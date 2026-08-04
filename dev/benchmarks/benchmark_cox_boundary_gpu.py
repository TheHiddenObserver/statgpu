"""Physical-GPU audit for the final PR80 Cox public-boundary fixes."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import warnings

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu._config import Device, set_device  # noqa: E402
from statgpu.inference._covariance import (  # noqa: E402
    classify_covariance_spectrum,
)
from statgpu.linear_model import (  # noqa: E402
    PenalizedCoxPHModel,
    PenalizedGLM_CV,
)
from statgpu.losses import _cox_ph as cox_loss  # noqa: E402
from statgpu.penalties import ElasticNetPenalty  # noqa: E402
from statgpu.survival import CoxPH, CoxPHCV  # noqa: E402
from statgpu.survival import _cox as cox_model  # noqa: E402
from statgpu.survival import _cox_counting as cox_counting  # noqa: E402
from statgpu.survival import _cox_score as cox_score  # noqa: E402
from statgpu.survival import _risk_sets as risk_sets  # noqa: E402
from statgpu.survival._concordance import (  # noqa: E402
    MAX_CONCORDANCE_PAIR_ENTRIES,
    concordance_tile_shape,
)
from statgpu.survival._risk_sets import (  # noqa: E402
    counting_process_concordance,
    cox_counting_process_objective,
)


SOURCE_FILES = (
    "statgpu/linear_model/penalized/_fit_mixin.py",
    ".github/workflows/test.yml",
    "statgpu/__init__.py",
    "statgpu/backends/_array_ops.py",
    "statgpu/backends/_utils.py",
    "statgpu/cross_validation/_base.py",
    "statgpu/inference/_covariance.py",
    "statgpu/linear_model/penalized/_penalized_cox.py",
    "statgpu/linear_model/penalized/_penalized_cox_cv.py",
    "statgpu/linear_model/penalized/_penalized_cv.py",
    "statgpu/penalties/_base.py",
    "statgpu/losses/_cox_ph.py",
    "statgpu/survival/__init__.py",
    "statgpu/survival/_cox.py",
    "statgpu/survival/_cox_counting.py",
    "statgpu/survival/_cox_cv.py",
    "statgpu/survival/_cox_errors.py",
    "statgpu/survival/_cox_fit_adapter.py",
    "statgpu/survival/_cox_inference.py",
    "statgpu/survival/_cox_legacy.py",
    "statgpu/survival/_numeric.py",
    "statgpu/survival/_concordance.py",
    "dev/benchmarks/pr79/diagnose_cox_pen.py",
    "dev/benchmarks/pr79/validators/numerical.py",
    "statgpu/survival/_cox_score.py",
    "statgpu/survival/_risk_sets.py",
    "dev/benchmarks/benchmark_cox_boundary_gpu.py",
    "dev/benchmarks/benchmark_cox_cluster.py",
    "dev/tests/test_pr79_accuracy_pipeline.py",
    "dev/tests/test_pr79_complete_review_fixes.py",
    "dev/tests/test_pr79_cox_parity_smoke.py",
    "dev/tests/test_cox_core_completion.py",
    "dev/tests/test_cox_phase1_completion.py",
    "dev/tests/test_pr80_complete_review_cycle.py",
    "dev/tests/test_pr80_completion_contract_followup.py",
    "dev/tests/test_pr80_constructor_boundaries.py",
    "dev/tests/test_pr80_workspace_estimator.py",
    "dev/tests/test_pr80_fit_boundary.py",
    "dev/tests/test_pr80_cv_fit_boundary.py",
    "dev/tests/test_pr80_cox_stability_review.py",
    "dev/tests/test_cox_cv.py",
    "dev/tests/test_pr80_target_transfer_overflow_cache.py",
    "dev/tests/test_pr80_robust_inference_units.py",
    "dev/tests/test_pr80_penalized_inference_strata.py",
    "dev/tests/test_pr80_penalized_cox_cv_contracts.py",
)

TARGETED_TEST_FILES = (
    "dev/tests/test_pr79_accuracy_pipeline.py",
    "dev/tests/test_pr79_complete_review_fixes.py",
    "dev/tests/test_pr79_cox_parity_smoke.py",
    "dev/tests/test_cox_core_completion.py",
    "dev/tests/test_cox_phase1_completion.py",
    "dev/tests/test_pr80_complete_review_cycle.py",
    "dev/tests/test_pr80_completion_contract_followup.py",
    "dev/tests/test_pr80_constructor_boundaries.py",
    "dev/tests/test_pr80_workspace_estimator.py",
    "dev/tests/test_pr80_fit_boundary.py",
    "dev/tests/test_pr80_cv_fit_boundary.py",
    "dev/tests/test_pr80_cox_stability_review.py",
    "dev/tests/test_cox_cv.py",
    "dev/tests/test_pr80_target_transfer_overflow_cache.py",
    "dev/tests/test_pr80_robust_inference_units.py",
    "dev/tests/test_pr80_penalized_inference_strata.py",
    "dev/tests/test_pr80_penalized_cox_cv_contracts.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _sample(seed: int = 2280, n: int = 72, p: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.linspace(0.35, -0.2, p)
    failure = rng.exponential(scale=np.exp(-(X @ beta))) + 0.05
    censor = rng.exponential(scale=1.8, size=n) + 0.05
    stop = np.minimum(failure, censor)
    event = (failure <= censor).astype(np.float64)
    event[:4] = 1.0
    return X, stop, event


def _backend(name: str):
    if name == "cupy":
        import cupy as xp

        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy has no physical CUDA device")
        return xp
    import torch as xp

    if not xp.cuda.is_available():
        raise RuntimeError("Torch CUDA is unavailable")
    return xp


def _array(name: str, xp, value, *, complex_value: bool = False):
    if name == "cupy":
        dtype = xp.complex128 if complex_value else xp.float64
        return xp.asarray(value, dtype=dtype)
    dtype = xp.complex128 if complex_value else xp.float64
    return xp.as_tensor(value, dtype=dtype, device="cuda")


def _numpy(name: str, value):
    if name == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    return value.detach().cpu().numpy()


def _sync(name: str, xp) -> None:
    if name == "cupy":
        xp.cuda.Stream.null.synchronize()
    else:
        xp.cuda.synchronize()


def _case_boundary(name: str, xp) -> dict:
    device = "cuda" if name == "cupy" else "torch"
    expected = Device.CUDA if name == "cupy" else Device.TORCH
    X_np, stop_np, event_np = _sample()
    X = _array(name, xp, X_np)
    target = _array(name, xp, np.column_stack((stop_np, event_np)))
    constructor_rejections = {}
    for parameter in (
        "compute_inference",
        "compute_cindex",
        "gpu_memory_cleanup",
    ):
        try:
            CoxPH(device=device, **{parameter: "False"})
        except ValueError as exc:
            constructor_rejections[parameter] = parameter in str(exc)
        else:
            constructor_rejections[parameter] = False
    model = CoxPH(
        device="cpu",
        compute_inference=True,
        compute_cindex=False,
        max_iter=80,
    )
    model.set_params(device=device)

    target_host_copies = []
    original_loss_to_numpy = cox_loss._to_numpy

    def recording_loss_to_numpy(value):
        target_host_copies.append(tuple(int(v) for v in value.shape))
        return original_loss_to_numpy(value)

    cox_loss._to_numpy = recording_loss_to_numpy
    try:
        started = time.perf_counter()
        model.fit(X, target)
        _sync(name, xp)
        fit_seconds = time.perf_counter() - started
    finally:
        cox_loss._to_numpy = original_loss_to_numpy

    complex_X = _array(
        name,
        xp,
        X_np[:3].astype(np.complex128) + 1j,
        complex_value=True,
    )
    complex_rejected = False
    try:
        model.predict_survival(complex_X)
    except ValueError as exc:
        complex_rejected = "real-valued" in str(exc)

    device_normalized = model.device is expected
    target_transfer_disclosed = (
        target_host_copies == [(X_np.shape[0],), (X_np.shape[0],)]
        and model.full_host_transfer_performed_ is True
    )
    cpu_from_device = CoxPH(
        device="cpu",
        compute_inference=False,
        compute_cindex=False,
        max_iter=80,
    ).fit(X, target)
    cpu_input_transfer_disclosed = (
        cpu_from_device.full_host_transfer_performed_ is True
    )
    finite = bool(np.all(np.isfinite(model.coef_)))
    model.coef_ = np.array([800.0, 0.0])
    extreme_X = _array(
        name, xp, np.array([[1.0, 0.0], [2.0, 0.0]])
    )
    extreme_survival, _ = model.predict_survival(extreme_X)
    extreme_survival_np = _numpy(name, extreme_survival)
    extreme_survival_stable = bool(
        np.all(np.isfinite(extreme_survival_np))
        and np.all(extreme_survival_np >= 0.0)
        and np.all(extreme_survival_np <= 1.0)
    )

    failed_refit_cleared = False
    try:
        model.fit(complex_X, target)
    except ValueError:
        failed_refit_cleared = (
            not model._fitted
            and model.coef_ is None
            and model._X is None
            and model._time is None
            and model._event is None
        )

    return {
        "backend": name,
        "fit_seconds": fit_seconds,
        "loss_target_host_copy_shapes": target_host_copies,
        "target_transfer_disclosed": target_transfer_disclosed,
        "cpu_input_transfer_disclosed": cpu_input_transfer_disclosed,
        "extreme_survival_log_domain": extreme_survival_stable,
        "complex_prediction_rejected": complex_rejected,
        "device_normalized": device_normalized,
        "failed_refit_cleared": failed_refit_cleared,
        "constructor_truthy_strings_rejected": constructor_rejections,
        "finite": finite,
        "passed": all(
            (
                target_transfer_disclosed,
                cpu_input_transfer_disclosed,
                extreme_survival_stable,
                complex_rejected,
                device_normalized,
                failed_refit_cleared,
                all(constructor_rejections.values()),
                finite,
            )
        ),
    }


def _case_ordinary_cv_preparation(name: str, xp) -> dict:
    """Audit ordinary GPU CV target transfers and fold-level loss reuse."""
    device = "cuda" if name == "cupy" else "torch"
    X_np, stop_np, event_np = _sample(seed=2481, n=36, p=2)
    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    copy_shapes = []
    content_validation_calls = 0
    original_loss_to_numpy = cox_loss._to_numpy
    original_matches_content = (
        cox_counting._PreparedRightCensoredCox.matches_content
    )

    def recording_loss_to_numpy(value):
        copy_shapes.append(tuple(int(v) for v in value.shape))
        return original_loss_to_numpy(value)

    def recording_matches_content(*args, **kwargs):
        nonlocal content_validation_calls
        content_validation_calls += 1
        return original_matches_content(*args, **kwargs)

    cox_loss._to_numpy = recording_loss_to_numpy
    cox_counting._PreparedRightCensoredCox.matches_content = (
        recording_matches_content
    )
    try:
        model = CoxPHCV(
            penalties=np.array([0.1, 0.01]),
            cv=2,
            ties="efron",
            device=device,
            compute_inference=False,
            max_iter=60,
            tol=1e-7,
            random_state=2481,
        ).fit(X, stop, event)
        _sync(name, xp)
    finally:
        cox_loss._to_numpy = original_loss_to_numpy
        cox_counting._PreparedRightCensoredCox.matches_content = (
            original_matches_content
        )

    fold_n = X_np.shape[0] // 2
    expected_shapes = [(fold_n,), (fold_n,)] * 2 + [
        (X_np.shape[0],),
        (X_np.shape[0],),
    ]
    diagnostics = model.cv_results_
    passed = all(
        (
            copy_shapes == expected_shapes,
            content_validation_calls == 0,
            diagnostics["candidate_right_censored_preparation_count"] == 2,
            diagnostics["candidate_target_host_transfer_count"] == 2,
            diagnostics["candidate_target_host_transfer_count_this_call"] == 2,
            diagnostics["candidate_target_host_vector_transfer_count"] == 4,
            diagnostics["selection_cache_hit"] is False,
            diagnostics["fold_backend_preparation_count_this_call"] == 2,
            model.cv_full_host_transfer_performed_ is True,
            model.final_refit_full_host_transfer_performed_ is True,
            model.full_host_transfer_performed_ is True,
        )
    )
    return {
        "backend": name,
        "ties": "efron",
        "loss_target_host_copy_shapes": copy_shapes,
        "expected_loss_target_host_copy_shapes": expected_shapes,
        "strict_content_validation_calls": content_validation_calls,
        "candidate_right_censored_preparation_count": diagnostics[
            "candidate_right_censored_preparation_count"
        ],
        "candidate_target_host_transfer_count": diagnostics[
            "candidate_target_host_transfer_count"
        ],
        "candidate_target_host_vector_transfer_count": diagnostics[
            "candidate_target_host_vector_transfer_count"
        ],
        "selection_cache_hit": diagnostics["selection_cache_hit"],
        "fold_backend_preparation_count_this_call": diagnostics[
            "fold_backend_preparation_count_this_call"
        ],
        "cv_full_host_transfer_performed": (
            model.cv_full_host_transfer_performed_
        ),
        "final_refit_full_host_transfer_performed": (
            model.final_refit_full_host_transfer_performed_
        ),
        "full_host_transfer_performed": model.full_host_transfer_performed_,
        "passed": bool(passed),
    }


def _case_prepared_state_and_packed_target(name: str, xp) -> dict:
    """Audit prepared-state integrity and packed-target D2H provenance."""
    X_np, stop_np, event_np = _sample(seed=2485, n=24, p=1)
    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    prepared = cox_counting.prepare_right_censored_cox_fast_path(
        X, stop, event, ties="efron"
    )
    X_changed = X.copy() if name == "cupy" else X.clone()
    X_changed[0, 0] += 0.25
    prepared_mismatch_rejected = False
    try:
        cox_counting.fit_counting_process_cox(
            X_changed,
            stop,
            event,
            ties="efron",
            compute_baseline=False,
            compute_score_residuals=False,
            right_censored_fast_path=True,
            right_censored_prepared=prepared,
        )
    except ValueError as exc:
        prepared_mismatch_rejected = "dataset contents" in str(exc)

    packed_target = _array(
        name, xp, np.column_stack((stop_np, event_np))
    )
    model = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        random_state=2485,
        device="cpu",
        compute_inference=False,
        max_iter=40,
    ).fit(X_np, packed_target)
    expected_backend = "cupy" if name == "cupy" else "torch-device"
    packed_target_transfer_disclosed = all(
        (
            model.cv_full_host_transfer_performed_ is True,
            model.full_host_transfer_performed_ is True,
            expected_backend in model.cv_results_["input_backends"],
        )
    )
    return {
        "backend": name,
        "prepared_mismatch_rejected": prepared_mismatch_rejected,
        "packed_target_input_backends": model.cv_results_["input_backends"],
        "cv_full_host_transfer_performed": (
            model.cv_full_host_transfer_performed_
        ),
        "full_host_transfer_performed": model.full_host_transfer_performed_,
        "packed_target_transfer_disclosed": (
            packed_target_transfer_disclosed
        ),
        "passed": bool(
            prepared_mismatch_rejected and packed_target_transfer_disclosed
        ),
    }


def _case_prediction_fast_path_and_fit_controls(name: str, xp) -> dict:
    """Audit the post-schema-7 prediction, solver, and parameter contracts."""
    device = "cuda" if name == "cupy" else "torch"
    prediction_model = PenalizedCoxPHModel(
        penalty="l2", device=device, compute_inference=False
    )
    prediction_model.coef_ = np.array([0.5, -0.25])
    prediction_model._selected_backend_name = name
    one_row = _array(name, xp, np.array([2.0, -1.0]))
    risk = prediction_model.predict_risk_score(one_row, return_cpu=False)
    one_dimensional_row_ok = bool(
        tuple(risk.shape) == (1,)
        and np.allclose(_numpy(name, risk), np.array([1.25]))
    )
    shape_rejections = {}
    for label, value in (
        ("wrong_one_dimensional_length", np.array([1.0, 2.0, 3.0])),
        ("three_dimensional", np.ones((1, 2, 1))),
    ):
        try:
            prediction_model.predict_risk_score(
                _array(name, xp, value), return_cpu=False
            )
        except ValueError:
            shape_rejections[label] = True
        else:
            shape_rejections[label] = False

    X_np, stop_np, event_np = _sample(seed=2486, n=24, p=2)
    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    start = _array(name, xp, np.zeros_like(stop_np))
    invalid_start = _array(name, xp, np.zeros_like(stop_np))
    invalid_start[0] = stop[0] * 0.5
    one_stratum = _array(name, xp, np.full(stop_np.shape, 7.0))
    multiple_strata = _array(
        name, xp, np.r_[np.zeros(stop_np.shape[0] - 1), 1.0]
    )
    fast_path = {}
    for ties in ("breslow", "efron"):
        for label, kwargs in (
            ("nonzero_start_rejected", {"start": invalid_start}),
            ("multiple_strata_rejected", {"strata": multiple_strata}),
        ):
            try:
                cox_counting.fit_counting_process_cox(
                    X,
                    stop,
                    event,
                    ties=ties,
                    compute_baseline=False,
                    compute_score_residuals=False,
                    right_censored_fast_path=True,
                    **kwargs,
                )
            except ValueError as exc:
                fast_path[f"{ties}_{label}"] = (
                    "right_censored_fast_path requires" in str(exc)
                )
            else:
                fast_path[f"{ties}_{label}"] = False
        valid = cox_counting.fit_counting_process_cox(
            X,
            stop,
            event,
            start=start,
            strata=one_stratum,
            ties=ties,
            max_iter=20,
            compute_baseline=False,
            compute_score_residuals=False,
            right_censored_fast_path=True,
        )
        fast_path[f"{ties}_ordinary_inputs_accepted"] = bool(
            np.all(np.isfinite(_numpy(name, valid["coef"])))
        )

    set_penalty = np.float64(0.1)
    fit_model = CoxPH(
        device=device,
        compute_inference=0,
        compute_cindex=0,
        max_iter=np.int64(40),
        tol=np.float64(1e-7),
    ).set_params(
        ties="EFRON",
        cov_type="NONROBUST",
        inference_mode="STRICT",
        penalty=set_penalty,
    )
    before = fit_model.get_params().copy()
    fit_model.fit(X, stop, event)
    set_params_representation_stable = all(
        (
            fit_model.get_params() == before,
            fit_model.ties == "EFRON",
            fit_model.cov_type == "NONROBUST",
            fit_model.inference_mode == "STRICT",
            fit_model.penalty is set_penalty,
        )
    )
    active_controls_normalized = all(
        (
            fit_model._fit_controls.ties == "efron",
            fit_model._fit_controls.cov_type == "nonrobust",
            fit_model._fit_controls.inference_mode == "strict",
            fit_model._fit_controls.compute_inference is False,
            fit_model._fit_controls.compute_cindex is False,
        )
    )

    single_stratum_model = CoxPH(
        device=device,
        compute_inference=True,
        compute_cindex=False,
        max_iter=60,
        tol=1e-8,
    ).fit(X, stop, event, strata=one_stratum)
    single_stratum_prediction = {}
    try:
        single_stratum_model.predict_survival(X[:2], times=[0.2, 0.8])
    except ValueError as exc:
        single_stratum_prediction["missing_rejected"] = (
            "strata is required" in str(exc)
        )
    else:
        single_stratum_prediction["missing_rejected"] = False
    try:
        single_stratum_model.predict_survival(
            X[:2], times=[0.2, 0.8], strata=one_stratum[:2] + 1
        )
    except ValueError as exc:
        single_stratum_prediction["unknown_rejected"] = (
            "unknown prediction stratum" in str(exc)
        )
    else:
        single_stratum_prediction["unknown_rejected"] = False
    known_survival, _ = single_stratum_model.predict_survival(
        X[:2], times=[0.2, 0.8], strata=one_stratum[:2]
    )
    single_stratum_prediction["known_accepted"] = bool(
        tuple(known_survival.shape) == (2, 2)
        and np.all(np.isfinite(_numpy(name, known_survival)))
    )

    single_stratum_cv = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        device=device,
        compute_inference=True,
        max_iter=60,
        tol=1e-8,
        random_state=2486,
    ).fit(X, stop, event, strata=one_stratum)
    try:
        single_stratum_cv.predict_survival(X[:2], times=[0.2, 0.8])
    except ValueError as exc:
        single_stratum_prediction["cv_missing_rejected"] = (
            "strata is required" in str(exc)
        )
    else:
        single_stratum_prediction["cv_missing_rejected"] = False

    budget_model = CoxPH(
        device=device,
        compute_inference=False,
        compute_cindex=False,
        penalty=0.1,
        max_iter=1,
        tol=1e-15,
    ).fit(X, stop, event)
    termination_provenance = {
        "interpreted": budget_model.termination_reason_,
        "raw": budget_model.optimization_stop_reason_,
        "passed": bool(
            budget_model.termination_reason_ == "stalled_with_large_kkt"
            and budget_model.optimization_stop_reason_ == "max_iter"
        ),
    }
    passed = all(
        (
            one_dimensional_row_ok,
            all(shape_rejections.values()),
            all(fast_path.values()),
            set_params_representation_stable,
            active_controls_normalized,
            all(single_stratum_prediction.values()),
            termination_provenance["passed"],
        )
    )
    return {
        "backend": name,
        "one_dimensional_multifeature_row": one_dimensional_row_ok,
        "shape_rejections": shape_rejections,
        "fast_path_eligibility": fast_path,
        "constructor_parameters_stable": set_params_representation_stable,
        "set_params_representation_stable": set_params_representation_stable,
        "active_controls_normalized": active_controls_normalized,
        "single_explicit_stratum_prediction": single_stratum_prediction,
        "termination_provenance": termination_provenance,
        "passed": bool(passed),
    }


def _case_hazard_ratio_boundary(name: str, xp) -> dict:
    """Verify strict overflow behavior on both GPU public Cox estimators."""
    device = "cuda" if name == "cupy" else "torch"
    X_np, stop_np, event_np = _sample(seed=2482, n=36, p=1)
    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    X_one = _array(name, xp, np.ones((2, 1)))

    canonical = CoxPH(
        device=device,
        compute_inference=False,
        compute_cindex=False,
        max_iter=60,
    ).fit(X, stop, event)
    canonical_rejections = {}
    for value in (800.0, -800.0):
        canonical.coef_ = np.array([value])
        try:
            canonical.predict_hazard_ratio(X_one)
        except FloatingPointError as exc:
            canonical_rejections[str(value)] = (
                "finite positive float64" in str(exc)
            )
        else:
            canonical_rejections[str(value)] = False
    canonical.coef_ = np.array([800.0])
    log_risk = _numpy(name, canonical.predict_risk_score(X_one))

    penalized = PenalizedCoxPHModel(
        penalty="l2",
        alpha=0.1,
        device=device,
        compute_inference=False,
        max_iter=60,
    ).fit(X, _array(name, xp, np.column_stack((stop_np, event_np))))
    penalized_rejections = {}
    for value in (800.0, -800.0):
        penalized.coef_ = np.array([value])
        try:
            penalized.predict_hazard_ratio(X_one, return_cpu=False)
        except FloatingPointError as exc:
            penalized_rejections[str(value)] = (
                "finite positive float64" in str(exc)
            )
        else:
            penalized_rejections[str(value)] = False
    penalized.coef_ = np.array([800.0])
    penalized_log_risk = _numpy(
        name,
        penalized.predict_risk_score(X_one, return_cpu=False),
    )
    penalized_complex_rejected = False
    try:
        penalized.predict_risk_score(
            _array(
                name,
                xp,
                np.ones((2, 1), dtype=np.complex128) + 1j,
                complex_value=True,
            ),
            return_cpu=False,
        )
    except ValueError as exc:
        penalized_complex_rejected = "real-valued" in str(exc)

    passed = bool(
        all(canonical_rejections.values())
        and all(penalized_rejections.values())
        and penalized_complex_rejected
        and np.array_equal(np.asarray(log_risk), np.array([800.0, 800.0]))
        and np.array_equal(
            np.asarray(penalized_log_risk), np.array([800.0, 800.0])
        )
    )
    return {
        "backend": name,
        "canonical_range_rejections": canonical_rejections,
        "penalized_range_rejections": penalized_rejections,
        "penalized_complex_log_risk_rejected": penalized_complex_rejected,
        "canonical_log_risk": np.asarray(log_risk).tolist(),
        "penalized_log_risk": np.asarray(penalized_log_risk).tolist(),
        "passed": passed,
    }


def _case_cv(name: str, xp) -> dict:
    device = "cuda" if name == "cupy" else "torch"
    expected = Device.CUDA if name == "cupy" else Device.TORCH
    X_np, stop_np, event_np = _sample(seed=2293, n=36, p=2)
    constructor_rejections = {}
    for parameter in ("compute_inference", "gpu_memory_cleanup"):
        try:
            CoxPHCV(
                penalties=np.array([0.1]),
                cv=2,
                device=device,
                **{parameter: "False"},
            )
        except ValueError as exc:
            constructor_rejections[parameter] = parameter in str(exc)
        else:
            constructor_rejections[parameter] = False
    model = CoxPHCV(
        penalties=np.array([0.1, 0.01]),
        cv=2,
        device="cpu",
        compute_inference=False,
        gpu_memory_cleanup=True,
        max_iter=60,
        random_state=2293,
    )
    model.set_params(device=device)
    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    strata = _array(name, xp, np.arange(X_np.shape[0]) % 3)
    cluster = _array(name, xp, np.arange(X_np.shape[0]) % 5)
    subject_id = _array(name, xp, np.arange(X_np.shape[0]))
    started = time.perf_counter()
    model.fit(
        X,
        stop,
        event,
        strata=strata,
        cluster=cluster,
        subject_id=subject_id,
    )
    _sync(name, xp)
    fit_seconds = time.perf_counter() - started
    final_refit_skips_cindex = (
        model.estimator_.compute_cindex is False
        and model.estimator_.concordance_ is None
    )
    cleanup_operations = {
        "outer_cuda": 0,
        "outer_torch": 0,
        "inner_cuda": 0,
        "inner_torch": 0,
    }
    model._cleanup_cuda_memory = lambda: cleanup_operations.__setitem__(
        "outer_cuda", cleanup_operations["outer_cuda"] + 1
    )
    model._cleanup_torch_memory = lambda: cleanup_operations.__setitem__(
        "outer_torch", cleanup_operations["outer_torch"] + 1
    )

    def inner_cuda_cleanup():
        if model.estimator_.gpu_memory_cleanup:
            cleanup_operations["inner_cuda"] += 1

    def inner_torch_cleanup():
        if model.estimator_.gpu_memory_cleanup:
            cleanup_operations["inner_torch"] += 1

    model.estimator_._cleanup_cuda_memory = inner_cuda_cleanup
    model.estimator_._cleanup_torch_memory = inner_torch_cleanup
    model.predict(_array(name, xp, X_np[:4]))
    _sync(name, xp)
    # Snapshot the public-call counters before ``model`` leaves this case.
    # ``CoxPHCV.__del__`` legitimately invokes cleanup later; retaining the
    # mutable dictionary would rewrite the already-evaluated JSON evidence.
    cleanup_operations_after_predict = dict(cleanup_operations)
    single_cleanup_owner = cleanup_operations_after_predict == {
        "outer_cuda": 1,
        "outer_torch": 1,
        "inner_cuda": 0,
        "inner_torch": 0,
    }
    transfer_provenance = (
        model.cv_full_host_transfer_performed_ is True
        and model.final_refit_full_host_transfer_performed_ is True
        and model.full_host_transfer_performed_ is True
        and model.orchestration_device_ == "cpu"
        and model.cv_results_["input_backends"] == (
            "cupy" if name == "cupy" else "torch-device",
        )
    )
    candidate_label_preparation = (
        model.cv_results_["fold_backend_preparation_count"] == 2
        and model.cv_results_["candidate_cluster_used"] is False
        and model.cv_results_["candidate_subject_id_used"] is False
        and model.cv_results_["candidate_strata_preencoded"] is True
    )
    passed = (
        model.device is expected
        and model.estimator_ is not None
        and model.estimator_.device is expected
        and model.effective_device_ == device
        and bool(np.all(np.isfinite(model.coef_)))
        and all(constructor_rejections.values())
        and final_refit_skips_cindex
        and single_cleanup_owner
        and transfer_provenance
        and candidate_label_preparation
    )
    return {
        "backend": name,
        "fit_seconds": fit_seconds,
        "effective_device": model.effective_device_,
        "constructor_truthy_strings_rejected": constructor_rejections,
        "final_refit_skips_training_cindex": final_refit_skips_cindex,
        "cleanup_operations_after_predict": cleanup_operations_after_predict,
        "single_cleanup_owner": single_cleanup_owner,
        "transfer_provenance": transfer_provenance,
        "cv_full_host_transfer_performed": (
            model.cv_full_host_transfer_performed_
        ),
        "final_refit_full_host_transfer_performed": (
            model.final_refit_full_host_transfer_performed_
        ),
        "full_host_transfer_performed": model.full_host_transfer_performed_,
        "orchestration_device": model.orchestration_device_,
        "input_backends": model.cv_results_["input_backends"],
        "candidate_label_preparation": candidate_label_preparation,
        "fold_backend_preparation_count": model.cv_results_[
            "fold_backend_preparation_count"
        ],
        "candidate_cluster_used": model.cv_results_[
            "candidate_cluster_used"
        ],
        "candidate_subject_id_used": model.cv_results_[
            "candidate_subject_id_used"
        ],
        "candidate_strata_preencoded": model.cv_results_[
            "candidate_strata_preencoded"
        ],
        "finite": bool(np.all(np.isfinite(model.coef_))),
        "passed": bool(passed),
    }


def _case_workspace(name: str, xp) -> dict:
    rng = np.random.default_rng(2294)
    n, p = 8192, 3
    X_np = rng.normal(size=(n, p))
    stop_np = np.full(n, 6.0)
    stop_np[:4] = 5.0
    event_np = np.zeros(n)
    event_np[:4] = 1.0
    start_np = rng.uniform(0.0, 4.0, size=n)
    beta_np = np.array([0.2, -0.15, 0.1])
    reference = cox_counting_process_objective(
        beta_np,
        X_np,
        stop_np,
        event_np,
        start=start_np,
        ties="efron",
        score_residuals=True,
    )
    previous = os.environ.get("STATGPU_COX_GROUP_MAX_BYTES")
    os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = "4096"
    try:
        started = time.perf_counter()
        result = cox_counting_process_objective(
            _array(name, xp, beta_np),
            _array(name, xp, X_np),
            _array(name, xp, stop_np),
            _array(name, xp, event_np),
            start=_array(name, xp, start_np),
            ties="efron",
            score_residuals=True,
        )
        _sync(name, xp)
        seconds = time.perf_counter() - started
    finally:
        if previous is None:
            os.environ.pop("STATGPU_COX_GROUP_MAX_BYTES", None)
        else:
            os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = previous

    differences = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(reference[key])
                    - np.asarray(_numpy(name, result[key]))
                )
            )
        )
        for key in ("score", "information", "score_residuals")
    }
    differences["log_likelihood"] = float(
        abs(
            float(reference["log_likelihood"])
            - float(np.asarray(_numpy(name, result["log_likelihood"])))
        )
    )
    passed = max(differences.values()) <= 1e-9
    return {
        "backend": name,
        "n": n,
        "p": p,
        "workspace_limit_bytes": 4096,
        "seconds": seconds,
        "max_abs_differences": differences,
        "passed": passed,
    }


def _case_wide_workspace_route(name: str, xp) -> dict:
    rng = np.random.default_rng(2304)
    n, p = 4096, 128
    workspace_limit = 8 * 1024 * 1024
    X_np = rng.normal(size=(n, p))
    stop_np = np.full(n, 6.0)
    stop_np[:4] = 5.0
    event_np = np.zeros(n)
    event_np[:4] = 1.0
    start_np = rng.uniform(0.0, 4.0, size=n)
    beta_np = np.linspace(0.08, -0.04, p)
    itemsize = np.dtype(np.float64).itemsize

    # This is the exact pre-fdc5f00 estimate. It omitted the two possible
    # n-by-p weighted-design intermediates used by three-operand einsum.
    old_estimate = n * (2 + 8 * itemsize) + 6 * p * p * itemsize
    corrected_estimate = risk_sets._estimate_dense_group_workspace_bytes(
        n,
        p,
        itemsize,
        compute_derivatives=True,
        score_residuals=True,
    )
    routing_boundary_passed = (
        old_estimate <= workspace_limit < corrected_estimate
    )

    previous = os.environ.get("STATGPU_COX_GROUP_MAX_BYTES")
    os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = str(1 << 50)
    try:
        reference = cox_counting_process_objective(
            beta_np,
            X_np,
            stop_np,
            event_np,
            start=start_np,
            ties="efron",
            score_residuals=True,
        )
    finally:
        if previous is None:
            os.environ.pop("STATGPU_COX_GROUP_MAX_BYTES", None)
        else:
            os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = previous

    streaming_calls = []
    original_streamed = risk_sets._streamed_stratum_group_objective

    def recording_streamed(*args, **kwargs):
        streaming_calls.append(
            {
                "n": int(args[1].shape[0]),
                "p": int(args[1].shape[1]),
                "workspace_limit_bytes": int(kwargs["max_workspace_bytes"]),
            }
        )
        return original_streamed(*args, **kwargs)

    risk_sets._streamed_stratum_group_objective = recording_streamed
    os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = str(workspace_limit)
    try:
        started = time.perf_counter()
        result = cox_counting_process_objective(
            _array(name, xp, beta_np),
            _array(name, xp, X_np),
            _array(name, xp, stop_np),
            _array(name, xp, event_np),
            start=_array(name, xp, start_np),
            ties="efron",
            score_residuals=True,
        )
        _sync(name, xp)
        seconds = time.perf_counter() - started
    finally:
        risk_sets._streamed_stratum_group_objective = original_streamed
        if previous is None:
            os.environ.pop("STATGPU_COX_GROUP_MAX_BYTES", None)
        else:
            os.environ["STATGPU_COX_GROUP_MAX_BYTES"] = previous

    differences = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(reference[key])
                    - np.asarray(_numpy(name, result[key]))
                )
            )
        )
        for key in ("score", "information", "score_residuals")
    }
    differences["log_likelihood"] = float(
        abs(
            float(reference["log_likelihood"])
            - float(np.asarray(_numpy(name, result["log_likelihood"])))
        )
    )
    route_was_streamed = streaming_calls == [
        {
            "n": n,
            "p": p,
            "workspace_limit_bytes": workspace_limit,
        }
    ]
    passed = (
        routing_boundary_passed
        and route_was_streamed
        and max(differences.values()) <= 1e-9
    )
    return {
        "backend": name,
        "n": n,
        "p": p,
        "workspace_limit_bytes": workspace_limit,
        "old_estimate_bytes": old_estimate,
        "corrected_estimate_bytes": corrected_estimate,
        "old_estimate_selects_dense": old_estimate <= workspace_limit,
        "corrected_estimate_selects_streaming": (
            corrected_estimate > workspace_limit
        ),
        "observed_streaming_calls": streaming_calls,
        "seconds": seconds,
        "max_abs_differences": differences,
        "passed": passed,
    }


def _case_concordance_boundaries(name: str, xp) -> dict:
    device = "cuda" if name == "cupy" else "torch"
    X_np, stop_np, event_np = _sample(seed=2406, n=72, p=2)
    X = _array(name, xp, X_np)
    target = _array(name, xp, np.column_stack((stop_np, event_np)))
    model = CoxPH(
        device=device,
        compute_inference=False,
        compute_cindex=False,
        max_iter=80,
        tol=1e-7,
    ).fit(X, target)

    X_score_np = X_np[:6]
    stop_score_np = np.arange(1, 7, dtype=np.float64)
    censored_np = np.zeros(6, dtype=np.float64)
    X_score = _array(name, xp, X_score_np)
    stop_score = _array(name, xp, stop_score_np)
    censored = _array(name, xp, censored_np)
    ordinary_value = model.score(X_score, stop_score, censored)
    counting_value = model.score(
        X_score,
        stop_score,
        censored,
        start=_array(name, xp, np.zeros(6)),
        strata=_array(name, xp, np.array([0, 0, 0, 1, 1, 1])),
    )

    penalized = PenalizedCoxPHModel(
        penalty="l2",
        alpha=0.2,
        device=device,
        max_iter=80,
        tol=1e-6,
        compute_inference=False,
    ).fit(X, target)
    penalized_value = penalized.score(
        X_score,
        _array(name, xp, np.column_stack((stop_score_np, censored_np))),
    )
    penalized_constructor_rejected = False
    try:
        PenalizedCoxPHModel(device=device, lla="False")
    except ValueError as exc:
        penalized_constructor_rejected = "lla must be" in str(exc)

    large_n = MAX_CONCORDANCE_PAIR_ENTRIES + 1
    event_tile, sample_tile = concordance_tile_shape(1, large_n)
    X_large_np = np.linspace(0.0, 1.0, large_n).reshape(-1, 1)
    stop_large_np = np.full(large_n, 2.0)
    stop_large_np[0] = 1.0
    event_large_np = np.zeros(large_n)
    event_large_np[0] = 1.0
    start_large_np = np.zeros(large_n)
    started = time.perf_counter()
    large_value_raw = counting_process_concordance(
        _array(name, xp, np.array([0.25])),
        _array(name, xp, X_large_np),
        _array(name, xp, stop_large_np),
        _array(name, xp, event_large_np),
        start=_array(name, xp, start_large_np),
    )
    _sync(name, xp)
    large_seconds = time.perf_counter() - started
    large_value = float(np.asarray(_numpy(name, large_value_raw)))

    passed = all(
        (
            ordinary_value == 0.5,
            counting_value == 0.5,
            penalized_value == 0.5,
            penalized_constructor_rejected,
            event_tile * sample_tile <= MAX_CONCORDANCE_PAIR_ENTRIES,
            sample_tile < large_n,
            large_value == 0.0,
        )
    )
    return {
        "backend": name,
        "all_censored_public_coxph": ordinary_value,
        "all_censored_counting_coxph": counting_value,
        "all_censored_penalized_cox": penalized_value,
        "penalized_truthy_string_rejected": penalized_constructor_rejected,
        "large_pair_case": {
            "n_events": 1,
            "n_samples": large_n,
            "event_tile": event_tile,
            "sample_tile": sample_tile,
            "tile_entries": event_tile * sample_tile,
            "limit_entries": MAX_CONCORDANCE_PAIR_ENTRIES,
            "comparison_tiles": (large_n + sample_tile - 1) // sample_tile,
            "concordance": large_value,
            "seconds": large_seconds,
        },
        "passed": passed,
    }


def _case_completion_contract(name: str, xp) -> dict:
    from statgpu.survival._cox_legacy import _LegacyCoxReferenceMixin
    from statgpu.survival import _cox as cox_module

    device = "cuda" if name == "cupy" else "torch"
    X_np, stop_np, event_np = _sample(seed=2410, n=72, p=2)
    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    model = CoxPH(
        device=device,
        compute_inference=True,
        compute_cindex=False,
        gpu_memory_cleanup=True,
        max_iter=80,
        tol=1e-8,
    ).fit(X, stop, event)

    cleanup_calls = {"cuda": 0, "torch": 0}

    def cleanup_cuda():
        cleanup_calls["cuda"] += 1

    def cleanup_torch():
        cleanup_calls["torch"] += 1

    model._cleanup_cuda_memory = cleanup_cuda
    model._cleanup_torch_memory = cleanup_torch
    model.predict_risk_score(X[:4])
    success_cleanup = dict(cleanup_calls)
    complex_rejected = False
    try:
        model.predict_hazard_ratio(
            _array(
                name,
                xp,
                X_np[:4].astype(np.complex128) + 1j,
                complex_value=True,
            )
        )
    except ValueError as exc:
        complex_rejected = "real-valued" in str(exc)
    error_cleanup = {
        key: cleanup_calls[key] - success_cleanup[key]
        for key in cleanup_calls
    }

    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        model.summary()
    summary_text = summary_buffer.getvalue()
    summary_truthful = all(
        token in summary_text
        for token in (
            "interface='matrix'",
            "counting_process=False",
            "stratified=False",
        )
    ) and "coxph(formula = Surv(time, event) ~ ." not in summary_text

    invalid_subject = np.arange(X_np.shape[0], dtype=np.float64)
    invalid_subject[0] = 0.1
    subject_rejected = False
    try:
        counting_process_concordance(
            _array(name, xp, model.coef_),
            X,
            stop,
            event,
            subject_id=invalid_subject,
        )
    except ValueError as exc:
        subject_rejected = "subject_id" in str(exc) and "integer-valued" in str(exc)

    sync_calls = []
    original_sync = cox_score._sync_scalars
    original_tiles = cox_score._concordance_tile_shape

    def recording_sync(*values, backend):
        sync_calls.append({"values": len(values), "backend": backend})
        return original_sync(*values, backend=backend)

    cox_score._sync_scalars = recording_sync
    cox_score._concordance_tile_shape = lambda _events, _samples: (1, 2)
    try:
        score_value = model.score(X, stop, event)
    finally:
        cox_score._sync_scalars = original_sync
        cox_score._concordance_tile_shape = original_tiles

    inference_result = model._inference_result
    inference_contract = all(
        (
            inference_result is not None,
            type(inference_result).__name__ == "ParameterInferenceResult",
            np.allclose(model._params, model.coef_),
            np.allclose(inference_result.bse, model._bse),
            np.allclose(inference_result.pvalues, model._pvalues),
            np.allclose(inference_result.conf_int, model._conf_int),
        )
    )
    dispatch_source = inspect.getsource(CoxPH._fit_counting_process_dispatch)
    dispatch_direct_backend_imports_absent = (
        "import cupy" not in dispatch_source
        and "import torch" not in dispatch_source
    )
    import_time_adapter_absent = CoxPH.fit.__module__ == "statgpu.survival._cox"
    legacy_methods = tuple(
        method
        for method, value in vars(_LegacyCoxReferenceMixin).items()
        if callable(value)
    )
    legacy_mixin_isolated = (
        _LegacyCoxReferenceMixin not in CoxPH.__mro__
        and all(not hasattr(CoxPH, method) for method in legacy_methods)
        and "_cox_legacy" not in inspect.getsource(cox_module)
    )
    passed = all(
        (
            success_cleanup == {"cuda": 1, "torch": 1},
            error_cleanup == {"cuda": 1, "torch": 1},
            complex_rejected,
            summary_truthful,
            subject_rejected,
            sync_calls == [{"values": 3, "backend": name}],
            np.isfinite(score_value),
            inference_contract,
            dispatch_direct_backend_imports_absent,
            import_time_adapter_absent,
            legacy_mixin_isolated,
        )
    )
    return {
        "backend": name,
        "cleanup_calls_after_success": success_cleanup,
        "cleanup_calls_after_error": error_cleanup,
        "complex_prediction_rejected": complex_rejected,
        "summary_truthful": summary_truthful,
        "fractional_subject_id_rejected": subject_rejected,
        "ordinary_concordance_sync_calls": sync_calls,
        "concordance": score_value,
        "inference_result_contract": inference_contract,
        "dispatch_direct_backend_imports_absent": dispatch_direct_backend_imports_absent,
        "import_time_adapter_absent": import_time_adapter_absent,
        "legacy_mixin_isolated": legacy_mixin_isolated,
        "passed": bool(passed),
    }


def _case_robust_inference_units(name: str, xp) -> dict:
    """Exercise strict robust-inference unit gates on a physical GPU."""
    device = "cuda" if name == "cupy" else "torch"
    X_np, stop_np, event_np = _sample(seed=9344, n=48, p=3)
    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    one_unit = _array(name, xp, np.zeros(X_np.shape[0]))
    p_units = _array(
        name, xp, np.arange(X_np.shape[0]) % X_np.shape[1]
    )
    two_units = _array(name, xp, np.arange(X_np.shape[0]) % 2)
    p_plus_one_units = _array(
        name, xp, np.arange(X_np.shape[0]) % (X_np.shape[1] + 1)
    )

    def rejected(cov_type, *, cluster=None, subject_id=None):
        model = CoxPH(
            device=device,
            cov_type=cov_type,
            compute_inference=True,
            compute_cindex=False,
            max_iter=100,
            tol=1e-9,
        )
        try:
            model.fit(
                X,
                stop,
                event,
                cluster=cluster,
                subject_id=subject_id,
            )
        except RuntimeError as exc:
            return {
                "error": str(exc),
                "state_cleared": model.coef_ is None and not model._fitted,
            }
        return {"error": "", "state_cleared": False}

    single_cluster = rejected("cluster", cluster=one_unit)
    single_subject_hc0 = rejected("hc0", subject_id=one_unit)
    single_subject_hc1 = rejected("hc1", subject_id=one_unit)
    equal_units_hc1 = rejected("hc1", subject_id=p_units)
    estimation_only = CoxPH(
        device=device,
        cov_type="cluster",
        compute_inference=False,
        compute_cindex=False,
        max_iter=100,
        tol=1e-9,
    ).fit(X, stop, event, cluster=one_unit)

    common = {
        "device": device,
        "compute_inference": True,
        "compute_cindex": False,
        "max_iter": 100,
        "tol": 1e-9,
    }
    hc0 = CoxPH(cov_type="hc0", **common).fit(
        X, stop, event, subject_id=p_plus_one_units
    )
    hc1 = CoxPH(cov_type="hc1", **common).fit(
        X, stop, event, subject_id=p_plus_one_units
    )
    rank_deficient_cluster = CoxPH(cov_type="cluster", **common).fit(
        X, stop, event, cluster=two_units
    )
    rank_deficient_hc0 = CoxPH(cov_type="hc0", **common).fit(
        X, stop, event, subject_id=p_units
    )
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        rank_deficient_cluster.summary()
    rank_deficient_summary = summary_buffer.getvalue()
    hc0_variance = np.asarray(hc0._var_matrix)
    hc1_variance = np.asarray(hc1._var_matrix)
    hc1_bse = np.asarray(hc1._bse)
    hc1_pvalues = np.asarray(hc1._pvalues)
    correction = (X_np.shape[1] + 1) / (
        X_np.shape[1] + 1 - X_np.shape[1]
    )
    variance_ratio_matches = np.allclose(
        hc1_variance,
        correction * hc0_variance,
        rtol=2e-8,
        atol=2e-10,
    )

    forced_spectrum = classify_covariance_spectrum(
        np.array(
            [
                [1.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    original_classifier = cox_model._classify_covariance_spectrum
    indefinite_model = CoxPH(cov_type="cluster", **common)
    indefinite_error = ""
    indefinite_cv = CoxPHCV(
        penalties=np.array([0.1]),
        cv=2,
        device=device,
        cov_type="cluster",
        compute_inference=True,
        max_iter=60,
        tol=1e-8,
    )
    indefinite_cv_error = ""
    try:
        cox_model._classify_covariance_spectrum = (
            lambda _covariance: forced_spectrum
        )
        try:
            indefinite_model.fit(
                X,
                stop,
                event,
                cluster=p_plus_one_units,
            )
        except RuntimeError as exc:
            indefinite_error = str(exc)
        try:
            indefinite_cv.fit(
                X,
                stop,
                event,
                cluster=p_plus_one_units,
            )
        except RuntimeError as exc:
            indefinite_cv_error = str(exc)
    finally:
        cox_model._classify_covariance_spectrum = original_classifier

    passed = all(
        (
            "cluster covariance requires at least two" in single_cluster["error"],
            "hc0 covariance requires at least two"
            in single_subject_hc0["error"],
            "hc1 covariance requires at least two"
            in single_subject_hc1["error"],
            "HC1 covariance requires n_units > n_features"
            in equal_units_hc1["error"],
            single_cluster["state_cleared"],
            single_subject_hc0["state_cleared"],
            single_subject_hc1["state_cleared"],
            equal_units_hc1["state_cleared"],
            estimation_only._fitted,
            estimation_only._bse is None,
            np.all(np.isfinite(estimation_only.coef_)),
            np.all(np.isfinite(hc1_bse)),
            np.all(hc1_bse > 0.0),
            np.all(np.isfinite(hc1_pvalues)),
            variance_ratio_matches,
            hc0.wald_test_available_,
            hc1.wald_test_available_,
            np.isfinite(hc0._wald_test_stat),
            np.isfinite(hc1._wald_test_stat),
            not rank_deficient_cluster.wald_test_available_,
            not rank_deficient_hc0.wald_test_available_,
            np.all(np.isfinite(rank_deficient_cluster._bse)),
            np.all(np.isfinite(rank_deficient_hc0._bse)),
            "Robust Wald test unavailable: robust covariance is rank-deficient"
            in rank_deficient_summary,
            "Classical likelihood-ratio test:" in rank_deficient_summary,
            "Classical score (logrank) test:" in rank_deficient_summary,
            "Wald test: nan" not in rank_deficient_summary,
            "not positive semidefinite" in indefinite_error,
            indefinite_model.coef_ is None,
            not indefinite_model._fitted,
            "not positive semidefinite" in indefinite_cv_error,
            indefinite_cv.estimator_ is None,
            not indefinite_cv._fitted,
        )
    )
    return {
        "backend": name,
        "single_cluster": single_cluster,
        "single_subject_hc0": single_subject_hc0,
        "single_subject_hc1": single_subject_hc1,
        "equal_units_hc1": equal_units_hc1,
        "single_cluster_estimation_only": {
            "fitted": bool(estimation_only._fitted),
            "inference_unset": estimation_only._bse is None,
            "coefficients": np.asarray(estimation_only.coef_).tolist(),
        },
        "p_plus_one_units": {
            "n_features": int(X_np.shape[1]),
            "n_units": int(X_np.shape[1] + 1),
            "finite_sample_correction": correction,
            "standard_errors": hc1_bse.tolist(),
            "pvalues": hc1_pvalues.tolist(),
            "variance_ratio_matches": bool(variance_ratio_matches),
            "hc0_wald_available": bool(hc0.wald_test_available_),
            "hc1_wald_available": bool(hc1.wald_test_available_),
        },
        "rank_deficient_joint_wald": {
            "cluster_units": 2,
            "subject_units": int(X_np.shape[1]),
            "cluster_marginal_standard_errors": np.asarray(
                rank_deficient_cluster._bse
            ).tolist(),
            "subject_hc0_marginal_standard_errors": np.asarray(
                rank_deficient_hc0._bse
            ).tolist(),
            "cluster_wald_available": bool(
                rank_deficient_cluster.wald_test_available_
            ),
            "subject_hc0_wald_available": bool(
                rank_deficient_hc0.wald_test_available_
            ),
            "failure_reason": rank_deficient_cluster.wald_test_failure_reason_,
            "summary_contract": bool(
                "Robust Wald test unavailable: robust covariance is rank-deficient"
                in rank_deficient_summary
                and "Wald test: nan" not in rank_deficient_summary
            ),
        },
        "materially_indefinite_covariance": {
            "classification": forced_spectrum.classification,
            "minimum_eigenvalue": forced_spectrum.minimum_eigenvalue,
            "cox_error": indefinite_error,
            "cox_state_cleared": bool(
                indefinite_model.coef_ is None and not indefinite_model._fitted
            ),
            "cv_error": indefinite_cv_error,
            "cv_state_cleared": bool(
                indefinite_cv.estimator_ is None and not indefinite_cv._fitted
            ),
        },
        "passed": bool(passed),
    }


def _case_penalized_inference_and_strata(name: str, xp) -> dict:
    """Audit fixed-penalty covariance and shared GPU strata validation."""
    device = "cuda" if name == "cupy" else "torch"
    X_np, stop_np, event_np = _sample(seed=2287, n=72, p=3)
    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    penalty = 0.4

    model = CoxPH(
        ties="efron",
        penalty=penalty,
        device=device,
        cov_type="nonrobust",
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
    ).fit(X, stop, event)
    objective = cox_counting_process_objective(
        model.coef_, X_np, stop_np, event_np, ties="efron"
    )
    meat = np.asarray(objective["information"], dtype=np.float64)
    derivative = meat + 2.0 * penalty * np.eye(X_np.shape[1])
    bread = np.linalg.inv(derivative)
    expected = bread @ meat @ bread
    covariance_error = float(
        np.max(np.abs(np.asarray(model._var_matrix) - expected))
    )
    curvature_difference = float(
        np.max(np.abs(np.asarray(model._var_matrix) - bread))
    )
    metadata = model._inference_result.metadata

    cv_model = CoxPHCV(
        penalties=[penalty],
        cv=2,
        random_state=19,
        ties="efron",
        device=device,
        compute_inference=True,
        max_iter=80,
    ).fit(X, stop, event)

    strata_np = np.arange(X_np.shape[0], dtype=np.int64) % 2
    strata = (
        xp.asarray(strata_np, dtype=xp.int64)
        if name == "cupy"
        else xp.as_tensor(strata_np, dtype=xp.int64, device="cuda")
    )
    stratified = CoxPH(
        device=device,
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
    ).fit(X, stop, event, strata=strata)

    score = stratified.score(X, stop, event, strata=strata)
    survival, times = stratified.predict_survival(
        X[:4], strata=strata[:4]
    )
    survival_np = _numpy(name, survival)

    def rejection(call):
        try:
            call()
        except ValueError as exc:
            return str(exc)
        return ""

    shape_errors = {
        "scalar": rejection(
            lambda: stratified.score(X, stop, event, strata=strata[0])
        ),
        "two_dimensional": rejection(
            lambda: stratified.score(
                X, stop, event, strata=strata.reshape(-1, 1)
            )
        ),
        "wrong_length": rejection(
            lambda: stratified.score(
                X, stop, event, strata=strata[:-1]
            )
        ),
        "prediction_two_dimensional": rejection(
            lambda: stratified.predict_survival(
                X[:4], strata=strata[:4].reshape(-1, 1)
            )
        ),
    }
    unknown = strata + 10
    unknown_score_error = rejection(
        lambda: stratified.score(X, stop, event, strata=unknown)
    )
    unknown_prediction_error = rejection(
        lambda: stratified.predict_survival(X[:4], strata=unknown[:4])
    )
    missing_score_error = rejection(
        lambda: stratified.score(X, stop, event)
    )

    passed = all(
        (
            covariance_error < 2e-8,
            curvature_difference > 1e-8,
            model.inference_method_ == "m_estimation",
            model.inference_target_ == "penalized_estimating_equation",
            model.penalty_conditioning_ == "fixed_penalty",
            model.penalty_selection_adjusted_ is False,
            metadata["meat_information"]
            == "unpenalized_observed_information",
            metadata["covariance_convention"]
            == "fixed_penalty_model_based_sandwich",
            metadata["score_test_contract"] == "suppressed_penalized_fit",
            not model.score_test_available_,
            cv_model.inference_method_ == "m_estimation",
            cv_model.penalty_selection_adjusted_ is False,
            np.isfinite(score),
            survival_np.shape == (4, int(times.shape[0])),
            np.all(np.isfinite(survival_np)),
            all(
                error == "strata must have shape (n_samples,)"
                for error in shape_errors.values()
            ),
            "unknown scoring stratum" in unknown_score_error,
            "unknown prediction stratum" in unknown_prediction_error,
            "strata is required when scoring" in missing_score_error,
        )
    )
    return {
        "backend": name,
        "penalty": penalty,
        "covariance_contract": "A^-1 J A^-1",
        "covariance_max_abs_error": covariance_error,
        "differs_from_penalized_curvature_inverse": curvature_difference,
        "inference_method": model.inference_method_,
        "inference_target": model.inference_target_,
        "penalty_conditioning": model.penalty_conditioning_,
        "penalty_selection_adjusted": model.penalty_selection_adjusted_,
        "covariance_convention": metadata["covariance_convention"],
        "score_test_contract": metadata["score_test_contract"],
        "cv_inference_method": cv_model.inference_method_,
        "cv_penalty_selection_adjusted": (
            cv_model.penalty_selection_adjusted_
        ),
        "valid_stratified_score": float(score),
        "valid_survival_shape": list(survival_np.shape),
        "shape_errors": shape_errors,
        "unknown_score_error": unknown_score_error,
        "unknown_prediction_error": unknown_prediction_error,
        "missing_score_error": missing_score_error,
        "passed": bool(passed),
    }


def _case_eventless_stratum_survival(name: str, xp) -> dict:
    """Audit the valid zero-baseline survival contract on physical GPU."""
    device = "cuda" if name == "cupy" else "torch"
    rng = np.random.default_rng(2288)
    split = 48
    X_np = rng.normal(size=(64, 1))
    beta = np.array([0.35])
    failure = rng.exponential(scale=np.exp(-(X_np[:split] @ beta))) + 0.05
    censor = rng.exponential(scale=2.0, size=split) + 0.05
    stop_np = np.empty(X_np.shape[0], dtype=np.float64)
    stop_np[:split] = np.minimum(failure, censor)
    stop_np[split:] = rng.uniform(0.1, 3.0, size=X_np.shape[0] - split)
    event_np = np.zeros(X_np.shape[0], dtype=np.float64)
    event_np[:split] = failure <= censor
    event_np[:16] = 1.0
    strata_np = np.zeros(X_np.shape[0], dtype=np.int64)
    strata_np[split:] = 1

    X = _array(name, xp, X_np)
    stop = _array(name, xp, stop_np)
    event = _array(name, xp, event_np)
    strata = (
        xp.asarray(strata_np, dtype=xp.int64)
        if name == "cupy"
        else xp.as_tensor(strata_np, dtype=xp.int64, device="cuda")
    )
    model = CoxPH(
        ties="efron",
        device=device,
        compute_inference=True,
        compute_cindex=False,
        max_iter=100,
    ).fit(X, stop, event, strata=strata)

    explicit_times = np.array(
        [
            0.0,
            np.median(stop_np[event_np == 1.0]),
            np.max(stop_np[event_np == 1.0]) + 1.0,
        ]
    )
    explicit, returned_times = model.predict_survival(
        X[split : split + 3],
        times=explicit_times,
        strata=strata[split : split + 3],
    )
    automatic, automatic_times = model.predict_survival(
        X[split : split + 3],
        strata=strata[split : split + 3],
    )
    mixed_indices = np.array([0, split])
    mixed_X = _array(name, xp, X_np[mixed_indices])
    mixed_strata_np = strata_np[mixed_indices]
    mixed_strata = (
        xp.asarray(mixed_strata_np, dtype=xp.int64)
        if name == "cupy"
        else xp.as_tensor(mixed_strata_np, dtype=xp.int64, device="cuda")
    )
    mixed, mixed_times = model.predict_survival(
        mixed_X, times=explicit_times, strata=mixed_strata
    )

    cv_model = CoxPHCV(
        penalties=[0.2],
        cv=2,
        random_state=23,
        ties="efron",
        device=device,
        compute_inference=True,
        max_iter=100,
        tol=1e-8,
    ).fit(X, stop, event, strata=strata)
    cv_survival, cv_times = cv_model.predict_survival(
        X[split : split + 2],
        times=explicit_times,
        strata=strata[split : split + 2],
    )

    explicit_np = _numpy(name, explicit)
    returned_times_np = _numpy(name, returned_times)
    automatic_np = _numpy(name, automatic)
    automatic_times_np = _numpy(name, automatic_times)
    mixed_np = _numpy(name, mixed)
    mixed_times_np = _numpy(name, mixed_times)
    cv_np = _numpy(name, cv_survival)
    cv_times_np = _numpy(name, cv_times)
    empty_baseline = model._baseline_by_stratum[1]
    cv_empty_baseline = cv_model.estimator_._baseline_by_stratum[1]

    passed = all(
        (
            empty_baseline["time"].shape == (0,),
            empty_baseline["cumulative_hazard"].shape == (0,),
            cv_empty_baseline["time"].shape == (0,),
            explicit_np.shape == (3, explicit_times.size),
            np.all(np.isfinite(explicit_np)),
            np.array_equal(explicit_np, np.ones_like(explicit_np)),
            np.allclose(returned_times_np, explicit_times),
            automatic_np.shape == (3, automatic_times_np.size),
            automatic_times_np.size > 0,
            np.all(np.isfinite(automatic_np)),
            np.array_equal(automatic_np, np.ones_like(automatic_np)),
            mixed_np.shape == (2, mixed_times_np.size),
            np.all(np.isfinite(mixed_np)),
            np.any(mixed_np[0] < 1.0),
            np.array_equal(mixed_np[1], np.ones_like(mixed_np[1])),
            cv_np.shape == (2, explicit_times.size),
            np.all(np.isfinite(cv_np)),
            np.array_equal(cv_np, np.ones_like(cv_np)),
            np.allclose(cv_times_np, explicit_times),
        )
    )
    return {
        "backend": name,
        "baseline_contract": "no failures => cumulative hazard 0 => survival 1",
        "empty_baseline_shape": list(empty_baseline["time"].shape),
        "explicit_times_shape": list(explicit_np.shape),
        "explicit_max_abs_error_from_one": float(
            np.max(np.abs(explicit_np - 1.0))
        ),
        "automatic_times_count": int(automatic_times_np.size),
        "automatic_max_abs_error_from_one": float(
            np.max(np.abs(automatic_np - 1.0))
        ),
        "mixed_shape": list(mixed_np.shape),
        "mixed_eventless_max_abs_error_from_one": float(
            np.max(np.abs(mixed_np[1] - 1.0))
        ),
        "mixed_eventful_min_survival": float(np.min(mixed_np[0])),
        "cv_shape": list(cv_np.shape),
        "cv_max_abs_error_from_one": float(np.max(np.abs(cv_np - 1.0))),
        "passed": bool(passed),
    }


def _case_penalized_cox_cv_and_backend_pin(name: str, xp) -> dict:
    """Audit supported-alpha evidence, final refit, and auto-backend pinning."""
    device = "cuda" if name == "cupy" else "torch"
    X_np, stop_np, event_np = _sample(seed=2291, n=32, p=2)
    event_np[16:20] = 1.0
    X = _array(name, xp, X_np)
    target_np = np.column_stack((stop_np, event_np))
    target = _array(name, xp, target_np)
    alpha_grid = np.array([0.15, 0.03], dtype=np.float64)
    penalty_results = {}

    for penalty in ("l1", "l2", "elasticnet", "scad", "mcp"):
        cv_model = PenalizedGLM_CV(
            loss="cox_ph",
            penalty=penalty,
            alpha_grid=alpha_grid,
            l1_ratio=0.4,
            cv=2,
            random_state=29,
            device=device,
            max_iter=400,
            tol=1e-6,
            loss_kwargs={"ties": "efron"},
        ).fit(X, target)
        direct = PenalizedCoxPHModel(
            penalty=penalty,
            alpha=cv_model.alpha_,
            l1_ratio=0.4,
            ties="efron",
            device=device,
            max_iter=400,
            tol=1e-6,
            compute_inference=False,
        ).fit(X, target)
        mean_scores = np.asarray(
            cv_model.cv_results_["mean_score"], dtype=np.float64
        )
        valid_counts = np.asarray(
            cv_model.cv_results_["valid_score_counts"], dtype=np.int64
        )
        required_count = int(
            cv_model.cv_results_["required_valid_score_count"]
        )
        coefficient_error = float(
            np.max(
                np.abs(
                    np.asarray(cv_model.coef_, dtype=np.float64)
                    - np.asarray(direct.coef_, dtype=np.float64)
                )
            )
        )
        penalty_passed = all(
            (
                cv_model.alpha_ in alpha_grid,
                np.all(np.isfinite(mean_scores)),
                np.all(valid_counts == required_count),
                required_count == 2,
                cv_model.cv_results_["fit_intercept"] is False,
                cv_model.cv_results_["final_refit_class"]
                == "PenalizedCoxPHModel",
                cv_model.intercept_ == 0.0,
                coefficient_error <= 1e-9,
            )
        )
        penalty_results[penalty] = {
            "selected_alpha": float(cv_model.alpha_),
            "mean_partial_likelihood_loss": mean_scores.tolist(),
            "valid_score_counts": valid_counts.tolist(),
            "required_valid_score_count": required_count,
            "final_refit_coefficient_max_abs_error": coefficient_error,
            "passed": bool(penalty_passed),
        }

    automatic_grid_results = {}
    custom_folds = [
        (np.arange(0, 16, dtype=np.int64), np.arange(16, 18, dtype=np.int64)),
        (np.arange(0, 16, dtype=np.int64), np.arange(18, 20, dtype=np.int64)),
    ]
    for case_name, penalty, estimator_ratio, expected_ratio in (
        ("string", "elasticnet", 0.4, 0.4),
        (
            "object",
            ElasticNetPenalty(alpha=9.0, l1_ratio=0.25),
            0.8,
            0.25,
        ),
        ("pure_l2", "elasticnet", 0.0, 0.0),
    ):
        auto_model = PenalizedGLM_CV(
            loss="cox_ph",
            penalty=penalty,
            n_alphas=3,
            l1_ratio=estimator_ratio,
            cv=2,
            cv_splits=custom_folds,
            random_state=29,
            device=device,
            max_iter=400,
            tol=1e-6,
            loss_kwargs={"ties": "efron"},
        ).fit(X, target)
        reference_loss = cox_loss.CoxPartialLikelihoodLoss(ties="efron")
        try:
            gradient = reference_loss.gradient(
                X,
                target,
                _array(name, xp, np.zeros(X_np.shape[1], dtype=np.float64)),
            )
        finally:
            reference_loss.release_fit_cache()
        raw_zero_score = float(np.max(np.abs(_numpy(name, gradient))))
        expected_alpha_max = (
            raw_zero_score / expected_ratio
            if expected_ratio > 0.0
            else raw_zero_score
        )
        actual_alpha_max = float(auto_model.alpha_grid_[0])
        expected_rule = (
            "elasticnet_zero_score_kkt"
            if expected_ratio > 0.0
            else "zero_score_l2_heuristic"
        )
        case_passed = all(
            (
                np.isclose(actual_alpha_max, expected_alpha_max, rtol=1e-10),
                auto_model.cv_results_["alpha_grid_rule"] == expected_rule,
                np.isclose(
                    auto_model.cv_results_["alpha_grid_l1_ratio"],
                    expected_ratio,
                ),
                len(auto_model.cv_results_["fold_indices"]) == 2,
            )
        )
        automatic_grid_results[case_name] = {
            "raw_zero_score_inf_norm": raw_zero_score,
            "l1_ratio": expected_ratio,
            "expected_alpha_max": expected_alpha_max,
            "actual_alpha_max": actual_alpha_max,
            "alpha_grid_rule": auto_model.cv_results_["alpha_grid_rule"],
            "general_disjoint_split_count": len(
                auto_model.cv_results_["fold_indices"]
            ),
            "passed": bool(case_passed),
        }

    class CapturingFoldCountCV(PenalizedGLM_CV):
        def _effective_cv_device(
            self, X_value, penalty_name, n_alphas, *, n_folds=None
        ):
            self.observed_device_sizing_fold_count = n_folds
            return device

    scalar_X_np = np.linspace(-1.0, 1.0, 60).reshape(20, 3)
    scalar_y_np = scalar_X_np @ np.array([0.7, -0.2, 0.4])
    scalar_X = _array(name, xp, scalar_X_np)
    scalar_y = _array(name, xp, scalar_y_np)
    scalar_single_folds = [
        (
            np.arange(5, 20, dtype=np.int64),
            np.arange(0, 5, dtype=np.int64),
        )
    ]
    scalar_four_folds = [
        (
            np.setdiff1d(np.arange(20), validation, assume_unique=True),
            validation,
        )
        for validation in np.array_split(np.arange(20), 4)
    ]
    scalar_generator_iterations = []

    def scalar_one_shot_folds():
        scalar_generator_iterations.append(1)
        if len(scalar_generator_iterations) > 1:
            raise RuntimeError("scalar custom fold generator was consumed twice")
        yield from scalar_four_folds

    scalar_single = CapturingFoldCountCV(
        loss="squared_error",
        penalty="l2",
        alpha_grid=[0.1],
        cv=99,
        cv_splits=scalar_single_folds,
        device="auto",
        max_iter=200,
        tol=1e-7,
    ).fit(scalar_X, scalar_y)
    scalar_generator = CapturingFoldCountCV(
        loss="squared_error",
        penalty="l2",
        alpha_grid=[0.1],
        cv=99,
        cv_splits=scalar_one_shot_folds(),
        device="auto",
        max_iter=200,
        tol=1e-7,
    ).fit(scalar_X, scalar_y)

    scalar_grid_values = np.array(
        [0.2, -1.0, np.nan, 0.0, np.inf, 0.05], dtype=np.float64
    )
    with warnings.catch_warnings(record=True) as filtered_warning_records:
        warnings.simplefilter("always")
        scalar_filtered_grid = PenalizedGLM_CV(
            loss="squared_error",
            penalty="l2",
            alpha_grid=_array(name, xp, scalar_grid_values),
            cv=2,
            random_state=31,
            device=device,
            max_iter=200,
            tol=1e-7,
        ).fit(scalar_X, scalar_y)
    filtered_warning_messages = [
        str(record.message) for record in filtered_warning_records
    ]

    with warnings.catch_warnings(record=True) as default_warning_records:
        warnings.simplefilter("always")
        scalar_default_grid = PenalizedGLM_CV(
            loss="squared_error",
            penalty="l2",
            alpha_grid=_array(
                name,
                xp,
                np.array([-1.0, np.nan, 0.0, np.inf], dtype=np.float64),
            ),
            n_alphas=3,
            cv=2,
            random_state=32,
            device=device,
            max_iter=200,
            tol=1e-7,
        ).fit(scalar_X, scalar_y)
    default_warning_messages = [
        str(record.message) for record in default_warning_records
    ]

    family_rng = np.random.default_rng(2293)
    family_X_np = family_rng.normal(size=(30, 4))
    family_y_np = family_X_np @ np.array([0.8, -0.35, 0.2, 0.1])
    family_y_np += family_rng.normal(scale=0.05, size=family_X_np.shape[0])
    family_X = _array(name, xp, family_X_np)
    family_y = _array(name, xp, family_y_np)
    group_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    family_specs = (
        ("l1", {}),
        ("l2", {}),
        ("elasticnet", {}),
        ("scad", {"a": 3.7}),
        ("mcp", {"gamma": 3.0}),
        (
            "adaptive_l1",
            {"weights": np.ones(4, dtype=np.float64)},
        ),
        ("group_lasso", {"groups": group_ids}),
        ("group_scad", {"groups": group_ids, "a": 3.7}),
        ("group_mcp", {"groups": group_ids, "gamma": 3.0}),
    )
    scalar_penalty_family_results = {}
    for penalty_name, penalty_kwargs in family_specs:
        with warnings.catch_warnings(record=True) as family_warning_records:
            warnings.simplefilter("always")
            family_model = PenalizedGLM_CV(
                loss="squared_error",
                penalty=penalty_name,
                penalty_kwargs=penalty_kwargs,
                alpha_grid=_array(
                    name,
                    xp,
                    np.array(
                        [0.2, -1.0, np.nan, 0.0, 0.05],
                        dtype=np.float64,
                    ),
                ),
                l1_ratio=0.4,
                cv=2,
                random_state=33,
                device=device,
                max_iter=300,
                tol=1e-6,
            ).fit(family_X, family_y)
        family_warnings = [
            str(record.message) for record in family_warning_records
        ]
        family_grid = np.asarray(family_model.alpha_grid_, dtype=np.float64)
        family_scores = np.asarray(
            family_model.cv_results_["all_scores"], dtype=np.float64
        )
        family_passed = all(
            (
                np.array_equal(family_grid, np.array([0.2, 0.05])),
                np.array_equal(
                    family_model.cv_results_["alpha"],
                    np.array([0.2, 0.05]),
                ),
                family_scores.shape == (2, 2),
                np.all(np.isfinite(family_scores)),
                family_model.alpha_ in {0.2, 0.05},
                np.isclose(
                    float(family_model.estimator_.alpha),
                    float(family_model.alpha_),
                ),
                family_model.estimator_.penalty == penalty_name,
                np.all(np.isfinite(np.asarray(family_model.coef_))),
                any("Filtered 3" in value for value in family_warnings),
            )
        )
        scalar_penalty_family_results[penalty_name] = {
            "filtered_grid": family_grid.tolist(),
            "selected_alpha": float(family_model.alpha_),
            "score_shape": list(family_scores.shape),
            "warning_messages": family_warnings,
            "final_refit_penalty": family_model.estimator_.penalty,
            "final_refit_alpha": float(family_model.estimator_.alpha),
            "passed": bool(family_passed),
        }

    invalid_grid_work_calls = []

    class RejectingInvalidGridCV(PenalizedGLM_CV):
        def _effective_cv_device(self, *args, **kwargs):
            invalid_grid_work_calls.append("device")
            raise AssertionError("invalid grid reached device routing")

        def _compute_cv_scores(self, *args, **kwargs):
            invalid_grid_work_calls.append("candidate")
            raise AssertionError("invalid grid reached candidate work")

        def _refit_best(self, *args, **kwargs):
            invalid_grid_work_calls.append("refit")
            raise AssertionError("invalid grid reached final refit")

    invalid_grid_specs = {
        "two_dimensional": _array(
            name,
            xp,
            np.array([[0.2, 0.1]], dtype=np.float64),
        ),
        "mixed_true_float": [True, 0.1],
        "mixed_false_float": [False, 0.1],
        "object_mixed_bool": np.array([True, 0.1], dtype=object),
        "mixed_numeric_string": ["0.2", 0.1],
        "object_numeric_string": np.array(["0.2", 0.1], dtype=object),
    }
    invalid_grid_errors = {}
    for invalid_name, invalid_grid in invalid_grid_specs.items():
        try:
            RejectingInvalidGridCV(
                loss="squared_error",
                penalty="l2",
                alpha_grid=invalid_grid,
                cv=2,
                device=device,
            ).fit(scalar_X, scalar_y)
        except ValueError as exc:
            invalid_grid_errors[invalid_name] = str(exc)

    filtered_grid_np = np.asarray(
        scalar_filtered_grid.alpha_grid_, dtype=np.float64
    )
    default_grid_np = np.asarray(
        scalar_default_grid.alpha_grid_, dtype=np.float64
    )
    scalar_alpha_grid_passed = all(
        (
            np.array_equal(filtered_grid_np, np.array([0.2, 0.05])),
            scalar_filtered_grid.alpha_ in {0.2, 0.05},
            np.all(np.isfinite(np.asarray(scalar_filtered_grid.coef_))),
            any("Filtered 4" in value for value in filtered_warning_messages),
            default_grid_np.shape == (3,),
            np.all(np.isfinite(default_grid_np)),
            np.all(default_grid_np > 0.0),
            scalar_default_grid.alpha_ in set(default_grid_np),
            any(
                "automatically generated default" in value
                for value in default_warning_messages
            ),
            all(
                result["passed"]
                for result in scalar_penalty_family_results.values()
            ),
            set(invalid_grid_errors) == set(invalid_grid_specs),
            "one-dimensional"
            in invalid_grid_errors.get("two_dimensional", ""),
            "not booleans"
            in invalid_grid_errors.get("mixed_true_float", ""),
            "not booleans"
            in invalid_grid_errors.get("mixed_false_float", ""),
            "not booleans"
            in invalid_grid_errors.get("object_mixed_bool", ""),
            "strings or bytes"
            in invalid_grid_errors.get("mixed_numeric_string", ""),
            "strings or bytes"
            in invalid_grid_errors.get("object_numeric_string", ""),
            invalid_grid_work_calls == [],
        )
    )

    support_rng = np.random.default_rng(2292)
    support_X_np = support_rng.normal(size=(36, 2))
    support_event_np = np.zeros(36, dtype=np.float64)
    support_event_np[:8] = 1.0
    support_target_np = np.column_stack(
        (np.arange(1.0, 37.0), support_event_np)
    )
    support_folds = [
        (
            np.array([0, 1, 2, 3, *range(8, 20)], dtype=np.int64),
            np.array([4, 5, 6, 7, *range(20, 24)], dtype=np.int64),
        )
    ]
    for start in range(20, 36, 4):
        support_folds.append(
            (
                np.arange(0, 20, dtype=np.int64),
                np.arange(start, start + 4, dtype=np.int64),
            )
        )
    support_model = CapturingFoldCountCV(
        loss="cox_ph",
        penalty="l2",
        alpha_grid=[0.1],
        cv=99,
        cv_splits=support_folds,
        device="auto",
        max_iter=400,
        tol=1e-6,
    ).fit(
        _array(name, xp, support_X_np),
        _array(name, xp, support_target_np),
    )
    public_fold_routing_passed = all(
        (
            scalar_single.observed_device_sizing_fold_count == 1,
            scalar_single.cv_results_["device_sizing_fold_count"] == 1,
            scalar_generator.observed_device_sizing_fold_count == 4,
            scalar_generator.cv_results_["device_sizing_fold_count"] == 4,
            scalar_generator_iterations == [1],
            len(support_folds) == 5,
            support_model.observed_device_sizing_fold_count == 1,
            support_model.cv_results_["n_effective_folds"] == 1,
            support_model.cv_results_["device_sizing_fold_count"] == 1,
            np.array_equal(
                support_model.cv_results_["fold_valid"],
                np.array([True, False, False, False, False]),
            ),
        )
    )

    fold_work_model = PenalizedGLM_CV(
        loss="cox_ph",
        penalty="l2",
        n_alphas=100,
        cv=99,
        device="auto",
    )
    fold_work_X = np.empty((2000, 100), dtype=np.float64)
    single_fold_device = fold_work_model._effective_cv_device(
        fold_work_X,
        "l2",
        100,
        n_folds=1,
    )
    repeated_fold_device = fold_work_model._effective_cv_device(
        fold_work_X,
        "l2",
        100,
        n_folds=5,
    )
    fold_work_passed = all(
        (
            single_fold_device == "cpu",
            repeated_fold_device == "torch",
        )
    )

    set_device(device)
    try:
        pinned_model = CoxPH(
            device="auto",
            ties="efron",
            compute_inference=False,
            compute_cindex=False,
            max_iter=100,
        ).fit(X, target)
        fitted_backend = pinned_model._fitted_backend_name
        effective_device = pinned_model.effective_device_
        set_device("cpu")
        pinned_prediction = pinned_model.predict_risk_score(X[:4])
        pinned_prediction_np = _numpy(name, pinned_prediction)
        pinned_score = float(pinned_model.score(X[:12], target[:12]))
    finally:
        set_device("auto")

    expected_backend = name
    expected_effective = device
    backend_pin_passed = all(
        (
            fitted_backend == expected_backend,
            effective_device == expected_effective,
            type(pinned_prediction).__module__.startswith(expected_backend),
            np.all(np.isfinite(pinned_prediction_np)),
            np.isfinite(pinned_score),
        )
    )
    passed = all(
        (
            backend_pin_passed,
            all(result["passed"] for result in penalty_results.values()),
            all(result["passed"] for result in automatic_grid_results.values()),
            fold_work_passed,
            public_fold_routing_passed,
            scalar_alpha_grid_passed,
        )
    )
    return {
        "backend": name,
        "penalty_families": penalty_results,
        "automatic_elasticnet_grid": automatic_grid_results,
        "scalar_alpha_grid": {
            "input_backend": name,
            "contract": (
                "filter non-positive/non-finite values before routing; "
                "regenerate defaults when none remain; reject malformed shape"
            ),
            "filtered_grid": filtered_grid_np.tolist(),
            "filtered_selected_alpha": float(scalar_filtered_grid.alpha_),
            "filtered_warning_messages": filtered_warning_messages,
            "default_grid": default_grid_np.tolist(),
            "default_selected_alpha": float(scalar_default_grid.alpha_),
            "default_warning_messages": default_warning_messages,
            "penalty_families": scalar_penalty_family_results,
            "malformed_errors": invalid_grid_errors,
            "malformed_work_calls": invalid_grid_work_calls,
            "passed": bool(scalar_alpha_grid_passed),
        },
        "public_fold_routing": {
            "scalar_list_observed_count": (
                scalar_single.observed_device_sizing_fold_count
            ),
            "scalar_generator_observed_count": (
                scalar_generator.observed_device_sizing_fold_count
            ),
            "scalar_generator_iterations": len(
                scalar_generator_iterations
            ),
            "cox_normalized_fold_count": len(support_folds),
            "cox_evaluable_fold_count": int(
                support_model.cv_results_["n_effective_folds"]
            ),
            "cox_observed_device_sizing_fold_count": (
                support_model.observed_device_sizing_fold_count
            ),
            "passed": bool(public_fold_routing_passed),
        },
        "actual_fold_count_auto_device": {
            "configured_cv": 99,
            "n_samples": 2000,
            "n_features": 100,
            "n_alphas": 100,
            "single_fold_device": single_fold_device,
            "five_fold_device": repeated_fold_device,
            "contract": "generic fallback uses supplied work-fold count",
            "passed": bool(fold_work_passed),
        },
        "selection_contract": (
            "finite held-out Cox partial likelihood from every evaluable fold"
        ),
        "final_refit_contract": "PenalizedCoxPHModel without intercept",
        "fitted_backend": fitted_backend,
        "effective_device": effective_device,
        "prediction_backend_after_global_device_change": (
            type(pinned_prediction).__module__.split(".")[0]
        ),
        "score_after_global_device_change": pinned_score,
        "backend_pin_passed": bool(backend_pin_passed),
        "passed": bool(passed),
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-targeted-tests", action="store_true")
    args = parser.parse_args()
    head = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    report = {
        "schema_version": 21,
        "validation_tier": "remote-full",
        "source_commit": head,
        "source_clean": not dirty,
        "source_sha256": {
            path: _sha256(REPO_ROOT / path) for path in SOURCE_FILES
        },
        "python": sys.version,
        "numpy": np.__version__,
        "backends": {},
        "gate_failures": [],
        "command": (
            "python dev/benchmarks/benchmark_cox_boundary_gpu.py "
            "--output <path>"
            + (" --run-targeted-tests" if args.run_targeted_tests else "")
        ),
    }
    for name in ("cupy", "torch"):
        try:
            xp = _backend(name)
            device_name = (
                xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                if name == "cupy"
                else xp.cuda.get_device_name(0)
            )
            cases = {
                "public_boundary": _case_boundary(name, xp),
                "cv_device_normalization": _case_cv(name, xp),
                "ordinary_cv_preparation": _case_ordinary_cv_preparation(
                    name, xp
                ),
                "prepared_state_and_packed_target": (
                    _case_prepared_state_and_packed_target(name, xp)
                ),
                "prediction_fast_path_and_fit_controls": (
                    _case_prediction_fast_path_and_fit_controls(name, xp)
                ),
                "hazard_ratio_boundary": _case_hazard_ratio_boundary(name, xp),
                "single_group_workspace": _case_workspace(name, xp),
                "wide_workspace_route": _case_wide_workspace_route(name, xp),
                "concordance_boundaries": _case_concordance_boundaries(name, xp),
                "completion_contract": _case_completion_contract(name, xp),
                "robust_inference_units": _case_robust_inference_units(
                    name, xp
                ),
                "penalized_inference_and_strata": (
                    _case_penalized_inference_and_strata(name, xp)
                ),
                "eventless_stratum_survival": (
                    _case_eventless_stratum_survival(name, xp)
                ),
                "penalized_cox_cv_and_backend_pin": (
                    _case_penalized_cox_cv_and_backend_pin(name, xp)
                ),
            }
            report["backends"][name] = {
                "version": xp.__version__,
                "device": device_name,
                "cases": cases,
            }
            for case_name, case in cases.items():
                if not case["passed"]:
                    report["gate_failures"].append(f"{name}:{case_name}")
        except Exception as exc:
            report["backends"][name] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            report["gate_failures"].append(f"{name}:execution")

    if args.run_targeted_tests:
        test_command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *TARGETED_TEST_FILES,
        ]
        test_env = os.environ.copy()
        test_env["STATGPU_REQUIRE_PHYSICAL_GPU"] = "1"
        completed = subprocess.run(
            test_command,
            cwd=REPO_ROOT,
            env=test_env,
            capture_output=True,
            text=True,
            check=False,
        )
        test_output = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        summary_line = next(
            (line for line in reversed(test_output.splitlines()) if " passed" in line),
            "",
        )
        passed_match = re.search(r"(\d+) passed", summary_line)
        report["targeted_tests"] = {
            "command": "STATGPU_REQUIRE_PHYSICAL_GPU=1 "
            + " ".join(test_command),
            "returncode": completed.returncode,
            "passed_count": (
                int(passed_match.group(1)) if passed_match is not None else None
            ),
            "summary": summary_line,
            "output_tail": "\n".join(test_output.splitlines()[-20:]),
            "passed": completed.returncode == 0,
        }
        if completed.returncode != 0:
            report["gate_failures"].append("targeted_tests")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["gate_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
