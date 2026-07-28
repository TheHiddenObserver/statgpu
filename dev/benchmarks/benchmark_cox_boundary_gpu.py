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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu._config import Device  # noqa: E402
from statgpu.linear_model import PenalizedCoxPHModel  # noqa: E402
from statgpu.survival import CoxPH, CoxPHCV  # noqa: E402
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
    ".github/workflows/test.yml",
    "statgpu/backends/_array_ops.py",
    "statgpu/backends/_utils.py",
    "statgpu/linear_model/penalized/_penalized_cox.py",
    "statgpu/survival/__init__.py",
    "statgpu/survival/_cox.py",
    "statgpu/survival/_cox_cv.py",
    "statgpu/survival/_cox_fit_adapter.py",
    "statgpu/survival/_cox_legacy.py",
    "statgpu/survival/_concordance.py",
    "statgpu/survival/_cox_score.py",
    "statgpu/survival/_risk_sets.py",
    "dev/benchmarks/benchmark_cox_boundary_gpu.py",
    "dev/tests/test_pr80_complete_review_cycle.py",
    "dev/tests/test_pr80_completion_contract_followup.py",
    "dev/tests/test_pr80_constructor_boundaries.py",
    "dev/tests/test_pr80_workspace_estimator.py",
    "dev/tests/test_pr80_fit_boundary.py",
    "dev/tests/test_pr80_cv_fit_boundary.py",
    "dev/tests/test_pr80_cox_stability_review.py",
)

TARGETED_TEST_FILES = (
    "dev/tests/test_pr80_complete_review_cycle.py",
    "dev/tests/test_pr80_completion_contract_followup.py",
    "dev/tests/test_pr80_constructor_boundaries.py",
    "dev/tests/test_pr80_workspace_estimator.py",
    "dev/tests/test_pr80_fit_boundary.py",
    "dev/tests/test_pr80_cv_fit_boundary.py",
    "dev/tests/test_pr80_cox_stability_review.py",
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

    def reject_public_host_copy(*_args, **_kwargs):
        raise AssertionError("packed target crossed the public host boundary")

    model._to_numpy = reject_public_host_copy
    started = time.perf_counter()
    model.fit(X, target)
    _sync(name, xp)
    fit_seconds = time.perf_counter() - started

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
    packed_target_stayed_native = model._entry is None
    finite = bool(np.all(np.isfinite(model.coef_)))

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
        "packed_target_stayed_native": packed_target_stayed_native,
        "complex_prediction_rejected": complex_rejected,
        "device_normalized": device_normalized,
        "failed_refit_cleared": failed_refit_cleared,
        "constructor_truthy_strings_rejected": constructor_rejections,
        "finite": finite,
        "passed": all(
            (
                packed_target_stayed_native,
                complex_rejected,
                device_normalized,
                failed_refit_cleared,
                all(constructor_rejections.values()),
                finite,
            )
        ),
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
        penalties=np.array([0.1]),
        cv=2,
        device="cpu",
        compute_inference=False,
        max_iter=60,
    )
    model.set_params(device=device)
    started = time.perf_counter()
    model.fit(
        _array(name, xp, X_np),
        _array(name, xp, stop_np),
        _array(name, xp, event_np),
    )
    _sync(name, xp)
    fit_seconds = time.perf_counter() - started
    final_refit_skips_cindex = (
        model.estimator_.compute_cindex is False
        and model.estimator_.concordance_ is None
    )
    passed = (
        model.device is expected
        and model.estimator_ is not None
        and model.estimator_.device is expected
        and model.effective_device_ == device
        and bool(np.all(np.isfinite(model.coef_)))
        and all(constructor_rejections.values())
        and final_refit_skips_cindex
    )
    return {
        "backend": name,
        "fit_seconds": fit_seconds,
        "effective_device": model.effective_device_,
        "constructor_truthy_strings_rejected": constructor_rejections,
        "final_refit_skips_training_cindex": final_refit_skips_cindex,
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
    direct_backend_imports_absent = (
        "import cupy" not in dispatch_source
        and "import torch" not in dispatch_source
    )
    import_time_adapter_absent = CoxPH.fit.__module__ == "statgpu.survival._cox"
    legacy_mixin_isolated = all(
        method not in CoxPH.__dict__
        and getattr(CoxPH, method) is getattr(_LegacyCoxReferenceMixin, method)
        for method in CoxPH._legacy_reference_methods
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
            direct_backend_imports_absent,
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
        "direct_backend_imports_absent": direct_backend_imports_absent,
        "import_time_adapter_absent": import_time_adapter_absent,
        "legacy_mixin_isolated": legacy_mixin_isolated,
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
        "schema_version": 4,
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
                "single_group_workspace": _case_workspace(name, xp),
                "wide_workspace_route": _case_wide_workspace_route(name, xp),
                "concordance_boundaries": _case_concordance_boundaries(name, xp),
                "completion_contract": _case_completion_contract(name, xp),
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
