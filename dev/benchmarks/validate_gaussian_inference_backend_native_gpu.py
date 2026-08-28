#!/usr/bin/env python3
"""Physical CuPy/Torch acceptance for issue #127 Gaussian inference.

Canonical evidence requires both GPU backends, exact source identity, a clean
candidate tree, explicit validation tier, backend/device provenance, numerical
parity, a behavioral pre-reporting host-transfer guard, and the maintained
small-df tail boundary.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.linear_model import LinearRegression, PenalizedGeneralizedLinearModel, RidgeCV
import statgpu.inference._distributions_backend as _dist_module
import statgpu.linear_model._gaussian_inference as _gi_module
import statgpu.linear_model.penalized._base as _pglm_base_module
from statgpu.linear_model._gaussian_inference import (
    build_gaussian_fit_state,
    compute_gaussian_inference,
    robust_covariance_gpu,
    robust_covariance_numpy,
)

SCHEMA_VERSION = 5
_REQUIRED_BACKENDS = ("cupy", "torch")
_VALIDATION_TIERS = ("local-minimal", "local-full", "remote-full")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _clean_worktree() -> bool:
    return _git("status", "--porcelain") == ""


def _validate_backend_list(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if values != _REQUIRED_BACKENDS:
        raise ValueError(
            "--backends must be exactly 'cupy,torch' in that order for canonical acceptance"
        )
    return values


def _gpu_model() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        return output[0].strip() if output else "unknown"
    except Exception:
        return "unknown"


def _backend_runtime(backend: str):
    if backend == "cupy":
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError("CuPy imported but no CUDA device is available")
        device_id = int(cp.cuda.runtime.getDevice())
        return cp, f"cuda:{device_id}", cp.__version__

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch imported but CUDA is unavailable")
    device_id = int(torch.cuda.current_device())
    return torch, f"cuda:{device_id}", torch.__version__


def _as_backend(value, backend: str):
    if backend == "cupy":
        import cupy as cp

        return cp.asarray(value, dtype=cp.float64)
    import torch

    return torch.as_tensor(
        np.asarray(value),
        dtype=torch.float64,
        device=f"cuda:{torch.cuda.current_device()}",
    )


def _assert_same_native_device(value, backend: str, expected_device: str):
    if backend == "cupy":
        import cupy as cp

        if not isinstance(value, cp.ndarray):
            raise AssertionError(f"expected CuPy ndarray, got {type(value)!r}")
        actual = f"cuda:{int(value.device.id)}"
    else:
        import torch

        if not isinstance(value, torch.Tensor):
            raise AssertionError(f"expected Torch tensor, got {type(value)!r}")
        actual = str(value.device)
    if actual != expected_device:
        raise AssertionError(f"device mismatch: expected {expected_device}, got {actual}")


def _problem(seed: int = 127):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(96, 5))
    beta = np.asarray([0.8, -0.35, 0.2, 0.55, -0.1])
    y = -0.4 + X @ beta + rng.normal(scale=0.3, size=X.shape[0])
    weights = np.linspace(0.5, 1.75, X.shape[0])
    return X, y, weights


def _max_error(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0



def _distribution_backend_name(distribution) -> str:
    sf_name = type(getattr(distribution, "_sf", None)).__name__
    return {
        "CuPySpecialFunctions": "cupy",
        "TorchSpecialFunctions": "torch",
        "ScipySpecialFunctions": "numpy",
    }.get(sf_name, sf_name or "unknown")


def _linear_regression_case(backend: str, concrete_device: str, cov_type: str):
    X, y, weights = _problem(seed=131)
    sw = weights if cov_type == "hc3" else None
    cpu = LinearRegression(
        device="cpu",
        compute_inference=True,
        cov_type=cov_type,
        hac_maxlags=2,
    ).fit(X, y, sample_weight=sw)

    X_native = _as_backend(X, backend)
    y_native = _as_backend(y, backend)
    sw_native = None if sw is None else _as_backend(sw, backend)
    _assert_same_native_device(X_native, backend, concrete_device)
    _assert_same_native_device(y_native, backend, concrete_device)

    distribution_backends = []
    helper_calls = {"count": 0}
    real_resolve = _dist_module.DistributionProxy._resolve

    def guarded_resolve(proxy, kwargs, *arrays):
        distribution = real_resolve(proxy, kwargs, *arrays)
        resolved = _distribution_backend_name(distribution)
        distribution_backends.append(resolved)
        if resolved != backend:
            raise AssertionError(
                f"LinearRegression {cov_type} distribution fallback: "
                f"expected={backend}, resolved={resolved}"
            )
        return distribution

    _dist_module.DistributionProxy._resolve = guarded_resolve
    if backend == "cupy":
        import statgpu.backends._gpu_inference_cupy as inference_module
        helper_name = "compute_inference_gpu"
    else:
        import statgpu.backends._gpu_inference_torch as inference_module
        helper_name = "compute_inference_torch"
    real_helper = getattr(inference_module, helper_name)

    def guarded_helper(X_design, resid, scale, df_resid, params, *args, **kwargs):
        _assert_same_native_device(X_design, backend, concrete_device)
        _assert_same_native_device(resid, backend, concrete_device)
        _assert_same_native_device(params, backend, concrete_device)
        helper_calls["count"] += 1
        return real_helper(X_design, resid, scale, df_resid, params, *args, **kwargs)

    setattr(inference_module, helper_name, guarded_helper)
    try:
        gpu = LinearRegression(
            device="cuda" if backend == "cupy" else "torch",
            compute_inference=True,
            cov_type=cov_type,
            hac_maxlags=2,
        ).fit(X_native, y_native, sample_weight=sw_native)
    finally:
        setattr(inference_module, helper_name, real_helper)
        _dist_module.DistributionProxy._resolve = real_resolve

    result = gpu._inference_result
    if result is None:
        raise AssertionError("LinearRegression GPU inference result is missing")
    meta = dict(result.metadata)
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"LinearRegression backend provenance mismatch: {meta}")
    if str(meta.get("numerical_device", "")) != concrete_device:
        raise AssertionError(f"LinearRegression device provenance mismatch: {meta}")
    if meta.get("reporting_boundary") != "post_numerical_inference":
        raise AssertionError(f"LinearRegression reporting boundary missing: {meta}")
    if cov_type == "nonrobust" and helper_calls["count"] != 1:
        raise AssertionError(
            f"LinearRegression {backend} nonrobust helper count={helper_calls['count']}"
        )
    if cov_type != "nonrobust" and not distribution_backends:
        raise AssertionError("LinearRegression distribution backend was not observed")

    errors = {
        "coef": _max_error(gpu.coef_, cpu.coef_),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    limits = {
        "coef": 2e-6,
        "bse": 2e-5,
        "statistic": 2e-4,
        "pvalue": 2e-4,
        "ci": 5e-5,
    }
    for key, limit in limits.items():
        if not np.isfinite(errors[key]) or errors[key] > limit:
            raise AssertionError(
                f"LinearRegression {backend} {cov_type} {key} error "
                f"{errors[key]:.3e} exceeds {limit:.3e}"
            )
    return {
        "case": f"linear_regression_{cov_type}",
        "requested_backend": backend,
        "executed_backend": meta.get("numerical_backend"),
        "executed_inference_backend": meta.get("numerical_backend"),
        "executed_inference_device": meta.get("numerical_device"),
        "reporting_backend": meta.get("reporting_backend"),
        "reporting_boundary": meta.get("reporting_boundary"),
        "distribution_backends": distribution_backends,
        "native_nonrobust_helper_calls": int(helper_calls["count"]),
        "errors": errors,
        "limits": limits,
        "no_silent_fallback": True,
        "status": "success",
    }


def _cupy_nonrank_failure_case(concrete_device: str):
    import cupy as cp
    from statgpu.backends._gpu_inference_cupy import compute_inference_gpu

    X = _as_backend(np.column_stack([np.ones(8), np.arange(8.0)]), "cupy")
    resid = _as_backend(np.linspace(-0.2, 0.2, 8), "cupy")
    params = _as_backend(np.asarray([0.5, 0.1]), "cupy")
    _assert_same_native_device(X, "cupy", concrete_device)

    real_cholesky = cp.linalg.cholesky

    def synthetic_nonrank_failure(*args, **kwargs):
        raise RuntimeError("synthetic CUDA out-of-memory sentinel")

    cp.linalg.cholesky = synthetic_nonrank_failure
    try:
        try:
            compute_inference_gpu(X, resid, 1.0, 6, params)
        except RuntimeError as exc:
            if "synthetic CUDA out-of-memory sentinel" not in str(exc):
                raise
        else:
            raise AssertionError(
                "CuPy non-rank linalg failure was swallowed by pseudoinverse recovery"
            )
    finally:
        cp.linalg.cholesky = real_cholesky

    return {
        "case": "cupy_nonrank_failure_fail_closed",
        "requested_backend": "cupy",
        "executed_inference_backend": "cupy",
        "executed_inference_device": concrete_device,
        "nonrank_failure_propagated": True,
        "status": "success",
    }

def _fit_pglm(X, y, *, device: str, cov_type: str, sample_weight=None):
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.05,
        fit_intercept=True,
        device=device,
        solver="fista",
        max_iter=5000,
        tol=1e-10,
        compute_inference=True,
        cov_type=cov_type,
        hac_maxlags=2,
    )
    model.fit(X, y, sample_weight=sample_weight)
    if model._inference_result is None:
        raise AssertionError("PGLM L2 inference result is missing")
    return model


def _ridge_covariance_from_state(state, cov_type: str, backend: str, alpha: float):
    ridge_alpha = float(state.normalization) * float(alpha)
    X = state.X_design
    k = int(X.shape[1])
    if backend == "numpy":
        penalty = np.eye(k, dtype=np.float64) * ridge_alpha
        penalty[0, 0] = 0.0
        XtX = X.T @ X
        bread_inv = np.linalg.pinv(XtX + penalty)
        if cov_type == "nonrobust":
            return float(np.asarray(state.scale)) * (bread_inv @ XtX @ bread_inv)
        return robust_covariance_numpy(
            X,
            state.resid,
            bread_inv,
            cov_type,
            hac_maxlags=2,
            df_resid=state.df_resid,
        )

    if backend == "cupy":
        import cupy as xp
    else:
        import torch as xp
    XtX = X.T @ X
    if backend == "torch":
        penalty_diag = xp.full((k,), ridge_alpha, dtype=X.dtype, device=X.device)
        penalty_diag[0] = 0.0
        bread_inv = xp.linalg.pinv(XtX + xp.diag(penalty_diag))
    else:
        penalty_diag = xp.full((k,), ridge_alpha, dtype=X.dtype)
        penalty_diag[0] = 0.0
        bread_inv = xp.linalg.pinv(XtX + xp.diag(penalty_diag))
    if cov_type == "nonrobust":
        return state.scale * (bread_inv @ XtX @ bread_inv)
    return robust_covariance_gpu(
        X,
        state.resid,
        bread_inv,
        cov_type,
        xp,
        hac_maxlags=2,
        df_resid=state.df_resid,
    )


def _consumer_case(backend: str, cov_type: str, weighted: bool):
    X, y, weights = _problem()
    sw = weights if weighted else None
    cpu = _fit_pglm(X, y, device="cpu", cov_type=cov_type, sample_weight=sw)

    X_gpu = _as_backend(X, backend)
    y_gpu = _as_backend(y, backend)
    sw_gpu = None if sw is None else _as_backend(sw, backend)
    device_arg = "cuda" if backend == "cupy" else "torch"
    gpu = _fit_pglm(
        X_gpu,
        y_gpu,
        device=device_arg,
        cov_type=cov_type,
        sample_weight=sw_gpu,
    )

    actual_backend = str(getattr(gpu, "_selected_backend_name", "")).lower()
    if actual_backend != backend:
        raise AssertionError(
            f"fit backend mismatch: requested={backend}, executed={actual_backend}"
        )
    meta = dict(gpu._inference_result.metadata)
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"inference backend mismatch: {meta}")
    inference_device = str(meta.get("numerical_device", ""))
    if not inference_device.startswith("cuda:"):
        raise AssertionError(f"inference device is not concrete CUDA: {meta}")
    if meta.get("reporting_boundary") != "post_numerical_inference":
        raise AssertionError(f"reporting boundary metadata missing: {meta}")

    cpu_state = build_gaussian_fit_state(
        X,
        y,
        cpu.coef_,
        cpu.intercept_,
        True,
        sample_weight=sw,
        backend="numpy",
    )
    gpu_state = build_gaussian_fit_state(
        X_gpu,
        y_gpu,
        gpu.coef_,
        gpu.intercept_,
        True,
        sample_weight=sw_gpu,
        backend=backend,
        device=inference_device,
    )
    gpu_cov = _ridge_covariance_from_state(gpu_state, cov_type, backend, 0.05)
    cpu_cov = _ridge_covariance_from_state(cpu_state, cov_type, "numpy", 0.05)

    errors = {
        "coef": _max_error(gpu.coef_, cpu.coef_),
        "covariance": _max_error(_to_numpy(gpu_cov), cpu_cov),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    limits = {
        "coef": 2e-6,
        "covariance": 5e-6,
        "bse": 2e-5,
        "statistic": 2e-4,
        "pvalue": 2e-4,
        "ci": 5e-5,
    }
    for key, limit in limits.items():
        if not np.isfinite(errors[key]) or errors[key] > limit:
            raise AssertionError(
                f"{backend} {cov_type} weighted={weighted} {key} error "
                f"{errors[key]:.3e} exceeds {limit:.3e}"
            )
    return {
        "case": f"pglm_{cov_type}_{'weighted' if weighted else 'unweighted'}",
        "requested_backend": backend,
        "executed_backend": actual_backend,
        "executed_inference_backend": meta.get("numerical_backend"),
        "executed_inference_device": inference_device,
        "reporting_backend": meta.get("reporting_backend"),
        "reporting_boundary": meta.get("reporting_boundary"),
        "no_silent_fallback": True,
        "errors": errors,
        "limits": limits,
        "status": "success",
    }


def _host_transfer_case(backend: str, concrete_device: str):
    """Behaviorally prove no Gaussian/PGLM host snapshot precedes distribution work."""
    X, y, _ = _problem(seed=130)
    X_native = _as_backend(X, backend)
    y_native = _as_backend(y, backend)
    _assert_same_native_device(X_native, backend, concrete_device)
    _assert_same_native_device(y_native, backend, concrete_device)

    phase = {
        "reporting_allowed": False,
        "gi_snapshots": 0,
        "pglm_snapshots": 0,
        "reference_distribution_completed": False,
    }
    real_gi_to_numpy = _gi_module._to_numpy
    real_pglm_to_numpy = _pglm_base_module._to_numpy
    real_reference = _gi_module.two_sided_reference_inference

    def guarded_gi_to_numpy(value):
        if not phase["reporting_allowed"]:
            raise AssertionError(
                "Gaussian numerical inference attempted a host snapshot before "
                "reference-distribution work completed"
            )
        phase["gi_snapshots"] += 1
        return real_gi_to_numpy(value)

    def guarded_pglm_to_numpy(value):
        if not phase["reporting_allowed"]:
            raise AssertionError(
                "PGLM reporting state attempted a host snapshot before numerical "
                "Gaussian inference completed"
            )
        phase["pglm_snapshots"] += 1
        return real_pglm_to_numpy(value)

    def guarded_reference(
        statistic_abs,
        *,
        distribution,
        alpha,
        backend: str,
        xp,
        df=None,
        device=None,
    ):
        if backend != requested_backend:
            raise AssertionError(
                f"reference distribution backend mismatch: {backend} != {requested_backend}"
            )
        _assert_same_native_device(statistic_abs, requested_backend, concrete_device)
        result = real_reference(
            statistic_abs,
            distribution=distribution,
            alpha=alpha,
            backend=backend,
            xp=xp,
            df=df,
            device=device,
        )
        phase["reference_distribution_completed"] = True
        phase["reporting_allowed"] = True
        return result

    requested_backend = backend
    _gi_module._to_numpy = guarded_gi_to_numpy
    _pglm_base_module._to_numpy = guarded_pglm_to_numpy
    _gi_module.two_sided_reference_inference = guarded_reference
    try:
        device_arg = "cuda" if backend == "cupy" else "torch"
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error",
            penalty="l2",
            alpha=0.05,
            fit_intercept=True,
            device=device_arg,
            compute_inference=True,
            cov_type="nonrobust",
        )
        model._penalty = model._resolve_penalty()
        model._selected_backend_name = backend
        model.coef_ = np.asarray([0.8, -0.35, 0.2, 0.55, -0.1])
        model.intercept_ = -0.4
        model._compute_post_fit_gaussian_inference(X_native, y_native)
    finally:
        _gi_module._to_numpy = real_gi_to_numpy
        _pglm_base_module._to_numpy = real_pglm_to_numpy
        _gi_module.two_sided_reference_inference = real_reference

    if not phase["reference_distribution_completed"]:
        raise AssertionError("reference distribution did not complete on the requested backend")
    if phase["gi_snapshots"] <= 0 or phase["pglm_snapshots"] <= 0:
        raise AssertionError(
            "host-transfer guard did not observe the established post-inference reporting snapshot"
        )
    result = model._inference_result
    if result is None:
        raise AssertionError("host-transfer case did not publish inference result")
    meta = dict(result.metadata)
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"host-transfer case backend mismatch: {meta}")
    if str(meta.get("numerical_device", "")) != concrete_device:
        raise AssertionError(f"host-transfer case device mismatch: {meta}")

    return {
        "case": "host_transfer_provenance",
        "requested_backend": backend,
        "executed_inference_backend": meta.get("numerical_backend"),
        "executed_inference_device": meta.get("numerical_device"),
        "reporting_boundary": meta.get("reporting_boundary"),
        "pre_reporting_host_transfers": 0,
        "gaussian_reporting_snapshots": int(phase["gi_snapshots"]),
        "pglm_reporting_snapshots": int(phase["pglm_snapshots"]),
        "reference_distribution_completed_on_backend": True,
        "instrumented_modules": [
            "statgpu.linear_model._gaussian_inference",
            "statgpu.linear_model.penalized._base",
        ],
        "no_silent_fallback": True,
        "status": "success",
    }


def _functional_edge_cases(backend: str, concrete_device: str):
    # Rank-deficient native inference and NumPy parity.
    x = np.arange(1.0, 9.0)
    design_np = np.column_stack([np.ones_like(x), x, x])
    params_np = np.asarray([0.5, 0.25, 0.25])
    resid_np = np.asarray([0.2, -0.1, 0.05, -0.15, 0.1, -0.1, 0.05, -0.05])
    scale = float(np.sum(resid_np**2) / (len(x) - 3))
    design = _as_backend(design_np, backend)
    params = _as_backend(params_np, backend)
    resid = _as_backend(resid_np, backend)
    _assert_same_native_device(design, backend, concrete_device)
    result = compute_gaussian_inference(
        design,
        params,
        resid,
        scale,
        len(x) - 3,
        "nonrobust",
        backend=backend,
        device=concrete_device,
    )
    rank_ref = compute_gaussian_inference(
        design_np,
        params_np,
        resid_np,
        scale,
        len(x) - 3,
        "nonrobust",
        backend="numpy",
    )
    if result.metadata.get("numerical_device") != concrete_device:
        raise AssertionError("rank-deficient case lost concrete inference device")
    rank_errors = {
        "bse": _max_error(result.bse, rank_ref.bse),
        "statistic": _max_error(result.tvalues, rank_ref.tvalues),
        "pvalue": _max_error(result.pvalues, rank_ref.pvalues),
        "ci": _max_error(result.conf_int, rank_ref.conf_int),
    }

    # Multi-target numerical state stays native until the one reporting snapshot.
    X, y, _ = _problem(seed=128)
    coef = np.column_stack([
        np.asarray([0.8, -0.35, 0.2, 0.55, -0.1]),
        np.asarray([-0.2, 0.4, 0.1, -0.3, 0.25]),
    ])
    intercept = np.asarray([-0.4, 0.2])
    y2 = np.column_stack([y, intercept[1] + X @ coef[:, 1] + 0.05])
    state = build_gaussian_fit_state(
        _as_backend(X, backend),
        _as_backend(y2, backend),
        coef,
        intercept,
        True,
        backend=backend,
        device=concrete_device,
    )
    _assert_same_native_device(state.X_design, backend, concrete_device)
    multi = compute_gaussian_inference(
        state.X_design,
        state.params,
        state.resid,
        state.scale,
        state.df_resid,
        "hc3",
        backend=backend,
        device=concrete_device,
    )
    state_ref = build_gaussian_fit_state(
        X, y2, coef, intercept, True, backend="numpy"
    )
    multi_ref = compute_gaussian_inference(
        state_ref.X_design,
        state_ref.params,
        state_ref.resid,
        state_ref.scale,
        state_ref.df_resid,
        "hc3",
        backend="numpy",
    )
    if multi.metadata.get("numerical_device") != concrete_device:
        raise AssertionError("multi-target case lost concrete inference device")
    if tuple(multi.bse.shape) != tuple(state.params.shape):
        raise AssertionError("multi-target inference shape mismatch")
    multi_errors = {
        "bse": _max_error(multi.bse, multi_ref.bse),
        "statistic": _max_error(multi.tvalues, multi_ref.tvalues),
        "pvalue": _max_error(multi.pvalues, multi_ref.pvalues),
        "ci": _max_error(multi.conf_int, multi_ref.conf_int),
    }
    for family, values in (("rank", rank_errors), ("multi_target", multi_errors)):
        for key, value in values.items():
            if not np.isfinite(value) or value > 5e-6:
                raise AssertionError(
                    f"{backend} {family} {key} error {value:.3e} exceeds 5e-6"
                )

    return {
        "case": "functional_rank_and_multitarget",
        "requested_backend": backend,
        "executed_inference_backend": multi.metadata.get("numerical_backend"),
        "executed_inference_device": multi.metadata.get("numerical_device"),
        "reporting_boundary": multi.metadata.get("reporting_boundary"),
        "rank_errors": rank_errors,
        "multi_target_errors": multi_errors,
        "limit": 5e-6,
        "no_silent_fallback": True,
        "status": "success",
    }


def _expected_t2_two_sided(statistic: float) -> float:
    value = abs(float(statistic))
    root = np.hypot(value, np.sqrt(2.0))
    return (2.0 / root) / (root + value)


def _small_df_tail_case(backend: str, concrete_device: str):
    design = _as_backend(np.ones((3, 1), dtype=np.float64), backend)
    params = _as_backend(np.asarray([1.0e154]), backend)
    resid = _as_backend(np.asarray([1.0, -1.0, 0.0]), backend)
    _assert_same_native_device(design, backend, concrete_device)
    result = compute_gaussian_inference(
        design,
        params,
        resid,
        1.0,
        2,
        "nonrobust",
        backend=backend,
        device=concrete_device,
    )
    statistic = float(result.tvalues[0])
    pvalue = float(result.pvalues[0])
    expected = _expected_t2_two_sided(statistic)
    if not np.isfinite(pvalue) or pvalue <= 0.0:
        raise AssertionError(
            f"{backend}: representable Student-t(2) tail collapsed to {pvalue!r}"
        )
    rel_error = abs(pvalue - expected) / expected
    if not np.isfinite(rel_error) or rel_error > 5e-12:
        raise AssertionError(
            f"{backend}: Student-t(2) tail relative error {rel_error:.3e} exceeds 5e-12"
        )
    return {
        "case": "student_t_df2_extreme_tail",
        "requested_backend": backend,
        "executed_inference_backend": result.metadata.get("numerical_backend"),
        "executed_inference_device": result.metadata.get("numerical_device"),
        "df": 2,
        "statistic": statistic,
        "pvalue": pvalue,
        "expected_pvalue": expected,
        "relative_error": rel_error,
        "limit": 5e-12,
        "pvalue_nonzero": True,
        "status": "success",
    }


def _ridgecv_case(backend: str):
    X, y, _ = _problem(seed=129)
    alphas = np.asarray([0.01, 0.1, 1.0])
    cpu = RidgeCV(
        alphas=alphas,
        cv=3,
        device="cpu",
        compute_inference=True,
        random_state=129,
    ).fit(X, y)
    device_arg = "cuda" if backend == "cupy" else "torch"
    gpu = RidgeCV(
        alphas=alphas,
        cv=3,
        device=device_arg,
        compute_inference=True,
        random_state=129,
    ).fit(_as_backend(X, backend), _as_backend(y, backend))

    if float(gpu.alpha_) != float(cpu.alpha_):
        raise AssertionError(
            f"RidgeCV selected alpha differs: gpu={gpu.alpha_}, cpu={cpu.alpha_}"
        )
    estimator_backend = str(
        getattr(gpu.estimator_, "_selected_backend_name", "")
    ).lower()
    if estimator_backend != backend:
        raise AssertionError(
            f"RidgeCV final-refit backend mismatch: {estimator_backend}"
        )
    result = gpu.estimator_._inference_result
    if result is None:
        raise AssertionError("RidgeCV final-refit inference is missing")
    if result.metadata.get("numerical_backend") != backend:
        raise AssertionError(
            f"RidgeCV inference backend mismatch: {result.metadata}"
        )
    inference_device = str(result.metadata.get("numerical_device", ""))
    if not inference_device.startswith("cuda:"):
        raise AssertionError(
            f"RidgeCV inference device is not concrete CUDA: {result.metadata}"
        )
    if result.metadata.get("reporting_boundary") != "post_numerical_inference":
        raise AssertionError("RidgeCV inference reporting boundary is missing")
    errors = {
        "coef": _max_error(gpu.coef_, cpu.coef_),
        "bse": _max_error(gpu.estimator_._bse, cpu.estimator_._bse),
        "statistic": _max_error(gpu.estimator_._tvalues, cpu.estimator_._tvalues),
        "pvalue": _max_error(gpu.estimator_._pvalues, cpu.estimator_._pvalues),
        "ci": _max_error(gpu.estimator_._conf_int, cpu.estimator_._conf_int),
    }
    limits = {"coef": 2e-6, "bse": 2e-5, "statistic": 2e-4, "pvalue": 2e-4, "ci": 5e-5}
    for key, limit in limits.items():
        if not np.isfinite(errors[key]) or errors[key] > limit:
            raise AssertionError(
                f"RidgeCV {backend} {key} error {errors[key]:.3e} exceeds {limit:.3e}"
            )
    return {
        "case": "ridgecv_final_refit_inference",
        "requested_backend": backend,
        "executed_backend": estimator_backend,
        "executed_inference_backend": result.metadata.get("numerical_backend"),
        "executed_inference_device": inference_device,
        "reporting_boundary": result.metadata.get("reporting_boundary"),
        "selected_alpha": float(gpu.alpha_),
        "errors": errors,
        "limits": limits,
        "no_silent_fallback": True,
        "status": "success",
    }


def run_backend(backend: str):
    _, concrete_device, runtime_version = _backend_runtime(backend)
    cases = []
    for cov_type in ("nonrobust", "hc3", "hac"):
        cases.append(_linear_regression_case(backend, concrete_device, cov_type))
    if backend == "cupy":
        cases.append(_cupy_nonrank_failure_case(concrete_device))
    for cov_type in ("nonrobust", "hc3", "hac"):
        cases.append(_consumer_case(backend, cov_type, weighted=False))
    cases.append(_consumer_case(backend, "nonrobust", weighted=True))
    cases.append(_host_transfer_case(backend, concrete_device))
    cases.append(_functional_edge_cases(backend, concrete_device))
    cases.append(_small_df_tail_case(backend, concrete_device))
    cases.append(_ridgecv_case(backend))
    return {
        "backend": backend,
        "concrete_device": concrete_device,
        "runtime_version": runtime_version,
        "cases": cases,
        "status": "success",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--validation-tier", required=True, choices=_VALIDATION_TIERS
    )
    parser.add_argument("--backends", required=True)
    args = parser.parse_args(argv)

    backends = _validate_backend_list(args.backends)
    git_sha = _git("rev-parse", "HEAD")
    if git_sha != args.expected_sha:
        raise RuntimeError(
            f"exact-source gate failed: HEAD={git_sha}, expected={args.expected_sha}"
        )
    clean_before = _clean_worktree()
    if not clean_before:
        raise RuntimeError("physical acceptance requires a clean working tree")

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha,
        "validation_tier": args.validation_tier,
        "working_tree_clean_before": clean_before,
        "working_tree_clean_after_checks": False,
        "python_version": platform.python_version(),
        "gpu_model": _gpu_model(),
        "required_backends": list(_REQUIRED_BACKENDS),
        "results": [],
        "status": "running",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for backend in backends:
            artifact["results"].append(run_backend(backend))
        clean_after = _clean_worktree()
        if not clean_after:
            raise RuntimeError(
                "working tree changed during physical validation before artifact write"
            )
        artifact["working_tree_clean_after_checks"] = True
        artifact["status"] = "success"
    except Exception as exc:
        artifact["status"] = "failure"
        artifact["error_type"] = type(exc).__name__
        artifact["error"] = str(exc)
        out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        raise

    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
