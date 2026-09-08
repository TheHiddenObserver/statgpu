#!/usr/bin/env python3
"""Physical CuPy/Torch validation for post-selection OLS inference (#137).

Schema v7 keeps the original four direct-Lasso post-selection cases and closes
all production numerical branches added by the repeated canonical review/fix
loops: ElasticNet/generic sparse Gaussian routing, centered weighted/unweighted
debiased inference, real weighted multi-alpha LassoCV selection plus final
refit, rank-deficient design-level SVD refits, Penalty-object AUTO-native
routing, empty-active robust reporting, and intercept-inclusive simultaneous
max-|Z| inference.

Explicit CUDA/Torch requests must fail rather than silently fall back. Hosted
logic tests cover alias/device orthogonality and explicit-CPU heterogeneous input
conversion; this runner is reserved for branches that require physical CUDA.
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

from statgpu._config import Device, set_device
import statgpu.inference._distributions_backend as _distribution_module
from statgpu.linear_model import (
    ElasticNet,
    Lasso,
    LassoCV,
    PenalizedGeneralizedLinearModel,
)
import statgpu.linear_model._debiased_intercept_parameterization_contract as _intercept_contract
from statgpu.penalties import get_penalty

_REQUIRED_BACKENDS = ("cupy", "torch")
_POST_LIMITS = {
    "penalized_coef": 2e-6,
    "post_selection_params": 2e-7,
    "bse": 2e-7,
    "statistic": 2e-5,
    "pvalue": 2e-6,
    "ci": 5e-7,
}
_DEBIASED_LIMITS = {
    "penalized_coef": 2e-6,
    "params": 5e-5,
    "bse": 2e-5,
    "statistic": 5e-4,
    "pvalue": 5e-4,
    "ci": 1e-4,
}
_CV_LIMITS = {
    "mse_path": 5e-6,
    "mean_mse": 2e-6,
    "penalized_coef": 2e-6,
    "post_selection_params": 2e-7,
    "bse": 2e-7,
    "statistic": 2e-5,
    "pvalue": 2e-6,
    "ci": 5e-7,
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _problem(seed: int = 137):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(360, 8))
    beta = np.zeros(8)
    beta[:4] = [1.4, -0.95, 0.7, 0.35]
    y = 0.3 + X @ beta + rng.normal(scale=0.55, size=X.shape[0])
    weights = rng.uniform(0.35, 1.8, size=X.shape[0])
    return X, y, weights


def _native(value, backend: str):
    if backend == "cupy":
        import cupy as cp

        return cp.asarray(value, dtype=cp.float64)

    import torch

    return torch.as_tensor(
        np.asarray(value),
        dtype=torch.float64,
        device=f"cuda:{torch.cuda.current_device()}",
    )


def _runtime(backend: str):
    if backend == "cupy":
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError("CuPy imported but no CUDA device is available")
        device_id = int(cp.cuda.runtime.getDevice())
        return f"cuda:{device_id}", cp.__version__

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch imported but CUDA is unavailable")
    device_id = int(torch.cuda.current_device())
    return f"cuda:{device_id}", torch.__version__


def _requested_device(backend: str) -> str:
    return "cuda" if backend == "cupy" else "torch"


def _max_error(a, b) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise AssertionError(
            f"shape mismatch: left={left.shape}, right={right.shape}"
        )
    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    if not np.array_equal(left_nan, right_nan):
        raise AssertionError("CPU/GPU inference NaN masks differ")
    finite = ~left_nan
    if not np.any(finite):
        return 0.0
    if not np.all(np.isfinite(left[finite])) or not np.all(np.isfinite(right[finite])):
        raise AssertionError("non-finite non-NaN values encountered in parity check")
    return float(np.max(np.abs(left[finite] - right[finite])))


def _assert_limits(label: str, errors: dict[str, float], limits: dict[str, float]):
    for key, limit in limits.items():
        value = float(errors[key])
        if not np.isfinite(value) or value > float(limit):
            raise AssertionError(
                f"{label} {key} error {value:.3e} exceeds {float(limit):.3e}"
            )


def _distribution_backend_name(distribution) -> str:
    sf_name = type(getattr(distribution, "_sf", None)).__name__
    return {
        "CuPySpecialFunctions": "cupy",
        "TorchSpecialFunctions": "torch",
        "ScipySpecialFunctions": "numpy",
    }.get(sf_name, sf_name or "unknown")


def _make_post_selection_model(kind: str, *, device: str):
    common = dict(
        alpha=0.05,
        solver="fista",
        inference_method="post_selection_ols",
        compute_inference=True,
        device=device,
        max_iter=6000,
        tol=1e-9,
    )
    if kind == "lasso":
        return Lasso(**common)
    if kind == "elasticnet":
        return ElasticNet(l1_ratio=0.7, **common)
    if kind == "generic_l1":
        return PenalizedGeneralizedLinearModel(
            loss="squared_error",
            penalty="l1",
            **common,
        )
    raise ValueError(f"unknown post-selection model kind: {kind}")


def _fit_post_selection(kind: str, X, y, *, device: str, sample_weight=None):
    model = _make_post_selection_model(kind, device=device)
    model.fit(X, y, sample_weight=sample_weight)
    if model._inference_result is None:
        raise AssertionError(f"{kind} post-selection OLS inference result is missing")
    return model


def _assert_post_selection_provenance(gpu, cpu, backend: str, expected_device: str):
    meta = dict(gpu._inference_result.metadata)
    if gpu._selected_backend_name != backend:
        raise AssertionError(
            f"fit backend mismatch: expected {backend}, got {gpu._selected_backend_name!r}"
        )
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"inference backend provenance mismatch: {meta}")
    if str(meta.get("numerical_device")) != expected_device:
        raise AssertionError(f"inference device provenance mismatch: {meta}")
    if meta.get("reporting_boundary") != "post_numerical_inference":
        raise AssertionError(f"reporting boundary mismatch: {meta}")
    if meta.get("resolved_method") != "post_selection_ols":
        raise AssertionError(f"method provenance mismatch: {meta}")
    if gpu._inference_result.statistic_name != "t":
        raise AssertionError("nonrobust post-selection OLS must preserve t-statistic semantics")
    if gpu._inference_result.distribution != "t":
        raise AssertionError("nonrobust post-selection OLS must preserve Student-t inference")
    if not meta.get("inactive_inference_placeholders"):
        raise AssertionError("inactive-coordinate compatibility metadata is missing")

    cpu_selected = cpu._inference_result.metadata["selected_feature_indices"]
    gpu_selected = gpu._inference_result.metadata["selected_feature_indices"]
    if cpu_selected != gpu_selected:
        raise AssertionError(
            f"active-set mismatch: cpu={cpu_selected}, {backend}={gpu_selected}"
        )
    if meta.get("refit_df_resid") != cpu._inference_result.metadata.get("refit_df_resid"):
        raise AssertionError("CPU/GPU post-selection residual degrees of freedom differ")
    return meta, gpu_selected


def _post_selection_errors(gpu, cpu):
    return {
        "penalized_coef": _max_error(gpu.coef_, cpu.coef_),
        "post_selection_params": _max_error(gpu._params, cpu._params),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }


def _post_selection_case(backend: str, *, weighted: bool, kind: str = "lasso"):
    seed_offset = {"lasso": 0, "elasticnet": 20, "generic_l1": 40}[kind]
    X, y, weights = _problem(seed=137 + seed_offset + int(weighted))
    sw = weights if weighted else None
    cpu = _fit_post_selection(kind, X, y, device="cpu", sample_weight=sw)

    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)
    gpu = _fit_post_selection(
        kind,
        _native(X, backend),
        _native(y, backend),
        device=requested_device,
        sample_weight=None if sw is None else _native(sw, backend),
    )

    meta, gpu_selected = _assert_post_selection_provenance(
        gpu,
        cpu,
        backend,
        expected_device,
    )
    errors = _post_selection_errors(gpu, cpu)
    _assert_limits(f"{kind} {backend} weighted={weighted}", errors, _POST_LIMITS)

    return {
        "model": kind,
        "case": "weighted" if weighted else "unweighted",
        "backend": backend,
        "backend_version": version,
        "requested_device": requested_device,
        "executed_backend": gpu._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "reporting_boundary": meta.get("reporting_boundary"),
        "selected_feature_indices": gpu_selected,
        "statistic_name": gpu._inference_result.statistic_name,
        "distribution": gpu._inference_result.distribution,
        "refit_df_resid": meta.get("refit_df_resid"),
        "errors": errors,
        "limits": dict(_POST_LIMITS),
        "status": "success",
    }


def _rank_deficient_post_selection_case(backend: str):
    rng = np.random.default_rng(251)
    n = 220
    x = rng.normal(size=n)
    X = np.column_stack([x, x, rng.normal(size=n), rng.normal(size=n)])
    y = 0.25 + 2.0 * x + 0.6 * X[:, 2] + rng.normal(scale=0.35, size=n)
    weights = rng.uniform(0.4, 1.7, size=n)
    common = dict(
        alpha=0.02,
        l1_ratio=0.5,
        solver="fista",
        inference_method="post_selection_ols",
        compute_inference=True,
        max_iter=6000,
        tol=1e-9,
    )
    cpu = ElasticNet(device="cpu", **common).fit(X, y, sample_weight=weights)
    cpu_meta = dict(cpu._inference_result.metadata)
    cpu_selected = list(cpu_meta["selected_feature_indices"])
    if 0 not in cpu_selected or 1 not in cpu_selected:
        raise AssertionError(
            f"rank-deficient fixture did not retain both duplicate columns: {cpu_selected}"
        )
    if not cpu_meta.get("refit_rank_deficient"):
        raise AssertionError(f"CPU rank-deficient fixture was not detected: {cpu_meta}")
    if int(cpu_meta.get("refit_rank")) >= int(cpu_meta.get("refit_parameter_count")):
        raise AssertionError(f"CPU rank metadata is inconsistent: {cpu_meta}")

    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)
    gpu = ElasticNet(device=requested_device, **common).fit(
        _native(X, backend),
        _native(y, backend),
        sample_weight=_native(weights, backend),
    )
    meta, gpu_selected = _assert_post_selection_provenance(
        gpu,
        cpu,
        backend,
        expected_device,
    )
    for key in ("refit_rank", "refit_parameter_count", "refit_rank_deficient"):
        if meta.get(key) != cpu_meta.get(key):
            raise AssertionError(
                f"rank-deficient {backend} metadata mismatch for {key}: "
                f"cpu={cpu_meta.get(key)!r}, gpu={meta.get(key)!r}"
            )
    if gpu_selected != cpu_selected:
        raise AssertionError("rank-deficient active-set identity changed across backends")

    errors = _post_selection_errors(gpu, cpu)
    _assert_limits(f"rank-deficient {backend}", errors, _POST_LIMITS)
    return {
        "model": "elasticnet",
        "case": "weighted_rank_deficient_active_refit",
        "backend": backend,
        "backend_version": version,
        "requested_device": requested_device,
        "executed_backend": gpu._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "selected_feature_indices": gpu_selected,
        "refit_rank": meta.get("refit_rank"),
        "refit_parameter_count": meta.get("refit_parameter_count"),
        "refit_rank_deficient": meta.get("refit_rank_deficient"),
        "refit_df_resid": meta.get("refit_df_resid"),
        "errors": errors,
        "limits": dict(_POST_LIMITS),
        "status": "success",
    }


def _penalty_object_auto_native_case(backend: str):
    X, y, weights = _problem(seed=263)
    common = dict(
        loss="squared_error",
        alpha=0.05,
        fit_intercept=True,
        solver="fista",
        inference_method="post_selection_ols",
        compute_inference=True,
        max_iter=6000,
        tol=1e-9,
    )
    cpu = PenalizedGeneralizedLinearModel(
        penalty=get_penalty("l1", alpha=0.05),
        device="cpu",
        **common,
    ).fit(X, y, sample_weight=weights)

    set_device("auto")
    expected_device, version = _runtime(backend)
    gpu = PenalizedGeneralizedLinearModel(
        penalty=get_penalty("l1", alpha=0.05),
        device="auto",
        **common,
    ).fit(
        _native(X, backend),
        _native(y, backend),
        sample_weight=_native(weights, backend),
    )
    meta, gpu_selected = _assert_post_selection_provenance(
        gpu,
        cpu,
        backend,
        expected_device,
    )
    if gpu._device != Device.AUTO or str(gpu.device).lower() not in {"auto", "device.auto"}:
        raise AssertionError("AUTO native-input routing mutated the estimator's public device")
    if getattr(gpu.penalty, "name", None) != "l1":
        raise AssertionError("Penalty-object identity/name was lost during AUTO routing")

    errors = _post_selection_errors(gpu, cpu)
    _assert_limits(f"Penalty-object AUTO {backend}", errors, _POST_LIMITS)
    return {
        "model": "generic_l1_penalty_object",
        "case": "weighted_auto_native_penalty_object",
        "backend": backend,
        "backend_version": version,
        "requested_device": "auto",
        "executed_backend": gpu._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "selected_feature_indices": gpu_selected,
        "public_device_restored": True,
        "errors": errors,
        "limits": dict(_POST_LIMITS),
        "status": "success",
    }


def _empty_active_robust_case(backend: str):
    rng = np.random.default_rng(271)
    X = rng.normal(size=(180, 5))
    y = rng.normal(scale=0.2, size=X.shape[0])
    common = dict(
        loss="squared_error",
        penalty="l1",
        alpha=100.0,
        fit_intercept=False,
        solver="fista",
        inference_method="post_selection_ols",
        compute_inference=True,
        cov_type="hc3",
        max_iter=3000,
        tol=1e-9,
    )
    cpu = PenalizedGeneralizedLinearModel(device="cpu", **common).fit(X, y)
    cpu_result = cpu._inference_result
    if cpu_result is None or cpu_result.metadata.get("n_selected") != 0:
        raise AssertionError("empty-active robust CPU fixture did not select zero features")
    if cpu_result.cov_type != "hc3" or cpu_result.distribution != "normal":
        raise AssertionError("empty-active robust CPU result lost requested covariance semantics")
    if cpu_result.statistic_name != "z" or cpu_result.df is not None:
        raise AssertionError("empty-active robust CPU reference family is inconsistent")

    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)
    gpu = PenalizedGeneralizedLinearModel(device=requested_device, **common).fit(
        _native(X, backend),
        _native(y, backend),
    )
    result = gpu._inference_result
    if result is None:
        raise AssertionError("empty-active robust GPU result is missing")
    meta = dict(result.metadata)
    if gpu._selected_backend_name != backend:
        raise AssertionError("empty-active robust fit executed on the wrong backend")
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"empty-active robust inference backend mismatch: {meta}")
    if str(meta.get("numerical_device")) != expected_device:
        raise AssertionError(f"empty-active robust device provenance mismatch: {meta}")
    if result.cov_type != "hc3" or result.distribution != "normal":
        raise AssertionError("empty-active robust GPU result lost HC3/normal semantics")
    if result.statistic_name != "z" or result.df is not None:
        raise AssertionError("empty-active robust GPU reference family is inconsistent")
    if int(meta.get("n_selected", -1)) != 0 or int(meta.get("refit_parameter_count", -1)) != 0:
        raise AssertionError(f"empty-active robust metadata is inconsistent: {meta}")

    errors = _post_selection_errors(gpu, cpu)
    _assert_limits(f"empty-active robust {backend}", errors, _POST_LIMITS)
    return {
        "model": "generic_l1",
        "case": "empty_active_hc3_no_intercept",
        "backend": backend,
        "backend_version": version,
        "requested_device": requested_device,
        "executed_backend": gpu._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "cov_type": result.cov_type,
        "distribution": result.distribution,
        "statistic_name": result.statistic_name,
        "refit_df_resid": meta.get("refit_df_resid"),
        "errors": errors,
        "limits": dict(_POST_LIMITS),
        "status": "success",
    }


def _debiased_case(backend: str, *, weighted: bool):
    X, y, weights = _problem(seed=211 + int(weighted))
    sw = weights if weighted else None
    common = dict(
        alpha=0.05,
        solver="fista",
        inference_method="debiased",
        compute_inference=True,
        max_iter=6000,
        tol=1e-9,
    )
    cpu = Lasso(device="cpu", **common).fit(X, y, sample_weight=sw)

    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)
    distribution_backends = []
    real_resolve = _distribution_module.DistributionProxy._resolve

    def guarded_resolve(proxy, kwargs, *arrays):
        distribution = real_resolve(proxy, kwargs, *arrays)
        if _intercept_contract._DISTRIBUTION_ROUTE.get() is not None:
            resolved = _distribution_backend_name(distribution)
            distribution_backends.append(resolved)
            if resolved != backend:
                raise AssertionError(
                    f"debiased distribution fallback: expected={backend}, resolved={resolved}"
                )
        return distribution

    _distribution_module.DistributionProxy._resolve = guarded_resolve
    try:
        gpu = Lasso(device=requested_device, **common).fit(
            _native(X, backend),
            _native(y, backend),
            sample_weight=None if sw is None else _native(sw, backend),
        )
    finally:
        _distribution_module.DistributionProxy._resolve = real_resolve
    if not distribution_backends:
        raise AssertionError("debiased GPU inference did not execute distribution helpers")

    result = gpu._inference_result
    if result is None or result.method != "debiased":
        raise AssertionError("debiased GPU inference result is missing")
    meta = dict(result.metadata)
    if gpu._selected_backend_name != backend:
        raise AssertionError("debiased fit backend provenance mismatch")
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"debiased numerical backend mismatch: {meta}")
    if str(meta.get("numerical_device")) != expected_device:
        raise AssertionError(f"debiased numerical device mismatch: {meta}")
    if meta.get("reporting_boundary") != "post_numerical_inference":
        raise AssertionError(f"debiased reporting boundary mismatch: {meta}")
    if bool(meta.get("sample_weighted")) != bool(weighted):
        raise AssertionError(f"debiased sample-weight provenance mismatch: {meta}")
    if weighted and meta.get("backend_path") != f"{backend}_debiased_weighted":
        raise AssertionError(f"weighted debiased backend path mismatch: {meta}")
    if meta.get("intercept_estimator") != "centered_debiased":
        raise AssertionError(f"debiased intercept estimator metadata mismatch: {meta}")
    if meta.get("intercept_influence") != "centered_nodewise":
        raise AssertionError(f"debiased intercept influence metadata mismatch: {meta}")

    errors = {
        "penalized_coef": _max_error(gpu.coef_, cpu.coef_),
        "params": _max_error(gpu._params, cpu._params),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    _assert_limits(f"debiased {backend} weighted={weighted}", errors, _DEBIASED_LIMITS)
    return {
        "model": "lasso",
        "case": "weighted_debiased" if weighted else "unweighted_debiased",
        "backend": backend,
        "backend_version": version,
        "requested_device": requested_device,
        "executed_backend": gpu._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "backend_path": meta.get("backend_path"),
        "reporting_boundary": meta.get("reporting_boundary"),
        "distribution_backends": distribution_backends,
        "errors": errors,
        "limits": dict(_DEBIASED_LIMITS),
        "status": "success",
    }


def _lassocv_weighted_selection_case(backend: str):
    X, y, weights = _problem(seed=229)
    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)
    alphas = np.asarray([0.12, 0.06, 0.03, 0.015], dtype=np.float64)
    common = dict(
        alphas=alphas,
        cv=3,
        fit_intercept=True,
        compute_inference=True,
        inference_method="post_selection_ols",
        solver="fista",
        cv_solver="fista",
        gpu_cv_mixed_precision=False,
        max_iter=5000,
        tol=1e-9,
        random_state=7,
    )
    cpu = LassoCV(device="cpu", **common).fit(X, y, sample_weight=weights)
    gpu = LassoCV(device=requested_device, **common).fit(
        _native(X, backend),
        _native(y, backend),
        sample_weight=_native(weights, backend),
    )
    if gpu.estimator_ is None or gpu.estimator_._inference_result is None:
        raise AssertionError("LassoCV final refit inference result is missing")
    if getattr(gpu, "_inference_result", None) is None:
        raise AssertionError("LassoCV outer inference result is missing")
    meta = dict(gpu.estimator_._inference_result.metadata)
    if gpu.estimator_._selected_backend_name != backend:
        raise AssertionError("LassoCV final refit executed on the wrong backend")
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"LassoCV inference backend mismatch: {meta}")
    if str(meta.get("numerical_device")) != expected_device:
        raise AssertionError(f"LassoCV inference device mismatch: {meta}")
    if meta.get("resolved_method") != "post_selection_ols":
        raise AssertionError(f"LassoCV inference method mismatch: {meta}")
    if dict(gpu._inference_result.metadata) != meta:
        raise AssertionError("LassoCV outer inference metadata differs from final estimator")
    for attr in ("_params", "_bse", "_tvalues", "_pvalues", "_conf_int"):
        if _max_error(getattr(gpu, attr), getattr(gpu.estimator_, attr)) != 0.0:
            raise AssertionError(f"LassoCV outer {attr} differs from final estimator")
    if float(gpu.alpha_) != float(cpu.alpha_):
        raise AssertionError(
            f"LassoCV selected-alpha mismatch: cpu={cpu.alpha_}, {backend}={gpu.alpha_}"
        )
    if np.asarray(gpu.alphas_).size <= 1:
        raise AssertionError("LassoCV physical closure did not execute a multi-alpha path")

    errors = {
        "mse_path": _max_error(gpu.mse_path_, cpu.mse_path_),
        "mean_mse": _max_error(gpu.mean_mse_, cpu.mean_mse_),
        "penalized_coef": _max_error(gpu.coef_, cpu.coef_),
        "post_selection_params": _max_error(gpu._params, cpu._params),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    _assert_limits(f"weighted multi-alpha LassoCV {backend}", errors, _CV_LIMITS)
    return {
        "model": "lassocv",
        "case": "weighted_multi_alpha_selection_final_refit",
        "backend": backend,
        "backend_version": version,
        "requested_device": requested_device,
        "executed_backend": gpu.estimator_._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "selected_alpha": float(gpu.alpha_),
        "n_alphas": int(np.asarray(gpu.alphas_).size),
        "outer_inference_surface": True,
        "errors": errors,
        "limits": dict(_CV_LIMITS),
        "status": "success",
    }


def _simultaneous_intercept_case(backend: str):
    X, y, weights = _problem(seed=293)
    B = 96
    seed = 20260908
    common = dict(
        alpha=0.045,
        solver="fista",
        inference_method="debiased",
        compute_inference=True,
        fit_intercept=True,
        max_iter=6000,
        tol=1e-9,
        enable_simultaneous_inference=True,
        simultaneous_include_intercept=True,
        simultaneous_n_bootstrap=B,
        simultaneous_random_state=seed,
    )
    cpu = Lasso(device="cpu", **common).fit(X, y, sample_weight=weights)
    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)

    # A centered/intercept-capable explicit GPU path must never re-enter the
    # historical NumPy simultaneous helper. Make any such fallback fatal while
    # the physical GPU fit runs; the backend-native finalizer bypasses it.
    original_cpu_simultaneous = (
        PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap
    )

    def forbid_cpu_simultaneous(*args, **kwargs):
        raise AssertionError(
            "explicit GPU simultaneous debiased inference fell back to the NumPy helper"
        )

    PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap = (
        forbid_cpu_simultaneous
    )
    try:
        gpu = Lasso(device=requested_device, **common).fit(
            _native(X, backend),
            _native(y, backend),
            sample_weight=_native(weights, backend),
        )
    finally:
        PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap = (
            original_cpu_simultaneous
        )

    result = gpu._inference_result
    if result is None or result.simultaneous_conf_int is None:
        raise AssertionError("intercept-inclusive simultaneous GPU result is missing")
    meta = dict(result.metadata)
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"simultaneous debiased backend mismatch: {meta}")
    if str(meta.get("numerical_device")) != expected_device:
        raise AssertionError(f"simultaneous debiased device mismatch: {meta}")
    if meta.get("simultaneous_numerical_backend") != backend:
        raise AssertionError(f"simultaneous numerical backend mismatch: {meta}")
    if str(meta.get("simultaneous_numerical_device")) != expected_device:
        raise AssertionError(f"simultaneous numerical device mismatch: {meta}")
    if meta.get("simultaneous_reporting_backend") != "numpy":
        raise AssertionError(f"simultaneous reporting backend mismatch: {meta}")
    if meta.get("simultaneous_reporting_boundary") != "post_numerical_inference":
        raise AssertionError(f"simultaneous reporting boundary mismatch: {meta}")

    target_mask = np.asarray(result.simultaneous_target_mask, dtype=bool)
    if target_mask.shape != np.asarray(gpu._params).shape or not np.all(target_mask):
        raise AssertionError("simultaneous intercept target mask does not include all parameters")

    influence_error = _max_error(
        gpu._debiased_intercept_influence_cpu,
        cpu._debiased_intercept_influence_cpu,
    )
    if influence_error > 2e-6:
        raise AssertionError(
            f"simultaneous intercept influence {backend} error {influence_error:.3e} "
            "exceeds 2.000e-06"
        )

    critical = float(result.simultaneous_critical_value)
    if not np.isfinite(critical) or critical < 0.0:
        raise AssertionError(
            f"simultaneous {backend} critical value is invalid: {critical!r}"
        )
    expected_ci = np.column_stack(
        [
            gpu._params - critical * gpu._bse,
            gpu._params + critical * gpu._bse,
        ]
    )
    ci_reconstruction_error = _max_error(gpu._conf_int_simultaneous, expected_ci)
    if ci_reconstruction_error > 1e-10:
        raise AssertionError(
            "simultaneous intercept CI does not match the backend-native critical value"
        )

    return {
        "model": "lasso",
        "case": "weighted_debiased_simultaneous_intercept",
        "backend": backend,
        "backend_version": version,
        "requested_device": requested_device,
        "executed_backend": gpu._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "simultaneous_numerical_backend": meta.get("simultaneous_numerical_backend"),
        "simultaneous_numerical_device": meta.get("simultaneous_numerical_device"),
        "cpu_simultaneous_fallback_blocked": True,
        "intercept_influence_error": influence_error,
        "intercept_influence_limit": 2e-6,
        "critical_value": critical,
        "ci_reconstruction_error": ci_reconstruction_error,
        "status": "success",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        default="cupy,torch",
        help="Canonical acceptance requires exactly cupy,torch.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    backends = tuple(
        part.strip().lower() for part in args.backends.split(",") if part.strip()
    )
    if backends != _REQUIRED_BACKENDS:
        raise ValueError("--backends must be exactly 'cupy,torch' in that order")

    set_device("auto")
    payload = {
        "schema_version": 7,
        "issue": 137,
        "head_sha": _git("rev-parse", "HEAD"),
        "worktree_clean": _git("status", "--porcelain") == "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cases": [],
        "closure_cases": [],
    }
    if not payload["worktree_clean"]:
        raise RuntimeError("physical acceptance requires a clean worktree")

    # Preserve the original four direct-Lasso acceptance cases in meaning so
    # historical evidence remains directly comparable.
    for backend in backends:
        for weighted in (False, True):
            payload["cases"].append(
                _post_selection_case(backend, weighted=weighted, kind="lasso")
            )

    # Nine closure cases per physical backend. Together with the original four
    # this is the canonical schema-v7 22-case matrix.
    for backend in backends:
        payload["closure_cases"].append(
            _post_selection_case(backend, weighted=True, kind="elasticnet")
        )
        payload["closure_cases"].append(
            _post_selection_case(backend, weighted=True, kind="generic_l1")
        )
        payload["closure_cases"].append(_debiased_case(backend, weighted=False))
        payload["closure_cases"].append(_debiased_case(backend, weighted=True))
        payload["closure_cases"].append(_lassocv_weighted_selection_case(backend))
        payload["closure_cases"].append(_rank_deficient_post_selection_case(backend))
        payload["closure_cases"].append(_penalty_object_auto_native_case(backend))
        payload["closure_cases"].append(_empty_active_robust_case(backend))
        payload["closure_cases"].append(_simultaneous_intercept_case(backend))

    if len(payload["cases"]) != 4 or len(payload["closure_cases"]) != 18:
        raise AssertionError("schema-v7 acceptance matrix must contain exactly 22 cases")

    payload["status"] = "success"
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
