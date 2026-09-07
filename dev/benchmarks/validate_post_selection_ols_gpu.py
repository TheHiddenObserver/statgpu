#!/usr/bin/env python3
"""Physical CuPy/Torch validation for post-selection OLS inference (#137).

Canonical acceptance covers the original direct-Lasso parity matrix plus the
public closure surfaces touched by #138: ElasticNet, the generic squared-error
penalized estimator, LassoCV final refit, preservation of weighted debiased GPU
inference, and rank-deficient active-set refit semantics. Explicit CUDA/Torch
requests must fail rather than fall back.
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

from statgpu.linear_model import (
    ElasticNet,
    Lasso,
    LassoCV,
    PenalizedGeneralizedLinearModel,
)

_REQUIRED_BACKENDS = ("cupy", "torch")
_POST_LIMITS = {
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


def _post_selection_case(backend: str, *, weighted: bool, kind: str = "lasso"):
    seed_offset = {"lasso": 0, "elasticnet": 20, "generic_l1": 40}[kind]
    X, y, weights = _problem(seed=137 + seed_offset + int(weighted))
    sw = weights if weighted else None
    cpu = _fit_post_selection(kind, X, y, device="cpu", sample_weight=sw)

    X_native = _native(X, backend)
    y_native = _native(y, backend)
    sw_native = None if sw is None else _native(sw, backend)
    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)
    gpu = _fit_post_selection(
        kind,
        X_native,
        y_native,
        device=requested_device,
        sample_weight=sw_native,
    )

    meta, gpu_selected = _assert_post_selection_provenance(
        gpu,
        cpu,
        backend,
        expected_device,
    )
    errors = {
        "penalized_coef": _max_error(gpu.coef_, cpu.coef_),
        "post_selection_params": _max_error(gpu._params, cpu._params),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    for key, limit in _POST_LIMITS.items():
        value = errors[key]
        if not np.isfinite(value) or value > limit:
            raise AssertionError(
                f"{kind} {backend} weighted={weighted} {key} error {value:.3e} "
                f"exceeds {limit:.3e}"
            )

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
    X = np.column_stack(
        [
            x,
            x,
            rng.normal(size=n),
            rng.normal(size=n),
        ]
    )
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

    errors = {
        "penalized_coef": _max_error(gpu.coef_, cpu.coef_),
        "post_selection_params": _max_error(gpu._params, cpu._params),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    for key, limit in _POST_LIMITS.items():
        value = errors[key]
        if not np.isfinite(value) or value > limit:
            raise AssertionError(
                f"rank-deficient {backend} {key} error {value:.3e} exceeds {limit:.3e}"
            )

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


def _weighted_debiased_case(backend: str):
    X, y, weights = _problem(seed=211)
    cpu = Lasso(
        alpha=0.05,
        solver="fista",
        compute_inference=False,
        device="cpu",
        max_iter=6000,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)
    gpu = Lasso(
        alpha=0.05,
        solver="fista",
        inference_method="debiased",
        compute_inference=True,
        device=requested_device,
        max_iter=6000,
        tol=1e-9,
    ).fit(
        _native(X, backend),
        _native(y, backend),
        sample_weight=_native(weights, backend),
    )
    result = gpu._inference_result
    if result is None or result.method != "debiased":
        raise AssertionError("weighted debiased GPU inference result is missing")
    meta = dict(result.metadata)
    if gpu._selected_backend_name != backend:
        raise AssertionError("weighted debiased fit backend provenance mismatch")
    if meta.get("backend_path") != f"{backend}_debiased_weighted":
        raise AssertionError(f"weighted debiased inference silently changed backend: {meta}")
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"weighted debiased numerical backend mismatch: {meta}")
    if str(meta.get("numerical_device")) != expected_device:
        raise AssertionError(f"weighted debiased numerical device mismatch: {meta}")
    if meta.get("reporting_boundary") != "post_numerical_inference":
        raise AssertionError(f"weighted debiased reporting boundary mismatch: {meta}")
    if meta.get("sample_weighted") is not True:
        raise AssertionError("weighted debiased metadata lost sample_weight provenance")

    coef_error = _max_error(gpu.coef_, cpu.coef_)
    if not np.isfinite(coef_error) or coef_error > _POST_LIMITS["penalized_coef"]:
        raise AssertionError(
            f"weighted debiased {backend} penalized_coef error {coef_error:.3e} exceeds "
            f"{_POST_LIMITS['penalized_coef']:.3e}"
        )
    expected_dim = X.shape[1] + 1
    for name, value in (
        ("params", gpu._params),
        ("bse", gpu._bse),
        ("statistic", gpu._tvalues),
        ("pvalue", gpu._pvalues),
    ):
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (expected_dim,) or not np.all(np.isfinite(arr)):
            raise AssertionError(f"weighted debiased {name} layout/finite check failed")
    ci = np.asarray(gpu._conf_int, dtype=np.float64)
    if ci.shape != (expected_dim, 2) or not np.all(np.isfinite(ci)):
        raise AssertionError("weighted debiased confidence-interval layout/finite check failed")

    return {
        "model": "lasso",
        "case": "weighted_debiased_backend_preservation",
        "backend": backend,
        "backend_version": version,
        "requested_device": requested_device,
        "executed_backend": gpu._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "backend_path": meta.get("backend_path"),
        "reporting_boundary": meta.get("reporting_boundary"),
        "penalized_coef_error": coef_error,
        "penalized_coef_limit": _POST_LIMITS["penalized_coef"],
        "status": "success",
    }


def _lassocv_final_refit_case(backend: str):
    X, y, weights = _problem(seed=229)
    requested_device = _requested_device(backend)
    expected_device, version = _runtime(backend)
    common = dict(
        alphas=np.asarray([0.05], dtype=np.float64),
        cv=2,
        fit_intercept=True,
        compute_inference=True,
        inference_method="post_selection_ols",
        solver="fista",
        cv_solver="fista",
        gpu_cv_mixed_precision=False,
        max_iter=3000,
        tol=1e-8,
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
    meta = dict(gpu.estimator_._inference_result.metadata)
    if gpu.estimator_._selected_backend_name != backend:
        raise AssertionError("LassoCV final refit executed on the wrong backend")
    if meta.get("numerical_backend") != backend:
        raise AssertionError(f"LassoCV inference backend mismatch: {meta}")
    if str(meta.get("numerical_device")) != expected_device:
        raise AssertionError(f"LassoCV inference device mismatch: {meta}")
    if meta.get("resolved_method") != "post_selection_ols":
        raise AssertionError(f"LassoCV inference method mismatch: {meta}")
    coef_error = _max_error(gpu.coef_, cpu.coef_)
    if coef_error > _POST_LIMITS["penalized_coef"]:
        raise AssertionError(
            f"LassoCV {backend} final-refit coef error {coef_error:.3e} exceeds "
            f"{_POST_LIMITS['penalized_coef']:.3e}"
        )
    return {
        "model": "lassocv",
        "case": "weighted_final_refit",
        "backend": backend,
        "backend_version": version,
        "requested_device": requested_device,
        "executed_backend": gpu.estimator_._selected_backend_name,
        "executed_device": meta.get("numerical_device"),
        "selected_alpha": float(gpu.alpha_),
        "penalized_coef_error": coef_error,
        "penalized_coef_limit": _POST_LIMITS["penalized_coef"],
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

    backends = tuple(part.strip().lower() for part in args.backends.split(",") if part.strip())
    if backends != _REQUIRED_BACKENDS:
        raise ValueError("--backends must be exactly 'cupy,torch' in that order")

    payload = {
        "schema_version": 5,
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

    # Preserve the original four direct-Lasso acceptance cases verbatim in
    # meaning so historical evidence remains comparable.
    for backend in backends:
        for weighted in (False, True):
            payload["cases"].append(
                _post_selection_case(backend, weighted=weighted, kind="lasso")
            )

    # Review closure for the public surfaces whose routing shares or consumes
    # the repaired path. Schema v5 adds one rank-deficient active-set case per
    # physical backend because effective-rank df is now backend-native work.
    for backend in backends:
        payload["closure_cases"].append(
            _post_selection_case(backend, weighted=True, kind="elasticnet")
        )
        payload["closure_cases"].append(
            _post_selection_case(backend, weighted=True, kind="generic_l1")
        )
        payload["closure_cases"].append(_weighted_debiased_case(backend))
        payload["closure_cases"].append(_lassocv_final_refit_case(backend))
        payload["closure_cases"].append(_rank_deficient_post_selection_case(backend))

    payload["status"] = "success"
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
