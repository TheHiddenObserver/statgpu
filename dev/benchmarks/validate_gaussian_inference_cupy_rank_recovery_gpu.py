#!/usr/bin/env python3
"""Focused physical CuPy rank-recovery acceptance for PR #129.

This companion to the full Gaussian inference matrix exercises the CuPy
cuSOLVER failure paths that can otherwise return silent NaNs for singular
normal equations. Canonical evidence requires exact source identity, a clean
tree, remote-full provenance, and CPU parity for the maintained recovery paths.
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

from statgpu.backends._array_ops import _linalg_exception_is_rank_failure
from statgpu.backends._utils import xp_cholesky_solve
from statgpu.linear_model import LinearRegression, PenalizedGeneralizedLinearModel
from statgpu.linear_model._gaussian_inference import compute_gaussian_inference

SCHEMA_VERSION = 1
_VALIDATION_TIERS = ("local-minimal", "local-full", "remote-full")
_LIMITS = {
    "prediction": 5e-6,
    "coef": 5e-6,
    "bse": 5e-6,
    "statistic": 5e-5,
    "pvalue": 5e-5,
    "ci": 5e-5,
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _clean_worktree() -> bool:
    return _git("status", "--porcelain") == ""


def _gpu_model() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        return output[0].strip() if output else "unknown"
    except Exception:
        return "unknown"


def _to_numpy(value):
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _max_error(a, b) -> float:
    aa = np.asarray(_to_numpy(a), dtype=np.float64)
    bb = np.asarray(_to_numpy(b), dtype=np.float64)
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


def _assert_errors(case: str, errors: dict[str, float]) -> None:
    for key, value in errors.items():
        limit = _LIMITS[key]
        if not np.isfinite(value) or value > limit:
            raise AssertionError(
                f"{case}: {key} error {value:.3e} exceeds {limit:.3e}"
            )


def _problem():
    x = np.arange(1.0, 9.0)
    X = np.column_stack([x, x])
    resid = np.asarray([0.2, -0.1, 0.05, -0.15, 0.1, -0.1, 0.05, -0.05])
    y = 0.5 + 0.5 * x + resid
    return x, X, y, resid


def _direct_inference_case(cp, concrete_device: str):
    x, _, _, resid = _problem()
    design_np = np.column_stack([np.ones_like(x), x, x])
    params_np = np.asarray([0.5, 0.25, 0.25])
    scale = float(np.sum(resid**2) / (len(x) - 3))

    result = compute_gaussian_inference(
        cp.asarray(design_np, dtype=cp.float64),
        cp.asarray(params_np, dtype=cp.float64),
        cp.asarray(resid, dtype=cp.float64),
        scale,
        len(x) - 3,
        "nonrobust",
        backend="cupy",
        device=concrete_device,
    )
    reference = compute_gaussian_inference(
        design_np,
        params_np,
        resid,
        scale,
        len(x) - 3,
        "nonrobust",
        backend="numpy",
    )
    errors = {
        "bse": _max_error(result.bse, reference.bse),
        "statistic": _max_error(result.tvalues, reference.tvalues),
        "pvalue": _max_error(result.pvalues, reference.pvalues),
        "ci": _max_error(result.conf_int, reference.conf_int),
    }
    _assert_errors("direct_inference_rank_recovery", errors)
    if result.metadata.get("numerical_backend") != "cupy":
        raise AssertionError(f"direct inference backend mismatch: {result.metadata}")
    if str(result.metadata.get("numerical_device", "")) != concrete_device:
        raise AssertionError(f"direct inference device mismatch: {result.metadata}")
    return {
        "case": "direct_inference_rank_recovery",
        "errors": errors,
        "limits": {key: _LIMITS[key] for key in errors},
        "executed_inference_backend": result.metadata.get("numerical_backend"),
        "executed_inference_device": result.metadata.get("numerical_device"),
        "status": "success",
    }


def _linear_case(cp, concrete_device: str, cov_type: str):
    _, X, y, _ = _problem()
    cpu = LinearRegression(
        device="cpu", compute_inference=True, cov_type=cov_type
    ).fit(X, y)
    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    gpu = LinearRegression(
        device="cuda", compute_inference=True, cov_type=cov_type
    ).fit(X_gpu, y_gpu)

    if int(cpu.rank_) != 2 or int(gpu.rank_) != 2:
        raise AssertionError(
            f"LinearRegression {cov_type} rank mismatch: cpu={cpu.rank_}, gpu={gpu.rank_}"
        )
    result = gpu._inference_result
    if result is None:
        raise AssertionError(f"LinearRegression {cov_type} inference is missing")
    if result.metadata.get("numerical_backend") != "cupy":
        raise AssertionError(f"LinearRegression {cov_type} backend mismatch: {result.metadata}")
    if str(result.metadata.get("numerical_device", "")) != concrete_device:
        raise AssertionError(f"LinearRegression {cov_type} device mismatch: {result.metadata}")

    errors = {
        "prediction": _max_error(gpu.predict(X_gpu), cpu.predict(X)),
        "coef": _max_error(gpu.coef_, cpu.coef_),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    _assert_errors(f"linear_rank_recovery_{cov_type}", errors)
    return {
        "case": f"linear_rank_recovery_{cov_type}",
        "rank": int(gpu.rank_),
        "errors": errors,
        "limits": {key: _LIMITS[key] for key in errors},
        "executed_inference_backend": result.metadata.get("numerical_backend"),
        "executed_inference_device": result.metadata.get("numerical_device"),
        "status": "success",
    }


def _exact_l2_case(cp, concrete_device: str):
    _, X, y, _ = _problem()
    kwargs = dict(
        loss="squared_error",
        penalty="l2",
        alpha=0.0,
        fit_intercept=True,
        solver="exact",
        compute_inference=True,
        cov_type="nonrobust",
    )
    cpu = PenalizedGeneralizedLinearModel(device="cpu", **kwargs).fit(X, y)
    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    gpu = PenalizedGeneralizedLinearModel(device="cuda", **kwargs).fit(X_gpu, y_gpu)

    result = gpu._inference_result
    if result is None:
        raise AssertionError("exact-L2 alpha=0 CuPy inference is missing")
    if str(getattr(gpu, "_selected_backend_name", "")).lower() != "cupy":
        raise AssertionError(
            f"exact-L2 fit backend mismatch: {getattr(gpu, '_selected_backend_name', None)!r}"
        )
    if result.metadata.get("numerical_backend") != "cupy":
        raise AssertionError(f"exact-L2 inference backend mismatch: {result.metadata}")
    if str(result.metadata.get("numerical_device", "")) != concrete_device:
        raise AssertionError(f"exact-L2 inference device mismatch: {result.metadata}")

    errors = {
        "prediction": _max_error(gpu.predict(X_gpu), cpu.predict(X)),
        "coef": _max_error(gpu.coef_, cpu.coef_),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._tvalues, cpu._tvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    _assert_errors("exact_l2_alpha0_rank_recovery", errors)
    return {
        "case": "exact_l2_alpha0_rank_recovery",
        "alpha": 0.0,
        "errors": errors,
        "limits": {key: _LIMITS[key] for key in errors},
        "executed_backend": getattr(gpu, "_selected_backend_name", None),
        "executed_inference_backend": result.metadata.get("numerical_backend"),
        "executed_inference_device": result.metadata.get("numerical_device"),
        "status": "success",
    }


def _shared_solve_failure_case(cp, concrete_device: str):
    with cp.cuda.Device(int(concrete_device.split(":", 1)[1])):
        matrix = cp.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=cp.float64)
        rhs = cp.asarray([1.0, 1.0], dtype=cp.float64)

    try:
        value = xp_cholesky_solve(matrix, rhs, cp)
    except Exception as exc:
        if not _linalg_exception_is_rank_failure(exc):
            raise AssertionError(
                "shared CuPy solve raised an unrelated failure instead of a rank failure: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        error_type = type(exc).__name__
        error_message = str(exc)
    else:
        returned = np.asarray(_to_numpy(value), dtype=np.float64)
        raise AssertionError(
            "shared CuPy solve silently returned for a singular system instead of "
            f"surfacing a rank failure: {returned!r}"
        )

    return {
        "case": "shared_xp_solve_rank_failure_visible",
        "executed_device": concrete_device,
        "error_type": error_type,
        "error": error_message,
        "status": "success",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--validation-tier", required=True, choices=_VALIDATION_TIERS
    )
    args = parser.parse_args(argv)

    git_sha = _git("rev-parse", "HEAD")
    if git_sha != args.expected_sha:
        raise RuntimeError(
            f"exact-source gate failed: HEAD={git_sha}, expected={args.expected_sha}"
        )
    clean_before = _clean_worktree()
    if not clean_before:
        raise RuntimeError("physical acceptance requires a clean working tree")

    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() <= 0:
        raise RuntimeError("CuPy imported but no CUDA device is available")
    device_id = int(cp.cuda.runtime.getDevice())
    concrete_device = f"cuda:{device_id}"

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha,
        "validation_tier": args.validation_tier,
        "working_tree_clean_before": clean_before,
        "working_tree_clean_after_checks": False,
        "python_version": platform.python_version(),
        "gpu_model": _gpu_model(),
        "cupy_version": cp.__version__,
        "executed_device": concrete_device,
        "results": [],
        "status": "running",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with cp.cuda.Device(device_id):
            artifact["results"].append(_direct_inference_case(cp, concrete_device))
            artifact["results"].append(_linear_case(cp, concrete_device, "nonrobust"))
            artifact["results"].append(_linear_case(cp, concrete_device, "hc3"))
            artifact["results"].append(_exact_l2_case(cp, concrete_device))
            artifact["results"].append(
                _shared_solve_failure_case(cp, concrete_device)
            )
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
