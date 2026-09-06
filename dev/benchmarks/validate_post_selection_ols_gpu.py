#!/usr/bin/env python3
"""Physical CuPy/Torch validation for post-selection OLS inference (#137).

This validator is intentionally separate from hosted CPU CI. Canonical physical
acceptance requires both public explicit-GPU routes, fitted-backend provenance,
active-set identity, and coefficient/SE/statistic/p-value/CI parity with the
NumPy reference. Explicit CUDA/Torch requests must fail rather than fall back.
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

from statgpu.linear_model import Lasso

_REQUIRED_BACKENDS = ("cupy", "torch")


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


def _max_error(a, b) -> float:
    """Compare finite values after requiring identical NaN reporting masks."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise AssertionError(
            f"shape mismatch: left={left.shape}, right={right.shape}"
        )
    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    if not np.array_equal(left_nan, right_nan):
        raise AssertionError(
            "CPU/GPU inference NaN masks differ; inactive/uninferred coordinates "
            "must agree exactly before finite-value parity is assessed"
        )
    finite = ~left_nan
    if not np.any(finite):
        return 0.0
    if not np.all(np.isfinite(left[finite])) or not np.all(np.isfinite(right[finite])):
        raise AssertionError("non-finite non-NaN values encountered in parity check")
    return float(np.max(np.abs(left[finite] - right[finite])))


def _fit(X, y, *, device: str, sample_weight=None):
    model = Lasso(
        alpha=0.05,
        solver="fista",
        inference_method="post_selection_ols",
        compute_inference=True,
        device=device,
        max_iter=6000,
        tol=1e-9,
    )
    model.fit(X, y, sample_weight=sample_weight)
    if model._inference_result is None:
        raise AssertionError("post-selection OLS inference result is missing")
    return model


def _case(backend: str, *, weighted: bool):
    X, y, weights = _problem(seed=137 + int(weighted))
    sw = weights if weighted else None
    cpu = _fit(X, y, device="cpu", sample_weight=sw)

    X_native = _native(X, backend)
    y_native = _native(y, backend)
    sw_native = None if sw is None else _native(sw, backend)
    requested_device = "cuda" if backend == "cupy" else "torch"
    expected_device, version = _runtime(backend)
    gpu = _fit(
        X_native,
        y_native,
        device=requested_device,
        sample_weight=sw_native,
    )

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
    if gpu._inference_result.statistic_name != "z":
        raise AssertionError("post-selection OLS must preserve z-statistic semantics")
    if gpu._inference_result.distribution != "normal":
        raise AssertionError("post-selection OLS must preserve normal reference inference")

    cpu_selected = cpu._inference_result.metadata["selected_feature_indices"]
    gpu_selected = gpu._inference_result.metadata["selected_feature_indices"]
    if cpu_selected != gpu_selected:
        raise AssertionError(
            f"active-set mismatch: cpu={cpu_selected}, {backend}={gpu_selected}"
        )

    errors = {
        "penalized_coef": _max_error(gpu.coef_, cpu.coef_),
        "post_selection_params": _max_error(gpu._params, cpu._params),
        "bse": _max_error(gpu._bse, cpu._bse),
        "statistic": _max_error(gpu._zvalues, cpu._zvalues),
        "pvalue": _max_error(gpu._pvalues, cpu._pvalues),
        "ci": _max_error(gpu._conf_int, cpu._conf_int),
    }
    limits = {
        "penalized_coef": 2e-6,
        "post_selection_params": 2e-7,
        "bse": 2e-7,
        "statistic": 2e-5,
        "pvalue": 2e-6,
        "ci": 5e-7,
    }
    for key, limit in limits.items():
        value = errors[key]
        if not np.isfinite(value) or value > limit:
            raise AssertionError(
                f"{backend} weighted={weighted} {key} error {value:.3e} "
                f"exceeds {limit:.3e}"
            )

    return {
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
        "errors": errors,
        "limits": limits,
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
        "schema_version": 2,
        "issue": 137,
        "head_sha": _git("rev-parse", "HEAD"),
        "worktree_clean": _git("status", "--porcelain") == "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cases": [],
    }
    if not payload["worktree_clean"]:
        raise RuntimeError("physical acceptance requires a clean worktree")

    for backend in backends:
        for weighted in (False, True):
            payload["cases"].append(_case(backend, weighted=weighted))

    payload["status"] = "success"
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
