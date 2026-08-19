#!/usr/bin/env python3
"""Physical CuPy/Torch gate for Fama-MacBeth Gram-RHS cancellation safety."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.panel import FamaMacBeth


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True) == ""


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _intercept_tail_fixture(n_periods: int = 3):
    amplitude = float(2.0**55)
    x_period = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    y_period = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    X = np.tile(x_period, n_periods)[:, None]
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    return X, y, time, amplitude


def _ambiguous_nonconstant_rhs_fixture(n_periods: int = 3):
    amplitude = float(2.0**55)
    x_period = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    y_period = np.asarray([-3.0, amplitude, 2.0, -amplitude], dtype=np.float64)
    X = np.tile(x_period, n_periods)[:, None]
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    design = np.column_stack([np.ones(x_period.size), x_period])
    if np.linalg.cond(design) != 1.0:
        raise AssertionError("ambiguous-RHS fixture must remain condition-one")
    return X, y, time


def _genuine_zero_rhs_fixture(n_periods: int = 3):
    x_period = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    y_period = np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    X = np.tile(x_period, n_periods)[:, None]
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    design = np.column_stack([np.ones(x_period.size), x_period])
    if np.linalg.cond(design) != 1.0:
        raise AssertionError("genuine-zero fixture must remain condition-one")
    if not np.array_equal(design.T @ y_period, np.zeros(2, dtype=np.float64)):
        raise AssertionError("genuine-zero fixture must have an exact zero RHS")
    return X, y, time


def _backend_arrays(backend: str, X, y, time):
    if backend == "cupy":
        import cupy as cp

        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(y, dtype=cp.float64),
            cp.asarray(time, dtype=cp.int64),
            "cuda",
        )
    if backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is not available")
        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
            torch.as_tensor(time, dtype=torch.int64, device="cuda"),
            "torch",
        )
    raise ValueError(backend)


def _gpu_name(backend: str) -> str:
    if backend == "cupy":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props.get("name", "unknown")
        return name.decode() if isinstance(name, bytes) else str(name)
    import torch

    return str(torch.cuda.get_device_name(torch.cuda.current_device()))


def _assert_executed_backend(model, backend: str):
    executed = getattr(model, "_backend_name", None)
    if executed != backend:
        raise AssertionError(f"requested {backend}, executed {executed!r}")


def run(backend: str):
    if not _git_clean():
        raise RuntimeError("physical validation requires a clean git worktree")

    X_np, y_np, time_np, amplitude = _intercept_tail_fixture()
    X, y, time, device = _backend_arrays(backend, X_np, y_np, time_np)
    model = FamaMacBeth(device=device, bandwidth=0).fit(X, y, time_ids=time)
    _assert_executed_backend(model, backend)
    betas = np.asarray(_to_numpy(model.betas_), dtype=np.float64)
    coef = np.asarray(_to_numpy(model.coef_), dtype=np.float64)
    np.testing.assert_allclose(
        betas[:, 0],
        np.full(model.n_periods, 1.0 / 3.0),
        rtol=1.2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(coef[0], 1.0 / 3.0, rtol=1.2e-14, atol=0.0)
    np.testing.assert_allclose(betas[:, 1], -amplitude, rtol=5.0e-15, atol=0.0)
    if model._period_solver_mode != "gram-certified":
        raise AssertionError(
            f"intercept-tail fixture used {model._period_solver_mode!r}, expected gram-certified"
        )
    if model._period_svd_fallbacks != 0:
        raise AssertionError("stable intercept Gram RHS unexpectedly required SVD fallback")

    X_bad_np, y_bad_np, time_bad_np = _ambiguous_nonconstant_rhs_fixture()
    X_bad, y_bad, time_bad, device = _backend_arrays(
        backend, X_bad_np, y_bad_np, time_bad_np
    )
    failed_closed = False
    try:
        FamaMacBeth(device=device, bandwidth=0).fit(
            X_bad, y_bad, time_ids=time_bad
        )
    except FloatingPointError as exc:
        if "FamaMacBeth period coefficient resolution exceeds float64 precision" not in str(exc):
            raise
        failed_closed = True
    if not failed_closed:
        raise AssertionError("lost nonconstant Gram-RHS tail did not fail closed")

    X_zero_np, y_zero_np, time_zero_np = _genuine_zero_rhs_fixture()
    X_zero, y_zero, time_zero, device = _backend_arrays(
        backend, X_zero_np, y_zero_np, time_zero_np
    )
    zero_model = FamaMacBeth(device=device, bandwidth=0).fit(
        X_zero, y_zero, time_ids=time_zero
    )
    _assert_executed_backend(zero_model, backend)
    zero_betas = np.asarray(_to_numpy(zero_model.betas_), dtype=np.float64)
    zero_coef = np.asarray(_to_numpy(zero_model.coef_), dtype=np.float64)
    np.testing.assert_allclose(zero_betas, np.zeros_like(zero_betas), rtol=0.0, atol=5.0e-15)
    np.testing.assert_allclose(zero_coef, np.zeros_like(zero_coef), rtol=0.0, atol=5.0e-15)
    if zero_model._period_svd_fallbacks != zero_model.n_periods:
        raise AssertionError("genuine-zero RHS should be rechecked by the SVD fallback")

    return {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "clean_worktree": True,
        "requested_backend": backend,
        "executed_backend": getattr(model, "_backend_name", None),
        "gpu": _gpu_name(backend),
        "packages": {
            "statgpu": _version("statgpu"),
            "numpy": _version("numpy"),
            "cupy": _version("cupy-cuda11x") or _version("cupy-cuda12x") or _version("cupy"),
            "torch": _version("torch"),
        },
        "intercept_tail": {
            "intercept": float(coef[0]),
            "slope": float(coef[1]),
            "solver_mode": model._period_solver_mode,
            "svd_fallbacks": int(model._period_svd_fallbacks),
        },
        "lost_nonconstant_rhs_tail_fail_closed": failed_closed,
        "genuine_zero_rhs": {
            "max_abs_beta": float(np.max(np.abs(zero_betas))),
            "max_abs_coef": float(np.max(np.abs(zero_coef))),
            "svd_fallbacks": int(zero_model._period_svd_fallbacks),
        },
        "status": "success",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("cupy", "torch"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.backend)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
