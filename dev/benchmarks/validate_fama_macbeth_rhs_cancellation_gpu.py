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

import statgpu.panel._fama_macbeth as fmb_module
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


def _nonzero_rhs_svd_drift_fixture(n_periods: int = 3):
    amplitude = float(2.0**55)
    x_period = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    y_period = np.asarray([-16.0, amplitude, -16.0, -amplitude], dtype=np.float64)
    design = np.column_stack([np.ones(x_period.size), x_period])
    if np.linalg.cond(design) != 1.0:
        raise AssertionError("SVD-drift fixture must remain condition-one")
    expected_rhs = np.asarray([-32.0, -32.0], dtype=np.float64)
    if not np.array_equal(design.T @ y_period, expected_rhs):
        raise AssertionError("SVD-drift fixture must retain a nonzero ordinary RHS")
    X = np.tile(x_period, n_periods)[:, None]
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    return X, y, time


def _genuine_zero_rhs_fixture(n_periods: int = 3):
    amplitude = float(2.0**55)
    x_period = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    y_period = np.asarray([amplitude, -amplitude, -amplitude, amplitude], dtype=np.float64)
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

        X_dev = cp.asarray(X, dtype=cp.float64)
        y_dev = cp.asarray(y, dtype=cp.float64)
        time_dev = cp.asarray(time, dtype=cp.int64)
        return X_dev, y_dev, time_dev, "cuda", int(X_dev.device.id)
    if backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is not available")
        X_dev = torch.as_tensor(X, dtype=torch.float64, device="cuda")
        y_dev = torch.as_tensor(y, dtype=torch.float64, device="cuda")
        time_dev = torch.as_tensor(time, dtype=torch.int64, device="cuda")
        return X_dev, y_dev, time_dev, "torch", str(X_dev.device)
    raise ValueError(backend)


def _gpu_name(backend: str, execution_device) -> str:
    if backend == "cupy":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(int(execution_device))
        name = props.get("name", "unknown")
        return name.decode() if isinstance(name, bytes) else str(name)
    import torch

    return str(torch.cuda.get_device_name(torch.device(str(execution_device))))


def _trace_public_fallback(backend: str):
    """Wrap the exact FMB fallback global and capture native input/output devices."""
    trace = []
    if backend == "cupy":
        import cupy as cp

        original = fmb_module.panel_lstsq_deferred_rank

        def tracked(X, y, xp):
            params, rank = original(X, y, xp)
            trace.append(
                {
                    "namespace": getattr(xp, "__name__", ""),
                    "x_native": isinstance(X, cp.ndarray),
                    "y_native": isinstance(y, cp.ndarray),
                    "params_native": isinstance(params, cp.ndarray),
                    "rank_native": isinstance(rank, cp.ndarray),
                    "x_device": int(X.device.id) if isinstance(X, cp.ndarray) else None,
                    "y_device": int(y.device.id) if isinstance(y, cp.ndarray) else None,
                    "params_device": int(params.device.id)
                    if isinstance(params, cp.ndarray)
                    else None,
                    "rank_device": int(rank.device.id)
                    if isinstance(rank, cp.ndarray)
                    else None,
                }
            )
            return params, rank

        fmb_module.panel_lstsq_deferred_rank = tracked
        return trace, "panel_lstsq_deferred_rank", original

    if backend == "torch":
        import torch

        original = fmb_module.panel_lstsq_batched

        def tracked(X, y, xp):
            params, ranks = original(X, y, xp)
            trace.append(
                {
                    "namespace": getattr(xp, "__name__", ""),
                    "x_native": isinstance(X, torch.Tensor),
                    "y_native": isinstance(y, torch.Tensor),
                    "params_native": isinstance(params, torch.Tensor),
                    "rank_native": isinstance(ranks, torch.Tensor),
                    "x_device": str(X.device) if isinstance(X, torch.Tensor) else None,
                    "y_device": str(y.device) if isinstance(y, torch.Tensor) else None,
                    "params_device": str(params.device)
                    if isinstance(params, torch.Tensor)
                    else None,
                    "rank_device": str(ranks.device)
                    if isinstance(ranks, torch.Tensor)
                    else None,
                    "x_is_cuda": bool(X.is_cuda) if isinstance(X, torch.Tensor) else False,
                    "y_is_cuda": bool(y.is_cuda) if isinstance(y, torch.Tensor) else False,
                    "params_is_cuda": bool(params.is_cuda)
                    if isinstance(params, torch.Tensor)
                    else False,
                    "rank_is_cuda": bool(ranks.is_cuda)
                    if isinstance(ranks, torch.Tensor)
                    else False,
                }
            )
            return params, ranks

        fmb_module.panel_lstsq_batched = tracked
        return trace, "panel_lstsq_batched", original

    raise ValueError(backend)


def _restore_public_fallback(name: str, original):
    setattr(fmb_module, name, original)


def _validate_public_trace(backend: str, trace, execution_device):
    if not trace:
        raise AssertionError(f"{backend}: expected public SVD fallback was not executed")
    for call in trace:
        if call["namespace"] != backend:
            raise AssertionError(f"{backend}: fallback namespace mismatch: {call}")
        if not all(
            (call["x_native"], call["y_native"], call["params_native"], call["rank_native"])
        ):
            raise AssertionError(f"{backend}: fallback crossed to non-native arrays: {call}")
        devices = (
            call["x_device"],
            call["y_device"],
            call["params_device"],
            call["rank_device"],
        )
        if len(set(devices)) != 1 or devices[0] != execution_device:
            raise AssertionError(f"{backend}: inconsistent fallback device provenance: {call}")
        if backend == "torch" and not all(
            (
                call["x_is_cuda"],
                call["y_is_cuda"],
                call["params_is_cuda"],
                call["rank_is_cuda"],
            )
        ):
            raise AssertionError(f"torch: fallback left CUDA: {call}")


def _assert_success_backend(model, backend: str, execution_device):
    executed = getattr(model, "_backend_name", None)
    if executed != backend:
        raise AssertionError(f"requested {backend}, executed {executed!r}")
    if backend == "cupy":
        import cupy as cp

        for name in ("coef_", "betas_"):
            value = getattr(model, name)
            if not isinstance(value, cp.ndarray) or int(value.device.id) != execution_device:
                raise AssertionError(f"cupy: {name} is not resident on device {execution_device}")
    else:
        import torch

        for name in ("coef_", "betas_"):
            value = getattr(model, name)
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_cuda
                or str(value.device) != execution_device
            ):
                raise AssertionError(f"torch: {name} is not resident on {execution_device}")


def _expected_precision_failure(backend, X, y, time, device, execution_device):
    trace, traced_name, original = _trace_public_fallback(backend)
    failed_closed = False
    try:
        try:
            FamaMacBeth(device=device, bandwidth=0).fit(X, y, time_ids=time)
        except FloatingPointError as exc:
            if "FamaMacBeth period coefficient resolution exceeds float64 precision" not in str(exc):
                raise
            failed_closed = True
    finally:
        _restore_public_fallback(traced_name, original)
    if not failed_closed:
        raise AssertionError("expected Fama-MacBeth precision failure did not occur")
    _validate_public_trace(backend, trace, execution_device)
    return trace


def run(backend: str):
    if not _git_clean():
        raise RuntimeError("physical validation requires a clean git worktree")

    cupy_runtime_version = None
    torch_runtime_version = None
    X_np, y_np, time_np, amplitude = _intercept_tail_fixture()
    X, y, time, device, execution_device = _backend_arrays(
        backend, X_np, y_np, time_np
    )
    if backend == "cupy":
        import cupy as cp

        cupy_runtime_version = str(cp.__version__)
    else:
        import torch

        torch_runtime_version = str(torch.__version__)

    model = FamaMacBeth(device=device, bandwidth=0).fit(X, y, time_ids=time)
    _assert_success_backend(model, backend, execution_device)
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
    X_bad, y_bad, time_bad, device, bad_device = _backend_arrays(
        backend, X_bad_np, y_bad_np, time_bad_np
    )
    lost_zero_trace = _expected_precision_failure(
        backend, X_bad, y_bad, time_bad, device, bad_device
    )

    X_drift_np, y_drift_np, time_drift_np = _nonzero_rhs_svd_drift_fixture()
    X_drift, y_drift, time_drift, device, drift_device = _backend_arrays(
        backend, X_drift_np, y_drift_np, time_drift_np
    )
    nonzero_drift_trace = _expected_precision_failure(
        backend, X_drift, y_drift, time_drift, device, drift_device
    )

    X_zero_np, y_zero_np, time_zero_np = _genuine_zero_rhs_fixture()
    X_zero, y_zero, time_zero, device, zero_device = _backend_arrays(
        backend, X_zero_np, y_zero_np, time_zero_np
    )
    zero_model = FamaMacBeth(device=device, bandwidth=0).fit(
        X_zero, y_zero, time_ids=time_zero
    )
    _assert_success_backend(zero_model, backend, zero_device)
    zero_betas = np.asarray(_to_numpy(zero_model.betas_), dtype=np.float64)
    zero_coef = np.asarray(_to_numpy(zero_model.coef_), dtype=np.float64)
    np.testing.assert_array_equal(zero_betas, np.zeros_like(zero_betas))
    np.testing.assert_array_equal(zero_coef, np.zeros_like(zero_coef))
    if zero_model._period_svd_fallbacks != zero_model.n_periods:
        raise AssertionError("genuine-zero RHS should be rechecked by the SVD fallback")

    return {
        "schema_version": 3,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "validation_tier": "remote-full",
        "clean_worktree": True,
        "requested_backend": backend,
        "executed_backend": getattr(model, "_backend_name", None),
        "execution_device": execution_device,
        "gpu": _gpu_name(backend, execution_device),
        "packages": {
            "statgpu": _version("statgpu"),
            "numpy": str(np.__version__),
            "cupy": cupy_runtime_version
            or _version("cupy-cuda11x")
            or _version("cupy-cuda12x")
            or _version("cupy"),
            "torch": torch_runtime_version or _version("torch"),
        },
        "intercept_tail": {
            "intercept": float(coef[0]),
            "slope": float(coef[1]),
            "solver_mode": model._period_solver_mode,
            "svd_fallbacks": int(model._period_svd_fallbacks),
        },
        "lost_nonconstant_rhs_tail_fail_closed": True,
        "lost_nonconstant_rhs_trace": lost_zero_trace,
        "nonzero_rhs_svd_drift_fail_closed": True,
        "nonzero_rhs_svd_drift_trace": nonzero_drift_trace,
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
