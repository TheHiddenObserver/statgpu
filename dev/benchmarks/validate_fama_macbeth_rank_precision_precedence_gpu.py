#!/usr/bin/env python3
"""Physical CuPy/Torch gate for Fama-MacBeth rank/precision error precedence."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import statgpu.panel._fama_macbeth as fmb_module
from statgpu.panel import FamaMacBeth
from statgpu.panel._linalg import panel_lstsq_batched, panel_lstsq_deferred_rank


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True) == ""


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fixture(n_periods: int = 3):
    amplitude = float(2.0**55)
    x_period = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    X_period = np.column_stack([x_period, x_period])
    y_period = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    design = np.column_stack([np.ones(x_period.size), X_period])
    if np.linalg.matrix_rank(design) != 2:
        raise AssertionError("rank/precision fixture must remain rank two")
    if (design.T @ y_period)[0] != 0.0:
        raise AssertionError("rank/precision fixture must erase the raw intercept tail")
    X = np.tile(X_period, (n_periods, 1))
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    return X, y, time, design, y_period


def _gpu_name(backend: str) -> str:
    if backend == "cupy":
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props.get("name", "unknown")
        return name.decode() if isinstance(name, bytes) else str(name)
    import torch

    return str(torch.cuda.get_device_name(torch.cuda.current_device()))


def _trace_public_fallback(backend: str):
    """Wrap the actual FMB fallback global and capture backend-native execution."""
    trace = []
    if backend == "cupy":
        import cupy as cp

        original = fmb_module.panel_lstsq_deferred_rank

        def tracked(X, y, xp):
            trace.append(
                {
                    "namespace": getattr(xp, "__name__", ""),
                    "x_native": isinstance(X, cp.ndarray),
                    "y_native": isinstance(y, cp.ndarray),
                    "x_device": int(X.device.id) if isinstance(X, cp.ndarray) else None,
                    "y_device": int(y.device.id) if isinstance(y, cp.ndarray) else None,
                }
            )
            return original(X, y, xp)

        fmb_module.panel_lstsq_deferred_rank = tracked
        return trace, "panel_lstsq_deferred_rank", original

    if backend == "torch":
        import torch

        original = fmb_module.panel_lstsq_batched

        def tracked(X, y, xp):
            trace.append(
                {
                    "namespace": getattr(xp, "__name__", ""),
                    "x_native": isinstance(X, torch.Tensor),
                    "y_native": isinstance(y, torch.Tensor),
                    "x_device": str(X.device) if isinstance(X, torch.Tensor) else None,
                    "y_device": str(y.device) if isinstance(y, torch.Tensor) else None,
                    "x_is_cuda": bool(X.is_cuda) if isinstance(X, torch.Tensor) else False,
                    "y_is_cuda": bool(y.is_cuda) if isinstance(y, torch.Tensor) else False,
                }
            )
            return original(X, y, xp)

        fmb_module.panel_lstsq_batched = tracked
        return trace, "panel_lstsq_batched", original

    raise ValueError(backend)


def _restore_public_fallback(name: str, original):
    setattr(fmb_module, name, original)


def _validate_public_trace(backend: str, trace):
    if not trace:
        raise AssertionError(f"{backend}: public FamaMacBeth fit never entered GPU fallback")
    for call in trace:
        if call["namespace"] != backend:
            raise AssertionError(
                f"{backend}: public fallback namespace={call['namespace']!r}, expected {backend!r}"
            )
        if not call["x_native"] or not call["y_native"]:
            raise AssertionError(f"{backend}: public fallback received non-native arrays: {call}")
        if backend == "cupy":
            if call["x_device"] is None or call["y_device"] is None:
                raise AssertionError(f"cupy: missing CUDA device provenance: {call}")
        else:
            if not call["x_is_cuda"] or not call["y_is_cuda"]:
                raise AssertionError(f"torch: public fallback left CUDA: {call}")
            if not str(call["x_device"]).startswith("cuda") or not str(
                call["y_device"]
            ).startswith("cuda"):
                raise AssertionError(f"torch: unexpected fallback devices: {call}")


def run(backend: str):
    if not _git_clean():
        raise RuntimeError("physical validation requires a clean git worktree")

    cupy_runtime_version = None
    torch_runtime_version = None
    X_np, y_np, time_np, design_np, y_period_np = _fixture()
    if backend == "cupy":
        import cupy as cp

        cupy_runtime_version = str(cp.__version__)
        design = cp.asarray(design_np, dtype=cp.float64)
        y_period = cp.asarray(y_period_np, dtype=cp.float64)
        _params, rank_backend = panel_lstsq_deferred_rank(design, y_period, cp)
        rank = int(cp.asnumpy(rank_backend).item())
        X = cp.asarray(X_np, dtype=cp.float64)
        y = cp.asarray(y_np, dtype=cp.float64)
        time = cp.asarray(time_np, dtype=cp.int64)
        device = "cuda"
    elif backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is not available")
        torch_runtime_version = str(torch.__version__)
        design = torch.as_tensor(design_np, dtype=torch.float64, device="cuda")
        y_period = torch.as_tensor(y_period_np, dtype=torch.float64, device="cuda")
        _params, ranks = panel_lstsq_batched(
            design.reshape(1, *design.shape),
            y_period.reshape(1, -1),
            torch,
        )
        rank = int(ranks[0].item())
        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
        y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")
        time = torch.as_tensor(time_np, dtype=torch.int64, device="cuda")
        device = "torch"
    else:
        raise ValueError(backend)

    if rank != 2:
        raise AssertionError(f"{backend}: fallback rank={rank}, expected 2")

    public_rank_failure = False
    model = FamaMacBeth(device=device, bandwidth=0)
    trace, traced_name, original = _trace_public_fallback(backend)
    try:
        try:
            model.fit(X, y, time_ids=time)
        except FloatingPointError as exc:
            raise AssertionError(
                f"{backend}: rank deficiency was misreported as precision failure: {exc}"
            ) from exc
        except ValueError as exc:
            text = str(exc)
            if "rank deficient" not in text or "rank=2, columns=3" not in text:
                raise
            public_rank_failure = True
    finally:
        _restore_public_fallback(traced_name, original)

    if not public_rank_failure:
        raise AssertionError(f"{backend}: rank-deficient period did not fail closed")
    _validate_public_trace(backend, trace)
    executed_backend = str(trace[0]["namespace"])

    leaked_state = [
        name
        for name in ("coef_", "betas_", "_backend_name", "_inference_result", "_xp")
        if hasattr(model, name)
    ]
    if leaked_state or bool(getattr(model, "_fitted", False)):
        raise AssertionError(
            f"{backend}: failed public fit leaked fitted/backend state: {leaked_state}"
        )

    return {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "clean_worktree": True,
        "requested_backend": backend,
        "executed_backend": executed_backend,
        "public_fallback_trace": trace,
        "public_failure_state_clean": True,
        "inference_state_published": False,
        "gpu": _gpu_name(backend),
        "packages": {
            "statgpu": _version("statgpu"),
            "numpy": str(np.__version__),
            "cupy": cupy_runtime_version
            or _version("cupy-cuda11x")
            or _version("cupy-cuda12x")
            or _version("cupy"),
            "torch": torch_runtime_version or _version("torch"),
        },
        "fallback_svd_rank": rank,
        "public_rank_deficiency_value_error": public_rank_failure,
        "precision_failure_misclassification": False,
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
