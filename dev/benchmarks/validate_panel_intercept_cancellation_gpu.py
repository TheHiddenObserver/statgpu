#!/usr/bin/env python3
"""Physical CuPy/Torch gate for cancellation-safe panel response paths."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.panel import BetweenOLS, PooledOLS, RandomEffects


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    )
    return status == ""


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fixture():
    amplitude = float(2.0**55)
    X = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    y = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    return X, y, amplitude


def _random_effects_component_loss_fixture():
    amplitude = float(2.0**55)
    within = 8.0
    levels = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    y = np.concatenate(
        [np.asarray([level + within, level - within]) for level in levels]
    )
    X = np.ones((y.size, 1), dtype=np.float64)
    entity = np.repeat(np.arange(levels.size, dtype=np.int64), 2)
    return X, y, entity


def _random_effects_common_scale_loss_fixture(*, amplitude, tiny_within):
    y = np.asarray(
        [amplitude, amplitude, tiny_within, -tiny_within, -amplitude, -amplitude],
        dtype=np.float64,
    )
    X = np.ones((y.size, 1), dtype=np.float64)
    entity = np.repeat(np.arange(3, dtype=np.int64), 2)
    return X, y, entity


def _backend_arrays(backend: str, X, y, entity=None):
    if backend == "cupy":
        import cupy as cp

        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(y, dtype=cp.float64),
            None if entity is None else cp.asarray(entity, dtype=cp.int64),
            cp,
            "cuda",
        )
    if backend == "torch":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Torch CUDA is not available")
        return (
            torch.as_tensor(X, dtype=torch.float64, device="cuda"),
            torch.as_tensor(y, dtype=torch.float64, device="cuda"),
            None
            if entity is None
            else torch.as_tensor(entity, dtype=torch.int64, device="cuda"),
            torch,
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


def _assert_model(model, *, backend: str, amplitude: float, label: str):
    coef = np.asarray(_to_numpy(model.coef_), dtype=np.float64).ravel()
    np.testing.assert_allclose(
        coef[0], 1.0 / 3.0, rtol=8.0e-15, atol=0.0,
        err_msg=f"{backend}: {label} cancellation-tail intercept",
    )
    np.testing.assert_allclose(
        coef[1], -amplitude, rtol=8.0e-15, atol=0.0,
        err_msg=f"{backend}: {label} cancellation-tail slope",
    )
    executed = getattr(model, "_backend_name", None)
    if executed != backend:
        raise AssertionError(
            f"{backend}: {label} requested GPU backend but executed {executed!r}"
        )
    if not np.all(np.isfinite(coef)):
        raise AssertionError(f"{backend}: {label} produced non-finite coefficients")
    return coef


def _assert_random_effects_fail_closed(backend: str, fixture, message: str, label: str):
    X_np, y_np, entity_np = fixture
    X, y, entity, _xp, device = _backend_arrays(
        backend, X_np, y_np, entity_np
    )
    model = RandomEffects(device=device)
    try:
        model.fit(X, y, entity_ids=entity)
    except FloatingPointError as exc:
        if message not in str(exc):
            raise
    else:
        raise AssertionError(f"{backend}: RandomEffects did not fail closed on {label}")
    if getattr(model, "_backend_name", None) != backend:
        raise AssertionError(
            f"{backend}: RandomEffects fail-closed path lost backend provenance"
        )
    return True


def run(backend: str):
    if not _git_clean():
        raise RuntimeError("physical validation requires a clean git worktree")

    X_np, y_np, amplitude = _fixture()
    X, y, _unused, _xp, device = _backend_arrays(backend, X_np, y_np)
    pooled = PooledOLS(device=device).fit(X, y)
    pooled_coef = _assert_model(
        pooled, backend=backend, amplitude=amplitude, label="PooledOLS"
    )

    X_level_np = np.repeat(X_np, 2, axis=0)
    y_level_np = np.repeat(y_np, 2)
    entity_np = np.repeat(np.arange(3, dtype=np.int64), 2)
    X_level, y_level, entity, _xp, device = _backend_arrays(
        backend, X_level_np, y_level_np, entity_np
    )
    between = BetweenOLS(device=device).fit(
        X_level, y_level, entity_ids=entity
    )
    between_coef = _assert_model(
        between, backend=backend, amplitude=amplitude, label="BetweenOLS"
    )

    quasi_fail = _assert_random_effects_fail_closed(
        backend,
        _random_effects_component_loss_fixture(),
        "quasi-demeaning exceeds the float64 component range",
        "lost quasi-demean component",
    )
    normalization_fail = _assert_random_effects_fail_closed(
        backend,
        _random_effects_common_scale_loss_fixture(
            amplitude=1.0e308,
            tiny_within=1.0e-100,
        ),
        "variance-component scaling exceeds the float64 common-residual range",
        "common normalization loss",
    )
    square_fail = _assert_random_effects_fail_closed(
        backend,
        _random_effects_common_scale_loss_fixture(
            amplitude=1.0e100,
            tiny_within=1.0e-100,
        ),
        "variance-component scaling exceeds the float64 common-residual range",
        "common RSS square underflow",
    )
    theta_rounding_fail = _assert_random_effects_fail_closed(
        backend,
        _random_effects_common_scale_loss_fixture(
            amplitude=1.0e17,
            tiny_within=1.0,
        ),
        "quasi-demeaning exceeds the float64 component range",
        "pre-rounded theta-complement loss",
    )

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "clean_worktree": True,
        "requested_backend": backend,
        "pooled_executed_backend": getattr(pooled, "_backend_name", None),
        "between_executed_backend": getattr(between, "_backend_name", None),
        "gpu": _gpu_name(backend),
        "packages": {
            "statgpu": _version("statgpu"),
            "numpy": _version("numpy"),
            "cupy": _version("cupy-cuda11x") or _version("cupy-cuda12x") or _version("cupy"),
            "torch": _version("torch"),
        },
        "fixture": {
            "amplitude": amplitude,
            "expected_intercept": 1.0 / 3.0,
            "expected_slope": -amplitude,
        },
        "pooled_coef": pooled_coef.tolist(),
        "between_coef": between_coef.tolist(),
        "random_effects_quasi_component_fail_closed": quasi_fail,
        "random_effects_common_normalization_fail_closed": normalization_fail,
        "random_effects_common_square_fail_closed": square_fail,
        "random_effects_theta_rounding_fail_closed": theta_rounding_fail,
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
