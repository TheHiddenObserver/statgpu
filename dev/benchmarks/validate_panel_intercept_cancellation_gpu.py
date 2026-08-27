#!/usr/bin/env python3
"""Physical CuPy/Torch gate for cancellation-safe panel least-squares paths."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from statgpu.backends import _to_numpy
from statgpu.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _git_clean() -> bool:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True) == ""


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


def _common_level_fixture():
    amplitude = float(2.0**55)
    X = np.asarray([[1.0], [1.0], [-1.0], [-1.0]], dtype=np.float64)
    y = amplitude + np.asarray([24.0, 0.0, -8.0, -32.0], dtype=np.float64)
    return X, y, 16.0


def _mixed_coefficient_resolution_fixture():
    amplitude = float(2.0**55)
    # Zero-mean near-collinear columns.  The zero column means keep the
    # constant-response anchor at zero even after PooledOLS prepends its
    # intercept, so the huge response scale is preserved for the resolution
    # check.  Adding the intercept keeps the design full-rank (the constant
    # direction is orthogonal to both perturbation directions) while the
    # condition number stays ~1e12, so the small coefficient sits below the
    # float64 resolution of the huge response and SVD roundoff is unavoidable
    # on every backend: the least-squares fit must fail closed.  The previous
    # well-conditioned fixture only tripped the resolution check through a
    # NumPy-SVD roundoff accident and could not fail closed after the
    # numerically exact Torch gesvd driver was introduced.
    delta = 1.0e-12
    col1 = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    col2 = col1 + delta * np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    X = np.column_stack([col1, col2])
    y = X @ np.asarray([16.0, amplitude], dtype=np.float64)
    return X, y


def _fama_macbeth_resolution_fixture(n_periods=3):
    # Well-conditioned period with a huge intercept and a small slope.  The
    # certified Gram path applies a deterministic RHS-rounding error bound
    # (independent of any SVD roundoff accident), so every backend must fail
    # closed on the unresolved slope coordinate.
    amplitude = float(2.0**55)
    X_period = np.asarray(
        [[2.0, 2.0], [-1.0, 2.0], [2.0, -2.0], [-2.0, 1.0]],
        dtype=np.float64,
    )
    y_period = X_period @ np.asarray([amplitude, 16.0], dtype=np.float64)
    X = np.tile(X_period, (n_periods, 1))
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), X_period.shape[0])
    return X, y, time


def _large_common_intercept_fmb_fixture(n_periods=4):
    x_period = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
    X = np.tile(x_period, n_periods)[:, None]
    y = np.full(X.shape[0], 6.0e307, dtype=np.float64)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    return X, y, time


def _first_difference_from_transformed(X_diff, y_diff):
    n = int(X_diff.shape[0])
    X = np.zeros((2 * n, int(X_diff.shape[1])), dtype=np.float64)
    y = np.zeros(2 * n, dtype=np.float64)
    X[1::2] = X_diff
    y[1::2] = y_diff
    entity = np.repeat(np.arange(n, dtype=np.int64), 2)
    time = np.tile(np.arange(2, dtype=np.int64), n)
    return X, y, entity, time


def _first_difference_extreme_r2_fixture():
    y_diff = np.concatenate(
        [np.asarray([1.0e308]), np.full(10, -1.0e308, dtype=np.float64)]
    )
    X_diff = (y_diff / 1.0e308).reshape(-1, 1)
    return _first_difference_from_transformed(X_diff, y_diff)


def _random_effects_component_loss_fixture():
    amplitude = float(2.0**55)
    within = 8.0
    levels = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    y = np.concatenate([np.asarray([level + within, level - within]) for level in levels])
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


def _backend_arrays(backend: str, X, y, metadata=None):
    if backend == "cupy":
        import cupy as cp
        return (
            cp.asarray(X, dtype=cp.float64),
            cp.asarray(y, dtype=cp.float64),
            None if metadata is None else cp.asarray(metadata, dtype=cp.int64),
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
            if metadata is None
            else torch.as_tensor(metadata, dtype=torch.int64, device="cuda"),
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


def _assert_backend(model, backend: str, label: str):
    executed = getattr(model, "_backend_name", None)
    if executed != backend:
        raise AssertionError(f"{label}: requested {backend}, executed {executed!r}")


def _assert_resolution_failure(callable_fit, *, label: str):
    try:
        callable_fit()
    except FloatingPointError as exc:
        if "coefficient resolution exceeds float64 precision" not in str(exc):
            raise
    else:
        raise AssertionError(f"{label}: unresolved coefficient fit did not fail closed")
    return True


def _assert_random_effects_fail_closed(backend: str, fixture, message: str, label: str):
    X_np, y_np, entity_np = fixture
    X, y, entity, device = _backend_arrays(backend, X_np, y_np, entity_np)
    model = RandomEffects(device=device, cov_type="hc0")
    try:
        model.fit(X, y, entity_ids=entity)
    except FloatingPointError as exc:
        if message not in str(exc):
            raise
    else:
        raise AssertionError(f"{backend}: RandomEffects did not fail closed on {label}")
    _assert_backend(model, backend, f"RandomEffects {label}")
    return True


def run(backend: str, validation_tier: str):
    if not _git_clean():
        raise RuntimeError("physical validation requires a clean git worktree")

    X_np, y_np, amplitude = _fixture()
    X, y, _unused, device = _backend_arrays(backend, X_np, y_np)
    pooled = PooledOLS(device=device, cov_type="hc0").fit(X, y)
    _assert_backend(pooled, backend, "PooledOLS cancellation")
    pooled_coef = np.asarray(_to_numpy(pooled.coef_), dtype=np.float64).ravel()
    np.testing.assert_allclose(pooled_coef[0], 1.0 / 3.0, rtol=8.0e-15, atol=0.0)
    np.testing.assert_allclose(pooled_coef[1], -amplitude, rtol=8.0e-15, atol=0.0)

    panel_X_np = np.ones((3, 1), dtype=np.float64)
    panel_X, panel_y, _unused, device = _backend_arrays(backend, panel_X_np, y_np)
    panel = PanelOLS(device=device, cov_type="hc0").fit(panel_X, panel_y)
    _assert_backend(panel, backend, "PanelOLS constant cancellation")
    panel_coef = np.asarray(_to_numpy(panel.coef_), dtype=np.float64).ravel()
    np.testing.assert_allclose(panel_coef[0], 1.0 / 3.0, rtol=8.0e-15, atol=0.0)

    X_level_np = np.repeat(X_np, 2, axis=0)
    y_level_np = np.repeat(y_np, 2)
    entity_np = np.repeat(np.arange(3, dtype=np.int64), 2)
    X_level, y_level, entity, device = _backend_arrays(
        backend, X_level_np, y_level_np, entity_np
    )
    between = BetweenOLS(device=device, cov_type="hc0").fit(
        X_level, y_level, entity_ids=entity
    )
    _assert_backend(between, backend, "BetweenOLS cancellation")
    between_coef = np.asarray(_to_numpy(between.coef_), dtype=np.float64).ravel()
    np.testing.assert_allclose(between_coef, pooled_coef, rtol=8.0e-15, atol=0.0)

    X_common_np, y_common_np, expected_slope = _common_level_fixture()
    X_common, y_common, _unused, device = _backend_arrays(
        backend, X_common_np, y_common_np
    )
    pooled_common = PooledOLS(device=device, cov_type="hc0").fit(X_common, y_common)
    _assert_backend(pooled_common, backend, "PooledOLS common level")
    np.testing.assert_allclose(
        np.asarray(_to_numpy(pooled_common.coef_))[1],
        expected_slope,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )

    X_common_level_np = np.repeat(X_common_np, 2, axis=0)
    y_common_level_np = np.repeat(y_common_np, 2)
    common_entity_np = np.repeat(np.arange(X_common_np.shape[0], dtype=np.int64), 2)
    X_common_level, y_common_level, common_entity, device = _backend_arrays(
        backend, X_common_level_np, y_common_level_np, common_entity_np
    )
    between_common = BetweenOLS(device=device, cov_type="hc0").fit(
        X_common_level, y_common_level, entity_ids=common_entity
    )
    _assert_backend(between_common, backend, "BetweenOLS common level")
    np.testing.assert_allclose(
        np.asarray(_to_numpy(between_common.coef_))[1],
        expected_slope,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )

    X_bad_np, y_bad_np = _mixed_coefficient_resolution_fixture()
    X_bad, y_bad, _unused, device = _backend_arrays(backend, X_bad_np, y_bad_np)
    panel_resolution_fail = _assert_resolution_failure(
        lambda: PanelOLS(device=device, cov_type="hc0").fit(X_bad, y_bad),
        label=f"{backend} PanelOLS",
    )
    pooled_resolution_fail = _assert_resolution_failure(
        lambda: PooledOLS(device=device, cov_type="hc0").fit(X_bad, y_bad),
        label=f"{backend} PooledOLS",
    )

    X_fd_bad_np, y_fd_bad_np, fd_bad_entity_np, fd_bad_time_np = (
        _first_difference_from_transformed(X_bad_np, y_bad_np)
    )
    X_fd_bad, y_fd_bad, fd_bad_entity, device = _backend_arrays(
        backend, X_fd_bad_np, y_fd_bad_np, fd_bad_entity_np
    )
    _, _, fd_bad_time, _ = _backend_arrays(
        backend, X_fd_bad_np, y_fd_bad_np, fd_bad_time_np
    )
    fd_resolution_fail = _assert_resolution_failure(
        lambda: FirstDifferenceOLS(device=device, cov_type="hc0").fit(
            X_fd_bad,
            y_fd_bad,
            entity_ids=fd_bad_entity,
            time_ids=fd_bad_time,
        ),
        label=f"{backend} FirstDifferenceOLS",
    )

    X_fmb_np, y_fmb_np, time_fmb_np = _fama_macbeth_resolution_fixture()
    X_fmb, y_fmb, time_fmb, device = _backend_arrays(
        backend, X_fmb_np, y_fmb_np, time_fmb_np
    )
    fmb_resolution_fail = _assert_resolution_failure(
        lambda: FamaMacBeth(device=device, bandwidth=0).fit(
            X_fmb, y_fmb, time_ids=time_fmb
        ),
        label=f"{backend} FamaMacBeth",
    )

    X_fmb_big_np, y_fmb_big_np, time_fmb_big_np = _large_common_intercept_fmb_fixture()
    X_fmb_big, y_fmb_big, time_fmb_big, device = _backend_arrays(
        backend, X_fmb_big_np, y_fmb_big_np, time_fmb_big_np
    )
    fmb_big = FamaMacBeth(device=device, bandwidth=0).fit(
        X_fmb_big, y_fmb_big, time_ids=time_fmb_big
    )
    _assert_backend(fmb_big, backend, "FamaMacBeth common intercept")
    if fmb_big._period_solver_mode != "gram-certified" or fmb_big._period_svd_fallbacks != 0:
        raise AssertionError("large common FamaMacBeth intercept did not stay Gram-certified")

    X_fd_np, y_fd_np, fd_entity_np, fd_time_np = _first_difference_extreme_r2_fixture()
    X_fd, y_fd, fd_entity, device = _backend_arrays(backend, X_fd_np, y_fd_np, fd_entity_np)
    _, _, fd_time, _ = _backend_arrays(backend, X_fd_np, y_fd_np, fd_time_np)
    fd = FirstDifferenceOLS(device=device, cov_type="hc0").fit(
        X_fd, y_fd, entity_ids=fd_entity, time_ids=fd_time
    )
    _assert_backend(fd, backend, "FirstDifferenceOLS extreme R2")
    if fd.rsquared != 1.0:
        raise AssertionError(f"{backend}: FirstDifferenceOLS extreme R2={fd.rsquared!r}")
    np.testing.assert_allclose(
        np.asarray(_to_numpy(fd.coef_))[0], 1.0e308, rtol=5.0e-15, atol=0.0
    )

    quasi_fail = _assert_random_effects_fail_closed(
        backend,
        _random_effects_component_loss_fixture(),
        "quasi-demeaning exceeds the float64 component range",
        "lost quasi-demean component",
    )
    normalization_fail = _assert_random_effects_fail_closed(
        backend,
        _random_effects_common_scale_loss_fixture(amplitude=1.0e308, tiny_within=1.0e-100),
        "variance-component scaling exceeds the float64 common-residual range",
        "common normalization loss",
    )
    square_fail = _assert_random_effects_fail_closed(
        backend,
        _random_effects_common_scale_loss_fixture(amplitude=1.0e100, tiny_within=1.0e-100),
        "variance-component scaling exceeds the float64 common-residual range",
        "common RSS square underflow",
    )
    theta_rounding_fail = _assert_random_effects_fail_closed(
        backend,
        _random_effects_common_scale_loss_fixture(amplitude=1.0e17, tiny_within=1.0),
        "quasi-demeaning exceeds the float64 component range",
        "pre-rounded theta-complement loss",
    )

    return {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "validation_tier": validation_tier,
        "clean_worktree": True,
        "requested_backend": backend,
        "executed_backends": {
            "pooled": getattr(pooled, "_backend_name", None),
            "panel": getattr(panel, "_backend_name", None),
            "between": getattr(between, "_backend_name", None),
            "fama_macbeth": getattr(fmb_big, "_backend_name", None),
            "first_difference": getattr(fd, "_backend_name", None),
        },
        "gpu": _gpu_name(backend),
        "packages": {
            "statgpu": _version("statgpu"),
            "numpy": _version("numpy"),
            "cupy": _version("cupy-cuda11x") or _version("cupy-cuda12x") or _version("cupy"),
            "torch": _version("torch"),
        },
        "pooled_coef": pooled_coef.tolist(),
        "panel_constant_coef": panel_coef.tolist(),
        "between_coef": between_coef.tolist(),
        "common_level_expected_slope": expected_slope,
        "panel_resolution_fail_closed": panel_resolution_fail,
        "pooled_resolution_fail_closed": pooled_resolution_fail,
        "first_difference_resolution_fail_closed": fd_resolution_fail,
        "fama_macbeth_resolution_fail_closed": fmb_resolution_fail,
        "fama_macbeth_common_intercept_solver_mode": fmb_big._period_solver_mode,
        "fama_macbeth_common_intercept_svd_fallbacks": fmb_big._period_svd_fallbacks,
        "first_difference_extreme_r2": fd.rsquared,
        "random_effects_quasi_component_fail_closed": quasi_fail,
        "random_effects_common_normalization_fail_closed": normalization_fail,
        "random_effects_common_square_fail_closed": square_fail,
        "random_effects_theta_rounding_fail_closed": theta_rounding_fail,
        "status": "success",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=("cupy", "torch"))
    parser.add_argument(
        "--validation-tier",
        required=True,
        choices=("local-minimal", "local-full", "remote-full"),
        help=(
            "evidence tier supplied by the runner orchestrator; the script never "
            "infers remote execution so local runs cannot silently claim remote-full"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.backend, args.validation_tier)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
