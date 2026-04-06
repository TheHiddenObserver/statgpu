"""Optional performance guard for large-scale inference resampling.

This test is skipped by default. Enable it in CI by setting:
    STATGPU_RUN_PERF_GUARD=1

Recommended to run on stable, dedicated GPU runners to reduce noise.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from statgpu.inference import bootstrap_statistic, permutation_test


def _have_cupy():
    try:
        import cupy as cp

        return cp
    except Exception:
        return None


def _corr_stat_maybe_vectorized(X_in, y_in, xp_module):
    x = X_in[:, 0]
    y_arr = xp_module.asarray(y_in)

    if y_arr.ndim == 1:
        return xp_module.corrcoef(x, y_arr)[0, 1]

    x_centered = x - xp_module.mean(x)
    y_centered = y_arr - xp_module.mean(y_arr, axis=1, keepdims=True)
    x_norm_sq = xp_module.sum(x_centered * x_centered)
    y_norm_sq = xp_module.sum(y_centered * y_centered, axis=1)
    denom = xp_module.sqrt(x_norm_sq * y_norm_sq)
    denom = xp_module.where(denom > 0.0, denom, xp_module.inf)
    numer = xp_module.sum(y_centered * x_centered.reshape(1, -1), axis=1)
    return numer / denom


def _time_once(fn, sync=None):
    t0 = time.perf_counter()
    fn()
    if sync is not None:
        sync()
    return (time.perf_counter() - t0) * 1000.0


def _warmup_once(fn, sync=None):
    fn()
    if sync is not None:
        sync()


@pytest.mark.skipif(os.getenv("STATGPU_RUN_PERF_GUARD", "0") != "1", reason="Performance guard disabled")
def test_large_scale_gpu_vectorized_faster_than_cpu_vectorized():
    cp = _have_cupy()
    if cp is None:
        pytest.skip("CuPy/CUDA unavailable")

    rng = np.random.default_rng(20260405)

    # Match large-scale benchmark regime to reduce noise and JIT sensitivity.
    x_np = rng.normal(loc=1.0, scale=1.0, size=30_000)
    X_np = rng.normal(size=(18_000, 1))
    y_np = 1.7 * X_np[:, 0] + rng.normal(scale=0.6, size=18_000)

    x_cp = cp.asarray(x_np)
    X_cp = cp.asarray(X_np)
    y_cp = cp.asarray(y_np)

    bootstrap_cpu_vec_fn = lambda: bootstrap_statistic(
            lambda arr: np.mean(arr, axis=-1),
            x_np,
            n_resamples=2500,
            strategy="iid",
            random_state=17,
            backend="numpy",
            force_vectorized=True,
        )
    bootstrap_gpu_vec_fn = lambda: bootstrap_statistic(
            lambda arr: cp.mean(arr, axis=-1),
            x_cp,
            n_resamples=2500,
            strategy="iid",
            random_state=17,
            backend="cupy",
            force_vectorized=True,
            statistic_hint="mean",
        )

    _warmup_once(bootstrap_cpu_vec_fn)
    _warmup_once(bootstrap_gpu_vec_fn, sync=cp.cuda.runtime.deviceSynchronize)
    bootstrap_cpu_vec = _time_once(bootstrap_cpu_vec_fn)
    bootstrap_gpu_vec = _time_once(bootstrap_gpu_vec_fn, sync=cp.cuda.runtime.deviceSynchronize)

    permutation_cpu_vec_fn = lambda: permutation_test(
            lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, np),
            X_np,
            y_np,
            n_resamples=2000,
            strategy="iid",
            alternative="two-sided",
            random_state=23,
            backend="numpy",
            force_vectorized=True,
        )
    permutation_gpu_vec_fn = lambda: permutation_test(
            lambda X_in, y_in: _corr_stat_maybe_vectorized(X_in, y_in, cp),
            X_cp,
            y_cp,
            n_resamples=2000,
            strategy="iid",
            alternative="two-sided",
            random_state=23,
            backend="cupy",
            force_vectorized=True,
            statistic_hint="pearson_corr",
        )

    _warmup_once(permutation_cpu_vec_fn)
    _warmup_once(permutation_gpu_vec_fn, sync=cp.cuda.runtime.deviceSynchronize)
    permutation_cpu_vec = _time_once(permutation_cpu_vec_fn)
    permutation_gpu_vec = _time_once(permutation_gpu_vec_fn, sync=cp.cuda.runtime.deviceSynchronize)

    # Guard goals: GPU vectorized path should beat CPU vectorized path.
    assert bootstrap_gpu_vec < bootstrap_cpu_vec
    assert permutation_gpu_vec < permutation_cpu_vec
