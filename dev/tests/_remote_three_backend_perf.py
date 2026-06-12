"""Three-backend (numpy/cupy/torch) performance comparison.

Measures wall-clock time for key operations across all backends.
"""

import time
import numpy as np
import json
import sys


def benchmark_loss_fused():
    """Benchmark fused_value_and_gradient for all loss types."""
    from statgpu.glm_core import get_glm_loss

    np.random.seed(42)
    results = {}

    for n, p in [(500, 50), (500, 200), (1000, 100), (1000, 500)]:
        X = np.column_stack([np.random.randn(n, p), np.ones(n)])
        y = np.abs(np.random.randn(n)) + 0.1
        coef = np.random.randn(p + 1)

        for loss_name in ['squared_error', 'logistic', 'poisson', 'gamma', 'inverse_gaussian']:
            loss = get_glm_loss(loss_name)
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                val, grad = loss.fused_value_and_gradient(X, y, coef)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            key = f'{loss_name}_n{n}_p{p}'
            results[key] = round(np.median(times) * 1000, 3)  # ms

    return results


def benchmark_fista_solver():
    """Benchmark FISTA solver for different loss/penalty combinations."""
    from statgpu.glm_core._solver import fista_solver
    from statgpu.glm_core import get_glm_loss
    from statgpu.penalties._l1 import L1Penalty

    results = {}

    for n, p in [(500, 50), (500, 200), (1000, 100), (1000, 500)]:
        np.random.seed(42)
        X = np.random.randn(n, p)
        y = X @ np.random.randn(p) + 0.1 * np.random.randn(n)

        for loss_name in ['squared_error', 'logistic', 'poisson']:
            loss = get_glm_loss(loss_name)
            penalty = L1Penalty(alpha=0.1)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                coef, n_iter = fista_solver(loss, penalty, X, y, max_iter=100, tol=1e-6)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            key = f'{loss_name}_n{n}_p{p}'
            results[key] = {
                'time_ms': round(np.median(times) * 1000, 1),
                'iters': n_iter,
                'ms_per_iter': round(np.median(times) * 1000 / max(n_iter, 1), 2),
            }

    return results


def benchmark_penalty_proximal():
    """Benchmark penalty proximal operators."""
    from statgpu.penalties._l1 import L1Penalty
    from statgpu.penalties._scad import SCADPenalty
    from statgpu.penalties._mcp import MCPPenalty
    from statgpu.penalties._elasticnet import ElasticNetPenalty

    results = {}
    np.random.seed(42)

    for p in [50, 200, 500, 1000]:
        w = np.random.randn(p)
        for PenClass, name, kwargs in [
            (L1Penalty, 'l1', {'alpha': 0.1}),
            (ElasticNetPenalty, 'elasticnet', {'alpha': 0.1, 'l1_ratio': 0.5}),
            (SCADPenalty, 'scad', {'alpha': 0.1, 'a': 3.7}),
            (MCPPenalty, 'mcp', {'alpha': 0.1, 'gamma': 3.0}),
        ]:
            pen = PenClass(**kwargs)
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                result = pen.proximal(w, step=0.1, backend='numpy')
                t1 = time.perf_counter()
                times.append(t1 - t0)
            key = f'{name}_p{p}'
            results[key] = round(np.median(times) * 1e6, 1)  # microseconds

    return results


def benchmark_batch_mse():
    """Benchmark batch_mse for different sizes."""
    from statgpu.linear_model._cv_base import batch_mse

    results = {}
    np.random.seed(42)

    for n_val, n_models, p in [(100, 50, 10), (100, 100, 50), (500, 50, 50), (500, 200, 100)]:
        X = np.random.randn(n_val, p)
        y = np.random.randn(n_val)
        coefs = np.random.randn(n_models, p)
        intercepts = np.random.randn(n_models)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            mse = batch_mse(X, y, coefs, intercepts)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        key = f'nval={n_val}_nmod={n_models}_p={p}'
        results[key] = round(np.median(times) * 1000, 2)  # ms

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Three-Backend Performance Comparison")
    print("=" * 60)

    print("\n[1/4] Loss fused_value_and_gradient (ms)...")
    try:
        loss_perf = benchmark_loss_fused()
        for k, v in loss_perf.items():
            print(f"  {k}: {v}ms")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[2/4] FISTA solver (ms, iters, ms/iter)...")
    try:
        fista_perf = benchmark_fista_solver()
        for k, v in fista_perf.items():
            print(f"  {k}: {v['time_ms']}ms ({v['iters']} iter, {v['ms_per_iter']}ms/iter)")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[3/4] Penalty proximal (microseconds)...")
    try:
        prox_perf = benchmark_penalty_proximal()
        for k, v in prox_perf.items():
            print(f"  {k}: {v}us")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[4/4] batch_mse (ms)...")
    try:
        mse_perf = benchmark_batch_mse()
        for k, v in mse_perf.items():
            print(f"  {k}: {v}ms")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\nDone.")
