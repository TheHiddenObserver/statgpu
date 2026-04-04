"""
Benchmark script for Ridge and Lasso regression.
Shows GPU speedup vs CPU.
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from statgpu.linear_model import Ridge, Lasso


def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features) * 2
    y = X @ true_coef + noise * np.random.randn(n_samples)
    return X, y, true_coef


def benchmark_ridge():
    """Benchmark Ridge regression."""
    print("\n" + "="*80)
    print("RIDGE REGRESSION BENCHMARK")
    print("="*80)

    sizes = [
        (1000, 50),
        (5000, 100),
        (10000, 200),
        (50000, 500),
        (100000, 1000),
    ]

    print(f"\n{'Size':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for n_samples, n_features in sizes:
        X, y, _ = generate_data(n_samples, n_features)

        # CPU benchmark
        ridge_cpu = Ridge(alpha=1.0, device='cpu')
        t0 = time.perf_counter()
        ridge_cpu.fit(X, y)
        cpu_time = (time.perf_counter() - t0) * 1000

        # GPU benchmark (if available)
        try:
            ridge_gpu = Ridge(alpha=1.0, device='cuda')
            t0 = time.perf_counter()
            ridge_gpu.fit(X, y)
            gpu_time = (time.perf_counter() - t0) * 1000
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"{n_samples}x{n_features:<8} {cpu_time:<15.2f} {gpu_time:<15.2f} {speedup:<10.2f}x")
        except Exception as e:
            print(f"{n_samples}x{n_features:<8} {cpu_time:<15.2f} {'N/A':<15} {'N/A':<10}")


def benchmark_lasso():
    """Benchmark Lasso regression."""
    print("\n" + "="*80)
    print("LASSO REGRESSION BENCHMARK (Coordinate Descent)")
    print("="*80)

    sizes = [
        (1000, 50),
        (5000, 100),
        (10000, 200),
    ]

    print(f"\n{'Size':<15} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10} {'Iters':<8}")
    print("-" * 70)

    for n_samples, n_features in sizes:
        X, y, _ = generate_data(n_samples, n_features)

        # CPU benchmark
        lasso_cpu = Lasso(alpha=0.1, max_iter=500, device='cpu')
        t0 = time.perf_counter()
        lasso_cpu.fit(X, y)
        cpu_time = (time.perf_counter() - t0) * 1000
        iters = lasso_cpu.n_iter_

        # GPU benchmark (if available)
        try:
            lasso_gpu = Lasso(alpha=0.1, max_iter=500, device='cuda')
            t0 = time.perf_counter()
            lasso_gpu.fit(X, y)
            gpu_time = (time.perf_counter() - t0) * 1000
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"{n_samples}x{n_features:<8} {cpu_time:<15.2f} {gpu_time:<15.2f} {speedup:<10.2f}x {iters:<8}")
        except Exception as e:
            print(f"{n_samples}x{n_features:<8} {cpu_time:<15.2f} {'N/A':<15} {'N/A':<10} {iters:<8}")


def benchmark_vs_sklearn():
    """Benchmark against sklearn."""
    print("\n" + "="*80)
    print("BENCHMARK VS SKLEARN")
    print("="*80)

    try:
        from sklearn.linear_model import Ridge as SklearnRidge
        from sklearn.linear_model import Lasso as SklearnLasso
    except ImportError:
        print("sklearn not available")
        return

    n_samples, n_features = 10000, 500
    X, y, _ = generate_data(n_samples, n_features)

    print(f"\nDataset: {n_samples} samples x {n_features} features")
    print(f"\n{'Model':<20} {'Library':<15} {'Time (ms)':<15} {'Relative':<10}")
    print("-" * 65)

    # statgpu Ridge
    ridge_sg = Ridge(alpha=1.0, device='cpu')
    t0 = time.perf_counter()
    ridge_sg.fit(X, y)
    sg_time = (time.perf_counter() - t0) * 1000
    print(f"{'Ridge':<20} {'statgpu':<15} {sg_time:<15.2f} {'1.00x':<10}")

    # sklearn Ridge
    ridge_sk = SklearnRidge(alpha=1.0, fit_intercept=True)
    t0 = time.perf_counter()
    ridge_sk.fit(X, y)
    sk_time = (time.perf_counter() - t0) * 1000
    relative = sk_time / sg_time
    print(f"{'Ridge':<20} {'sklearn':<15} {sk_time:<15.2f} {relative:<10.2f}x")

    # statgpu Lasso
    lasso_sg = Lasso(alpha=0.1, max_iter=1000, device='cpu')
    t0 = time.perf_counter()
    lasso_sg.fit(X, y)
    sg_time = (time.perf_counter() - t0) * 1000
    print(f"{'Lasso':<20} {'statgpu':<15} {sg_time:<15.2f} {'1.00x':<10}")

    # sklearn Lasso
    lasso_sk = SklearnLasso(alpha=0.1, fit_intercept=True, max_iter=1000)
    t0 = time.perf_counter()
    lasso_sk.fit(X, y)
    sk_time = (time.perf_counter() - t0) * 1000
    relative = sk_time / sg_time
    print(f"{'Lasso':<20} {'sklearn':<15} {sk_time:<15.2f} {relative:<10.2f}x")


if __name__ == "__main__":
    print("StatGPU Ridge & Lasso Benchmark")
    print(f"NumPy version: {np.__version__}")

    benchmark_ridge()
    benchmark_lasso()
    benchmark_vs_sklearn()

    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)
