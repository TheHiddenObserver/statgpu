"""
Run statgpu on saved benchmark data and compare with glmnet.
Generates a full comparison table with detailed timing.
"""
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from statgpu.linear_model import ElasticNet


def run_statgpu_benchmark(name, n, p, results_dir=Path("/root")):
    """Run statgpu on saved data"""

    # Load data (skip header row from R write.csv)
    X = np.loadtxt(results_dir / f"benchmark_X_{name}.csv", delimiter=",", skiprows=1)
    y = np.loadtxt(results_dir / f"benchmark_y_{name}.csv", delimiter=",", skiprows=1)

    results = {
        'name': name,
        'n_samples': n,
        'n_features': p,
        'backends': {}
    }

    # statgpu CPU
    print(f"\n[statgpu CPU] {name} (n={n}, p={p})")
    enet_cpu = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-8, device='cpu')

    start = time.perf_counter()
    enet_cpu.fit(X, y)
    time_cpu = time.perf_counter() - start

    results['backends']['statgpu_cpu'] = {
        'coef': enet_cpu.coef_.tolist(),
        'intercept': float(enet_cpu.intercept_),
        'coef_norm': float(np.linalg.norm(enet_cpu.coef_)),
        'fit_time_ms': float(time_cpu * 1000),
        'n_iter': int(enet_cpu.n_iter_)
    }
    print(f"  fit_time: {time_cpu*1000:.2f} ms")

    # statgpu GPU (CuPy)
    print(f"\n[statgpu GPU (CuPy)] {name}")
    try:
        import cupy as cp

        enet_gpu = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-8, device='cuda')

        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        enet_gpu.fit(X, y)
        cp.cuda.Stream.null.synchronize()
        time_gpu = time.perf_counter() - start

        results['backends']['statgpu_gpu_cupy'] = {
            'coef': enet_gpu.coef_.tolist(),
            'intercept': float(enet_gpu.intercept_),
            'coef_norm': float(np.linalg.norm(enet_gpu.coef_)),
            'fit_time_ms': float(time_gpu * 1000),
            'n_iter': int(enet_gpu.n_iter_)
        }
        print(f"  fit_time: {time_gpu*1000:.2f} ms")
    except Exception as e:
        print(f"  Skipped: {e}")
        results['backends']['statgpu_gpu_cupy'] = {'error': str(e)}

    # statgpu GPU (Torch)
    print(f"\n[statgpu GPU (Torch)] {name}")
    try:
        import torch

        if torch.cuda.is_available():
            enet_torch = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-8, device='cuda')

            start = time.perf_counter()
            enet_torch.fit(X, y)
            time_torch = time.perf_counter() - start

            results['backends']['statgpu_gpu_torch'] = {
                'coef': enet_torch.coef_.tolist(),
                'intercept': float(enet_torch.intercept_),
                'coef_norm': float(np.linalg.norm(enet_torch.coef_)),
                'fit_time_ms': float(time_torch * 1000),
                'n_iter': int(enet_torch.n_iter_)
            }
            print(f"  fit_time: {time_torch*1000:.2f} ms")
        else:
            print(f"  Skipped: CUDA not available")
            results['backends']['statgpu_gpu_torch'] = {'error': 'CUDA not available'}
    except Exception as e:
        print(f"  Skipped: {e}")
        results['backends']['statgpu_gpu_torch'] = {'error': str(e)}

    return results


def main():
    print("=" * 70)
    print("Elastic Net Benchmark: statgpu vs glmnet")
    print("=" * 70)

    test_configs = [
        ("small_data", 200, 20),
        ("medium_data", 1000, 50),
        ("large_data", 5000, 100),
        ("high_dim_data", 100, 200),
        ("sparse_coef", 500, 100),
        ("high_noise", 500, 50)
    ]

    all_results = []
    for name, n, p in test_configs:
        result = run_statgpu_benchmark(name, n, p)
        all_results.append(result)

    # Save combined results
    combined = {
        'timestamp': datetime.now().isoformat(),
        'results': all_results
    }

    with open("/root/benchmark_statgpu_all.json", 'w') as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to: /root/benchmark_statgpu_all.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
