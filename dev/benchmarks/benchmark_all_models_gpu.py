"""
Benchmark all five models with FULL GPU computation.
"""

import numpy as np
import time
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print("=" * 80)
print("All Five Models - Full GPU Benchmark")
print("=" * 80)

from statgpu._config import set_device, cuda_available
import cupy as cp

if not cuda_available():
    print("No GPU available")
    exit()

print("✓ GPU available\n")

# Import models
from statgpu.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from statgpu.survival import CoxPH

# Test configuration
np.random.seed(42)
sizes = [
    (10000, 100),
    (50000, 200),
    (100000, 500),
]

models_config = [
    ('LinearRegression', LinearRegression, {'device': 'cuda'}, None),
    ('Ridge', Ridge, {'alpha': 1.0, 'device': 'cuda'}, None),
    ('Lasso', Lasso, {'alpha': 0.1, 'max_iter': 500, 'device': 'cuda'}, None),
]

print(f"{'Model':<20} {'Size':<15} {'CPU(ms)':<12} {'GPU(ms)':<12} {'Speedup':<10}")
print("=" * 80)

for model_name, ModelClass, kwargs, extra in models_config:
    for n_samples, n_features in sizes:
        # Generate data
        X_cpu = np.random.randn(n_samples, n_features).astype(np.float32)
        y_cpu = X_cpu @ np.random.randn(n_features).astype(np.float32)
        
        # CPU benchmark
        set_device('cpu')
        model = ModelClass(**{**kwargs, 'device': 'cpu'})
        t0 = time.perf_counter()
        model.fit(X_cpu, y_cpu)
        cpu_time = (time.perf_counter() - t0) * 1000
        
        # GPU benchmark
        set_device('cuda')
        X_gpu = cp.asarray(X_cpu)
        y_gpu = cp.asarray(y_cpu)
        cp.cuda.Device().synchronize()
        
        model = ModelClass(**kwargs)
        t0 = time.perf_counter()
        model.fit(X_gpu, y_gpu)
        cp.cuda.Device().synchronize()
        gpu_time = (time.perf_counter() - t0) * 1000
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        size_str = f"{n_samples//1000}K×{n_features}"
        print(f"{model_name:<20} {size_str:<15} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f}x")

print("\n" + "=" * 80)
print("✓ Benchmark complete!")
print("=" * 80)
