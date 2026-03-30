"""
Test n_jobs parallel performance for CPU.
"""

import numpy as np
import time
from statgpu.linear_model import LinearRegression
from statgpu._config import set_device
import joblib

print("=" * 60)
print("Testing n_jobs parallel performance (CPU)")
print("=" * 60)

# Generate data
np.random.seed(42)
n_samples, n_features = 100000, 100
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = X @ np.random.randn(n_features).astype(np.float32)

print(f"\nDataset: {n_samples:,} samples x {n_features} features")
print(f"Data size: {X.nbytes / 1e6:.1f} MB\n")

# Test different n_jobs
n_jobs_list = [1, 2, 4, -1]

print(f"{'n_jobs':>10} {'Time(ms)':>12} {'Speedup':>10}")
print("-" * 40)

set_device('cpu')
times = {}

for n_jobs in n_jobs_list:
    model = LinearRegression(device='cpu', n_jobs=n_jobs)
    
    # Warmup
    model.fit(X[:100], y[:100])
    
    # Benchmark
    start = time.time()
    model.fit(X, y)
    elapsed = (time.time() - start) * 1000
    times[n_jobs] = elapsed
    
    speedup = times[1] / elapsed if n_jobs != 1 else 1.0
    print(f"{n_jobs:>10} {elapsed:>12.2f} {speedup:>10.2f}x")

print("\n" + "=" * 60)
print(f"CPU cores available: {joblib.cpu_count()}")
print("=" * 60)
