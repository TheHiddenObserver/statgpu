"""
Optimized GPU benchmark with warmup and various data sizes.
"""

import numpy as np
import time
from statgpu.linear_model import LinearRegression
from statgpu._config import set_device, cuda_available
import cupy as cp

print("=" * 70)
print("GPU Performance Benchmark")
print("=" * 70)

if not cuda_available():
    print("CUDA not available!")
    exit(1)

# Warmup: initialize CUDA context
print("\n[Warmup] Initializing CUDA context...")
set_device('cuda')
X_warmup = cp.random.randn(100, 10).astype(cp.float32)
y_warmup = X_warmup @ cp.random.randn(10).astype(cp.float32)
model = LinearRegression(device='cuda')
model.fit(X_warmup, y_warmup)
print("[Warmup] Done.\n")

# Test configurations
test_cases = [
    (1000, 10, "Small"),
    (10000, 50, "Medium"),
    (100000, 100, "Large"),
    (500000, 200, "XLarge"),
    (1000000, 500, "XXLarge"),
]

print(f"{'Size':<12} {'Samples':>10} {'Features':>10} {'CPU(ms)':>12} {'GPU(ms)':>12} {'Speedup':>10}")
print("-" * 70)

for n_samples, n_features, label in test_cases:
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = X_np @ np.random.randn(n_features).astype(np.float32)
    
    # CPU benchmark
    set_device('cpu')
    model_cpu = LinearRegression(device='cpu')
    start = time.time()
    model_cpu.fit(X_np, y_np)
    cpu_time = (time.time() - start) * 1000
    
    # GPU benchmark
    set_device('cuda')
    model_gpu = LinearRegression(device='cuda')
    start = time.time()
    model_gpu.fit(X_np, y_np)
    gpu_time = (time.time() - start) * 1000
    
    speedup = cpu_time / gpu_time
    print(f"{label:<12} {n_samples:>10} {n_features:>10} {cpu_time:>12.2f} {gpu_time:>12.2f} {speedup:>10.2f}x")
    
    # Verify correctness
    if not np.allclose(model_cpu.coef_, model_gpu.coef_, rtol=1e-3):
        print(f"  ⚠ Warning: Results differ!")

print("\n" + "=" * 70)

# Memory pool test
print("\n[Memory Pool Test]")
print("Testing repeated fits with memory pool...")

set_device('cuda')
n_samples, n_features = 100000, 100

# Use CuPy memory pool
mem_pool = cp.get_default_memory_pool()

np.random.seed(42)
X_np = np.random.randn(n_samples, n_features).astype(np.float32)
y_np = X_np @ np.random.randn(n_features).astype(np.float32)

model = LinearRegression(device='cuda')

times = []
for i in range(10):
    start = time.time()
    model.fit(X_np, y_np)
    times.append((time.time() - start) * 1000)

print(f"  First fit:  {times[0]:.2f}ms")
print(f"  Mean(2-10): {np.mean(times[1:]):.2f}ms")
print(f"  Std:        {np.std(times[1:]):.2f}ms")

# Synchronize to get accurate timing
cp.cuda.Device().synchronize()
print("\n[Done]")
