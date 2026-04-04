"""
GPU benchmark with warmup and pure computation timing.
Excludes CUDA initialization overhead.
"""

import numpy as np
import time
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from statgpu.linear_model import LinearRegression, Ridge
from statgpu._config import set_device, cuda_available

print("=" * 70)
print("GPU Benchmark with Warmup (Excluding CUDA Init)")
print("=" * 70)

# Check GPU
has_gpu = cuda_available()
print(f"GPU available: {has_gpu}\n")

if not has_gpu:
    print("No GPU available, exiting.")
    exit()

import cupy as cp

# Test sizes
sizes = [
    (1000, 50),
    (5000, 100),
    (10000, 100),
    (50000, 200),
]

# ============================================
# 1. Warmup CUDA
# ============================================
print("=" * 70)
print("WARMUP: Initializing CUDA context...")
print("=" * 70)

set_device('cuda')

# Create small arrays to warm up CUDA
X_warm = cp.random.randn(100, 10).astype(cp.float32)
y_warm = X_warm @ cp.random.randn(10).astype(cp.float32)

# Force CUDA initialization by doing a small computation
result = X_warm.T @ X_warm
result = cp.asnumpy(result)  # Sync to ensure completion

print("✓ CUDA warmup complete\n")

# ============================================
# 2. Benchmark with pure computation time
# ============================================
print("=" * 70)
print("PURE COMPUTATION TIME (excluding data transfer)")
print("=" * 70)

for n_samples, n_features in sizes:
    print(f"\nDataset: {n_samples:,} × {n_features}")
    print("-" * 70)
    
    # Generate data on CPU
    np.random.seed(42)
    X_cpu = np.random.randn(n_samples, n_features).astype(np.float32)
    y_cpu = X_cpu @ np.random.randn(n_features).astype(np.float32)
    
    # ========================================
    # CPU Time
    # ========================================
    set_device('cpu')
    model = Ridge(alpha=1.0, device='cpu')
    
    t0 = time.perf_counter()
    model.fit(X_cpu, y_cpu)
    cpu_time = (time.perf_counter() - t0) * 1000
    
    print(f"CPU Total:     {cpu_time:8.2f} ms")
    
    # ========================================
    # GPU Time (with detailed breakdown)
    # ========================================
    set_device('cuda')
    
    # Step 1: Data transfer CPU -> GPU
    t0 = time.perf_counter()
    X_gpu = cp.asarray(X_cpu)
    y_gpu = cp.asarray(y_cpu)
    cp.cuda.Device().synchronize()  # Wait for transfer
    transfer_to_gpu = (time.perf_counter() - t0) * 1000
    
    # Step 2: Pure computation on GPU
    model = Ridge(alpha=1.0, device='cuda')
    t0 = time.perf_counter()
    model.fit(X_gpu, y_gpu)  # This should use pre-transferred data
    cp.cuda.Device().synchronize()  # Wait for computation
    gpu_compute = (time.perf_counter() - t0) * 1000
    
    # Step 3: Transfer results back
    t0 = time.perf_counter()
    coef = model.coef_  # This triggers .get()
    cp.cuda.Device().synchronize()
    transfer_from_gpu = (time.perf_counter() - t0) * 1000
    
    gpu_total = transfer_to_gpu + gpu_compute + transfer_from_gpu
    
    print(f"GPU Transfer:  {transfer_to_gpu:8.2f} ms (CPU->GPU)")
    print(f"GPU Compute:   {gpu_compute:8.2f} ms (pure)")
    print(f"GPU Transfer:  {transfer_from_gpu:8.2f} ms (GPU->CPU)")
    print(f"GPU Total:     {gpu_total:8.2f} ms")
    
    if gpu_compute > 0:
        print(f"Speedup (compute only): {cpu_time/gpu_compute:.2f}x")
        print(f"Speedup (total):        {cpu_time/gpu_total:.2f}x")

# ============================================
# 3. Multiple runs (amortize overhead)
# ============================================
print("\n" + "=" * 70)
print("MULTIPLE RUNS (amortizing overhead)")
print("=" * 70)

n_samples, n_features = 10000, 100
np.random.seed(42)
X_cpu = np.random.randn(n_samples, n_features).astype(np.float32)
y_cpu = X_cpu @ np.random.randn(n_features).astype(np.float32)

# Transfer once
X_gpu = cp.asarray(X_cpu)
y_gpu = cp.asarray(y_cpu)

n_runs = 10

# CPU multiple runs
set_device('cpu')
model = Ridge(alpha=1.0, device='cpu')
t0 = time.perf_counter()
for _ in range(n_runs):
    model.fit(X_cpu, y_cpu)
cpu_multi = (time.perf_counter() - t0) * 1000 / n_runs

# GPU multiple runs
set_device('cuda')
model = Ridge(alpha=1.0, device='cuda')
t0 = time.perf_counter()
for _ in range(n_runs):
    model.fit(X_gpu, y_gpu)
    cp.cuda.Device().synchronize()
gpu_multi = (time.perf_counter() - t0) * 1000 / n_runs

print(f"\nDataset: {n_samples:,} × {n_features}, {n_runs} runs averaged")
print(f"CPU:  {cpu_multi:.2f} ms")
print(f"GPU:  {gpu_multi:.2f} ms")
print(f"Speedup: {cpu_multi/gpu_multi:.2f}x")

print("\n" + "=" * 70)
print("Benchmark complete!")
print("=" * 70)
