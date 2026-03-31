"""
Test RidgeFullGPU: full GPU computation.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '/root/.openclaw/workspace-coding/statgpu')

print("=" * 70)
print("RidgeFullGPU Test - Full GPU Computation")
print("=" * 70)

from statgpu.linear_model._ridge_full_gpu import RidgeFullGPU
from statgpu._config import set_device, cuda_available
import cupy as cp

# Check GPU
if not cuda_available():
    print("No GPU available")
    exit()

print("✓ GPU available\n")

# Test sizes
sizes = [
    (1000, 50),
    (5000, 100),
    (10000, 100),
    (50000, 200),
]

for n_samples, n_features in sizes:
    print(f"\n{'=' * 70}")
    print(f"Dataset: {n_samples:,} × {n_features}")
    print(f"{'=' * 70}")
    
    # Generate data on CPU
    np.random.seed(42)
    X_cpu = np.random.randn(n_samples, n_features).astype(np.float32)
    y_cpu = X_cpu @ np.random.randn(n_features).astype(np.float32)
    
    # ========================================
    # Full GPU Pipeline
    # ========================================
    set_device('cuda')
    
    # Transfer to GPU (once)
    t0 = time.perf_counter()
    X_gpu = cp.asarray(X_cpu)
    y_gpu = cp.asarray(y_cpu)
    cp.cuda.Device().synchronize()
    transfer_time = (time.perf_counter() - t0) * 1000
    
    # Fit on GPU
    model = RidgeFullGPU(alpha=1.0, device='cuda')
    t0 = time.perf_counter()
    model.fit(X_gpu, y_gpu)
    cp.cuda.Device().synchronize()
    fit_time = (time.perf_counter() - t0) * 1000
    
    # Predict on GPU
    t0 = time.perf_counter()
    y_pred_gpu = model.predict(X_gpu)
    cp.cuda.Device().synchronize()
    predict_time = (time.perf_counter() - t0) * 1000
    
    # Score on GPU
    t0 = time.perf_counter()
    r2 = model.score(X_gpu, y_gpu)
    cp.cuda.Device().synchronize()
    score_time = (time.perf_counter() - t0) * 1000
    
    total_gpu = transfer_time + fit_time + predict_time + score_time
    
    print(f"GPU Transfer:  {transfer_time:8.2f} ms")
    print(f"GPU Fit:       {fit_time:8.2f} ms")
    print(f"GPU Predict:   {predict_time:8.2f} ms")
    print(f"GPU Score:     {score_time:8.2f} ms")
    print(f"GPU Total:     {total_gpu:8.2f} ms")
    print(f"R² = {r2:.6f}")
    
    # ========================================
    # CPU Comparison
    # ========================================
    set_device('cpu')
    from statgpu.linear_model import Ridge
    
    model_cpu = Ridge(alpha=1.0, device='cpu')
    t0 = time.perf_counter()
    model_cpu.fit(X_cpu, y_cpu)
    y_pred_cpu = model_cpu.predict(X_cpu)
    r2_cpu = model_cpu.score(X_cpu, y_cpu)
    cpu_time = (time.perf_counter() - t0) * 1000
    
    print(f"CPU Total:     {cpu_time:8.2f} ms")
    print(f"Speedup:       {cpu_time/total_gpu:8.2f}x")

print("\n" + "=" * 70)
print("✓ Full GPU pipeline complete!")
print("=" * 70)
print("\nNote: Data stays on GPU for multiple predictions")
print("      Only initial transfer and final results need CPU")
