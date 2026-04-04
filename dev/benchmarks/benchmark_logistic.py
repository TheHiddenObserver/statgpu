"""
Benchmark script for LogisticRegression GPU vs CPU.
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from statgpu.linear_model import LogisticRegression
from statgpu._config import set_device, cuda_available

print("=" * 80)
print("Logistic Regression GPU Benchmark")
print("=" * 80)

# Check CUDA availability
has_cuda = cuda_available()
print(f"\nCUDA available: {has_cuda}")

# Benchmark configurations
configs = [
    (1000, 10, "Small"),
    (5000, 20, "Medium"),
    (10000, 50, "Large"),
    (50000, 100, "Very Large"),
]

if has_cuda:
    configs.append((100000, 200, "Huge"))

results = []

for n_samples, n_features, label in configs:
    print(f"\n{'=' * 80}")
    print(f"Dataset: {label} ({n_samples:,} samples, {n_features} features)")
    print(f"{'=' * 80}")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    true_coef = np.random.randn(n_features)
    z = X @ true_coef + np.random.randn(n_samples) * 0.1
    y = (z > 0).astype(int)
    
    print(f"Class balance: {np.mean(y):.2%} positive")
    
    # CPU benchmark
    set_device('cpu')
    
    # Warmup
    model_cpu = LogisticRegression(device='cpu', max_iter=50)
    model_cpu.fit(X[:100], y[:100])
    
    # Actual benchmark
    times_cpu = []
    for _ in range(3):
        start = time.time()
        model_cpu = LogisticRegression(device='cpu', max_iter=50)
        model_cpu.fit(X, y)
        times_cpu.append(time.time() - start)
    
    time_cpu = np.median(times_cpu)
    print(f"\nCPU time: {time_cpu:.4f}s (median of 3 runs)")
    
    # GPU benchmark
    if has_cuda:
        set_device('cuda')
        
        # Warmup
        model_gpu = LogisticRegression(device='cuda', max_iter=50)
        model_gpu.fit(X[:100], y[:100])
        
        # Actual benchmark
        times_gpu = []
        for _ in range(3):
            start = time.time()
            model_gpu = LogisticRegression(device='cuda', max_iter=50)
            model_gpu.fit(X, y)
            times_gpu.append(time.time() - start)
        
        time_gpu = np.median(times_gpu)
        speedup = time_cpu / time_gpu
        
        print(f"GPU time: {time_gpu:.4f}s (median of 3 runs)")
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results match
        coef_diff = np.max(np.abs(model_cpu.coef_ - model_gpu.coef_))
        intercept_diff = abs(model_cpu.intercept_ - model_gpu.intercept_)
        print(f"\nResult verification:")
        print(f"  Max coefficient diff: {coef_diff:.2e}")
        print(f"  Intercept diff: {intercept_diff:.2e}")
        
        results.append({
            'label': label,
            'n_samples': n_samples,
            'n_features': n_features,
            'time_cpu': time_cpu,
            'time_gpu': time_gpu,
            'speedup': speedup,
            'coef_diff': coef_diff
        })
    else:
        results.append({
            'label': label,
            'n_samples': n_samples,
            'n_features': n_features,
            'time_cpu': time_cpu,
            'time_gpu': None,
            'speedup': None,
            'coef_diff': None
        })

# Summary table
print(f"\n{'=' * 80}")
print("Summary")
print(f"{'=' * 80}")
print(f"\n{'Dataset':<15} {'Samples':>10} {'Features':>10} {'CPU (s)':>12} {'GPU (s)':>12} {'Speedup':>10}")
print("-" * 80)

for r in results:
    gpu_time = f"{r['time_gpu']:.4f}" if r['time_gpu'] else "N/A"
    speedup = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
    print(f"{r['label']:<15} {r['n_samples']:>10,} {r['n_features']:>10} {r['time_cpu']:>12.4f} {gpu_time:>12} {speedup:>10}")

# Performance scaling analysis
if has_cuda and len(results) > 1:
    print(f"\n{'=' * 80}")
    print("Performance Scaling Analysis")
    print(f"{'=' * 80}")
    
    print("\nObservations:")
    
    # Find best speedup
    best = max(results, key=lambda x: x['speedup'] or 0)
    print(f"  - Best speedup: {best['speedup']:.2f}x on {best['label']} dataset")
    
    # Check if speedup increases with size
    speedups = [r['speedup'] for r in results if r['speedup']]
    if len(speedups) >= 2 and speedups[-1] > speedups[0]:
        print(f"  - GPU scaling: Speedup increases with dataset size ({speedups[0]:.2f}x to {speedups[-1]:.2f}x)")
    
    # Check numerical accuracy
    max_diff = max(r['coef_diff'] for r in results if r['coef_diff'])
    print(f"  - Numerical accuracy: Max coefficient difference = {max_diff:.2e}")
    if max_diff < 1e-3:
        print(f"    ✓ Results are numerically accurate")
    else:
        print(f"    ⚠ Results differ more than expected")

print(f"\n{'=' * 80}")
print("Benchmark Complete")
print(f"{'=' * 80}")
