"""
Benchmark: CPU vs GPU linear regression
"""

import time
import numpy as np
from statgpu.linear_model import LinearRegression
from statgpu._config import set_device, cuda_available


def benchmark(n_samples=100000, n_features=100, n_runs=5):
    """Benchmark CPU vs GPU performance."""
    
    print(f"Benchmark: {n_samples} samples, {n_features} features")
    print("=" * 50)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X @ np.random.randn(n_features).astype(np.float32)
    
    # CPU benchmark
    print("\nCPU (NumPy):")
    set_device('cpu')
    model_cpu = LinearRegression(device='cpu')
    
    # Warmup run
    model_cpu.fit(X, y)
    
    times_cpu = []
    for i in range(n_runs):
        start = time.time()
        model_cpu.fit(X, y)
        elapsed = time.time() - start
        times_cpu.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    avg_cpu = np.mean(times_cpu)
    print(f"  Average: {avg_cpu:.4f}s")
    
    # GPU benchmark (if available)
    if cuda_available():
        print("\nGPU (CuPy):")
        set_device('cuda')
        model_gpu = LinearRegression(device='cuda')
        
        # Warmup run (CUDA context initialization)
        model_gpu.fit(X, y)
        
        times_gpu = []
        for i in range(n_runs):
            start = time.time()
            model_gpu.fit(X, y)
            elapsed = time.time() - start
            times_gpu.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.4f}s")
        
        avg_gpu = np.mean(times_gpu)
        print(f"  Average: {avg_gpu:.4f}s")
        print(f"\nSpeedup: {avg_cpu/avg_gpu:.2f}x")
        
        # Verify results match
        print(f"\nResults match: {np.allclose(model_cpu.coef_, model_gpu.coef_, rtol=1e-3)}")
    else:
        print("\nGPU not available, skipping GPU benchmark")


if __name__ == "__main__":
    # Small test
    benchmark(n_samples=10000, n_features=50, n_runs=3)
    
    print("\n" + "=" * 50)
    print("Larger test:")
    benchmark(n_samples=100000, n_features=200, n_runs=3)
