"""
Compare Ridge implementations: original vs GPU-only.
"""

import numpy as np
import time
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print("=" * 60)
print("Ridge Implementation Comparison")
print("=" * 60)

# Import both implementations
from statgpu.linear_model import Ridge

# Test different sizes
sizes = [
    (1000, 50),
    (5000, 100),
    (10000, 100),
]

for n_samples, n_features in sizes:
    print(f"\n{'=' * 60}")
    print(f"Dataset: {n_samples:,} x {n_features}")
    print(f"{'=' * 60}")
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X @ np.random.randn(n_features).astype(np.float32)
    
    # CPU version
    from statgpu._config import set_device
    set_device('cpu')
    model = Ridge(alpha=1.0, device='cpu')
    t0 = time.time()
    model.fit(X, y)
    cpu_time = (time.time() - t0) * 1000
    print(f"CPU: {cpu_time:.2f} ms")
    
    # Check GPU available
    from statgpu._config import cuda_available
    if cuda_available():
        set_device('cuda')
        model = Ridge(alpha=1.0, device='cuda')
        t0 = time.time()
        model.fit(X, y)
        gpu_time = (time.time() - t0) * 1000
        print(f"GPU: {gpu_time:.2f} ms")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("GPU not available")

print("\n" + "=" * 60)
print("Comparison complete!")
print("=" * 60)
