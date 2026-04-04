"""
Final test of all five models with full GPU computation.
"""

import numpy as np
import time
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print("=" * 80)
print("All Five Models - Full GPU Computation Test")
print("=" * 80)

from statgpu._config import set_device, cuda_available
import cupy as cp

if not cuda_available():
    print("No GPU available")
    exit()

print("✓ GPU available\n")

from statgpu.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from statgpu.survival import CoxPH

# Generate data
np.random.seed(42)
n_samples, n_features = 10000, 100

X_cpu = np.random.randn(n_samples, n_features).astype(np.float32)
y_reg = X_cpu @ np.random.randn(n_features).astype(np.float32)
y_binary = (y_reg > np.median(y_reg)).astype(np.int32)
time_surv = np.random.exponential(10, n_samples).astype(np.float32)
event_surv = np.random.binomial(1, 0.7, n_samples).astype(np.int32)

print(f"Dataset: {n_samples:,} × {n_features}")
print(f"  - Regression: continuous y")
print(f"  - Classification: binary y ({np.sum(y_binary)} positive)")
print(f"  - Survival: {np.sum(event_surv)} events\n")

# Transfer to GPU once
X_gpu = cp.asarray(X_cpu)
y_reg_gpu = cp.asarray(y_reg)
y_binary_gpu = cp.asarray(y_binary)
time_gpu = cp.asarray(time_surv)
event_gpu = cp.asarray(event_surv)
cp.cuda.Device().synchronize()

results = {}

# 1. Linear Regression
print("=" * 80)
print("1. LINEAR REGRESSION")
print("=" * 80)

set_device('cuda')
model = LinearRegression(device='cuda')
t0 = time.perf_counter()
model.fit(X_gpu, y_reg_gpu)
cp.cuda.Device().synchronize()
gpu_time = (time.perf_counter() - t0) * 1000

print(f"GPU fit: {gpu_time:.2f} ms")
print(f"R² (GPU computed): {model.rsquared:.6f}")
print(f"Coefficients: {model.coef_[:5]}")

# 2. Ridge
print("\n" + "=" * 80)
print("2. RIDGE REGRESSION")
print("=" * 80)

model = Ridge(alpha=1.0, device='cuda')
t0 = time.perf_counter()
model.fit(X_gpu, y_reg_gpu)
cp.cuda.Device().synchronize()
gpu_time = (time.perf_counter() - t0) * 1000

print(f"GPU fit: {gpu_time:.2f} ms")
print(f"R²: {model.rsquared:.6f}")

# 3. Lasso
print("\n" + "=" * 80)
print("3. LASSO REGRESSION")
print("=" * 80)

model = Lasso(alpha=0.1, max_iter=500, device='cuda')
t0 = time.perf_counter()
model.fit(X_gpu, y_reg_gpu)
cp.cuda.Device().synchronize()
gpu_time = (time.perf_counter() - t0) * 1000

print(f"GPU fit: {gpu_time:.2f} ms")
print(f"Iterations: {model.n_iter_}")
print(f"R²: {model.rsquared:.6f}")
print(f"Non-zero coefs: {np.sum(np.abs(model.coef_) > 1e-10)}")

# 4. Logistic
print("\n" + "=" * 80)
print("4. LOGISTIC REGRESSION")
print("=" * 80)

model = LogisticRegression(max_iter=100, device='cuda')
t0 = time.perf_counter()
model.fit(X_gpu, y_binary_gpu)
cp.cuda.Device().synchronize()
gpu_time = (time.perf_counter() - t0) * 1000

print(f"GPU fit: {gpu_time:.2f} ms")
print(f"Iterations: {model.n_iter_}")

# 5. CoxPH
print("\n" + "=" * 80)
print("5. COX PROPORTIONAL HAZARDS")
print("=" * 80)

try:
    model = CoxPH(ties='breslow', max_iter=50, device='cuda')
    t0 = time.perf_counter()
    model.fit(X_gpu, time_gpu, event_gpu)
    cp.cuda.Device().synchronize()
    gpu_time = (time.perf_counter() - t0) * 1000
    
    print(f"GPU fit: {gpu_time:.2f} ms")
    print(f"Converged: {model._converged}")
    print(f"Iterations: {model._iterations}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("✓ All five models tested with full GPU computation!")
print("=" * 80)
