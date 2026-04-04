"""
Test GPU full pipeline: data on GPU -> fit on GPU -> predict on GPU -> results on GPU.
Only final results transferred to CPU.
"""

import numpy as np
import time
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print("=" * 70)
print("GPU Full Pipeline Test (Data->Fit->Predict all on GPU)")
print("=" * 70)

from statgpu._config import set_device, cuda_available
import cupy as cp

# Check GPU
if not cuda_available():
    print("No GPU available")
    exit()

print("✓ GPU available\n")

# Generate data on CPU
np.random.seed(42)
n_samples, n_features = 10000, 100
X_cpu = np.random.randn(n_samples, n_features).astype(np.float32)
y_cpu = X_cpu @ np.random.randn(n_features).astype(np.float32)

print(f"Dataset: {n_samples:,} × {n_features}")

# ============================================
# Full GPU Pipeline
# ============================================
print("\n" + "=" * 70)
print("FULL GPU PIPELINE")
print("=" * 70)

# Step 1: Transfer data to GPU (once)
t0 = time.perf_counter()
X_gpu = cp.asarray(X_cpu)
y_gpu = cp.asarray(y_cpu)
cp.cuda.Device().synchronize()
transfer_to = (time.perf_counter() - t0) * 1000
print(f"1. Data transfer CPU->GPU: {transfer_to:.2f} ms")

# Step 2: Fit on GPU
t0 = time.perf_counter()
from statgpu.linear_model import Ridge
set_device('cuda')
model = Ridge(alpha=1.0, device='cuda')
model.fit(X_gpu, y_gpu)
cp.cuda.Device().synchronize()
fit_time = (time.perf_counter() - t0) * 1000
print(f"2. Fit on GPU: {fit_time:.2f} ms")

# Step 3: Predict on GPU (using GPU data)
t0 = time.perf_counter()
# Convert coef to cupy for GPU prediction
coef_gpu = cp.asarray(model.coef_)
intercept_gpu = float(model.intercept_)
y_pred_gpu = X_gpu @ coef_gpu + intercept_gpu
cp.cuda.Device().synchronize()
predict_time = (time.perf_counter() - t0) * 1000
print(f"3. Predict on GPU: {predict_time:.2f} ms")

# Step 4: Compute R² on GPU
t0 = time.perf_counter()
ss_res_gpu = cp.sum((y_gpu - y_pred_gpu) ** 2)
y_mean_gpu = cp.mean(y_gpu)
ss_tot_gpu = cp.sum((y_gpu - y_mean_gpu) ** 2)
r2_gpu = 1 - ss_res_gpu / ss_tot_gpu
cp.cuda.Device().synchronize()
r2_time = (time.perf_counter() - t0) * 1000
print(f"4. R² compute on GPU: {r2_time:.2f} ms")

# Step 5: Transfer results back (once at the end)
t0 = time.perf_counter()
r2_result = float(r2_gpu.get())
y_pred_result = cp.asnumpy(y_pred_gpu)
transfer_back = (time.perf_counter() - t0) * 1000
print(f"5. Results transfer GPU->CPU: {transfer_back:.2f} ms")

total_gpu = transfer_to + fit_time + predict_time + r2_time + transfer_back
print(f"\nTotal GPU pipeline: {total_gpu:.2f} ms")
print(f"R² = {r2_result:.6f}")

# ============================================
# Compare with CPU
# ============================================
print("\n" + "=" * 70)
print("CPU PIPELINE (for comparison)")
print("=" * 70)

t0 = time.perf_counter()
set_device('cpu')
model_cpu = Ridge(alpha=1.0, device='cpu')
model_cpu.fit(X_cpu, y_cpu)
y_pred_cpu = model_cpu.predict(X_cpu)
r2_cpu = model_cpu.score(X_cpu, y_cpu)
cpu_time = (time.perf_counter() - t0) * 1000

print(f"CPU total: {cpu_time:.2f} ms")
print(f"R² = {r2_cpu:.6f}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"GPU pipeline: {total_gpu:.2f} ms")
print(f"CPU pipeline: {cpu_time:.2f} ms")
print(f"Speedup: {cpu_time/total_gpu:.2f}x")

print("\nBreakdown:")
print(f"  - Data transfer: {transfer_to + transfer_back:.2f} ms ({(transfer_to + transfer_back)/total_gpu*100:.1f}%)")
print(f"  - GPU compute: {fit_time + predict_time + r2_time:.2f} ms ({(fit_time + predict_time + r2_time)/total_gpu*100:.1f}%)")

print("\n" + "=" * 70)
print("Note: For multiple predictions, data stays on GPU")
print("      Only first transfer and last transfer needed!")
print("=" * 70)
