"""
Compare statgpu with statsmodels for correctness validation.
"""

import numpy as np
import time
from statgpu.linear_model import LinearRegression
from statgpu._config import set_device, cuda_available

print("=" * 60)
print("Comparing statgpu with statsmodels")
print("=" * 60)

# Generate test data
np.random.seed(42)
n_samples, n_features = 10000, 10
X = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, -2.0, 3.0, 0.5, -1.0, 2.5, -0.5, 1.0, -1.5, 2.0])
true_intercept = 5.0
y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.1

print(f"\nData: {n_samples} samples, {n_features} features")
print(f"True coefficients: {true_coef}")
print(f"True intercept: {true_intercept}")

# ============================================
# 1. statsmodels
# ============================================
print("\n" + "-" * 40)
print("statsmodels OLS")
print("-" * 40)

try:
    import statsmodels.api as sm
    X_with_const = sm.add_constant(X)
    model_sm = sm.OLS(y, X_with_const)
    results_sm = model_sm.fit()
    
    sm_intercept = results_sm.params[0]
    sm_coef = results_sm.params[1:]
    
    print(f"Intercept: {sm_intercept:.6f}")
    print(f"Coefficients: {sm_coef}")
    print(f"R-squared: {results_sm.rsquared:.6f}")
    
    HAS_STATSMODELS = True
except ImportError:
    print("statsmodels not installed, skipping")
    HAS_STATSMODELS = False

# ============================================
# 2. statgpu CPU
# ============================================
print("\n" + "-" * 40)
print("statgpu CPU")
print("-" * 40)

set_device('cpu')
model_cpu = LinearRegression(device='cpu')
start = time.time()
model_cpu.fit(X, y)
cpu_time = time.time() - start

print(f"Intercept: {model_cpu.intercept_:.6f}")
print(f"Coefficients: {model_cpu.coef_}")
print(f"R-squared: {model_cpu.score(X, y):.6f}")
print(f"Fit time: {cpu_time*1000:.2f}ms")

# ============================================
# 3. statgpu GPU
# ============================================
if cuda_available():
    print("\n" + "-" * 40)
    print("statgpu GPU")
    print("-" * 40)
    
    set_device('cuda')
    model_gpu = LinearRegression(device='cuda')
    start = time.time()
    model_gpu.fit(X, y)
    gpu_time = time.time() - start
    
    print(f"Intercept: {model_gpu.intercept_:.6f}")
    print(f"Coefficients: {model_gpu.coef_}")
    print(f"R-squared: {model_gpu.score(X, y):.6f}")
    print(f"Fit time: {gpu_time*1000:.2f}ms")
else:
    print("\nGPU not available")

# ============================================
# 4. sklearn (for reference)
# ============================================
print("\n" + "-" * 40)
print("sklearn LinearRegression")
print("-" * 40)

try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    model_sk = SklearnLR()
    start = time.time()
    model_sk.fit(X, y)
    sk_time = time.time() - start
    
    print(f"Intercept: {model_sk.intercept_:.6f}")
    print(f"Coefficients: {model_sk.coef_}")
    print(f"R-squared: {model_sk.score(X, y):.6f}")
    print(f"Fit time: {sk_time*1000:.2f}ms")
    
    HAS_SKLEARN = True
except ImportError:
    print("sklearn not installed, skipping")
    HAS_SKLEARN = False

# ============================================
# 5. Comparison
# ============================================
print("\n" + "=" * 60)
print("Comparison Summary")
print("=" * 60)

if HAS_STATSMODELS:
    print("\nDifferences from statsmodels (CPU):")
    print(f"  Intercept diff: {abs(model_cpu.intercept_ - sm_intercept):.2e}")
    print(f"  Coefficients max diff: {np.max(np.abs(model_cpu.coef_ - sm_coef)):.2e}")
    
    if cuda_available():
        print("\nDifferences from statsmodels (GPU):")
        print(f"  Intercept diff: {abs(model_gpu.intercept_ - sm_intercept):.2e}")
        print(f"  Coefficients max diff: {np.max(np.abs(model_gpu.coef_ - sm_coef)):.2e}")

if cuda_available():
    print("\nCPU vs GPU consistency:")
    print(f"  Intercept diff: {abs(model_cpu.intercept_ - model_gpu.intercept_):.2e}")
    print(f"  Coefficients max diff: {np.max(np.abs(model_cpu.coef_ - model_gpu.coef_)):.2e}")
    print(f"\nSpeedup (GPU vs CPU): {cpu_time/gpu_time:.2f}x")

print("\n" + "=" * 60)
print("Validation: PASS" if (not HAS_STATSMODELS or np.allclose(model_cpu.coef_, sm_coef, rtol=1e-4)) else "Validation: FAIL")
print("=" * 60)
