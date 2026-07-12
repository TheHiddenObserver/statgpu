"""
Comprehensive benchmark: statgpu vs sklearn vs statsmodels vs R.
Compares computation time and numerical accuracy under equivalent objectives.
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("StatGPU Benchmark: Time & Accuracy Comparison")
print("=" * 80)

np.random.seed(42)
N_SAMPLES = 10000
N_FEATURES = 50
NOISE = 0.1

X = np.random.randn(N_SAMPLES, N_FEATURES)
true_coef = np.random.randn(N_FEATURES) * 2
true_intercept = 5.0
y = X @ true_coef + true_intercept + np.random.randn(N_SAMPLES) * NOISE
y_binary = (y > np.median(y)).astype(int)

print(f"\nDataset: {N_SAMPLES} samples × {N_FEATURES} features")
print(f"Data size: {X.nbytes / 1e6:.1f} MB")
print()

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from statgpu.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from statgpu._config import set_device, cuda_available

has_gpu = cuda_available()
print(f"GPU available: {has_gpu}")
print()

# ============================================================================
# 1. LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("1. LINEAR REGRESSION")
print("=" * 80)

results_lr = {}

print("\n--- statgpu CPU ---")
set_device('cpu')
model = LinearRegression(device='cpu')
t0 = time.perf_counter()
model.fit(X, y)
results_lr['statgpu_cpu'] = {
    'time': (time.perf_counter() - t0) * 1000,
    'coef': model.coef_.copy(),
    'intercept': model.intercept_,
    'r2': model.rsquared,
}
print(f"Time: {results_lr['statgpu_cpu']['time']:.2f} ms")
print(f"R²: {results_lr['statgpu_cpu']['r2']:.6f}")

if has_gpu:
    print("\n--- statgpu GPU ---")
    set_device('cuda')
    model = LinearRegression(device='cuda')
    t0 = time.perf_counter()
    model.fit(X, y)
    results_lr['statgpu_gpu'] = {
        'time': (time.perf_counter() - t0) * 1000,
        'coef': model.coef_.copy(),
        'intercept': model.intercept_,
        'r2': model.rsquared,
    }
    print(f"Time: {results_lr['statgpu_gpu']['time']:.2f} ms")
    print(f"R²: {results_lr['statgpu_gpu']['r2']:.6f}")
    speedup = results_lr['statgpu_cpu']['time'] / results_lr['statgpu_gpu']['time']
    print(f"Speedup vs CPU: {speedup:.2f}x")

try:
    from sklearn.linear_model import LinearRegression as SklearnLR
    print("\n--- sklearn ---")
    model = SklearnLR()
    t0 = time.perf_counter()
    model.fit(X, y)
    results_lr['sklearn'] = {
        'time': (time.perf_counter() - t0) * 1000,
        'coef': model.coef_.copy(),
        'intercept': model.intercept_,
        'r2': model.score(X, y),
    }
    print(f"Time: {results_lr['sklearn']['time']:.2f} ms")
    print(f"R²: {results_lr['sklearn']['r2']:.6f}")
except ImportError:
    print("sklearn not available")

try:
    import statsmodels.api as sm
    print("\n--- statsmodels ---")
    X_const = sm.add_constant(X)
    t0 = time.perf_counter()
    model = sm.OLS(y, X_const).fit()
    results_lr['statsmodels'] = {
        'time': (time.perf_counter() - t0) * 1000,
        'coef': np.array(model.params[1:]),
        'intercept': float(model.params[0]),
        'r2': model.rsquared,
    }
    print(f"Time: {results_lr['statsmodels']['time']:.2f} ms")
    print(f"R²: {results_lr['statsmodels']['r2']:.6f}")
except ImportError:
    print("statsmodels not available")

print("\n--- Accuracy Comparison (vs sklearn) ---")
if 'sklearn' in results_lr:
    for name, result in results_lr.items():
        if name != 'sklearn':
            coef_diff = np.max(np.abs(result['coef'] - results_lr['sklearn']['coef']))
            intercept_diff = abs(result['intercept'] - results_lr['sklearn']['intercept'])
            r2_diff = abs(result['r2'] - results_lr['sklearn']['r2'])
            print(f"{name:20s}: coef_diff={coef_diff:.2e}, intercept_diff={intercept_diff:.2e}, R²_diff={r2_diff:.2e}")

# ============================================================================
# 2. RIDGE REGRESSION
# ============================================================================
STATGPU_RIDGE_ALPHA = 1.0
SKLEARN_RIDGE_ALPHA = N_SAMPLES * STATGPU_RIDGE_ALPHA
print("\n" + "=" * 80)
print(
    "2. RIDGE REGRESSION "
    f"(statgpu alpha={STATGPU_RIDGE_ALPHA}, sklearn alpha={SKLEARN_RIDGE_ALPHA})"
)
print("=" * 80)
print("Mapping: sklearn_alpha = n_samples * statgpu_alpha")

results_ridge = {}

print("\n--- statgpu CPU ---")
model = Ridge(alpha=STATGPU_RIDGE_ALPHA, device='cpu')
t0 = time.perf_counter()
model.fit(X, y)
results_ridge['statgpu_cpu'] = {
    'time': (time.perf_counter() - t0) * 1000,
    'coef': model.coef_.copy(),
    'r2': model.rsquared,
}
print(f"Time: {results_ridge['statgpu_cpu']['time']:.2f} ms")
print(f"R²: {results_ridge['statgpu_cpu']['r2']:.6f}")

if has_gpu:
    print("\n--- statgpu GPU ---")
    model = Ridge(alpha=STATGPU_RIDGE_ALPHA, device='cuda')
    t0 = time.perf_counter()
    model.fit(X, y)
    results_ridge['statgpu_gpu'] = {
        'time': (time.perf_counter() - t0) * 1000,
        'coef': model.coef_.copy(),
        'r2': model.rsquared,
    }
    print(f"Time: {results_ridge['statgpu_gpu']['time']:.2f} ms")
    speedup = results_ridge['statgpu_cpu']['time'] / results_ridge['statgpu_gpu']['time']
    print(f"Speedup vs CPU: {speedup:.2f}x")

try:
    from sklearn.linear_model import Ridge as SklearnRidge
    print("\n--- sklearn ---")
    model = SklearnRidge(alpha=SKLEARN_RIDGE_ALPHA)
    t0 = time.perf_counter()
    model.fit(X, y)
    results_ridge['sklearn'] = {
        'time': (time.perf_counter() - t0) * 1000,
        'coef': model.coef_.copy(),
        'r2': model.score(X, y),
    }
    print(f"Time: {results_ridge['sklearn']['time']:.2f} ms")
    print(f"R²: {results_ridge['sklearn']['r2']:.6f}")
except ImportError:
    print("sklearn not available")

print("\n--- Accuracy Comparison ---")
if 'sklearn' in results_ridge:
    for name, result in results_ridge.items():
        if name != 'sklearn':
            coef_diff = np.max(np.abs(result['coef'] - results_ridge['sklearn']['coef']))
            r2_diff = abs(result['r2'] - results_ridge['sklearn']['r2'])
            print(f"{name:20s}: coef_diff={coef_diff:.2e}, R²_diff={r2_diff:.2e}")

# ============================================================================
# 3. LASSO REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("3. LASSO REGRESSION (alpha=0.1)")
print("=" * 80)

results_lasso = {}

print("\n--- statgpu CPU ---")
model = Lasso(alpha=0.1, device='cpu', max_iter=1000)
t0 = time.perf_counter()
model.fit(X, y)
results_lasso['statgpu_cpu'] = {
    'time': (time.perf_counter() - t0) * 1000,
    'coef': model.coef_.copy(),
    'r2': model.rsquared,
    'n_iter': model.n_iter_,
}
print(f"Time: {results_lasso['statgpu_cpu']['time']:.2f} ms")
print(f"R²: {results_lasso['statgpu_cpu']['r2']:.6f}")
print(f"Iterations: {results_lasso['statgpu_cpu']['n_iter']}")
print(f"Non-zero coefs: {np.sum(np.abs(model.coef_) > 1e-10)}")

if has_gpu:
    print("\n--- statgpu GPU ---")
    model = Lasso(alpha=0.1, device='cuda', max_iter=1000)
    t0 = time.perf_counter()
    model.fit(X, y)
    results_lasso['statgpu_gpu'] = {
        'time': (time.perf_counter() - t0) * 1000,
        'coef': model.coef_.copy(),
        'r2': model.rsquared,
        'n_iter': model.n_iter_,
    }
    print(f"Time: {results_lasso['statgpu_gpu']['time']:.2f} ms")
    print(f"Iterations: {results_lasso['statgpu_gpu']['n_iter']}")
    speedup = results_lasso['statgpu_cpu']['time'] / results_lasso['statgpu_gpu']['time']
    print(f"Speedup vs CPU: {speedup:.2f}x")
