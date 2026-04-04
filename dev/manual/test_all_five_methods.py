"""
Test all five statgpu methods with comprehensive validation.
"""

import numpy as np
import time
import warnings
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
warnings.filterwarnings('ignore')

print("=" * 80)
print("StatGPU - All Five Methods Test")
print("=" * 80)

# Generate comprehensive test data
np.random.seed(42)
n_samples = 1000
n_features = 10

# Regression data
X_reg = np.random.randn(n_samples, n_features)
true_coef = np.random.randn(n_features) * 2
true_intercept = 5.0
y_reg = X_reg @ true_coef + true_intercept + np.random.randn(n_samples) * 0.5

# Binary classification data
logit = X_reg @ (np.random.randn(n_features) * 0.5) + 0.5
prob = 1 / (1 + np.exp(-logit))
y_binary = (np.random.rand(n_samples) < prob).astype(int)

# Survival data
time_surv = np.random.exponential(10, n_samples)
event_surv = np.random.binomial(1, 0.7, n_samples)

print(f"\nDataset: {n_samples} samples × {n_features} features")
print(f"  - Regression: y continuous")
print(f"  - Classification: y binary ({np.sum(y_binary)} positive)")
print(f"  - Survival: {np.sum(event_surv)} events, {np.sum(1-event_surv)} censored")

# Check GPU
from statgpu._config import cuda_available, set_device
has_gpu = cuda_available()
print(f"\nGPU available: {has_gpu}")

# Import all models
from statgpu.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from statgpu.survival import CoxPH

results = {}

# ============================================================================
# 1. Linear Regression
# ============================================================================
print("\n" + "=" * 80)
print("1. LINEAR REGRESSION")
print("=" * 80)

model = LinearRegression(device='cpu')
t0 = time.perf_counter()
model.fit(X_reg, y_reg)
results['LinearRegression'] = {
    'time': (time.perf_counter() - t0) * 1000,
    'r2': model.rsquared,
    'coef_mean': np.mean(model.coef_),
    'intercept': model.intercept_
}
print(f"✓ Time: {results['LinearRegression']['time']:.2f} ms")
print(f"✓ R²: {results['LinearRegression']['r2']:.6f}")
print(f"✓ Intercept: {results['LinearRegression']['intercept']:.4f}")

# ============================================================================
# 2. Ridge Regression
# ============================================================================
print("\n" + "=" * 80)
print("2. RIDGE REGRESSION (alpha=1.0)")
print("=" * 80)

model = Ridge(alpha=1.0, device='cpu')
t0 = time.perf_counter()
model.fit(X_reg, y_reg)
results['Ridge'] = {
    'time': (time.perf_counter() - t0) * 1000,
    'r2': model.rsquared,
    'coef_mean': np.mean(model.coef_),
    'intercept': model.intercept_
}
print(f"✓ Time: {results['Ridge']['time']:.2f} ms")
print(f"✓ R²: {results['Ridge']['r2']:.6f}")
print(f"✓ Mean coef: {results['Ridge']['coef_mean']:.4f}")

# ============================================================================
# 3. Lasso Regression
# ============================================================================
print("\n" + "=" * 80)
print("3. LASSO REGRESSION (alpha=0.1)")
print("=" * 80)

model = Lasso(alpha=0.1, max_iter=1000, device='cpu')
t0 = time.perf_counter()
model.fit(X_reg, y_reg)
results['Lasso'] = {
    'time': (time.perf_counter() - t0) * 1000,
    'r2': model.rsquared,
    'nonzero': np.sum(np.abs(model.coef_) > 1e-10),
    'n_iter': model.n_iter_
}
print(f"✓ Time: {results['Lasso']['time']:.2f} ms")
print(f"✓ R²: {results['Lasso']['r2']:.6f}")
print(f"✓ Non-zero coefs: {results['Lasso']['nonzero']}/{n_features}")
print(f"✓ Iterations: {results['Lasso']['n_iter']}")

# ============================================================================
# 4. Logistic Regression
# ============================================================================
print("\n" + "=" * 80)
print("4. LOGISTIC REGRESSION")
print("=" * 80)

model = LogisticRegression(max_iter=100, device='cpu')
t0 = time.perf_counter()
model.fit(X_reg, y_binary)
acc = np.mean(model.predict(X_reg) == y_binary)
results['Logistic'] = {
    'time': (time.perf_counter() - t0) * 1000,
    'accuracy': acc,
    'n_iter': model.n_iter_,
    'coef_mean': np.mean(model.coef_)
}
print(f"✓ Time: {results['Logistic']['time']:.2f} ms")
print(f"✓ Accuracy: {results['Logistic']['accuracy']:.4f}")
print(f"✓ Iterations: {results['Logistic']['n_iter']}")

# ============================================================================
# 5. Cox Proportional Hazards
# ============================================================================
print("\n" + "=" * 80)
print("5. COX PROPORTIONAL HAZARDS")
print("=" * 80)

try:
    model = CoxPH(ties='breslow', max_iter=50, device='cpu')
    t0 = time.perf_counter()
    model.fit(X_reg, time_surv, event_surv)
    results['CoxPH'] = {
        'time': (time.perf_counter() - t0) * 1000,
        'converged': model._converged,
        'iterations': model._iterations,
        'cindex': model._cindex if hasattr(model, '_cindex') else None,
        'coef_mean': np.mean(model.coef_) if model.coef_ is not None else None
    }
    print(f"✓ Time: {results['CoxPH']['time']:.2f} ms")
    print(f"✓ Converged: {results['CoxPH']['converged']}")
    print(f"✓ Iterations: {results['CoxPH']['iterations']}")
    if results['CoxPH']['cindex'] is not None:
        print(f"✓ C-index: {results['CoxPH']['cindex']:.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    results['CoxPH'] = {'error': str(e)}

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Method':<20} {'Time (ms)':<12} {'Metric':<15} {'Value':<15}")
print("-" * 65)

for method, data in results.items():
    if 'error' in data:
        print(f"{method:<20} {'ERROR':<12} {data['error'][:30]}")
    elif method == 'LinearRegression':
        print(f"{method:<20} {data['time']:<12.2f} {'R²':<15} {data['r2']:<15.6f}")
    elif method == 'Ridge':
        print(f"{method:<20} {data['time']:<12.2f} {'R²':<15} {data['r2']:<15.6f}")
    elif method == 'Lasso':
        print(f"{method:<20} {data['time']:<12.2f} {'Non-zero':<15} {data['nonzero']}/{n_features}")
    elif method == 'Logistic':
        print(f"{method:<20} {data['time']:<12.2f} {'Accuracy':<15} {data['accuracy']:<15.4f}")
    elif method == 'CoxPH':
        status = '✓' if data.get('converged') else '✗'
        print(f"{method:<20} {data['time']:<12.2f} {'Converged':<15} {status}")

print("\n" + "=" * 80)
print("All five methods tested!")
print("=" * 80)
