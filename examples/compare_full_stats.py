"""
Compare statgpu full summary with statsmodels and R-style output.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Full Statistical Output Comparison")
print("statgpu vs statsmodels (R-style)")
print("=" * 80)

# Generate data (same as R's set.seed(42))
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, -2.0, 3.0, 0.5, -1.0])
true_intercept = 5.0
y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.5

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Noise level: 0.5")
print()

# ============================================
# 1. statsmodels (reference)
# ============================================
print("\n" + "=" * 80)
print("statsmodels OLS (Reference)")
print("=" * 80)

import statsmodels.api as sm
X_with_const = sm.add_constant(X)
model_sm = sm.OLS(y, X_with_const)
results_sm = model_sm.fit()
print(results_sm.summary().as_text())

# ============================================
# 2. statgpu CPU
# ============================================
print("\n" + "=" * 80)
print("statgpu CPU")
print("=" * 80)

from statgpu.linear_model import LinearRegression
from statgpu._config import set_device

set_device('cpu')
model_cpu = LinearRegression(device='cpu')
model_cpu.fit(X, y)
model_cpu.summary()

# ============================================
# 3. statgpu GPU (if available)
# ============================================
from statgpu._config import cuda_available

if cuda_available():
    print("\n" + "=" * 80)
    print("statgpu GPU")
    print("=" * 80)
    
    set_device('cuda')
    model_gpu = LinearRegression(device='cuda')
    model_gpu.fit(X, y)
    model_gpu.summary()

# ============================================
# 4. Detailed Comparison
# ============================================
print("\n" + "=" * 80)
print("Detailed Comparison: statgpu vs statsmodels")
print("=" * 80)

print(f"\n{'Metric':<25} {'statsmodels':>15} {'statgpu CPU':>15} {'Diff':>15}")
print("-" * 75)

metrics = [
    ('Intercept', results_sm.params[0], model_cpu._params[0]),
    ('x1 coef', results_sm.params[1], model_cpu._params[1]),
    ('x2 coef', results_sm.params[2], model_cpu._params[2]),
    ('x3 coef', results_sm.params[3], model_cpu._params[3]),
    ('x4 coef', results_sm.params[4], model_cpu._params[4]),
    ('x5 coef', results_sm.params[5], model_cpu._params[5]),
]

for name, sm_val, sg_val in metrics:
    diff = abs(sm_val - sg_val)
    print(f"{name:<25} {sm_val:>15.6f} {sg_val:>15.6f} {diff:>15.2e}")

print()
print(f"{'Model Stats':<25} {'statsmodels':>15} {'statgpu CPU':>15} {'Diff':>15}")
print("-" * 75)

stats_compare = [
    ('R-squared', results_sm.rsquared, model_cpu.rsquared),
    ('Adj. R-squared', results_sm.rsquared_adj, model_cpu.rsquared_adj),
    ('F-statistic', results_sm.fvalue, model_cpu.fvalue),
    ('AIC', results_sm.aic, model_cpu.aic),
    ('BIC', results_sm.bic, model_cpu.bic),
    ('Log-Likelihood', results_sm.llf, model_cpu.llf),
]

for name, sm_val, sg_val in stats_compare:
    if sm_val is not None and sg_val is not None:
        diff = abs(sm_val - sg_val)
        print(f"{name:<25} {sm_val:>15.4f} {sg_val:>15.4f} {diff:>15.2e}")

print("\n" + "=" * 80)
print("Validation: All differences should be < 1e-10 (machine precision)")
print("=" * 80)
