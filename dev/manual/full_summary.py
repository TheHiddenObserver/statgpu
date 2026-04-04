"""
Full summary output comparison with statsmodels and R.
"""

import numpy as np
import time

print("=" * 80)
print("Full Summary Comparison: statgpu vs statsmodels vs R")
print("=" * 80)

# Generate data (similar to R's set.seed(42))
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)

# True coefficients
true_coef = np.array([1.5, -2.0, 3.0, 0.5, -1.0])
true_intercept = 5.0

# Generate y with some noise
y = X @ true_coef + true_intercept + np.random.randn(n_samples) * 0.5

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"True intercept: {true_intercept}")
print(f"True coefficients: {true_coef}")
print()

# ============================================
# 1. statsmodels (reference)
# ============================================
print("-" * 80)
print("statsmodels OLS (Python equivalent to R's lm())")
print("-" * 80)

try:
    import statsmodels.api as sm
    
    X_with_const = sm.add_constant(X)
    model_sm = sm.OLS(y, X_with_const)
    results_sm = model_sm.fit()
    
    print(results_sm.summary())
    HAS_STATSMODELS = True
except ImportError:
    print("statsmodels not installed")
    HAS_STATSMODELS = False

# ============================================
# 2. R-style output using statsmodels
# ============================================
if HAS_STATSMODELS:
    print("\n" + "=" * 80)
    print("Detailed Comparison Table")
    print("=" * 80)
    
    print(f"\n{'Parameter':<15} {'statsmodels':>15} {'statgpu':>15} {'Difference':>15}")
    print("-" * 65)
    
    # Intercept
    print(f"{'Intercept':<15} {results_sm.params[0]:>15.6f} {'(TODO)':>15} {'-':>15}")
    
    # Coefficients
    for i in range(n_features):
        print(f"{'x' + str(i+1):<15} {results_sm.params[i+1]:>15.6f} {'(TODO)':>15} {'-':>15}")
    
    print(f"\n{'Metric':<20} {'statsmodels':>20} {'statgpu':>20}")
    print("-" * 65)
    print(f"{'R-squared':<20} {results_sm.rsquared:>20.6f} {'(TODO)':>20}")
    print(f"{'Adj. R-squared':<20} {results_sm.rsquared_adj:>20.6f} {'(TODO)':>20}")
    print(f"{'F-statistic':<20} {results_sm.fvalue:>20.4f} {'(TODO)':>20}")
    print(f"{'Prob (F-statistic)':<20} {results_sm.f_pvalue:>20.4e} {'(TODO)':>20}")
    print(f"{'AIC':<20} {results_sm.aic:>20.4f} {'(TODO)':>20}")
    print(f"{'BIC':<20} {results_sm.bic:>20.4f} {'(TODO)':>20}")
    print(f"{'Log-Likelihood':<20} {results_sm.llf:>20.4f} {'(TODO)':>20}")

print("\n" + "=" * 80)
print("Next: Implement full summary output in statgpu")
print("=" * 80)
