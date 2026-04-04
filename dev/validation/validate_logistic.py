"""
Validation script for LogisticRegression against sklearn and statsmodels.
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Logistic Regression Validation")
print("=" * 80)

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
true_coef = np.array([0.5, -1.0, 0.3, 0.0, 2.0])
true_intercept = 0.1

# Generate binary outcomes
z = X @ true_coef + true_intercept
p = 1 / (1 + np.exp(-z))
y = (np.random.rand(n_samples) < p).astype(int)

print(f"\nDataset: {n_samples} samples, {n_features} features")
print(f"Class balance: {np.mean(y):.2%} positive")

# Split into train/test
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ========================================================================
# 1. statgpu LogisticRegression (CPU)
# ========================================================================
print("\n" + "=" * 80)
print("1. statgpu LogisticRegression (CPU)")
print("=" * 80)

from statgpu.linear_model import LogisticRegression
from statgpu._config import set_device

set_device('cpu')

start = time.time()
sg_model = LogisticRegression(device='cpu', max_iter=100, C=1e10)  # No regularization
sg_model.fit(X_train, y_train)
sg_time_cpu = time.time() - start

print(f"\nFitting time: {sg_time_cpu:.4f}s")
print(f"Iterations: {sg_model.n_iter_}")
print(f"\nCoefficients:")
print(f"  Intercept: {sg_model.intercept_:.6f}")
for i, c in enumerate(sg_model.coef_):
    print(f"  x{i+1}: {c:.6f}")

print(f"\nAccuracy: {sg_model.score(X_test, y_test):.4f}")
print(f"Log-likelihood: {sg_model.loglikelihood:.4f}")
print(f"AIC: {sg_model.aic:.4f}")
print(f"BIC: {sg_model.bic:.4f}")
print(f"Pseudo R-squared: {sg_model.pseudo_rsquared:.4f}")

# ========================================================================
# 2. sklearn LogisticRegression
# ========================================================================
print("\n" + "=" * 80)
print("2. sklearn LogisticRegression")
print("=" * 80)

from sklearn.linear_model import LogisticRegression as SklearnLR

start = time.time()
sk_model = SklearnLR(max_iter=100, C=1e10, solver='lbfgs', penalty='l2')
sk_model.fit(X_train, y_train)
sk_time = time.time() - start

print(f"\nFitting time: {sk_time:.4f}s")
print(f"Iterations: {sk_model.n_iter_}")
print(f"\nCoefficients:")
print(f"  Intercept: {sk_model.intercept_[0]:.6f}")
for i, c in enumerate(sk_model.coef_[0]):
    print(f"  x{i+1}: {c:.6f}")

print(f"\nAccuracy: {sk_model.score(X_test, y_test):.4f}")

# ========================================================================
# 3. statsmodels Logit
# ========================================================================
print("\n" + "=" * 80)
print("3. statsmodels Logit")
print("=" * 80)

try:
    import statsmodels.api as sm
    
    X_train_const = sm.add_constant(X_train)
    
    start = time.time()
    sm_model = sm.Logit(y_train, X_train_const)
    sm_result = sm_model.fit(disp=0, maxiter=100)
    sm_time = time.time() - start
    
    print(f"\nFitting time: {sm_time:.4f}s")
    print(f"Iterations: {sm_result.mle_retvals.get('iterations', 'N/A')}")
    print(f"\nCoefficients:")
    for i, name in enumerate(['const'] + [f'x{i+1}' for i in range(n_features)]):
        print(f"  {name}: {sm_result.params[i]:.6f}")
    
    y_pred_sm = (sm_result.predict(sm.add_constant(X_test)) >= 0.5).astype(int)
    sm_acc = np.mean(y_pred_sm == y_test)
    print(f"\nAccuracy: {sm_acc:.4f}")
    print(f"Log-likelihood: {sm_result.llf:.4f}")
    print(f"AIC: {sm_result.aic:.4f}")
    print(f"BIC: {sm_result.bic:.4f}")
    print(f"Pseudo R-squared: {sm_result.prsquared:.4f}")
    
    has_statsmodels = True
except ImportError:
    print("statsmodels not installed, skipping comparison")
    has_statsmodels = False

# ========================================================================
# 4. Comparison
# ========================================================================
print("\n" + "=" * 80)
print("4. Comparison Summary")
print("=" * 80)

print("\nCoefficients comparison:")
print(f"{'Parameter':<15} {'statgpu':>12} {'sklearn':>12} {'statsmodels':>12}")
print("-" * 55)
print(f"{'Intercept':<15} {sg_model.intercept_:>12.6f} {sk_model.intercept_[0]:>12.6f} {sm_result.params[0] if has_statsmodels else 'N/A':>12}")
for i in range(n_features):
    sm_val = sm_result.params[i+1] if has_statsmodels else float('nan')
    print(f"{'x'+str(i+1):<15} {sg_model.coef_[i]:>12.6f} {sk_model.coef_[0][i]:>12.6f} {sm_val:>12.6f}")

# Compute differences
print("\nMax absolute difference:")
print(f"  statgpu vs sklearn: {np.max(np.abs(sg_model.coef_ - sk_model.coef_[0])):.8f}")
if has_statsmodels:
    sm_coef = sm_result.params[1:]
    print(f"  statgpu vs statsmodels: {np.max(np.abs(sg_model.coef_ - sm_coef)):.8f}")
    print(f"  sklearn vs statsmodels: {np.max(np.abs(sk_model.coef_[0] - sm_coef)):.8f}")

# ========================================================================
# 5. GPU Benchmark (if available)
# ========================================================================
print("\n" + "=" * 80)
print("5. GPU Benchmark")
print("=" * 80)

from statgpu._config import cuda_available

if cuda_available():
    print("CUDA is available, running GPU benchmark...")
    
    set_device('cuda')
    
    # Warmup
    _ = LogisticRegression(device='cuda', max_iter=10)
    _.fit(X_train[:100], y_train[:100])
    
    # Benchmark
    start = time.time()
    sg_model_gpu = LogisticRegression(device='cuda', max_iter=100, C=1e10)
    sg_model_gpu.fit(X_train, y_train)
    sg_time_gpu = time.time() - start
    
    print(f"\nGPU fitting time: {sg_time_gpu:.4f}s")
    print(f"CPU fitting time: {sg_time_cpu:.4f}s")
    print(f"Speedup: {sg_time_cpu / sg_time_gpu:.2f}x")
    
    # Verify results match
    print(f"\nGPU-CPU coefficient match: {np.allclose(sg_model.coef_, sg_model_gpu.coef_, rtol=1e-3)}")
    print(f"Max coefficient difference: {np.max(np.abs(sg_model.coef_ - sg_model_gpu.coef_)):.8f}")
else:
    print("CUDA not available, skipping GPU benchmark")

# ========================================================================
# 6. Summary Statistics Comparison
# ========================================================================
if has_statsmodels:
    print("\n" + "=" * 80)
    print("6. Statistical Output Comparison")
    print("=" * 80)
    
    print("\nStandard Errors:")
    print(f"{'Parameter':<15} {'statgpu':>12} {'statsmodels':>12} {'Diff':>12}")
    print("-" * 55)
    for i, name in enumerate(['Intercept'] + [f'x{i+1}' for i in range(n_features)]):
        sg_se = sg_model._bse[i]
        sm_se = sm_result.bse[i]
        print(f"{name:<15} {sg_se:>12.6f} {sm_se:>12.6f} {abs(sg_se-sm_se):>12.6f}")
    
    print("\np-values:")
    print(f"{'Parameter':<15} {'statgpu':>12} {'statsmodels':>12} {'Diff':>12}")
    print("-" * 55)
    for i, name in enumerate(['Intercept'] + [f'x{i+1}' for i in range(n_features)]):
        sg_p = sg_model._pvalues[i]
        sm_p = sm_result.pvalues[i]
        print(f"{name:<15} {sg_p:>12.6f} {sm_p:>12.6f} {abs(sg_p-sm_p):>12.6f}")

print("\n" + "=" * 80)
print("Validation Complete")
print("=" * 80)
