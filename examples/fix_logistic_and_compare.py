"""
Fix Logistic regression and compare with sklearn in detail.
"""

import numpy as np
import time
import warnings
import sys
warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/.openclaw/workspace-coding/statgpu')

print("=" * 80)
print("Logistic Regression Fix & Detailed Comparison")
print("=" * 80)

# Generate data
np.random.seed(42)
N_SAMPLES = 1000
N_FEATURES = 10

X = np.random.randn(N_SAMPLES, N_FEATURES)
true_coef = np.random.randn(N_FEATURES) * 0.5
true_intercept = 0.5
logit = X @ true_coef + true_intercept
prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(N_SAMPLES) < prob).astype(int)

print(f"\nDataset: {N_SAMPLES} samples × {N_FEATURES} features")
print(f"Positive class: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
print(f"Negative class: {np.sum(1-y)} ({(1-np.mean(y))*100:.1f}%)")

# Import models
from statgpu.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLR

# Test 1: Default settings
print("\n" + "=" * 80)
print("Test 1: Default Settings")
print("=" * 80)

print("\n--- statgpu ---")
sg_model = LogisticRegression(C=1.0, max_iter=100, tol=1e-4, device='cpu')
t0 = time.perf_counter()
sg_model.fit(X, y)
sg_time = (time.perf_counter() - t0) * 1000
print(f"Time: {sg_time:.2f} ms")
print(f"Iterations: {sg_model.n_iter_}")
print(f"Converged: {getattr(sg_model, '_converged', 'N/A')}")
print(f"Intercept: {sg_model.intercept_:.6f}")
print(f"Coef (first 5): {sg_model.coef_[:5]}")

print("\n--- sklearn ---")
sk_model = SklearnLR(C=1.0, max_iter=100, tol=1e-4, solver='lbfgs')
t0 = time.perf_counter()
sk_model.fit(X, y)
sk_time = (time.perf_counter() - t0) * 1000
print(f"Time: {sk_time:.2f} ms")
print(f"Iterations: {sk_model.n_iter_[0]}")
print(f"Intercept: {sk_model.intercept_[0]:.6f}")
print(f"Coef (first 5): {sk_model.coef_[0][:5]}")

# Compare
print("\n--- Comparison ---")
intercept_diff = abs(sg_model.intercept_ - sk_model.intercept_[0])
coef_diff = np.max(np.abs(sg_model.coef_ - sk_model.coef_[0]))
print(f"Intercept diff: {intercept_diff:.6f}")
print(f"Max coef diff: {coef_diff:.6f}")

# Test 2: With higher precision
print("\n" + "=" * 80)
print("Test 2: Higher Precision (tol=1e-6)")
print("=" * 80)

print("\n--- statgpu ---")
sg_model2 = LogisticRegression(C=1.0, max_iter=200, tol=1e-6, device='cpu')
sg_model2.fit(X, y)
print(f"Iterations: {sg_model2.n_iter_}")
print(f"Intercept: {sg_model2.intercept_:.6f}")

print("\n--- sklearn ---")
sk_model2 = SklearnLR(C=1.0, max_iter=200, tol=1e-6, solver='lbfgs')
sk_model2.fit(X, y)
print(f"Iterations: {sk_model2.n_iter_[0]}")
print(f"Intercept: {sk_model2.intercept_[0]:.6f}")

print("\n--- Comparison ---")
intercept_diff2 = abs(sg_model2.intercept_ - sk_model2.intercept_[0])
coef_diff2 = np.max(np.abs(sg_model2.coef_ - sk_model2.coef_[0]))
print(f"Intercept diff: {intercept_diff2:.6f}")
print(f"Max coef diff: {coef_diff2:.6f}")

# Test 3: Predictions
print("\n" + "=" * 80)
print("Test 3: Prediction Comparison")
print("=" * 80)

sg_proba = sg_model.predict_proba(X)[:, 1]
sk_proba = sk_model.predict_proba(X)[:, 1]

sg_pred = sg_model.predict(X)
sk_pred = sk_model.predict(X)

print(f"\nProba diff (max): {np.max(np.abs(sg_proba - sk_proba)):.6f}")
print(f"Prediction agreement: {np.mean(sg_pred == sk_pred) * 100:.2f}%")

# Test 4: Different regularization strengths
print("\n" + "=" * 80)
print("Test 4: Different C values")
print("=" * 80)

for C in [0.01, 0.1, 1.0, 10.0]:
    sg_m = LogisticRegression(C=C, max_iter=100, device='cpu')
    sg_m.fit(X, y)
    
    sk_m = SklearnLR(C=C, max_iter=100, solver='lbfgs')
    sk_m.fit(X, y)
    
    diff = np.max(np.abs(sg_m.coef_ - sk_m.coef_[0]))
    print(f"C={C:5.2f}: max coef diff = {diff:.6f}")

print("\n" + "=" * 80)
print("Analysis Complete")
print("=" * 80)

# Summary
print("\n=== SUMMARY ===")
print(f"Default settings - coef diff: {coef_diff:.6f}")
print(f"High precision   - coef diff: {coef_diff2:.6f}")
print(f"Prediction agreement: {np.mean(sg_pred == sk_pred) * 100:.2f}%")

if coef_diff > 0.01:
    print("\n⚠ WARNING: Large coefficient differences detected!")
    print("Possible causes:")
    print("1. Different optimization algorithms (IRLS vs L-BFGS)")
    print("2. Different convergence criteria")
    print("3. Numerical precision issues")
else:
    print("\n✓ Coefficients match well!")
