"""
Test regression diagnostics.
"""

import numpy as np
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print("=" * 60)
print("Regression Diagnostics Test")
print("=" * 60)

# Generate data with some multicollinearity
np.random.seed(42)
n_samples = 200
n_features = 8

X = np.random.randn(n_samples, n_features)
# Make x7 and x8 correlated with x1 (multicollinearity)
X[:, 6] = X[:, 0] + np.random.randn(n_samples) * 0.1
X[:, 7] = X[:, 0] + np.random.randn(n_samples) * 0.1

true_coef = np.array([2, -1.5, 3, 0.5, -2, 1, 0, 0])  # x6, x7 not used
y = X @ true_coef + np.random.randn(n_samples) * 0.5

print(f"\nDataset: {n_samples} samples × {n_features} features")
print("Note: x7 and x8 are correlated with x1 (multicollinearity)")

# Fit model
from statgpu.linear_model import LinearRegression
from statgpu.diagnostics import diagnose_model, RegressionDiagnostics

model = LinearRegression(device='cpu')
model.fit(X, y)

print(f"\nR²: {model.rsquared:.4f}")

# Run diagnostics
print("\n")
diag = diagnose_model(model)

# Additional tests
print("\n" + "=" * 60)
print("Additional Diagnostic Values")
print("=" * 60)

print(f"\nResiduals range: [{np.min(diag.residuals):.4f}, {np.max(diag.residuals):.4f}]")
print(f"Standardized residuals range: [{np.min(diag.standardized_residuals):.4f}, {np.max(diag.standardized_residuals):.4f}]")
print(f"Studentized residuals range: [{np.min(diag.studentized_residuals):.4f}, {np.max(diag.studentized_residuals):.4f}]")

# Top 3 influential points
cooks = diag.cooks_distance
top3 = np.argsort(cooks)[-3:]
print(f"\nTop 3 influential points (Cook's D): {top3}")
print(f"  Cook's D: {cooks[top3]}")

print("\n" + "=" * 60)
print("Diagnostics test completed!")
print("=" * 60)
