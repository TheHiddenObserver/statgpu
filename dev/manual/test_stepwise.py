"""
Test stepwise model selection.
"""

import numpy as np
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print("=" * 60)
print("Stepwise Model Selection Test")
print("=" * 60)

# Generate data with some irrelevant features
np.random.seed(42)
n_samples = 200
n_features = 10

X = np.random.randn(n_samples, n_features)
# Only first 5 features are relevant
true_coef = np.array([2, -1.5, 3, 0.5, -2, 0, 0, 0, 0, 0])
y = X @ true_coef + np.random.randn(n_samples) * 0.5

print(f"\nDataset: {n_samples} samples × {n_features} features")
print(f"True relevant features: 0, 1, 2, 3, 4")
print(f"True irrelevant features: 5, 6, 7, 8, 9")

# Test 1: Forward selection with LinearRegression
print("\n" + "=" * 60)
print("Test 1: Forward Selection (AIC)")
print("=" * 60)

from statgpu.feature_selection import StepwiseSelector
from statgpu.linear_model import LinearRegression

selector = StepwiseSelector(
    LinearRegression,
    criterion='aic',
    direction='forward',
    device='cpu'
)
selector.fit(X, y)
selector.summary()

print(f"\nSelected features: {selector.selected_features_}")
print(f"Expected: [0, 1, 2, 3, 4]")

# Test 2: Backward elimination
print("\n" + "=" * 60)
print("Test 2: Backward Elimination (BIC)")
print("=" * 60)

selector2 = StepwiseSelector(
    LinearRegression,
    criterion='bic',
    direction='backward',
    device='cpu'
)
selector2.fit(X, y)
selector2.summary()

print(f"\nSelected features: {selector2.selected_features_}")

# Test 3: Bidirectional
print("\n" + "=" * 60)
print("Test 3: Bidirectional (AIC)")
print("=" * 60)

selector3 = StepwiseSelector(
    LinearRegression,
    criterion='aic',
    direction='both',
    device='cpu'
)
selector3.fit(X, y)
selector3.summary()

print(f"\nSelected features: {selector3.selected_features_}")

# Test 4: With Ridge
print("\n" + "=" * 60)
print("Test 4: Stepwise with Ridge")
print("=" * 60)

from statgpu.linear_model import Ridge

selector4 = StepwiseSelector(
    Ridge,
    criterion='aic',
    direction='forward',
    alpha=1.0,
    device='cpu'
)
selector4.fit(X, y)
selector4.summary()

# Test 5: With Lasso
print("\n" + "=" * 60)
print("Test 5: Stepwise with Lasso")
print("=" * 60)

from statgpu.linear_model import Lasso

selector5 = StepwiseSelector(
    Lasso,
    criterion='aic',
    direction='forward',
    alpha=0.1,
    device='cpu'
)
selector5.fit(X, y)
selector5.summary()

print("\n" + "=" * 60)
print("All stepwise tests completed!")
print("=" * 60)
