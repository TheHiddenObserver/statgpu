"""
Simple test for CoxPH model.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CoxPH Simple Test")
print("=" * 80)

# Generate survival data
np.random.seed(42)
n = 200
p = 5

X = np.random.randn(n, p)
true_coef = np.array([0.5, -0.3, 0.8, -0.2, 0.4])

# Generate survival times (exponential with hazard depending on X)
hazard = np.exp(X @ true_coef)
time = np.random.exponential(1.0 / hazard)

# Censoring (30% censored)
event = np.random.binomial(1, 0.7, n)

print(f"\nData: {n} samples, {p} features")
print(f"Events: {np.sum(event)} ({np.mean(event)*100:.1f}%)")
print(f"Censored: {np.sum(1-event)} ({(1-np.mean(event))*100:.1f}%)")

# Fit CoxPH
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from statgpu.survival import CoxPH

print("\nFitting CoxPH (CPU)...")
model = CoxPH(ties='breslow', device='cpu', max_iter=50)
model.fit(X, time, event)

print(f"\n✓ Converged: {model._converged}")
print(f"✓ Iterations: {model._iterations}")
print(f"✓ C-index: {model._cindex:.4f}")

print("\nCoefficients:")
for i, coef in enumerate(model.coef_):
    print(f"  x{i+1}: {coef:.4f} (HR: {np.exp(coef):.4f})")

print("\n" + "=" * 80)
print("CoxPH test completed!")
print("=" * 80)
