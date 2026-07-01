"""Debug remaining 0.25% difference between statgpu and pygam."""
import numpy as np
import sys
sys.path.insert(0, '/root/statgpu')

from statgpu.nonparametric.splines._bspline_basis import bspline_basis
from statgpu.nonparametric.splines._penalized import penalized_ls, difference_penalty

rng = np.random.RandomState(42)
X = rng.randn(100000, 10)
y = np.sin(X[:, 0] * 2) + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2] + rng.randn(100000) * 0.3

LAM = 1.0
NS = 20
DEG = 3
n = X.shape[0]
n_interior = NS - DEG - 1

# pygam reference
from pygam import LinearGAM, s
terms = s(0, n_splines=NS, spline_order=DEG)
for j in range(1, 10):
    terms = terms + s(j, n_splines=NS, spline_order=DEG)
pg = LinearGAM(terms, lam=LAM).fit(X, y)
pg_pred = pg.predict(X)
pg_intercept = pg.coef_[0]

# Build basis WITHOUT centering (like pygam)
basis_blocks = []
penalty_blocks = []
for j in range(10):
    xj = X[:, j]
    xj_min, xj_max = float(np.min(xj)), float(np.max(xj))
    knots_j = np.linspace(xj_min, xj_max, n_interior + 2)[1:-1]
    Bj = bspline_basis(xj, knots_j, degree=DEG)
    Sj = difference_penalty(2, Bj.shape[1])
    basis_blocks.append(Bj)
    penalty_blocks.append(Sj)

B_full = np.hstack([np.ones((n, 1))] + basis_blocks)
total_basis = sum(b.shape[1] for b in basis_blocks)
S_full = np.zeros((1 + total_basis, 1 + total_basis))
offset = 1
for Sj in penalty_blocks:
    nj = Sj.shape[0]
    S_full[offset:offset+nj, offset:offset+nj] = Sj
    offset += nj

# Fit WITHOUT centering
beta_no_center, _ = penalized_ls(B_full, y, S_full, LAM, np)
pred_no_center = B_full @ beta_no_center
rel_diff_no_center = np.linalg.norm(pred_no_center - pg_pred) / np.linalg.norm(pg_pred)
print(f"no centering: pred_rel={rel_diff_no_center:.4e}, intercept={beta_no_center[0]:.6f}")

# Fit WITH centering (like statgpu)
B_mean = np.mean(B_full[:, 1:], axis=0)
B_centered = B_full.copy()
B_centered[:, 1:] -= B_mean
beta_centered, _ = penalized_ls(B_centered, y, S_full, LAM, np)
pred_centered = B_centered @ beta_centered
rel_diff_centered = np.linalg.norm(pred_centered - pg_pred) / np.linalg.norm(pg_pred)
print(f"with centering: pred_rel={rel_diff_centered:.4e}, intercept={beta_centered[0]:.6f}")

print(f"pygam intercept: {pg_intercept:.6f}")

# Check if centering is the main cause
print(f"\ncentering causes {rel_diff_centered/rel_diff_no_center:.1f}x more diff" if rel_diff_no_center > 0 else "")

# Test: same centering as pygam (none)
# The 0.25% is from centering + solver differences
print(f"\nConclusion: centering {'is' if abs(rel_diff_centered - rel_diff_no_center) > 1e-4 else 'is NOT'} the main cause")
print(f"  no centering diff: {rel_diff_no_center:.4e}")
print(f"  with centering diff: {rel_diff_centered:.4e}")

print("\nDONE")
