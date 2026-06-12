# -*- coding: utf-8 -*-
"""Debug why CuPy fused kernel diverges for SCAD/MCP in fista_lla_path."""
import numpy as np
import cupy as cp
from statgpu.glm_core._solver import _get_sqerr_proximal_cupy
from statgpu.penalties._scad import SCADPenalty
from statgpu.backends import _to_numpy

rng = np.random.RandomState(42)
n, p = 100, 5
X_np = rng.randn(n, p)
y_np = X_np @ np.array([1, 2, 3, 4, 5]) + rng.randn(n) * 0.5
X_cu = cp.asarray(X_np)
y_cu = cp.asarray(y_np)

# Precompute
X_c = X_cu
y_c = y_c = y_cu
n_samples = n
XtX = X_c.T @ X_c
Xty = X_c.T @ y_c
L = float(cp.linalg.eigvalsh(XtX)[-1]) / n_samples
step = 1.0 / L

penalty = SCADPenalty(alpha=0.1)
fused = _get_sqerr_proximal_cupy()

# Run FISTA with LLA weight updates (like fista_lla_path does)
coef = cp.zeros(p, dtype=cp.float64)
y_k = coef.copy()

# LLA outer loop (3 iterations)
for lla_iter in range(3):
    # Update LLA weights based on current coefficients
    lla_w = penalty.lla_weights(coef)
    thresh = lla_w * step

    # FISTA inner loop (20 iterations)
    t_k = 1.0
    for i in range(20):
        coef_old = coef.copy()
        grad = (XtX @ y_k - Xty) / n_samples

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        beta_mom = (t_k - 1.0) / t_new
        t_k = t_new

        # Fused step
        coef, y_k = fused(coef, grad, step, thresh, coef_old, beta_mom)

    coef_norm = float(cp.linalg.norm(coef))
    print(f"  LLA iter {lla_iter}: coef_norm={coef_norm:.6f} thresh={_to_numpy(thresh)[:3]}")

print(f"\nFinal coef: {_to_numpy(coef)}")
print(f"NaN: {bool(cp.any(cp.isnan(coef)))}")
