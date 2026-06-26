"""Debug: compare basis matrices between statgpu and pygam."""
import numpy as np
import sys
sys.path.insert(0, '/root/statgpu')

from statgpu.nonparametric.splines._bspline_basis import bspline_basis

rng = np.random.RandomState(42)
x = rng.randn(10000)

NS = 20
DEG = 3
n_interior = NS - DEG - 1

# statgpu: uniform knots in original space
x_min, x_max = float(np.min(x)), float(np.max(x))
knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
B_sg = bspline_basis(x, knots, degree=DEG)

# pygam: uniform knots in [0,1] space
from pygam.utils import b_spline_basis
edge_knots = np.linspace(0, 1, n_interior + 2)

# pygam operates on data scaled to [0,1]
x_scaled = (x - x_min) / (x_max - x_min)
B_pg = b_spline_basis(x_scaled, n_splines=NS, spline_order=DEG, edge_knots=edge_knots)

# Convert sparse to dense if needed
if hasattr(B_pg, 'toarray'):
    B_pg = B_pg.toarray()

print(f"statgpu basis shape: {B_sg.shape}")
print(f"pygam basis shape: {B_pg.shape}")

# Compare
if B_sg.shape == B_pg.shape:
    diff = np.linalg.norm(B_sg - B_pg) / np.linalg.norm(B_pg)
    print(f"basis rel diff: {diff:.4e}")

    # Check if they're proportional (same shape, different scale)
    # Find best scale factor
    from numpy.linalg import lstsq
    scale = lstsq(B_pg.reshape(-1, 1), B_sg.reshape(-1, 1), rcond=None)[0][0, 0]
    print(f"best scale factor: {scale:.6f}")
    B_pg_scaled = B_pg * scale
    diff_scaled = np.linalg.norm(B_sg - B_pg_scaled) / np.linalg.norm(B_sg)
    print(f"basis rel diff (after scale): {diff_scaled:.4e}")
else:
    print("shapes differ!")

print(f"\nstatgpu row sums: {B_sg.sum(axis=1)[:5]}")
print(f"pygam row sums: {B_pg.sum(axis=1)[:5]}")

print("\nDONE")
