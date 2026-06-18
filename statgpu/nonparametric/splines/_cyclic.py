"""Cyclic (periodic) cubic spline basis with GPU acceleration."""

from __future__ import annotations

__all__ = ["cyclic_cubic_spline_basis"]

import numpy as np

from statgpu.backends import _LINALG_ERRORS, _to_numpy
from statgpu.nonparametric.splines._bspline_basis import bspline_basis, _get_xp


def cyclic_cubic_spline_basis(x, knots, xp=None):
    """Construct a cyclic (periodic) cubic B-spline basis.

    Enforces periodicity constraints at the boundary:
    f(a) = f(b), f'(a) = f'(b), f''(a) = f''(b)

    where a = min(knots), b = max(knots).  This reduces the basis by
    3 functions compared to a standard B-spline basis.

    Parameters
    ----------
    x : array-like, shape (n,)
        Evaluation points.
    knots : array-like, shape (m,)
        Interior knots (must be strictly increasing).
    xp : module, optional
        Array module (numpy, cupy, or torch).  If None, uses numpy.

    Returns
    -------
    B : array, shape (n, m + degree + 1 - 3)
        Cyclic cubic spline basis matrix.

    Notes
    -----
    Uses the null-space projection method: build a constraint matrix C
    from the periodicity conditions, compute the null space of C, and
    project the standard B-spline basis onto it.
    """
    xp = _get_xp(xp)

    x = xp.asarray(x, dtype=xp.float64).ravel()
    knots = xp.asarray(knots, dtype=xp.float64).ravel()

    if knots.shape[0] < 1:
        raise ValueError("At least one knot is required")

    # Use x range as boundaries (not knot range) so knots are strictly interior
    x_lo = float(_to_scalar(xp.min(x)))
    x_hi = float(_to_scalar(xp.max(x)))
    knot_lo = float(_to_scalar(knots[0]))
    knot_hi = float(_to_scalar(knots[-1]))
    boundary_lo = min(x_lo, knot_lo)
    boundary_hi = max(x_hi, knot_hi)

    # Build full cubic B-spline basis
    B = bspline_basis(x, knots, degree=3, xp=xp,
                      boundary_lo=boundary_lo, boundary_hi=boundary_hi)
    n_basis = B.shape[1]

    # Build constraint matrix C (3 constraints x n_basis)
    # Constraint 1: f(a) - f(b) = 0
    # Constraint 2: f'(a) - f'(b) = 0
    # Constraint 3: f''(a) - f''(b) = 0

    # Evaluate basis at boundary points
    a_arr = xp.asarray([boundary_lo], dtype=xp.float64)
    b_arr = xp.asarray([boundary_hi], dtype=xp.float64)

    B_a = bspline_basis(a_arr, knots, degree=3, xp=xp,
                        boundary_lo=boundary_lo, boundary_hi=boundary_hi)
    B_b = bspline_basis(b_arr, knots, degree=3, xp=xp,
                        boundary_lo=boundary_lo, boundary_hi=boundary_hi)

    # Finite difference step size (relative to boundary range)
    rng = boundary_hi - boundary_lo
    eps = max(1e-6 * rng, 1e-10)

    # First derivatives: central difference f'(x) ≈ (f(x+h) - f(x-h)) / 2h
    a_lo = xp.asarray([boundary_lo - eps], dtype=xp.float64)
    a_hi = xp.asarray([boundary_lo + eps], dtype=xp.float64)
    b_lo = xp.asarray([boundary_hi - eps], dtype=xp.float64)
    b_hi = xp.asarray([boundary_hi + eps], dtype=xp.float64)

    B_a_lo = bspline_basis(a_lo, knots, degree=3, xp=xp,
                           boundary_lo=boundary_lo, boundary_hi=boundary_hi)
    B_a_hi = bspline_basis(a_hi, knots, degree=3, xp=xp,
                           boundary_lo=boundary_lo, boundary_hi=boundary_hi)
    B_b_lo = bspline_basis(b_lo, knots, degree=3, xp=xp,
                           boundary_lo=boundary_lo, boundary_hi=boundary_hi)
    B_b_hi = bspline_basis(b_hi, knots, degree=3, xp=xp,
                           boundary_lo=boundary_lo, boundary_hi=boundary_hi)

    dB_a = (B_a_hi - B_a_lo) / (2 * eps)  # (1, n_basis)
    dB_b = (B_b_hi - B_b_lo) / (2 * eps)

    # Second derivatives: central difference f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
    ddB_a = (B_a_hi - 2 * B_a + B_a_lo) / (eps ** 2)
    ddB_b = (B_b_hi - 2 * B_b + B_b_lo) / (eps ** 2)

    # Constraint matrix C: shape (3, n_basis)
    C = xp.zeros((3, n_basis), dtype=xp.float64)
    C[0, :] = B_a - B_b           # f(a) = f(b)
    C[1, :] = dB_a - dB_b         # f'(a) = f'(b)
    C[2, :] = ddB_a - ddB_b       # f''(a) = f''(b)

    # Find null space of C via SVD
    try:
        U, S_vec, Vt = xp.linalg.svd(C, full_matrices=True)
    except (AttributeError, _LINALG_ERRORS):
        # Fallback: use numpy for SVD
        C_np = _to_numpy(C)
        U_np, S_np, Vt_np = np.linalg.svd(C_np, full_matrices=True)
        Vt = xp.asarray(Vt_np, dtype=xp.float64)

    # Null space = rows of Vt corresponding to zero singular values
    # Determine rank from singular values (not hardcoded)
    if S_vec.shape[0] > 0:
        tol = max(C.shape) * float(_to_scalar(S_vec[0])) * np.finfo(np.float64).eps
        rank = int(np.sum(np.abs(_to_numpy(S_vec)) > tol))
    else:
        rank = 0
    null_space = Vt[rank:].T  # (n_basis, n_basis - rank)

    # Project B-spline basis onto null space
    B_cyclic = B @ null_space  # (n, n_basis - 3)

    return B_cyclic


def _to_scalar(x):
    """Extract a Python scalar from a backend array."""
    if hasattr(x, 'item'):
        return x.item()
    return float(x)
