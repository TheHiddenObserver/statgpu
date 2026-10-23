"""Cyclic (periodic) cubic spline basis with GPU acceleration."""

from __future__ import annotations

__all__ = ["cyclic_cubic_spline_basis"]

import numpy as np

from statgpu.backends import _to_numpy, xp_asarray
from statgpu.nonparametric.splines._bspline_basis import bspline_basis, _get_xp


def cyclic_cubic_spline_basis(x, knots, xp=None):
    """Construct a cyclic (periodic) cubic B-spline basis.

    Enforces matching values and first two one-sided derivatives at min(x)
    and max(x). Interior knots must lie strictly between these boundaries.
    Boundaries and the arbitrary basis orientation are recomputed each call;
    this function does not save a training basis or extrapolate periodically.

    Parameters
    ----------
    x : array-like, shape (n,)
        Finite evaluation points with a nonzero range.
    knots : array-like, shape (m,)
        Finite, strictly increasing interior knots, with m >= 1.
    xp : module, optional
        Array module (numpy, cupy, or torch). If None, uses numpy.

    Returns
    -------
    B : array, shape (n, m + 1)
        Float64 periodic cubic basis. Columns have arbitrary SVD orientation;
        constants belong to its span, but no column is a dedicated intercept.

    Raises
    ------
    ValueError
        If inputs are invalid or the three periodic constraints cannot be
        resolved reliably in float64 at the supplied knot spacing.

    Notes
    -----
    Exact clamped-cubic endpoint derivative coefficients replace numerical
    differences. Constraint rows are balanced before a backend-native SVD.
    Only knot metadata and singular values are inspected on the host; basis
    evaluation, SVD, and projection use the requested backend/device. No host
    SVD fallback is used.
    """
    xp = _get_xp(xp)
    x = xp_asarray(x, dtype=xp.float64, xp=xp, ref_arr=x).ravel()
    knots = xp_asarray(knots, dtype=xp.float64, xp=xp, ref_arr=x).ravel()

    # The ordinary constructor owns finite/order/interior-knot validation.
    # Explicit boundaries retain the evaluation-range contract.
    if x.shape[0] == 0:
        raise ValueError("x must contain at least one value")
    if knots.shape[0] < 1:
        raise ValueError("At least one knot is required")
    a = float(xp.min(x))
    b = float(xp.max(x))
    B = bspline_basis(x, knots, degree=3, xp=xp,
                      boundary_lo=a, boundary_hi=b)
    width = b - a
    if not np.isfinite(width):
        raise ValueError("Evaluation range is too large for reliable periodic constraints")

    k = np.asarray(_to_numpy(knots), dtype=np.float64)
    # Dimensionless endpoint spans. The second span uses the opposite
    # boundary when there is just one interior knot.
    l1, r1 = (k[0] - a) / width, (b - k[-1]) / width
    l2 = (k[1] - a) / width if len(k) > 1 else 1.0
    r2 = (b - k[-2]) / width if len(k) > 1 else 1.0
    h = min(l1, r1)
    if h <= 0:
        raise ValueError("Knot spacing cannot resolve reliable periodic constraints")

    C = np.zeros((3, B.shape[1]), dtype=np.float64)
    C[0, 0], C[0, -1] = 1.0, -1.0
    # d1 multiplied by h/3; d2 multiplied by h**2/6. Form
    # bounded ratios directly, avoiding overflow from inverse tiny spans.
    C[1, :2] += np.array([-1.0, 1.0]) * (h / l1)
    C[1, -2:] -= np.array([-1.0, 1.0]) * (h / r1)
    C[2, :3] += np.array([1.0, -1.0 - l1/l2, l1/l2]) * (h/l1)**2
    C[2, -3:] -= np.array([r1/r2, -1.0 - r1/r2, 1.0]) * (h/r1)**2
    C /= np.linalg.norm(C, axis=1)[:, None]
    constraints = xp_asarray(C, dtype=xp.float64, xp=xp, ref_arr=x)
    _, singular_values, vt = xp.linalg.svd(constraints, full_matrices=True)
    s = np.asarray(_to_numpy(singular_values))
    if not np.isfinite(s).all() or s[-1] <= max(C.shape) * s[0] * np.finfo(float).eps:
        raise ValueError("Knot spacing cannot resolve three independent periodic constraints")
    result = B @ vt[3:].T
    if not bool(xp.all(xp.isfinite(result))):
        raise ValueError("Knot spacing cannot produce a finite periodic basis")
    return result
