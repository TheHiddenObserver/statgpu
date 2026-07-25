"""Thin plate spline basis with GPU acceleration."""

from __future__ import annotations

__all__ = ["thin_plate_spline_basis"]

import numpy as np

from statgpu.backends import _to_float_scalar, xp_asarray, xp_maximum, xp_ones
from statgpu.nonparametric.splines._bspline_basis import _get_xp


def thin_plate_spline_basis(x, knots, penalty_order=2, xp=None):
    """Construct a thin plate spline basis.

    Thin plate splines use radial basis functions of the form
    φ(r) = r^{2m-d} log(r) for even d, or r^{2m-d} for odd d,
    where m is the penalty order and d is the input dimensionality.

    For 1-D data (d=1, odd) with m=2: φ(r) = r^3 (where r = |x - knot|).
    For 2-D data (d=2, even) with m=2: φ(r) = r^2 log(r).

    Parameters
    ----------
    x : array-like, shape (n,) or (n, d)
        Evaluation points.  Can be 1-D or multi-dimensional.
    knots : array-like, shape (m,) or (m, d)
        Knot positions.  Must have the same dimensionality as x.
    penalty_order : int, default=2
        Penalty order m.  Controls the smoothness (m=2 gives cubic
        smoothing splines in 1-D).
    xp : module, optional
        Array module (numpy, cupy, or torch).  If None, uses numpy.

    Returns
    -------
    B : array, shape (n, m + d + 1)
        Thin plate spline basis matrix.  Includes the radial basis
        functions plus a polynomial term (intercept + linear terms).

    Notes
    -----
    The thin plate spline basis consists of:
    1. Radial basis functions: φ(||x - ξ_j||) for each knot ξ_j
    2. Polynomial terms: [1, x_1, ..., x_d] (to ensure completeness)

    For 1-D data, this is equivalent to the cubic smoothing spline basis.
    """
    xp = _get_xp(xp)

    x = xp_asarray(x, dtype=xp.float64, xp=xp)
    knots = xp_asarray(knots, dtype=xp.float64, xp=xp, ref_arr=x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if knots.ndim == 1:
        knots = knots.reshape(-1, 1)

    if x.ndim != 2 or knots.ndim != 2 or x.shape[0] == 0 or knots.shape[0] == 0:
        raise ValueError("x and knots must be non-empty one- or two-dimensional arrays")
    if isinstance(penalty_order, bool) or int(penalty_order) != penalty_order or penalty_order < 1:
        raise ValueError("penalty_order must be a positive integer")
    if not bool(_to_float_scalar(xp.all(xp.isfinite(x)))) or not bool(
        _to_float_scalar(xp.all(xp.isfinite(knots)))
    ):
        raise ValueError("x and knots must contain only finite values")

    n, d = x.shape
    m = knots.shape[0]

    if knots.shape[1] != d:
        raise ValueError(
            f"x has {d} dimensions but knots has {knots.shape[1]}"
        )

    # Compute pairwise distances: r_ij = ||x_i - ξ_j||
    # x: (n, d), knots: (m, d)
    # diff: (n, m, d)
    diff = x[:, None, :] - knots[None, :, :]
    # r: (n, m)
    r_sq = xp.sum(diff * diff, axis=2)
    r = xp.sqrt(xp_maximum(r_sq, 1e-30, xp))  # avoid log(0); 1e-30 safe for log

    # Radial basis functions
    if d % 2 == 0:
        # Even dimension: φ(r) = r^{2m-d} log(r)
        # For d=2, m=2: φ(r) = r^2 log(r)
        exponent = 2 * penalty_order - d
        if exponent <= 0:
            raise ValueError(
                f"penalty_order={penalty_order} too small for d={d} dimensions; "
                f"need 2*penalty_order > d (got {2*penalty_order} <= {d})"
            )
        phi = r ** exponent * xp.log(xp_maximum(r, 1e-30, xp))
    else:
        # Odd dimension: φ(r) = r^{2m-d}
        # For d=1, m=2: φ(r) = r^3
        # For d=1, m=1: φ(r) = r
        exponent = 2 * penalty_order - d
        if exponent <= 0:
            raise ValueError(
                f"penalty_order={penalty_order} too small for d={d} dimensions; "
                f"need 2*penalty_order > d (got {2*penalty_order} <= {d})"
            )
        phi = r ** exponent

    # Polynomial terms: [1, x_1, ..., x_d]
    poly = xp_ones((n, d + 1), xp.float64, xp, x)
    if d >= 1:
        poly[:, 1:] = x

    # Concatenate: [phi_1, ..., phi_m, 1, x_1, ..., x_d]
    B = xp.concatenate([phi, poly], axis=1)

    return B
