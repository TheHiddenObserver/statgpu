"""
B-spline and natural cubic spline basis construction with GPU support.

Implements De Boor's recursive algorithm for B-spline basis evaluation,
vectorized over sample points for efficient GPU computation.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _LINALG_ERRORS, _torch_dev, _to_numpy, xp_zeros, xp_eye, xp_full, xp_astype, xp_asarray


def _get_xp(xp):
    """Return the array module (numpy if xp is None)."""
    return xp if xp is not None else np


def bspline_basis(x, knots, degree=3, xp=None):
    """
    Evaluate B-spline basis matrix at points x.

    Uses De Boor's recursive algorithm, vectorized over all sample points
    for efficient GPU computation.

    Parameters
    ----------
    x : array-like, shape (n,)
        Evaluation points.
    knots : array-like, shape (m,)
        Interior knots (must be strictly increasing).
    degree : int, default=3
        Spline degree (3 = cubic).
    xp : module, optional
        Array module (numpy, cupy, or torch). If None, uses numpy.

    Returns
    -------
    B : array, shape (n, m + degree + 1)
        B-spline basis matrix. Each row corresponds to a sample point,
        each column to a basis function.
    """
    xp = _get_xp(xp)

    x = xp.asarray(x, dtype=xp.float64).ravel()
    knots = xp.asarray(knots, dtype=xp.float64).ravel()
    n = x.shape[0]
    m = knots.shape[0]

    if m == 0:
        raise ValueError("At least one interior knot is required")

    # Construct augmented knot vector:
    # t = [x_min]*(degree+1), knots..., [x_max]*(degree+1)
    # Use the wider of (x range, knots range) to define boundary knots,
    # so evaluation at boundary points is valid.
    x_min = float(xp.min(x))
    x_max = float(xp.max(x))
    knot_min = float(xp.min(knots))
    knot_max = float(xp.max(knots))

    # Boundary knots should be at or beyond both x range and knot range
    boundary_lo = min(x_min, knot_min)
    boundary_hi = max(x_max, knot_max)

    # Ensure interior knots are strictly within boundary
    if knot_min <= boundary_lo or knot_max >= boundary_hi:
        raise ValueError(
            "Interior knots must be strictly within the boundary range "
            f"({boundary_lo}, {boundary_hi})"
        )

    left_pad = xp_full(degree + 1, boundary_lo, xp.float64, xp, x)
    right_pad = xp_full(degree + 1, boundary_hi, xp.float64, xp, x)
    t = xp.concatenate([left_pad, knots, right_pad])

    n_knots = len(t)
    n_basis = n_knots - degree - 1  # = m + degree + 1

    # Pre-extract all knot values to CPU in one transfer
    t_cpu = _to_numpy(t).tolist()

    # De Boor recursion, vectorized over x
    # Initialize degree-0 indicator functions for all n_knots-1 intervals.
    n_intervals = n_knots - 1
    B = xp_zeros((n, n_intervals), xp.float64, xp, x)

    # B_{i,0}(x) = 1 if t_i <= x < t_{i+1} else 0
    # For the last non-degenerate interval, include right endpoint.
    last_nondeg = -1
    for i in range(n_intervals):
        if t_cpu[i + 1] > t_cpu[i]:
            last_nondeg = i

    # Vectorized degree-0 initialization
    for i in range(n_intervals):
        t_i = t_cpu[i]
        t_ip1 = t_cpu[i + 1]
        if t_ip1 > t_i:
            if i == last_nondeg:
                mask = (x >= t_i) & (x <= t_ip1)
            else:
                mask = (x >= t_i) & (x < t_ip1)
            B[:, i] = xp_astype(mask, xp.float64, xp)

    # Recursive computation for degrees 1, 2, ..., degree
    # Outer loop has data dependencies (each k uses B from k-1).
    # Inner loop over basis functions is vectorized.
    for k in range(1, degree + 1):
        n_cur = n_intervals - k

        # Precompute knot arrays for all basis functions at once
        # Use xp_asarray with ref_arr=x to ensure same device (GPU if x is on GPU)
        t_lo = xp_asarray([t_cpu[i] for i in range(n_cur)], dtype=xp.float64, xp=xp, ref_arr=x)
        t_hi = xp_asarray([t_cpu[i + k] for i in range(n_cur)], dtype=xp.float64, xp=xp, ref_arr=x)
        t_ip1 = xp_asarray([t_cpu[i + 1] for i in range(n_cur)], dtype=xp.float64, xp=xp, ref_arr=x)
        t_ip1k = xp_asarray([t_cpu[i + 1 + k] for i in range(n_cur)], dtype=xp.float64, xp=xp, ref_arr=x)

        denom1 = t_hi - t_lo
        denom2 = t_ip1k - t_ip1

        # Safe denominators (replace 0 with 1 to avoid division by zero)
        safe_denom1 = xp.where(denom1 > 0, denom1, 1.0)
        safe_denom2 = xp.where(denom2 > 0, denom2, 1.0)

        # Masks: (n_cur,)
        mask1 = denom1 > 0
        mask2 = denom2 > 0

        # w1, w2: (n, n_cur) — broadcast x (n,) with knot arrays (n_cur,)
        w1 = xp.where(mask1[None, :], (x[:, None] - t_lo[None, :]) / safe_denom1[None, :], 0.0)
        w2 = xp.where(mask2[None, :], (x[:, None] - t_ip1[None, :]) / safe_denom2[None, :], 0.0)

        # Vectorized De Boor step: (n, n_cur)
        B_new = w1 * B[:, :n_cur] + (1.0 - w2) * B[:, 1:n_cur + 1]
        B = B_new

    # Final result has exactly n_basis columns
    return B


def natural_cubic_spline_basis(x, knots, xp=None):
    """
    Natural cubic spline basis (linear beyond boundary knots).

    Constructs a cubic B-spline basis and applies boundary constraints
    to enforce linearity beyond the boundary knots. This reduces the
    effective number of basis functions by 2 compared to a regular
    cubic B-spline.

    Parameters
    ----------
    x : array-like, shape (n,)
        Evaluation points.
    knots : array-like, shape (m,)
        Interior knots (must be strictly increasing).
    xp : module, optional
        Array module (numpy, cupy, or torch). If None, uses numpy.

    Returns
    -------
    B : array, shape (n, m + 1)
        Natural cubic spline basis matrix. The first column is typically
        the intercept (constant), and the remaining columns are the
        natural spline basis functions.
    """
    xp = _get_xp(xp)

    x = xp.asarray(x, dtype=xp.float64).ravel()
    knots = xp.asarray(knots, dtype=xp.float64).ravel()
    n = x.shape[0]
    m = knots.shape[0]

    if m < 1:
        raise ValueError("At least one interior knot is required for natural cubic splines")

    x_min = float(xp.min(x))
    x_max = float(xp.max(x))

    # Build cubic B-spline basis
    # Use a range that covers both x and knots for bspline_basis
    knot_min = float(xp.min(knots))
    knot_max = float(xp.max(knots))
    eval_min = min(x_min, knot_min - 1.0)
    eval_max = max(x_max, knot_max + 1.0)

    B_cubic = bspline_basis(x, knots, degree=3, xp=xp)
    n_basis = B_cubic.shape[1]

    # Apply boundary constraints to enforce linearity beyond boundary knots.
    # The constraint is that the second derivative is zero at the boundary knots.
    # Build the constraint matrix C such that C @ beta = 0
    # where beta are the coefficients of the cubic B-spline basis.

    # For numerical differentiation, we use points near the boundaries
    # but with a range wide enough to cover the knots.
    eps = 1e-6

    # Create evaluation arrays wide enough for bspline_basis validation
    # Second derivative at x_min (near left boundary knot)
    x_eval_lo = xp_asarray([x_min, x_min + eps, x_min + 2 * eps,
                            x_max, x_max - eps, x_max - 2 * eps],
                           dtype=xp.float64, xp=xp, ref_arr=x)

    # Build basis at all 6 evaluation points at once
    B_eval = bspline_basis(x_eval_lo, knots, degree=3, xp=xp)

    # Extract individual evaluations
    B_lo = B_eval[0:1, :]
    B_lo_eps = B_eval[1:2, :]
    B_lo_eps2 = B_eval[2:3, :]
    B_hi = B_eval[3:4, :]
    B_hi_eps = B_eval[4:5, :]
    B_hi_eps2 = B_eval[5:6, :]

    d2_lo = (B_lo_eps2 - 2 * B_lo_eps + B_lo) / (eps ** 2)
    d2_hi = (B_hi_eps2 - 2 * B_hi_eps + B_hi) / (eps ** 2)

    # Stack constraints: C is (2, n_basis)
    C = xp.vstack([d2_lo, d2_hi])

    # Find null space of C using SVD.
    # C is (2, n_basis).  SVD gives U(2,2), S(2,), Vh(n_basis, n_basis).
    # The null space is spanned by the last (n_basis - rank) rows of Vh.
    try:
        U, S_vals, Vh = xp.linalg.svd(C)
        n_rank = int(xp.sum(S_vals > max(C.shape) * S_vals[0] * xp.finfo(xp.float64).eps))
        null_space = Vh[n_rank:].T  # shape: (n_basis, n_basis - n_rank)
    except _LINALG_ERRORS:
        # Fallback: use QR with mode='reduced' and manual extension
        Q_full, _ = xp.linalg.qr(xp_eye(n_basis, xp.float64, xp, x))
        Q_c, _ = xp.linalg.qr(C.T)
        # Orthogonal complement of column space of C.T
        null_space = Q_full[:, C.shape[0]:]

    # Project the B-spline basis onto the null space
    # B_natural = B_cubic @ null_space
    B_natural = B_cubic @ null_space

    return B_natural


def _bspline_basis_derivative(x, knots, degree=3, deriv_order=1, xp=None):
    """
    Evaluate derivative of B-spline basis.

    Uses the derivative formula for B-splines:
    B'_{i,k}(x) = k/(t_{i+k} - t_i) * B_{i,k-1}(x) - k/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)

    Parameters
    ----------
    x : array-like, shape (n,)
        Evaluation points.
    knots : array-like, shape (m,)
        Interior knots.
    degree : int, default=3
        Spline degree.
    deriv_order : int, default=1
        Order of derivative (must be <= degree).
    xp : module, optional
        Array module.

    Returns
    -------
    dB : array, shape (n, n_basis)
        Derivative of B-spline basis matrix.
    """
    xp = _get_xp(xp)

    if deriv_order > degree:
        return xp_zeros((len(x), len(knots) + degree + 1), xp.float64, xp, x)

    if deriv_order == 0:
        return bspline_basis(x, knots, degree=degree, xp=xp)

    # Compute derivative using the recursive formula
    # For first derivative of degree k B-spline:
    # B'_{i,k} = k/(t_{i+k}-t_i) * B_{i,k-1} - k/(t_{i+k+1}-t_{i+1}) * B_{i+1,k-1}

    x = xp_asarray(x, dtype=xp.float64, xp=xp).ravel()
    knots = xp_asarray(knots, dtype=xp.float64, xp=xp, ref_arr=x).ravel()

    x_min = float(xp.min(x))
    x_max = float(xp.max(x))

    left_pad = xp_full(degree + 1, x_min, xp.float64, xp, x)
    right_pad = xp_full(degree + 1, x_max, xp.float64, xp, x)
    t = xp.concatenate([left_pad, knots, right_pad])

    # Get B-spline basis of degree (degree - deriv_order)
    reduced_degree = degree - deriv_order
    B_reduced = bspline_basis(x, knots, degree=reduced_degree, xp=xp)

    n_basis = len(t) - degree - 1
    n_basis_reduced = len(t) - reduced_degree - 1

    # Apply the derivative formula recursively
    # For each derivative order, we apply:
    # dB_{i,k} = k/(t_{i+k}-t_i) * B_{i,k-1} - k/(t_{i+k+1}-t_{i+1}) * B_{i+1,k-1}

    dB = B_reduced
    for d in range(deriv_order):
        current_degree = reduced_degree + d
        n_current = dB.shape[1]
        dB_new = xp_zeros((len(x), n_current - 1), xp.float64, xp, x)

        for i in range(n_current - 1):
            denom1 = float(t[i + current_degree] - t[i])
            denom2 = float(t[i + current_degree + 1] - t[i + 1])

            term1 = (current_degree / denom1 * dB[:, i]) if denom1 > 0 else xp_zeros(len(x), xp.float64, xp, x)
            term2 = (current_degree / denom2 * dB[:, i + 1]) if denom2 > 0 else xp_zeros(len(x), xp.float64, xp, x)

            dB_new[:, i] = term1 - term2

        dB = dB_new

    return dB
