"""
Penalized least squares utilities for spline smoothing.

Provides functions for solving penalized regression problems and
constructing difference penalty matrices for spline smoothing.
"""

from __future__ import annotations

import numpy as np

from statgpu.backends import _LINALG_ERRORS, _torch_dev, _to_numpy, xp_zeros, xp_eye, xp_asarray, xp_cholesky_solve


def _get_xp(xp):
    """Return the array module (numpy if xp is None)."""
    return xp if xp is not None else np


def difference_penalty(order, n_coef, xp=None):
    """
    Construct difference penalty matrix of given order.

    The penalty matrix S = D^T @ D penalizes differences between
    adjacent coefficients, encouraging smoothness.

    Parameters
    ----------
    order : int
        Order of differences.  order=1 penalizes first differences
        (piecewise linear), order=2 penalizes second differences
        (piecewise quadratic, the default for smoothing).
    n_coef : int
        Number of spline coefficients (basis functions).
    xp : module, optional
        Array module (numpy, cupy, or torch). If None, uses numpy.

    Returns
    -------
    S : array, shape (n_coef, n_coef)
        Penalty matrix (positive semi-definite).
    """
    xp = _get_xp(xp)

    if order < 1:
        raise ValueError("Penalty order must be >= 1")
    if n_coef <= order:
        raise ValueError(
            f"n_coef ({n_coef}) must be greater than order ({order})"
        )

    # Construct difference matrix D of shape (n_coef - order, n_coef)
    # For order=1: D[i, i] = -1, D[i, i+1] = 1
    # For order=2: D[i, i] = 1, D[i, i+1] = -2, D[i, i+2] = 1

    # Build D using iterative differencing of identity matrix
    D = xp_eye(n_coef, xp.float64, xp)
    for _ in range(order):
        # First differences of current D
        D = D[1:, :] - D[:-1, :]

    # Penalty matrix S = D^T @ D
    S = D.T @ D

    return S


def penalized_ls(B, y, penalty_matrix, lambda_, xp=None):
    """
    Solve penalized least squares problem.

    Minimizes: ||y - B @ beta||^2 + lambda_ * beta^T @ S @ beta

    Parameters
    ----------
    B : array, shape (n, p)
        Basis matrix (design matrix for the spline).
    y : array, shape (n,) or (n, 1)
        Response vector.
    penalty_matrix : array, shape (p, p)
        Penalty matrix S (positive semi-definite).
    lambda_ : float
        Smoothing parameter (must be non-negative).
    xp : module, optional
        Array module (numpy, cupy, or torch). If None, uses numpy.

    Returns
    -------
    beta : array, shape (p,) or (p, 1)
        Fitted coefficients.
    edf : float
        Effective degrees of freedom: trace(B @ (B^T @ B + lambda_ * S)^{-1} @ B^T).
    """
    xp = _get_xp(xp)

    B = xp_asarray(B, dtype=xp.float64, xp=xp)
    y = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=B)
    penalty_matrix = xp_asarray(penalty_matrix, dtype=xp.float64, xp=xp, ref_arr=B)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n, p = B.shape

    # Normal equations: (B^T @ B + lambda_ * S) @ beta = B^T @ y
    BtB = B.T @ B
    Bty = B.T @ y
    A = BtB + lambda_ * penalty_matrix

    # Solve using Cholesky decomposition (more efficient for symmetric positive definite)
    A_used = A  # track which matrix was actually used (for edf consistency)
    try:
        # Add small jitter for numerical stability
        jitter = 1e-10 * xp.trace(A) / p
        A_stable = A + jitter * xp_eye(p, xp.float64, xp, A)
        A_used = A_stable
        beta = xp_cholesky_solve(A_stable, Bty, xp)
    except _LINALG_ERRORS:
        # Fallback to general solve
        try:
            beta = xp.linalg.solve(A, Bty)
        except _LINALG_ERRORS:
            # Last resort: least squares
            beta = xp.linalg.lstsq(A, Bty, rcond=None)[0]

    # Effective degrees of freedom: edf = tr(A^{-1} @ B^T @ B)
    # Use the same matrix as the beta solve for consistency.
    try:
        A_inv_BtB = xp.linalg.solve(A_used, BtB)
        edf = xp.trace(A_inv_BtB)
        # Clamp edf to valid range [0, p]
        # Keep as GPU scalar — use clip/clamp for device compatibility
        if hasattr(edf, 'clamp'):  # torch
            edf = edf.clamp(0.0, float(p))
        else:  # numpy/cupy
            edf = xp.clip(edf, 0.0, float(p))
    except _LINALG_ERRORS:
        edf = float(p)

    # Flatten beta if y was 1D
    if y.shape[1] == 1:
        beta = beta.ravel()

    return beta, edf


def generalized_cross_validation(B, y, penalty_matrix, lambda_, xp=None):
    """
    Compute Generalized Cross-Validation (GCV) score.

    GCV = n * RSS / (n - edf)^2

    where RSS is the residual sum of squares and edf is the effective
    degrees of freedom.

    Parameters
    ----------
    B : array, shape (n, p)
        Basis matrix.
    y : array, shape (n,)
        Response vector.
    penalty_matrix : array, shape (p, p)
        Penalty matrix.
    lambda_ : float
        Smoothing parameter.
    xp : module, optional
        Array module.

    Returns
    -------
    gcv : float
        GCV score (lower is better).
    """
    xp = _get_xp(xp)

    B = xp_asarray(B, dtype=xp.float64, xp=xp)
    y = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=B)

    beta, edf = penalized_ls(B, y, penalty_matrix, lambda_, xp)

    resid = y - B @ beta
    n = len(y)

    rss = xp.sum(resid ** 2)  # GPU scalar, no sync

    # Avoid division by zero or negative denom (edf >= n)
    denom = 1.0 - edf / n
    # Keep denom as GPU scalar for xp.where compatibility
    if hasattr(denom, 'item'):  # torch/cupy scalar
        gcv = xp.where(denom > 1e-10, rss / n / (denom ** 2), xp.tensor(float('inf')) if hasattr(xp, 'tensor') else float('inf'))
    else:
        gcv = rss / n / (denom ** 2) if denom > 1e-10 else float('inf')

    return gcv


def select_lambda_gcv(B, y, penalty_matrix, lambda_grid=None, xp=None):
    """
    Select smoothing parameter via Generalized Cross-Validation.

    Searches over a grid of lambda values and selects the one that
    minimizes the GCV score.

    Parameters
    ----------
    B : array, shape (n, p)
        Basis matrix.
    y : array, shape (n,)
        Response vector.
    penalty_matrix : array, shape (p, p)
        Penalty matrix.
    lambda_grid : array-like, optional
        Grid of lambda values to search over. If None, uses a
        log-spaced grid from 1e-10 to 1e10.
    xp : module, optional
        Array module.

    Returns
    -------
    best_lambda : float
        Lambda value that minimizes GCV.
    gcv_scores : array
        GCV scores for each lambda in the grid.
    """
    xp = _get_xp(xp)

    B = xp_asarray(B, dtype=xp.float64, xp=xp)
    y = xp_asarray(y, dtype=xp.float64, xp=xp, ref_arr=B)
    penalty_matrix = xp_asarray(penalty_matrix, dtype=xp.float64, xp=xp, ref_arr=B)

    if lambda_grid is None:
        lambda_grid = xp.logspace(-10, 10, 100)

    lambda_grid = xp_asarray(lambda_grid, dtype=xp.float64, xp=xp, ref_arr=B)

    # GCV loop on device. penalized_ls and generalized_cross_validation
    # return GPU scalars — no per-iteration sync.
    gcv_list = []
    for i in range(len(lambda_grid)):
        gcv_val = generalized_cross_validation(
            B, y, penalty_matrix, lambda_grid[i], xp
        )
        gcv_list.append(gcv_val)

    gcv_vec = xp.stack(gcv_list)
    gcv_np = _to_numpy(gcv_vec)  # single sync
    best_idx = int(np.argmin(gcv_np))
    best_lambda = float(_to_numpy(lambda_grid)[best_idx])
    gcv_scores = xp_asarray(gcv_np, dtype=xp.float64, xp=xp, ref_arr=B)

    return best_lambda, gcv_scores


def fit_penalized_spline(x, y, knots, degree=3, penalty_order=2,
                          lambda_=1.0, xp=None):
    """
    Fit a penalized spline to data.

    Parameters
    ----------
    x : array-like, shape (n,)
        Predictor variable.
    y : array-like, shape (n,)
        Response variable.
    knots : array-like, shape (m,)
        Interior knots.
    degree : int, default=3
        Spline degree.
    penalty_order : int, default=2
        Order of the difference penalty.
    lambda_ : float, default=1.0
        Smoothing parameter.
    xp : module, optional
        Array module.

    Returns
    -------
    beta : array, shape (n_basis,)
        Fitted spline coefficients.
    edf : float
        Effective degrees of freedom.
    B : array, shape (n, n_basis)
        Basis matrix.
    S : array, shape (n_basis, n_basis)
        Penalty matrix.
    """
    from statgpu.nonparametric.splines._bspline_basis import bspline_basis

    xp = _get_xp(xp)

    x = xp.asarray(x, dtype=xp.float64).ravel()
    y = xp.asarray(y, dtype=xp.float64).ravel()

    # Build basis matrix
    B = bspline_basis(x, knots, degree=degree, xp=xp)

    # Build penalty matrix
    n_basis = B.shape[1]
    S = difference_penalty(penalty_order, n_basis, xp)

    # Solve penalized least squares
    beta, edf = penalized_ls(B, y, S, lambda_, xp)

    return beta, edf, B, S


def predict_penalized_spline(x_new, beta, knots, degree=3, xp=None,
                             boundary_lo=None, boundary_hi=None):
    """
    Predict using a fitted penalized spline.

    Parameters
    ----------
    x_new : array-like, shape (n_new,)
        New predictor values.
    beta : array, shape (n_basis,)
        Fitted spline coefficients.
    knots : array-like, shape (m,)
        Interior knots used for fitting.
    degree : int, default=3
        Spline degree.
    xp : module, optional
        Array module.
    boundary_lo : float, optional
        Lower boundary knot (from training data). Required for small batches.
    boundary_hi : float, optional
        Upper boundary knot (from training data). Required for small batches.

    Returns
    -------
    y_pred : array, shape (n_new,)
        Predicted values.
    """
    from statgpu.nonparametric.splines._bspline_basis import bspline_basis

    xp = _get_xp(xp)

    x_new = xp.asarray(x_new, dtype=xp.float64).ravel()
    beta = xp.asarray(beta, dtype=xp.float64)

    # Build basis matrix for new points, using training boundaries
    B_new = bspline_basis(x_new, knots, degree=degree, xp=xp,
                          boundary_lo=boundary_lo, boundary_hi=boundary_hi)

    # Predict
    y_pred = B_new @ beta

    return y_pred
