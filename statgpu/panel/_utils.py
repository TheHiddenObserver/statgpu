"""
Panel data utility functions.

Provides demeaning / within-transformation routines used by fixed effects
and random effects estimators.  All functions accept an ``xp`` module
(numpy / cupy / torch) so they work on any backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from statgpu.backends import xp_asarray, xp_zeros, _to_numpy


def within_transform(y, groups, xp=None):
    """Remove group means (fixed-effect projection).

    Computes ``y_within[i] = y[i] - mean(y[groups == g[i]])`` for every
    observation.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome vector.
    groups : array-like, shape (n,)
        Integer group labels.
    xp : module, optional
        Array module (numpy / cupy / torch).  Defaults to numpy.

    Returns
    -------
    y_within : array, shape (n,)
        Demeaned outcome.
    """
    if xp is None:
        xp = np

    y = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
    groups = xp_asarray(groups, xp=xp, ref_arr=y).ravel()

    # Vectorized group demeaning — avoids per-group Python loop
    # (O(n_groups) kernel launches → O(1) vectorized ops)
    # Do NOT cast to int64 before unique — float/string labels would be
    # truncated/crashed.  np.unique(..., return_inverse=True) already returns
    # contiguous integer indices regardless of the input dtype.
    groups_np = _to_numpy(groups)
    y_np = _to_numpy(y)

    # Map group labels to contiguous indices [0, n_groups)
    unique_labels, group_idx = np.unique(groups_np, return_inverse=True)
    n_groups = len(unique_labels)

    # Compute group means via bincount (O(n), single pass)
    group_sum = np.bincount(group_idx, weights=y_np, minlength=n_groups)
    group_count = np.bincount(group_idx, minlength=n_groups)
    group_mean = group_sum / np.maximum(group_count, 1)

    # Subtract group means
    result_np = y_np - group_mean[group_idx]

    # Convert back to original backend
    if hasattr(y, 'clone'):  # torch
        import torch
        result = torch.from_numpy(result_np).to(dtype=y.dtype, device=y.device)
    elif hasattr(y, 'get'):  # cupy
        import cupy as cp
        result = cp.asarray(result_np)
    else:
        result = result_np

    return result


def _within_transform_matrix(X, groups, xp=None):
    """Group-demean all columns of X at once (vectorized, no per-column loop).

    Parameters
    ----------
    X : array-like, shape (n, k)
        Regressor matrix.
    groups : array-like, shape (n,)
        Group labels.
    xp : module, optional
        Array module.  Defaults to numpy.

    Returns
    -------
    X_demeaned : array, shape (n, k)
        Group-demeaned regressors.
    """
    if xp is None:
        xp = np

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    groups_np = _to_numpy(groups)
    X_np = _to_numpy(X)

    unique_labels, group_idx = np.unique(groups_np, return_inverse=True)
    n_groups = len(unique_labels)

    # Vectorized group means for all columns: (n_groups, k)
    # group_idx is (n,), X_np is (n, k)
    group_sum = np.zeros((n_groups, X_np.shape[1]), dtype=np.float64)
    np.add.at(group_sum, group_idx, X_np)
    group_count = np.bincount(group_idx, minlength=n_groups).reshape(-1, 1)
    group_mean = group_sum / np.maximum(group_count, 1)

    result_np = X_np - group_mean[group_idx]

    # Convert back to original backend
    if hasattr(X, 'clone'):  # torch
        import torch
        result = torch.from_numpy(result_np).to(dtype=X.dtype, device=X.device)
    elif hasattr(X, 'get'):  # cupy
        import cupy as cp
        result = cp.asarray(result_np)
    else:
        result = result_np

    return result


def make_group_dummies(groups, xp=None):
    """Create dummy variable matrix from group labels.

    Parameters
    ----------
    groups : array-like, shape (n,)
        Integer group labels.
    xp : module, optional
        Array module.  Defaults to numpy.

    Returns
    -------
    D : array, shape (n, n_groups)
        Dummy matrix with ones indicating group membership.
    """
    if xp is None:
        xp = np

    groups = xp_asarray(groups, xp=xp).ravel()
    unique = xp.unique(groups)
    n = len(groups)
    n_groups = len(unique)
    # Batch-transfer unique group values to CPU (single sync)
    unique_cpu = _to_numpy(unique).tolist()

    D = xp_zeros((n, n_groups), xp.float64, xp, groups)
    for i, g_val in enumerate(unique_cpu):
        mask = groups == g_val
        D[mask, i] = 1.0

    return D


def demean_variables(y, X, entity_ids, time_ids=None, xp=None):
    """Demean *y* and *X* for fixed-effects estimation.

    If *time_ids* is also provided, performs two-way demeaning (entity
    and time effects).

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome vector.
    X : array-like, shape (n, k)
        Regressor matrix.
    entity_ids : array-like, shape (n,)
        Entity (individual) identifiers.
    time_ids : array-like, shape (n,), optional
        Time-period identifiers.  If provided, two-way demeaning is applied.
    xp : module, optional
        Array module.  Defaults to numpy.

    Returns
    -------
    y_d : array, shape (n,)
        Demeaned outcome.
    X_d : array, shape (n, k)
        Demeaned regressors.
    """
    if xp is None:
        xp = np

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Demean: for two-way FE with unbalanced panels, use iterative
    # alternating demeaning (entity then time) until convergence.
    # For one-way FE or balanced panels, a single pass suffices.
    y_d = y.copy() if hasattr(y, 'copy') else y - 0.0
    X_d = X.copy() if hasattr(X, 'copy') else X - 0.0

    if entity_ids is not None and time_ids is not None:
        # Two-way FE: iterate until convergence
        for _ in range(50):
            y_prev = y_d.copy() if hasattr(y_d, 'copy') else y_d - 0.0
            X_prev = X_d.copy() if hasattr(X_d, 'copy') else X_d - 0.0
            # Entity demean
            y_d = within_transform(y_d, entity_ids, xp)
            X_d = _within_transform_matrix(X_d, entity_ids, xp)
            # Time demean
            y_d = within_transform(y_d, time_ids, xp)
            X_d = _within_transform_matrix(X_d, time_ids, xp)
            # Check convergence on both y and X
            delta_y = float(xp.max(xp.abs(y_d - y_prev)))
            delta_X = float(xp.max(xp.abs(X_d - X_prev)))
            if max(delta_y, delta_X) < 1e-10:
                break
    else:
        # One-way FE: single pass
        if entity_ids is not None:
            y_d = within_transform(y_d, entity_ids, xp)
            X_d = _within_transform_matrix(X_d, entity_ids, xp)
        if time_ids is not None:
            y_d = within_transform(y_d, time_ids, xp)
            X_d = _within_transform_matrix(X_d, time_ids, xp)

    return y_d, X_d


def group_means(y, groups, xp=None):
    """Compute group-level means aligned to each observation.

    Returns an array of shape (n,) where element *i* is the mean of *y*
    over all observations belonging to the same group as observation *i*.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome vector.
    groups : array-like, shape (n,)
        Group labels.
    xp : module, optional
        Array module.  Defaults to numpy.

    Returns
    -------
    y_bar : array, shape (n,)
        Group means aligned to each observation.
    """
    if xp is None:
        xp = np

    y = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
    groups = xp_asarray(groups, xp=xp, ref_arr=y).ravel()

    # Vectorized group means using bincount (O(n), no per-group loop)
    groups_np = _to_numpy(groups)
    y_np = _to_numpy(y)
    unique_labels, group_idx = np.unique(groups_np, return_inverse=True)
    n_groups = len(unique_labels)
    group_sum = np.bincount(group_idx, weights=y_np, minlength=n_groups)
    group_count = np.bincount(group_idx, minlength=n_groups)
    group_mean = group_sum / np.maximum(group_count, 1)
    result_np = group_mean[group_idx]

    # Convert back to original backend
    if hasattr(y, 'clone'):  # torch
        import torch
        result = torch.from_numpy(result_np).to(dtype=y.dtype, device=y.device)
    elif hasattr(y, 'get'):  # cupy
        import cupy as cp
        result = cp.asarray(result_np)
    else:
        result = result_np

    return result


def group_sizes(groups, xp=None):
    """Return an array of per-observation group sizes.

    Element *i* is the number of observations in the group of
    observation *i*.

    Parameters
    ----------
    groups : array-like, shape (n,)
        Group labels.
    xp : module, optional
        Array module.  Defaults to numpy.

    Returns
    -------
    T_i : array, shape (n,)
        Group size for each observation.
    """
    if xp is None:
        xp = np

    groups = xp_asarray(groups, xp=xp).ravel()

    # Vectorized group sizes using bincount (O(n), no per-group loop)
    groups_np = _to_numpy(groups)
    unique_labels, group_idx = np.unique(groups_np, return_inverse=True)
    n_groups = len(unique_labels)
    count = np.bincount(group_idx, minlength=n_groups)
    result_np = count[group_idx].astype(np.float64)

    # Convert back to original backend
    if hasattr(groups, 'clone'):  # torch
        import torch
        result = torch.from_numpy(result_np).to(dtype=groups.dtype, device=groups.device)
    elif hasattr(groups, 'get'):  # cupy
        import cupy as cp
        result = cp.asarray(result_np)
    else:
        result = result_np

    return result


def ols_inference_nonrobust(params, X, scale, df, alpha=0.05):
    """Compute non-robust OLS inference (SE, t, p, CI).

    Parameters
    ----------
    params : ndarray, shape (k,)
        Estimated coefficients.
    X : ndarray, shape (n, k)
        Design matrix (numpy).
    scale : float
        Residual variance (RSS / df).
    df : int
        Residual degrees of freedom.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    bse, tvalues, pvalues, conf_int : ndarrays
    """
    from scipy import stats

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    cov_params = scale * XtX_inv
    bse = np.sqrt(np.diag(cov_params))
    tvalues = params / (bse + 1e-30)
    pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), df))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    conf_int = np.column_stack([
        params - t_crit * bse,
        params + t_crit * bse,
    ])
    return bse, tvalues, pvalues, conf_int
