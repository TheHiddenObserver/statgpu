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

    unique_groups = xp.unique(groups)
    # Batch-transfer unique group values to CPU (single sync, not per-group)
    unique_cpu = _to_numpy(unique_groups).tolist()
    result = xp.copy(y) if hasattr(xp, 'copy') else y.clone() if hasattr(y, 'clone') else y - 0.0

    for g_val in unique_cpu:
        mask = groups == g_val
        group_mean = xp.mean(y[mask])
        if hasattr(result, '__setitem__'):
            result[mask] = y[mask] - group_mean
        else:
            result = xp.where(mask, y - group_mean, result)

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

    # Entity demeaning (skip if entity_ids is None)
    if entity_ids is not None:
        y_d = within_transform(y, entity_ids, xp)
        X_d = xp.zeros_like(X)
        for j in range(X.shape[1]):
            X_d[:, j] = within_transform(X[:, j], entity_ids, xp)
    else:
        y_d = y.copy() if hasattr(y, 'copy') else y - 0.0
        X_d = X.copy() if hasattr(X, 'copy') else X - 0.0

    # Time demeaning (two-way FE, or time-only FE)
    if time_ids is not None:
        y_d = within_transform(y_d, time_ids, xp)
        for j in range(X.shape[1]):
            X_d[:, j] = within_transform(X_d[:, j], time_ids, xp)

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

    result = xp.zeros_like(y)
    # Batch-transfer unique group values to CPU (single sync)
    unique_cpu = _to_numpy(xp.unique(groups)).tolist()
    for g_val in unique_cpu:
        mask = groups == g_val
        result[mask] = xp.mean(y[mask])

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
    result = xp_zeros(len(groups), xp.float64, xp, groups)
    # Batch-transfer unique group values to CPU (single sync)
    unique_cpu = _to_numpy(xp.unique(groups)).tolist()

    for g_val in unique_cpu:
        mask = groups == g_val
        result[mask] = float(xp.sum(mask))

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
