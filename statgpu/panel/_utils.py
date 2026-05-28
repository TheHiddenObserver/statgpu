"""
Panel data utility functions.

Provides demeaning / within-transformation routines used by fixed effects
and random effects estimators.  All functions accept an ``xp`` module
(numpy / cupy / torch) so they work on any backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from statgpu.backends import xp_asarray, xp_zeros


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
    result = xp.copy(y) if hasattr(xp, 'copy') else y.clone() if hasattr(y, 'clone') else y - 0.0

    for g in unique_groups:
        # Works for numpy, cupy, and torch tensors
        g_val = g.item() if hasattr(g, 'item') else g
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

    D = xp_zeros((n, n_groups), xp.float64, xp, groups)
    for i, g in enumerate(unique):
        g_val = g.item() if hasattr(g, 'item') else g
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

    # Entity demeaning
    y_d = within_transform(y, entity_ids, xp)
    X_d = xp.zeros_like(X)
    for j in range(X.shape[1]):
        X_d[:, j] = within_transform(X[:, j], entity_ids, xp)

    # Time demeaning (two-way FE)
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
    for g in xp.unique(groups):
        g_val = g.item() if hasattr(g, 'item') else g
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

    for g in xp.unique(groups):
        g_val = g.item() if hasattr(g, 'item') else g
        mask = groups == g_val
        result[mask] = float(xp.sum(mask))

    return result
