"""
Clustered covariance estimators for panel data models.

Implements one-way and two-way clustered standard errors following
Cameron & Miller (2015) and Cameron, Gelbach & Miller (2011).
"""
from __future__ import annotations

__all__ = ["clustered_covariance", "two_way_clustered_covariance"]

from typing import Optional

import numpy as np

from statgpu.backends import (
    _LINALG_ERRORS,
    _get_torch_device_str,
    _torch_dev,
    _to_numpy,
    xp_asarray,
    xp_zeros,
)


def _ensure_xp(xp=None):
    """Return the array module, defaulting to numpy."""
    return xp if xp is not None else np


def clustered_covariance(X, resid, clusters, xp=None):
    """One-way clustered robust covariance matrix.

    Implements the cluster-robust sandwich estimator:

        V = (X'X/n)^{-1} @ meat @ (X'X/n)^{-1}

    where ``meat = sum_g (X_g' e_g)(X_g' e_g)'`` summed over clusters.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Design matrix (including intercept if applicable).
    resid : array-like, shape (n,)
        OLS residuals.
    clusters : array-like, shape (n,)
        Cluster assignment labels (integer or categorical).
    xp : module, optional
        Array module (numpy / cupy / torch).  Defaults to numpy.

    Returns
    -------
    V : array, shape (k, k)
        Cluster-robust covariance matrix of the coefficient estimates.
    """
    xp = _ensure_xp(xp)

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    clusters = xp_asarray(clusters, xp=xp, ref_arr=X).ravel()

    n, k = X.shape

    # Bread: (X'X / n)^{-1}
    XtX = X.T @ X / n
    try:
        bread = xp.linalg.inv(XtX)
    except _LINALG_ERRORS:
        bread = xp.linalg.pinv(XtX)

    # Meat: sum over clusters of (X_g' e_g)(X_g' e_g)'
    # Batch-transfer unique cluster values to CPU (single sync, not per-cluster)
    unique_clusters_cpu = _to_numpy(xp.unique(clusters)).tolist()
    meat = xp_zeros((k, k), xp.float64, xp, X)
    for g_val in unique_clusters_cpu:
        mask = clusters == g_val
        Xg = X[mask]
        eg = resid[mask]
        Xe = Xg.T @ eg  # shape (k,)
        meat = meat + xp.outer(Xe, Xe)

    # Sandwich: V = bread @ meat @ bread / n^2
    V = bread @ meat @ bread / (n * n)
    return V


def two_way_clustered_covariance(X, resid, cluster1, cluster2, xp=None):
    """Two-way clustered robust covariance matrix.

    Implements the Cameron, Gelbach & Miller (2011) intersection
    correction::

        V = V_cluster1 + V_cluster2 - V_intersection

    where the intersection clusters are formed from all unique
    ``(cluster1, cluster2)`` pairs.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Design matrix.
    resid : array-like, shape (n,)
        OLS residuals.
    cluster1 : array-like, shape (n,)
        First cluster dimension (e.g. entity).  Accepts integer or
        categorical labels (will be factorized to integers internally).
    cluster2 : array-like, shape (n,)
        Second cluster dimension (e.g. time).  Same as cluster1.
    xp : module, optional
        Array module (numpy / cupy / torch).  Defaults to numpy.

    Returns
    -------
    V : array, shape (k, k)
        Two-way cluster-robust covariance matrix.
    """
    xp = _ensure_xp(xp)

    V1 = clustered_covariance(X, resid, cluster1, xp)
    V2 = clustered_covariance(X, resid, cluster2, xp)

    # Intersection clusters: unique (c1, c2) pairs via Cantor-pair hash
    # Factorize labels to integers (supports string/categorical labels)
    c1_raw = _to_numpy(xp_asarray(cluster1, xp=xp, ref_arr=V1).ravel())
    c2_raw = _to_numpy(xp_asarray(cluster2, xp=xp, ref_arr=V1).ravel())
    _, c1 = np.unique(c1_raw, return_inverse=True)
    _, c2 = np.unique(c2_raw, return_inverse=True)
    # Use Python int for Cantor-pair to avoid int64 overflow with
    # large cluster counts (>~3 billion unique combinations).
    c1_int = [int(x) for x in c1]
    c2_int = [int(x) for x in c2]
    combined_np = np.array(
        [s * (s + 1) // 2 + c2i for s, c2i in zip(
            [a + b for a, b in zip(c1_int, c2_int)], c2_int
        )],
        dtype=np.int64,
    )
    combined = xp_asarray(combined_np, dtype=xp.int64, xp=xp, ref_arr=V1)

    V12 = clustered_covariance(X, resid, combined, xp)
    return V1 + V2 - V12
