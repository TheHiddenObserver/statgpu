"""
Clustered covariance estimators for panel data models.

Implements one-way and two-way clustered standard errors following
Cameron & Miller (2015) and Cameron, Gelbach & Miller (2011).
"""
from __future__ import annotations

__all__ = ["clustered_covariance", "two_way_clustered_covariance", "hac_covariance"]

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
    # Vectorized: scores = X * e, then scatter_add per cluster, then S'@S
    scores = X * resid[:, None]  # (n, k)

    # Factorize cluster labels to contiguous indices
    clusters_np = _to_numpy(clusters)
    unique_labels, cluster_idx = np.unique(clusters_np, return_inverse=True)
    n_clusters = len(unique_labels)
    cluster_idx_xp = xp_asarray(cluster_idx, dtype=xp.int64, xp=xp, ref_arr=X)

    # Scatter-add: sum scores per cluster → S shape (n_clusters, k)
    S = xp_zeros((n_clusters, k), dtype=xp.float64, xp=xp, ref_arr=X)
    if hasattr(S, 'scatter_add_'):
        # torch
        S.scatter_add_(0, cluster_idx_xp.unsqueeze(1).expand_as(scores), scores)
    elif hasattr(S, 'device') and not hasattr(S, 'get'):
        # cupy — fall back to numpy loop
        S_np = np.zeros((n_clusters, k), dtype=np.float64)
        np.add.at(S_np, cluster_idx, _to_numpy(scores))
        S = xp_asarray(S_np, dtype=xp.float64, xp=xp, ref_arr=X)
    else:
        # numpy
        np.add.at(S, cluster_idx, scores)

    # meat = S' @ S  (k, k)
    meat = S.T @ S

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
    # Vectorized Cantor-pair hash: s = c1 + c2, hash = s*(s+1)/2 + c2
    s = c1.astype(np.int64) + c2.astype(np.int64)
    combined_np = s * (s + 1) // 2 + c2.astype(np.int64)
    combined = xp_asarray(combined_np, dtype=xp.int64, xp=xp, ref_arr=V1)

    V12 = clustered_covariance(X, resid, combined, xp)
    return V1 + V2 - V12


def hac_covariance(X, resid, bandwidth=None, kernel="bartlett", xp=None):
    """Heteroskedasticity and Autocorrelation Consistent (HAC) covariance.

    Implements the Newey-West (1987) HAC estimator with Bartlett kernel:

        V = (X'X/n)^{-1} @ Omega_hat @ (X'X/n)^{-1}

    where ``Omega_hat = Gamma_0 + sum_{h=1}^{bw} w(h) (Gamma_h + Gamma_h')``
    and ``Gamma_h = (1/n) sum_i (x_i e_i)(x_{i-h} e_{i-h})'``.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Design matrix (must be sorted by time within each entity).
    resid : array-like, shape (n,)
        OLS residuals.
    bandwidth : int or None, default=None
        Bandwidth (number of lags).  If None, uses ``floor(4 * (n/100)^{2/9})``
        (Newey-West 1994 rule of thumb).
    kernel : str, default='bartlett'
        Kernel function.  Currently only ``'bartlett'`` is supported.
    xp : module, optional
        Array module (numpy / cupy / torch).  Defaults to numpy.

    Returns
    -------
    V : array, shape (k, k)
        HAC covariance matrix of the coefficient estimates.

    References
    ----------
    Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    *Econometrica*, 55(3), 703-708.
    """
    xp = _ensure_xp(xp)

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()

    n, k = X.shape

    # Default bandwidth: Newey-West (1994) rule
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(bandwidth, n - 1))

    # Bread: (X'X / n)^{-1}
    XtX = X.T @ X / n
    try:
        bread = xp.linalg.inv(XtX)
    except _LINALG_ERRORS:
        bread = xp.linalg.pinv(XtX)

    # Meat: Omega_hat
    # Score matrix: s_i = x_i * e_i, shape (n, k)
    scores = X * resid[:, None]  # (n, k)

    # Gamma_0 = (1/n) * sum_i s_i s_i' = scores' @ scores / n
    meat = scores.T @ scores / n

    # Gamma_h for h = 1..bandwidth
    for h in range(1, bandwidth + 1):
        # Bartlett kernel weight: w(h) = 1 - h/(bandwidth+1)
        w = 1.0 - h / (bandwidth + 1.0)

        # Gamma_h = (1/n) * sum_{i=h}^{n-1} s_i s_{i-h}'
        Gamma_h = scores[h:].T @ scores[:n - h] / n

        meat = meat + w * (Gamma_h + Gamma_h.T)

    # Sandwich: V = bread @ meat @ bread / n
    V = bread @ meat @ bread / n
    return V
