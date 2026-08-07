"""
Covariance estimators and internal dispatch for panel data models.

The public one-way/two-way cluster and HAC functions retain their existing
contracts. Stage A of issue #93 adds ``ols_covariance`` as an internal
behavior-preserving dispatcher for residual-based OLS/transformed-OLS models.
"""
from __future__ import annotations

__all__ = [
    "clustered_covariance",
    "two_way_clustered_covariance",
    "hac_covariance",
    "ols_covariance",
]

from typing import Optional

import numpy as np

from statgpu.backends import (
    _LINALG_ERRORS,
    _to_numpy,
    xp_asarray,
    xp_zeros,
)


def _ensure_xp(xp=None):
    """Return the array module, defaulting to numpy."""
    return xp if xp is not None else np


def clustered_covariance(X, resid, clusters, xp=None):
    """One-way clustered robust covariance matrix.

    Implements the cluster-robust sandwich estimator::

        V = (X'X/n)^{-1} @ meat @ (X'X/n)^{-1} / n^2

    where ``meat = sum_g (X_g' e_g)(X_g' e_g)'``.
    """
    xp = _ensure_xp(xp)

    clusters_np = np.asarray(_to_numpy(clusters)).ravel()
    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    n, k = X.shape
    if resid.shape[0] != n or clusters_np.shape[0] != n:
        raise ValueError("X, resid, and clusters must have the same number of observations")

    XtX = X.T @ X / n
    try:
        bread = xp.linalg.inv(XtX)
    except _LINALG_ERRORS:
        bread = xp.linalg.pinv(XtX)

    scores = X * resid[:, None]
    unique_labels, cluster_idx = np.unique(clusters_np, return_inverse=True)
    n_clusters = len(unique_labels)
    cluster_idx_xp = xp_asarray(cluster_idx, dtype=xp.int64, xp=xp, ref_arr=X)

    S = xp_zeros((n_clusters, k), dtype=xp.float64, xp=xp, ref_arr=X)
    if hasattr(S, "scatter_add_"):
        S.scatter_add_(0, cluster_idx_xp.unsqueeze(1).expand_as(scores), scores)
    elif type(S).__module__.startswith("cupy"):
        xp.add.at(S, cluster_idx_xp, scores)
    else:
        np.add.at(S, cluster_idx, scores)

    meat = S.T @ S
    return bread @ meat @ bread / (n * n)


def two_way_clustered_covariance(X, resid, cluster1, cluster2, xp=None):
    """Two-way clustered covariance with intersection correction."""
    xp = _ensure_xp(xp)

    V1 = clustered_covariance(X, resid, cluster1, xp)
    V2 = clustered_covariance(X, resid, cluster2, xp)

    c1_raw = np.asarray(_to_numpy(cluster1)).ravel()
    c2_raw = np.asarray(_to_numpy(cluster2)).ravel()
    n = int(X.shape[0])
    if c1_raw.shape[0] != n or c2_raw.shape[0] != n:
        raise ValueError("cluster arrays must match the number of observations")
    _, c1 = np.unique(c1_raw, return_inverse=True)
    _, c2 = np.unique(c2_raw, return_inverse=True)
    s = c1.astype(np.int64) + c2.astype(np.int64)
    combined_np = s * (s + 1) // 2 + c2.astype(np.int64)
    combined = xp_asarray(combined_np, dtype=xp.int64, xp=xp, ref_arr=V1)

    V12 = clustered_covariance(X, resid, combined, xp)
    return V1 + V2 - V12


def hac_covariance(X, resid, bandwidth=None, kernel="bartlett", xp=None):
    """Newey-West HAC covariance using the Bartlett kernel."""
    xp = _ensure_xp(xp)
    if str(kernel).lower() != "bartlett":
        raise ValueError("kernel must be 'bartlett'")
    if bandwidth is not None:
        if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, np.integer)):
            raise ValueError("bandwidth must be a non-negative integer or None")
        if int(bandwidth) < 0:
            raise ValueError("bandwidth must be a non-negative integer or None")

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()

    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n, _ = X.shape

    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(bandwidth, n - 1))

    XtX = X.T @ X / n
    try:
        bread = xp.linalg.inv(XtX)
    except _LINALG_ERRORS:
        bread = xp.linalg.pinv(XtX)

    scores = X * resid[:, None]
    meat = scores.T @ scores / n

    for h in range(1, bandwidth + 1):
        w = 1.0 - h / (bandwidth + 1.0)
        gamma_h = scores[h:].T @ scores[: n - h] / n
        meat = meat + w * (gamma_h + gamma_h.T)

    return bread @ meat @ bread / n


def ols_covariance(
    X,
    resid,
    *,
    cov_type,
    scale=None,
    df_resid=None,
    cluster=None,
    bandwidth=None,
    kernel="bartlett",
    xp=None,
    allowed=None,
    hc1_correction=None,
):
    """Dispatch existing residual-based panel covariance definitions.

    This helper deliberately does not infer model-specific degrees of freedom.
    Callers pass ``scale`` and ``df_resid`` (or an explicit HC1 correction) so
    Stage A cannot silently harmonize distinct panel-model conventions.
    """
    xp = _ensure_xp(xp)
    name = str(cov_type).lower()
    if allowed is not None and name not in {str(value).lower() for value in allowed}:
        choices = ", ".join(sorted(str(value) for value in allowed))
        raise ValueError(f"cov_type={cov_type!r} is not supported here; expected one of: {choices}")

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])

    if name == "nonrobust":
        if scale is None:
            raise ValueError("scale is required for nonrobust covariance")
        XtX = X.T @ X
        try:
            bread = xp.linalg.inv(XtX)
        except _LINALG_ERRORS:
            bread = xp.linalg.pinv(XtX)
        return float(scale) * bread

    if name == "robust":
        XtX = X.T @ X
        try:
            bread = xp.linalg.inv(XtX)
        except _LINALG_ERRORS:
            bread = xp.linalg.pinv(XtX)
        scores = X * resid[:, None]
        meat = scores.T @ scores
        correction = hc1_correction
        if correction is None:
            if df_resid is None or int(df_resid) <= 0:
                raise ValueError("positive df_resid or hc1_correction is required for robust covariance")
            correction = n / float(df_resid)
        return bread @ meat @ bread * float(correction)

    if name == "clustered":
        if cluster is None:
            raise ValueError("cluster is required for cov_type='clustered'")
        cluster_np = np.asarray(_to_numpy(cluster))
        if cluster_np.ndim == 2 and cluster_np.shape[1] == 2:
            return two_way_clustered_covariance(
                X, resid, cluster_np[:, 0], cluster_np[:, 1], xp=xp
            )
        # Preserve the historical PanelOLS one-way behavior for a column-vector
        # cluster array: clustered_covariance() ravelled an (n, 1) input.
        if cluster_np.ndim == 2 and cluster_np.shape[1] == 1:
            cluster_np = cluster_np[:, 0]
        if cluster_np.ndim != 1:
            raise ValueError("cluster must be one-dimensional, (n, 1), or (n, 2)")
        return clustered_covariance(X, resid, cluster_np, xp=xp)

    if name == "hac":
        return hac_covariance(
            X, resid, bandwidth=bandwidth, kernel=kernel, xp=xp
        )

    raise ValueError(
        "cov_type must be one of 'nonrobust', 'robust', 'clustered', or 'hac'"
    )
