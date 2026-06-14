"""
Panel data utility functions.

Provides demeaning / within-transformation routines used by fixed effects
and random effects estimators.  All functions accept an ``xp`` module
(numpy / cupy / torch) so they work on any backend.

Performance note: all group-level operations use scatter-add to compute
group sums and counts in a single kernel launch, avoiding per-group
Python loops and their associated GPU-CPU synchronization overhead.
"""

from __future__ import annotations

__all__ = ["demean", "within_transform", "group_means"]

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from statgpu.backends import xp_asarray, xp_copy, xp_ones, xp_zeros, _to_float_scalar, _to_numpy


@dataclass
class PanelSummary:
    """Structured result container for panel model summaries.

    Attributes
    ----------
    model_type : str
        ``'PanelOLS'`` or ``'RandomEffects'``.
    nobs : int
        Number of observations.
    df_resid : int
        Residual degrees of freedom.
    coef : ndarray, shape (k,)
        Estimated coefficients.
    bse : ndarray, shape (k,)
        Standard errors.
    tvalues : ndarray, shape (k,)
        t-statistics.
    pvalues : ndarray, shape (k,)
        Two-sided p-values.
    conf_int : ndarray, shape (k, 2)
        Confidence intervals.
    feature_names : list of str
        Feature names (auto-generated as ``x1, x2, ...`` if not provided).
    rsquared_within : float or None
        Within R-squared (PanelOLS only).
    cov_type : str or None
        Covariance type (PanelOLS only).
    entity_effects : bool or None
        Whether entity effects were included (PanelOLS only).
    time_effects : bool or None
        Whether time effects were included (PanelOLS only).
    variance_components : dict or None
        ``{'sigma2_e': float, 'sigma2_a': float}`` (RandomEffects only).
    theta : float or None
        GLS transformation parameter (RandomEffects only).
    alpha : float
        Significance level for confidence intervals.
    extra : dict
        Additional model-specific metadata.
    """

    model_type: str
    nobs: int
    df_resid: int
    coef: np.ndarray
    bse: np.ndarray
    tvalues: np.ndarray
    pvalues: np.ndarray
    conf_int: np.ndarray
    feature_names: List[str]
    rsquared_within: Optional[float] = None
    cov_type: Optional[str] = None
    entity_effects: Optional[bool] = None
    time_effects: Optional[bool] = None
    variance_components: Optional[Dict[str, float]] = None
    theta: Optional[float] = None
    alpha: float = 0.05
    extra: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        """Formatted text table."""
        lines = []
        lines.append("=" * 72)
        lines.append(f"{'':>20}{self.model_type} Results")
        lines.append("=" * 72)

        if self.entity_effects is not None:
            lines.append(f"Entity effects:     {str(self.entity_effects):>10}")
        if self.time_effects is not None:
            lines.append(f"Time effects:       {str(self.time_effects):>10}")
        if self.cov_type is not None:
            lines.append(f"Covariance type:    {self.cov_type:>10}")
        lines.append(f"No. Observations:   {self.nobs:>10}")
        lines.append(f"Degrees of Freedom: {self.df_resid:>10}")
        if self.rsquared_within is not None:
            lines.append(f"Within R-squared:   {self.rsquared_within:>10.4f}")
        if self.variance_components is not None:
            lines.append(f"sigma2_e:           {self.variance_components['sigma2_e']:>10.6f}")
            lines.append(f"sigma2_a:           {self.variance_components['sigma2_a']:>10.6f}")
        if self.theta is not None:
            lines.append(f"theta (avg):        {self.theta:>10.4f}")

        ci_label = f"[{self.alpha/2:.3f}" if self.alpha != 0.05 else "[0.025"
        ci_label2 = f"{1-self.alpha/2:.3f}]" if self.alpha != 0.05 else "0.975]"
        lines.append("-" * 72)
        lines.append(f"{'':<12} {'coef':>10} {'std err':>10} {'t':>8} {'P>|t|':>10} {ci_label:>10} {ci_label2:>10}")
        lines.append("-" * 72)
        for i, name in enumerate(self.feature_names):
            lines.append(
                f"{name:<12} {self.coef[i]:>10.4f} {self.bse[i]:>10.4f} "
                f"{self.tvalues[i]:>8.3f} {self.pvalues[i]:>10.4f} "
                f"{self.conf_int[i, 0]:>10.4f} {self.conf_int[i, 1]:>10.4f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Return a JSON-serializable dictionary."""
        return {
            'model_type': self.model_type,
            'nobs': self.nobs,
            'df_resid': self.df_resid,
            'coef': self.coef.tolist(),
            'bse': self.bse.tolist(),
            'tvalues': self.tvalues.tolist(),
            'pvalues': self.pvalues.tolist(),
            'conf_int': self.conf_int.tolist(),
            'feature_names': self.feature_names,
            'rsquared_within': self.rsquared_within,
            'cov_type': self.cov_type,
            'entity_effects': self.entity_effects,
            'time_effects': self.time_effects,
            'variance_components': self.variance_components,
            'theta': self.theta,
            'alpha': self.alpha,
        }


def _scatter_add(xp, indices, values, n_groups):
    """Scatter-add values into bins defined by indices.

    Returns an array ``out`` of shape ``(n_groups,)`` where
    ``out[j] = sum(values[indices == j])``.

    Works across NumPy, CuPy, and PyTorch with a single kernel launch.
    """
    if hasattr(xp, 'scatter_add'):
        # PyTorch: scatter_add(dim, index, src)
        out = xp.zeros(n_groups, dtype=values.dtype, device=values.device)
        out.scatter_add_(0, indices.long(), values)
        return out
    elif hasattr(xp, 'add') and hasattr(xp, 'zeros') and xp.__name__ == 'cupy':
        # CuPy: use cupyx.scatter_add or cp.add.at
        try:
            out = xp.zeros(n_groups, dtype=values.dtype)
            from cupyx import scatter_add as _scatter_add_cu
            _scatter_add_cu(out, indices, values)
            return out
        except ImportError:
            # Fallback: compute on CPU then transfer back to GPU
            out_np = np.zeros(n_groups, dtype=values.dtype)
            np.add.at(out_np, _to_numpy(indices), _to_numpy(values))
            return xp.asarray(out_np)
    else:
        # NumPy: np.add.at
        out = np.zeros(n_groups, dtype=values.dtype)
        np.add.at(out, _to_numpy(indices), _to_numpy(values))
        return out


def _remap_to_contiguous(groups, xp):
    """Remap group labels to contiguous 0..n_groups-1 indices.

    Returns (indices, n_groups, unique_labels) where indices[i] is the
    contiguous index of group groups[i].
    """
    groups_np = _to_numpy(groups).ravel()
    unique_labels, indices_np = np.unique(groups_np, return_inverse=True)
    n_groups = len(unique_labels)
    indices = xp_asarray(indices_np, dtype=xp.int64, xp=xp, ref_arr=groups)
    return indices, n_groups, unique_labels


def within_transform(y, groups, xp=None):
    """Remove group means (fixed-effect projection).

    Computes ``y_within[i] = y[i] - mean(y[groups == g[i]])`` for every
    observation.  Uses scatter-add for a single-kernel group reduction
    instead of per-group Python loops.

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

    # Remap groups to contiguous indices (single CPU sync for unique)
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)

    # Group sums and counts via scatter-add (2 kernel launches total)
    group_sums = _scatter_add(xp, idx, y, n_groups)
    group_counts = _scatter_add(xp, idx, xp.ones_like(y), n_groups)

    # Group means (element-wise, no loop)
    group_means = group_sums / xp.maximum(group_counts, 1.0)

    # Broadcast back: y_within = y - group_means[idx]
    return y - group_means[idx]


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
    n = len(groups)
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)

    # Build dummy matrix using advanced indexing (no per-group loop)
    D = xp_zeros((n, n_groups), xp.float64, xp, groups)
    row_idx = xp.arange(n, device=getattr(groups, 'device', None)
                        if hasattr(groups, 'device') else None)
    D[row_idx, idx] = 1.0

    return D


def _within_transform_matrix(M, groups, xp):
    """Remove group means from each column of matrix M (batched).

    Uses scatter-add on the full matrix to compute all column-group
    means in one pass, instead of looping over columns.

    Parameters
    ----------
    M : array, shape (n, k)
        Input matrix.
    groups : array, shape (n,)
        Integer group labels.
    xp : module
        Array module.

    Returns
    -------
    M_within : array, shape (n, k)
        Column-demeaned matrix.
    """
    n, k = M.shape
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)

    # Compute group counts once (n_groups,) — reuse across all columns
    ones_col = xp_ones(n, M.dtype, xp, M)
    group_counts = _scatter_add(xp, idx, ones_col, n_groups)
    inv_counts = 1.0 / xp.maximum(group_counts, 1.0)

    # For each column, compute group sums and subtract
    # This is still O(k) scatter-adds, but each operates on a full column
    # which is much faster than per-group Python loops
    result = M.copy() if hasattr(M, 'copy') else M.clone()
    for j in range(k):
        col = M[:, j]
        group_sums_j = _scatter_add(xp, idx, col, n_groups)
        group_means_j = group_sums_j * inv_counts
        result[:, j] = col - group_means_j[idx]

    return result


def demean_variables(y, X, entity_ids, time_ids=None, xp=None,
                     max_iter=100, tol=1e-10):
    """Demean *y* and *X* for fixed-effects estimation.

    If *time_ids* is also provided, performs two-way demeaning (entity
    and time effects) using the alternating projection method (Mundlak
    1978).  For balanced panels convergence occurs in one pass; for
    unbalanced panels the iteration continues until the maximum change
    across all variables is below *tol*.

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
    max_iter : int, default=100
        Maximum alternating-projection iterations for two-way FE.
    tol : float, default=1e-10
        Convergence tolerance for two-way FE (max absolute change).

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

    y_d = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
    X_d = X.copy() if hasattr(X, 'copy') else X.clone() if hasattr(X, 'clone') else X - 0.0

    # Entity demeaning (skip if entity_ids is None, e.g. time-only FE)
    if entity_ids is not None:
        y_d = within_transform(y_d, entity_ids, xp)
        X_d = _within_transform_matrix(X_d, entity_ids, xp)

    # Time demeaning (two-way FE) with alternating projection
    # Each iteration applies BOTH entity and time demeaning to ensure
    # convergence to the true two-way fixed effects (Mundlak 1978).
    if time_ids is not None:
        for iteration in range(max_iter):
            y_d_old = y_d.copy() if hasattr(y_d, 'copy') else y_d.clone()

            # Alternate: entity demeaning then time demeaning
            # Only apply entity demeaning if entity_ids is provided (two-way FE)
            if entity_ids is not None:
                y_d = within_transform(y_d, entity_ids, xp)
                X_d = _within_transform_matrix(X_d, entity_ids, xp)
            y_d = within_transform(y_d, time_ids, xp)
            X_d = _within_transform_matrix(X_d, time_ids, xp)

            # Check convergence (single sync)
            max_change = _to_float_scalar(xp.max(xp.abs(y_d - y_d_old)))
            if max_change < tol:
                break

    return y_d, X_d


def group_means(y, groups, xp=None):
    """Compute group-level means aligned to each observation.

    Returns an array of shape (n,) where element *i* is the mean of *y*
    over all observations belonging to the same group as observation *i*.

    Uses scatter-add for single-kernel group reduction.

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

    idx, n_groups, _ = _remap_to_contiguous(groups, xp)

    # Group sums and counts via scatter-add (2 kernel launches)
    group_sums = _scatter_add(xp, idx, y, n_groups)
    group_counts = _scatter_add(xp, idx, xp.ones_like(y), n_groups)

    means = group_sums / xp.maximum(group_counts, 1.0)
    return means[idx]


def group_sizes(groups, xp=None):
    """Return an array of per-observation group sizes.

    Element *i* is the number of observations in the group of
    observation *i*.

    Uses scatter-add for single-kernel group counting.

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
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)

    # Group counts via scatter-add (1 kernel launch)
    ones = xp_ones(len(groups), xp.float64, xp, groups)
    counts = _scatter_add(xp, idx, ones, n_groups)
    return counts[idx]


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
    _eps = np.finfo(np.float64).tiny
    tvalues = params / np.maximum(bse, _eps)
    pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), df))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    conf_int = np.column_stack([
        params - t_crit * bse,
        params + t_crit * bse,
    ])
    return bse, tvalues, pvalues, conf_int
