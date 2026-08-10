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

__all__ = [
    "PanelSummary",
    "demean_variables",
    "within_transform",
    "group_means",
    "group_sizes",
    "make_group_dummies",
    "compute_panel_inference",
    "factorize_panel_metadata",
    "factorize_panel_labels",
    "validate_panel_numeric_data",
    "validate_panel_alpha",
]

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from statgpu.backends import (
    xp_asarray,
    xp_maximum,
    xp_ones,
    xp_zeros,
    _to_float_scalar,
    _to_numpy,
)


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
        statistic_label = "t" if self.cov_type in (None, "nonrobust") else "z"
        pvalue_label = "P>|t|" if statistic_label == "t" else "P>|z|"
        lines.append(
            f"{'':<12} {'coef':>10} {'std err':>10} "
            f"{statistic_label:>8} {pvalue_label:>10} {ci_label:>10} {ci_label2:>10}"
        )
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
            "model_type": self.model_type,
            "nobs": self.nobs,
            "df_resid": self.df_resid,
            "coef": self.coef.tolist(),
            "bse": self.bse.tolist(),
            "tvalues": self.tvalues.tolist(),
            "pvalues": self.pvalues.tolist(),
            "conf_int": self.conf_int.tolist(),
            "feature_names": self.feature_names,
            "rsquared_within": self.rsquared_within,
            "cov_type": self.cov_type,
            "entity_effects": self.entity_effects,
            "time_effects": self.time_effects,
            "variance_components": self.variance_components,
            "theta": self.theta,
            "alpha": self.alpha,
        }


def _scatter_add(xp, indices, values, n_groups):
    """Scatter-add values into bins defined by indices on the selected backend."""
    backend_name = getattr(xp, "__name__", "")
    if backend_name == "torch":
        out = xp.zeros(n_groups, dtype=values.dtype, device=values.device)
        out.scatter_add_(0, indices.long(), values)
        return out
    if backend_name == "cupy":
        # CuPy implements ``ufunc.at`` natively.  Do not route numerical values
        # through host NumPy when optional cupyx helpers are unavailable.
        out = xp.zeros(n_groups, dtype=values.dtype)
        xp.add.at(out, indices, values)
        return out

    out = np.zeros(n_groups, dtype=values.dtype)
    np.add.at(
        out,
        np.asarray(indices, dtype=np.int64),
        np.asarray(values),
    )
    return out


def _remap_to_contiguous(groups, xp):
    """Remap group labels to contiguous 0..n_groups-1 indices."""
    groups_np = _to_numpy(groups).ravel()
    unique_labels, indices_np = np.unique(groups_np, return_inverse=True)
    n_groups = len(unique_labels)
    indices = xp_asarray(indices_np, dtype=xp.int64, xp=xp, ref_arr=groups)
    return indices, n_groups, unique_labels


def validate_panel_alpha(alpha):
    """Validate the confidence-interval significance level."""
    if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")


def validate_panel_numeric_data(X, y, xp):
    """Validate panel design/response shape and finiteness on the backend."""
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be one-dimensional with one value per row of X")
    finite_X = bool(_to_float_scalar(xp.all(xp.isfinite(X))))
    finite_y = bool(_to_float_scalar(xp.all(xp.isfinite(y))))
    if not finite_X or not finite_y:
        raise ValueError("X and y must contain only finite values")


def _ordered_categorical_metadata(values, *, name: str, expected_n=None):
    """Preserve declared pandas ordered-categorical chronology when present."""
    candidate = getattr(values, "array", values)
    dtype = getattr(candidate, "dtype", None)
    categories = getattr(dtype, "categories", None)
    codes = getattr(candidate, "codes", None)
    if categories is None or not bool(getattr(dtype, "ordered", False)) or codes is None:
        return None

    codes_np = np.asarray(codes, dtype=np.int64).ravel()
    if codes_np.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if expected_n is not None and codes_np.shape[0] != int(expected_n):
        raise ValueError(f"{name} must have {int(expected_n)} observations")
    if np.any(codes_np < 0):
        raise ValueError(f"{name} must not contain missing or non-finite values")

    observed = np.unique(codes_np)
    labels = np.asarray(categories)[observed]
    remapped = np.searchsorted(observed, codes_np).astype(np.int64, copy=False)
    return labels, remapped


def _object_label_is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return not bool(np.isfinite(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    value_type = type(value)
    if value_type.__module__.startswith("pandas") and value_type.__name__ in {
        "NAType",
        "NaTType",
    }:
        return True
    return False


def factorize_panel_metadata(values, *, name="labels", expected_n=None):
    """Validate and factorize observation metadata into CPU labels/codes.

    Entity, time and cluster labels share the same missing-value contract:
    ``None``, NaN/Inf, NaT and pandas missing sentinels are rejected rather than
    silently becoming legitimate panel groups.  Ordered categoricals retain
    their declared ordering.
    """
    if values is None:
        return None, None

    categorical = _ordered_categorical_metadata(
        values, name=name, expected_n=expected_n
    )
    if categorical is not None:
        return categorical

    values_np = np.asarray(_to_numpy(values))
    if values_np.ndim == 2 and values_np.shape[1] == 1:
        values_np = values_np[:, 0]
    if values_np.ndim != 1 or values_np.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if expected_n is not None and values_np.shape[0] != int(expected_n):
        raise ValueError(f"{name} must have {int(expected_n)} observations")

    invalid = False
    if np.issubdtype(values_np.dtype, np.number):
        invalid = bool(np.any(~np.isfinite(values_np)))
    elif np.issubdtype(values_np.dtype, np.datetime64) or np.issubdtype(
        values_np.dtype, np.timedelta64
    ):
        invalid = bool(np.any(np.isnat(values_np)))
    elif values_np.dtype.kind == "O":
        invalid = any(_object_label_is_missing(value) for value in values_np)
    if invalid:
        raise ValueError(f"{name} must not contain missing or non-finite values")

    try:
        unique_labels, codes = np.unique(values_np, return_inverse=True)
    except TypeError as exc:
        raise ValueError(
            f"{name} must contain mutually comparable labels with deterministic identity"
        ) from exc
    return unique_labels, codes.astype(np.int64, copy=False)


def factorize_panel_labels(values, xp, ref_arr=None, name="labels", expected_n=None):
    """Factorize validated metadata on CPU and copy only compact codes to device."""
    if values is None:
        return None, None
    unique_labels, codes = factorize_panel_metadata(
        values, name=name, expected_n=expected_n
    )
    codes_dev = xp_asarray(codes, dtype=xp.int64, xp=xp, ref_arr=ref_arr)
    return codes_dev, unique_labels


def within_transform(y, groups, xp=None):
    """Remove group means (fixed-effect projection)."""
    if xp is None:
        xp = np

    y = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
    groups = xp_asarray(groups, xp=xp, ref_arr=y).ravel()
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)
    group_sums = _scatter_add(xp, idx, y, n_groups)
    group_counts = _scatter_add(xp, idx, xp.ones_like(y), n_groups)
    group_means = group_sums / xp_maximum(group_counts, 1.0, xp)
    return y - group_means[idx]


def make_group_dummies(groups, xp=None):
    """Create dummy variable matrix from group labels."""
    if xp is None:
        xp = np

    groups = xp_asarray(groups, xp=xp).ravel()
    n = len(groups)
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)
    D = xp_zeros((n, n_groups), xp.float64, xp, groups)
    if getattr(xp, "__name__", "") == "torch":
        row_idx = xp.arange(
            n,
            device=getattr(groups, "device", None)
            if hasattr(groups, "device")
            else None,
        )
    else:
        row_idx = xp.arange(n)
    D[row_idx, idx] = 1.0
    return D


def _within_transform_matrix(M, groups, xp):
    """Remove group means from each column of matrix M (batched)."""
    n, k = M.shape
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)
    ones_col = xp_ones(n, M.dtype, xp, M)
    group_counts = _scatter_add(xp, idx, ones_col, n_groups)
    inv_counts = 1.0 / xp_maximum(group_counts, 1.0, xp)

    result = M.copy() if hasattr(M, "copy") else M.clone()
    for j in range(k):
        col = M[:, j]
        group_sums_j = _scatter_add(xp, idx, col, n_groups)
        group_means_j = group_sums_j * inv_counts
        result[:, j] = col - group_means_j[idx]
    return result


def demean_variables(
    y,
    X,
    entity_ids,
    time_ids=None,
    xp=None,
    max_iter=100,
    tol=1e-10,
):
    """Demean *y* and *X* for fixed-effects estimation."""
    if xp is None:
        xp = np

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y_d = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
    X_d = (
        X.copy()
        if hasattr(X, "copy")
        else X.clone()
        if hasattr(X, "clone")
        else X - 0.0
    )

    if entity_ids is not None:
        y_d = within_transform(y_d, entity_ids, xp)
        X_d = _within_transform_matrix(X_d, entity_ids, xp)

    if time_ids is not None:
        for _iteration in range(max_iter):
            y_d_old = y_d.copy() if hasattr(y_d, "copy") else y_d.clone()
            if entity_ids is not None:
                y_d = within_transform(y_d, entity_ids, xp)
                X_d = _within_transform_matrix(X_d, entity_ids, xp)
            y_d = within_transform(y_d, time_ids, xp)
            X_d = _within_transform_matrix(X_d, time_ids, xp)
            max_change = _to_float_scalar(xp.max(xp.abs(y_d - y_d_old)))
            if max_change < tol:
                break

    return y_d, X_d


def group_means(y, groups, xp=None):
    """Compute group-level means aligned to each observation."""
    if xp is None:
        xp = np

    y = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
    groups = xp_asarray(groups, xp=xp, ref_arr=y).ravel()
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)
    group_sums = _scatter_add(xp, idx, y, n_groups)
    group_counts = _scatter_add(xp, idx, xp.ones_like(y), n_groups)
    means = group_sums / xp_maximum(group_counts, 1.0, xp)
    return means[idx]


def group_sizes(groups, xp=None):
    """Return an array of per-observation group sizes."""
    if xp is None:
        xp = np

    groups = xp_asarray(groups, xp=xp).ravel()
    idx, n_groups, _ = _remap_to_contiguous(groups, xp)
    ones = xp_ones(len(groups), xp.float64, xp, groups)
    counts = _scatter_add(xp, idx, ones, n_groups)
    return counts[idx]


def ols_inference_nonrobust(params, X, scale, df, alpha=0.05):
    """Compute non-robust OLS inference (SE, t, p, CI)."""
    from scipy import stats

    X_pinv = np.linalg.pinv(X)
    cov_params = scale * (X_pinv @ X_pinv.T)
    diag = np.diag(cov_params)
    tol = 4096.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(diag))) if diag.size else 1.0
    )
    if np.any(diag < -tol):
        raise ValueError("covariance has materially negative diagonal variance")
    bse = np.sqrt(np.maximum(diag, 0.0))
    tvalues = params / np.maximum(bse, np.finfo(np.float64).tiny)
    pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), df))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    conf_int = np.column_stack(
        [params - t_crit * bse, params + t_crit * bse]
    )
    return bse, tvalues, pvalues, conf_int


def compute_panel_inference(
    model,
    X,
    resid,
    params,
    scale,
    n,
    k,
    xp,
    backend_name,
    cov_type,
    alpha,
    dist_df=None,
):
    """Legacy shared OLS inference, using an X-pseudoinverse stable bread."""
    from statgpu.backends import _to_numpy

    X_pinv = xp.linalg.pinv(X)
    bread = X_pinv @ X_pinv.T
    if cov_type == "nonrobust":
        cov_params = scale * bread
    elif cov_type == "robust":
        influence = X_pinv.T * resid[:, None]
        cov_params = influence.T @ influence * n / (n - k)
    elif cov_type == "clustered":
        raise ValueError(
            "cov_type='clustered' requires cluster labels. "
            "Use PooledOLS which accepts a 'cluster' parameter."
        )
    elif cov_type == "hac":
        raise NotImplementedError(
            "cov_type='hac' is not supported in compute_panel_inference. "
            "Use PooledOLS or FamaMacBeth which have native HAC support."
        )
    else:
        cov_params = scale * bread

    diag_cov = xp.diag(cov_params)
    diag_abs_max = _to_float_scalar(xp.max(xp.abs(diag_cov)))
    tol = 4096.0 * np.finfo(np.float64).eps * max(1.0, diag_abs_max)
    if _to_float_scalar(xp.min(diag_cov)) < -tol:
        raise ValueError("covariance has materially negative diagonal variance")
    diag_cov = xp_maximum(diag_cov, 0.0, xp)
    bse_dev = xp.sqrt(diag_cov)
    tvalues_dev = params / xp_maximum(
        bse_dev, np.finfo(np.float64).tiny, xp
    )

    df = dist_df if dist_df is not None else n - k
    from statgpu.inference._distributions_backend import get_distribution

    dist_name = "norm" if cov_type in ("robust", "clustered", "hac") else "t"
    t_dist = get_distribution(dist_name, backend=backend_name)
    if dist_name == "t":
        pvalues_dev = 2 * t_dist.sf(xp.abs(tvalues_dev), df)
        t_crit = t_dist.isf(alpha / 2, df)
    else:
        pvalues_dev = 2 * t_dist.sf(xp.abs(tvalues_dev))
        t_crit = t_dist.isf(alpha / 2)

    t_crit = xp_asarray(t_crit, dtype=params.dtype, xp=xp, ref_arr=params)
    conf_low = params - t_crit * bse_dev
    conf_high = params + t_crit * bse_dev

    model.coef_ = _to_numpy(params)
    model.bse_ = _to_numpy(bse_dev)
    model.tvalues_ = _to_numpy(tvalues_dev)
    model.pvalues_ = _to_numpy(pvalues_dev)
    model.conf_int_ = _to_numpy(xp.stack([conf_low, conf_high], axis=1))
