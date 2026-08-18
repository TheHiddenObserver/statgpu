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


def _prepare_group_projection(groups, xp):
    """Factorize one group vector once and cache backend-native projection data."""
    groups = xp_asarray(groups, xp=xp).ravel()
    idx, n_groups, labels = _remap_to_contiguous(groups, xp)
    ones = xp_ones(int(idx.shape[0]), xp.float64, xp, idx)
    counts = _scatter_add(xp, idx, ones, n_groups)
    inv_counts = 1.0 / xp_maximum(counts, 1.0, xp)
    return idx, n_groups, labels, counts, inv_counts


def _compact_group_means(values, projection, xp):
    idx, n_groups, _labels, counts, inv_counts = projection
    # A group sum can overflow even though its mean is finite. Only groups whose
    # raw accumulation is at risk are divided by their own count before the
    # scatter-add; safe groups keep the historical arithmetic exactly.
    counts_aligned = counts[idx]
    limit = np.finfo(np.float64).max / xp_maximum(counts_aligned, 1.0, xp)
    dangerous_obs = (xp.abs(values) > limit) * 1.0
    dangerous_count = _scatter_add(xp, idx, dangerous_obs, n_groups)
    factor_compact = xp.where(
        dangerous_count > 0.0, counts, xp.ones_like(counts)
    )
    factor_aligned = factor_compact[idx]
    sums = _scatter_add(xp, idx, values / factor_aligned, n_groups)
    return sums * inv_counts * factor_compact


def _within_preindexed(values, projection, xp):
    means = _compact_group_means(values, projection, xp)
    return values - means[projection[0]]


def _within_matrix_preindexed(matrix, projection, xp):
    result = matrix.copy() if hasattr(matrix, "copy") else matrix.clone()
    for j in range(int(matrix.shape[1])):
        result[:, j] = _within_preindexed(matrix[:, j], projection, xp)
    return result


def _column_max_abs(matrix, xp):
    if getattr(xp, "__name__", "") == "torch":
        return xp.max(xp.abs(matrix), dim=0).values
    return xp.max(xp.abs(matrix), axis=0)


def _matrix_group_mean_max(matrix, projection, xp):
    values = xp_zeros(
        int(matrix.shape[1]), matrix.dtype, xp, matrix
    )
    for j in range(int(matrix.shape[1])):
        means = _compact_group_means(matrix[:, j], projection, xp)
        values[j] = xp.max(xp.abs(means))
    return values


def _convergence_allowance(current_scale, level_scale, tol, xp):
    """Return relative tolerance unless a transformed direction is numerically absorbed."""
    tiny = np.finfo(np.float64).tiny
    level_scale = xp_maximum(level_scale, tiny, xp)
    roundoff_floor = 8.0 * np.finfo(np.float64).eps * level_scale
    relative = float(tol) * xp_maximum(current_scale, tiny, xp)
    return xp.where(current_scale > roundoff_floor, relative, roundoff_floor)


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
    projection = _prepare_group_projection(groups, xp)
    return _within_preindexed(y, projection, xp)


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
    projection = _prepare_group_projection(groups, xp)
    return _within_matrix_preindexed(M, projection, xp)


def demean_variables(
    y,
    X,
    entity_ids,
    time_ids=None,
    xp=None,
    max_iter=1_000_000,
    tol=1e-10,
):
    """Demean *y* and *X* for one- or two-way fixed-effects estimation.

    Two-way projections factorize group metadata once, keep all iterative
    projection state on the selected backend, and stop only when residual means
    are small for *both* effect dimensions.  A scale-aware roundoff floor is used
    only for directions that are themselves numerically absorbed by the fixed
    effects, avoiding both premature convergence from large removable levels and
    non-termination of exactly absorbed columns.
    """
    if xp is None:
        xp = np
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ) or int(max_iter) <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and positive")
    max_iter = int(max_iter)

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
    y_level_scale = xp.max(xp.abs(y_d))
    X_level_scale = _column_max_abs(X_d, xp)

    entity_projection = (
        None if entity_ids is None else _prepare_group_projection(entity_ids, xp)
    )
    time_projection = (
        None if time_ids is None else _prepare_group_projection(time_ids, xp)
    )

    if entity_projection is not None:
        y_d = _within_preindexed(y_d, entity_projection, xp)
        X_d = _within_matrix_preindexed(X_d, entity_projection, xp)

    if time_projection is None:
        return y_d, X_d

    # A one-way time effect is an exact single projection. Alternation is needed
    # only when both entity and time effects are present.
    if entity_projection is None:
        y_d = _within_preindexed(y_d, time_projection, xp)
        X_d = _within_matrix_preindexed(X_d, time_projection, xp)
        return y_d, X_d

    converged = False
    final_metric = float("inf")
    for _iteration in range(max_iter):
        y_d = _within_preindexed(y_d, entity_projection, xp)
        X_d = _within_matrix_preindexed(X_d, entity_projection, xp)
        y_d = _within_preindexed(y_d, time_projection, xp)
        X_d = _within_matrix_preindexed(X_d, time_projection, xp)

        entity_y_means = _compact_group_means(y_d, entity_projection, xp)
        time_y_means = _compact_group_means(y_d, time_projection, xp)
        y_violation = xp.maximum(
            xp.max(xp.abs(entity_y_means)), xp.max(xp.abs(time_y_means))
        )
        entity_X_violation = _matrix_group_mean_max(
            X_d, entity_projection, xp
        )
        time_X_violation = _matrix_group_mean_max(X_d, time_projection, xp)
        X_violation = xp.maximum(entity_X_violation, time_X_violation)

        y_scale = xp.max(xp.abs(y_d))
        X_scale = _column_max_abs(X_d, xp)
        y_allowance = _convergence_allowance(
            y_scale, y_level_scale, tol, xp
        )
        X_allowance = _convergence_allowance(
            X_scale, X_level_scale, tol, xp
        )
        y_metric = y_violation / xp_maximum(
            y_allowance, np.finfo(np.float64).tiny, xp
        )
        X_metric = xp.max(
            X_violation
            / xp_maximum(X_allowance, np.finfo(np.float64).tiny, xp)
        )
        final_metric = _to_float_scalar(xp.maximum(y_metric, X_metric))
        if final_metric <= 1.0:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "two-way fixed-effect demeaning did not converge within "
            f"max_iter={max_iter}; final normalized group-mean violation="
            f"{final_metric:.6e}"
        )

    return y_d, X_d


def _recover_two_way_effects(
    values,
    entity_ids,
    time_ids,
    xp,
    *,
    max_iter=1_000_000,
    tol=1e-10,
):
    """Recover additive two-way effects by backend-native alternating least squares.

    The returned compact entity/time effects reproduce the joint least-squares
    projection on observed cells.  The time effects are normalized to have zero
    observation-weighted mean, with the compensating shift applied to entity
    effects so fitted values are unchanged.
    """
    values = xp_asarray(values, dtype=xp.float64, xp=xp).ravel()
    entity_projection = _prepare_group_projection(entity_ids, xp)
    time_projection = _prepare_group_projection(time_ids, xp)
    e_idx, n_entities, _e_labels, _e_counts, _e_inv = entity_projection
    t_idx, n_times, _t_labels, t_counts, _t_inv = time_projection
    entity_effects = xp_zeros(n_entities, xp.float64, xp, values)
    time_effects = xp_zeros(n_times, xp.float64, xp, values)
    level_scale = xp.max(xp.abs(values))

    converged = False
    final_metric = float("inf")
    for _iteration in range(int(max_iter)):
        entity_effects = _compact_group_means(
            values - time_effects[t_idx], entity_projection, xp
        )
        time_effects = _compact_group_means(
            values - entity_effects[e_idx], time_projection, xp
        )

        shift = xp.sum(time_effects * t_counts) / float(values.shape[0])
        time_effects = time_effects - shift
        entity_effects = entity_effects + shift

        residual = values - entity_effects[e_idx] - time_effects[t_idx]
        entity_means = _compact_group_means(residual, entity_projection, xp)
        time_means = _compact_group_means(residual, time_projection, xp)
        violation = xp.maximum(
            xp.max(xp.abs(entity_means)), xp.max(xp.abs(time_means))
        )
        residual_scale = xp.max(xp.abs(residual))
        allowance = _convergence_allowance(
            residual_scale, level_scale, tol, xp
        )
        final_metric = _to_float_scalar(
            violation
            / xp_maximum(allowance, np.finfo(np.float64).tiny, xp)
        )
        if final_metric <= 1.0:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            "two-way fixed-effect recovery did not converge within "
            f"max_iter={int(max_iter)}; final normalized group-mean violation="
            f"{final_metric:.6e}"
        )
    return entity_effects, time_effects


def group_means(y, groups, xp=None):
    """Compute group-level means aligned to each observation."""
    if xp is None:
        xp = np

    y = xp_asarray(y, dtype=xp.float64, xp=xp).ravel()
    groups = xp_asarray(groups, xp=xp, ref_arr=y).ravel()
    projection = _prepare_group_projection(groups, xp)
    means = _compact_group_means(y, projection, xp)
    return means[projection[0]]


def group_sizes(groups, xp=None):
    """Return an array of per-observation group sizes."""
    if xp is None:
        xp = np

    groups = xp_asarray(groups, xp=xp).ravel()
    projection = _prepare_group_projection(groups, xp)
    return projection[3][projection[0]]


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