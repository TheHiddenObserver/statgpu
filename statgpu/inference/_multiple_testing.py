"""Multiple-testing utilities (FDR/FWER p-value adjustments)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import chi2

from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar


def _to_bool_scalar(x) -> bool:
    if hasattr(x, "item"):
        return bool(x.item())
    return bool(x)


def _cumextreme(arr, mode: str, xp):
    """Backend-compatible cumulative min/max with CuPy fallback."""
    if xp is np:
        if mode == "min":
            return np.minimum.accumulate(arr)
        if mode == "max":
            return np.maximum.accumulate(arr)
        raise ValueError("mode must be 'min' or 'max'")

    # CuPy does not support minimum/maximum.accumulate on all versions.
    import cupy as cp

    arr_np = cp.asnumpy(arr)
    if mode == "min":
        out_np = np.minimum.accumulate(arr_np)
    elif mode == "max":
        out_np = np.maximum.accumulate(arr_np)
    else:
        raise ValueError("mode must be 'min' or 'max'")
    return cp.asarray(out_np)


_METHOD_ALIASES = {
    "bh": "bh",
    "fdr_bh": "bh",
    "benjamini-hochberg": "bh",
    "benjamini_hochberg": "bh",
    "by": "by",
    "fdr_by": "by",
    "benjamini-yekutieli": "by",
    "benjamini_yekutieli": "by",
    "holm": "holm",
    "holm-bonferroni": "holm",
    "holm_bonferroni": "holm",
    "bonferroni": "bonferroni",
    "bonf": "bonferroni",
}


_COMBINE_METHOD_ALIASES = {
    "fisher": "fisher",
    "fisher-combination": "fisher",
    "fisher_combination": "fisher",
    "cauchy": "cauchy",
    "cauchy-combination": "cauchy",
    "cauchy_combination": "cauchy",
    "acat": "cauchy",
}


def _normalize_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in _METHOD_ALIASES:
        allowed = sorted(set(_METHOD_ALIASES.values()))
        raise ValueError(f"Unknown method='{method}'. Supported methods: {allowed}")
    return _METHOD_ALIASES[key]


def _normalize_combine_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in _COMBINE_METHOD_ALIASES:
        allowed = sorted(set(_COMBINE_METHOD_ALIASES.values()))
        raise ValueError(f"Unknown method='{method}'. Supported methods: {allowed}")
    return _COMBINE_METHOD_ALIASES[key]


def _validate_alpha(alpha: float) -> float:
    alpha_f = float(alpha)
    if alpha_f <= 0.0 or alpha_f >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return alpha_f


def _validate_1d_pvalues(pvalues, xp):
    p = xp.asarray(pvalues, dtype=xp.float64).reshape(-1)
    if _to_bool_scalar(xp.any(~xp.isfinite(p))):
        raise ValueError("pvalues must be finite")
    if _to_bool_scalar(xp.any((p < 0.0) | (p > 1.0))):
        raise ValueError("pvalues must be within [0, 1]")
    return p


def _validate_pvalues_array(arr, xp):
    p = xp.asarray(arr, dtype=xp.float64)
    if _to_bool_scalar(xp.any(~xp.isfinite(p))):
        raise ValueError("pvalues must be finite")
    if _to_bool_scalar(xp.any((p < 0.0) | (p > 1.0))):
        raise ValueError("pvalues must be within [0, 1]")
    return p


def _adjust_1d_pvalues(pvalues_1d, method: str, xp):
    m = int(pvalues_1d.size)
    if m == 0:
        return xp.asarray([], dtype=xp.float64)

    order = xp.argsort(pvalues_1d)
    p_sorted = pvalues_1d[order]

    if method == "bonferroni":
        adj_sorted = xp.minimum(p_sorted * m, 1.0)
    elif method == "holm":
        factors = m - xp.arange(m, dtype=xp.float64)
        raw = factors * p_sorted
        adj_sorted = _cumextreme(raw, mode="max", xp=xp)
        adj_sorted = xp.minimum(adj_sorted, 1.0)
    elif method == "bh":
        ranks = xp.arange(1.0, m + 1.0)
        raw = p_sorted * m / ranks
        adj_sorted = _cumextreme(raw[::-1], mode="min", xp=xp)[::-1]
        adj_sorted = xp.minimum(adj_sorted, 1.0)
    elif method == "by":
        ranks = xp.arange(1.0, m + 1.0)
        c_m = xp.sum(1.0 / ranks)
        raw = p_sorted * m * c_m / ranks
        adj_sorted = _cumextreme(raw[::-1], mode="min", xp=xp)[::-1]
        adj_sorted = xp.minimum(adj_sorted, 1.0)
    else:
        raise ValueError(f"Unsupported normalized method: {method}")

    adj = xp.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj


def _validate_weights(weights, m: int, xp):
    if weights is None:
        return xp.full(m, 1.0 / m, dtype=xp.float64)

    w = xp.asarray(weights, dtype=xp.float64).reshape(-1)
    if int(w.size) != int(m):
        raise ValueError("weights must be 1D and have the same length as the combine axis")
    if _to_bool_scalar(xp.any(~xp.isfinite(w))):
        raise ValueError("weights must be finite")
    if _to_bool_scalar(xp.any(w < 0.0)):
        raise ValueError("weights must be non-negative")

    w_sum = xp.sum(w)
    if _to_float_scalar(w_sum) <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w / w_sum


def _chi2_sf(statistics, df: int, xp):
    """Chi-square survival function for NumPy/CuPy arrays."""
    if xp is np:
        return np.asarray(chi2.sf(statistics, df=df), dtype=np.float64)

    import cupy as cp

    stats_cp = cp.asarray(statistics, dtype=cp.float64)

    # Prefer GPU-native computation when cupyx is available.
    try:
        from cupyx.scipy.special import gammaincc

        return cp.asarray(gammaincc(0.5 * df, 0.5 * stats_cp), dtype=cp.float64)
    except Exception:
        stats_np = cp.asnumpy(stats_cp)
        sf_np = chi2.sf(stats_np, df=df)
        return cp.asarray(sf_np, dtype=cp.float64)


def _combine_1d_fisher(pvalues_1d, xp):
    p = _validate_1d_pvalues(pvalues_1d, xp)
    m = int(p.size)
    if m == 0:
        raise ValueError("pvalues must contain at least one value")

    # Avoid log(0) while keeping statistical meaning for very small p-values.
    eps = np.finfo(np.float64).tiny
    p_safe = xp.clip(p.astype(xp.float64), eps, 1.0)
    statistic = -2.0 * xp.sum(xp.log(p_safe))

    pvalue = _chi2_sf(statistic, df=2 * m, xp=xp)
    return statistic.astype(xp.float64), pvalue.astype(xp.float64)


def _combine_1d_cauchy(pvalues_1d, weights, xp):
    p = _validate_1d_pvalues(pvalues_1d, xp)
    m = int(p.size)
    if m == 0:
        raise ValueError("pvalues must contain at least one value")

    w = _validate_weights(weights, m, xp)

    eps = np.finfo(np.float64).eps
    p_safe = xp.clip(p.astype(xp.float64), eps, 1.0 - eps)
    statistic = xp.sum(w * xp.tan((0.5 - p_safe) * np.pi))
    pvalue = 0.5 - xp.arctan(statistic) / np.pi
    pvalue = xp.clip(pvalue, 0.0, 1.0)
    return statistic.astype(xp.float64), pvalue.astype(xp.float64)


def _combine_1d_pvalues(pvalues_1d, method: str, weights, xp):
    if method == "fisher":
        if weights is not None:
            raise ValueError("weights are only supported for method='cauchy'")
        return _combine_1d_fisher(pvalues_1d, xp)
    if method == "cauchy":
        return _combine_1d_cauchy(pvalues_1d, weights, xp)
    raise ValueError(f"Unsupported normalized combine method: {method}")


def adjust_pvalues(
    pvalues,
    method: str = "bh",
    alpha: float = 0.05,
    axis: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust p-values for multiple testing.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values.
    method : str, default='bh'
        One of: 'bh', 'by', 'holm', 'bonferroni'.
        Common aliases are accepted (e.g., 'fdr_bh', 'bonf').
    alpha : float, default=0.05
        Rejection threshold in (0, 1).
    axis : int or None, default=None
        Axis along which to adjust p-values.
        If None, adjusts over all values flattened.
    backend : {'auto', 'numpy', 'cupy'}, default='auto'
        Compute backend. ``'auto'`` infers from input array type.

    Returns
    -------
    reject : ndarray of bool
        Rejection mask for adjusted p-values at level ``alpha``.
    pvalues_adjusted : ndarray of float
        Adjusted p-values with same shape as input.
    """
    method_n = _normalize_method(method)
    alpha_f = _validate_alpha(alpha)
    backend_name = _resolve_backend(backend, pvalues)
    xp = _get_xp(backend_name)

    arr = xp.asarray(pvalues, dtype=xp.float64)

    if axis is None:
        flat = _validate_1d_pvalues(arr, xp)
        adj_flat = _adjust_1d_pvalues(flat, method_n, xp)
        reject_flat = adj_flat <= alpha_f
        return reject_flat.reshape(arr.shape), adj_flat.reshape(arr.shape)

    if arr.ndim == 0:
        raise ValueError("axis must be None for scalar pvalues")

    axis_n = int(np.core.numeric.normalize_axis_index(axis, arr.ndim))
    moved = xp.moveaxis(arr, axis_n, -1)
    matrix = moved.reshape(-1, moved.shape[-1])

    adj_matrix = xp.empty_like(matrix, dtype=xp.float64)
    reject_matrix = xp.empty_like(matrix, dtype=bool)

    for i in range(matrix.shape[0]):
        row = _validate_1d_pvalues(matrix[i], xp)
        adj_row = _adjust_1d_pvalues(row, method_n, xp)
        adj_matrix[i] = adj_row
        reject_matrix[i] = adj_row <= alpha_f

    adj_moved = adj_matrix.reshape(moved.shape)
    reject_moved = reject_matrix.reshape(moved.shape)

    return (
        xp.moveaxis(reject_moved, -1, axis_n),
        xp.moveaxis(adj_moved, -1, axis_n),
    )


def combine_pvalues(
    pvalues,
    method: str = "fisher",
    weights=None,
    axis: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine p-values into a global p-value.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values to combine.
    method : {'fisher', 'cauchy'}, default='fisher'
        Combination method. Aliases accepted (e.g., 'acat').
    weights : array-like, optional
        Optional non-negative weights for method='cauchy'.
    axis : int or None, default=None
        Axis along which to combine p-values. If None, flattens all values.
    backend : {'auto', 'numpy', 'cupy'}, default='auto'
        Compute backend. ``'auto'`` infers from input array type.

    Returns
    -------
    statistic : ndarray or scalar
        Combined test statistic.
    pvalue : ndarray or scalar
        Combined p-value(s).
    """
    method_n = _normalize_combine_method(method)
    backend_name = _resolve_backend(backend, pvalues)
    xp = _get_xp(backend_name)

    arr = xp.asarray(pvalues, dtype=xp.float64)

    if axis is None:
        flat = _validate_1d_pvalues(arr, xp)
        return _combine_1d_pvalues(flat, method_n, weights, xp)

    if arr.ndim == 0:
        raise ValueError("axis must be None for scalar pvalues")

    arr = _validate_pvalues_array(arr, xp)
    axis_n = int(np.core.numeric.normalize_axis_index(axis, arr.ndim))
    moved = xp.moveaxis(arr, axis_n, -1)
    m = int(moved.shape[-1])
    if m == 0:
        raise ValueError("pvalues must contain at least one value")

    if method_n == "fisher":
        if weights is not None:
            raise ValueError("weights are only supported for method='cauchy'")
        eps = np.finfo(np.float64).tiny
        p_safe = xp.clip(moved.astype(xp.float64), eps, 1.0)
        statistics = -2.0 * xp.sum(xp.log(p_safe), axis=-1)
        pvals = _chi2_sf(statistics, df=2 * m, xp=xp)
        return statistics.astype(xp.float64), pvals.astype(xp.float64)

    if method_n == "cauchy":
        w = _validate_weights(weights, m, xp)
        eps = np.finfo(np.float64).eps
        p_safe = xp.clip(moved.astype(xp.float64), eps, 1.0 - eps)
        w_shape = (1,) * (p_safe.ndim - 1) + (m,)
        statistics = xp.sum(w.reshape(w_shape) * xp.tan((0.5 - p_safe) * np.pi), axis=-1)
        pvals = 0.5 - xp.arctan(statistics) / np.pi
        pvals = xp.clip(pvals, 0.0, 1.0)
        return statistics.astype(xp.float64), pvals.astype(xp.float64)

    raise ValueError(f"Unsupported normalized combine method: {method_n}")


def multipletests(
    pvalues,
    alpha: float = 0.05,
    method: str = "bh",
    axis: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias compatible with common scientific naming."""
    return adjust_pvalues(
        pvalues,
        method=method,
        alpha=alpha,
        axis=axis,
        backend=backend,
    )
