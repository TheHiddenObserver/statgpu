"""Multiple-testing utilities (FDR/FWER p-value adjustments)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from statgpu.backends import get_backend, _resolve_backend, _to_float_scalar
from statgpu.inference._distributions_backend import chi2, norm


def _to_bool_scalar(x) -> bool:
    if hasattr(x, "item"):
        return bool(x.item())
    return bool(x)


def _normalize_axis_index(axis, ndim):
    try:
        return int(np._core.numeric.normalize_axis_index(axis, ndim))
    except AttributeError:
        return int(np.core.numeric.normalize_axis_index(axis, ndim))


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
    "hochberg": "hochberg",
    "fdr_hochberg": "hochberg",
    "step_up": "hochberg",
    "stepup": "hochberg",
}


_COMBINE_METHOD_ALIASES = {
    "fisher": "fisher",
    "fisher-combination": "fisher",
    "fisher_combination": "fisher",
    "cauchy": "cauchy",
    "cauchy-combination": "cauchy",
    "cauchy_combination": "cauchy",
    "acat": "cauchy",
    "stouffer": "stouffer",
    "z-test": "stouffer",
    "ztest": "stouffer",
    "weighted_z": "stouffer",
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


def _validate_1d_pvalues(pvalues, backend):
    p = backend.asarray(pvalues, dtype=backend.float64).reshape(-1)
    if _to_bool_scalar(backend.xp.any(~backend.xp.isfinite(p))):
        raise ValueError("pvalues must be finite")
    if _to_bool_scalar(backend.xp.any((p < 0.0) | (p > 1.0))):
        raise ValueError("pvalues must be within [0, 1]")
    return p


def _validate_pvalues_array(arr, backend):
    p = backend.asarray(arr, dtype=backend.float64)
    if _to_bool_scalar(backend.xp.any(~backend.xp.isfinite(p))):
        raise ValueError("pvalues must be finite")
    if _to_bool_scalar(backend.xp.any((p < 0.0) | (p > 1.0))):
        raise ValueError("pvalues must be within [0, 1]")
    return p


def _adjust_1d_pvalues(pvalues_1d, method: str, backend):
    m = int(pvalues_1d.shape[0])
    if m == 0:
        return backend.asarray([], dtype=backend.float64)

    order = backend.xp.argsort(pvalues_1d)
    p_sorted = pvalues_1d[order]

    if method == "bonferroni":
        adj_sorted = backend.minimum(p_sorted * m, 1.0)
    elif method == "holm":
        factors = m - backend.arange(m, dtype=backend.float64)
        raw = factors * p_sorted
        adj_sorted = backend.cummax(raw)
        adj_sorted = backend.minimum(adj_sorted, 1.0)
    elif method == "bh":
        ranks = backend.arange(1.0, m + 1.0)
        raw = p_sorted * m / ranks
        adj_sorted = backend.flip(backend.cummin(backend.flip(raw, 0)), 0)
        adj_sorted = backend.minimum(adj_sorted, 1.0)
    elif method == "by":
        ranks = backend.arange(1.0, m + 1.0)
        c_m = backend.xp.sum(1.0 / ranks)
        raw = p_sorted * m * c_m / ranks
        adj_sorted = backend.flip(backend.cummin(backend.flip(raw, 0)), 0)
        adj_sorted = backend.minimum(adj_sorted, 1.0)
    elif method == "hochberg":
        factors = backend.arange(m, 0, -1, dtype=backend.float64)
        raw = p_sorted * factors
        adj_sorted = backend.flip(backend.cummin(backend.flip(raw, 0)), 0)
        adj_sorted = backend.minimum(adj_sorted, 1.0)
    else:
        raise ValueError(f"Unsupported normalized method: {method}")

    adj = backend.xp.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj


def _validate_weights(weights, m: int, backend):
    if weights is None:
        return backend.full((m,), 1.0 / m, dtype=backend.float64)

    w = backend.asarray(weights, dtype=backend.float64).reshape(-1)
    if int(w.shape[0]) != int(m):
        raise ValueError("weights must be 1D and have the same length as the combine axis")
    if _to_bool_scalar(backend.xp.any(~backend.xp.isfinite(w))):
        raise ValueError("weights must be finite")
    if _to_bool_scalar(backend.xp.any(w < 0.0)):
        raise ValueError("weights must be non-negative")

    w_sum = backend.xp.sum(w)
    if _to_float_scalar(w_sum) <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w / w_sum




def _combine_1d_fisher(pvalues_1d, backend):
    p = _validate_1d_pvalues(pvalues_1d, backend)
    m = int(p.shape[0])
    if m == 0:
        raise ValueError("pvalues must contain at least one value")

    # Avoid log(0) while keeping statistical meaning for very small p-values.
    eps = np.finfo(np.float64).tiny
    p_safe = backend.xp.clip(backend.astype(p, backend.float64), eps, 1.0)
    statistic = -2.0 * backend.xp.sum(backend.xp.log(p_safe))

    pvalue = chi2.sf(statistic, df=2 * m)
    return backend.astype(statistic, backend.float64), backend.astype(pvalue, backend.float64)


def _combine_1d_cauchy(pvalues_1d, weights, backend):
    p = _validate_1d_pvalues(pvalues_1d, backend)
    m = int(p.shape[0])
    if m == 0:
        raise ValueError("pvalues must contain at least one value")

    w = _validate_weights(weights, m, backend)

    eps = np.finfo(np.float64).eps
    p_safe = backend.xp.clip(backend.astype(p, backend.float64), eps, 1.0 - eps)
    statistic = backend.xp.sum(w * backend.xp.tan((0.5 - p_safe) * np.pi))
    pvalue = 0.5 - backend.xp.arctan(statistic) / np.pi
    pvalue = backend.xp.clip(pvalue, 0.0, 1.0)
    return backend.astype(statistic, backend.float64), backend.astype(pvalue, backend.float64)


def _combine_1d_stouffer(pvalues_1d, weights, backend):
    p = _validate_1d_pvalues(pvalues_1d, backend)
    m = int(p.shape[0])
    if m == 0:
        raise ValueError("pvalues must contain at least one value")

    w = _validate_weights(weights, m, backend)

    eps = np.finfo(np.float64).eps
    p_safe = backend.xp.clip(backend.astype(p, backend.float64), eps, 1.0 - eps)
    z_scores = norm.ppf(1.0 - p_safe)
    z = backend.xp.sum(w * z_scores) / backend.xp.sqrt(backend.xp.sum(w * w))
    pvalue = norm.sf(z)
    pvalue = backend.xp.clip(pvalue, 0.0, 1.0)
    return backend.astype(z, backend.float64), backend.astype(pvalue, backend.float64)


def _combine_1d_pvalues(pvalues_1d, method: str, weights, backend):
    if method == "fisher":
        if weights is not None:
            raise ValueError(
                "weights are only supported for method='cauchy' or method='stouffer'"
            )
        return _combine_1d_fisher(pvalues_1d, backend)
    if method == "cauchy":
        return _combine_1d_cauchy(pvalues_1d, weights, backend)
    if method == "stouffer":
        return _combine_1d_stouffer(pvalues_1d, weights, backend)
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
        One of: 'bh', 'by', 'holm', 'bonferroni', 'hochberg'.
        Common aliases are accepted (e.g., 'fdr_bh', 'bonf', 'step_up').
    alpha : float, default=0.05
        Rejection threshold in (0, 1).
    axis : int or None, default=None
        Axis along which to adjust p-values.
        If None, adjusts over all values flattened.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
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
    backend = get_backend(backend_name)

    arr = backend.asarray(pvalues, dtype=backend.float64)

    if axis is None:
        flat = _validate_1d_pvalues(arr, backend)
        adj_flat = _adjust_1d_pvalues(flat, method_n, backend)
        reject_flat = adj_flat <= alpha_f
        return reject_flat.reshape(arr.shape), adj_flat.reshape(arr.shape)

    if arr.ndim == 0:
        raise ValueError("axis must be None for scalar pvalues")

    axis_n = _normalize_axis_index(axis, arr.ndim)
    moved = backend.xp.moveaxis(arr, axis_n, -1)
    matrix = moved.reshape(-1, moved.shape[-1])

    adj_matrix = backend.xp.empty_like(matrix, dtype=backend.float64)
    reject_matrix = backend.xp.empty_like(matrix, dtype=bool)

    for i in range(matrix.shape[0]):
        row = _validate_1d_pvalues(matrix[i], backend)
        adj_row = _adjust_1d_pvalues(row, method_n, backend)
        adj_matrix[i] = adj_row
        reject_matrix[i] = adj_row <= alpha_f

    adj_moved = adj_matrix.reshape(moved.shape)
    reject_moved = reject_matrix.reshape(moved.shape)

    return (
        backend.xp.moveaxis(reject_moved, -1, axis_n),
        backend.xp.moveaxis(adj_moved, -1, axis_n),
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
    method : {'fisher', 'cauchy', 'stouffer'}, default='fisher'
        Combination method. Aliases accepted (e.g., 'acat').
    weights : array-like, optional
        Optional non-negative weights for method='cauchy' or 'stouffer'.
    axis : int or None, default=None
        Axis along which to combine p-values. If None, flattens all values.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
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
    backend = get_backend(backend_name)

    arr = backend.asarray(pvalues, dtype=backend.float64)

    if axis is None:
        flat = _validate_1d_pvalues(arr, backend)
        return _combine_1d_pvalues(flat, method_n, weights, backend)

    if arr.ndim == 0:
        raise ValueError("axis must be None for scalar pvalues")

    arr = _validate_pvalues_array(arr, backend)
    axis_n = _normalize_axis_index(axis, arr.ndim)
    moved = backend.xp.moveaxis(arr, axis_n, -1)
    m = int(moved.shape[-1])
    if m == 0:
        raise ValueError("pvalues must contain at least one value")

    if method_n == "fisher":
        if weights is not None:
            raise ValueError("weights are only supported for method='cauchy' or method='stouffer'")
        eps = np.finfo(np.float64).tiny
        p_safe = backend.xp.clip(backend.astype(moved, backend.float64), eps, 1.0)
        statistics = -2.0 * backend.xp.sum(backend.xp.log(p_safe), axis=-1)
        pvals = chi2.sf(statistics, df=2 * m)
        return backend.astype(statistics, backend.float64), backend.astype(pvals, backend.float64)

    if method_n == "cauchy":
        w = _validate_weights(weights, m, backend)
        eps = np.finfo(np.float64).eps
        p_safe = backend.xp.clip(backend.astype(moved, backend.float64), eps, 1.0 - eps)
        w_shape = (1,) * (p_safe.ndim - 1) + (m,)
        statistics = backend.xp.sum(w.reshape(w_shape) * backend.xp.tan((0.5 - p_safe) * np.pi), axis=-1)
        pvals = 0.5 - backend.xp.arctan(statistics) / np.pi
        pvals = backend.xp.clip(pvals, 0.0, 1.0)
        return backend.astype(statistics, backend.float64), backend.astype(pvals, backend.float64)

    if method_n == "stouffer":
        w = _validate_weights(weights, m, backend)
        eps = np.finfo(np.float64).eps
        p_safe = backend.xp.clip(backend.astype(moved, backend.float64), eps, 1.0 - eps)
        z_scores = norm.ppf(1.0 - p_safe)
        w_shape = (1,) * (p_safe.ndim - 1) + (m,)
        statistics = backend.xp.sum(w.reshape(w_shape) * z_scores, axis=-1) / backend.xp.sqrt(backend.xp.sum(w * w))
        pvals = norm.sf(statistics)
        pvals = backend.xp.clip(pvals, 0.0, 1.0)
        return backend.astype(statistics, backend.float64), backend.astype(pvals, backend.float64)

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
