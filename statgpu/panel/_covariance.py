"""Covariance estimators and internal dispatch for panel data models.

Stage C of Issue #93 extends the Stage-A residual-OLS covariance registry with
HC0/HC2/HC3, explicit cluster group debiasing, and Driscoll-Kraay covariance.
Covariance breads are built from the design pseudoinverse rather than normal
equations so full-rank but ill-conditioned designs do not square the condition
number before inference.
"""
from __future__ import annotations

__all__ = [
    "clustered_covariance",
    "two_way_clustered_covariance",
    "hac_covariance",
    "driscoll_kraay_covariance",
    "normalize_covariance_type",
    "ols_covariance",
]

from typing import Optional

import numpy as np

from statgpu.backends import (
    _get_xp,
    _resolve_backend,
    _to_float_scalar,
    _to_numpy,
    xp_asarray,
    xp_zeros,
)
from statgpu.panel._linalg import (
    panel_svd_pseudoinverse,
    panel_svd_working_pseudoinverse,
)
from statgpu.panel._utils import factorize_panel_metadata


_COVARIANCE_ALIASES = {
    "hc1": "robust",
    "dk": "driscoll-kraay",
    "kernel": "driscoll-kraay",
}

_KERNEL_ALIASES = {
    "bartlett": "bartlett",
    "newey-west": "bartlett",
    "parzen": "parzen",
    "gallant": "parzen",
    "quadratic-spectral": "qs",
    "qs": "qs",
    "andrews": "qs",
}


def normalize_covariance_type(cov_type: str) -> str:
    """Return the canonical Stage-C covariance name."""
    name = str(cov_type).strip().lower()
    return _COVARIANCE_ALIASES.get(name, name)


def _ensure_xp(xp=None, *arrays):
    """Return an explicit array module or infer it from public inputs."""
    if xp is not None:
        return xp
    return _get_xp(_resolve_backend("auto", *arrays))


def _is_torch(xp) -> bool:
    return getattr(xp, "__name__", "") == "torch"


def _design_pseudoinverse(X, xp):
    """Compatibility wrapper around the shared panel SVD policy."""
    return panel_svd_pseudoinverse(X, xp)


def _gram_inverse(X, xp, *, rank_aware: bool = False):
    """Compatibility helper returning a stable generalized inverse of X'X.

    ``rank_aware`` is retained for the existing internal signature.  The stable
    implementation always derives the bread from ``X+``; it never attempts
    ``inv(X'X)`` merely because ``X`` is still classified as full rank.
    """
    del rank_aware
    _X_pinv, bread, rank = _design_pseudoinverse(X, xp)
    return bread, rank


def _column_abs_max(values, xp):
    if _is_torch(xp):
        return xp.max(xp.abs(values), dim=0).values
    return xp.max(xp.abs(values), axis=0)


def _projection_product_working_values(projection_rows, resid, xp):
    """Scale projection coordinates only when projection*residual could overflow.

    The scale is the smallest per-coordinate factor needed to leave a factor-two
    range margin.  If the raw multiplication is already safe, the projection is
    left bit-for-bit unchanged; in particular, ordinary/subnormal-design paths
    do not pay an avoidable normalization round trip.
    """
    projection_max = _column_abs_max(projection_rows, xp)
    resid_max = xp.max(xp.abs(resid))
    resid_bound = xp.maximum(resid_max, xp.ones_like(resid_max))
    safe_projection = (0.5 * float(np.finfo(np.float64).max)) / resid_bound
    one = xp.ones_like(projection_max)
    scale = xp.where(
        projection_max > safe_projection,
        projection_max / safe_projection,
        one,
    )
    return projection_rows / scale, scale


def _gram_working_values(values, xp, *, max_multiplier: float = 1.0):
    """Apply only the per-coordinate scaling required for a safe Gram reduction."""
    n_terms = max(1, int(values.shape[0]))
    multiplier = max(1.0, abs(float(max_multiplier)))
    threshold = float(
        np.sqrt(0.25 * np.finfo(np.float64).max / (float(n_terms) * multiplier))
    )
    max_abs = _column_abs_max(values, xp)
    one = xp.ones_like(max_abs)
    scale = xp.where(max_abs > threshold, max_abs / threshold, one)
    return values / scale, scale


def _common_gram_working_values(values_list, xp, *, max_multiplier: float = 1.0):
    """Put several grouped-score matrices on one minimally scaled Gram space."""
    max_abs = _column_abs_max(values_list[0], xp)
    n_terms = int(values_list[0].shape[0])
    for values in values_list[1:]:
        max_abs = xp.maximum(max_abs, _column_abs_max(values, xp))
        n_terms = max(n_terms, int(values.shape[0]))
    multiplier = max(1.0, abs(float(max_multiplier)))
    threshold = float(
        np.sqrt(0.25 * np.finfo(np.float64).max / (float(max(1, n_terms)) * multiplier))
    )
    one = xp.ones_like(max_abs)
    scale = xp.where(max_abs > threshold, max_abs / threshold, one)
    return [values / scale for values in values_list], scale


def _cross_reduction_is_safe(
    left, right, xp, *, max_multiplier: float = 1.0
) -> bool:
    """Return whether ``left.T @ right`` has a conservative rowwise range bound.

    Matrix-product terms pair values from the same observation/group row.  Use
    per-row infinity norms so unrelated column maxima do not reject a safe
    mixed-range cross term.  This is only a sufficient condition; uncertain
    cases remain on the established scaled covariance path.
    """
    n_terms = max(1, int(left.shape[0]))
    multiplier = max(1.0, abs(float(max_multiplier)))
    if _is_torch(xp):
        left_row_max = xp.max(xp.abs(left), dim=1).values
        right_row_max = xp.max(xp.abs(right), dim=1).values
    else:
        left_row_max = xp.max(xp.abs(left), axis=1)
        right_row_max = xp.max(xp.abs(right), axis=1)
    finite = xp.isfinite(left_row_max) & xp.isfinite(right_row_max)
    large = xp.maximum(left_row_max, right_row_max)
    small = xp.minimum(left_row_max, right_row_max)
    safe_large = xp.maximum(large, xp.ones_like(large))
    per_term_limit = (
        0.25
        * float(np.finfo(np.float64).max)
        / (float(n_terms) * multiplier)
    )
    safe = finite & (small <= float(per_term_limit) / safe_large)
    return bool(_to_float_scalar(xp.all(safe)))


def _influence_rows(X, resid, xp):
    """Return finite influence rows with only unavoidable product scaling delayed.

    The SVD working-design pseudoinverse is rescaled only if multiplying a
    coordinate by the largest finite residual could overflow.  The original
    residual vector is never globally normalized, so unrelated tiny/subnormal
    residual contributions survive beside very large observations.
    """
    X_work, X_pinv_work, design_scale, rank = panel_svd_working_pseudoinverse(
        X, xp
    )
    projection_rows = X_pinv_work.T
    projection_work, projection_scale = _projection_product_working_values(
        projection_rows, resid, xp
    )
    influence_work = projection_work * resid[:, None]
    return (
        influence_work,
        projection_scale,
        design_scale,
        X_pinv_work,
        X_work,
        rank,
    )


def _restore_coordinate_covariance(covariance, scale, xp):
    """Restore a per-coordinate covariance scale without forming scale_i*scale_j."""
    row = scale[:, None]
    col = scale[None, :]
    large = xp.maximum(row, col)
    small = xp.minimum(row, col)
    return _symmetrize((covariance * large) * small)


def _restore_scalar_covariance(covariance, scale, xp):
    """Restore one positive scalar score scale without materializing scale**2."""
    return _symmetrize((covariance * scale) * scale)


def _restore_influence_covariance(
    covariance, covariance_scale, projection_scale, design_scale, xp
):
    """Restore Gram, projection, then tiny-design scales after cancellation."""
    covariance = _restore_coordinate_covariance(covariance, covariance_scale, xp)
    covariance = _restore_coordinate_covariance(covariance, projection_scale, xp)
    return _restore_scalar_covariance(covariance, design_scale, xp)


def _cluster_grouped_scores(
    scores, codes, *, n_groups: int, nobs: int, group_debias: bool, xp,
    return_compensation: bool = False,
):
    if int(n_groups) < 2:
        raise ValueError(
            "clustered covariance requires at least two distinct clusters"
        )
    grouped = _grouped_score_sums(
        scores,
        codes,
        n_groups=int(n_groups),
        xp=xp,
        return_compensation=return_compensation,
    )
    correction = (
        _group_debias_factor(int(n_groups), int(nobs)) if group_debias else 1.0
    )
    if return_compensation:
        high, low = grouped
        return high, low, float(correction)
    return grouped, float(correction)


def _cluster_meat_from_grouped(grouped, correction: float, xp):
    grouped_work, grouped_scale = _gram_working_values(
        grouped, xp, max_multiplier=correction
    )
    meat = _symmetrize(grouped_work.T @ grouped_work * float(correction))
    return meat, grouped_scale


def _symmetrize(matrix):
    """Average a matrix with its transpose without avoidable overflow.

    ``0.5 * (a + b)`` preserves subnormal entries but can overflow when both
    finite operands are near DBL_MAX. Only those same-sign risky entries use
    ``0.5*a + 0.5*b``; ordinary and subnormal entries retain the direct sum.
    """
    xp = _ensure_xp(None, matrix)
    other = matrix.T
    abs_left = xp.abs(matrix)
    abs_right = xp.abs(other)
    same_sign = ((matrix >= 0.0) & (other >= 0.0)) | (
        (matrix < 0.0) & (other < 0.0)
    )
    risk = same_sign & (abs_left > float(np.finfo(np.float64).max) - abs_right)
    left = xp.where(risk, 0.5 * matrix, matrix)
    right = xp.where(risk, 0.5 * other, other)
    summed = left + right
    return xp.where(risk, summed, 0.5 * summed)


def _covariance_accumulator_start(initial, *, max_terms: int, xp):
    """Start a per-entry reduction-length covariance accumulator.

    Entries are scaled only when one of at most ``max_terms`` same-sign
    contributions could overflow a representable final reduction. Safe and
    subnormal entries remain on their original scale.
    """
    max_terms = max(1, int(max_terms))
    threshold = float(np.finfo(np.float64).max) / float(max_terms)
    scaled = xp.abs(initial) > threshold
    divisor = xp.where(
        scaled,
        xp.full_like(initial, float(max_terms)),
        xp.ones_like(initial),
    )
    return initial / divisor, scaled


def _covariance_accumulator_add(
    accumulator,
    scaled,
    base,
    multiplier: float,
    *,
    max_terms: int,
    xp,
):
    """Add ``multiplier * base`` without materializing a risky term/sum."""
    max_terms = max(1, int(max_terms))
    multiplier = float(multiplier)
    if multiplier == 0.0:
        return accumulator, scaled

    abs_multiplier = abs(multiplier)
    max_float = float(np.finfo(np.float64).max)
    accumulator_threshold = max_float / float(max_terms)
    base_threshold = max_float / (float(max_terms) * abs_multiplier)
    needs_scale = (~scaled) & (
        (xp.abs(accumulator) > accumulator_threshold)
        | (xp.abs(base) > base_threshold)
    )

    divisor = xp.where(
        needs_scale,
        xp.full_like(accumulator, float(max_terms)),
        xp.ones_like(accumulator),
    )
    accumulator = accumulator / divisor
    scaled = scaled | needs_scale
    coefficient = xp.where(
        scaled,
        xp.full_like(base, multiplier / float(max_terms)),
        xp.full_like(base, multiplier),
    )
    return accumulator + base * coefficient, scaled


def _covariance_accumulator_finish(accumulator, scaled, *, max_terms: int, xp):
    """Restore the original scale after safe lag-term accumulation."""
    max_terms = max(1, int(max_terms))
    factor = xp.where(
        scaled,
        xp.full_like(accumulator, float(max_terms)),
        xp.ones_like(accumulator),
    )
    return accumulator * factor


def _stable_inclusion_exclusion(V1, V2, V12, xp):
    """Return ``V1 + V2 - V12`` with cancellation before risky addition.

    If either marginal entry has the same sign as the intersection entry,
    subtract that pair first; subtraction of same-sign finite values cannot
    overflow. If neither does, all three inclusion-exclusion terms have the same
    sign and any overflow in their direct sum reflects an unrepresentable result.
    """
    same1 = ((V1 >= 0.0) & (V12 >= 0.0)) | ((V1 < 0.0) & (V12 < 0.0))
    same2 = ((V2 >= 0.0) & (V12 >= 0.0)) | ((V2 < 0.0) & (V12 < 0.0))
    use1 = same1
    use2 = (~use1) & same2
    stable_mask = use1 | use2

    first = xp.where(use1, V1, V2)
    remaining = xp.where(use1, V2, V1)
    first_safe = xp.where(stable_mask, first, xp.zeros_like(first))
    remaining_safe = xp.where(stable_mask, remaining, xp.zeros_like(remaining))
    intersection_safe = xp.where(stable_mask, V12, xp.zeros_like(V12))
    stable = (first_safe - intersection_safe) + remaining_safe

    fallback_V1 = xp.where(stable_mask, xp.zeros_like(V1), V1)
    fallback_V2 = xp.where(stable_mask, xp.zeros_like(V2), V2)
    fallback_V12 = xp.where(stable_mask, xp.zeros_like(V12), V12)
    direct = (fallback_V1 + fallback_V2) - fallback_V12
    return xp.where(stable_mask, stable, direct)



def _grouped_score_sums(
    scores, codes_np, *, n_groups: int, xp, return_compensation: bool = False
):
    "Sum scores by group while preserving recursively separated magnitude tiers."
    codes_np = np.asarray(codes_np, dtype=np.int64).ravel()
    if codes_np.shape[0] != int(scores.shape[0]):
        raise ValueError("group codes must match the number of score rows")
    if int(n_groups) <= 0:
        raise ValueError("at least one group is required")
    codes = xp_asarray(
        codes_np,
        dtype=xp.int64,
        xp=xp,
        ref_arr=scores,
    )
    shape = (int(n_groups), int(scores.shape[1]))
    index = (
        codes.unsqueeze(1).expand_as(scores)
        if hasattr(codes, "unsqueeze")
        else None
    )

    def _group_abs_max(values):
        out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=scores)
        absolute = xp.abs(values)
        if hasattr(out, "scatter_reduce_"):
            out.scatter_reduce_(
                0, index, absolute, reduce="amax", include_self=True
            )
        elif type(out).__module__.startswith("cupy"):
            xp.maximum.at(out, codes, absolute)
        else:
            np.maximum.at(out, codes_np, absolute)
        return out

    def _signed_parts(values):
        positive = xp.where(values > 0.0, values, xp.zeros_like(values))
        negative = xp.where(values < 0.0, values, xp.zeros_like(values))
        positive_out = xp_zeros(
            shape, dtype=xp.float64, xp=xp, ref_arr=scores
        )
        negative_out = xp_zeros(
            shape, dtype=xp.float64, xp=xp, ref_arr=scores
        )
        if hasattr(positive_out, "scatter_add_"):
            positive_out.scatter_add_(0, index, positive)
            negative_out.scatter_add_(0, index, negative)
        elif type(positive_out).__module__.startswith("cupy"):
            xp.add.at(positive_out, codes, positive)
            xp.add.at(negative_out, codes, negative)
        else:
            np.add.at(positive_out, codes_np, positive)
            np.add.at(negative_out, codes_np, negative)
        return positive_out, negative_out

    def _two_sum(left, right):
        summed = left + right
        virtual_right = summed - left
        residual = (
            left - (summed - virtual_right)
        ) + (right - virtual_right)
        return summed, residual

    max_abs = _group_abs_max(scores)
    counts_np = np.bincount(
        codes_np, minlength=int(n_groups)
    ).astype(np.float64)
    counts = xp_asarray(
        counts_np,
        dtype=xp.float64,
        xp=xp,
        ref_arr=scores,
    ).reshape(-1, 1)
    limit = float(np.finfo(np.float64).max) / counts
    factor = xp.where(max_abs > limit, counts, xp.ones_like(max_abs))
    remaining = scores / factor[codes]
    split_ratio = xp.minimum(
        counts * float(8.0 * np.finfo(np.float64).eps),
        xp.full_like(counts, 0.25),
    )

    tiers = []
    for _ in range(128):
        tier_max = _group_abs_max(remaining)
        threshold = tier_max * split_ratio
        tail_mask = (
            (xp.abs(remaining) < threshold[codes])
            & (remaining != 0.0)
        )
        main = xp.where(tail_mask, xp.zeros_like(remaining), remaining)
        positive_out, negative_out = _signed_parts(main)
        tier_sum, tier_error = _two_sum(positive_out, negative_out)
        tiers.append((tier_sum, tier_error))
        if not bool(_to_float_scalar(xp.any(tail_mask))):
            break
        remaining = xp.where(
            tail_mask, remaining, xp.zeros_like(remaining)
        )
    else:
        raise RuntimeError(
            "grouped score cancellation exceeded the float64 tier budget"
        )

    def _collapse(parts):
        total = xp.zeros_like(tiers[0][0])
        correction = xp.zeros_like(total)
        for part in parts:
            summed, error = _two_sum(total, part)
            total = summed
            correction = correction + error
        summed, error = _two_sum(total, correction)
        return summed + error

    lower_parts = []
    for tier_sum, tier_error in reversed(tiers[1:]):
        lower_parts.extend((tier_error, tier_sum))
    lower_parts.append(tiers[0][1])
    low = _collapse(lower_parts)
    high = tiers[0][0]

    if return_compensation:
        return high * factor, low * factor

    summed, error = _two_sum(high, low)
    return (summed + error) * factor


def _factorize_1d_labels(values, *, nobs: int, name: str):
    labels, codes = factorize_panel_metadata(
        values, name=name, expected_n=int(nobs)
    )
    return labels, codes


def _paired_codes(left, right):
    pairs = np.column_stack(
        [np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)]
    )
    _, codes = np.unique(pairs, axis=0, return_inverse=True)
    return codes.astype(np.int64, copy=False)


def _same_partition(left, right) -> bool:
    """Return whether two integer code vectors induce the same partition.

    Code values themselves are arbitrary labels.  Two partitions are equal iff
    they have the same number of groups and each observed left/right pair gives
    a one-to-one mapping between those groups.
    """
    left = np.asarray(left, dtype=np.int64).ravel()
    right = np.asarray(right, dtype=np.int64).ravel()
    if left.shape != right.shape:
        return False
    n_left = int(np.unique(left).size)
    n_right = int(np.unique(right).size)
    if n_left != n_right:
        return False
    pairs = np.column_stack([left, right])
    return int(np.unique(pairs, axis=0).shape[0]) == n_left


def _group_debias_factor(n_groups: int, nobs: int) -> float:
    n_groups = int(n_groups)
    nobs = int(nobs)
    if n_groups < 2:
        raise ValueError("group_debias requires at least two groups")
    if nobs <= 0:
        raise ValueError("group_debias requires a positive observation count")
    return (n_groups / (n_groups - 1.0)) * ((nobs - 1.0) / nobs)


def _validate_group_debias(value) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("group_debias must be boolean")
    return bool(value)


def clustered_covariance(
    X,
    resid,
    clusters,
    xp=None,
    *,
    group_debias: bool = False,
    metadata: Optional[dict] = None,
):
    """One-way clustered robust covariance matrix."""
    xp = _ensure_xp(xp, X)
    group_debias = _validate_group_debias(group_debias)

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    n, _k = X.shape
    labels, cluster_idx = _factorize_1d_labels(
        clusters, nobs=int(n), name="clusters"
    )
    if resid.shape[0] != n:
        raise ValueError("X and resid must have the same number of observations")

    (
        influence,
        projection_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    n_clusters = int(len(labels))
    grouped, correction = _cluster_grouped_scores(
        influence,
        cluster_idx,
        n_groups=n_clusters,
        nobs=int(n),
        group_debias=group_debias,
        xp=xp,
    )
    cov_work, grouped_scale = _cluster_meat_from_grouped(
        grouped, correction, xp
    )
    cov = _restore_influence_covariance(
        cov_work, grouped_scale, projection_scale, design_scale, xp
    )
    if metadata is not None:
        metadata.update(
            {
                "cluster_dimensions": 1,
                "cluster_group_counts": [n_clusters],
                "group_debias": bool(group_debias),
                "group_debias_factors": [float(correction)],
            }
        )
    return cov


def two_way_clustered_covariance(
    X,
    resid,
    cluster1,
    cluster2,
    xp=None,
    *,
    group_debias: bool = False,
    metadata: Optional[dict] = None,
):
    """Two-way clustered covariance with exact intersection factorization."""
    xp = _ensure_xp(xp, X)
    group_debias = _validate_group_debias(group_debias)
    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])
    labels1, c1 = _factorize_1d_labels(cluster1, nobs=n, name="cluster1")
    labels2, c2 = _factorize_1d_labels(cluster2, nobs=n, name="cluster2")
    c12 = _paired_codes(c1, c2)
    n12 = int(np.max(c12)) + 1 if c12.size else 0

    # All Cameron-Gelbach-Miller grouping is performed before covariance-scale
    # normalization. Nested cluster dimensions are simplified algebraically, so
    # an exactly cancelling marginal/intersection pair never has to be rounded.
    (
        influence,
        projection_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    nested_c1 = _same_partition(c12, c1)
    nested_c2 = _same_partition(c12, c2)
    if nested_c1 or nested_c2:
        grouped1, correction1 = _cluster_grouped_scores(
            influence, c1, n_groups=int(len(labels1)), nobs=n,
            group_debias=group_debias, xp=xp
        )
        grouped2, correction2 = _cluster_grouped_scores(
            influence, c2, n_groups=int(len(labels2)), nobs=n,
            group_debias=group_debias, xp=xp
        )
        grouped12, correction12 = _cluster_grouped_scores(
            influence, c12, n_groups=n12, nobs=n,
            group_debias=group_debias, xp=xp
        )
    else:
        grouped1, grouped1_low, correction1 = _cluster_grouped_scores(
            influence, c1, n_groups=int(len(labels1)), nobs=n,
            group_debias=group_debias, xp=xp, return_compensation=True
        )
        grouped2, grouped2_low, correction2 = _cluster_grouped_scores(
            influence, c2, n_groups=int(len(labels2)), nobs=n,
            group_debias=group_debias, xp=xp, return_compensation=True
        )
        grouped12, grouped12_low, correction12 = _cluster_grouped_scores(
            influence, c12, n_groups=n12, nobs=n,
            group_debias=group_debias, xp=xp, return_compensation=True
        )
    if nested_c1:
        cov_work, common_scale = _cluster_meat_from_grouped(
            grouped2, correction2, xp
        )
    elif nested_c2:
        cov_work, common_scale = _cluster_meat_from_grouped(
            grouped1, correction1, xp
        )
    else:
        multiplier = max(correction1, correction2, correction12)
        (grouped1_work, grouped2_work, grouped12_work), common_scale = (
            _common_gram_working_values(
                [grouped1, grouped2, grouped12],
                xp,
                max_multiplier=multiplier,
            )
        )
        V1_work = _symmetrize(
            grouped1_work.T @ grouped1_work * float(correction1)
        )
        V2_work = _symmetrize(
            grouped2_work.T @ grouped2_work * float(correction2)
        )
        V12_work = _symmetrize(
            grouped12_work.T @ grouped12_work * float(correction12)
        )
        cov_work = _stable_inclusion_exclusion(V1_work, V2_work, V12_work, xp)

        # The grouped score is represented as a float64 high part plus the
        # exact TwoSum residual from its final positive/negative combination.
        # This rounding loss occurs at grouping time and is independent of
        # whether the subsequent Gram needs range scaling, so compensate any
        # nonzero low part in the nonnested CGM combination.  Each component
        # keeps its own finite-sample group-debias factor.
        low_max = xp.maximum(
            xp.maximum(
                xp.max(xp.abs(grouped1_low)),
                xp.max(xp.abs(grouped2_low)),
            ),
            xp.max(xp.abs(grouped12_low)),
        )
        need_compensation = _to_float_scalar(low_max) > 0.0
        if need_compensation:
            # First try to keep the low-order expansion in the *same* coordinate
            # scale as the high Gram.  This is the important path when individual
            # physical high-low cross terms would overflow but the CGM
            # inclusion-exclusion is finite: cancellation happens before any
            # physical-scale restoration instead of silently dropping the tail.
            low1_work = grouped1_low / common_scale
            low2_work = grouped2_low / common_scale
            low12_work = grouped12_low / common_scale
            low_work_triples = (
                (grouped1_low, low1_work),
                (grouped2_low, low2_work),
                (grouped12_low, low12_work),
            )
            low_underflowed = any(
                bool(
                    _to_float_scalar(
                        xp.any((low != 0.0) & (low_work == 0.0))
                    )
                )
                for low, low_work in low_work_triples
            )

            if not low_underflowed:
                def _low_order_covariance_work(high_work, low_work, correction):
                    cross = high_work.T @ low_work
                    low_square = _symmetrize(low_work.T @ low_work)
                    return _symmetrize(
                        (
                            cross
                            + cross.T
                            + low_square
                        )
                        * float(correction)
                    )

                low_correction_work = _stable_inclusion_exclusion(
                    _low_order_covariance_work(
                        grouped1_work, low1_work, correction1
                    ),
                    _low_order_covariance_work(
                        grouped2_work, low2_work, correction2
                    ),
                    _low_order_covariance_work(
                        grouped12_work, low12_work, correction12
                    ),
                    xp,
                )
                cov_work = _symmetrize(cov_work + low_correction_work)
            else:
                # If dividing by the high-Gram coordinate scale would erase a
                # nonzero low part, preserve it on the physical score scale.
                # Such a fallback is used only after proving every required
                # high-low and low-low reduction safe.  Never fail open by
                # omitting the low component.
                correction_triples = (
                    (grouped1, grouped1_low, correction1),
                    (grouped2, grouped2_low, correction2),
                    (grouped12, grouped12_low, correction12),
                )
                safe_correction = all(
                    _cross_reduction_is_safe(
                        high, low, xp, max_multiplier=correction
                    )
                    and _cross_reduction_is_safe(
                        low, low, xp, max_multiplier=correction
                    )
                    for high, low, correction in correction_triples
                )
                if not safe_correction:
                    raise FloatingPointError(
                        "two-way cluster low-order correction cannot be "
                        "evaluated safely without losing a nonzero tail"
                    )

                def _low_order_covariance(high, low, correction):
                    cross = high.T @ low
                    low_square = _symmetrize(low.T @ low)
                    return _symmetrize(
                        (
                            cross
                            + cross.T
                            + low_square
                        )
                        * float(correction)
                    )

                low_correction = _stable_inclusion_exclusion(
                    _low_order_covariance(
                        grouped1, grouped1_low, correction1
                    ),
                    _low_order_covariance(
                        grouped2, grouped2_low, correction2
                    ),
                    _low_order_covariance(
                        grouped12, grouped12_low, correction12
                    ),
                    xp,
                )
                cov_work = _restore_coordinate_covariance(
                    cov_work, common_scale, xp
                )
                cov_work = _symmetrize(cov_work + low_correction)
                common_scale = xp.ones_like(projection_scale)
    cov = _restore_influence_covariance(
        cov_work, common_scale, projection_scale, design_scale, xp
    )
    if metadata is not None:
        metadata.update(
            {
                "cluster_dimensions": 2,
                "cluster_group_counts": [
                    int(len(labels1)),
                    int(len(labels2)),
                    n12,
                ],
                "group_debias": bool(group_debias),
                "group_debias_factors": [
                    correction1,
                    correction2,
                    correction12,
                ],
            }
        )
    return cov


def hac_covariance(X, resid, bandwidth=None, kernel="bartlett", xp=None):
    """Historical row-order Newey-West HAC covariance using Bartlett weights."""
    xp = _ensure_xp(xp, X)
    if str(kernel).lower() != "bartlett":
        raise ValueError("kernel must be 'bartlett'")
    if bandwidth is not None:
        if isinstance(bandwidth, bool) or not isinstance(
            bandwidth, (int, np.integer)
        ):
            raise ValueError("bandwidth must be a non-negative integer or None")
        if int(bandwidth) < 0:
            raise ValueError("bandwidth must be a non-negative integer or None")

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])

    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(int(bandwidth), n - 1))

    (
        influence,
        projection_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _gram_working_values(
        influence, xp, max_multiplier=2.0
    )
    max_terms = int(bandwidth) + 1
    cov, scaled = _covariance_accumulator_start(
        _symmetrize(influence_work.T @ influence_work), max_terms=max_terms, xp=xp
    )
    for h in range(1, bandwidth + 1):
        w = 1.0 - h / (bandwidth + 1.0)
        gamma_h = influence_work[h:].T @ influence_work[: n - h]
        cov, scaled = _covariance_accumulator_add(
            cov,
            scaled,
            _symmetrize(gamma_h),
            2.0 * float(w),
            max_terms=max_terms,
            xp=xp,
        )
    cov = _covariance_accumulator_finish(
        cov, scaled, max_terms=max_terms, xp=xp
    )
    return _restore_influence_covariance(
        cov, influence_scale, projection_scale, design_scale, xp
    )


def _canonical_kernel(kernel: str) -> str:
    name = str(kernel).strip().lower()
    if name not in _KERNEL_ALIASES:
        choices = ", ".join(sorted(_KERNEL_ALIASES))
        raise ValueError(
            f"unsupported Driscoll-Kraay kernel {kernel!r}; expected one of: {choices}"
        )
    return _KERNEL_ALIASES[name]


def _validate_dk_bandwidth(bandwidth, n_periods: int) -> int:
    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0)))
    if isinstance(bandwidth, bool) or not isinstance(
        bandwidth, (int, np.integer)
    ):
        raise ValueError(
            "Driscoll-Kraay bandwidth must be a non-negative integer or None"
        )
    bandwidth = int(bandwidth)
    if bandwidth < 0:
        raise ValueError(
            "Driscoll-Kraay bandwidth must be a non-negative integer or None"
        )
    # Explicit oversized bandwidths are smoothing parameters, not silently
    # capped lags. Only observed lags can contribute to the covariance.
    return bandwidth


def _dk_kernel_weights(
    kernel: str, bandwidth: int, max_lag: int
) -> tuple[str, np.ndarray]:
    """Return canonical DK kernel name and weights for lags 0..max_lag."""
    canonical = _canonical_kernel(kernel)
    max_lag = int(max_lag)
    bandwidth = int(bandwidth)
    weights = np.zeros(max_lag + 1, dtype=np.float64)
    weights[0] = 1.0
    if max_lag == 0 or bandwidth == 0:
        return canonical, weights

    if canonical == "bartlett":
        stop = min(bandwidth, max_lag)
        lag = np.arange(1, stop + 1, dtype=np.float64)
        weights[1 : stop + 1] = 1.0 - lag / (bandwidth + 1.0)
        return canonical, weights

    if canonical == "parzen":
        stop = min(bandwidth, max_lag)
        lag = np.arange(1, stop + 1, dtype=np.float64)
        z = lag / (bandwidth + 1.0)
        low = z <= 0.5
        w = np.empty_like(z)
        w[low] = 1.0 - 6.0 * z[low] ** 2 + 6.0 * z[low] ** 3
        w[~low] = 2.0 * (1.0 - z[~low]) ** 3
        weights[1 : stop + 1] = w
        return canonical, weights

    # Quadratic spectral is not truncated at bandwidth.  Its direct expression
    # catastrophically cancels for x -> 0, so use the analytic series there:
    # 3/x^2 (sin(x)/x - cos(x))
    #   = 1 - x^2/10 + x^4/280 - x^6/15120 + O(x^8).
    lag = np.arange(1, max_lag + 1, dtype=np.float64)
    x = 6.0 * np.pi * lag / (5.0 * bandwidth)
    small = np.abs(x) < 1.0e-3
    w = np.empty_like(x)
    x2 = x[small] * x[small]
    w[small] = 1.0 - x2 / 10.0 + x2 * x2 / 280.0 - x2 * x2 * x2 / 15120.0
    regular = ~small
    xr = x[regular]
    w[regular] = 3.0 / (xr * xr) * (np.sin(xr) / xr - np.cos(xr))
    weights[1:] = w
    return canonical, weights


def driscoll_kraay_covariance(
    X,
    resid,
    time_ids,
    *,
    bandwidth=None,
    kernel="bartlett",
    extra_df: int = 0,
    xp=None,
    metadata: Optional[dict] = None,
):
    """Driscoll-Kraay covariance on fit-space influence scores."""
    xp = _ensure_xp(xp, X)
    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])
    labels, time_codes = _factorize_1d_labels(
        time_ids, nobs=n, name="time_ids"
    )
    n_periods = int(len(labels))
    if n_periods < 2:
        raise ValueError(
            "Driscoll-Kraay covariance requires at least two time periods"
        )

    if isinstance(extra_df, bool) or not isinstance(extra_df, (int, np.integer)):
        raise ValueError("extra_df must be a non-negative integer")
    extra_df = int(extra_df)
    if extra_df < 0:
        raise ValueError("extra_df must be a non-negative integer")

    (
        influence,
        projection_scale,
        design_scale,
        _X_pinv,
        _X_work,
        rank,
    ) = _influence_rows(X, resid, xp)
    k_columns = int(X.shape[1])
    rank_deficient = rank < k_columns
    df_model = rank if rank_deficient else k_columns
    denom = n - extra_df - df_model
    if denom <= 0:
        raise ValueError(
            "Driscoll-Kraay covariance requires positive debiased residual degrees of freedom"
        )

    grouped = _grouped_score_sums(
        influence, time_codes, n_groups=n_periods, xp=xp
    )
    grouped_work, grouped_scale = _gram_working_values(
        grouped, xp, max_multiplier=2.0
    )
    bw = _validate_dk_bandwidth(bandwidth, n_periods)
    canonical_kernel, weights_np = _dk_kernel_weights(
        kernel, bw, n_periods - 1
    )
    weights = xp_asarray(
        weights_np,
        dtype=xp.float64,
        xp=xp,
        ref_arr=grouped_work,
    )

    max_terms = 1 + int(np.count_nonzero(weights_np[1:]))
    cov, scaled = _covariance_accumulator_start(
        _symmetrize(grouped_work.T @ grouped_work), max_terms=max_terms, xp=xp
    )
    for lag in range(1, n_periods):
        if weights_np[lag] == 0.0:
            continue
        gamma = grouped_work[lag:].T @ grouped_work[: n_periods - lag]
        cov, scaled = _covariance_accumulator_add(
            cov,
            scaled,
            _symmetrize(gamma),
            2.0 * float(weights_np[lag]),
            max_terms=max_terms,
            xp=xp,
        )
    cov = _covariance_accumulator_finish(
        cov, scaled, max_terms=max_terms, xp=xp
    )

    cov = _restore_influence_covariance(
        cov, grouped_scale, projection_scale, design_scale, xp
    )
    scale = float(n) / float(denom)
    cov = _symmetrize(scale * cov)
    if metadata is not None:
        nonzero_lags = np.flatnonzero(np.abs(weights_np[1:]) > 0.0) + 1
        metadata.update(
            {
                "covariance": "driscoll-kraay",
                "kernel": canonical_kernel,
                "bandwidth": int(bw),
                "n_periods": n_periods,
                "max_weighted_lag": int(nonzero_lags.max())
                if nonzero_lags.size
                else 0,
                "all_observed_lags_weighted": bool(
                    canonical_kernel == "qs" and bw > 0
                ),
                "extra_df": int(extra_df),
                "design_rank": int(rank),
                "design_columns": int(k_columns),
                "rank_deficient_extension": bool(rank_deficient),
                "df_scale": float(scale),
            }
        )
    return cov


def _hc_covariance(
    X, resid, *, kind: str, xp, metadata: Optional[dict] = None
):
    (
        influence,
        projection_scale,
        design_scale,
        X_pinv_work,
        X_work,
        rank,
    ) = _influence_rows(X, resid, xp)
    if kind == "hc0":
        influence_work, influence_scale = _gram_working_values(influence, xp)
        cov_work = _symmetrize(influence_work.T @ influence_work)
        if metadata is not None:
            metadata.update(
                {
                    "covariance": "hc0",
                    "design_rank": int(rank),
                    "design_columns": int(X.shape[1]),
                }
            )
        return _restore_influence_covariance(
            cov_work, influence_scale, projection_scale, design_scale, xp
        )

    projection_rows = X_pinv_work.T
    # Leverage is invariant to the uniform working-design rescaling:
    # X X+ = X_work X_work+.  Evaluate it entirely at the safe working scale.
    if _is_torch(xp):
        leverage = xp.sum(X_work * projection_rows, dim=1)
    else:
        leverage = xp.sum(X_work * projection_rows, axis=1)
    leverage_min = _to_float_scalar(xp.min(leverage))
    leverage_max = _to_float_scalar(xp.max(leverage))
    tol = 4096.0 * np.finfo(np.float64).eps
    if leverage_min < -tol:
        raise ValueError("HC2/HC3 leverage is materially negative")
    if leverage_max > 1.0 + tol:
        raise ValueError("HC2/HC3 leverage is materially greater than one")
    if _is_torch(xp):
        leverage = xp.clamp(leverage, min=0.0, max=1.0)
    else:
        leverage = xp.clip(leverage, 0.0, 1.0)
    denominator = 1.0 - leverage
    denominator_min = _to_float_scalar(xp.min(denominator))
    if denominator_min <= tol:
        raise ValueError(
            "HC2/HC3 covariance is undefined when leverage is numerically one"
        )
    if kind == "hc2":
        adjusted_influence = influence / xp.sqrt(denominator)[:, None]
    elif kind == "hc3":
        adjusted_influence = influence / denominator[:, None]
    else:
        raise ValueError(f"unknown HC covariance kind {kind!r}")
    adjusted_work, adjusted_scale = _gram_working_values(
        adjusted_influence, xp
    )
    cov_work = _symmetrize(adjusted_work.T @ adjusted_work)
    if metadata is not None:
        metadata.update(
            {
                "covariance": kind,
                "design_rank": int(rank),
                "design_columns": int(X.shape[1]),
                "leverage_min": float(leverage_min),
                "leverage_max": float(leverage_max),
            }
        )
    return _restore_influence_covariance(
        cov_work, adjusted_scale, projection_scale, design_scale, xp
    )


def ols_covariance(
    X,
    resid,
    *,
    cov_type,
    scale=None,
    df_resid=None,
    cluster=None,
    time_ids=None,
    bandwidth=None,
    kernel="bartlett",
    group_debias: bool = False,
    extra_df: int = 0,
    xp=None,
    allowed=None,
    hc1_correction=None,
    metadata: Optional[dict] = None,
):
    """Dispatch residual-based panel covariance definitions."""
    xp = _ensure_xp(xp, X)
    group_debias = _validate_group_debias(group_debias)
    name = normalize_covariance_type(cov_type)
    if allowed is not None:
        allowed_names = {normalize_covariance_type(value) for value in allowed}
        if name not in allowed_names:
            choices = ", ".join(sorted(str(value) for value in allowed_names))
            raise ValueError(
                f"cov_type={cov_type!r} is not supported here; expected one of: {choices}"
            )

    if group_debias and name != "clustered":
        raise ValueError("group_debias=True requires cov_type='clustered'")

    X = xp_asarray(X, dtype=xp.float64, xp=xp)
    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()
    if X.ndim != 2 or resid.shape[0] != X.shape[0]:
        raise ValueError("X and resid must have matching observation counts")
    n = int(X.shape[0])

    if metadata is not None:
        metadata.clear()
        metadata["covariance"] = name

    if name == "nonrobust":
        if scale is None:
            raise ValueError("scale is required for nonrobust covariance")
        scale_value = float(scale)
        if np.isnan(scale_value) or scale_value < 0.0:
            raise ValueError("scale must be non-negative and not NaN")
        _X_work, X_pinv_work, design_scale, _rank = (
            panel_svd_working_pseudoinverse(X, xp)
        )

        suspicious_scale = (
            not np.isfinite(scale_value)
            or (0.0 < scale_value < np.finfo(np.float64).tiny)
            or (scale_value == 0.0 and _to_float_scalar(xp.max(xp.abs(resid))) > 0.0)
        )
        if suspicious_scale:
            if df_resid is None or int(df_resid) <= 0:
                raise ValueError(
                    "positive df_resid is required when nonrobust scale must be "
                    "reconstructed from residuals"
                )
            resid_scale = xp.max(xp.abs(resid))
            resid_scale_value = _to_float_scalar(resid_scale)
            if resid_scale_value == 0.0:
                scaled_pinv = X_pinv_work * 0.0
            else:
                from statgpu.panel._diagnostics import _scaled_unit_values

                resid_unit = _scaled_unit_values(resid, resid_scale, xp)
                norm_sq = _to_float_scalar(xp.sum(resid_unit * resid_unit))
                rms_unit = float(np.sqrt(norm_sq / float(int(df_resid))))
                # sqrt(scale) = resid_scale * rms_unit. Apply the dimensionless
                # RMS factor before the potentially large residual/design-scale
                # restoration so a representable final covariance does not
                # overflow an intermediate solely because of multiplication order.
                scaled_pinv = (
                    ((X_pinv_work * rms_unit) * resid_scale) * design_scale
                )
            if metadata is not None:
                metadata["nonrobust_scale_reconstructed"] = True
                metadata["residual_scale"] = float(resid_scale_value)
        else:
            # scale * X+ X+^T = A A^T with
            # A = sqrt(scale) * design_scale * X_work+.
            scaled_pinv = (
                X_pinv_work * float(np.sqrt(scale_value))
            ) * design_scale
        return _symmetrize(scaled_pinv @ scaled_pinv.T)

    if name == "robust":
        (
            influence,
            projection_scale,
            design_scale,
            _X_pinv,
            _X_work,
            _rank,
        ) = _influence_rows(X, resid, xp)
        correction = hc1_correction
        if correction is None:
            if df_resid is None or int(df_resid) <= 0:
                raise ValueError(
                    "positive df_resid or hc1_correction is required for robust covariance"
                )
            correction = n / float(df_resid)
        if metadata is not None:
            metadata.update(
                {
                    "covariance": "robust",
                    "hc_equivalent": "hc1",
                    "hc1_correction": float(correction),
                }
            )
        influence_work, influence_scale = _gram_working_values(
            influence, xp, max_multiplier=float(correction)
        )
        cov_work = _symmetrize(
            influence_work.T @ influence_work * float(correction)
        )
        return _restore_influence_covariance(
            cov_work, influence_scale, projection_scale, design_scale, xp
        )

    if name in {"hc0", "hc2", "hc3"}:
        return _hc_covariance(
            X, resid, kind=name, xp=xp, metadata=metadata
        )

    if name == "clustered":
        if cluster is None:
            raise ValueError("cluster is required for cov_type='clustered'")
        cluster_np = np.asarray(_to_numpy(cluster))
        if cluster_np.ndim == 2 and cluster_np.shape[1] == 2:
            return two_way_clustered_covariance(
                X,
                resid,
                cluster_np[:, 0],
                cluster_np[:, 1],
                xp=xp,
                group_debias=group_debias,
                metadata=metadata,
            )
        if cluster_np.ndim == 2 and cluster_np.shape[1] == 1:
            cluster_np = cluster_np[:, 0]
        if cluster_np.ndim != 1:
            raise ValueError(
                "cluster must be one-dimensional, (n, 1), or (n, 2)"
            )
        return clustered_covariance(
            X,
            resid,
            cluster_np,
            xp=xp,
            group_debias=group_debias,
            metadata=metadata,
        )

    if name == "hac":
        if metadata is not None:
            metadata.update(
                {
                    "covariance": "hac",
                    "kernel": "bartlett",
                    "bandwidth": bandwidth,
                    "row_order_hac": True,
                }
            )
        return hac_covariance(
            X, resid, bandwidth=bandwidth, kernel=kernel, xp=xp
        )

    if name == "driscoll-kraay":
        if time_ids is None:
            raise ValueError(
                "time_ids is required for Driscoll-Kraay covariance"
            )
        return driscoll_kraay_covariance(
            X,
            resid,
            time_ids,
            bandwidth=bandwidth,
            kernel=kernel,
            extra_df=extra_df,
            xp=xp,
            metadata=metadata,
        )

    raise ValueError(
        "cov_type must be one of 'nonrobust', 'robust', 'hc0', 'hc1', "
        "'hc2', 'hc3', 'clustered', 'hac', or 'driscoll-kraay'"
    )
