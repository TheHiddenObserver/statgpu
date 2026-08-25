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
    panel_working_pseudoinverse,
)
from statgpu.panel._reductions import grouped_score_sums as _grouped_score_sums
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


def _validate_covariance_finite_inputs(X, resid, xp):
    """Fail closed before non-finite scores enter signed/group reductions."""
    finite = xp.all(xp.isfinite(X)) & xp.all(xp.isfinite(resid))
    if not bool(_to_float_scalar(finite)):
        raise ValueError("X and resid must contain only finite values")


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
    X_work, X_pinv_work, design_scale, rank = panel_working_pseudoinverse(
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


def _range_safe_float_product(values, factors, xp):
    """Multiply finite float64 factors without a range-unsafe association.

    Decompose each operand with ``frexp`` and accumulate exponents separately.
    Every mantissa stays in a compact interval, so compensating large/small
    factors can cancel in exponent space before the one final ``ldexp``.
    """
    mantissa, exponent = xp.frexp(values)
    for factor in factors:
        if np.isscalar(factor):
            factor_mantissa, factor_exponent = np.frexp(float(factor))
        else:
            factor_mantissa, factor_exponent = xp.frexp(factor)
        mantissa = mantissa * factor_mantissa
        exponent = exponent + factor_exponent
    mantissa, normalization_exponent = xp.frexp(mantissa)
    exponent = exponent + normalization_exponent
    zero = mantissa == 0.0
    underflow = exponent < -1073
    overflow = exponent > 1024
    if getattr(xp, "__name__", "") == "torch":
        safe_exponent = xp.clamp(exponent, min=-1073, max=1024)
        pow_fn = xp.pow
    else:
        safe_exponent = xp.clip(exponent, -1073, 1024)
        pow_fn = xp.power
    first_exponent = safe_exponent // 2
    second_exponent = safe_exponent - first_exponent
    base = xp.full_like(mantissa, 2.0)
    result = mantissa * pow_fn(base, first_exponent)
    result = result * pow_fn(base, second_exponent)
    result = xp.where(zero | underflow, xp.zeros_like(result), result)
    infinity = xp.full_like(result, float("inf"))
    signed_infinity = xp.where(mantissa < 0.0, -infinity, infinity)
    return xp.where(overflow & (~zero), signed_infinity, result)


def _restore_coordinate_covariance(covariance, scale, xp):
    """Restore a per-coordinate covariance scale without transient range loss."""
    row = scale[:, None]
    col = scale[None, :]
    return _symmetrize(_range_safe_float_product(covariance, (row, col), xp))


def _restore_scalar_covariance(covariance, scale, xp):
    """Restore one positive scalar score scale without transient range loss."""
    return _symmetrize(_range_safe_float_product(covariance, (scale, scale), xp))


def _restore_influence_covariance(
    covariance, covariance_scale, projection_scale, design_scale, xp
):
    """Restore all delayed influence scales in one exponent-safe product."""
    return _symmetrize(
        _range_safe_float_product(
            covariance,
            (
                covariance_scale[:, None],
                covariance_scale[None, :],
                projection_scale[:, None],
                projection_scale[None, :],
                design_scale,
                design_scale,
            ),
            xp,
        )
    )


def _cluster_grouped_scores(
    scores,
    codes,
    *,
    n_groups: int,
    nobs: int,
    group_debias: bool,
    xp,
    return_compensation: bool = False,
    return_components: bool = False,
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
        return_components=return_components,
    )
    correction = (
        _group_debias_factor(int(n_groups), int(nobs)) if group_debias else 1.0
    )
    if return_components:
        return grouped, float(correction)
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



def _dyadic_float_terms(value: float):
    """Return an exact finite-float coefficient as signed powers of two."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("expansion coefficient must be finite")
    if value == 0.0:
        return []
    sign = -1.0 if value < 0.0 else 1.0
    numerator, denominator = abs(value).as_integer_ratio()
    denominator_shift = int(denominator).bit_length() - 1
    powers = []
    bit = 0
    while numerator:
        if numerator & 1:
            powers.append(sign * (2.0 ** (bit - denominator_shift)))
        numerator >>= 1
        bit += 1
    powers.reverse()
    return powers


def _two_sum_matrix(left, right):
    summed = left + right
    virtual_right = summed - left
    error = (left - (summed - virtual_right)) + (right - virtual_right)
    return summed, error


def _twofold_matrix_sum(terms, xp):
    """Accumulate row-level matrix terms with two compensated error streams."""
    if not terms:
        raise ValueError("at least one matrix term is required")
    total = xp.zeros_like(terms[0])
    correction = xp.zeros_like(total)
    residual_correction = xp.zeros_like(total)
    for term in terms:
        total, error = _two_sum_matrix(total, term)
        correction, second_error = _two_sum_matrix(correction, error)
        residual_correction, third_error = _two_sum_matrix(
            residual_correction, second_error
        )
        residual_correction = residual_correction + third_error
    return _stable_matrix_expansion_sum(
        [residual_correction, correction, total], xp
    )


def _stable_matrix_expansion_sum(terms, xp):
    """Sum finite matrix terms with a floating-point expansion.

    The caller must put terms on a common range-safe working scale.  General
    TwoSum then keeps cancellation residuals as separate expansion components,
    so a smaller covariance tier is not discarded before later large terms
    cancel.  This is used only for nonnested multiway-cluster inclusion-
    exclusion, where cancellation across marginal/intersection components is
    part of the estimator definition.
    """
    if not terms:
        raise ValueError("at least one matrix term is required")

    partials = []
    for term in terms:
        carry = term
        next_partials = []
        for partial in partials:
            summed = carry + partial
            virtual_partial = summed - carry
            residual = (
                carry - (summed - virtual_partial)
            ) + (partial - virtual_partial)
            next_partials.append(residual)
            carry = summed
        next_partials.append(carry)
        partials = next_partials

    # Grow-expansion keeps residual components before the final carry.  At this
    # point all large cancellation has already occurred; an ascending estimate
    # performs the one unavoidable float64 rounding of the final covariance.
    total = xp.zeros_like(partials[0])
    for partial in partials:
        total = total + partial
    return _symmetrize(total)



def _append_vectorized_component_terms(terms, components, correction, sign, xp):
    """Append vectorized component-pair Gram terms for one cluster dimension.

    Every retained tier of ``components`` contributes its own Gram and its
    cross-Gram with each earlier tier.  The terms are consumed by
    ``_stable_matrix_expansion_sum``; exact dyadic coefficient handling is
    intentionally not applied here (BLAS reductions keep ordinary designs fast).
    """
    coefficient = float(sign) * float(correction)
    for i, left in enumerate(components):
        terms.append(_symmetrize(left.T @ left) * coefficient)
        for right in components[:i]:
            cross = left.T @ right
            terms.append((cross + cross.T) * coefficient)


def _row_expansion_residual_acceptable(component_sets, cov_work, xp) -> bool:
    """Return whether a vectorized Gram may ignore the tier residuals.

    The row-level fallback protects group-score tiers that a BLAS Gram would
    round away and that a later inclusion-exclusion cancellation could expose.
    ``component_sets`` are the retiered common-scale working components; the
    first component of each set is its dominant tier and the remaining
    components are residuals.  The retiering floor ``sqrt(16 n eps)`` already
    promises that every retained tier is Gram-safe on its own; the residual is
    accepted when it stays within that same floor relative to the dominant
    component and when its cross/self Gram contribution stays within the floor
    of the already computed vectorized result.  Ordinary designs therefore
    stay vectorized, while deep-cancellation designs (whose vectorized result
    collapses toward zero) still fall back to the exact row products.
    """
    eps = float(np.finfo(np.float64).eps)
    residual_max = None
    dominant_max = None
    n_rows = 1
    for components in component_sets:
        for index, component in enumerate(components):
            scale = float(_to_float_scalar(xp.max(xp.abs(component))))
            if index == 0:
                dominant_max = (
                    scale if dominant_max is None else max(dominant_max, scale)
                )
                n_rows = max(n_rows, int(component.shape[0]))
            else:
                residual_max = (
                    scale if residual_max is None else max(residual_max, scale)
                )
    if residual_max is None or residual_max == 0.0:
        return True
    if dominant_max is None or dominant_max == 0.0:
        return False
    tier_floor = float(np.sqrt(16.0 * n_rows * eps)) * 16.0
    if residual_max > dominant_max * tier_floor:
        return False
    result_scale = float(_to_float_scalar(xp.max(xp.abs(cov_work))))
    if result_scale <= 0.0 or not np.isfinite(result_scale):
        return False
    # A residual row contributes roughly residual * dominant per Gram entry
    # through the cross term (summed over n_rows rows) and residual^2 through
    # the self term; keep both within the retiering floor of the vectorized
    # result.
    cross_contribution = float(residual_max) * float(dominant_max) * n_rows
    self_contribution = float(residual_max) ** 2 * n_rows
    limit = result_scale * tier_floor
    return (cross_contribution <= limit) and (self_contribution <= limit)


def _component_row_reduction_needs_expansion(component_sets, xp) -> bool:
    """Return whether a BLAS row reduction can hide a recoverable component.

    Component-pair Grams are vectorized whenever every nonzero grouped value is
    comfortably above the roundoff floor induced by the largest value in the
    same coordinate.  A self-product squares the value ratio, so the relevant
    value threshold is ``sqrt(n * eps)`` rather than ``n * eps``.  A factor-16
    margin covers reduction order and the later inclusion-exclusion.  Only the
    rare high-dynamic-range case falls back to explicit row outer products.
    """
    risk = None
    eps = float(np.finfo(np.float64).eps)
    for components in component_sets:
        for component in components:
            n_rows = max(1, int(component.shape[0]))
            absolute = xp.abs(component)
            max_abs = _column_abs_max(component, xp)
            sentinel = xp.full_like(absolute, float(np.inf))
            nonzero = xp.where(absolute > 0.0, absolute, sentinel)
            if _is_torch(xp):
                min_nonzero = xp.min(nonzero, dim=0).values
            else:
                min_nonzero = xp.min(nonzero, axis=0)
            ratio_floor = float(np.sqrt(16.0 * float(n_rows) * eps))
            local = xp.any(
                xp.isfinite(min_nonzero)
                & (min_nonzero < max_abs * ratio_floor)
            )
            risk = local if risk is None else (risk | local)
    if risk is None:
        return False
    return bool(_to_float_scalar(risk))



def _row_min_nonzero(values, xp):
    absolute = xp.abs(values)
    sentinel = xp.full_like(absolute, float("inf"))
    nonzero = xp.where(absolute > 0.0, absolute, sentinel)
    if _is_torch(xp):
        return xp.min(nonzero, dim=1).values
    return xp.min(nonzero, axis=1)


def _common_scale_product_range_lost(component_sets, xp) -> bool:
    """Return whether a nonzero row product underflows on the common scale.

    Multi-tier two-way covariance must put estimator components on one working
    scale before inclusion-exclusion. A component can survive that division as
    a nonzero subnormal while its self/cross product is already below the
    smallest float64 subnormal. In that case silently evaluating the Gram would
    discard a potentially recoverable final covariance contribution after the
    larger estimator terms cancel, so the current float64 representation must
    fail closed.
    """
    minimum = float(np.nextafter(0.0, 1.0))
    risk = None
    for components in component_sets:
        row_minima = [_row_min_nonzero(component, xp) for component in components]
        for i, left_min in enumerate(row_minima):
            for right_min in row_minima[: i + 1]:
                active = xp.isfinite(left_min) & xp.isfinite(right_min)
                # Division avoids materializing the underflowing product itself.
                threshold = minimum / right_min
                local = xp.any(active & (left_min < threshold))
                risk = local if risk is None else (risk | local)
    if risk is None:
        return False
    return bool(_to_float_scalar(risk))


def _retier_component_for_safe_gram(component, xp):
    """Split one grouped component into globally Gram-safe magnitude tiers.

    Local group summation tiers protect cancellation *within* each cluster.  A
    later Gram reduction also sums across cluster rows, so singleton/intersection
    groups can still place vastly different magnitudes in the same component.
    Split those rows by a per-coordinate ``sqrt(n * eps)`` threshold before the
    BLAS reduction.  Each returned tier can then use a vectorized Gram without
    erasing a smaller row contribution that CGM cancellation may later expose.
    """
    n_rows = max(1, int(component.shape[0]))
    ratio_floor = float(
        np.sqrt(16.0 * float(n_rows) * float(np.finfo(np.float64).eps))
    )
    remaining = component
    tiers = []
    for _ in range(128):
        max_abs = _column_abs_max(remaining, xp)
        threshold = max_abs * ratio_floor
        tail_mask = (
            (xp.abs(remaining) < threshold[None, :])
            & (remaining != 0.0)
        )
        tiers.append(
            xp.where(tail_mask, xp.zeros_like(remaining), remaining)
        )
        if not bool(_to_float_scalar(xp.any(tail_mask))):
            break
        remaining = xp.where(
            tail_mask, remaining, xp.zeros_like(remaining)
        )
    else:
        raise RuntimeError(
            "two-way cluster global magnitude tiers exceeded the float64 budget"
        )
    return tiers


def _retier_component_sets_for_safe_gram(component_sets, xp):
    refined = []
    for components in component_sets:
        current = []
        for component in components:
            current.extend(_retier_component_for_safe_gram(component, xp))
        refined.append(current)
    return tuple(refined)




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
    _validate_covariance_finite_inputs(X, resid, xp)

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
    _validate_covariance_finite_inputs(X, resid, xp)
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
        if nested_c1:
            cov_work, common_scale = _cluster_meat_from_grouped(
                grouped2, correction2, xp
            )
        else:
            cov_work, common_scale = _cluster_meat_from_grouped(
                grouped1, correction1, xp
            )
    else:
        components1, correction1 = _cluster_grouped_scores(
            influence,
            c1,
            n_groups=int(len(labels1)),
            nobs=n,
            group_debias=group_debias,
            xp=xp,
            return_components=True,
        )
        components2, correction2 = _cluster_grouped_scores(
            influence,
            c2,
            n_groups=int(len(labels2)),
            nobs=n,
            group_debias=group_debias,
            xp=xp,
            return_components=True,
        )
        components12, correction12 = _cluster_grouped_scores(
            influence,
            c12,
            n_groups=n12,
            nobs=n,
            group_debias=group_debias,
            xp=xp,
            return_components=True,
        )

        extra_max = xp.zeros_like(xp.max(xp.abs(components1[0])))
        for components in (components1, components2, components12):
            for component in components[1:]:
                extra_max = xp.maximum(extra_max, xp.max(xp.abs(component)))

        if _to_float_scalar(extra_max) == 0.0:
            (grouped1_work, grouped2_work, grouped12_work), common_scale = (
                _common_gram_working_values(
                    [components1[0], components2[0], components12[0]],
                    xp,
                    max_multiplier=max(correction1, correction2, correction12),
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
            cov_work = _stable_inclusion_exclusion(
                V1_work, V2_work, V12_work, xp
            )
        else:
            component_sets = (components1, components2, components12)
            needs_row_expansion = _component_row_reduction_needs_expansion(
                component_sets, xp
            )
            if needs_row_expansion:
                component_sets = _retier_component_sets_for_safe_gram(
                    component_sets, xp
                )
                components1, components2, components12 = component_sets

            all_components = components1 + components2 + components12
            max_rows = max(
                int(components[0].shape[0]) for components in component_sets
            )
            # Count scalar products rather than matrix terms: an off-diagonal
            # expansion pair contributes both u v' and v u'.  The existing
            # common-Gram scaler already multiplies its bound by max_rows, so
            # convert the total product count to an equivalent per-max-row
            # multiplier and keep a factor-of-two margin for intermediate sums.
            product_count = sum(
                int(components[0].shape[0]) * (len(components) ** 2)
                for components in component_sets
            )
            max_correction = max(correction1, correction2, correction12)
            product_multiplier = (
                2.0
                * float(max(1, product_count))
                / float(max(1, max_rows))
                * float(max_correction)
            )
            working_components, common_scale = _common_gram_working_values(
                all_components,
                xp,
                max_multiplier=product_multiplier,
            )

            lost_component_backend = None
            for original, working in zip(all_components, working_components):
                local_loss = xp.any(
                    (original != 0.0) & (working == 0.0)
                )
                lost_component_backend = (
                    local_loss
                    if lost_component_backend is None
                    else (lost_component_backend | local_loss)
                )
            if bool(_to_float_scalar(lost_component_backend)):
                raise FloatingPointError(
                    "two-way cluster score expansion exceeds the float64 "
                    "common-scale dynamic range"
                )

            n1 = len(components1)
            n2 = len(components2)
            work1 = working_components[:n1]
            work2 = working_components[n1:n1 + n2]
            work12 = working_components[n1 + n2:]

            if _common_scale_product_range_lost((work1, work2, work12), xp):
                raise FloatingPointError(
                    "two-way cluster score expansion exceeds the float64 "
                    "common-scale product range"
                )

            if needs_row_expansion:
                # The row-level fallback exists because a BLAS Gram can round
                # away a small group-score tier that a later inclusion-exclusion
                # cancellation would otherwise expose.  Ordinary designs only
                # trip the conservative min/max trigger without carrying any
                # recoverable tier residual; try the vectorized Gram first and
                # fall back to the exact row products only when the residual
                # contribution is not eps-negligible.
                vectorized_terms = []
                _append_vectorized_component_terms(
                    vectorized_terms, work1, correction1, 1.0, xp
                )
                _append_vectorized_component_terms(
                    vectorized_terms, work2, correction2, 1.0, xp
                )
                _append_vectorized_component_terms(
                    vectorized_terms, work12, correction12, -1.0, xp
                )
                cov_work = _stable_matrix_expansion_sum(vectorized_terms, xp)
                if not _row_expansion_residual_acceptable(
                    (work1, work2, work12), cov_work, xp
                ):
                    terms = []

                    def _append_scaled_row_term(term, coefficient):
                        for dyadic in _dyadic_float_terms(coefficient):
                            terms.append(term * float(dyadic))

                    def _append_component_terms(components, correction, sign):
                        coefficient = float(sign) * float(correction)
                        for i, left in enumerate(components):
                            for row in range(int(left.shape[0])):
                                vector = left[row]
                                outer = vector[:, None] * vector[None, :]
                                _append_scaled_row_term(outer, coefficient)
                            for right in components[:i]:
                                for row in range(int(left.shape[0])):
                                    cross = left[row, :, None] * right[
                                        row, None, :
                                    ]
                                    _append_scaled_row_term(
                                        cross + cross.T, coefficient
                                    )

                    # Only the pathological cross-group dynamic-range branch
                    # uses row-level compensated accumulation.  Keep row
                    # products out of signed BLAS reductions and expand float
                    # debias corrections into exact dyadic factors before
                    # estimator-level cancellation.
                    _append_component_terms(work1, correction1, 1.0)
                    _append_component_terms(work2, correction2, 1.0)
                    _append_component_terms(work12, correction12, -1.0)
                    cov_work = _twofold_matrix_sum(terms, xp)
            else:
                terms = []
                # Local grouped-score tiers do not by themselves require the
                # row-level fallback.  If every cross-group component pair is
                # certified safe, retain the vectorized BLAS/GPU expansion.
                _append_vectorized_component_terms(
                    terms, work1, correction1, 1.0, xp
                )
                _append_vectorized_component_terms(
                    terms, work2, correction2, 1.0, xp
                )
                _append_vectorized_component_terms(
                    terms, work12, correction12, -1.0, xp
                )
                cov_work = _stable_matrix_expansion_sum(terms, xp)
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
    _validate_covariance_finite_inputs(X, resid, xp)
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
    _validate_covariance_finite_inputs(X, resid, xp)
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
