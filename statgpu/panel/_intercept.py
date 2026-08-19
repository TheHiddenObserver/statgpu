"""Precision helpers for panel response projections and quasi-demeaning."""
from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar
from statgpu.panel._linalg import (
    _lstsq_working_design,
    _svd_inverse_factors,
    panel_lstsq,
)
from statgpu.panel._reductions import grouped_score_sums, stable_reduction_flags


def panel_lstsq_stable_response(X, y, xp):
    """Use the shared SVD policy with a cancellation-safe response projection.

    The historical SVD path computes ``weighted_u_t @ y`` inside BLAS; reduction
    order can erase a representable low-order component. For ordinary responses
    this helper is exactly :func:`panel_lstsq`. Only responses already classified
    as cancellation/range sensitive replace that one matrix-vector reduction
    with the shared magnitude-tiered grouped sum. SVD factors, rank cutoff,
    design rescaling, and minimum-norm parameterization remain unchanged.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    if not bool(stable_reduction_flags(y, xp)[0]):
        return panel_lstsq(X, y, xp)

    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    products = weighted_u_t.T * y.reshape(-1, 1)
    codes_np = np.zeros(int(products.shape[0]), dtype=np.int64)
    projected = grouped_score_sums(
        products,
        codes_np,
        n_groups=1,
        xp=xp,
    )[0]
    return (Vh.T @ projected) * design_scale, rank


def panel_lstsq_exact_constant(X, y, xp, *, constant_index: int = 0):
    """Stable-response solve for callers that construct an exact constant column."""
    constant_index = int(constant_index)
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    if not 0 <= constant_index < int(X.shape[1]):
        raise ValueError("constant_index is out of range")
    return panel_lstsq_stable_response(X, y, xp)


def guarded_random_effects_common_sumsquares(left, right, xp):
    """Return common-scale RSS terms or fail if normalization loses a residual.

    Swamy-Arora compares within and between residual variances on one common
    physical scale. A single float64 normalization cannot safely represent two
    nonzero residual series whose magnitudes are separated by more than the
    subnormal range. Even when the normalized residual itself survives, its
    square can underflow before RSS accumulation. Either loss would force a
    false zero variance component, so reject that unsupported range.
    """
    from statgpu.panel._diagnostics import _scaled_unit_values

    scale = xp.maximum(xp.max(xp.abs(left)), xp.max(xp.abs(right)))
    scale_value = _to_float_scalar(scale)
    if scale_value == 0.0:
        return 0.0, 0.0, 0.0
    left_scaled = _scaled_unit_values(left, scale, xp)
    right_scaled = _scaled_unit_values(right, scale, xp)
    left_squared = left_scaled * left_scaled
    right_squared = right_scaled * right_scaled
    lost = (
        xp.any((left != 0.0) & (left_scaled == 0.0))
        | xp.any((right != 0.0) & (right_scaled == 0.0))
        | xp.any((left_scaled != 0.0) & (left_squared == 0.0))
        | xp.any((right_scaled != 0.0) & (right_squared == 0.0))
    )
    if bool(_to_float_scalar(lost)):
        raise FloatingPointError(
            "RandomEffects variance-component scaling exceeds the float64 "
            "common-residual range"
        )
    left_ss = _to_float_scalar(xp.sum(left_squared))
    right_ss = _to_float_scalar(xp.sum(right_squared))
    return float(left_ss), float(right_ss), float(scale_value)


def _quasi_component_loss(within, between, direct, candidate, xp):
    """Return a backend scalar marking unrecoverable component materialization."""
    between_product_lost = (
        (between[0] != 0.0) & (between[1] != 0.0) & (between[2] == 0.0)
    )
    within_values = within
    between_values = between[2]
    use_within = xp.abs(within_values) >= xp.abs(between_values)
    large = xp.where(use_within, within_values, between_values)
    small = xp.where(use_within, between_values, within_values)
    addition_lost = (small != 0.0) & (candidate == large)

    finite_components = xp.isfinite(within_values) & xp.isfinite(between_values)
    candidate_nonfinite = finite_components & (~xp.isfinite(candidate))

    comparison_scale = xp.maximum(xp.abs(direct), xp.abs(candidate))
    safe_scale = xp.maximum(
        comparison_scale,
        xp.full_like(comparison_scale, float(np.finfo(np.float64).tiny)),
    )
    relative_difference = xp.abs(direct / safe_scale - candidate / safe_scale)
    material_disagreement = (
        xp.isfinite(direct)
        & xp.isfinite(candidate)
        & (relative_difference > 4096.0 * float(np.finfo(np.float64).eps))
    )
    return xp.any(
        between_product_lost
        | addition_lost
        | candidate_nonfinite
        | material_disagreement
    )


def guarded_random_effects_quasi_demean(
    y,
    X,
    y_bar,
    X_bar,
    y_within,
    X_within,
    theta,
    xp,
    *,
    one_minus_theta=None,
):
    """Materialize the historical RE transform, failing closed on component loss.

    ``y - theta*y_bar`` equals ``y_within + (1-theta)*y_bar`` and likewise for
    every design column. The direct form preserves the established ordinary
    rounding path, but at extreme scales it can discard a nonzero component that
    a later regression would still identify. We evaluate the algebraically
    equivalent component decomposition only as a certificate. If multiplication,
    addition, or catastrophic direct/decomposed disagreement loses information,
    raise rather than publish a finite but incorrect RandomEffects fit.

    ``one_minus_theta`` may be supplied from the original positive square-root
    variance ratio. That preserves a nonzero complement even when ``1-root`` has
    already rounded to an exact ``theta == 1.0``.
    """
    one_minus = 1.0 - theta if one_minus_theta is None else one_minus_theta

    y_between = one_minus * y_bar
    y_candidate = y_within + y_between
    y_direct = y - theta * y_bar
    y_risk = _quasi_component_loss(
        y_within,
        (one_minus, y_bar, y_between),
        y_direct,
        y_candidate,
        xp,
    )

    one_minus_col = one_minus.reshape(-1, 1)
    X_between = one_minus_col * X_bar
    X_candidate = X_within + X_between
    X_direct = X - theta.reshape(-1, 1) * X_bar
    X_risk = _quasi_component_loss(
        X_within,
        (one_minus_col, X_bar, X_between),
        X_direct,
        X_candidate,
        xp,
    )

    if bool(_to_float_scalar(y_risk | X_risk)):
        raise FloatingPointError(
            "RandomEffects quasi-demeaning exceeds the float64 component range; "
            "the transformed fit would discard a nonzero within/between component"
        )
    return y_direct, X_direct
