"""Range- and cancellation-safe panel reduction primitives."""
from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar, _to_numpy, xp_asarray, xp_zeros


def _is_torch(xp) -> bool:
    return getattr(xp, "__name__", "") == "torch"


def _column_abs_max(values, xp):
    if _is_torch(xp):
        return xp.max(xp.abs(values), dim=0).values
    return xp.max(xp.abs(values), axis=0)


def _group_abs_max(values, codes, codes_np, *, n_groups: int, xp):
    shape = (int(n_groups), int(values.shape[1]))
    out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=values)
    absolute = xp.abs(values)
    index = codes.unsqueeze(1).expand_as(values) if hasattr(codes, "unsqueeze") else None
    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(0, index, absolute, reduce="amax", include_self=True)
    elif type(out).__module__.startswith("cupy"):
        xp.maximum.at(out, codes, absolute)
    else:
        np.maximum.at(out, codes_np, absolute)
    return out


def _signed_parts(values, codes, codes_np, *, n_groups: int, xp):
    shape = (int(n_groups), int(values.shape[1]))
    positive = xp.where(values > 0.0, values, xp.zeros_like(values))
    negative = xp.where(values < 0.0, values, xp.zeros_like(values))
    positive_out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=values)
    negative_out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=values)
    index = codes.unsqueeze(1).expand_as(values) if hasattr(codes, "unsqueeze") else None
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
    residual = (left - (summed - virtual_right)) + (right - virtual_right)
    return summed, residual


def grouped_score_sums(
    scores,
    codes_np,
    *,
    n_groups: int,
    xp,
    return_compensation: bool = False,
    return_components: bool = False,
):
    """Sum grouped float64 scores while preserving recursive magnitude tiers."""
    codes_np = np.asarray(codes_np, dtype=np.int64).ravel()
    if codes_np.shape[0] != int(scores.shape[0]):
        raise ValueError("group codes must match the number of score rows")
    if int(n_groups) <= 0:
        raise ValueError("at least one group is required")
    if return_compensation and return_components:
        raise ValueError(
            "return_compensation and return_components are mutually exclusive"
        )
    codes = xp_asarray(codes_np, dtype=xp.int64, xp=xp, ref_arr=scores)
    max_abs = _group_abs_max(
        scores, codes, codes_np, n_groups=int(n_groups), xp=xp
    )
    counts_np = np.bincount(codes_np, minlength=int(n_groups)).astype(np.float64)
    counts = xp_asarray(
        counts_np, dtype=xp.float64, xp=xp, ref_arr=scores
    ).reshape(-1, 1)
    limit = float(np.finfo(np.float64).max) / counts
    # This primitive returns a sum, not a mean. Scale only when a group value is
    # strictly above the conservative same-sign limit. At exact equality a
    # large positive/negative pair can cancel and expose a representable
    # subnormal tail; pre-dividing that tail by the group count would erase it.
    # Mean callers apply their own >= guard because their final target remains
    # representable even when an unscaled same-sign *sum* at equality may not be.
    factor = xp.where(max_abs > limit, counts, xp.ones_like(max_abs))
    remaining = scores / factor[codes]
    split_ratio = xp.minimum(
        counts * float(8.0 * np.finfo(np.float64).eps),
        xp.full_like(counts, 0.25),
    )

    tiers = []
    for _ in range(128):
        tier_max = _group_abs_max(
            remaining, codes, codes_np, n_groups=int(n_groups), xp=xp
        )
        threshold = tier_max * split_ratio
        tail_mask = (
            (xp.abs(remaining) < threshold[codes]) & (remaining != 0.0)
        )
        main = xp.where(tail_mask, xp.zeros_like(remaining), remaining)
        positive_out, negative_out = _signed_parts(
            main, codes, codes_np, n_groups=int(n_groups), xp=xp
        )
        tier_sum, tier_error = _two_sum(positive_out, negative_out)
        tiers.append((tier_sum, tier_error))
        if not bool(_to_float_scalar(xp.any(tail_mask))):
            break
        remaining = xp.where(tail_mask, remaining, xp.zeros_like(remaining))
    else:
        raise RuntimeError(
            "grouped score cancellation exceeded the float64 tier budget"
        )

    if return_components:
        components = []
        for tier_sum, tier_error in tiers:
            components.extend((tier_sum * factor, tier_error * factor))
        return components

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


def stable_mean(values, xp):
    """Return an axis-0 mean without avoidable overflow or lost cancellation."""
    original_ndim = int(values.ndim)
    matrix = values.reshape(-1, 1) if original_ndim == 1 else values
    if int(matrix.ndim) != 2 or int(matrix.shape[0]) <= 0:
        raise ValueError("stable_mean requires a non-empty one- or two-dimensional array")
    n = int(matrix.shape[0])
    max_abs = _column_abs_max(matrix, xp)
    limit = float(np.finfo(np.float64).max) / float(n)
    dangerous = max_abs >= float(limit)
    factor = xp.where(
        dangerous,
        xp.full_like(max_abs, float(n)),
        xp.ones_like(max_abs),
    )
    working = matrix / factor[None, :]
    codes_np = np.zeros(n, dtype=np.int64)
    total = grouped_score_sums(
        working, codes_np, n_groups=1, xp=xp
    )[0]
    mean = xp.where(dangerous, total, total / float(n))
    first = matrix[0]
    if _is_torch(xp):
        constant = xp.all(matrix == first[None, :], dim=0)
    else:
        constant = xp.all(matrix == first[None, :], axis=0)
    mean = xp.where(constant, first, mean)
    return mean[0] if original_ndim == 1 else mean


def stable_group_means_preindexed(
    values,
    codes,
    codes_np,
    *,
    n_groups: int,
    counts,
    xp,
):
    """Return compact group means with cancellation-safe tiered accumulation."""
    values_2d = values.reshape(-1, 1)
    codes_np = np.asarray(codes_np, dtype=np.int64).ravel()
    counts_col = counts.reshape(-1, 1)
    max_abs = _group_abs_max(
        values_2d, codes, codes_np, n_groups=int(n_groups), xp=xp
    )
    limit = float(np.finfo(np.float64).max) / counts_col
    dangerous = max_abs >= limit
    factor = xp.where(dangerous, counts_col, xp.ones_like(max_abs))
    working = values_2d / factor[codes]
    total = grouped_score_sums(
        working, codes_np, n_groups=int(n_groups), xp=xp
    )[:, 0]
    mean = xp.where(
        dangerous[:, 0],
        total,
        total / counts,
    )

    # Preserve an exactly constant group exactly.  This matters for degenerate
    # TSS/within transformations: repeated finite constants should not acquire a
    # synthetic residual solely from reduction order.
    group_min = xp.full_like(counts, float("inf"))
    group_max = xp.full_like(counts, float("-inf"))
    if hasattr(group_min, "scatter_reduce_"):
        group_min.scatter_reduce_(0, codes, values, reduce="amin", include_self=True)
        group_max.scatter_reduce_(0, codes, values, reduce="amax", include_self=True)
    elif type(group_min).__module__.startswith("cupy"):
        xp.minimum.at(group_min, codes, values)
        xp.maximum.at(group_max, codes, values)
    else:
        np.minimum.at(group_min, codes_np, values)
        np.maximum.at(group_max, codes_np, values)
    constant = group_min == group_max
    return xp.where(constant, group_min, mean)


def stable_reduction_flags(matrix, xp):
    """Return host boolean flags for columns needing tiered group reduction.

    One packed transfer classifies all columns.  Ordinary columns keep the
    historical one-scatter fast path; only columns whose dynamic range can hide
    a recoverable float64 component, or whose same-sign reduction can overflow,
    use the tiered path during iterative fixed-effect projections.
    """
    values = matrix.reshape(-1, 1) if int(matrix.ndim) == 1 else matrix
    n = max(1, int(values.shape[0]))
    absolute = xp.abs(values)
    max_abs = _column_abs_max(values, xp)
    sentinel = xp.full_like(absolute, float("inf"))
    nonzero = xp.where(absolute > 0.0, absolute, sentinel)
    if _is_torch(xp):
        min_nonzero = xp.min(nonzero, dim=0).values
    else:
        min_nonzero = xp.min(nonzero, axis=0)
    ratio_floor = min(
        0.25,
        16.0 * float(n) * float(np.finfo(np.float64).eps),
    )
    cancellation_risk = xp.isfinite(min_nonzero) & (
        min_nonzero < max_abs * float(ratio_floor)
    )
    overflow_risk = max_abs >= (
        float(np.finfo(np.float64).max) / float(n)
    )
    flags = cancellation_risk | overflow_risk
    return np.asarray(_to_numpy(flags), dtype=bool).reshape(-1)
