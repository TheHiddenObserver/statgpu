"""Range- and cancellation-safe panel reduction primitives."""
from __future__ import annotations

import numpy as np

from statgpu.backends._utils import _CUPY_UFUNC_AT_SAFE_MAX
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
        # CuPy ``maximum.at``/``cupyx.scatter_max`` corrupt float64 magnitudes
        # around 1e7..1e308 into inf (observed on CuPy 13.6), even with unique
        # indices.  Ordinary magnitudes are far below the corruption range, so
        # keep the native GPU scatter for them and fall back to the sequential
        # host scatter only when a magnitude could hit the corruption window.
        if float(_to_float_scalar(xp.max(absolute))) <= _CUPY_UFUNC_AT_SAFE_MAX:
            xp.maximum.at(out, codes, absolute)
            return out
        out_np = np.zeros(shape, dtype=np.float64)
        np.maximum.at(out_np, codes_np, _to_numpy(absolute))
        out = xp_asarray(out_np, dtype=xp.float64, xp=xp, ref_arr=values)
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
        # CuPy ``add.at`` uses atomic accumulation, so repeated indices are
        # summed exactly like the NumPy sequential scatter; unlike CuPy's
        # non-atomic ``maximum.at``/``scatter_max`` (which corrupt float64
        # magnitudes around 1e7..1e308), the additive path is safe.
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


def _validate_group_reduction_inputs(scores, codes_np, *, n_groups: int, xp):
    codes_np = np.asarray(codes_np, dtype=np.int64).ravel()
    if codes_np.shape[0] != int(scores.shape[0]):
        raise ValueError("group codes must match the number of score rows")
    if int(n_groups) <= 0:
        raise ValueError("at least one group is required")
    codes = xp_asarray(codes_np, dtype=xp.int64, xp=xp, ref_arr=scores)
    counts_np = np.bincount(codes_np, minlength=int(n_groups)).astype(np.float64)
    counts = xp_asarray(
        counts_np, dtype=xp.float64, xp=xp, ref_arr=scores
    ).reshape(-1, 1)
    return codes_np, codes, counts


def _grouped_tier_components(
    scores,
    codes_np,
    codes,
    counts,
    *,
    n_groups: int,
    xp,
    target: str,
):
    """Return high/error components for grouped sums or grouped means.

    Magnitude separation happens before range scaling. For a sum, a risky tier
    is divided by its group count and restored after signed accumulation. For a
    mean, a risky tier is already on its final mean scale after that division;
    a safe tier is summed first and divided by the group count afterwards. This
    keeps collectively representable subnormal tails from being erased by a
    whole-group predivision performed only because another tier is huge.
    """
    if target not in {"sum", "mean"}:
        raise ValueError("target must be 'sum' or 'mean'")

    split_ratio = xp.minimum(
        counts * float(8.0 * np.finfo(np.float64).eps),
        xp.full_like(counts, 0.25),
    )
    remaining = scores
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

        main_max = _group_abs_max(
            main, codes, codes_np, n_groups=int(n_groups), xp=xp
        )
        limit = float(np.finfo(np.float64).max) / counts
        dangerous = main_max >= limit
        work_factor = xp.where(dangerous, counts, xp.ones_like(main_max))
        main_work = main / work_factor[codes]
        positive_out, negative_out = _signed_parts(
            main_work, codes, codes_np, n_groups=int(n_groups), xp=xp
        )
        tier_sum, tier_error = _two_sum(positive_out, negative_out)

        if target == "sum":
            output_factor = work_factor
        else:
            output_factor = xp.where(
                dangerous,
                xp.ones_like(counts),
                1.0 / counts,
            )
        tiers.append((tier_sum * output_factor, tier_error * output_factor))

        if not bool(_to_float_scalar(xp.any(tail_mask))):
            break
        remaining = xp.where(tail_mask, remaining, xp.zeros_like(remaining))
    else:
        raise RuntimeError(
            "grouped score cancellation exceeded the float64 tier budget"
        )
    return tiers


def _collapse_tier_components(tiers, xp, *, return_compensation: bool = False):
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
        return high, low
    summed, error = _two_sum(high, low)
    return summed + error


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
    if return_compensation and return_components:
        raise ValueError(
            "return_compensation and return_components are mutually exclusive"
        )
    codes_np, codes, counts = _validate_group_reduction_inputs(
        scores, codes_np, n_groups=int(n_groups), xp=xp
    )
    tiers = _grouped_tier_components(
        scores,
        codes_np,
        codes,
        counts,
        n_groups=int(n_groups),
        xp=xp,
        target="sum",
    )
    if return_components:
        components = []
        for tier_sum, tier_error in tiers:
            components.extend((tier_sum, tier_error))
        return components
    return _collapse_tier_components(
        tiers, xp, return_compensation=return_compensation
    )


def grouped_score_means(scores, codes_np, *, n_groups: int, xp):
    """Return grouped float64 means with tier-local range protection."""
    codes_np, codes, counts = _validate_group_reduction_inputs(
        scores, codes_np, n_groups=int(n_groups), xp=xp
    )
    tiers = _grouped_tier_components(
        scores,
        codes_np,
        codes,
        counts,
        n_groups=int(n_groups),
        xp=xp,
        target="mean",
    )
    return _collapse_tier_components(tiers, xp)


def stable_mean(values, xp):
    """Return an axis-0 mean without avoidable overflow or lost cancellation."""
    original_ndim = int(values.ndim)
    matrix = values.reshape(-1, 1) if original_ndim == 1 else values
    if int(matrix.ndim) != 2 or int(matrix.shape[0]) <= 0:
        raise ValueError("stable_mean requires a non-empty one- or two-dimensional array")
    n = int(matrix.shape[0])
    codes_np = np.zeros(n, dtype=np.int64)
    mean = grouped_score_means(matrix, codes_np, n_groups=1, xp=xp)[0]

    # Exactly constant finite columns have exactly that mean. Preserve the input
    # value rather than introducing an avoidable divide/multiply rounding step.
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
    del counts  # counts are reconstructed once inside the shared reducer.
    values_2d = values.reshape(-1, 1)
    codes_np = np.asarray(codes_np, dtype=np.int64).ravel()
    mean = grouped_score_means(
        values_2d, codes_np, n_groups=int(n_groups), xp=xp
    )[:, 0]

    # Preserve exactly constant groups exactly, including huge repeated values.
    group_min = xp.full_like(mean, float("inf"))
    group_max = xp.full_like(mean, float("-inf"))
    if hasattr(group_min, "scatter_reduce_"):
        group_min.scatter_reduce_(0, codes, values, reduce="amin", include_self=True)
        group_max.scatter_reduce_(0, codes, values, reduce="amax", include_self=True)
    elif type(group_min).__module__.startswith("cupy"):
        # CuPy ``minimum.at``/``maximum.at`` return inf for float64 magnitudes
        # around 1e7..1e308 (observed on CuPy 13.6).  Ordinary magnitudes keep
        # the native GPU scatter; only risky magnitudes use the sequential
        # host scatter and restore the group extrema on the reference device.
        if float(_to_float_scalar(xp.max(xp.abs(values)))) <= _CUPY_UFUNC_AT_SAFE_MAX:
            xp.minimum.at(group_min, codes, values)
            xp.maximum.at(group_max, codes, values)
        else:
            values_np = _to_numpy(values)
            codes_flat = _to_numpy(codes).ravel()
            out_np = np.zeros(mean.shape, dtype=np.float64)
            group_min_np = np.full(out_np.shape, float("inf"))
            group_max_np = np.full(out_np.shape, float("-inf"))
            np.minimum.at(group_min_np, codes_flat, values_np)
            np.maximum.at(group_max_np, codes_flat, values_np)
            group_min = xp_asarray(
                group_min_np, dtype=xp.float64, xp=xp, ref_arr=mean
            )
            group_max = xp_asarray(
                group_max_np, dtype=xp.float64, xp=xp, ref_arr=mean
            )
    else:
        np.minimum.at(group_min, codes_np, values)
        np.maximum.at(group_max, codes_np, values)
    constant = group_min == group_max
    return xp.where(constant, group_min, mean)


def stable_reduction_flags(matrix, xp):
    """Return host boolean flags for columns needing tiered group reduction.

    One packed transfer classifies all columns. Ordinary columns keep the
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
