from pathlib import Path


def function_span(text: str, name: str):
    marker = f"def {name}("
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"function {name} not found")
    next_def = text.find("\ndef ", start + len(marker))
    end = len(text) if next_def < 0 else next_def + 1
    return start, end


def replace_function(path: str, name: str, replacement: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start, end = function_span(text, name)
    p.write_text(
        text[:start] + replacement.rstrip() + "\n\n" + text[end:],
        encoding="utf-8",
    )


def edit_function(path: str, name: str, replacements) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start, end = function_span(text, name)
    block = text[start:end]
    for old, new in replacements:
        if old not in block:
            raise RuntimeError(f"anchor not found in {path}:{name}: {old[:100]!r}")
        block = block.replace(old, new)
    p.write_text(text[:start] + block + text[end:], encoding="utf-8")


def insert_once(path: str, marker: str, addition: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if addition.strip() in text:
        return
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError(f"insert anchor not found in {path}: {marker!r}")
    p.write_text(text[:pos] + addition + text[pos:], encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"replace anchor not found in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared cancellation-safe reduction primitive.  This is deliberately separate
# from covariance and utils to avoid a circular dependency: covariance already
# depends on panel metadata utilities, while fit-space group means must reuse the
# same magnitude-tiered summation policy.
# ---------------------------------------------------------------------------
Path("statgpu/panel/_reductions.py").write_text(
    r'''"""Range- and cancellation-safe panel reduction primitives."""
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
    # Equality is dangerous too: DBL_MAX / m can round upward enough that m
    # equal terms overflow although their mean remains representable.
    factor = xp.where(max_abs >= limit, counts, xp.ones_like(max_abs))
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
    namespace = getattr(xp, "__name__", "")
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
''',
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# Covariance now imports the shared reducer.  Keep the old private name as an
# alias so maintained tests and internal callers retain their contract.
# ---------------------------------------------------------------------------
replace_function("statgpu/panel/_covariance.py", "_grouped_score_sums", "")
insert_once(
    "statgpu/panel/_covariance.py",
    "from statgpu.panel._utils import factorize_panel_metadata\n",
    "from statgpu.panel._reductions import grouped_score_sums as _grouped_score_sums\n",
)


# ---------------------------------------------------------------------------
# Fit-space group means: one packed stability classification per initial y/X
# matrix, then the ordinary one-scatter path remains unchanged for normal-scale
# columns.  Extreme columns use the shared tiered reduction throughout the
# alternating projection so no per-column host decision is introduced inside
# each iteration.
# ---------------------------------------------------------------------------
insert_once(
    "statgpu/panel/_utils.py",
    "\n\n@dataclass\nclass PanelSummary:",
    "\nfrom statgpu.panel._reductions import (\n    stable_group_means_preindexed,\n    stable_reduction_flags,\n)\n",
)

replace_function(
    "statgpu/panel/_utils.py",
    "_remap_to_contiguous",
    r'''def _remap_to_contiguous(groups, xp):
    """Remap group labels to contiguous 0..n_groups-1 indices."""
    groups_np = _to_numpy(groups).ravel()
    unique_labels, indices_np = np.unique(groups_np, return_inverse=True)
    indices_np = indices_np.astype(np.int64, copy=False)
    n_groups = len(unique_labels)
    indices = xp_asarray(indices_np, dtype=xp.int64, xp=xp, ref_arr=groups)
    return indices, n_groups, unique_labels, indices_np
''',
)

replace_function(
    "statgpu/panel/_utils.py",
    "_prepare_group_projection",
    r'''def _prepare_group_projection(groups, xp):
    """Factorize one group vector once and cache backend/host projection codes."""
    groups = xp_asarray(groups, xp=xp).ravel()
    idx, n_groups, labels, codes_np = _remap_to_contiguous(groups, xp)
    ones = xp_ones(int(idx.shape[0]), xp.float64, xp, idx)
    counts = _scatter_add(xp, idx, ones, n_groups)
    inv_counts = 1.0 / xp_maximum(counts, 1.0, xp)
    return idx, n_groups, labels, counts, inv_counts, codes_np
''',
)

replace_function(
    "statgpu/panel/_utils.py",
    "_compact_group_means",
    r'''def _compact_group_means(values, projection, xp, *, stable=None):
    idx, n_groups, _labels, counts, inv_counts, codes_np = projection
    if stable is None:
        stable = bool(stable_reduction_flags(values, xp)[0])
    if stable:
        return stable_group_means_preindexed(
            values,
            idx,
            codes_np,
            n_groups=int(n_groups),
            counts=counts,
            xp=xp,
        )

    # Common path: one scatter-add, with only the range scaling needed to avoid
    # an overflowing same-sign group sum. Equality is included because
    # DBL_MAX / m can round upward enough that m equal terms overflow.
    counts_aligned = counts[idx]
    limit = np.finfo(np.float64).max / xp_maximum(counts_aligned, 1.0, xp)
    dangerous_obs = (xp.abs(values) >= limit) * 1.0
    dangerous_count = _scatter_add(xp, idx, dangerous_obs, n_groups)
    factor_compact = xp.where(
        dangerous_count > 0.0, counts, xp.ones_like(counts)
    )
    factor_aligned = factor_compact[idx]
    sums = _scatter_add(xp, idx, values / factor_aligned, n_groups)
    return sums * inv_counts * factor_compact
''',
)

replace_function(
    "statgpu/panel/_utils.py",
    "_within_preindexed",
    r'''def _within_preindexed(values, projection, xp, *, stable=None):
    means = _compact_group_means(values, projection, xp, stable=stable)
    return values - means[projection[0]]
''',
)

replace_function(
    "statgpu/panel/_utils.py",
    "_within_matrix_preindexed",
    r'''def _within_matrix_preindexed(matrix, projection, xp, *, stable_columns=None):
    result = matrix.copy() if hasattr(matrix, "copy") else matrix.clone()
    for j in range(int(matrix.shape[1])):
        stable = None if stable_columns is None else bool(stable_columns[j])
        result[:, j] = _within_preindexed(
            matrix[:, j], projection, xp, stable=stable
        )
    return result
''',
)

replace_function(
    "statgpu/panel/_utils.py",
    "_matrix_group_mean_max",
    r'''def _matrix_group_mean_max(matrix, projection, xp, *, stable_columns=None):
    values = xp_zeros(int(matrix.shape[1]), matrix.dtype, xp, matrix)
    for j in range(int(matrix.shape[1])):
        stable = None if stable_columns is None else bool(stable_columns[j])
        means = _compact_group_means(
            matrix[:, j], projection, xp, stable=stable
        )
        values[j] = xp.max(xp.abs(means))
    return values
''',
)

edit_function(
    "statgpu/panel/_utils.py",
    "demean_variables",
    [
        (
            "    X_level_scale = _column_max_abs(X_d, xp)\n\n    entity_projection = (",
            "    X_level_scale = _column_max_abs(X_d, xp)\n"
            "    stability_matrix = (\n"
            "        xp.cat([y_d[:, None], X_d], dim=1)\n"
            "        if getattr(xp, '__name__', '') == 'torch'\n"
            "        else xp.concatenate([y_d[:, None], X_d], axis=1)\n"
            "    )\n"
            "    stability_flags = stable_reduction_flags(stability_matrix, xp)\n"
            "    y_stable = bool(stability_flags[0])\n"
            "    X_stable = stability_flags[1:]\n\n"
            "    entity_projection = (",
        ),
        (
            "_within_preindexed(y_d, entity_projection, xp)",
            "_within_preindexed(y_d, entity_projection, xp, stable=y_stable)",
        ),
        (
            "_within_matrix_preindexed(X_d, entity_projection, xp)",
            "_within_matrix_preindexed(\n            X_d, entity_projection, xp, stable_columns=X_stable\n        )",
        ),
        (
            "_within_preindexed(y_d, time_projection, xp)",
            "_within_preindexed(y_d, time_projection, xp, stable=y_stable)",
        ),
        (
            "_within_matrix_preindexed(X_d, time_projection, xp)",
            "_within_matrix_preindexed(\n            X_d, time_projection, xp, stable_columns=X_stable\n        )",
        ),
        (
            "_compact_group_means(y_d, entity_projection, xp)",
            "_compact_group_means(\n            y_d, entity_projection, xp, stable=y_stable\n        )",
        ),
        (
            "_compact_group_means(y_d, time_projection, xp)",
            "_compact_group_means(\n            y_d, time_projection, xp, stable=y_stable\n        )",
        ),
        (
            "_matrix_group_mean_max(\n            X_d, entity_projection, xp\n        )",
            "_matrix_group_mean_max(\n            X_d, entity_projection, xp, stable_columns=X_stable\n        )",
        ),
        (
            "_matrix_group_mean_max(X_d, time_projection, xp)",
            "_matrix_group_mean_max(\n            X_d, time_projection, xp, stable_columns=X_stable\n        )",
        ),
    ],
)

edit_function(
    "statgpu/panel/_utils.py",
    "_recover_two_way_effects",
    [
        (
            "    level_scale = xp.max(xp.abs(values))\n\n    converged = False",
            "    level_scale = xp.max(xp.abs(values))\n"
            "    stable = bool(stable_reduction_flags(values, xp)[0])\n\n"
            "    converged = False",
        ),
        (
            "            values - time_effects[t_idx], entity_projection, xp\n        )",
            "            values - time_effects[t_idx], entity_projection, xp, stable=stable\n        )",
        ),
        (
            "            values - entity_effects[e_idx], time_projection, xp\n        )",
            "            values - entity_effects[e_idx], time_projection, xp, stable=stable\n        )",
        ),
        (
            "_compact_group_means(residual, entity_projection, xp)",
            "_compact_group_means(residual, entity_projection, xp, stable=stable)",
        ),
        (
            "_compact_group_means(residual, time_projection, xp)",
            "_compact_group_means(residual, time_projection, xp, stable=stable)",
        ),
    ],
)

# Six-element projection tuple: update the only full tuple unpackings and the
# dummy helper that calls _remap_to_contiguous directly. Indexed consumers keep
# their historical positions unchanged.
replace_once(
    "statgpu/panel/_utils.py",
    "    idx, n_groups, _ = _remap_to_contiguous(groups, xp)\n",
    "    idx, n_groups, _, _codes_np = _remap_to_contiguous(groups, xp)\n",
)
replace_once(
    "statgpu/panel/_utils.py",
    "    e_idx, n_entities, _e_labels, _e_counts, _e_inv = entity_projection\n    t_idx, n_times, _t_labels, t_counts, _t_inv = time_projection\n",
    "    e_idx, n_entities, _e_labels, _e_counts, _e_inv, _e_codes_np = entity_projection\n    t_idx, n_times, _t_labels, t_counts, _t_inv, _t_codes_np = time_projection\n",
)


# ---------------------------------------------------------------------------
# Diagnostics and Fama-MacBeth use the same stable mean. This fixes the current
# Torch 2.0 CI regression where a huge constant y acquired synthetic TSS, and it
# also fixes arbitrary period-beta order such as [A, small, -A].
# ---------------------------------------------------------------------------
insert_once(
    "statgpu/panel/_diagnostics.py",
    "from statgpu.panel._results import PanelFitStatistics, PanelTestResult\n",
    "from statgpu.panel._reductions import stable_mean\n",
)
replace_function(
    "statgpu/panel/_diagnostics.py",
    "_scaled_mean",
    r'''def _scaled_mean(values, xp):
    """Return an axis-0 mean without avoidable overflow or lost cancellation."""
    return stable_mean(values, xp)
''',
)
replace_function(
    "statgpu/panel/_diagnostics.py",
    "_scaled_group_means",
    r'''def _scaled_group_means(values, groups, xp):
    """Return cancellation-safe group means aligned to observations."""
    return group_means(values, groups, xp=xp)
''',
)

insert_once(
    "statgpu/panel/_fama_macbeth.py",
    "from statgpu.panel._utils import (\n",
    "from statgpu.panel._reductions import stable_mean\n",
)

fmb_path = Path("statgpu/panel/_fama_macbeth.py")
fmb_text = fmb_path.read_text(encoding="utf-8")
old_mean = '''        # Protect the T-term reduction with the minimum scale needed for\n        # overflow safety.  Magnitude-normalizing each coordinate can underflow\n        # a small but representable remainder when large period coefficients\n        # cancel (e.g. +1e154, -1e154, 1e-170).\n        if xp.__name__ == "torch":\n            beta_scale = xp.max(xp.abs(betas), dim=0).values\n        else:\n            beta_scale = xp.max(xp.abs(betas), axis=0)\n        reduction_limit = np.finfo(np.float64).max / float(T)\n        mean_scale = xp.where(\n            beta_scale > reduction_limit,\n            xp.full_like(beta_scale, float(T)),\n            xp.ones_like(beta_scale),\n        )\n        avg_beta = xp.sum(betas / mean_scale, axis=0) * (\n            mean_scale / float(T)\n        )\n'''
new_mean = '''        # Use the shared tiered mean so period order cannot erase a small\n        # representable coefficient between large cancelling periods. Ordinary\n        # same-scale inputs collapse to one tier; only risky dynamic ranges pay\n        # for additional magnitude components.\n        avg_beta = stable_mean(betas, xp)\n'''
if old_mean not in fmb_text:
    raise RuntimeError("FamaMacBeth average-beta reduction anchor not found")
fmb_path.write_text(fmb_text.replace(old_mean, new_mean, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Focused regressions: fit-space group mean, diagnostics, subnormal/boundary
# preservation, maintained Torch failure, and Fama-MacBeth beta ordering.
# ---------------------------------------------------------------------------
Path("dev/tests/test_panel_diagnostic_cancellation_precision.py").write_text(
    r'''import numpy as np
import pytest

from statgpu.panel import FamaMacBeth
from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean
from statgpu.panel._utils import group_means, within_transform


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_shared_group_mean_preserves_small_term_after_huge_cancellation_numpy():
    values = np.asarray([1.0e308, 1.0, -1.0e308], dtype=np.float64)
    groups = np.zeros(3, dtype=np.int64)
    actual = np.asarray(group_means(values, groups, xp=np))
    np.testing.assert_allclose(actual, np.full(3, 1.0 / 3.0), rtol=0.0, atol=0.0)


def test_scaled_mean_preserves_small_term_after_huge_cancellation_numpy():
    values = np.asarray([1.0e308, 1.0, -1.0e308], dtype=np.float64)
    assert float(_scaled_mean(values, np)) == 1.0 / 3.0


def test_scaled_group_means_preserve_small_term_after_huge_cancellation_numpy():
    values = np.asarray(
        [1.0e308, 1.0, -1.0e308, 5.0, 5.0, 5.0], dtype=np.float64
    )
    groups = np.asarray([7, 7, 7, 3, 3, 3], dtype=np.int64)
    actual = np.asarray(_scaled_group_means(values, groups, np))
    expected = np.asarray([1.0 / 3.0] * 3 + [5.0] * 3, dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_shared_group_mean_scales_at_exact_overflow_boundary_numpy():
    value = float(np.finfo(np.float64).max / 3.0)
    values = np.asarray([value, value, value], dtype=np.float64)
    groups = np.zeros(3, dtype=np.int64)
    actual = np.asarray(group_means(values, groups, xp=np))
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, np.full(3, value))
    np.testing.assert_array_equal(
        np.asarray(within_transform(values, groups, xp=np)), np.zeros(3)
    )


def test_shared_group_mean_preserves_smallest_subnormal_numpy():
    tiny = np.nextafter(0.0, 1.0)
    values = np.asarray([tiny, tiny, tiny], dtype=np.float64)
    groups = np.zeros(3, dtype=np.int64)
    actual = np.asarray(group_means(values, groups, xp=np))
    assert np.all(actual == tiny)
    assert float(_scaled_mean(values, np)) == tiny


def _multiscale_fmb_fixture():
    x = np.asarray([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)
    intercepts = np.asarray([1.0e150, 1.0, -1.0e150], dtype=np.float64)
    X = np.tile(x, intercepts.size)[:, None]
    y = np.concatenate([np.full(x.size, value) for value in intercepts])
    time = np.repeat(np.arange(intercepts.size), x.size)
    return X, y, time


def test_fama_macbeth_average_preserves_middle_period_after_large_cancellation_numpy():
    X, y, time = _multiscale_fmb_fixture()
    model = FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)
    np.testing.assert_allclose(float(model.coef_[0]), 1.0 / 3.0, rtol=2e-14, atol=0.0)


def test_torch_cpu_group_mean_and_fmb_cancellation_match_numpy():
    torch = pytest.importorskip("torch")
    values = torch.tensor([1.0e308, 1.0, -1.0e308], dtype=torch.float64)
    groups = torch.zeros(3, dtype=torch.int64)
    grouped = group_means(values, groups, xp=torch)
    np.testing.assert_allclose(
        _to_numpy(grouped), np.full(3, 1.0 / 3.0), rtol=0.0, atol=0.0
    )
    assert float(_scaled_mean(values, torch)) == 1.0 / 3.0

    X, y, time = _multiscale_fmb_fixture()
    expected = FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)
    actual = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    np.testing.assert_allclose(
        _to_numpy(actual.coef_), np.asarray(expected.coef_), rtol=3e-13, atol=0.0
    )


def test_torch_cpu_huge_constant_response_remains_degenerate():
    torch = pytest.importorskip("torch")
    x_period = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
    X = np.tile(x_period, 4)[:, None]
    y = np.full(X.shape[0], 6.0e307, dtype=np.float64)
    time = np.repeat(np.arange(4), x_period.size)
    model = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    assert model.fit_statistics_.rsquared_overall == 0.0
    assert model.fit_statistics_.metadata["degenerate_total_ss"]["overall"] is True
''',
    encoding="utf-8",
)
