from pathlib import Path


p = Path("statgpu/panel/_covariance.py")
text = p.read_text(encoding="utf-8")
start = text.find("def _grouped_score_sums(")
end = text.find("def _factorize_1d_labels", start)
if start < 0 or end < 0:
    raise RuntimeError("grouped-score function boundaries not found")
new_func = r'''def _grouped_score_sums(
    scores, codes_np, *, n_groups: int, xp, return_compensation: bool = False
):
    """Sum scores by group, optionally as a two-component high/low expansion."""
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
    index = codes.unsqueeze(1).expand_as(scores) if hasattr(codes, "unsqueeze") else None

    abs_scores = xp.abs(scores)
    max_abs = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=scores)
    if hasattr(max_abs, "scatter_reduce_"):
        max_abs.scatter_reduce_(
            0, index, abs_scores, reduce="amax", include_self=True
        )
    elif type(max_abs).__module__.startswith("cupy"):
        xp.maximum.at(max_abs, codes, abs_scores)
    else:
        np.maximum.at(max_abs, codes_np, abs_scores)

    counts_np = np.bincount(codes_np, minlength=int(n_groups)).astype(np.float64)
    counts = xp_asarray(
        counts_np, dtype=xp.float64, xp=xp, ref_arr=scores
    ).reshape(-1, 1)
    limit = float(np.finfo(np.float64).max) / counts
    factor = xp.where(max_abs > limit, counts, xp.ones_like(max_abs))
    working = scores / factor[codes]

    def _signed_parts(values):
        positive = xp.where(values > 0.0, values, xp.zeros_like(values))
        negative = xp.where(values < 0.0, values, xp.zeros_like(values))
        positive_out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=scores)
        negative_out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=scores)
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

    positive_out, negative_out = _signed_parts(working)
    summed, cancellation_residual = _two_sum(positive_out, negative_out)
    if not return_compensation:
        return summed * factor

    # A final positive/negative TwoSum cannot recover a tiny same-sign term that
    # was already swallowed inside one scatter reduction (A + eps -> A).  Only
    # in compensation mode, split observations that lie inside a conservative
    # groupwise rounding envelope into a secondary accumulator.  The main and
    # tail accumulators are disjoint, so high+low has no backend-order-dependent
    # double counting.  Ordinary/nested cluster paths do not request this mode.
    group_max_work = max_abs / factor
    split_ratio = xp.minimum(
        counts * float(8.0 * np.finfo(np.float64).eps),
        xp.full_like(counts, 0.25),
    )
    threshold = group_max_work * split_ratio
    tail_mask = (xp.abs(working) < threshold[codes]) & (working != 0.0)
    has_tail = bool(_to_float_scalar(xp.any(tail_mask)))
    if not has_tail:
        return summed * factor, cancellation_residual * factor

    main = xp.where(tail_mask, xp.zeros_like(working), working)
    tail = xp.where(tail_mask, working, xp.zeros_like(working))
    main_positive, main_negative = _signed_parts(main)
    tail_positive, tail_negative = _signed_parts(tail)
    main_sum, main_residual = _two_sum(main_positive, main_negative)
    tail_sum, tail_residual = _two_sum(tail_positive, tail_negative)

    low_sum, low_error1 = _two_sum(main_residual, tail_sum)
    low_sum, low_error2 = _two_sum(low_sum, tail_residual)
    low = (low_sum + low_error1) + low_error2
    return main_sum * factor, low * factor


'''
text = text[:start] + new_func + text[end:]

old_calls = '''    grouped1, grouped1_low, correction1 = _cluster_grouped_scores(
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
    if _same_partition(c12, c1):
'''
new_calls = '''    nested_c1 = _same_partition(c12, c1)
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
'''
if old_calls not in text:
    raise RuntimeError("post-consolidation two-way grouped-score anchor not found")
text = text.replace(old_calls, new_calls, 1)
text = text.replace(
    '''    elif _same_partition(c12, c2):
''',
    '''    elif nested_c2:
''',
    1,
)
p.write_text(text.rstrip() + "\n", encoding="utf-8")
