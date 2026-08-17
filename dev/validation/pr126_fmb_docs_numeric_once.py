from pathlib import Path

cases = [
    (
        Path('docs/en/panel/fama-macbeth.md'),
        'At least two valid periods must remain after filtering; otherwise `.fit()` raises an error because the variability of the coefficient series cannot be estimated from fewer than two periods. Every retained period must also have full column rank under the shared panel SVD cutoff; a rank-deficient retained period fails closed before inference. On GPU, the Gram certificate is evaluated far from that numerical boundary and only selects whether the coefficient solve may use the fast path. Uncertified periods retain the original SVD cutoff and fail-closed behavior, and all paths preserve the chronological order of `betas_`; if several periods are deficient, the public error still identifies the earliest deficient retained period in chronological order.',
        'At least two valid periods must remain after filtering; otherwise `.fit()` raises an error because the variability of the coefficient series cannot be estimated from fewer than two periods. Every retained period must also have full column rank under the shared panel SVD cutoff; a rank-deficient retained period fails closed before inference. On GPU, the Gram certificate is evaluated far from that numerical boundary and only selects whether the coefficient solve may use the fast path. Uncertified periods retain the original SVD cutoff and fail-closed behavior, and all paths preserve the chronological order of `betas_`; if several periods are deficient, the public error still identifies the earliest deficient retained period in chronological order.\n\nNumerical safety is fail-closed across this path. A non-finite batched Gram right-hand side or candidate solution is treated as uncertified and routed through the rank-revealing SVD fallback. Period-coefficient means and covariance use scaled reductions so finite, representable results are not lost to avoidable float64 intermediate overflow; if the final covariance itself is non-finite or has a negative diagonal variance, inference raises instead of publishing clipped or non-finite standard errors. The parameter-based overall, within, and between $R^2$ diagnostics use the same scale-invariant principle, including scaled entity group means when `entity_ids` is supplied, so finite large-level panels do not turn fit statistics into overflow-driven `NaN` values.',
        'Numerical safety: the certified Gram fast path treats a non-finite batched right-hand side or solution as unsafe and routes that period through the rank-revealing SVD fallback. The period-coefficient mean and covariance use scaled reductions to avoid avoidable float64 overflow; if the final covariance itself is non-finite or has a negative diagonal variance, inference fails closed instead of publishing clipped or non-finite standard errors.',
    ),
    (
        Path('docs/cn/panel/fama-macbeth.md'),
        '过滤后至少需要两个有效 period，否则 `.fit()` 会报错，因为少于两个 period 无法估计 coefficient series 的波动。每个 retained period 还必须在共享 panel SVD cutoff 下 full column rank；若某个 retained period rank deficient，会在 inference 之前 fail closed。GPU 上的 Gram certificate 被刻意放在远离 numerical rank boundary 的区域，它只决定 coefficient solve 是否可以使用 fast path；uncertified periods 继续采用原有 SVD cutoff 与 fail-closed 语义。所有路径都保持 `betas_` chronology；若多个时期 rank deficient，公开错误仍定位 chronology 上最早的 deficient retained period。',
        '过滤后至少需要两个有效 period，否则 `.fit()` 会报错，因为少于两个 period 无法估计 coefficient series 的波动。每个 retained period 还必须在共享 panel SVD cutoff 下 full column rank；若某个 retained period rank deficient，会在 inference 之前 fail closed。GPU 上的 Gram certificate 被刻意放在远离 numerical rank boundary 的区域，它只决定 coefficient solve 是否可以使用 fast path；uncertified periods 继续采用原有 SVD cutoff 与 fail-closed 语义。所有路径都保持 `betas_` chronology；若多个时期 rank deficient，公开错误仍定位 chronology 上最早的 deficient retained period。\n\n这一数值路径采用 fail-closed 语义。若批量 Gram 右端项或候选解出现非有限值，该时期会被视为 uncertified，并转入秩揭示 SVD 回退。时期系数均值与协方差使用缩放后的归约，避免有限且可表示的结果因为 float64 中间量的可避免溢出而丢失；若最终协方差本身仍为非有限值或出现负的对角方差，则 inference 会直接报错，而不会发布截断后或非有限的标准误。parameter-based overall、within 与 between $R^2$ 也采用同样的 scale-invariant 原则；提供 `entity_ids` 时，entity group mean 同样先缩放后聚合，从而避免有限的大量级 panel 把公开 fit statistics 变成由溢出造成的 `NaN`。',
        '数值安全性：经认证的 Gram 快速路径若发现批量右端项或求解结果为非有限值，会将对应时期转入秩揭示 SVD 回退路径。时期系数均值与协方差采用缩放后的归约以避免可避免的 float64 溢出；若最终协方差本身仍为非有限值或出现负的对角方差，则推断会直接报错，而不会发布截断后或非有限的标准误。',
    ),
]

for path, anchor, replacement, stale_tail in cases:
    text = path.read_text(encoding='utf-8')
    if text.count(anchor) != 1:
        raise RuntimeError(f'numerical behavior anchor count != 1: {path}')
    if text.count(stale_tail) != 1:
        raise RuntimeError(f'stale tail note count != 1: {path}')
    text = text.replace(anchor, replacement, 1)
    text = text.replace('\n\n' + stale_tail, '', 1)
    path.write_text(text, encoding='utf-8')
