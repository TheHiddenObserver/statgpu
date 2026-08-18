from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# The accumulator supersedes the one-off weighted symmetric helper.
replace_once(
    "statgpu/panel/_covariance.py",
    '''def _weighted_symmetric_sum(matrix, weight):\n    """Return ``weight * (matrix + matrix.T)`` without an unsafe raw sum."""\n    return _symmetrize(matrix) * (2.0 * weight)\n\n\n''',
    "",
    "remove dead weighted symmetric helper",
)

replace_once(
    "docs/en/panel/covariance.md",
    "Symmetric covariance combinations are evaluated with range-aware arithmetic: final symmetrization avoids overflowing a finite same-sign average, two-way inclusion-exclusion subtracts a same-sign intersection component before a risky addition when possible, and HAC/Driscoll-Kraay lag pairs apply the kernel weight without first materializing an overflowing unweighted symmetric sum. These reorderings are algebraically equivalent to the displayed definitions whenever the final float64 result is representable.",
    "Symmetric covariance combinations are evaluated with range-aware arithmetic: final symmetrization avoids overflowing a finite same-sign average, and two-way inclusion-exclusion subtracts a same-sign intersection component before a risky addition when possible. HAC/Driscoll-Kraay first form the symmetric lag average rather than an overflowing unweighted symmetric sum, then accumulate the lag sequence with a per-entry reduction-length working scale only where a transient partial sum could overflow. Safe and subnormal entries remain on their original scale. These reorderings are algebraically equivalent to the displayed definitions whenever the final float64 result is representable; as elsewhere, they do not claim higher-precision recovery from arbitrarily ill-conditioned cancellation.",
    "English HAC accumulator docs",
)
replace_once(
    "docs/cn/panel/covariance.md",
    "对 symmetric covariance combination，statgpu 使用 range-aware arithmetic：最终 symmetrization 会避免有限同号平均值在相加阶段先溢出；two-way inclusion-exclusion 会在可能时先减去同号 intersection component；HAC/Driscoll-Kraay lag pair 也不会先构造可能溢出的未加权 symmetric sum，再乘较小 kernel weight。只要最终 float64 结果可表示，这些重排与上面的统计定义代数等价。",
    "对 symmetric covariance combination，statgpu 使用 range-aware arithmetic：最终 symmetrization 会避免有限同号平均值在相加阶段先溢出；two-way inclusion-exclusion 会在可能时先减去同号 intersection component。HAC/Driscoll-Kraay 会先形成 symmetric lag average，而不是先构造可能溢出的未加权 symmetric sum；随后对整个 lag sequence 只在某个 entry 的 transient partial sum 存在溢出风险时使用 per-entry reduction-length working scale，安全与 subnormal entry 保持原尺度。只要最终 float64 结果可表示，这些重排与上面的统计定义代数等价；与其他数值路径一样，并不宣称可以用更高精度恢复任意病态 cancellation。",
    "Chinese HAC accumulator docs",
)

replace_once(
    "CHANGELOG.md",
    "- **Panel covariance extreme-scale arithmetic**: clustered score grouping selectively rescales only overflow-risk group/coordinate reductions and combines positive/negative contributions after safe accumulation; covariance symmetrization, two-way cluster inclusion-exclusion, and HAC/Driscoll-Kraay symmetric lag terms use range-aware algebraic reorderings so finite representable results are not lost to avoidable intermediate overflow. The physical Stage-C validator now exercises these primitives on both CuPy and Torch CUDA.\n",
    "- **Panel covariance extreme-scale arithmetic**: clustered score grouping selectively rescales only overflow-risk group/coordinate reductions and combines positive/negative contributions after safe accumulation; covariance symmetrization and two-way cluster inclusion-exclusion use range-aware algebraic reorderings. HAC/Driscoll-Kraay additionally use a per-entry reduction-length accumulator across the complete lag sequence, so a transient positive partial sum cannot become `Inf` before later finite negative lag contributions restore a representable covariance. The physical Stage-C validator exercises these primitives on both CuPy and Torch CUDA.\n",
    "root changelog HAC accumulator",
)
replace_once(
    "docs/en/changelog.md",
    "Cluster grouping, covariance symmetrization, two-way inclusion-exclusion, and HAC/Driscoll-Kraay lag combinations now also use range-aware arithmetic to avoid preventable intermediate overflow. The maintained physical Stage-C runner includes the diagnostic-scale, zero-variance, and covariance extreme-scale branches for both CuPy and Torch CUDA.",
    "Cluster grouping, covariance symmetrization, and two-way inclusion-exclusion now use range-aware arithmetic to avoid preventable intermediate overflow. HAC/Driscoll-Kraay also use a per-entry reduction-length accumulator across the full lag sequence, so an overflowing transient partial sum cannot erase a later cancellation when the final covariance remains representable. The maintained physical Stage-C runner includes the diagnostic-scale, zero-variance, covariance extreme-scale, and lag-accumulation branches for both CuPy and Torch CUDA.",
    "English changelog HAC accumulator",
)
replace_once(
    "docs/cn/changelog.md",
    "cluster grouping、covariance symmetrization、two-way inclusion-exclusion 以及 HAC/Driscoll-Kraay lag combination 也改用 range-aware arithmetic，避免可表示有限结果因不必要的中间溢出而丢失。maintained physical Stage-C runner 已把 diagnostic-scale、zero-variance 与 covariance extreme-scale 分支加入 CuPy 与 Torch CUDA 验证。",
    "cluster grouping、covariance symmetrization 与 two-way inclusion-exclusion 也改用 range-aware arithmetic，避免可表示有限结果因不必要的中间溢出而丢失。HAC/Driscoll-Kraay 进一步对完整 lag sequence 使用 per-entry reduction-length accumulator，避免 transient partial sum 先变成 `Inf` 后使后续本可恢复的 cancellation 失效。maintained physical Stage-C runner 已把 diagnostic-scale、zero-variance、covariance extreme-scale 与 lag-accumulation 分支加入 CuPy 与 Torch CUDA 验证。",
    "Chinese changelog HAC accumulator",
)
