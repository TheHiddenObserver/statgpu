from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:160]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/panel/_utils.py",
    """Performance note: all group-level operations use scatter-add to compute\ngroup sums and counts in a single kernel launch, avoiding per-group\nPython loops and their associated GPU-CPU synchronization overhead.\n""",
    """Performance note: ordinary-scale group reductions retain the single-scatter\nfast path. Columns whose dynamic range can hide a representable cancellation\nremainder use the shared backend-native magnitude-tiered reducer after one packed\nrisk classification; neither path introduces per-group Python loops.\n""",
)

replace_once(
    "docs/en/panel/fama-macbeth.md",
    """Period-coefficient averaging and parameter-R² scalar/group means use reduction-length scaling only when a raw sum could overflow. This avoids the additional information loss caused by magnitude-normalizing an entire reduction; it does not claim compensated or higher-precision recovery for arbitrarily ill-conditioned floating-point cancellation remainder.""",
    """Period-coefficient averaging and parameter-R² scalar/group means use the shared magnitude-tiered float64 reduction policy. Ordinary-scale inputs remain on the single-tier/single-scatter fast path; extra magnitude tiers are activated only when dynamic range can hide a representable low-order contribution or an unscaled same-sign reduction can overflow. The stable path separates magnitude bands before signed accumulation, while mean callers apply observation/group-count scaling only where the final mean is representable but the raw sum is not. This remains float64 arithmetic rather than arbitrary-precision or exact summation, but representable low-order cancellation remainders are not intentionally discarded merely because much larger terms are present.""",
)

replace_once(
    "docs/cn/panel/fama-macbeth.md",
    """period coefficient average 与 parameter-R² 的 scalar/group mean 只在 raw reduction 存在溢出风险时按 reduction length 做最小缩放。这样可以避免对整个 reduction 做 magnitude normalization 所额外引入的信息损失，但并不承诺用 compensated summation 或更高精度恢复任意病态的浮点 cancellation remainder。""",
    """period coefficient average 与 parameter-R² 的 scalar/group mean 现在共享 magnitude-tiered float64 reduction policy。普通尺度输入仍走 single-tier/single-scatter fast path；只有当动态范围可能吞掉仍可表示的低阶贡献，或未缩放的同号求和可能溢出时，才启用额外 magnitude tier。稳定路径会先按数量级分离再做 signed accumulation，而 mean caller 只在“最终均值可表示、raw sum 可能不可表示”的位置按 observation/group count 做缩放。该实现仍属于 float64 arithmetic，并不等同于 arbitrary-precision 或 exact summation；但不会仅仅因为同时存在更大项，就主动丢弃仍可表示的低阶 cancellation remainder。""",
)

replace_once(
    "CHANGELOG.md",
    """- **Fama-MacBeth / shared panel numerical stability**: NumPy/CuPy/Torch now share the conservative Gram-certificate dispatch; non-finite Gram batches are masked before eigenspectrum evaluation, and non-finite Gram/RHS/solutions fall through to the maintained SVD rank policy. Shared SVD least-squares applies inverse singular values to $U^T$ before the raw-response reduction, uses a uniform safe working scale for collectively subnormal full-rank designs, and preserves the existing relative rank cutoff. Fama-MacBeth averages and parameter-R² scalar/group means use reduction-length scaling only when overflow is possible, avoiding the extra loss caused by magnitude-normalizing an entire reduction without claiming arbitrary compensated recovery; coefficient-series covariance uses per-coordinate scales with symmetric large-scale-first restoration. Positive inference variances are no longer replaced by an absolute floor, while exact-zero variance maps zero coefficients to statistic 0 and nonzero coefficients to signed infinity instead of using a fake tiny denominator.""",
    """- **Fama-MacBeth / shared panel numerical stability**: NumPy/CuPy/Torch now share the conservative Gram-certificate dispatch; non-finite Gram batches are masked before eigenspectrum evaluation, and non-finite Gram/RHS/solutions fall through to the maintained SVD rank policy. Shared SVD least-squares applies inverse singular values to $U^T$ before the raw-response reduction, uses a uniform safe working scale for collectively subnormal full-rank designs, and preserves the existing relative rank cutoff. Fama-MacBeth coefficient averages, fixed-effect group means, and parameter-R² scalar/group means now share a magnitude-tiered float64 reduction path for dynamic-range/cancellation risk while ordinary columns retain the existing fast scatter reduction. Mean-level count scaling is kept outside the generic grouped-sum primitive so representable subnormal cancellation tails are not erased at an exact range boundary; coefficient-series covariance continues to use per-coordinate scales with symmetric large-scale-first restoration. Positive inference variances are no longer replaced by an absolute floor, while exact-zero variance maps zero coefficients to statistic 0 and nonzero coefficients to signed infinity instead of using a fake tiny denominator.""",
)
