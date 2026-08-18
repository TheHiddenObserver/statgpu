from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "docs/en/panel/fama-macbeth.md",
    "Period-coefficient averaging and parameter-R² scalar/group means use only the reduction-length scaling needed to prevent a raw sum from overflowing, so a representable small cancellation remainder is not discarded because another value is large. Coefficient-series covariance uses per-coordinate centered scales and restores each symmetric entry with the larger scale first, preserving representable small-coordinate variance and cross-covariance. If the final covariance itself is non-finite or has a negative diagonal variance, inference fails closed; an exactly zero estimated variance uses the shared tiny statistic denominator so `0/0` does not leak `NaN` into the result surface.",
    "Period-coefficient averaging and parameter-R² scalar/group means use reduction-length scaling only when a raw sum could overflow. This avoids the additional information loss caused by magnitude-normalizing an entire reduction; it does not claim compensated or higher-precision recovery for arbitrarily ill-conditioned floating-point cancellation. Coefficient-series covariance uses per-coordinate centered scales and restores each symmetric entry with the larger scale first, preserving representable small-coordinate variance and cross-covariance. If the final covariance itself is non-finite or has a negative diagonal variance, inference fails closed. At exactly zero estimated variance, a zero coefficient has statistic 0 while a nonzero coefficient has signed-infinite statistic, so no dimensionful fake denominator or `0/0` `NaN` is introduced.",
    "English FMB numerical semantics",
)
replace_once(
    "docs/en/panel/fama-macbeth.md",
    "The current head changes valid Fama-MacBeth and shared panel least-squares paths, so fresh CuPy/Torch CUDA validation is required before those historical artifacts can be superseded.",
    "The current head changes valid Fama-MacBeth, shared panel inference, residual-covariance arithmetic, and physical-validator paths, so fresh CuPy/Torch CUDA validation is required before those historical artifacts can be superseded.",
    "English FMB current physical scope",
)

replace_once(
    "docs/cn/panel/fama-macbeth.md",
    "period coefficient average 与 parameter-R² 的 scalar/group mean 只在 raw reduction 存在溢出风险时按 reduction length 做最小缩放，因此不会仅因另一个值很大就丢掉可表示的 cancellation remainder。coefficient-series covariance 使用 per-coordinate centered scale，并对每个 symmetric entry 先恢复较大尺度、再恢复较小尺度，从而保留可表示的小尺度 variance 与 cross-covariance。若最终 covariance 本身不可表示或出现负 diagonal variance，inference 会 fail closed；exact-zero estimated variance 则使用共享的 tiny statistic denominator，避免 `0/0` 把 `NaN` 泄漏到公开结果。",
    "period coefficient average 与 parameter-R² 的 scalar/group mean 只在 raw reduction 存在溢出风险时按 reduction length 做最小缩放。这样可以避免对整个 reduction 做 magnitude normalization 所额外引入的信息损失，但并不承诺用 compensated summation 或更高精度恢复任意病态的浮点 cancellation remainder。coefficient-series covariance 使用 per-coordinate centered scale，并对每个 symmetric entry 先恢复较大尺度、再恢复较小尺度，从而保留可表示的小尺度 variance 与 cross-covariance。若最终 covariance 本身不可表示或出现负 diagonal variance，inference 会 fail closed。exact-zero estimated variance 下，零 coefficient 的 statistic 为 0，非零 coefficient 的 statistic 为带符号无穷，因此既不引入带单位的伪 denominator，也不会让 `0/0` `NaN` 泄漏到公开结果。",
    "Chinese FMB numerical semantics",
)
replace_once(
    "docs/cn/panel/fama-macbeth.md",
    "当前 head 已修改有效的 Fama-MacBeth 与 shared panel least-squares 数值路径，因此在替代历史 artifact 之前仍需重新完成 CuPy/Torch CUDA 物理验证。",
    "当前 head 已修改有效的 Fama-MacBeth、shared panel inference、residual-covariance arithmetic 与 physical-validator 路径，因此在替代历史 artifact 之前仍需重新完成 CuPy/Torch CUDA 物理验证。",
    "Chinese FMB current physical scope",
)

replace_once(
    "docs/en/panel/covariance.md",
    "> Last updated: 2026-08-16  \n",
    "> Last updated: 2026-08-18<br>\n",
    "English covariance date",
)
replace_once(
    "docs/en/panel/covariance.md",
    "Nonrobust coefficient inference uses a Student-t reference. HC, clustered, and Driscoll-Kraay inference use the asymptotic normal reference used by the panel API.\n",
    "Nonrobust coefficient inference uses a Student-t reference. HC, clustered, and Driscoll-Kraay inference use the asymptotic normal reference used by the panel API. Positive covariance diagonal entries are used without an absolute variance floor, so rescaling the response rescales coefficients and standard errors without changing finite t/z statistics. At an exactly zero diagonal variance, a zero coefficient has statistic 0 and a nonzero coefficient has signed-infinite statistic; p-values and confidence intervals are then derived from that explicit result rather than from a fabricated tiny denominator.\n",
    "English covariance exact-zero inference",
)
replace_once(
    "docs/en/panel/covariance.md",
    "Two-way clustering combines the two one-way cluster covariances and subtracts the covariance for the paired cluster labels:\n",
    "For extreme but finite score magnitudes, grouped score reductions selectively use a group-size working scale only where a same-sign partial sum could overflow, and positive/negative contributions are accumulated separately before the final cancellation. This protects the reduction itself without globally magnitude-normalizing unrelated safe groups. As with ordinary float64 linear algebra, this is not a promise to recover arbitrary tiny remainders after catastrophically ill-conditioned upstream cancellation.\n\nTwo-way clustering combines the two one-way cluster covariances and subtracts the covariance for the paired cluster labels:\n",
    "English grouped score numerical contract",
)
replace_once(
    "docs/en/panel/covariance.md",
    "With `bandwidth=None`, statgpu uses $\\lfloor4(T/100)^{2/9}\\rfloor$.",
    "Symmetric covariance combinations are evaluated with range-aware arithmetic: final symmetrization avoids overflowing a finite same-sign average, two-way inclusion-exclusion subtracts a same-sign intersection component before a risky addition when possible, and HAC/Driscoll-Kraay lag pairs apply the kernel weight without first materializing an overflowing unweighted symmetric sum. These reorderings are algebraically equivalent to the displayed definitions whenever the final float64 result is representable.\n\nWith `bandwidth=None`, statgpu uses $\\lfloor4(T/100)^{2/9}\\rfloor$.",
    "English covariance range-aware combinations",
)

replace_once(
    "docs/cn/panel/covariance.md",
    "> 最后更新：2026-08-16  \n",
    "> 最后更新：2026-08-18<br>\n",
    "Chinese covariance date",
)
replace_once(
    "docs/cn/panel/covariance.md",
    "nonrobust coefficient inference 使用 Student-t reference；HC、clustered 与 Driscoll-Kraay 使用 panel API 中的 asymptotic-normal reference。\n",
    "nonrobust coefficient inference 使用 Student-t reference；HC、clustered 与 Driscoll-Kraay 使用 panel API 中的 asymptotic-normal reference。正的 covariance diagonal 不再使用绝对 variance floor，因此整体缩放 response 会按同一比例缩放 coefficient 与 standard error，而不会改变有限 t/z statistic。若 diagonal variance 精确为 0，则零 coefficient 的 statistic 为 0，非零 coefficient 的 statistic 为带符号无穷；p-value 与 confidence interval 直接由这一显式结果得到，而不是通过伪造 tiny denominator。\n",
    "Chinese covariance exact-zero inference",
)
replace_once(
    "docs/cn/panel/covariance.md",
    "双向 clustering 将两个 one-way cluster covariance 相加，再减去 paired cluster labels 对应的 covariance：\n",
    "对极端但仍有限的 score，grouped reduction 只在同号 partial sum 存在溢出风险的 group/coordinate 上使用 group-size working scale，并先分别累计正项与负项，再做最终 cancellation；不会因为某个危险 group 而对其他安全 group 做全局 magnitude normalization。和一般 float64 线性代数一样，这并不承诺在上游已经发生灾难性病态 cancellation 后恢复任意微小 remainder。\n\n双向 clustering 将两个 one-way cluster covariance 相加，再减去 paired cluster labels 对应的 covariance：\n",
    "Chinese grouped score numerical contract",
)
replace_once(
    "docs/cn/panel/covariance.md",
    "`bandwidth=None` 时使用 $\\lfloor4(T/100)^{2/9}\\rfloor$。",
    "对 symmetric covariance combination，statgpu 使用 range-aware arithmetic：最终 symmetrization 会避免有限同号平均值在相加阶段先溢出；two-way inclusion-exclusion 会在可能时先减去同号 intersection component；HAC/Driscoll-Kraay lag pair 也不会先构造可能溢出的未加权 symmetric sum，再乘较小 kernel weight。只要最终 float64 结果可表示，这些重排与上面的统计定义代数等价。\n\n`bandwidth=None` 时使用 $\\lfloor4(T/100)^{2/9}\\rfloor$。",
    "Chinese covariance range-aware combinations",
)

old_root = "- **Fama-MacBeth / shared panel numerical stability**: NumPy/CuPy/Torch now share the conservative Gram-certificate dispatch; non-finite Gram batches are masked before eigenspectrum evaluation, and non-finite Gram/RHS/solutions fall through to the maintained SVD rank policy. Shared SVD least-squares applies inverse singular values to $U^T$ before the raw-response reduction, uses a uniform safe working scale for collectively subnormal full-rank designs, and preserves the existing relative rank cutoff. Fama-MacBeth averages and parameter-R² scalar/group means use only reduction-length scaling when overflow is possible, preserving representable cancellation remainders; coefficient-series covariance uses per-coordinate scales with symmetric large-scale-first restoration. Genuinely unrepresentable covariance still fails closed, while exact-zero variance avoids `0/0` inference NaNs.\n"
new_root = "- **Fama-MacBeth / shared panel numerical stability**: NumPy/CuPy/Torch now share the conservative Gram-certificate dispatch; non-finite Gram batches are masked before eigenspectrum evaluation, and non-finite Gram/RHS/solutions fall through to the maintained SVD rank policy. Shared SVD least-squares applies inverse singular values to $U^T$ before the raw-response reduction, uses a uniform safe working scale for collectively subnormal full-rank designs, and preserves the existing relative rank cutoff. Fama-MacBeth averages and parameter-R² scalar/group means use reduction-length scaling only when overflow is possible, avoiding the extra loss caused by magnitude-normalizing an entire reduction without claiming arbitrary compensated recovery; coefficient-series covariance uses per-coordinate scales with symmetric large-scale-first restoration. Positive inference variances are no longer replaced by an absolute floor, while exact-zero variance maps zero coefficients to statistic 0 and nonzero coefficients to signed infinity instead of using a fake tiny denominator.\n- **Panel covariance extreme-scale arithmetic**: clustered score grouping selectively rescales only overflow-risk group/coordinate reductions and combines positive/negative contributions after safe accumulation; covariance symmetrization, two-way cluster inclusion-exclusion, and HAC/Driscoll-Kraay symmetric lag terms use range-aware algebraic reorderings so finite representable results are not lost to avoidable intermediate overflow. The physical Stage-C validator now exercises these primitives on both CuPy and Torch CUDA.\n"
replace_once("CHANGELOG.md", old_root, new_root, "root changelog numerical bullet")

old_en = "The latest numerical hardening unifies NumPy/CuPy/Torch Fama-MacBeth period dispatch under the same conservative Gram certificate and maintained SVD fallback. The certificate rejects non-finite Gram/RHS/solutions before they can preempt fallback; shared SVD least-squares uses inverse-singular-value factor ordering plus a safe uniform working scale for collectively subnormal full-rank designs. Fama-MacBeth coefficient averages and shared parameter-R² means now use cancellation-safe reduction-length scaling, and coefficient-series covariance uses per-coordinate scales with symmetric restoration so representable small-coordinate variance/cross-covariance is retained. Classical model F, pooling F, and Breusch-Pagan LM now use the same overflow-safe centering and subnormal-safe backend normalization strategy, so scale-invariant diagnostics are computed before squared-unit restoration and do not become false exact fits or `Inf/Inf`/underflow artifacts. The maintained physical Stage-C runner now includes these diagnostic-scale branches for both CuPy and Torch CUDA.\n"
new_en = "The latest numerical hardening unifies NumPy/CuPy/Torch Fama-MacBeth period dispatch under the same conservative Gram certificate and maintained SVD fallback. The certificate rejects non-finite Gram/RHS/solutions before they can preempt fallback; shared SVD least-squares uses inverse-singular-value factor ordering plus a safe uniform working scale for collectively subnormal full-rank designs. Fama-MacBeth coefficient averages and shared parameter-R² means use reduction-length scaling only when overflow is possible, avoiding extra magnitude-normalization loss without claiming arbitrary compensated cancellation recovery; coefficient-series covariance uses per-coordinate scales with symmetric restoration. Shared panel inference no longer imposes an absolute variance floor: exact-zero variance maps a zero coefficient to statistic 0 and a nonzero coefficient to signed infinity. Classical model F, pooling F, and Breusch-Pagan LM use overflow-safe centering and subnormal-safe backend normalization. Cluster grouping, covariance symmetrization, two-way inclusion-exclusion, and HAC/Driscoll-Kraay lag combinations now also use range-aware arithmetic to avoid preventable intermediate overflow. The maintained physical Stage-C runner includes the diagnostic-scale, zero-variance, and covariance extreme-scale branches for both CuPy and Torch CUDA.\n"
replace_once("docs/en/changelog.md", old_en, new_en, "English public changelog numerical paragraph")

old_cn = "最新的 numerical hardening 让 NumPy/CuPy/Torch 的 Fama-MacBeth period solve 统一经过同一 conservative Gram certificate 与 maintained SVD fallback。certificate 会在 fallback 前拒绝 non-finite Gram/RHS/solution；shared SVD least-squares 使用 inverse-singular-value factor ordering，并对整体 subnormal 但 full-rank 的 design 使用安全统一 working scale。Fama-MacBeth coefficient average 与 shared parameter-R² mean 改为 cancellation-safe 的 reduction-length scaling；coefficient-series covariance 使用 per-coordinate scale 与 symmetric restoration，以保留本来可表示的小尺度 variance/cross-covariance。Classical model F、pooling F 与 Breusch-Pagan LM 现在也统一使用 overflow-safe centering 和 subnormal-safe backend normalization：先在归一化 working scale 上计算 scale-invariant diagnostic，再恢复平方单位，从而避免把有限结果误判为 exact fit，或产生 `Inf/Inf`/underflow artifact。maintained physical Stage-C runner 已把这些 diagnostic-scale 分支加入 CuPy 与 Torch CUDA 验证。\n"
new_cn = "最新的 numerical hardening 让 NumPy/CuPy/Torch 的 Fama-MacBeth period solve 统一经过同一 conservative Gram certificate 与 maintained SVD fallback。certificate 会在 fallback 前拒绝 non-finite Gram/RHS/solution；shared SVD least-squares 使用 inverse-singular-value factor ordering，并对整体 subnormal 但 full-rank 的 design 使用安全统一 working scale。Fama-MacBeth coefficient average 与 shared parameter-R² mean 只在存在溢出风险时使用 reduction-length scaling，以避免全局 magnitude normalization 额外造成的信息损失，但不宣称可以恢复任意病态 cancellation remainder；coefficient-series covariance 使用 per-coordinate scale 与 symmetric restoration。shared panel inference 不再施加绝对 variance floor：exact-zero variance 下，零 coefficient 的 statistic 为 0，非零 coefficient 为带符号无穷。Classical model F、pooling F 与 Breusch-Pagan LM 使用 overflow-safe centering 和 subnormal-safe backend normalization；cluster grouping、covariance symmetrization、two-way inclusion-exclusion 以及 HAC/Driscoll-Kraay lag combination 也改用 range-aware arithmetic，避免可表示有限结果因不必要的中间溢出而丢失。maintained physical Stage-C runner 已把 diagnostic-scale、zero-variance 与 covariance extreme-scale 分支加入 CuPy 与 Torch CUDA 验证。\n"
replace_once("docs/cn/changelog.md", old_cn, new_cn, "Chinese public changelog numerical paragraph")
