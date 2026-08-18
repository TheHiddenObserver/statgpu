from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Fama-MacBeth EN: all maintained backends now share the conservative Gram
# certificate, while SVD fallback uses factor ordering rather than RHS scaling.
replace_once(
    "docs/en/panel/fama-macbeth.md",
    "The NumPy reference path keeps the long-standing serial rank-revealing SVD policy. GPU paths group retained periods by their exact row count, without zero padding, and first form batched $G_t=X_t^\\top X_t$ and $X_t^\\top y_t$. A period may consume the Gram-solve candidate only when the backend-native spectrum satisfies $\\lambda_{\\min}(G_t)/\\lambda_{\\max}(G_t)>10^{-4}$, which restricts the fast path to clearly well-conditioned designs. The certificate is a performance gate rather than a new rank definition: every uncertified period falls back to the maintained SVD cutoff $\\max(n_t,k)\\epsilon s_{\\max,t}$. Torch can solve an unsafe subset with its documented stacked-SVD support, while CuPy keeps supported two-dimensional SVD fallbacks. Thus near-rank-boundary and rank-deficient behavior remains SVD-owned even though well-conditioned GPU periods avoid the substantially more expensive rank-revealing SVD.",
    "NumPy, CuPy, and Torch now use the same conservative exact-size Gram-certificate dispatch. Retained periods are grouped by their actual row count, without zero padding, and the candidate path forms batched $G_t=X_t^\\top X_t$ and $X_t^\\top y_t$. A period may consume the Gram solve only when the backend-native spectrum satisfies $\\lambda_{\\min}(G_t)/\\lambda_{\\max}(G_t)>10^{-4}$ and the Gram matrix, right-hand side, and candidate solution are finite. Non-finite Gram batches are masked before the spectrum calculation so the performance certificate itself cannot preempt the fallback with a backend linear-algebra error. The certificate is only a performance gate: every uncertified period remains governed by the maintained SVD cutoff $\\max(n_t,k)\\epsilon s_{\\max,t}$. Torch uses its documented stacked-SVD support for unsafe subsets, while NumPy/CuPy retain supported two-dimensional fallbacks. Near-rank-boundary and rank-deficient behavior therefore remains SVD-owned on every backend.",
    "EN estimator dispatch",
)
replace_once(
    "docs/en/panel/fama-macbeth.md",
    "Numerical safety is fail-closed across this path. A non-finite batched Gram right-hand side or candidate solution is treated as uncertified and routed through the rank-revealing SVD fallback. The shared serial, deferred-rank, and batched SVD solvers scale each response/target before the orthogonal projection and rescale the final coefficient, so a representable solution is not lost merely because the unscaled `U^T y` accumulation would overflow. Period-coefficient means and covariance use scaled reductions so finite, representable results are not lost to avoidable float64 intermediate overflow; if the final covariance itself is non-finite or has a negative diagonal variance, inference raises instead of publishing clipped or non-finite standard errors. The parameter-based overall, within, and between $R^2$ diagnostics use the same scale-invariant principle, including scaled entity group means when `entity_ids` is supplied, so finite large-level panels do not turn fit statistics into overflow-driven `NaN` values.",
    "Numerical safety is fail-closed across this path. A non-finite Gram matrix, batched right-hand side, or candidate solution is treated as uncertified and routed through the rank-revealing SVD fallback. The shared SVD solvers apply retained inverse singular values to the rows of $U^\\top$ before multiplying by the raw response, avoiding an overflowing `U^T y` intermediate without magnitude-normalizing away tiny response components. If an entire full-rank design is below $\\sqrt{\\mathrm{DBL\\_MIN}}$, it is uniformly raised to that safe working scale before SVD and the final coefficient is transformed back; this positive scalar change leaves the relative rank cutoff unchanged. Period-coefficient averaging and parameter-R² scalar/group means use only the reduction-length scaling needed to prevent a raw sum from overflowing, so a representable small cancellation remainder is not discarded because another value is large. Coefficient-series covariance uses per-coordinate centered scales and restores each symmetric entry with the larger scale first, preserving representable small-coordinate variance and cross-covariance. If the final covariance itself is non-finite or has a negative diagonal variance, inference fails closed; an exactly zero estimated variance uses the shared tiny statistic denominator so `0/0` does not leak `NaN` into the result surface.",
    "EN numerical behavior",
)
replace_once(
    "docs/en/panel/fama-macbeth.md",
    "Fresh Tesla P100 evidence on numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` reports",
    "Historical Tesla P100 evidence on numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` reports",
    "EN historical P100 wording",
)
replace_once(
    "docs/en/panel/fama-macbeth.md",
    "Numerical differences in the accepted P100 scaling artifact remain tight:",
    "For that historical source, numerical differences in the accepted P100 scaling artifact remain tight:",
    "EN historical artifact wording",
)
# Add current-source status after historical performance paragraph.
replace_once(
    "docs/en/panel/fama-macbeth.md",
    "These measurements demonstrate the maintained benchmark crossover on that P100 resident-array protocol; they are not a universal hardware guarantee.",
    "These measurements demonstrate the maintained benchmark crossover on that historical P100 resident-array protocol; they are not a universal hardware guarantee and are not physical acceptance for the current numerical head. The current head changes valid Fama-MacBeth and shared panel least-squares paths, so fresh CuPy/Torch CUDA validation is required before those historical artifacts can be superseded.",
    "EN current physical status",
)
replace_once(
    "docs/en/panel/fama-macbeth.md",
    "> Last updated: 2026-08-17  ",
    "> Last updated: 2026-08-18  ",
    "EN date",
)

# Chinese mirror.
replace_once(
    "docs/cn/panel/fama-macbeth.md",
    "NumPy reference path 保留原有 serial rank-revealing SVD policy。GPU 路径按真实 $n_t$ 做 exact-size grouping，不使用 zero padding，并先批量构造 $G_t=X_t^\\top X_t$ 与 $X_t^\\top y_t$。只有当 backend-native Gram spectrum 满足 $\\lambda_{\\min}(G_t)/\\lambda_{\\max}(G_t)>10^{-4}$ 时，该时期才允许使用 batched Gram solve candidate，因此 fast path 只覆盖明显 well-conditioned 的设计。这个 certificate 只是 performance gate，并没有替换 rank 定义：所有 uncertified period 都回退到原有 $\\max(n_t,k)\\epsilon s_{\\max,t}$ SVD cutoff。Torch 对 unsafe subset 可以使用其有文档保证的 stacked-SVD，CuPy 则继续使用受支持的二维 SVD fallback。这样 near-rank-boundary 与 rank-deficient 行为仍由原 SVD policy 决定，而 clearly well-conditioned 的 GPU periods 可以避免更昂贵的 rank-revealing SVD。",
    "NumPy、CuPy 与 Torch 现在统一使用同一套 conservative exact-size Gram-certificate dispatch。retained period 按真实 row count 分组，不使用 zero padding；candidate path 批量构造 $G_t=X_t^\\top X_t$ 与 $X_t^\\top y_t$。只有当 backend-native spectrum 满足 $\\lambda_{\\min}(G_t)/\\lambda_{\\max}(G_t)>10^{-4}$，且 Gram matrix、右端项与 candidate solution 都为有限值时，才允许消费 Gram solve。若 Gram batch 已经非有限，会先在 spectrum calculation 前用安全 placeholder 屏蔽，因此 performance certificate 自身不会先于 fallback 抛出 backend linear-algebra error。这个 certificate 仍然只是 performance gate：所有 uncertified period 都由既有 $\\max(n_t,k)\\epsilon s_{\\max,t}$ SVD cutoff 决定。Torch 对 unsafe subset 使用其有文档保证的 stacked-SVD，NumPy/CuPy 保持受支持的二维 fallback，因此所有 backend 的 near-rank-boundary 与 rank-deficient 行为仍由 SVD policy 负责。",
    "CN estimator dispatch",
)
replace_once(
    "docs/cn/panel/fama-macbeth.md",
    "这一数值路径采用 fail-closed 语义。若批量 Gram 右端项或候选解出现非有限值，该时期会被视为 uncertified，并转入秩揭示 SVD 回退。共享的 serial、deferred-rank 与 batched SVD solver 会在正交投影前按 response/target 缩放，并在最终 coefficient 上还原该尺度，因此不会仅因为未缩放的 `U^T y` 累加溢出而丢失本来可表示的解。时期系数均值与协方差使用缩放后的归约，避免有限且可表示的结果因为 float64 中间量的可避免溢出而丢失；若最终协方差本身仍为非有限值或出现负的对角方差，则 inference 会直接报错，而不会发布截断后或非有限的标准误。parameter-based overall、within 与 between $R^2$ 也采用同样的 scale-invariant 原则；提供 `entity_ids` 时，entity group mean 同样先缩放后聚合，从而避免有限的大量级 panel 把公开 fit statistics 变成由溢出造成的 `NaN`。",
    "这一数值路径采用 fail-closed 语义。若 Gram matrix、批量右端项或 candidate solution 出现非有限值，该 period 会被视为 uncertified，并进入 rank-revealing SVD fallback。共享 SVD solver 会先把 retained inverse singular value 作用到 $U^\\top$ 的各行，再与原始 response 相乘，从而避免 `U^T y` 中间量溢出，同时不通过 magnitude normalization 丢掉很小但仍可表示的 response component。若整个 full-rank design 都低于 $\\sqrt{\\mathrm{DBL\\_MIN}}$，会先做统一的正比例 working-scale 提升，并在最终 coefficient 上还原；该正比例变化不改变 relative rank cutoff。period coefficient average 与 parameter-R² 的 scalar/group mean 只在 raw reduction 存在溢出风险时按 reduction length 做最小缩放，因此不会仅因另一个值很大就丢掉可表示的 cancellation remainder。coefficient-series covariance 使用 per-coordinate centered scale，并对每个 symmetric entry 先恢复较大尺度、再恢复较小尺度，从而保留可表示的小尺度 variance 与 cross-covariance。若最终 covariance 本身不可表示或出现负 diagonal variance，inference 会 fail closed；exact-zero estimated variance 则使用共享的 tiny statistic denominator，避免 `0/0` 把 `NaN` 泄漏到公开结果。",
    "CN numerical behavior",
)
replace_once(
    "docs/cn/panel/fama-macbeth.md",
    "在 numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` 上的 fresh Tesla P100 evidence 中",
    "在 numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` 上的历史 Tesla P100 evidence 中",
    "CN historical P100 wording",
)
replace_once(
    "docs/cn/panel/fama-macbeth.md",
    "这些结果说明维护中的 P100 resident-array protocol 已经在三个 scale 上全部 crossover；它们并不是对所有硬件与数据搬运场景的普遍性能承诺。",
    "这些结果说明该历史 P100 resident-array protocol 在三个 scale 上全部 crossover；它们既不是对所有硬件与数据搬运场景的普遍性能承诺，也不是当前 numerical head 的 physical acceptance。当前 head 已修改有效的 Fama-MacBeth 与 shared panel least-squares 数值路径，因此在替代历史 artifact 之前仍需重新完成 CuPy/Torch CUDA 物理验证。",
    "CN current physical status",
)
replace_once(
    "docs/cn/panel/fama-macbeth.md",
    "accepted P100 scaling artifact 的数值差异仍然很小：",
    "对该历史 source，accepted P100 scaling artifact 的数值差异仍然很小：",
    "CN historical artifact wording",
)
replace_once(
    "docs/cn/panel/fama-macbeth.md",
    "> 最后更新：2026-08-17  ",
    "> 最后更新：2026-08-18  ",
    "CN date",
)

# Root changelog: replace stale response-scaling description and mark old GPU
# acceptance as historical after this loop's valid-path changes.
replace_once(
    "CHANGELOG.md",
    "- **Fama-MacBeth numerical stability**: certified Gram batching now rejects non-finite RHS/solutions and falls back to the shared SVD policy; coefficient-series means/covariances use scaled reductions to avoid representable float64 overflow, while genuinely non-finite or negative-variance covariance fails closed before inference. Parameter-based overall/within/between R² also uses scaled mean and group-mean reductions so finite large-level panels do not produce overflow-driven NaN fit statistics, including when `entity_ids` is supplied. The shared serial/deferred/batched panel SVD solvers also scale response RHS values before orthogonal projection, preventing representable coefficients from being lost when `U^T y` would overflow.",
    "- **Fama-MacBeth / shared panel numerical stability**: NumPy/CuPy/Torch now share the conservative Gram-certificate dispatch; non-finite Gram batches are masked before eigenspectrum evaluation, and non-finite Gram/RHS/solutions fall through to the maintained SVD rank policy. Shared SVD least-squares applies inverse singular values to $U^T$ before the raw-response reduction, uses a uniform safe working scale for collectively subnormal full-rank designs, and preserves the existing relative rank cutoff. Fama-MacBeth averages and parameter-R² scalar/group means use only reduction-length scaling when overflow is possible, preserving representable cancellation remainders; coefficient-series covariance uses per-coordinate scales with symmetric large-scale-first restoration. Genuinely unrepresentable covariance still fails closed, while exact-zero variance avoids `0/0` inference NaNs.",
    "root numerical changelog",
)
replace_once(
    "CHANGELOG.md",
    "- Standardized Fama-MacBeth inference aliases/results without changing backend-native public arrays, moved well-conditioned GPU period solves to a conservative Gram-certified exact-size batch with the original SVD rank policy as fail-closed fallback, removed duplicate direct-fit finite scanning, and retained backend-native distribution inference plus a single reporting snapshot.",
    "- Standardized Fama-MacBeth inference aliases/results without changing backend-native public arrays, moved clearly well-conditioned NumPy/CuPy/Torch period solves to a shared conservative Gram-certified exact-size batch with the original SVD rank policy as fail-closed fallback, removed duplicate direct-fit finite scanning, and retained backend-native distribution inference plus a single reporting snapshot.",
    "root backend changelog",
)
replace_once(
    "CHANGELOG.md",
    "- Completed exact-source Tesla P100 acceptance on numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4`:",
    "- Historical exact-source Tesla P100 acceptance on numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` recorded that",
    "root historical physical wording",
)
# Add explicit current-source status before the next bullet.
replace_once(
    "CHANGELOG.md",
    "- Added maintained Python/R external-definition checks and preserved the final physical artifacts under `results/pr126_p100_fama_fix/`; legacy Gaussian linear-model inference backend migration is tracked separately in #127.",
    "- The subsequent review-fix loops changed valid Fama-MacBeth and shared panel least-squares paths, so the `8c60db00...` P100 artifacts are historical-only for the current branch; fresh exact-head CuPy/Torch CUDA acceptance is required before merge readiness can be promoted.\n- Added maintained Python/R external-definition checks and preserved the historical physical artifacts under `results/pr126_p100_fama_fix/`; legacy Gaussian linear-model inference backend migration is tracked separately in #127.",
    "root current physical status",
)

# Public changelog mirrors: correct current acceptance status and add the latest
# numerical hardening summary without duplicating the detailed estimator page.
for path, language in (("docs/en/changelog.md", "en"), ("docs/cn/changelog.md", "cn")):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if language == "en":
        text = text.replace("> Last updated: 2026-08-17<br>", "> Last updated: 2026-08-18<br>", 1)
        physical_old = "Physical validation is recorded as an exact-source evidence chain. Historical Stage-C and Fama-MacBeth artifacts remain valid for their original numerical heads, while the final PR126 acceptance is anchored to clean numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4`."
        physical_new = "Physical validation is recorded as an exact-source evidence chain. Historical Stage-C and Fama-MacBeth artifacts remain valid only for their original numerical heads. The previously accepted P100 source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` is now historical for the current PR branch because later review-fix loops changed valid Fama-MacBeth and shared panel least-squares paths; fresh exact-head CuPy/Torch CUDA acceptance is required before merge readiness can be promoted."
        if physical_old not in text:
            raise RuntimeError("EN public changelog physical anchor missing")
        text = text.replace(physical_old, physical_new, 1)
        insertion = "\n\nThe latest numerical hardening unifies NumPy/CuPy/Torch Fama-MacBeth period dispatch under the same conservative Gram certificate and maintained SVD fallback. The certificate rejects non-finite Gram/RHS/solutions before they can preempt fallback; shared SVD least-squares uses inverse-singular-value factor ordering plus a safe uniform working scale for collectively subnormal full-rank designs. Fama-MacBeth coefficient averages and shared parameter-R² means now use cancellation-safe reduction-length scaling, and coefficient-series covariance uses per-coordinate scales with symmetric restoration so representable small-coordinate variance/cross-covariance is retained.\n"
        marker = "\n\nPhysical validation is recorded as an exact-source evidence chain."
    else:
        text = text.replace("> 最后更新：2026-08-17<br>", "> 最后更新：2026-08-18<br>", 1)
        physical_old = "物理验证按照 exact-source evidence chain 记录。历史 Stage-C 与 Fama-MacBeth artifact 继续对各自 numerical head 有效；PR126 最终 acceptance 则统一锚定 clean numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4`。"
        physical_new = "物理验证按照 exact-source evidence chain 记录。历史 Stage-C 与 Fama-MacBeth artifact 只继续对各自原始 numerical head 有效。此前接受的 P100 source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` 对当前 PR branch 已属于历史证据，因为后续 review-fix loop 修改了有效的 Fama-MacBeth 与 shared panel least-squares 路径；在提升 merge readiness 之前必须重新完成 exact-head CuPy/Torch CUDA acceptance。"
        if physical_old not in text:
            raise RuntimeError("CN public changelog physical anchor missing")
        text = text.replace(physical_old, physical_new, 1)
        insertion = "\n\n最新的 numerical hardening 让 NumPy/CuPy/Torch 的 Fama-MacBeth period solve 统一经过同一 conservative Gram certificate 与 maintained SVD fallback。certificate 会在 fallback 前拒绝 non-finite Gram/RHS/solution；shared SVD least-squares 使用 inverse-singular-value factor ordering，并对整体 subnormal 但 full-rank 的 design 使用安全统一 working scale。Fama-MacBeth coefficient average 与 shared parameter-R² mean 改为 cancellation-safe 的 reduction-length scaling；coefficient-series covariance 使用 per-coordinate scale 与 symmetric restoration，以保留本来可表示的小尺度 variance/cross-covariance。\n"
        marker = "\n\n物理验证按照 exact-source evidence chain 记录。"
    if marker not in text:
        raise RuntimeError(f"{language} public changelog insertion anchor missing")
    text = text.replace(marker, insertion + marker, 1)
    p.write_text(text, encoding="utf-8")

print("PR126 final numerical documentation synchronized")
