from pathlib import Path

updates = {
    "docs/en/changelog.md": (
        "The latest numerical hardening unifies NumPy/CuPy/Torch Fama-MacBeth period dispatch under the same conservative Gram certificate and maintained SVD fallback. The certificate rejects non-finite Gram/RHS/solutions before they can preempt fallback; shared SVD least-squares uses inverse-singular-value factor ordering plus a safe uniform working scale for collectively subnormal full-rank designs. Fama-MacBeth coefficient averages and shared parameter-R² means now use cancellation-safe reduction-length scaling, and coefficient-series covariance uses per-coordinate scales with symmetric restoration so representable small-coordinate variance/cross-covariance is retained.\n",
        "The latest numerical hardening unifies NumPy/CuPy/Torch Fama-MacBeth period dispatch under the same conservative Gram certificate and maintained SVD fallback. The certificate rejects non-finite Gram/RHS/solutions before they can preempt fallback; shared SVD least-squares uses inverse-singular-value factor ordering plus a safe uniform working scale for collectively subnormal full-rank designs. Fama-MacBeth coefficient averages and shared parameter-R² means now use cancellation-safe reduction-length scaling, and coefficient-series covariance uses per-coordinate scales with symmetric restoration so representable small-coordinate variance/cross-covariance is retained. Classical model F, pooling F, and Breusch-Pagan LM now use the same overflow-safe centering and subnormal-safe backend normalization strategy, so scale-invariant diagnostics are computed before squared-unit restoration and do not become false exact fits or `Inf/Inf`/underflow artifacts. The maintained physical Stage-C runner now includes these diagnostic-scale branches for both CuPy and Torch CUDA.\n",
    ),
    "docs/cn/changelog.md": (
        "最新的 numerical hardening 让 NumPy/CuPy/Torch 的 Fama-MacBeth period solve 统一经过同一 conservative Gram certificate 与 maintained SVD fallback。certificate 会在 fallback 前拒绝 non-finite Gram/RHS/solution；shared SVD least-squares 使用 inverse-singular-value factor ordering，并对整体 subnormal 但 full-rank 的 design 使用安全统一 working scale。Fama-MacBeth coefficient average 与 shared parameter-R² mean 改为 cancellation-safe 的 reduction-length scaling；coefficient-series covariance 使用 per-coordinate scale 与 symmetric restoration，以保留本来可表示的小尺度 variance/cross-covariance。\n",
        "最新的 numerical hardening 让 NumPy/CuPy/Torch 的 Fama-MacBeth period solve 统一经过同一 conservative Gram certificate 与 maintained SVD fallback。certificate 会在 fallback 前拒绝 non-finite Gram/RHS/solution；shared SVD least-squares 使用 inverse-singular-value factor ordering，并对整体 subnormal 但 full-rank 的 design 使用安全统一 working scale。Fama-MacBeth coefficient average 与 shared parameter-R² mean 改为 cancellation-safe 的 reduction-length scaling；coefficient-series covariance 使用 per-coordinate scale 与 symmetric restoration，以保留本来可表示的小尺度 variance/cross-covariance。Classical model F、pooling F 与 Breusch-Pagan LM 现在也统一使用 overflow-safe centering 和 subnormal-safe backend normalization：先在归一化 working scale 上计算 scale-invariant diagnostic，再恢复平方单位，从而避免把有限结果误判为 exact fit，或产生 `Inf/Inf`/underflow artifact。maintained physical Stage-C runner 已把这些 diagnostic-scale 分支加入 CuPy 与 Torch CUDA 验证。\n",
    ),
}

for path, (old, new) in updates.items():
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"public changelog anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")

print("PR126 public changelogs synchronized for final cleanup")
