from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "docs/en/models/panel.md",
    "Stage C of the Tier-1 panel roadmap completes the residual-sandwich covariance layer on top of the Stage-B diagnostics: historical defaults remain unchanged, while HC0/HC2/HC3, robust RandomEffects inference, explicit cluster group debiasing, and Driscoll-Kraay covariance are added with NumPy/CuPy/Torch-native accumulation. Physical CUDA acceptance is recorded from exact-clean-head Tesla P100 runs: all 26 estimator cases plus two direct public covariance primitives pass on both CuPy and Torch, together with synchronized performance evidence including the bounded high-T QS scenario.",
    "Stage C of the Tier-1 panel roadmap completes the residual-sandwich covariance layer on top of the Stage-B diagnostics: historical defaults remain unchanged, while HC0/HC2/HC3, robust RandomEffects inference, explicit cluster group debiasing, and Driscoll-Kraay covariance are added with NumPy/CuPy/Torch-native accumulation. Tesla P100 correctness and synchronized performance evidence were recorded for the pre-review exact-clean head `9c0b3050...`; the subsequent ordered-categorical chronology fix changes production covariance input handling, so those artifacts are now historical evidence and a fresh exact-head P100 rerun is required before PR #126 returns to Ready.",
)
replace_once(
    "docs/en/models/panel.md",
    "Hosted Stage-C tests pin HC2/HC3 against analytic/statsmodels fit-space calculations and cluster/Driscoll-Kraay definitions against `linearmodels==7.0`. Exact-clean-head Tesla P100 acceptance is recorded for both CuPy and Torch: 26/26 estimator covariance cases plus 2/2 direct public covariance primitives per backend, with requested/executed backend identity and no CPU fallback. The synchronized performance artifact also covers the explicit `N=10,000`, `k=2`, `T=200` QS all-lag scenario; it does not encode a speedup or CPU-baseline claim.",
    "Hosted Stage-C tests pin HC2/HC3 against analytic/statsmodels fit-space calculations and cluster/Driscoll-Kraay definitions against `linearmodels==7.0`. The previously recorded Tesla P100 artifacts passed 26/26 estimator covariance cases plus 2/2 direct public covariance primitives per backend and included synchronized `N=10,000`, `k=2`, `T=200` QS timing. Because the post-review ordered-categorical chronology fix modifies production covariance metadata handling, those artifacts remain historical evidence but no longer close the exact-head physical gate; the updated candidate must be rerun on CuPy and Torch before Ready/merge consideration.",
)

replace_once(
    "docs/cn/models/panel.md",
    "Tier-1 Panel 路线的 Stage C 在 Stage-B diagnostics 之上补齐 residual-sandwich covariance 层：历史默认行为保持不变，同时加入 HC0/HC2/HC3、RandomEffects robust inference、显式 cluster group debias 与 Driscoll-Kraay，并保持 NumPy/CuPy/Torch 数值累积后端原生。physical CUDA acceptance 已由 exact-clean-head Tesla P100 产物闭合：CuPy 与 Torch 均通过 26 个 estimator case 和 2 个 direct public covariance primitive，并记录了包含 bounded high-T QS 场景的同步 performance evidence。",
    "Tier-1 Panel 路线的 Stage C 在 Stage-B diagnostics 之上补齐 residual-sandwich covariance 层：历史默认行为保持不变，同时加入 HC0/HC2/HC3、RandomEffects robust inference、显式 cluster group debias 与 Driscoll-Kraay，并保持 NumPy/CuPy/Torch 数值累积后端原生。pre-review exact-clean head `9c0b3050...` 已记录 Tesla P100 correctness 与同步 performance evidence；随后 ordered-categorical chronology 修复改变了 production covariance input handling，因此这些产物现作为历史证据保留，PR #126 回到 Ready 前必须在新的 exact head 上重新完成 P100 验证。",
)
replace_once(
    "docs/cn/models/panel.md",
    "hosted Stage-C tests 已将 HC2/HC3 与 analytic/statsmodels fit-space 计算对齐，并将 cluster/Driscoll-Kraay definition 与 `linearmodels==7.0` 固定版本对齐。exact-clean-head Tesla P100 acceptance 已对 CuPy 与 Torch 闭合：每个 backend 均通过 26/26 estimator covariance case 与 2/2 direct public covariance primitive，并验证 requested/executed backend identity、无 CPU fallback。同步 performance artifact 还覆盖显式 `N=10,000`、`k=2`、`T=200` 的 QS all-lag 场景；其中不编码 speedup 或 CPU-baseline claim。",
    "hosted Stage-C tests 已将 HC2/HC3 与 analytic/statsmodels fit-space 计算对齐，并将 cluster/Driscoll-Kraay definition 与 `linearmodels==7.0` 固定版本对齐。此前 Tesla P100 产物已在每个 backend 上通过 26/26 estimator covariance case 与 2/2 direct public covariance primitive，并包含同步 `N=10,000`、`k=2`、`T=200` QS timing。由于 post-review ordered-categorical chronology 修复修改了 production covariance metadata handling，这些产物继续作为历史证据保留，但不再闭合新的 exact-head physical gate；更新后的 candidate 必须在 CuPy 与 Torch 上重跑后才能恢复 Ready/merge-ready 结论。",
)

print("PR126 post-review documentation state aligned")
