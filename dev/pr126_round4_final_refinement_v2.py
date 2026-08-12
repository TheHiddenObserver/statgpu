from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {count}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Keep one host synchronization per alternating-projection iteration.
replace_once(
    "statgpu/panel/_utils.py",
    '''            y_change = _to_float_scalar(xp.max(xp.abs(y_d - y_d_old)))\n            if getattr(xp, "__name__", "") == "torch":\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), dim=0).values\n            else:\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), axis=0)\n            y_relative_change = float(y_change) / y_scale_ref\n            X_relative_change = _to_float_scalar(\n                xp.max(X_change_columns / X_scale_ref)\n            )\n            max_change = max(float(y_relative_change), float(X_relative_change))\n''',
    '''            y_change = xp.max(xp.abs(y_d - y_d_old))\n            if getattr(xp, "__name__", "") == "torch":\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), dim=0).values\n            else:\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), axis=0)\n            y_relative_change = y_change / y_scale_ref\n            X_relative_change = xp.max(X_change_columns / X_scale_ref)\n            max_change = _to_float_scalar(\n                xp.maximum(y_relative_change, X_relative_change)\n            )\n''',
)

# Physical GPU evidence must independently enforce rank-deficient inference
# unavailability, not merely match the NumPy reference.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''            if name == "panel_entity_hc0" and snapshot["prediction_backend"] != backend:\n                raise AssertionError(\n                    f"{name}: prediction requested {backend}, executed {snapshot['prediction_backend']}"\n                )\n            payload["cases"][name] = {\n''',
    '''            fit_rank = _fit_rank(model)\n            parameter_count = int(snapshot["coef"].size)\n            if fit_rank < parameter_count:\n                if snapshot["coefficient_inference_applicable"]:\n                    raise AssertionError(\n                        f"{name}: rank-deficient fit published coordinate inference"\n                    )\n                reason = snapshot["coefficient_inference_reason"]\n                if not reason or "rank deficient" not in reason:\n                    raise AssertionError(\n                        f"{name}: rank-deficient inference reason is not auditable"\n                    )\n            if name == "panel_entity_hc0" and snapshot["prediction_backend"] != backend:\n                raise AssertionError(\n                    f"{name}: prediction requested {backend}, executed {snapshot['prediction_backend']}"\n                )\n            payload["cases"][name] = {\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''                "fit_rank": _fit_rank(model),\n                "parameter_count": int(snapshot["coef"].size),\n''',
    '''                "fit_rank": fit_rank,\n                "parameter_count": parameter_count,\n''',
)

# English durable statistical contract.
replace_once(
    "docs/en/models/panel.md",
    "Stage C is additive on historical full-column-rank fits: coefficient estimation, Stage-B fit statistics, and the historical default inference remain unchanged. The supported rank-deficient extension uses identified numerical rank so adding redundant columns does not change fit-space degrees of freedom or identified inference. Covariance names are normalized as follows.\n",
    "Stage C is additive on historical full-column-rank fits: coefficient estimation, Stage-B fit statistics, and the historical default inference remain unchanged. The supported rank-deficient extension uses identified numerical rank so adding redundant columns does not change fitted values, residual degrees of freedom, variance components, or other identified fit-space quantities. The reported coefficient vector is the shared Moore-Penrose minimum-norm representation, but the original coordinate coefficients are not uniquely identified; `bse_`, `tvalues_`, `pvalues_`, and `conf_int_` are therefore unavailable and `summary()` fails closed for an exact rank-deficient fit. Covariance names are normalized as follows.\n",
)
replace_once(
    "docs/en/models/panel.md",
    "Historical full-rank Swamy-Arora variance-component and coefficient estimation is unchanged. In the supported rank-deficient extension, the between/within/quasi-demeaned residual degrees of freedom use their identified numerical ranks, making variance components, theta, identified fitted values, and fit-space inference invariant to exactly redundant columns. Robust, HC, cluster, and Driscoll-Kraay covariance are computed from `X_star` and its residuals, so changing `cov_type` still changes inference only. The classical Stage-B Hausman test requires **both** the FE and RE fits to use nonrobust covariance; robust auxiliary Hausman remains out of scope and returns a structured inapplicable result.\n",
    "Historical full-rank Swamy-Arora variance-component and coefficient estimation is unchanged. In the supported rank-deficient extension, the between/within/quasi-demeaned residual degrees of freedom use their identified numerical ranks, making variance components, theta, fitted values, and fit-space covariance invariant to exactly redundant columns. Coordinate-wise BSE/test/p-value/CI output is unavailable because the original coefficient coordinates are not uniquely identified. Robust, HC, cluster, and Driscoll-Kraay covariance are still formed on `X_star` for identified fit-space auditing. The classical Stage-B Hausman test requires **both** the FE and RE fits to use nonrobust covariance; robust auxiliary Hausman remains out of scope and returns a structured inapplicable result.\n",
)
replace_once(
    "docs/en/models/panel.md",
    "Stage-B model-F restrictions use effective numerical rank rather than blindly using the raw column count.\n",
    "Stage-B model-F restrictions use effective numerical rank rather than blindly using the raw column count. The same coefficient-inference-unavailable rule applies to rank-deficient `PanelOLS`, `RandomEffects`, `BetweenOLS`, and `FirstDifferenceOLS` fits.\n\n### FirstDifference time semantics\n\nWhen `time_ids` are supplied, `FirstDifferenceOLS` requires every `(entity_id, time_id)` pair to be unique and rejects duplicates instead of constructing a meaningless within-time difference. Differences are taken between consecutive **observed** times within each entity; internal calendar gaps are allowed and are not implicitly filled or divided by gap length. Ordered categorical time labels preserve their declared chronology.\n",
)

# Chinese durable statistical contract using the current natural-language text.
replace_once(
    "docs/cn/models/panel.md",
    "对于历史 full-column-rank 拟合，Stage C 仍是增量式扩展：系数估计、Stage-B fit statistics 与历史默认推断不改变。对受支持的 rank-deficient extension，则使用 identified numerical rank，使冗余列不会改变 fit-space df 或已识别推断。协方差名称规范如下。\n",
    "对于历史 full-column-rank 拟合，Stage C 仍是增量式扩展：系数估计、Stage-B fit statistics 与历史默认推断不改变。对受支持的 rank-deficient extension，则使用 identified numerical rank，使冗余列不会改变 fitted value、residual df、variance component 或其他 identified fit-space quantity。公开 coefficient vector 是共享的 Moore-Penrose minimum-norm representation，但原始坐标下的 coefficient 不唯一可识别；因此 exact rank-deficient fit 的 `bse_`、`tvalues_`、`pvalues_` 与 `conf_int_` 不可用，`summary()` 也会 fail closed。协方差名称规范如下。\n",
)
replace_once(
    "docs/cn/models/panel.md",
    "历史 full-rank Swamy-Arora variance component 与 coefficient estimate 保持不变。对受支持的 rank-deficient extension，between/within/quasi-demeaned residual df 使用各自 identified numerical rank，从而使 variance component、theta、identified fitted value 与 fit-space inference 对精确冗余列保持不变。robust、HC、cluster 与 Driscoll-Kraay 仍基于 `X_star` 与相应 residual 计算，因此改变 `cov_type` 只改变 inference。Stage-B classical Hausman 要求 **FE 与 RE 两端都使用 nonrobust covariance**；robust auxiliary Hausman 不在 Stage C 范围内，并返回结构化 inapplicable 结果。\n",
    "历史 full-rank Swamy-Arora variance component 与 coefficient estimate 保持不变。对受支持的 rank-deficient extension，between/within/quasi-demeaned residual df 使用各自 identified numerical rank，从而使 variance component、theta、fitted value 与 fit-space covariance 对精确冗余列保持不变。由于原始 coefficient coordinate 不唯一可识别，逐坐标 BSE/test/p-value/CI 不可用；robust、HC、cluster 与 Driscoll-Kraay covariance 仍可在 `X_star` 上形成，用于 identified fit-space 审计。Stage-B classical Hausman 要求 **FE 与 RE 两端都使用 nonrobust covariance**；robust auxiliary Hausman 不在 Stage C 范围内，并返回结构化 inapplicable 结果。\n",
)
replace_once(
    "docs/cn/models/panel.md",
    "由于后续 production numerical behavior 已改变，`3dc7df19...` Tesla P100 结果仅保留为历史证据。修复后的实现尚未基于新的 physical evidence 完成 promotion：maintained acceptance matrix 要求 CuPy 与 Torch 各通过 **47/47 = 35 个 estimator integration + 12 个 public primitive**，其中包括 8 个记录 `fit_rank < parameter_count` 的 rank-deficient nonrobust/HC1 estimator case；同步 performance matrix 仍为 58/58 行。PR 专属 gate 状态记录在仓库 review 文档中，而不写入长期模型文档。显式 GPU device 继续禁止静默回退 CPU。\n",
    "已完成的 `f1546476...` Tesla P100 run 继续作为不可变历史证据保留：CuPy 与 Torch 均完成 47/47 correctness case，同步 performance matrix 为 58/58 行。随后新的 implementation hardening 再次改变 production numerical behavior，因此该历史 measurement 不能描述当前 numerical tree；更新后的实现需要新的 exact-head physical lineage。PR 专属 acceptance 状态只记录在仓库 review 文档中，而不写入长期模型文档。显式 GPU device 继续禁止静默回退 CPU。\n",
)
replace_once(
    "docs/cn/models/panel.md",
    "Stage-B model-F 的 restriction rank 使用有效数值 rank，而不是直接使用原始列数。\n",
    "Stage-B model-F 的 restriction rank 使用有效数值 rank，而不是直接使用原始列数。同样的 coefficient-inference-unavailable 规则适用于 rank-deficient `PanelOLS`、`RandomEffects`、`BetweenOLS` 与 `FirstDifferenceOLS`。\n\n### FirstDifference 时间语义\n\n提供 `time_ids` 时，`FirstDifferenceOLS` 要求每个 `(entity_id, time_id)` pair 唯一；重复 pair 会直接报错，不会构造没有时间含义的同一期差分。差分发生在每个 entity 的连续**已观测**时间点之间；内部 calendar gap 可以存在，不会自动补齐，也不会按 gap 长度缩放。ordered categorical time label 保留其声明的 chronology。\n",
)

print("PR126 round4 final refinement v2 applied")
