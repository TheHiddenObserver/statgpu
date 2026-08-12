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


# Preserve the historical single scalar synchronization per alternating-
# projection iteration.  Both relative changes remain backend scalars until the
# final maximum is copied to host once for the convergence branch.
replace_once(
    "statgpu/panel/_utils.py",
    '''            y_change = _to_float_scalar(xp.max(xp.abs(y_d - y_d_old)))\n            if getattr(xp, "__name__", "") == "torch":\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), dim=0).values\n            else:\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), axis=0)\n            y_relative_change = float(y_change) / y_scale_ref\n            X_relative_change = _to_float_scalar(\n                xp.max(X_change_columns / X_scale_ref)\n            )\n            max_change = max(float(y_relative_change), float(X_relative_change))\n''',
    '''            y_change = xp.max(xp.abs(y_d - y_d_old))\n            if getattr(xp, "__name__", "") == "torch":\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), dim=0).values\n            else:\n                X_change_columns = xp.max(xp.abs(X_d - X_d_old), axis=0)\n            y_relative_change = y_change / y_scale_ref\n            X_relative_change = xp.max(X_change_columns / X_scale_ref)\n            max_change = _to_float_scalar(\n                xp.maximum(y_relative_change, X_relative_change)\n            )\n''',
)

# The physical acceptance runner must independently fail closed on the new
# rank-deficient inference contract instead of relying only on NumPy/GPU parity.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''            if name == "panel_entity_hc0" and snapshot["prediction_backend"] != backend:\n                raise AssertionError(\n                    f"{name}: prediction requested {backend}, executed {snapshot['prediction_backend']}"\n                )\n            payload["cases"][name] = {\n''',
    '''            fit_rank = _fit_rank(model)\n            parameter_count = int(snapshot["coef"].size)\n            if fit_rank < parameter_count:\n                if snapshot["coefficient_inference_applicable"]:\n                    raise AssertionError(\n                        f"{name}: rank-deficient fit published coordinate inference"\n                    )\n                if not snapshot["coefficient_inference_reason"] or (\n                    "rank deficient" not in snapshot["coefficient_inference_reason"]\n                ):\n                    raise AssertionError(\n                        f"{name}: rank-deficient inference reason is not auditable"\n                    )\n            if name == "panel_entity_hc0" and snapshot["prediction_backend"] != backend:\n                raise AssertionError(\n                    f"{name}: prediction requested {backend}, executed {snapshot['prediction_backend']}"\n                )\n            payload["cases"][name] = {\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''                "fit_rank": _fit_rank(model),\n                "parameter_count": int(snapshot["coef"].size),\n''',
    '''                "fit_rank": fit_rank,\n                "parameter_count": parameter_count,\n''',
)

# Durable user docs: distinguish identified fit-space quantities from
# unavailable coordinate-wise inference and state FirstDifference time semantics.
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

replace_once(
    "docs/cn/models/panel.md",
    "Stage C 在历史 full-column-rank fit 上保持 additive：coefficient estimation、Stage-B fit statistics 与历史 default inference 不变。受支持的 rank-deficient extension 使用 identified numerical rank，因此加入冗余列不会改变 fit-space degrees of freedom 或 identified inference。covariance 名称规范化如下。\n",
    "Stage C 在历史 full-column-rank fit 上保持 additive：coefficient estimation、Stage-B fit statistics 与历史 default inference 不变。受支持的 rank-deficient extension 使用 identified numerical rank，因此加入冗余列不会改变 fitted value、residual degrees of freedom、variance component 或其他 identified fit-space quantity。公开的 coefficient vector 是共享 Moore-Penrose minimum-norm representation，但原始坐标下的 coefficient 并不唯一可识别；因此 exact rank-deficient fit 的 `bse_`、`tvalues_`、`pvalues_` 与 `conf_int_` 不可用，`summary()` 也会 fail closed。covariance 名称规范化如下。\n",
)
# Chinese file wording can evolve independently; append a durable semantic note
# after its rank-deficient PooledOLS section anchor if present.
p = ROOT / "docs/cn/models/panel.md"
text = p.read_text(encoding="utf-8")
anchor = "Stage-B model-F restriction 使用 effective numerical rank，而不会直接使用原始 column count。\n"
if anchor in text:
    text = text.replace(
        anchor,
        anchor
        + "同样的 coefficient-inference-unavailable 规则也适用于 rank-deficient `PanelOLS`、`RandomEffects`、`BetweenOLS` 与 `FirstDifferenceOLS`。\n\n"
        + "### FirstDifference 时间语义\n\n"
        + "提供 `time_ids` 时，`FirstDifferenceOLS` 要求每个 `(entity_id, time_id)` pair 唯一；重复 pair 会直接报错，不会构造没有时间含义的同一期差分。差分发生在每个 entity 的连续**已观测**时间点之间；内部 calendar gap 可以存在，不会自动补齐，也不会按 gap 长度缩放。ordered categorical time label 保留其声明的 chronology。\n",
        1,
    )
else:
    # Fail closed rather than silently skipping a bilingual documentation gate.
    raise RuntimeError("docs/cn/models/panel.md: rank-deficient anchor drifted")
p.write_text(text, encoding="utf-8")

print("PR126 round4 final refinements applied")
