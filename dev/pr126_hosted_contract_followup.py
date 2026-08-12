from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: replacement count={count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


# The old test encoded a raw-column-count df rule. Stage C now explicitly
# supports identified-rank df for rank-deficient BetweenOLS, so split the
# contract into a success case for redundant columns and a true zero-df failure.
path = "dev/tests/test_module_review_covariance_panel.py"
old = '''def test_between_requires_positive_residual_degrees_of_freedom():\n    X = np.arange(12.0).reshape(6, 2)\n    y = np.arange(6.0)\n    entity = np.array([0, 0, 1, 1, 2, 2])\n    with pytest.raises(ValueError, match="degrees of freedom"):\n        BetweenOLS().fit(X, y, entity_ids=entity)\n'''
new = '''def test_between_rank_deficient_df_uses_identified_rank():\n    # Three entity means and three raw columns after the automatic intercept,\n    # but only rank two: the supported rank-deficient extension has one\n    # residual degree of freedom and must not reject merely because groups <= k.\n    X = np.arange(12.0).reshape(6, 2)\n    y = np.arange(6.0)\n    entity = np.array([0, 0, 1, 1, 2, 2])\n    model = BetweenOLS().fit(X, y, entity_ids=entity)\n    assert model.df_resid == 1\n    assert model.fit_statistics_.metadata["diagnostic_rank"] == 2\n\n\ndef test_between_requires_positive_identified_residual_degrees_of_freedom():\n    # Three entity means span intercept + two slopes, so rank == n_groups and\n    # the identified residual degrees of freedom are genuinely zero.\n    X = np.array([\n        [0.0, 0.0], [0.0, 0.0],\n        [1.0, 0.0], [1.0, 0.0],\n        [0.0, 1.0], [0.0, 1.0],\n    ])\n    y = np.arange(6.0)\n    entity = np.array([0, 0, 1, 1, 2, 2])\n    with pytest.raises(ValueError, match="degrees of freedom"):\n        BetweenOLS().fit(X, y, entity_ids=entity)\n'''
replace_once(path, old, new)

# User model pages should describe evidence, not PR lifecycle tokens. Keep the
# exact current gate in dev/reviews and PR metadata instead.
replace_once(
    "docs/en/models/panel.md",
    "Current validation status is **PARTIAL_REMOTE_PENDING**. The `3dc7df19...` Tesla P100 run remains historical evidence only because production numerical behavior changed afterward. Fresh correctness acceptance must run the expanded matrix at the current exact-clean source: CuPy and Torch each require **47/47 = 35 estimator integrations + 12 public primitives**, including eight rank-deficient nonrobust/HC1 estimator cases that record `fit_rank < parameter_count`; the synchronized performance target remains 58/58 rows. Explicit GPU devices continue to forbid silent CPU fallback.",
    "The `3dc7df19...` Tesla P100 run is historical evidence only because production numerical behavior changed afterward. The post-fix implementation has not yet been promoted from fresh physical evidence: its maintained acceptance matrix requires **47/47 = 35 estimator integrations + 12 public primitives** on each of CuPy and Torch, including eight rank-deficient nonrobust/HC1 estimator cases that record `fit_rank < parameter_count`; the synchronized performance matrix remains 58/58 rows. PR-specific gate state is tracked in the repository review records rather than this long-lived model page. Explicit GPU devices continue to forbid silent CPU fallback.",
)
replace_once(
    "docs/cn/models/panel.md",
    "当前验证状态为 **PARTIAL_REMOTE_PENDING**。由于后续 production numerical behavior 已改变，`3dc7df19...` Tesla P100 结果仅保留为历史证据。新的 correctness acceptance 必须在当前 exact-clean source 上执行扩展矩阵：CuPy 与 Torch 各要求 **47/47 = 35 个 estimator integration + 12 个 public primitive**，其中新增 8 个 rank-deficient nonrobust/HC1 estimator case 并记录 `fit_rank < parameter_count`；同步 performance 目标仍为 58/58 行。显式 GPU device 继续禁止静默回退 CPU。",
    "由于后续 production numerical behavior 已改变，`3dc7df19...` Tesla P100 结果仅保留为历史证据。修复后的实现尚未基于新的 physical evidence 完成 promotion：maintained acceptance matrix 要求 CuPy 与 Torch 各通过 **47/47 = 35 个 estimator integration + 12 个 public primitive**，其中包括 8 个记录 `fit_rank < parameter_count` 的 rank-deficient nonrobust/HC1 estimator case；同步 performance matrix 仍为 58/58 行。PR 专属 gate 状态记录在仓库 review 文档中，而不写入长期模型文档。显式 GPU device 继续禁止静默回退 CPU。",
)
