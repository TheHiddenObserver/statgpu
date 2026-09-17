"""Runtime-help and public-documentation placement contracts."""

import inspect
from pathlib import Path

from statgpu import glm_core, solvers
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)


_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_DOC_PAIRS = (
    (
        "docs/en/README.md",
        "docs/cn/README.md",
    ),
    (
        "docs/en/guides/solver-algorithms.md",
        "docs/cn/guides/solver-algorithms.md",
    ),
    (
        "docs/en/guides/loss-penalty-solver-framework.md",
        "docs/cn/guides/loss-penalty-solver-framework.md",
    ),
    (
        "docs/en/guides/solver-penalty-matrix.md",
        "docs/cn/guides/solver-penalty-matrix.md",
    ),
    (
        "docs/en/models/losses.md",
        "docs/cn/models/losses.md",
    ),
    (
        "docs/en/models/quantile.md",
        "docs/cn/models/quantile.md",
    ),
    (
        "docs/en/guides/cross-validation.md",
        "docs/cn/guides/cross-validation.md",
    ),
    (
        "docs/en/guides/inference-api.md",
        "docs/cn/guides/inference-api.md",
    ),
    (
        "docs/en/guides/inference-modes.md",
        "docs/cn/guides/inference-modes.md",
    ),
    (
        "docs/en/guides/lbfgs-float32-precision-contract.md",
        "docs/cn/guides/lbfgs-float32-precision-contract.md",
    ),
    (
        "docs/en/guides/device-and-memory.md",
        "docs/cn/guides/device-and-memory.md",
    ),
    (
        "docs/en/guides/implemented-methods.md",
        "docs/cn/guides/implemented-methods.md",
    ),
    (
        "docs/en/guides/cox-cv-staged-safety.md",
        "docs/cn/guides/cox-cv-staged-safety.md",
    ),
    (
        "docs/en/guides/nodewise-alpha-migration.md",
        "docs/cn/guides/nodewise-alpha-migration.md",
    ),
    (
        "docs/en/guides/penalized-glm-inference.md",
        "docs/cn/guides/penalized-glm-inference.md",
    ),
    (
        "docs/en/guides/penalized-solver-api-migration.md",
        "docs/cn/guides/penalized-solver-api-migration.md",
    ),
    (
        "docs/en/panel/architecture.md",
        "docs/cn/panel/architecture.md",
    ),
    (
        "docs/en/usage.md",
        "docs/cn/usage.md",
    ),
)


def test_guarded_public_solver_docstrings_expose_quantile_boundary():
    fista_doc = inspect.getdoc(solvers.fista_bb_solver) or ""
    admm_doc = inspect.getdoc(solvers.admm_solver) or ""
    fista_text = " ".join(fista_doc.split())
    admm_text = " ".join(admm_doc.split())

    assert "Quantile" in fista_text
    assert "does not support Quantile" in fista_text
    assert "alternating BB1/BB2 steps" in fista_text
    assert "Supports numpy / cupy / torch backends" in fista_text

    assert "Nesterov-accelerated gradient descent" in admm_text
    assert "does not support Quantile" in admm_text
    assert "cg_max_iter" in admm_doc

    # glm_core re-exports the same guarded public callables, so runtime help
    # must remain identical across both public import paths.
    assert inspect.getdoc(glm_core.fista_bb_solver) == fista_doc
    assert inspect.getdoc(glm_core.admm_solver) == admm_doc

    assert "maintained" not in fista_text.lower()
    assert "maintained" not in admm_text.lower()
    assert "fail closed" not in fista_text.lower()
    assert "fail closed" not in admm_text.lower()


def test_penalized_glm_runtime_help_lists_public_admm_solver():
    doc = inspect.getdoc(PenalizedGeneralizedLinearModel) or ""
    assert "'admm'" in doc
    assert "Support depends on the loss and penalty" in doc
    assert "unsupported explicit combinations raise an error" in doc


def test_typed_quantile_runtime_help_names_ordinary_fista_boundary():
    doc = " ".join((inspect.getdoc(PenalizedQuantileRegression) or "").split())
    assert "L1/ElasticNet objectives use ordinary FISTA" in doc
    assert "explicit ordinary ``solver='fista'`` is also supported" in doc
    assert "executes the generic FISTA engine rather than being silently substituted by IRLS" in doc
    assert "IRLS remains the default automatic choice" in doc
    assert "FISTA-BB and shared ADMM do not support Quantile" in doc
    assert "shared L-BFGS implementation assumes a smooth loss gradient" in doc
    assert "maintained" not in doc.lower()
    assert "fail closed" not in doc.lower()


def test_typed_quantile_runtime_help_documents_loss_kwargs_precedence():
    doc = " ".join((inspect.getdoc(PenalizedQuantileRegression) or "").split())
    assert "loss_kwargs={'quantile': q}" in doc
    assert "takes precedence over the typed ``quantile=`` argument" in doc
    assert "public ``quantile`` attribute retains the outer constructor value" in doc


def test_repository_documentation_language_policy_is_canonicalized():
    style = (_ROOT / "dev/DOCUMENTATION_STYLE.md").read_text(encoding="utf-8")
    contributing = (_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    claude = (_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    assert "Internal documentation may describe capability/status" in style
    assert (
        "user-facing documentation should describe observable behavior/support"
        in style
    )
    # The canonical policy itself intentionally names the internal terms it
    # keeps out of ordinary user documentation.
    assert "maintained route" in style
    assert "fail closed" in style

    for entrypoint in (contributing, claude):
        assert "dev/DOCUMENTATION_STYLE.md" in entrypoint

    assert "User-facing docs describe observable support/behavior" in contributing


def test_documentation_root_portal_stays_user_facing():
    portal = (_ROOT / "docs/index.md").read_text(encoding="utf-8")

    assert "Choose a documentation entry point" in portal
    assert "maintained documentation entry point" not in portal.lower()
    assert "pull-request validation records" not in portal.lower()
    assert ".claude/skills/" not in portal


def test_loss_reference_keeps_quantile_content_at_loss_layer():
    en = (_ROOT / "docs/en/models/losses.md").read_text(encoding="utf-8")
    cn = (_ROOT / "docs/cn/models/losses.md").read_text(encoding="utf-8")

    # The loss reference should retain the mathematical/numerical properties
    # of QuantileLoss, while estimator routing belongs on quantile.md and the
    # shared solver references.
    assert "QuantileLoss" in en
    assert "non-smooth step-function subgradient and no Hessian" in en
    assert "[Quantile Regression](quantile.md)" in en
    assert "`QuantileLoss`" in cn
    assert "次梯度为阶梯函数，并且没有 Hessian" in cn
    assert "[分位数回归](quantile.md)" in cn

    estimator_solver_terms = (
        "PenalizedQuantileRegression",
        'solver="auto"',
        "FISTA-BB",
        "Proximal IRLS-CD",
        "L-BFGS",
        "ADMM",
    )
    for term in estimator_solver_terms:
        assert term not in en
        assert term not in cn


def test_solver_penalty_matrix_stays_a_cross_model_reference():
    en = (_ROOT / "docs/en/guides/solver-penalty-matrix.md").read_text(
        encoding="utf-8"
    )
    cn = (_ROOT / "docs/cn/guides/solver-penalty-matrix.md").read_text(
        encoding="utf-8"
    )

    # The page should keep the global direct/CV matrices and delegate model-
    # specific derivations and implementation narratives to model pages.
    for text in (en, cn):
        assert "**squared_error**" in text
        assert "**logistic**" in text
        assert "**poisson**" in text
        assert "**gamma**" in text
        assert "**inverse_gaussian**" in text
        assert "**negative_binomial**" in text
        assert "**tweedie**" in text
        assert "**quantile**" in text

    assert "Inverse-power Gamma smooth-domain contract" not in en
    assert "For Quantile CV" not in en
    assert "Because check loss is non-smooth" not in en
    assert "Poisson GPU L1 FISTA-BB rule is size-gated" not in en
    assert "direct low-level Quantile L-BFGS" not in en

    assert "`inverse_power` Gamma 的光滑定义域约定" not in cn
    assert "对 Quantile CV" not in cn
    assert "由于 check loss 非光滑" not in cn
    assert "Poisson GPU L1 FISTA-BB 规则" not in cn
    assert "底层直接 Quantile L-BFGS" not in cn


def test_loss_penalty_solver_framework_stays_at_composition_layer():
    en = (_ROOT / "docs/en/guides/loss-penalty-solver-framework.md").read_text(
        encoding="utf-8"
    )
    cn = (_ROOT / "docs/cn/guides/loss-penalty-solver-framework.md").read_text(
        encoding="utf-8"
    )

    # The framework owns the architecture and composition contracts.
    for heading in (
        "## 1. Runtime architecture",
        "## 2. Component responsibilities",
        "## 3. Contract composition",
        "## 4. Capability matching and solver dispatch",
        "## 5. `sample_weight` and objective consistency",
        "## 6. Backend and device boundary",
        "## 7. CV and meta-estimator boundary",
        "## 9. Documentation ownership",
    ):
        assert heading in en

    for heading in (
        "## 1. 运行架构",
        "## 2. 各组件的职责",
        "## 3. 契约如何组合",
        "## 4. Capability matching 与 solver 分发",
        "## 5. `sample_weight` 与目标函数一致性",
        "## 6. Backend 与 device 边界",
        "## 7. CV 与 meta-estimator 边界",
        "## 9. 文档职责分工",
    ):
        assert heading in cn

    # Detailed formulas, model-specific route narratives, full dispatch tables,
    # and unrelated backend examples belong to their canonical pages.
    for old_detail in (
        "### All Implemented Losses",
        "### Per-Sample Formulas",
        "### SCAD Formula",
        "### Specialized Solvers",
        "## 4. Backend Coverage",
        "| Priority | Solver | Condition |",
        "PenalizedQuantileRegression(",
        "DBSCAN",
    ):
        assert old_detail not in en

    for old_detail in (
        "### 已实现的全部损失函数",
        "### 逐样本公式",
        "### SCAD 公式",
        "### 专用求解器",
        "## 4. 后端覆盖",
        "| 优先级 | 求解器 | 条件 |",
        "PenalizedQuantileRegression(",
        "DBSCAN",
    ):
        assert old_detail not in cn

    assert "[Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md)" in en
    assert "[求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)" in cn


def test_solver_algorithm_pages_preserve_explicit_quantile_fista_contract():
    en = (_ROOT / "docs/en/guides/solver-algorithms.md").read_text(encoding="utf-8")
    cn = (_ROOT / "docs/cn/guides/solver-algorithms.md").read_text(encoding="utf-8")

    assert "explicit `solver=\"fista\"` executes ordinary FISTA" in en
    assert "not a claim that textbook smooth-gradient FISTA convergence theory applies" in en
    assert "only an explicit `solver=\"fista\"` request selects this ordinary-FISTA path" in en
    assert "an explicit `solver=\"fista\"` request remains authoritative for both CV child fits" in en
    assert "explicit `solver=\"fista\"` fails rather than being silently substituted by IRLS" not in en

    assert "显式 `solver=\"fista\"` 则真正执行普通 FISTA" in cn
    assert "并不声称 pinball loss 满足教科书式 smooth-gradient FISTA 的收敛假设" in cn
    assert "只有显式 `solver=\"fista\"` 才选择这条普通 FISTA 路径" in cn
    assert "该请求对 CV 子拟合和最终全数据重拟合都保持有效并执行普通 FISTA" in cn
    assert "显式 `solver=\"fista\"` 会失败而不是被静默替换成 IRLS" not in cn


def test_cross_validation_guide_stays_at_user_selection_layer():
    en = (_ROOT / "docs/en/guides/cross-validation.md").read_text(encoding="utf-8")
    cn = (_ROOT / "docs/cn/guides/cross-validation.md").read_text(encoding="utf-8")

    for text in (en, cn):
        assert "candidate" in text
        assert "final refit" in text
        assert "cv_solver" in text
        assert "Part II: Architecture and Implementation" not in text
        assert "_effective_cv_device" not in text
        assert "_compute_cv_scores" not in text
        assert "dev/tests/" not in text
        assert "dev/benchmarks/" not in text
        assert "PR #" not in text

    internal = (_ROOT / "dev/design/CROSS_VALIDATION.md").read_text(encoding="utf-8")
    assert "Scoring-path families" in internal
    assert "LassoCV selection cache" in internal
    assert "Device heuristics" in internal


def test_usage_portals_do_not_embed_contributor_or_validation_workflows():
    en = (_ROOT / "docs/en/usage.md").read_text(encoding="utf-8")
    cn = (_ROOT / "docs/cn/usage.md").read_text(encoding="utf-8")

    for text in (en, cn):
        assert "Cross-Validation" in text or "交叉验证" in text
        assert "Contributor Checklist" not in text
        assert "Validation and Evidence" not in text
        assert "dev/AGENTS.md" not in text
        assert ".claude/skills/" not in text
        assert "physical GPU" not in text.lower()


def test_inference_guides_do_not_expose_internal_execution_plumbing():
    en = (_ROOT / "docs/en/guides/inference-modes.md").read_text(encoding="utf-8")
    cn = (_ROOT / "docs/cn/guides/inference-modes.md").read_text(encoding="utf-8")

    for text in (en, cn):
        assert "post_selection_ols" in text
        assert "debiased" in text
        assert "bootstrap" in text
        assert "_selected_backend_name" not in text
        assert "simultaneous_reporting_boundary" not in text
        assert "cache provenance" not in text.lower()


def test_float32_lbfgs_page_is_guidance_not_validation_evidence():
    en = (_ROOT / "docs/en/guides/lbfgs-float32-precision-contract.md").read_text(
        encoding="utf-8"
    )
    cn = (_ROOT / "docs/cn/guides/lbfgs-float32-precision-contract.md").read_text(
        encoding="utf-8"
    )

    for text in (en, cn):
        assert "float64" in text
        assert "objective" in text
        assert "Issue #" not in text
        assert "acceptance bound" not in text.lower()
        assert "验收边界" not in text
        assert "dev/reviews/" not in text


def test_secondary_public_guides_keep_their_declared_layer():
    device_en = (_ROOT / "docs/en/guides/device-and-memory.md").read_text(
        encoding="utf-8"
    )
    methods_en = (_ROOT / "docs/en/guides/implemented-methods.md").read_text(
        encoding="utf-8"
    )
    panel_en = (_ROOT / "docs/en/panel/architecture.md").read_text(encoding="utf-8")
    nodewise_en = (_ROOT / "docs/en/guides/nodewise-alpha-migration.md").read_text(
        encoding="utf-8"
    )
    inference_en = (_ROOT / "docs/en/guides/penalized-glm-inference.md").read_text(
        encoding="utf-8"
    )

    assert "| Solver | NumPy | CuPy | Torch |" not in device_en
    assert "dev/benchmarks/" not in device_en
    assert "## Validation Scope" not in methods_en
    assert "## 5. Fixed Effects Example" not in panel_en
    assert "_linalg.py" not in panel_en
    assert "1e-8" not in nodewise_en
    assert "3000" not in nodewise_en
    assert "cache provenance" not in nodewise_en.lower()
    assert "Status: targeted" not in inference_en


def test_public_solver_docs_do_not_expose_internal_review_vocabulary():
    for en_path, cn_path in _PUBLIC_DOC_PAIRS:
        en = (_ROOT / en_path).read_text(encoding="utf-8")
        cn = (_ROOT / cn_path).read_text(encoding="utf-8")

        assert "maintained" not in en.lower(), en_path
        assert "fail closed" not in en.lower(), en_path
        assert "fail-closed" not in en.lower(), en_path
        assert "维护" not in cn, cn_path
        assert "fail closed" not in cn.lower(), cn_path
        assert "fail-closed" not in cn.lower(), cn_path
