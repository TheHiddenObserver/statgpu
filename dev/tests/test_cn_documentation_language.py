"""Focused language contracts for cleaned Chinese public documentation."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]

_CLEANED_CN_PAGES = (
    "docs/cn/README.md",
    "docs/cn/usage.md",
    "docs/cn/guides/cross-validation.md",
    "docs/cn/guides/cross-validation-design.md",
    "docs/cn/guides/device-and-memory.md",
    "docs/cn/guides/implemented-methods.md",
    "docs/cn/guides/inference-api.md",
    "docs/cn/guides/inference-modes.md",
    "docs/cn/guides/lbfgs-float32-precision-contract.md",
    "docs/cn/guides/loss-penalty-solver-framework.md",
    "docs/cn/guides/nodewise-alpha-migration.md",
    "docs/cn/guides/penalized-glm-inference.md",
    "docs/cn/guides/penalized-solver-api-migration.md",
    "docs/cn/guides/solver-penalty-matrix.md",
    "docs/cn/guides/solver-algorithms.md",
    "docs/cn/guides/cox-cv-staged-safety.md",
    "docs/cn/guides/pytorch-backend.md",
    "docs/cn/models/README.md",
    "docs/cn/models/losses.md",
    "docs/cn/models/quantile.md",
    "docs/cn/models/ridge.md",
    "docs/cn/models/lasso.md",
    "docs/cn/models/poisson-regression.md",
    "docs/cn/panel/architecture.md",
    "docs/cn/panel/covariance.md",
    "docs/cn/panel/panel-ols.md",
    "docs/cn/panel/pooled-ols.md",
    "docs/cn/panel/between-ols.md",
    "docs/cn/panel/first-difference-ols.md",
    "docs/cn/panel/random-effects.md",
    "docs/cn/panel/fama-macbeth.md",
    "docs/cn/panel/diagnostics.md",
    "docs/cn/panel/fit-statistics.md",
    "docs/cn/benchmarks.md",
    "docs/cn/guides/benchmarks.md",
    "docs/cn/models/coxph.md",
    "docs/cn/models/elastic-net.md",
    "docs/cn/models/scad.md",
    "docs/cn/releases/pr79-final-validation.md",
)

# These are prose-level English noun phrases that previously appeared inside
# otherwise Chinese sentences. Do not broaden this into a ban on individual
# English words: API identifiers and established algorithm names must remain
# searchable and recognizable.
_PROSE_FRAGMENTS_TO_AVOID = (
    "public estimator",
    "tuning grid",
    "final refit",
    "selection work",
    "selection cache",
    "selection evidence",
    "fitted model",
    "fitted state",
    "model-specific",
    "backend availability",
    "performance heuristic",
    "loss-level",
    "solver provenance",
    "Capability matching",
    "meta-estimator",
    "coefficient inference method",
    "inference target",
    "active set",
    "residual-bootstrap",
    "candidate tuning values",
    "held-out scoring",
    "warm start storage",
    "batching layout",
    "Validation Matrix",
    "Physical GPU",
    "exact head",
    "fail closed",
    "## Overview",
    "## Path",
    "## Parameters",
    "## Outputs",
    "## External Validation",
    "Source provenance",
    "parse report",
    "source inventory",
    "Runtime provenance",
    "provenance payload",
    "clean exact-head",
    "exact-head artifact",
    "exact-head CI",
    "前端 contract",
    "inference-only",
    "penalized coefficients",
    "active-refit",
    "active-set identity",
    "inference backend/device",
    "machine schema",
    "Validation tier",
    "Gate failures",
    "Campaign 文件名",
    "unsupported",
    "line search",
    "right-censored",
    "delayed entry",
    "child refit",
    "score residual",
    "log-link evaluator",
    "row-streaming",
    "failure-group",
    "mean-one",
    "effectively-uniform",
    "domain-capped",
    "compatibility-only alias",
    "estimation-only",
    "covariance estimator",
    "working problem",
    "inactive-coordinate placeholder",
    "attempt metadata",
    "estimator state",
    "failure transaction",
    "installer idempotence",
)

_QUANTILE_PROSE_FRAGMENTS_TO_AVOID = (
    "typed `PenalizedQuantileRegression.score()`",
    "generic `PenalizedGeneralizedLinearModel",
    "shared scalar-response",
    "strict Quantile 交叉验证",
    "target-level",
    "failure signal",
    "two-stage 的第一阶段",
    " relaxed",
    "strict refinement",
    "strict selection",
    "Quantile 非凸 continuation 路径",
    "alpha step",
    "continuation step",
    "目标 step",
    "Quantile solver 调用",
    "IRLS/continuation",
    "broadcasting",
    "直接 continuation 调用",
    "普通 estimator/CV",
    "import 兼容符号",
    "adaptive penalty weights",
    "固定 adaptive weights",
    "batched Quantile",
    "kernel/bootstrap 推断",
)

_PR166_CHANGELOG_FRAGMENTS_TO_AVOID = (
    "direct、CV 与底层公开 consumer",
    "response validation",
    "backend work",
    "child refit",
    "bootstrap child objective",
    "standalone 推断",
    "runtime help",
    "host sync",
    "focused regressions",
    "regression coverage",
    "generic FISTA diagnostic",
    "selected full-data refit",
)

_PR164_CHANGELOG_FRAGMENTS_TO_AVOID = (
    "solver identity",
    "FISTA family",
    "resolved/executed provenance",
    "solver= keyword",
    "numerical dispatch",
    "CV grid work",
    "fail closed",
    "direct fit、CV candidate/fold",
    "selected full-data refit",
    "requested/resolved/executed",
    "quantile level",
    "analytic validation weights",
    "clone-safe construction",
    "adaptive-L1 initialization",
    "pinball objective",
    "LLA surrogate",
    "diagonal approximation",
    "Torch Quantile execution",
    "penalty diagonal",
    "warm start",
    "fallback weights",
    "Torch-native clone",
    "numerical source",
    "physical gate",
    "case 全部通过",
    "tolerance 没有放宽",
    "coefficient/intercept",
    "CV-score",
    "penalized-objective",
    "canonical exact-source artifact",
    "documentation-only",
    "immutable numerical-source acceptance",
    "physical rerun",
)

_UNRELEASED_CHANGELOG_FRAGMENTS_TO_AVOID = (
    "fail closed",
    "physical CUDA acceptance",
    "physical CUDA gate",
    "physical CUDA validator",
    "physical acceptance",
    "physical validation",
    "physical gate",
    "hosted coverage",
    "hosted validation",
    "hosted checks",
    "hosted gates",
    "exact-source",
    "exact SHA",
    "exact clean-head",
    "canonical dispatch",
    "canonical artifact",
    "consumer",
    "tolerance",
    "child inference",
    "child optimization",
    "child context",
    "warm start",
    "log-link",
    "fixture",
    "resampling",
    "resample",
    "snapshot",
    "unsupported rows",
    "reconciliation boundary",
    "compatibility boundary",
    "ownership",
    "placeholder",
    "unfitted",
    "weighted-center",
    "row transform",
    "provenance publication",
    "result provenance",
    "closed-form",
    "checklist",
)

_PR166_SOLVER_PROSE_FRAGMENTS_TO_AVOID = (
    "smooth-gradient FISTA",
    "estimator/CV",
    "smooth-gradient difference",
    "non Cholesky fallback",
    "family/backend/problem-size",
    "estimator 算法",
)


def _read(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


def test_cleaned_chinese_pages_avoid_known_mixed_language_prose():
    for path in _CLEANED_CN_PAGES:
        text = _read(path)
        for fragment in _PROSE_FRAGMENTS_TO_AVOID:
            assert fragment not in text, f"{path}: prose-level English fragment {fragment!r}"


def test_quantile_page_uses_chinese_explanatory_prose():
    text = _read("docs/cn/models/quantile.md")
    for fragment in _QUANTILE_PROSE_FRAGMENTS_TO_AVOID:
        assert fragment not in text, (
            "docs/cn/models/quantile.md: explanatory prose fragment "
            f"{fragment!r}"
        )


def test_pr166_changelog_section_uses_chinese_explanatory_prose():
    text = _read("docs/cn/changelog.md")
    start = text.index("## 未发布 — Quantile 求解器与推断更新（PR #166")
    end = text.index("## 未发布 — Quantile 求解器来源对齐（PR #164", start)
    section = text[start:end]
    for fragment in _PR166_CHANGELOG_FRAGMENTS_TO_AVOID:
        assert fragment not in section, (
            "docs/cn/changelog.md PR166 section: explanatory prose fragment "
            f"{fragment!r}"
        )


def test_pr164_changelog_section_uses_chinese_explanatory_prose():
    text = _read("docs/cn/changelog.md")
    start = text.index("## 未发布 — Quantile 求解器来源对齐（PR #164")
    end = text.index("## 未发布 — Quantile IRLS 惩罚契约修复（PR #162", start)
    section = text[start:end]
    for fragment in _PR164_CHANGELOG_FRAGMENTS_TO_AVOID:
        assert fragment not in section, (
            "docs/cn/changelog.md PR164 section: explanatory prose fragment "
            f"{fragment!r}"
        )


def test_remaining_unreleased_changelog_sections_use_chinese_explanatory_prose():
    text = _read("docs/cn/changelog.md")
    start = text.index("## 未发布 — Quantile IRLS 惩罚契约修复（PR #162")
    end = text.index("## 0.2.5 — 2026-08-26（已发布）", start)
    section = text[start:end]
    for fragment in _UNRELEASED_CHANGELOG_FRAGMENTS_TO_AVOID:
        assert fragment not in section, (
            "docs/cn/changelog.md unreleased PR162-PR129 sections: "
            f"explanatory prose fragment {fragment!r}"
        )


def test_pr166_solver_algorithm_prose_uses_chinese_explanatory_vocabulary():
    text = _read("docs/cn/guides/solver-algorithms.md")
    for fragment in _PR166_SOLVER_PROSE_FRAGMENTS_TO_AVOID:
        assert fragment not in text, (
            "docs/cn/guides/solver-algorithms.md: explanatory prose fragment "
            f"{fragment!r}"
        )


def test_chinese_pages_keep_real_api_identifiers_searchable():
    cv = _read("docs/cn/guides/cross-validation.md")
    inference = _read("docs/cn/guides/inference-modes.md")
    quantile = _read("docs/cn/models/quantile.md")
    torch_guide = _read("docs/cn/guides/pytorch-backend.md")
    covariance = _read("docs/cn/panel/covariance.md")

    # Real API identifiers may appear with arguments or fitted-state suffixes;
    # the contract is that the identifier remains searchable, not that one
    # exact Markdown rendering is required.
    for token in ("LassoCV", "cv_solver", 'device="auto"', "sample_weight"):
        assert token in cv

    for token in ("debiased", "post_selection_ols", "bootstrap"):
        assert token in inference

    for token in ("FISTA", "IRLS", "SCAD", "MCP", 'solver="auto"'):
        assert token in quantile

    for token in ('device="torch"', "PyTorch", "Torch CUDA"):
        assert token in torch_guide

    for token in ("HC0", "HC3", "Driscoll", "cov_type"):
        assert token in covariance


def test_documentation_policy_states_chinese_language_rule():
    style = _read("dev/DOCUMENTATION_STYLE.md")

    assert "### Chinese prose consistency" in style
    assert "API identifiers" in style
    assert "Chinese syntax and Chinese explanatory vocabulary" in style
    assert "Do not pursue artificial “zero English.”" in style
