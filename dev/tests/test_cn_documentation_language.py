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
)


def _read(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


def test_cleaned_chinese_pages_avoid_known_mixed_language_prose():
    for path in _CLEANED_CN_PAGES:
        text = _read(path)
        for fragment in _PROSE_FRAGMENTS_TO_AVOID:
            assert fragment not in text, f"{path}: prose-level English fragment {fragment!r}"


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
