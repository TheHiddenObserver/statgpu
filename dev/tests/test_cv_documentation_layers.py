"""Bounded CV content/placement guards; passing is not a semantic audit."""

from pathlib import Path
import re

import pytest

from dev.validation.check_docs_contracts import validate_cv_audience


_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


def test_cv_public_guide_links_public_design_layer():
    en = _read("docs/en/guides/cross-validation.md")
    cn = _read("docs/cn/guides/cross-validation.md")

    for language, text in (("en", en), ("cn", cn)):
        assert re.search(r"\]\(cross-validation-design\.md(?:#[^)]*)?\)", text)
        assert not validate_cv_audience(_ROOT / f"docs/{language}/guides/cross-validation.md", text)

    assert "cv_solver" in en
    assert "final refit" in en
    assert "cv_solver" in cn
    assert "最终重拟合" in cn

    for text in (en, cn):
        assert "_effective_cv_device" not in text
        assert "_array_identity_token" not in text
        assert "_make_lasso_cv_auto_cache_key" not in text
        assert "STATGPU_LASSO_CV_CACHE_SIZE" not in text


def _check_design_concepts(text, language):
    # Editorial heading text is not an API. Check body concepts and navigation,
    # permitting reflow and alternate headings without accepting empty sections.
    body = re.sub(r"^#{1,6} .*$", "", text, flags=re.MULTILINE)
    body = " ".join(body.split()).lower()
    concepts = (
        (r"selection", r"final refit", r"candidate", r"validation criterion",
         r"warm.start", r"batching across folds", r"selection cache",
         r"sample_weight", r"device=\"auto\"", r"risk.set", r"p.values")
        if language == "en" else
        (r"选择", r"最终重拟合", r"候选", r"验证准则", r"热启动", r"批量处理",
         r"选择缓存", r"sample_weight", r"device=\"auto\"", r"风险集", r"p 值")
    )
    for concept in concepts:
        assert re.search(concept, body), (language, concept)
    for target in ("cross-validation.md", "inference-modes.md", "../models/coxph.md"):
        assert re.search(r"\]\(" + re.escape(target) + r"(?:#[^)]*)?\)", text), target


@pytest.mark.parametrize("language", ("en", "cn"))
def test_cv_public_design_explains_stable_execution_model(language):
    _check_design_concepts(_read(f"docs/{language}/guides/cross-validation-design.md"), language)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_design_guard_accepts_heading_edits_but_rejects_missing_concepts(language):
    text = _read(f"docs/{language}/guides/cross-validation-design.md")
    renamed = re.sub(r"^## .*", "## An editorial heading", text, flags=re.MULTILINE)
    _check_design_concepts(renamed, language)
    _check_design_concepts(re.sub(r"(?<!\n)\n(?!\n)", " ", text), language)
    # Remove a real concept, not merely its heading or a particular sentence.
    without_cache = re.sub(r"selection cache|选择缓存", "[removed]", text, flags=re.IGNORECASE)
    with pytest.raises(AssertionError):
        _check_design_concepts(without_cache, language)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ("cross-validation", "cross-validation-design"))
def test_cv_audience_guard_rejects_workflows_and_allows_user_contracts(language, page):
    path = _ROOT / f"docs/{language}/guides/{page}.md"
    text = path.read_text(encoding="utf-8")
    assert not validate_cv_audience(path, text)
    instruction = (
        "Maintainers should update internal test fixtures after every implementation "
        "change before approving this module." if language == "en" else
        "维护者必须在每次实现变更后更新内部测试夹具，并完成审批。"
    )
    assert validate_cv_audience(path, text + "\n\n" + instruction)
    assert validate_cv_audience(path, text + "\n\nUse `_compute_cv_scores` directly.")
    # Legitimate diagnostic/API vocabulary and mathematical boundaries remain
    # acceptable. Code and examples are not maintainer prose.
    allowed = "\n\nInspect `cv_results_`, `fold_valid` and `failure_path`; invalid folds cannot provide a validation score."
    assert not validate_cv_audience(path, text + allowed)
    assert not validate_cv_audience(path, text + "\n\n```text\n" + instruction + "\n```")


def test_cv_public_design_does_not_publish_private_implementation_contracts():
    en = _read("docs/en/guides/cross-validation-design.md")
    cn = _read("docs/cn/guides/cross-validation-design.md")

    private_tokens = (
        "_effective_cv_device",
        "_array_identity_token",
        "_make_lasso_cv_auto_cache_key",
        "STATGPU_LASSO_CV_CACHE_SIZE",
        "dev/tests/",
        "dev/benchmarks/",
        "dev/reviews/",
        "PR #",
        "Issue #",
    )
    for text in (en, cn):
        for token in private_tokens:
            assert token not in text

    assert "maintained route" not in en.lower()
    assert "fail closed" not in en.lower()
    assert "维护中的路径" not in cn
    assert "维护路径" not in cn
    assert "fail closed" not in cn.lower()


def test_cv_internal_design_keeps_private_contracts_and_names_public_layers():
    internal = _read("dev/design/CROSS_VALIDATION.md")

    for fragment in (
        "docs/en/guides/cross-validation.md",
        "docs/en/guides/cross-validation-design.md",
        "_effective_cv_device()",
        "_array_identity_token",
        "_make_lasso_cv_auto_cache_key",
        "gpu_cv_mixed_precision",
        "STATGPU_LASSO_CV_CACHE_SIZE",
        "64",
        "private call graphs",
        "heuristic thresholds",
    ):
        assert fragment in internal


def test_cv_design_is_discoverable_from_user_portals():
    en_readme = _read("docs/en/README.md")
    cn_readme = _read("docs/cn/README.md")
    en_usage = _read("docs/en/usage.md")
    cn_usage = _read("docs/cn/usage.md")

    assert "guides/cross-validation-design.md" in en_readme
    assert "guides/cross-validation-design.md" in cn_readme
    assert "guides/cross-validation-design.md" in en_usage
    assert "guides/cross-validation-design.md" in cn_usage


def test_documentation_policy_defines_public_design_layer():
    style = _read("dev/DOCUMENTATION_STYLE.md")

    assert "## Public design / architecture pages" in style
    assert "stable execution model" in style
    assert "conceptual acceleration strategies" in style
    assert "Exact helper names" in style
    assert "heuristic thresholds" in style
