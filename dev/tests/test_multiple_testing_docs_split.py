"""Protect the multiple-testing overview, three lessons, and complete API.

Execution and independent numerical checks cover every named CPU example.
Targeted prose/ordering guards preserve the teaching structure; they are not
statistical validation, and these tests do not claim physical GPU evidence.
"""

import ast
import inspect
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
from doc_examples import example_code, parse_examples, run_example
from doc_paths import markdown_anchors, model_doc, peer_link, reachable_pages
from scipy import stats

from statgpu.inference import adjust_pvalues, combine_pvalues, multipletests

ROOT = Path(__file__).resolve().parents[2]
LESSONS = {
    "multiple-testing-fwer": (
        "multiple-testing-fwer", "multiple-testing-bonferroni", "multiple-testing-hochberg",
    ),
    "multiple-testing-fdr": ("multiple-testing-fdr", "multiple-testing-by"),
    "pvalue-combination": (
        "pvalue-combination-fisher", "pvalue-combination-stouffer", "pvalue-combination-cauchy",
    ),
}
OVERVIEW = "models/testing/multiple-testing"
REFERENCE = "guides/multiple-testing-combine-pvalues"
PAGES = (OVERVIEW, *(model_doc(page) for page in LESSONS), REFERENCE)
PVALUES = np.array([0.001, 0.01, 0.03, 0.05, 0.5])


def _read(language, page):
    return (ROOT / "docs" / language / (page + ".md")).read_text(encoding="utf-8")


def _examples(text, page):
    examples = parse_examples(text, page)
    assert list(examples) == list(LESSONS[page]), f"{page}: named example inventory changed"
    assert all(not example.requires for example in examples.values())
    assert all(example.explicit_end for example in examples.values())
    return examples


def _run(language, page, name):
    text = _read(language, model_doc(page))
    _examples(text, page)
    return run_example(text, name, f"docs/{language}/{model_doc(page)}.md")


@pytest.mark.parametrize("language", ("en", "cn"))
def test_overview_routes_to_distinct_method_lessons_and_complete_api(language):
    overview = _read(language, OVERVIEW)
    reference = _read(language, REFERENCE)
    assert list(parse_examples(overview)) == ["multiple-testing-learner", "multiple-testing-families"]
    for page in LESSONS:
        lesson = _read(language, model_doc(page))
        assert f"]({page}.md)" in overview
        assert "](multiple-testing.md)" in lesson
        assert "](../../guides/multiple-testing-combine-pvalues.md)" in lesson
        assert f"](../{model_doc(page)}.md)" in reference
    assert "](../../guides/multiple-testing-combine-pvalues.md)" in overview
    assert "](../models/testing/multiple-testing.md)" in reference
    # Shared error criteria and family selection stay in the overview. The
    # detailed method derivations and comparisons belong to their own lessons.
    assert r"\mathrm{FWER}=P(V\ge1)" in overview
    assert r"\frac{V}{\max(R,1)}" in overview
    assert "axis=None" in overview and "axis=1" in overview
    assert r"\tilde p" not in overview and r"T_C=" not in overview
    assert "37.4167" not in overview


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("entry", ("README", "usage", "models/README", "models/testing/README"))
def test_index_navigation_exposes_every_split_page(language, entry):
    language_root = ROOT / "docs" / language
    reached = reachable_pages(language_root / (entry + ".md"), language_root)
    for page in PAGES:
        assert (language_root / (page + ".md")).resolve() in reached, (entry, page)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", PAGES)
def test_split_pages_have_valid_language_peers_and_public_facing_prose(language, page):
    text = _read(language, page)
    assert f"]({peer_link(language, page)})" in text
    for phrase in ("contract closure", "exact-head", "maintained route", "review-clean",
                   "证据闭环", "维护中的路径", "PR #171"):
        assert phrase not in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_all_links_to_split_pages_have_live_destinations_and_fragments(language):
    names = {Path(page).name for page in PAGES}
    for source in [ROOT / "README.md", *(ROOT / "docs" / language).rglob("*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
            link = urlsplit(target)
            if link.scheme or link.netloc:
                continue
            destination = (source.parent / unquote(link.path)).resolve() if link.path else source
            if destination.stem not in names:
                continue
            assert destination.is_file(), (source, target)
            if link.fragment:
                assert unquote(link.fragment) in markdown_anchors(destination.read_text(encoding="utf-8")), (source, target)


@pytest.mark.parametrize("page", LESSONS)
def test_every_method_has_matching_self_contained_bilingual_code(page):
    en, cn = (_read(language, model_doc(page)) for language in ("en", "cn"))
    for text in (en, cn):
        examples = _examples(text, page)
        for example in examples.values():
            assert len(example.blocks) >= 3
            assert max(len(block.splitlines()) for block in example.blocks) <= 16
            tree = ast.parse(example.code)
            assert any(isinstance(node, ast.Import) and any(alias.name == "numpy" for alias in node.names)
                       for node in tree.body)
            assert any(isinstance(node, ast.ImportFrom) and node.module == "statgpu.inference"
                       for node in tree.body)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name)
                     and node.func.id in {"adjust_pvalues", "combine_pvalues"}]
            assert calls
            for call in calls:
                assert any(keyword.arg == "backend" and isinstance(keyword.value, ast.Constant)
                           and keyword.value.value == "numpy" for keyword in call.keywords)
        fences = list(re.finditer(r"```python\n(.*?)```", text, re.DOTALL))
        assert all(any(example.start < fence.start() < example.end for example in examples.values())
                   for fence in fences), "No executable method fence may escape the strict inventory"
    for name in LESSONS[page]:
        assert ast.dump(ast.parse(example_code(en, name))) == ast.dump(ast.parse(example_code(cn, name)))


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", LESSONS)
def test_removing_any_method_wrapper_cannot_silently_shrink_execution_coverage(language, page):
    text = _read(language, model_doc(page))
    for name in LESSONS[page]:
        changed = re.sub(r"<!-- (?:api|learner)-example: " + name + r" -->", "", text)
        changed = changed.replace(f"<!-- example-end: {name} -->", "")
        with pytest.raises(AssertionError, match="named example inventory changed"):
            _examples(changed, page)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("method,name,p_name,reject_name,adjusted_name,alpha", [
    ("holm", "multiple-testing-fwer", "p", "reject", "adjusted", 0.05),
    ("bonferroni", "multiple-testing-bonferroni", "p", "reject_bonferroni", "adjusted_bonferroni", 0.05),
    ("hochberg", "multiple-testing-hochberg", "p_hochberg", "reject_hochberg", "adjusted_hochberg", 0.125),
    ("bh", "multiple-testing-fdr", "p", "reject_bh", "adjusted_bh", 0.05),
    ("by", "multiple-testing-by", "p", "reject_by", "adjusted_by", 0.05),
])
def test_documented_adjustments_match_independent_formulas(
    language, method, name, p_name, reject_name, adjusted_name, alpha
):
    page = "multiple-testing-fdr" if method in {"bh", "by"} else "multiple-testing-fwer"
    scope = _run(language, page, name)
    expected_input = [0.0625, 0.0625, 0.5] if method == "hochberg" else PVALUES
    np.testing.assert_array_equal(scope[p_name], expected_input)
    pvalues = scope[p_name]
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    m = len(pvalues)
    ranks = np.arange(1, m + 1)
    if method == "bonferroni":
        sorted_adjusted = m * sorted_p
    elif method == "holm":
        sorted_adjusted = np.maximum.accumulate((m - ranks + 1) * sorted_p)
    elif method == "hochberg":
        sorted_adjusted = np.minimum.accumulate(((m - ranks + 1) * sorted_p)[::-1])[::-1]
    else:
        sorted_adjusted = np.minimum.accumulate((m * sorted_p / ranks)[::-1])[::-1]
        if method == "by":
            sorted_adjusted *= np.sum(1 / ranks)
    expected = np.empty_like(pvalues)
    expected[order] = np.minimum(sorted_adjusted, 1)
    np.testing.assert_allclose(scope[adjusted_name], expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(scope[reject_name], expected <= alpha)
    assert scope[adjusted_name].shape == scope[reject_name].shape == pvalues.shape
    assert scope[adjusted_name].dtype == np.float64
    assert scope[reject_name].dtype == np.bool_
    if method == "hochberg":
        np.testing.assert_array_equal(scope[reject_name], [True, True, False])
        np.testing.assert_array_equal(scope[adjusted_name], [0.125, 0.125, 0.5])
    elif method == "by":
        np.testing.assert_array_equal(scope[reject_name], [True, False, False, False, False])
    elif method == "bonferroni":
        np.testing.assert_array_equal(scope[reject_name], [True, True, False, False, False])


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("method", ("fisher", "stouffer", "cauchy"))
def test_documented_combination_matches_independent_reference(language, method):
    scope = _run(language, "pvalue-combination", f"pvalue-combination-{method}")
    np.testing.assert_array_equal(scope["p"], PVALUES)
    if method == "fisher":
        statistic, probability = scope["statistic"], scope["global_p"]
        expected_stat = -2 * np.log(PVALUES).sum()
        expected_p = stats.chi2.sf(expected_stat, 2 * len(PVALUES))
    else:
        np.testing.assert_array_equal(scope["weights"], [2.0, 1.0, 1.0, 1.0, 1.0])
        statistic, probability = scope[f"statistic_{method}"], scope[f"global_p_{method}"]
        weights = scope["weights"]
        if method == "stouffer":
            expected_stat = np.dot(weights, stats.norm.isf(PVALUES)) / np.linalg.norm(weights)
            expected_p = stats.norm.sf(expected_stat)
        else:
            expected_stat = np.dot(weights / weights.sum(), np.tan(np.pi * (0.5 - PVALUES)))
            expected_p = stats.cauchy.sf(expected_stat)
    np.testing.assert_allclose(statistic, expected_stat, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(probability, expected_p, rtol=1e-9, atol=1e-12)
    assert np.ndim(statistic) == np.ndim(probability) == 0
    assert np.isfinite(statistic) and 0 < probability < 1
    text = _read(language, model_doc("pvalue-combination"))
    assert f"{float(statistic):.4f}" in text
    assert f"{float(probability):.6f}" in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_fwer_and_fdr_lessons_explain_target_assumptions_and_formulas_before_calls(language):
    fwer = _read(language, model_doc("multiple-testing-fwer"))
    fdr = _read(language, model_doc("multiple-testing-fdr"))
    first_fwer = fwer.index("<!-- api-example: multiple-testing-fwer -->")
    first_fdr = fdr.index("<!-- api-example: multiple-testing-fdr -->")
    for fragment in (r"\mathrm{FWER}=P(V\ge1)", "Simes", "[0,1]", r"P(p_i\le t)\le t"):
        assert fragment in fwer[:first_fwer]
    for fragment in (r"\frac{V}{\max(R,1)}", "PRDS", "[0,1]", r"P(p_i\le t)\le t"):
        assert fragment in fdr[:first_fdr]
    for text, formula, marker in (
        (fwer, r"\max_{j\le i}(m-j+1)p_{(j)}", "multiple-testing-fwer"),
        (fwer, r"\min(1,mp_{(i)})", "multiple-testing-bonferroni"),
        (fwer, r"\min_{j\ge i}(m-j+1)p_{(j)}", "multiple-testing-hochberg"),
        (fdr, r"\min_{j\ge i}(m/j)p_{(j)}", "multiple-testing-fdr"),
        (fdr, r"c_m\min_{j\ge i}(m/j)p_{(j)}", "multiple-testing-by"),
    ):
        assert text.index(formula) < text.index(f"<!-- api-example: {marker} -->")
    assert "[0.04, 0.04, 0.04]" in fdr
    assert "0.125" in fwer and "[0.0625, 0.0625, 0.5]" in fwer


@pytest.mark.parametrize("language", ("en", "cn"))
def test_combination_lesson_keeps_derivations_calibration_and_direction_boundaries(language):
    text = _read(language, model_doc("pvalue-combination"))
    for formula, marker in (
        (r"T_F=-2\sum", "fisher"),
        (r"T_Z=\frac", "stouffer"),
        (r"T_C=\sum", "cauchy"),
    ):
        assert text.index(formula) < text.index(f"<!-- api-example: pvalue-combination-{marker} -->")
    assert "arxiv.org/abs/1808.09011" in text
    assert "docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html" in text
    compact = " ".join(text.split())
    if language == "en":
        for phrase in ("independent, continuous uniform null p-values", "covariance-aware denominator",
                       "not exact calibration at every significance level", "consistently oriented one-sided",
                       "does not recover effect signs", "does not provide individual FDR control",
                       "Prespecify them, or justify their independence"):
            assert phrase in compact
    else:
        for phrase in ("相互独立", "连续且服从均匀分布", "协方差项", "也没有相应的分母校正",
                       "方向一致的单侧 P 值", "传入双侧 P 值不会自动恢复效应正负号",
                       "不是对任意联合分布、任意显著性水平都成立的精确校准",
                       "也不能控制逐项发现的 FDR", "应预先指定权重"):
            assert phrase in compact


@pytest.mark.parametrize("language", ("en", "cn"))
def test_complete_api_retains_signatures_parameters_aliases_and_numerical_boundaries(language):
    text = _read(language, REFERENCE)
    for function in (adjust_pvalues, combine_pvalues, multipletests):
        assert function.__name__ + "(" in text
        for name in inspect.signature(function).parameters:
            assert re.search(r"\b" + name + r"\b", text), (function.__name__, name)
    for token in ("fdr_hochberg", "float64", "nan_policy", "NaN", "eps", "weights", "axis=None"):
        assert token in text
