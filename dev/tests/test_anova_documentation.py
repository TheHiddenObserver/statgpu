"""Execute ANOVA lessons and protect the layered navigation/API inventory.

These bounded CPU/SciPy checks exercise the documented examples and observable
contracts. They do not replace statistical prose review or physical GPU tests.
"""

import ast
import dataclasses
import inspect
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
from scipy import stats

import statgpu.anova as anova
from doc_paths import model_doc, peer_link, reachable_pages
from doc_examples import example_code, parse_examples, run_example


ROOT = Path(__file__).resolve().parents[2]
LESSONS = {
    "anova-oneway": ("anova-basic",),
    "anova-welch": ("anova-welch",),
    "anova-twoway": ("anova-twoway",),
    "anova-posthoc": ("anova-tukey", "anova-bonferroni"),
}
FUNCTIONS = (
    "f_oneway", "f_welch", "f_twoway", "tukey_hsd", "bonferroni",
    "cohens_f", "partial_eta_squared",
)
OVERVIEW_CONCEPTS = (
    "factors-and-design", "variance-intuition", "assumptions",
    "choose-the-analysis", "effect-sizes", "reporting",
)
ENTRY_POINTS = ("README", "usage", "models/README", "guides/implemented-methods")
RESULTS = (
    "AnovaResult", "TwoWayAnovaResult", "TukeyResult", "PosthocResult",
    "PairwiseComparison",
)


def _read(language, page):
    return (ROOT / f"docs/{language}/{page}.md").read_text(encoding="utf-8")


def _lesson(language, page, example):
    return run_example(_read(language, model_doc(page)), example,
                       f"docs/{language}/{model_doc(page)}.md")


@pytest.mark.parametrize("page,names", LESSONS.items())
def test_bilingual_lessons_have_the_same_self_contained_examples(page, names):
    en = _read("en", model_doc(page))
    cn = _read("cn", model_doc(page))
    for text in (en, cn):
        examples = parse_examples(text)
        assert tuple(name for name in examples if name not in {"anova-cupy", "anova-torch"}) == names
        assert all(not examples[name].requires for name in names)
    for name in names:
        assert example_code(en, name) == example_code(cn, name)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_classical_example_matches_scipy_and_sample_effect_size(language):
    scope = _lesson(language, "anova-oneway", "anova-basic")
    groups = (scope["g1"], scope["g2"])
    result = scope["result"]
    expected = stats.f_oneway(*groups)
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-11)
    assert result.pvalue == pytest.approx(expected.pvalue, rel=1e-7)
    all_data = np.concatenate(groups)
    ss_total = np.sum((all_data - all_data.mean()) ** 2)
    ss_between = sum(len(g) * (g.mean() - all_data.mean()) ** 2 for g in groups)
    assert result.eta_squared == pytest.approx(ss_between / ss_total)
    assert anova.cohens_f(*groups, backend="numpy") == pytest.approx(
        np.sqrt(ss_between / (ss_total - ss_between)))
    assert round(result.statistic, 3) == 22.079
    assert round(result.eta_squared, 3) == 0.100


@pytest.mark.parametrize("language", ("en", "cn"))
def test_welch_example_matches_formula_and_scipy_tail(language):
    scope = _lesson(language, "anova-welch", "anova-welch")
    groups, result = scope["unequal_groups"], scope["welch"]
    n = np.array([len(g) for g in groups])
    means = np.array([np.mean(g) for g in groups])
    weights = n / np.array([np.var(g, ddof=1) for g in groups])
    k = len(groups)
    adjustment = np.sum((1 - weights / weights.sum()) ** 2 / (n - 1))
    center = np.average(means, weights=weights)
    expected_f = (np.sum(weights * (means - center) ** 2) / (k - 1)
                  / (1 + 2 * (k - 2) * adjustment / (k * k - 1)))
    expected_df = (k * k - 1) / (3 * adjustment)
    assert result.statistic == pytest.approx(expected_f, rel=1e-12)
    assert result.df_within == pytest.approx(expected_df, rel=1e-12)
    assert result.pvalue == pytest.approx(stats.f.sf(expected_f, k - 1, expected_df), rel=1e-7)
    assert np.isnan(result.eta_squared)
    assert not float(result.df_within).is_integer()
    # For two groups, Welch ANOVA F is the square of Welch's t statistic.
    two = anova.f_welch(*groups[:2], backend="numpy")
    t = stats.ttest_ind(*groups[:2], equal_var=False)
    assert two.statistic == pytest.approx(t.statistic ** 2, rel=1e-12)
    assert two.pvalue == pytest.approx(t.pvalue, rel=1e-7)


def _factor_design(data):
    """Centered factorial coding makes the balanced effects orthogonal."""
    a, b, n = len(data), len(data[0]), len(data[0][0])
    ia = np.repeat(np.arange(a), b * n)
    ib = np.tile(np.repeat(np.arange(b), n), a)
    # The final level is minus the sum of the others (sum-to-zero coding).
    ac = np.column_stack([(ia == i).astype(float) - (ia == a - 1)
                          for i in range(a - 1)])
    bc = np.column_stack([(ib == j).astype(float) - (ib == b - 1)
                          for j in range(b - 1)])
    interaction = np.column_stack([ac[:, i] * bc[:, j]
                                   for i in range(a - 1) for j in range(b - 1)])
    y = np.concatenate([g for row in data for g in row])
    return y, [np.ones((len(y), 1)), ac, bc, interaction]


def _sse(y, blocks):
    design = np.column_stack(blocks)
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(residual @ residual), len(y) - np.linalg.matrix_rank(design)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_twoway_example_matches_nested_linear_models(language):
    scope = _lesson(language, "anova-twoway", "anova-twoway")
    data, full, additive = scope["data"], scope["result"], scope["additive"]
    y, blocks = _factor_design(data)
    sse_full, df_full = _sse(y, blocks)
    sst = np.sum((y - y.mean()) ** 2)
    assert full.ss_within == pytest.approx(sse_full)
    assert full.df_within == df_full
    for index, prefix in ((1, "factor_a"), (2, "factor_b"), (3, "interaction")):
        sse_reduced, df_reduced = _sse(y, [v for i, v in enumerate(blocks) if i != index])
        ss_effect = sse_reduced - sse_full
        df_effect = df_reduced - df_full
        expected_f = (ss_effect / df_effect) / (sse_full / df_full)
        assert getattr(full, prefix + "_statistic") == pytest.approx(expected_f)
        assert getattr(full, prefix + "_pvalue") == pytest.approx(
            stats.f.sf(expected_f, df_effect, df_full), rel=1e-7)
        assert getattr(full, prefix + "_eta_squared") == pytest.approx(ss_effect / sst)
    sse_additive, df_additive = _sse(y, blocks[:3])
    assert additive.ss_within == pytest.approx(sse_additive)
    assert additive.df_within == df_additive
    assert sse_additive > sse_full
    assert df_additive == df_full + full.interaction_df
    for index, prefix in ((1, "factor_a"), (2, "factor_b")):
        reduced, df_reduced = _sse(y, [v for i, v in enumerate(blocks[:3]) if i != index])
        expected_f = ((reduced - sse_additive) / (df_reduced - df_additive)
                      / (sse_additive / df_additive))
        assert getattr(additive, prefix + "_statistic") == pytest.approx(expected_f)
    assert additive.interaction_statistic is None
    assert additive.interaction_pvalue is None


@pytest.mark.parametrize("language", ("en", "cn"))
def test_tukey_example_matches_scipy_adjusted_pvalues_and_intervals(language):
    scope = _lesson(language, "anova-posthoc", "anova-tukey")
    groups, result = scope["groups"], scope["posthoc"]
    # studentized_range is present at the package's minimum supported SciPy;
    # scipy.stats.tukey_hsd was added later, so use the distribution directly.
    df = sum(map(len, groups)) - len(groups)
    mse = sum(np.sum((g - np.mean(g)) ** 2) for g in groups) / df
    critical = stats.studentized_range.ppf(1 - result.alpha, len(groups), df)
    for pair in result.comparisons:
        gi, gj = groups[pair.group_i], groups[pair.group_j]
        difference = np.mean(gi) - np.mean(gj)
        se = np.sqrt(mse / 2 * (1 / len(gi) + 1 / len(gj)))
        pvalue = stats.studentized_range.sf(abs(difference) / se, len(groups), df)
        assert pair.mean_diff == pytest.approx(difference)
        assert pair.pvalue == pytest.approx(pvalue, rel=1e-7)
        assert pair.ci_lower == pytest.approx(difference - critical * se)
        assert pair.ci_upper == pytest.approx(difference + critical * se)
        assert pair.reject == (pvalue < result.alpha)
    assert [c.reject for c in result.comparisons] == [False, True, True]


@pytest.mark.parametrize("language", ("en", "cn"))
def test_bonferroni_example_keeps_raw_welch_pvalues_and_adjusts_decisions(language):
    scope = _lesson(language, "anova-posthoc", "anova-bonferroni")
    groups, result = scope["unequal_groups"], scope["bonf"]
    assert result.n_comparisons == len(groups) * (len(groups) - 1) // 2
    first = result.comparisons[0]
    assert result.alpha / result.n_comparisons < first.pvalue < result.alpha
    assert first.reject is False
    assert first.ci_lower < 0 < first.ci_upper
    for pair in result.comparisons:
        gi, gj = groups[pair.group_i], groups[pair.group_j]
        expected = stats.ttest_ind(gi, gj, equal_var=False)
        ni, nj = len(gi), len(gj)
        vi, vj = np.var(gi, ddof=1) / ni, np.var(gj, ddof=1) / nj
        df = (vi + vj) ** 2 / (vi ** 2 / (ni - 1) + vj ** 2 / (nj - 1))
        margin = stats.t.isf(result.alpha / result.n_comparisons / 2, df) * np.sqrt(vi + vj)
        difference = np.mean(gi) - np.mean(gj)
        assert pair.pvalue == pytest.approx(expected.pvalue, rel=1e-7)
        assert pair.reject == (expected.pvalue < result.alpha / result.n_comparisons)
        assert pair.ci_lower == pytest.approx(difference - margin)
        assert pair.ci_upper == pytest.approx(difference + margin)
        assert pair.mean_diff == pytest.approx(difference)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_reference_inventories_every_export_and_dataclass_field(language):
    text = _read(language, "reference/anova-api")
    assert set(anova.__all__) == set(FUNCTIONS + RESULTS)
    for name in FUNCTIONS + RESULTS:
        assert re.search(r"\b" + name + r"\b", text), name
    assert len(FUNCTIONS) == 7 and len(RESULTS) == 5
    assert sum(len(dataclasses.fields(getattr(anova, name))) for name in RESULTS) == 34
    for name in RESULTS:
        cls = getattr(anova, name)
        assert dataclasses.is_dataclass(cls)
        section = text.split("### `" + name + "`", 1)[1]
        section = re.split(r"\n##+ ", section, maxsplit=1)[0]
        documented_fields = re.findall(r"^\| `([A-Za-z_]\w*)` \|", section, re.MULTILINE)
        expected_fields = {field.name for field in dataclasses.fields(cls)}
        assert set(documented_fields) == expected_fields, name
        assert len(documented_fields) == len(expected_fields), name
        assert not hasattr(cls, "summary")
    for name in FUNCTIONS:
        function = getattr(anova, name)
        for parameter in inspect.signature(function).parameters.values():
            assert re.search(r"\b" + parameter.name + r"\b", text)


def _documented_parameters(code):
    node = ast.parse("def " + code.strip() + ":\n    pass\n").body[0]
    args = node.args
    positional = args.posonlyargs + args.args
    defaults = [inspect.Parameter.empty] * (len(positional) - len(args.defaults))
    defaults += [ast.literal_eval(value) for value in args.defaults]
    parameters = []
    for index, (argument, default) in enumerate(zip(positional, defaults)):
        kind = (inspect.Parameter.POSITIONAL_ONLY if index < len(args.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        parameters.append((argument.arg, kind, default))
    if args.vararg:
        parameters.append((args.vararg.arg, inspect.Parameter.VAR_POSITIONAL,
                           inspect.Parameter.empty))
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append((argument.arg, inspect.Parameter.KEYWORD_ONLY,
                           inspect.Parameter.empty if default is None else ast.literal_eval(default)))
    if args.kwarg:
        parameters.append((args.kwarg.arg, inspect.Parameter.VAR_KEYWORD,
                           inspect.Parameter.empty))
    return parameters


@pytest.mark.parametrize("language", ("en", "cn"))
def test_reference_signatures_match_runtime_parameters_and_defaults(language):
    text = _read(language, "reference/anova-api")
    signatures = {}
    for block in re.findall(r"```text\n(.*?)```", text, re.DOTALL):
        match = re.match(r"([A-Za-z_]\w*)\(", block)
        if match and match.group(1) in FUNCTIONS + RESULTS:
            signatures[match.group(1)] = block
    assert set(signatures) == set(FUNCTIONS + RESULTS)
    for name, code in signatures.items():
        actual = [(p.name, p.kind, p.default)
                  for p in inspect.signature(getattr(anova, name)).parameters.values()]
        assert _documented_parameters(code) == actual, name


@pytest.mark.parametrize("language", ("en", "cn"))
def test_overview_routes_to_all_lessons_and_reference_without_duplicating_examples(language):
    text = _read(language, "models/testing/anova")
    assert not parse_examples(text)
    for page in LESSONS:
        assert "](" + page + ".md" in text
    assert "](../../reference/anova-api.md)" in text
    assert "](multiple-testing.md)" in text
    for page in LESSONS:
        tutorial = _read(language, model_doc(page))
        assert "](../../reference/anova-api.md" in tutorial
    assert "](multiple-testing.md" in _read(language, "models/testing/anova-posthoc")


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ENTRY_POINTS)
def test_entry_points_link_both_anova_principles_and_complete_reference(language, page):
    language_root = ROOT / "docs" / language
    reached = reachable_pages(language_root / (page + ".md"), language_root)
    for destination in ("anova", *LESSONS):
        expected = (language_root / (model_doc(destination) + ".md")).resolve()
        assert expected in reached, (page, destination)
    assert (language_root / "reference/anova-api.md").resolve() in reached


def _anchored_section(text, anchor):
    """Stable section identifiers protect coverage without freezing headings."""
    marker = f'<a id="{anchor}"></a>'
    assert text.count(marker) == 1, anchor
    tail = text.split(marker, 1)[1]
    heading = re.match(r"\s*(?:<a id=\"[^\"]+\"></a>\s*)*(#{2,6}) ([^\n]+)\n", tail)
    assert heading, f"{anchor}: anchor must introduce a section"
    body = re.split(r"\n#{1," + str(len(heading.group(1))) + r"} ",
                    tail[heading.end():], maxsplit=1)[0]
    body = re.sub(r"<a id=\"[^\"]+\"></a>", "", body).strip()
    assert body, f"{anchor}: concept section must not be empty"
    return heading.group(2), body


@pytest.mark.parametrize("language", ("en", "cn"))
def test_overview_has_distinct_general_concept_sections(language):
    text = _read(language, "models/testing/anova")
    sections = [_anchored_section(text, anchor) for anchor in OVERVIEW_CONCEPTS]
    assert len({heading for heading, _ in sections}) == len(OVERVIEW_CONCEPTS)
    assert not re.findall(r"```python\n", text)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_classical_and_welch_own_separate_formulas_and_examples(language):
    classical = _read(language, "models/testing/anova-oneway")
    welch = _read(language, "models/testing/anova-welch")
    _anchored_section(classical, "classical-anova")
    _anchored_section(classical, "anova-basic")
    _, formula = _anchored_section(welch, "welch-intuition")
    _anchored_section(welch, "anova-welch")
    assert "$$" in formula
    for anchor in ("welch-intuition", "anova-welch"):
        _, redirect = _anchored_section(classical, anchor)
        assert f"](anova-welch.md#{anchor})" in redirect
        assert "$$" not in redirect
        assert not parse_examples(redirect)
    assert "anova-welch" not in parse_examples(classical)
    # Catch the previous duplicated Welch derivation without banning comparison
    # prose, references, or a link to the unequal-variance alternative.
    assert all("F_W" not in block for block in re.findall(r"\$\$(.*?)\$\$", classical, re.DOTALL))
    for code in re.findall(r"```python\n(.*?)```", classical, re.DOTALL):
        tree = ast.parse(code)
        assert not any(isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name) and node.func.id == "f_welch"
            or isinstance(node.func, ast.Attribute) and node.func.attr == "f_welch"
        ) for node in ast.walk(tree))
    assert "](anova-welch.md" in classical
    assert "](anova-oneway.md" in welch


@pytest.mark.parametrize("language", ("en", "cn"))
def test_reference_routes_to_overview_and_each_peer_tutorial(language):
    text = _read(language, "reference/anova-api")
    introduction = text.split('<a id="public-api">', 1)[0]
    for page in ("anova", *LESSONS):
        assert f"](../{model_doc(page)}.md)" in introduction


def test_root_readme_exposes_all_anova_layers():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for page in ("anova", *LESSONS):
        assert f"](https://statgpu.github.io/en/{model_doc(page)})" in text
    assert "](https://statgpu.github.io/en/reference/anova-api)" in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_posthoc_links_general_concepts_and_welch_example_to_their_owners(language):
    text = _read(language, "models/testing/anova-posthoc")
    _, question = _anchored_section(text, "the-question")
    assert "](anova.md)" in question
    assert "](anova-oneway.md)" not in question
    _, example = _anchored_section(text, "anova-bonferroni")
    assert "](anova-welch.md#anova-welch)" in example


def _anova_navigation_paths():
    return [ROOT / "README.md"] + [
        ROOT / f"docs/{language}/{page}.md"
        for language in ("en", "cn")
        for page in (*ENTRY_POINTS, "models/testing/anova", "reference/anova-api",
                     *(model_doc(lesson) for lesson in LESSONS))
    ]


def _markdown_anchors(text):
    anchors = set(re.findall(r'<a id="([^\"]+)"></a>', text))
    seen = {}
    # GitHub-style heading anchors, including duplicate heading suffixes.
    for heading in re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if not count else f"{slug}-{count}")
    return anchors


@pytest.mark.parametrize("path", _anova_navigation_paths(),
                         ids=lambda path: str(path.relative_to(ROOT)))
def test_anova_navigation_has_existing_targets_and_fragments(path):
    text = path.read_text(encoding="utf-8")
    for label, target in re.findall(r"\[([^\]]*)\]\(([^)]+)\)", text):
        link = urlsplit(target)
        if link.scheme or link.netloc:
            continue
        destination = (path.parent / unquote(link.path)).resolve() if link.path else path
        if not destination.stem.startswith("anova"):
            continue
        assert destination.is_file(), (path, target)
        if link.fragment:
            anchors = _markdown_anchors(destination.read_text(encoding="utf-8"))
            assert unquote(link.fragment) in anchors, (path, target)
        # "Welch" must not continue to label the ordinary one-way destination.
        if "Welch" in label:
            assert destination.stem != "anova-oneway", (path, label, target)
