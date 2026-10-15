"""Keep the two knockoff lessons distinct, runnable, and statistically explicit.

These CPU checks execute the published examples without hidden fixtures. They
check analytic pair/threshold properties, not empirical FDR or GPU performance.
"""

import ast
from dataclasses import fields
import inspect
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest

from doc_paths import model_doc, peer_link, reachable_pages
from doc_examples import example_code, parse_examples, run_example
from statgpu import feature_selection as selection

ROOT = Path(__file__).resolve().parents[2]
LESSONS = {
    "knockoff-fixed-x": "knockoff-selection",
    "knockoff-model-x": "knockoff-model-x",
}
PAGES = ("models/feature-selection/knockoff", *(model_doc(page) for page in LESSONS),
         "reference/feature-selection-api")
ENTRY_POINTS = ("README", "usage", "models/README", "guides/implemented-methods")
CALLABLES = (
    "fixed_x_knockoff_filter", "model_x_knockoff_filter", "knockoff_filter",
    "FixedXKnockoffSelector", "KnockoffSelector",
)


def _read(language, page):
    return (ROOT / f"docs/{language}/{page}.md").read_text(encoding="utf-8")


def _run(language, page, marker):
    return run_example(_read(language, page), marker, f"docs/{language}/{page}.md")


def _anchors(text):
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', text))
    seen = {}
    for heading in re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if not count else f"{slug}-{count}")
    return anchors


@pytest.mark.parametrize("language", ("en", "cn"))
def test_overview_routes_readers_to_independent_lessons_and_one_complete_api(language):
    overview = _read(language, "models/feature-selection/knockoff")
    reference = _read(language, "reference/feature-selection-api")
    assert not parse_examples(overview), "The overview must not duplicate a full tutorial"
    for page in LESSONS:
        assert f"]({page}.md" in overview
        lesson = _read(language, model_doc(page))
        assert "](knockoff.md" in lesson
        assert "](../../reference/feature-selection-api.md" in lesson
        assert f"](../{model_doc(page)}.md" in reference
    assert "](../../reference/feature-selection-api.md" in overview
    assert "](../models/feature-selection/knockoff.md" in reference
    for token in ("Fixed-X", "Model-X", "FDR", "knockoff_plus", "q", "W"):
        assert token in overview
    # Complete call signatures have one owner; small task-specific calls belong
    # in tutorials, and the overview retains concepts rather than an inventory.
    for name in CALLABLES:
        assert name + "(" not in overview


@pytest.mark.parametrize("page,marker", LESSONS.items())
def test_bilingual_lessons_have_identical_self_contained_cpu_and_selector_steps(page, marker):
    en = _read("en", model_doc(page))
    cn = _read("cn", model_doc(page))
    selector_marker = page + "-selector"
    for text in (en, cn):
        examples = parse_examples(text)
        assert next(iter(examples)) == marker
        assert not examples[marker].requires
        assert examples[marker].explicit_end
        assert examples[selector_marker].requires == (marker,)
        assert examples[selector_marker].explicit_end
    for name in (marker, selector_marker):
        assert ast.dump(ast.parse(example_code(en, name))) == ast.dump(ast.parse(example_code(cn, name)))


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", PAGES)
def test_split_pages_keep_language_peers_and_audience_boundaries(language, page):
    text = _read(language, page)
    other = "cn" if language == "en" else "en"
    assert f"]({peer_link(language, page)})" in text
    for phrase in ("contract closure", "exact-head", "maintained route", "review-clean",
                   "证据闭环", "维护中的路径", "PR #168"):
        assert phrase not in text


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ENTRY_POINTS)
def test_entry_points_expose_both_knockoff_lessons_and_the_complete_api(language, page):
    language_root = ROOT / "docs" / language
    reached = reachable_pages(language_root / (page + ".md"), language_root)
    for destination in ("knockoff", *LESSONS):
        expected = (language_root / (model_doc(destination) + ".md")).resolve()
        assert expected in reached, (page, destination)
    assert (language_root / "reference/feature-selection-api.md").resolve() in reached


def test_root_readme_exposes_all_knockoff_layers():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for page in ("knockoff", *LESSONS):
        assert f"](docs/en/{model_doc(page)}.md)" in text
    assert "](docs/en/reference/feature-selection-api.md)" in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_all_knockoff_and_feature_reference_links_resolve(language):
    paths = [ROOT / "README.md"] + list((ROOT / f"docs/{language}").rglob("*.md"))
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            link = urlsplit(target)
            if link.scheme or link.netloc:
                continue
            destination = (path.parent / unquote(link.path)).resolve() if link.path else path
            if not (destination.stem.startswith("knockoff")
                    or destination.stem == "feature-selection-api"):
                continue
            assert destination.is_file(), (path, target)
            if link.fragment:
                assert unquote(link.fragment) in _anchors(destination.read_text(encoding="utf-8")), (path, target)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_complete_api_retains_knockoff_signatures_defaults_methods_and_result_fields(language):
    reference = _read(language, "reference/feature-selection-api")
    for name in selection.__all__:
        assert name in reference
    for name in CALLABLES:
        signature = inspect.signature(getattr(selection, name))
        matches = re.findall(r"^" + name + r"\((.+)\)$", reference, re.MULTILINE)
        assert len(matches) == 1, (name, "one canonical full signature")
        call = ast.parse(name + "(" + matches[0] + ")").body[0].value
        parameters = [argument.id for argument in call.args] + [keyword.arg for keyword in call.keywords]
        assert parameters == list(signature.parameters), name
        for keyword in call.keywords:
            assert ast.literal_eval(keyword.value) == signature.parameters[keyword.arg].default, (name, keyword.arg)
    for cls in (selection.KnockoffSelector, selection.FixedXKnockoffSelector):
        assert set(cls().get_params()) == set(inspect.signature(cls).parameters)
        for name, method in inspect.getmembers(cls, callable):
            if not name.startswith("_"):
                assert re.search(r"\b" + name + r"\b", reference), (cls.__name__, name)
    for field in fields(selection.KnockoffResult):
        assert re.search(r"\b" + field.name + r"\b", reference), field.name
    for token in ("to_dict()", "get_support()", "selected_features_", "result_",
                  "q_trajectory", "rank", "fdr_hat", "n_selected"):
        assert token in reference


@pytest.mark.parametrize("language", ("en", "cn"))
def test_model_x_tutorial_uses_the_known_joint_feature_law_and_independent_response_noise(language):
    ns = _run(language, "models/feature-selection/knockoff-model-x", "knockoff-model-x")
    assert (ns["n"], ns["p"], ns["q"]) == (400, 20, .20)
    rng = np.random.default_rng(73)
    X = rng.normal(size=(400, 20))
    Xk = rng.normal(size=(400, 20))
    beta = np.r_[3, -2.5, 2, -1.5, 1.2, -1, np.zeros(14)]
    response = 2 + X @ beta + 2 * np.sin(X[:, 0]) + rng.normal(scale=1, size=400)
    np.testing.assert_array_equal(ns["X"], X)
    np.testing.assert_array_equal(ns["Xk"], Xk)
    np.testing.assert_array_equal(ns["beta"], beta)
    np.testing.assert_allclose(ns["y"], response, atol=1e-13)
    assert not np.shares_memory(ns["X"], ns["Xk"])
    result = ns["result"]
    assert result.metadata["xk_source"] == "provided"
    assert result.metadata["n_modelx_draws"] == 1
    assert result.knockoff_type == "model_x" and result.backend == "numpy"
    code = example_code(_read(language, "models/feature-selection/knockoff-model-x"), "knockoff-model-x")
    assert "modelx_draws=7" in code
    assert code.index("q = 0.20") < code.index("y =")
    assert code.index("Xk =") < code.index("y =")


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page,marker", LESSONS.items())
def test_each_lesson_matches_full_threshold_counts_and_numeric_interpretation(language, page, marker):
    ns = _run(language, model_doc(page), marker)
    X, Xk, y, result = (ns[key] for key in ("X", "Xk", "y", "result"))
    expected_w = abs(X.T @ (y - y.mean())) - abs(Xk.T @ (y - y.mean()))
    np.testing.assert_allclose(result.W, expected_w, atol=1e-10)
    assert len(np.unique(abs(expected_w))) == ns["p"]
    assert np.all(expected_w != 0)
    thresholds = np.unique(abs(expected_w))
    negative = np.array([np.count_nonzero(expected_w <= -t) for t in thresholds])
    positive = np.array([np.count_nonzero(expected_w >= t) for t in thresholds])
    ratios = (1 + negative) / np.maximum(positive, 1)
    eligible = thresholds[ratios <= ns["q"]]
    assert eligible.size
    threshold = eligible.min()
    assert result.threshold == pytest.approx(threshold)
    np.testing.assert_array_equal(result.selected_features, np.flatnonzero(expected_w >= threshold))
    selected_position = np.flatnonzero(thresholds == threshold).item()
    assert result.estimated_fdr == pytest.approx(ratios[selected_position])
    # Check every candidate, not only the selected threshold. These examples
    # deliberately have no ties; partial tied groups cannot establish this rule.
    assert len(result.q_trajectory) == len(thresholds)
    for rank, (t, pos, ratio) in enumerate(zip(thresholds[::-1], positive[::-1], ratios[::-1]), start=1):
        row = result.q_trajectory[rank - 1]
        assert row["rank"] == rank
        assert row["threshold"] == pytest.approx(t)
        assert row["n_selected"] == max(int(pos), 1)
        assert row["fdr_hat"] == pytest.approx(min(float(ratio), 1))
    if page == "knockoff-model-x":
        np.testing.assert_array_equal(result.selected_features, np.arange(6))
        assert result.threshold == pytest.approx(173.09154455462868)
        assert result.estimated_fdr == pytest.approx(1 / 6)
        assert result.W[0] == pytest.approx(1541.74561015)
        assert result.W[17] == pytest.approx(-90.69420842)
        assert positive[selected_position] == 6 and negative[selected_position] == 0
    else:
        np.testing.assert_array_equal(result.selected_features, [0, 1, 2, 3, 4, 5, 15])
        assert result.threshold == pytest.approx(1.019324242097392)
        assert result.estimated_fdr == pytest.approx(1 / 7)
        assert ns["beta"][15] == 0
    swapped_function = (selection.model_x_knockoff_filter if page == "knockoff-model-x"
                        else selection.fixed_x_knockoff_filter)
    swapped = swapped_function(Xk, y, Xk=X, q=ns["q"], backend="numpy")
    np.testing.assert_allclose(swapped.W, -result.W, atol=1e-10)
    # The model-X flip-sign property concerns swapping any subset of pairs,
    # not just swapping the entire matrices at once. Check each coordinate.
    for column in range(ns["p"]):
        x_swap, xk_swap = X.copy(), Xk.copy()
        x_swap[:, column], xk_swap[:, column] = Xk[:, column], X[:, column]
        one_swap = swapped_function(x_swap, y, Xk=xk_swap, q=ns["q"], backend="numpy")
        expected_swap = result.W.copy()
        expected_swap[column] *= -1
        np.testing.assert_allclose(one_swap.W, expected_swap, atol=1e-10)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page,marker", LESSONS.items())
def test_printed_outputs_match_the_explained_numerical_results(language, page, marker, capsys):
    ns = _run(language, model_doc(page), marker)
    output = capsys.readouterr().out
    result = ns["result"]
    assert f"Selected columns: {result.selected_features.tolist()}" in output
    assert f"Threshold: {round(result.threshold, 3)}" in output
    assert f"Threshold ratio: {round(result.estimated_fdr, 3)}" in output
    text = _read(language, model_doc(page))
    assert str(result.selected_features.tolist()) in text
    assert f"{result.threshold:.3f}" in text
    assert f"{result.estimated_fdr:.3f}" in text
    if page == "knockoff-model-x":
        assert "Knockoff source: provided" in output and "Actual draws: 1" in output
        assert "Positive / negative counts: 6 0" in output
        assert "W[0] / W[17]: 1541.746 -90.694" in output
        assert ns["positive_count"] == 6 and ns["negative_count"] == 0


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", LESSONS)
def test_each_selector_example_preserves_selection_original_layout_and_parameter_contract(language, page):
    ns = _run(language, model_doc(page), page + "-selector")
    selector, result = ns["selector"], ns["result"]
    np.testing.assert_array_equal(selector.selected_features_, result.selected_features)
    np.testing.assert_allclose(selector.result_.W, result.W, atol=1e-10)
    support = selector.get_support()
    assert support.dtype == np.bool_ and support.shape == (ns["p"],)
    np.testing.assert_array_equal(np.flatnonzero(support), result.selected_features)
    query = ns["X"][:10].astype(np.float32)
    transformed = selector.transform(query)
    assert transformed.dtype == query.dtype
    assert transformed.shape == (10, len(result.selected_features))
    np.testing.assert_array_equal(transformed, query[:, support])
    with pytest.raises(ValueError, match="features"):
        selector.transform(query[:, :-1])
    with pytest.raises(TypeError):
        selector.get_support(indices=True)
    assert all(not hasattr(selector, name) for name in ("predict", "score", "summary"))
    fresh = type(selector)(**selector.get_params())
    assert fresh.fit(ns["X"], ns["y"], Xk=ns["Xk"]) is fresh
    np.testing.assert_array_equal(fresh.get_support(), support)
    assert fresh.set_params(q=ns["q"]) is fresh
    assert fresh.result_ is None and fresh.selected_features_ is None
    with pytest.raises(RuntimeError, match="not been fitted"):
        fresh.get_support()


# Frozen explicit and implicit overview fragments from before the split.
LEGACY_ANCHORS = {'cn': {'centered-pair-example',
        'cpugpu-示例',
        'cupy',
        'knockoff-特征选择',
        'knockoff-规则',
        'knockoff-选择要解决什么问题',
        'reproducibility-of-generated-torch-model-x',
        'torch-cuda',
        'torch-自动-model-x-构造的可重复性',
        '严格与近似模式的差别',
        '估计方程',
        '使用有效外部配对的完整-cpu-示例',
        '先验证目标错误率',
        '协方差与推断',
        '参数',
        '参考references',
        '响应模型假设',
        '在选择前确定目标错误率',
        '外部验证external-validation',
        '如何理解选择结果及其不确定性',
        '如何选择-lasso-计算配置',
        '完整的-cpu-示例',
        '导入路径',
        '常见问题faq',
        '性能与统计边界',
        '构造并检查外部配对',
        '概览',
        '生成响应并选择特征',
        '目标函数',
        '统计量并列时的限制',
        '自动-fixed-x-构造的中心化限制',
        '路径',
        '输出outputs',
        '进阶参考',
        '重复计算-lasso-统计量',
        '阈值如何工作',
        '阈值规则与构造假设',
        '预先设定设计与目标错误率'},
 'en': {'a-complete-cpu-example',
        'a-complete-cpu-example-with-a-valid-supplied-pair',
        'advanced-reference',
        'centered-pair-example',
        'choosing-a-lasso-computational-profile',
        'construct-and-verify-the-supplied-pair',
        'covarianceinference',
        'cpugpu-examples',
        'cupy',
        'estimating-equation',
        'external-validation',
        'faq',
        'generate-the-response-and-select-features',
        'generated-fixed-x-centering-limitation',
        'how-the-threshold-works',
        'import-paths',
        'knockoff-feature-selection',
        'objective-function',
        'outputs',
        'overview',
        'parameters',
        'path',
        'performance-boundary',
        'plan-the-target-rate-before-selection',
        'prespecify-the-design-and-target',
        'read-the-selection-and-its-uncertainty',
        'references',
        'repeated-lasso-statistic-calls',
        'reproducibility-of-generated-torch-model-x',
        'response-model-assumptions',
        'the-knockoff-rule',
        'threshold-rules-and-construction-assumptions',
        'tied-statistic-limitation',
        'torch-cuda',
        'validate-the-target-rate',
        'what-problem-does-knockoff-selection-solve'}}


@pytest.mark.parametrize("language", ("en", "cn"))
def test_every_pre_split_overview_fragment_still_resolves(language):
    text = _read(language, "models/feature-selection/knockoff")
    assert len(LEGACY_ANCHORS[language]) == {"en": 36, "cn": 38}[language]
    assert LEGACY_ANCHORS[language] <= _anchors(text)
    explicit = re.findall(r'<a id="([^\"]+)"></a>', text)
    assert len(explicit) == len(set(explicit)), "Duplicate explicit anchor IDs"
