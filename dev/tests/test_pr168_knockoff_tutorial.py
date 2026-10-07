"""Keep the learner's first knockoff workflow statistically coherent."""

import re
from pathlib import Path

import numpy as np
import pytest

from statgpu import fixed_x_knockoff_filter

ROOT = Path(__file__).resolve().parents[2]


def _example(language, page, marker):
    path = ROOT / f"docs/{language}/{page}.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"<!-- " + re.escape(marker) + r" -->\s*```python\n(.*?)```",
        text, flags=re.DOTALL,
    )
    assert match is not None
    namespace = {}
    exec(compile(match.group(1), str(path), "exec"), namespace)  # noqa: S102
    return text, match.group(1), namespace


@pytest.mark.parametrize("language", ["en", "cn"])
def test_first_tutorial_supplies_centered_complete_matched_pair(language):
    _, _, ns = _example(language, "models/knockoff", "learner-example: knockoff-selection")
    X, Xk, y = (ns[name] for name in ("X", "Xk", "y"))
    n, p = X.shape
    assert (n, p, ns["q"]) == (240, 20, .20)
    assert n >= 2 * p + 1
    P = np.eye(n) - np.ones((n, n)) / n
    for design in (X, Xk):
        np.testing.assert_allclose(design.mean(0), 0, atol=1e-14)
        np.testing.assert_allclose(design.T @ P @ design, np.eye(p), atol=1e-14)
    np.testing.assert_allclose(X.T @ P @ Xk, np.zeros((p, p)), atol=1e-14)
    assert ns["result"].metadata["xk_source"] == "provided"

    # Match the documented intercept and independent homoskedastic Gaussian DGP.
    rng = np.random.default_rng(42)
    rng.normal(size=(n, 2 * p))
    expected_noise = rng.normal(scale=.5, size=n)
    np.testing.assert_allclose(y, 2 + X @ ns["beta"] + expected_noise, atol=1e-14)
    np.testing.assert_array_equal(np.flatnonzero(ns["beta"]), np.arange(6))


@pytest.mark.parametrize("language", ["en", "cn"])
def test_tutorial_statistics_and_complete_threshold_counts_agree(language):
    text, code, ns = _example(language, "models/knockoff", "learner-example: knockoff-selection")
    X, Xk, y, result = (ns[name] for name in ("X", "Xk", "y", "result"))
    expected_w = abs(X.T @ (y - y.mean())) - abs(Xk.T @ (y - y.mean()))
    np.testing.assert_allclose(result.W, expected_w, atol=1e-14)
    assert len(np.unique(abs(result.W))) == ns["p"]
    eligible = [
        t for t in np.unique(abs(result.W)) if t > 0
        and (1 + np.count_nonzero(result.W <= -t))
        / max(1, np.count_nonzero(result.W >= t)) <= ns["q"]
    ]
    assert result.threshold == pytest.approx(min(eligible))
    np.testing.assert_array_equal(result.selected_features, np.flatnonzero(result.W >= min(eligible)))
    assert result.estimated_fdr == pytest.approx(1 / 7)
    assert 15 in result.selected_features and ns["beta"][15] == 0
    assert "q = 0.20" in code
    assert code.index("q = 0.20") < code.index("y =")
    assert text.index("learner-example: knockoff-selection") < text.index("repeated-lasso-statistic-calls")

    swapped = fixed_x_knockoff_filter(Xk, y, Xk=X, q=ns["q"], backend="numpy")
    np.testing.assert_allclose(swapped.W, -result.W, atol=1e-14)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_selector_example_teaches_inevitable_empty_result(language):
    text, code, ns = _example(language, "reference/feature-selection-api", "api-example: knockoff-selector")
    result = ns["selector"].result_
    assert ns["p"] == 5 and ns["q"] == .1
    assert 1 / ns["p"] > ns["q"]
    assert code.index("q = 0.1") < code.index("y =")
    assert result.selected_features.size == 0
    assert np.isinf(result.threshold)
    assert result.estimated_fdr == 0
    assert ns["selected"].shape == (10, 0)
    assert "1/p=0.2>q" in text
    assert "../models/knockoff.md#centered-pair-example" in text


@pytest.mark.parametrize("language", ["en", "cn"])
def test_knockoff_learner_links_contributor_validation_instead_of_listing_scripts(language):
    text = (ROOT / f"docs/{language}/models/knockoff.md").read_text(encoding="utf-8")
    assert "../../../dev/references/model-validation.md#knockoff" in text
    assert "dev/benchmarks/benchmark_knockoff_" not in text
    assert "#centered-pair-example" in text
    for safeguard in ("q=np.nan", "W=[8,8,-8]", "n≥2p+1", "random_state=None",
                      "torch-lasso-device-routing", "repeated-lasso-statistic-calls"):
        assert safeguard in text


@pytest.mark.parametrize("language,anchors", [
    ("en", ("overview", "validate-the-target-rate", "a-complete-cpu-example",
            "path", "objective-function", "estimating-equation", "external-validation")),
    ("cn", ("概览", "先验证目标错误率", "完整的-cpu-示例", "路径", "目标函数",
            "估计方程", "外部验证external-validation", "严格与近似模式的差别")),
])
def test_moved_sections_preserve_inbound_heading_links(language, anchors):
    text = (ROOT / f"docs/{language}/models/knockoff.md").read_text(encoding="utf-8")
    for anchor in anchors:
        assert f'<a id="{anchor}"></a>' in text
