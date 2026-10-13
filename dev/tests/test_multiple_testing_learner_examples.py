"""Execute the bilingual multiple-testing lesson and check its numeric tables.

These bounded example/navigation checks do not replace statistical prose review.
"""

import math
from pathlib import Path
import re

import numpy as np
import pytest

from doc_examples import example_code, run_example
from doc_paths import reachable_pages


ROOT = Path(__file__).resolve().parents[2]
MARKER = "multiple-testing-learner"


def _lesson(language):
    return (ROOT / f"docs/{language}/models/testing/multiple-testing.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("language", ("en", "cn"))
def test_progressive_lesson_results_match_the_explanation(language):
    scope = run_example(_lesson(language), MARKER, f"{language}/multiple-testing")
    np.testing.assert_array_equal(scope["p"], [0.001, 0.01, 0.03, 0.05, 0.5])
    np.testing.assert_array_equal(scope["reject"], [True, True, False, False, False])
    np.testing.assert_allclose(scope["adjusted"], [0.005, 0.04, 0.09, 0.1, 0.5])
    np.testing.assert_array_equal(scope["reject_bh"], [True, True, True, False, False])
    np.testing.assert_allclose(scope["adjusted_bh"], [0.005, 0.025, 0.05, 0.0625, 0.5])

    # A first-failure stopping rule would reject none in this BH example.
    np.testing.assert_array_equal(scope["p_step_up"], [0.02, 0.03, 0.04])
    assert scope["p_step_up"][0] > 0.05 / 3
    assert scope["p_step_up"][-1] <= 0.05
    np.testing.assert_array_equal(scope["reject_step_up"], [True, True, True])
    np.testing.assert_allclose(scope["adjusted_step_up"], [0.04, 0.04, 0.04])

    # For chi-square(2m), the survival function has this finite gamma-series form.
    half_statistic = -sum(math.log(p) for p in scope["p"])
    global_p = math.exp(-half_statistic) * sum(
        half_statistic ** j / math.factorial(j) for j in range(len(scope["p"]))
    )
    np.testing.assert_allclose(scope["statistic"], 2 * half_statistic, rtol=1e-12)
    np.testing.assert_allclose(scope["global_p"], global_p, rtol=1e-9)
    assert round(float(scope["statistic"]), 4) == 37.4167
    assert round(float(scope["global_p"]), 6) == 0.000048


def test_english_and_chinese_execute_identical_progressive_steps():
    assert example_code(_lesson("en"), MARKER) == example_code(_lesson("cn"), MARKER)


def _numeric_table(text, method):
    # Identify the method in a table header, independent of headings/prose language.
    for table in re.findall(r"(?:^\|.*\|[ \t]*\n)+", text, flags=re.MULTILINE):
        lines = table.splitlines()
        if method in lines[0]:
            return [[cell.strip() for cell in row.strip("|").split("|")]
                    for row in lines[2:]]
    raise AssertionError(f"Missing {method} walkthrough table")


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("method", ("Holm", "BH"))
def test_rank_tables_match_the_documented_comparison_thresholds(language, method):
    rows = _numeric_table(_lesson(language), method)
    assert len(rows) == 5
    ranks = np.array([int(row[0]) for row in rows])
    np.testing.assert_array_equal(ranks, np.arange(1, 6))
    pvalues = np.array([float(row[1]) for row in rows])
    np.testing.assert_array_equal(pvalues, [0.001, 0.01, 0.03, 0.05, 0.5])
    cutoffs = np.array([float(re.split(r"[=≈]", row[2])[-1]) for row in rows])
    expected = 0.05 / (6 - ranks) if method == "Holm" else ranks * 0.05 / 5
    np.testing.assert_allclose(cutoffs, expected, atol=0.00005, rtol=0)
    # Check decisions with exact thresholds rather than their printed rounding.
    passes = pvalues <= expected
    expected_passes = [True, True, False, False, False] if method == "Holm" else [True, True, True, False, False]
    np.testing.assert_array_equal(passes, expected_passes)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ("usage.md", "guides/inference-api.md"))
def test_navigation_distinguishes_principles_from_complete_api(language, page):
    path = ROOT / f"docs/{language}/{page}"
    text = path.read_text(encoding="utf-8")
    prefix = "../" if page.startswith("guides/") else ""
    api_prefix = "" if page.startswith("guides/") else "guides/"
    if page == "usage.md":
        # The beginner path routes to the testing family before the two layers.
        reached = reachable_pages(path, ROOT / "docs" / language)
        for relative in ("models/testing/multiple-testing.md",
                         "guides/multiple-testing-combine-pvalues.md"):
            assert (ROOT / "docs" / language / relative).resolve() in reached
    else:
        assert re.search(r"\]\(" + re.escape(prefix + "models/testing/multiple-testing.md") + r"(?:#[^)]*)?\)", text)
        assert re.search(r"\]\(" + re.escape(api_prefix + "multiple-testing-combine-pvalues.md") + r"(?:#[^)]*)?\)", text)
