"""Numerical stepwise decisions and focused bilingual documentation guards.

The prose guards check wording only. The numerical cases independently exercise
the public API, including the failure-to-stop counterexample and exact equality.
"""

from pathlib import Path

import numpy as np
import pytest

from statgpu.inference import adjust_pvalues


@pytest.mark.parametrize(
    "method,pvalues,alpha,expected_reject,expected_adjusted",
    [
        ("holm", [0.04, 0.03], 0.05, [False, False], [0.06, 0.06]),
        ("holm", [0.09375, 0.0078125, 0.078125], 0.125,
         [False, True, False], [0.15625, 0.0234375, 0.15625]),
        ("holm", [0.125, 0.0625], 0.125, [True, True], [0.125, 0.125]),
        ("holm", [0.125], 0.125, [True], [0.125]),
        ("hochberg", [0.0625, 0.0625], 0.0625,
         [True, True], [0.0625, 0.0625]),
        ("hochberg", [0.5, 0.0625, 0.0625], 0.125,
         [False, True, True], [0.5, 0.125, 0.125]),
        ("hochberg", [0.25, 0.125], 0.0625,
         [False, False], [0.25, 0.25]),
        ("hochberg", [0.125], 0.125, [True], [0.125]),
    ],
)
def test_stepwise_decisions_stop_propagate_and_include_equality(
    method, pvalues, alpha, expected_reject, expected_adjusted
):
    # Test original input order and batched-axis dispatch independently.
    for values, axis in [(np.array(pvalues), None),
                         (np.array([pvalues, np.ones(len(pvalues))]), 1)]:
        reject, adjusted = adjust_pvalues(
            values, method=method, alpha=alpha, axis=axis, backend="numpy"
        )
        if axis is None:
            expected_mask = expected_reject
            expected_values = expected_adjusted
        else:
            # Distinct rows also detect accidental reuse of another row.
            expected_mask = [expected_reject, [False] * len(pvalues)]
            expected_values = [expected_adjusted, [1.0] * len(pvalues)]
        np.testing.assert_array_equal(reject, expected_mask)
        np.testing.assert_allclose(adjusted, expected_values, rtol=0, atol=1e-15)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_holm_documentation_keeps_inclusive_threshold_and_stopping_rule(language):
    # A lexical guard is not evidence of statistical validity.
    root = Path(__file__).resolve().parents[2]
    text = (root / f"docs/{language}/models/multiple-testing.md").read_text(
        encoding="utf-8"
    )
    holm = text.split("**Holm", 1)[1].split("**Benjamini", 1)[0]
    assert r"p_{(i)} \leq \alpha / (m - i + 1)" in holm
    assert r"p_{(i)} < \alpha / (m - i + 1)" not in holm
    if language == "en":
        assert "first failed comparison, stop" in holm
        assert "any remaining hypothesis" in holm
    else:
        assert "首次不满足条件时立即停止" in holm
        assert "后续假设均不拒绝" in holm
