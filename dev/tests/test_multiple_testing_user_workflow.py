"""Execute documented workflows and protect their public input/output contracts."""

import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.inference import adjust_pvalues, combine_pvalues


@pytest.mark.parametrize("language", ["en", "cn"])
def test_documented_cpu_examples_and_expected_results(language, capsys):
    path = Path(__file__).resolve().parents[2] / f"docs/{language}/models/multiple-testing.md"
    text = path.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    cpu_blocks = [block for block in blocks if "import torch" not in block]
    assert len(cpu_blocks) == 3
    namespaces = []
    # Each CPU example must be independently runnable, not depend on prior cells.
    for block in cpu_blocks:
        namespace = {}
        exec(compile(block, str(path), "exec"), namespace)  # noqa: S102 - trusted repository examples
        namespaces.append(namespace)
    first, batched, combined = namespaces
    np.testing.assert_array_equal(first["reject"], [True, True, True, False, False])
    np.testing.assert_allclose(first["pvals_adj"], [.005, .025, .05, .0625, .5])
    np.testing.assert_array_equal(batched["reject_all"], [[True, False], [False, False]])
    np.testing.assert_allclose(batched["adjusted_all"], [[.04, .12], [.4, .8]])
    np.testing.assert_array_equal(batched["reject_rows"], [[True, True], [False, False]])
    np.testing.assert_allclose(batched["adjusted_rows"], [[.02, .04], [.4, .8]])
    np.testing.assert_allclose(batched["row_p_global"], [.003529618404342517, .4532130341997297])
    assert batched["row_p_global"].shape == (2,)
    assert "37.4167, 0.000048" in capsys.readouterr().out
    assert float(combined["p_global"]) < .05
    # Keep literal displayed masks/p-values tied to the executed examples.
    assert "[True, True, True, False, False]" in text
    assert "[[True, False], [False, False]]" in text
    assert "[[True, True], [False, False]]" in text
    assert "[0.00353, 0.453213]" in text


@pytest.mark.parametrize("method", ["bh", "by", "holm", "bonferroni", "hochberg"])
def test_axis_family_shape_order_and_alpha_independence(method):
    p = np.array([[.04, .01], [.8, .2]])
    pooled_mask, pooled_adj = adjust_pvalues(p, method=method)
    flat_mask, flat_adj = adjust_pvalues(p.ravel(), method=method)
    np.testing.assert_array_equal(pooled_mask, flat_mask.reshape(p.shape))
    np.testing.assert_allclose(pooled_adj, flat_adj.reshape(p.shape))
    row_mask, row_adj = adjust_pvalues(p, method=method, axis=-1)
    for index, row in enumerate(p):
        expected_mask, expected_adj = adjust_pvalues(row, method=method)
        np.testing.assert_array_equal(row_mask[index], expected_mask)
        np.testing.assert_allclose(row_adj[index], expected_adj)
    _, changed_alpha = adjust_pvalues(p, method=method, alpha=.1)
    np.testing.assert_array_equal(pooled_adj, changed_alpha)
    np.testing.assert_array_equal(pooled_mask, pooled_adj <= .05)
    assert pooled_adj.dtype == np.float64
    assert pooled_mask.dtype == np.bool_
    assert isinstance(pooled_adj, np.ndarray)


@pytest.mark.parametrize("function", [adjust_pvalues, combine_pvalues])
@pytest.mark.parametrize("axis", [None, 1])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -.01, 1.01])
def test_invalid_pvalues_raise_without_omission(function, axis, invalid):
    with pytest.raises(ValueError):
        function(np.array([[.1, invalid], [.2, .8]]), axis=axis)


@pytest.mark.parametrize("method", ["fisher", "cauchy", "stouffer"])
def test_combination_shape_and_axis_matches_individual_calls(method):
    p = np.array([[.04, .01], [.8, .2]])
    stats, pvalues = combine_pvalues(p, method=method, axis=1)
    assert stats.shape == pvalues.shape == (2,)
    for index, row in enumerate(p):
        expected = combine_pvalues(row, method=method)
        np.testing.assert_allclose([stats[index], pvalues[index]], expected)
    pooled = combine_pvalues(p, method=method, axis=None)
    flat = combine_pvalues(p.ravel(), method=method)
    np.testing.assert_allclose(pooled, flat)
    assert all(np.ndim(value) == 0 for value in pooled)


@pytest.mark.parametrize("method", ["cauchy", "stouffer"])
def test_weights_are_shared_normalized_and_default_equal(method):
    p = np.array([[.04, .01], [.8, .2]])
    weights = np.array([1., 2.])
    result = combine_pvalues(p, method=method, weights=weights, axis=1)
    scaled = combine_pvalues(p, method=method, weights=10 * weights, axis=1)
    np.testing.assert_allclose(result, scaled)
    for index, row in enumerate(p):
        expected = combine_pvalues(row, method=method, weights=weights)
        np.testing.assert_allclose([result[0][index], result[1][index]], expected)
    equal = combine_pvalues(p, method=method, weights=np.ones(2), axis=1)
    default = combine_pvalues(p, method=method, axis=1)
    np.testing.assert_allclose(equal, default)


@pytest.mark.parametrize("method", ["cauchy", "stouffer"])
@pytest.mark.parametrize("weights", [[0., 0.], [-1., 2.], [1., np.nan], [1., np.inf], [1.]])
def test_invalid_weights_raise(method, weights):
    with pytest.raises(ValueError):
        combine_pvalues([[.1, .2], [.3, .4]], method=method, weights=weights, axis=1)


def test_fisher_rejects_weights_and_combination_rejects_empty_input():
    with pytest.raises(ValueError, match="weights"):
        combine_pvalues([.1, .2], method="fisher", weights=[1., 1.])
    with pytest.raises(ValueError, match="at least one"):
        combine_pvalues([])


@pytest.mark.parametrize("alpha", [0., 1., -.1, 1.1])
def test_alpha_interval_boundaries(alpha):
    with pytest.raises(ValueError, match="alpha"):
        adjust_pvalues([.1, .2], alpha=alpha)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_family_and_interpretation_guidance_is_present(language):
    path = Path(__file__).resolve().parents[2] / f"docs/{language}/models/multiple-testing.md"
    text = path.read_text(encoding="utf-8")
    assert "axis=None" in text and "axis=1" in text
    assert "nan_policy" in text and "[0, 1]" in text
    if language == "en":
        assert "does **not** automatically control" in text
        assert "not “proved true.”" in text
        assert "**among all rejections**" in text
        assert "Prespecify Cauchy/Stouffer weights" in text
    else:
        assert "**不会**自动控制" in text
        assert "并不表示“已证明为真”" in text
        assert "**全部拒绝中**" in text
        assert "权重应在查看单项 P 值之前确定" in text
