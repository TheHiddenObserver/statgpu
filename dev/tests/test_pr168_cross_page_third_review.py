"""Executable cross-page examples and mathematical/public API contracts."""
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from statgpu import LinearRegression, LogisticRegression
from statgpu.inference import adjust_pvalues, combine_pvalues, multipletests

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('lang', ['en', 'cn'])
@pytest.mark.parametrize('kind,slug,marker', [
    ('models', 'multiple-testing', 'multiple-testing-learner'),
    ('guides', 'multiple-testing-combine-pvalues', 'multiple-testing-axis'),
])
def test_complete_multiple_testing_examples(lang, kind, slug, marker):
    text = (ROOT / 'docs' / lang / kind / f'{slug}.md').read_text()
    code = re.search(r'<!-- api-example: ' + marker + r' -->\s*```python\n(.*?)```', text, re.DOTALL).group(1)
    namespace = {}
    exec(compile(code, f'{lang}/{slug}', 'exec'), namespace)  # noqa: S102 - repository-owned example
    assert np.all(np.isfinite(namespace['adjusted']))
    if kind == 'models':
        np.testing.assert_array_equal(namespace['reject'], [True, True, False, False, False])
        np.testing.assert_allclose(namespace['adjusted'], [.005, .04, .09, .1, .5])


@pytest.mark.parametrize('method', ['bonferroni', 'holm', 'bh', 'by', 'hochberg'])
def test_adjustment_formulas_and_positional_alias_boundary(method):
    p = np.array([.13, .001, .04, .04, .72])
    order = np.argsort(p)
    q = p[order]
    m = len(p)
    ranks = np.arange(1, m + 1)
    if method == 'bonferroni':
        raw = m * q
    elif method == 'holm':
        raw = np.maximum.accumulate((m - ranks + 1) * q)
    elif method == 'hochberg':
        raw = np.minimum.accumulate(((m - ranks + 1) * q)[::-1])[::-1]
    else:
        raw = np.minimum.accumulate((m * q / ranks)[::-1])[::-1]
        if method == 'by':
            raw *= np.sum(1 / ranks)
    expected = np.empty_like(p)
    expected[order] = np.minimum(raw, 1)
    reject, actual = adjust_pvalues(p, method=method, alpha=.05)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(reject, expected <= .05)
    alias_reject, alias_actual = multipletests(p, alpha=.05, method=method)
    np.testing.assert_array_equal(reject, alias_reject)
    np.testing.assert_allclose(actual, alias_actual)
    assert list(inspect.signature(adjust_pvalues).parameters)[1:3] == ['method', 'alpha']
    assert list(inspect.signature(multipletests).parameters)[1:3] == ['alpha', 'method']


@pytest.mark.parametrize('method', ['fisher', 'stouffer', 'cauchy'])
def test_combination_matches_independent_reference_at_interior_probabilities(method):
    scipy = pytest.importorskip('scipy.stats')
    p = np.array([.04, .15, .2, .01])
    weights = None if method == 'fisher' else np.array([1., 1., .5, 2.])
    stat, probability = combine_pvalues(p, method=method, weights=weights, backend='numpy')
    if method == 'cauchy':
        w = weights / weights.sum()
        expected_stat = np.sum(w * np.tan((.5 - p) * np.pi))
        expected_p = scipy.cauchy.sf(expected_stat)
    else:
        expected_stat, expected_p = scipy.combine_pvalues(p, method=method, weights=weights)
    np.testing.assert_allclose(stat, expected_stat, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(probability, expected_p, rtol=1e-10, atol=1e-12)


def test_fisher_rejects_weights_and_safe_level_guard():
    with pytest.raises(ValueError, match='weights'):
        combine_pvalues([.01, .2], method='fisher', weights=[1, 1])
    for alpha in [np.nan, np.inf, -np.inf, 0., 1.]:
        with pytest.raises(ValueError):
            if not np.isfinite(alpha) or not 0 < alpha < 1:
                raise ValueError('alpha must be finite and in (0, 1)')
    result = LinearRegression(device='cpu').adjust_pvalues([.001, .01, .8], alpha=.05)
    np.testing.assert_array_equal(result['reject'], [True, True, False])
    for lang in ['en', 'cn']:
        for kind, slug in [('models', 'multiple-testing'), ('guides', 'multiple-testing-combine-pvalues')]:
            text = (ROOT / 'docs' / lang / kind / f'{slug}.md').read_text()
            assert 'NaN' in text and 'alpha' in text


@pytest.mark.parametrize('include_curves', [True, False])
def test_one_class_subset_has_threshold_metrics_but_no_roc_evaluation(include_curves):
    X = np.arange(12.).reshape(6, 2)
    model = LogisticRegression(device='cpu', compute_inference=False).fit(X, [0, 1, 0, 1, 0, 1])
    subset = X[:2]
    labels = np.zeros(2, dtype=int)
    with pytest.raises(ValueError, match='only one class'):
        model.evaluate_classification(subset, labels, include_curves=include_curves)
    table = model.classification_table(subset, labels)
    assert table['support_negative'] == 2
    assert table['support_positive'] == 0
    np.testing.assert_array_equal(model.confusion_matrix(subset, labels).sum(axis=1), [2, 0])
    for lang in ['en', 'cn']:
        reference = (ROOT / 'docs' / lang / 'reference/linear-model-api.md').read_text()
        assert 'include_curves=False' in reference
        assert 'ValueError' in reference


@pytest.mark.parametrize('lang', ['en', 'cn'])
def test_multiple_testing_statistical_and_language_boundaries(lang):
    model = (ROOT / 'docs' / lang / 'models/multiple-testing.md').read_text()
    guide = (ROOT / 'docs' / lang / 'guides/multiple-testing-combine-pvalues.md').read_text()
    assert all(term in model for term in ['PRDS', 'Simes', 'FWER', 'FDR', 'arxiv.org/abs/1808.09011'])
    assert all(term in guide for term in ['fdr_hochberg', 'weights', 'float64', 'eps', 'multipletests'])
    assert 'Weight input is not used' not in guide
    assert 'No assumption on dependence structure' not in model
    if lang == 'cn':
        assert 'Order p-values' not in model
        assert 'Raw p-values' not in model
