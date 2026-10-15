"""Protect ordered learner workflows and their numerical transformations.

Structural prose checks do not establish statistical validity. Small independent
examples below exercise rank direction, clipping, unsorting, and family axes.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from statgpu.inference import adjust_pvalues

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('page,names', [
    ('multiple-testing-fwer', ['multiple-testing-fwer', 'multiple-testing-bonferroni',
                               'multiple-testing-hochberg']),
    ('multiple-testing-fdr', ['multiple-testing-fdr', 'multiple-testing-by']),
    ('pvalue-combination', ['pvalue-combination-fisher', 'pvalue-combination-stouffer',
                            'pvalue-combination-cauchy']),
])
def test_every_method_explains_numbered_flow_before_its_runnable_example(language, page, names):
    text = (ROOT / f'docs/{language}/models/testing/{page}.md').read_text()
    previous = 0
    for name in names:
        start = text.index(f'<!-- api-example: {name} -->')
        introduction = text[previous:start]
        numbers = [int(n) for n in re.findall(r'^([1-9])\. ', introduction, re.MULTILINE)]
        assert len(numbers) >= 4, (language, name)
        assert numbers == list(range(1, len(numbers) + 1)), (language, name, numbers)
        previous = text.index(f'<!-- example-end: {name} -->')


@pytest.mark.parametrize('method,expected', [
    ('bonferroni', [1.0, .04, .12, .08]),
    ('holm', [.9, .04, .06, .06]),
    ('hochberg', [.9, .04, .06, .06]),
    ('bh', [.9, .04, .04, .04]),
    ('by', [1., 1/12, 1/12, 1/12]),
])
def test_algorithm_flow_unsorts_clips_and_keeps_distinct_families(method, expected):
    # Unsorted values, unequal adjacent raw factors, and a cap at one.
    values = np.array([[.9, .01, .03, .02], [1., 1., 1., 1.]])
    reject, adjusted = adjust_pvalues(values, method=method, alpha=.05, axis=1, backend='numpy')
    np.testing.assert_allclose(adjusted, [expected, np.ones(4)], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(reject, np.asarray([expected, np.ones(4)]) <= .05)
    assert adjusted.shape == values.shape


@pytest.mark.parametrize('method', ['fisher', 'stouffer', 'cauchy'])
@pytest.mark.parametrize('endpoint', [0., 1.])
def test_combination_flow_clips_inputs_before_transforming(method, endpoint):
    from scipy import stats

    from statgpu.inference import combine_pvalues

    p = np.array([endpoint, .25, .75])
    statistic, probability = combine_pvalues(p, method=method, backend='numpy')
    if method == 'fisher':
        safe = np.clip(p, np.finfo(np.float64).tiny, 1.)
        expected = -2 * np.log(safe).sum()
        expected_probability = stats.chi2.sf(expected, 2 * len(p))
    else:
        eps = np.finfo(np.float64).eps
        safe = np.clip(p, eps, 1. - eps)
        if method == 'stouffer':
            expected = stats.norm.ppf(1. - safe).sum() / np.sqrt(len(p))
            expected_probability = stats.norm.sf(expected)
        else:
            expected = np.mean(np.tan((.5 - safe) * np.pi))
            expected_probability = np.clip(.5 - np.arctan(expected) / np.pi, 0., 1.)
    np.testing.assert_allclose(statistic, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(probability, expected_probability, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('alias,canonical', [('cpu', 'numpy'), ('cuda', 'cupy'), ('gpu', 'cupy')])
def test_documented_backend_aliases_resolve_before_input_conversion(alias, canonical):
    from statgpu.backends import _resolve_backend

    assert _resolve_backend(alias, np.array([.1, .2])) == canonical
