"""Execute safe resampling/provenance journeys from the PR168 third review.

These are CPU contract checks. Uninitialized-memory behavior is reproduced in a
separate investigation, never treated as a desired output in these tests.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from statgpu import LinearRegression
from statgpu.inference import bootstrap_statistic, permutation_test
from statgpu.linear_model import ElasticNet, Lasso, PenalizedLinearRegression

ROOT = Path(__file__).resolve().parents[2]


class _MissingPostSelectionProvenance(AssertionError):
    """Post-selection inference did not publish its method and target."""


def _run_example(language, guide, marker):
    text = (ROOT / 'docs' / language / 'guides' / guide).read_text()
    match = re.search(r'<!-- ' + re.escape(marker) + r' -->\s*```python\n(.*?)```',
                      text, flags=re.DOTALL)
    assert match is not None
    scope = {'__name__': '__doc_example__'}
    exec(compile(match[1], f'{language}/{guide}:{marker}', 'exec'), scope)  # noqa: S102
    return scope


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_numeric_resampling_label_example_validates_before_dispatch(language):
    scope = _run_example(language, 'inference-api.md',
                         'safety-example: validated-resampling-labels')
    result = scope['result']
    assert result.observed == pytest.approx(0.903193542115137)
    assert result.pvalue == pytest.approx(0.12)
    assert np.isfinite(result.samples).all()
    validate = scope['validated_labels']
    labels = validate([0, 0, 1, 1], 4)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    for bad in ([0, 0, np.nan, np.nan], [0, 0, np.inf, 1],
                [0, 0, -np.inf, 1], ['a', 'a', 'b', 'b']):
        with pytest.raises(ValueError, match='finite numeric codes'):
            validate(bad, 4)
    for bad in (0, [[0, 0], [1, 1]], [0, 1]):
        with pytest.raises(ValueError, match='one group label per observation'):
            validate(bad, 4)


@pytest.mark.parametrize('use_estimator', [False, True])
@pytest.mark.parametrize('strategy', ['stratified', 'grouped'])
def test_valid_grouped_permutations_preserve_response_multiset(use_estimator, strategy):
    x = np.arange(1.0, 7.0)
    labels = np.array([0, 0, 0, 1, 1, 1])
    # The full-response sum cannot change under any genuine permutation.
    def statistic(_X, y):
        return y.sum(axis=-1)

    function = (LinearRegression(device='cpu').permutation_test
                if use_estimator else permutation_test)
    options = {'strata' if strategy == 'stratified' else 'groups': labels}
    result = function(statistic, x, x, strategy=strategy, n_resamples=41,
                      random_state=7, backend='numpy', **options)
    np.testing.assert_array_equal(result.samples, np.full(41, x.sum()))
    assert result.pvalue == 1.0


@pytest.mark.parametrize('use_estimator', [False, True])
@pytest.mark.parametrize('strategy', ['iid', 'stratified', 'cluster', 'block'])
def test_valid_resampling_keeps_explicit_arrays_aligned(use_estimator, strategy):
    x = np.arange(6.0)
    y = 3 * x + 1
    labels = np.array([0, 0, 1, 1, 2, 2])

    def statistic(xs, ys):
        return np.max(np.abs(ys - 3 * xs - 1), axis=-1)

    options = {'stratified': {'strata': labels}, 'cluster': {'clusters': labels},
               'block': {'block_size': 2}, 'iid': {}}[strategy]
    function = (LinearRegression(device='cpu').bootstrap_statistic
                if use_estimator else bootstrap_statistic)
    result = function(statistic, x, y, strategy=strategy, n_resamples=41,
                      random_state=7, backend='numpy', **options)
    np.testing.assert_array_equal(result.samples, np.zeros(41))
    assert result.confidence_interval == (0.0, 0.0)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_post_selection_example_preserves_prediction_and_reports_refit(language):
    scope = _run_example(language, 'inference-modes.md',
                         'inference-example: post-selection')
    model, x, y = scope['model'], scope['X'], scope['y']
    result = model._inference_result
    assert result.method == 'post_selection_ols'
    selected = np.asarray(result.metadata['selected_feature_indices'], dtype=int)
    design = np.column_stack([np.ones(len(x)), x[:, selected]])
    expected = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(result.params[np.r_[0, selected + 1]], expected,
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.predict(x), model.intercept_ + x @ model.coef_)
    inactive = np.setdiff1d(np.arange(x.shape[1]), selected) + 1
    assert inactive.size == 1
    np.testing.assert_array_equal(result.params[inactive], 0.0)
    np.testing.assert_array_equal(result.pvalues[inactive], 1.0)
    assert not np.allclose(np.r_[model.intercept_, model.coef_], result.params)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_missing_label_and_provenance_caveats_are_in_both_user_journeys(language):
    guide = (ROOT / f'docs/{language}/guides/inference-api.md').read_text()
    reference = (ROOT / f'docs/{language}/reference/estimator-api.md').read_text()
    modes = (ROOT / f'docs/{language}/guides/inference-modes.md').read_text()
    penalized = (ROOT / f'docs/{language}/guides/penalized-glm-inference.md').read_text()
    for text in (guide, reference):
        assert 'NaN' in text
    assert 'validate-resampling-labels-before-calling' in reference
    assert ('## Validate resampling labels before calling' in guide
            if language == 'en' else 'validate-resampling-labels-before-calling' in guide)
    for text in (modes, penalized):
        assert 'model._inference_result.method' in text
        assert '`None`' in text
    if language == 'en':
        assert 'approximate inverse design precision matrix' not in modes
        assert 'inverse of the design Gram/covariance matrix' in modes
        assert 'equal-tail test' in guide
    else:
        assert '精度矩阵的近似逆' not in modes
        assert '协方差矩阵的近似逆' in modes
        assert '等尾检验' in guide


@pytest.mark.parametrize('function', [bootstrap_statistic, permutation_test,
                                     LinearRegression.bootstrap_statistic,
                                     LinearRegression.permutation_test])
def test_public_resampling_help_warns_about_missing_labels(function):
    text = inspect.getdoc(function)
    assert 'nonmissing' in text
    assert 'NaN' in text
    assert 'uninitialized' in text


# Intended public provenance contract; do not enshrine today's None fields as a
# passing assertion. The specialized post-selection path skips the publisher.
@pytest.mark.xfail(strict=True, raises=_MissingPostSelectionProvenance,
                   reason='post_selection_ols skips shared public provenance publication')
@pytest.mark.parametrize('cls', [Lasso, ElasticNet, PenalizedLinearRegression])
def test_post_selection_publishes_the_same_method_and_target_as_result(cls):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((80, 3))
    y = 1 + x @ np.array([1.5, 0, -0.8]) + rng.normal(scale=0.4, size=80)
    options = {'penalty': 'l1'} if cls is PenalizedLinearRegression else {}
    model = cls(alpha=0.1, solver='fista', compute_inference=True,
                inference_method='post_selection_ols', device='cpu', **options).fit(x, y)
    published_method = model.inference_method_
    result_method = model._inference_result.method
    published_target = model.inference_target_
    assert result_method == 'post_selection_ols'
    result = model._inference_result
    assert np.isfinite(result.params).all()
    assert np.isfinite(result.conf_int).all()
    active = np.flatnonzero(np.abs(model.coef_) > 1e-10)
    design = np.column_stack([np.ones(len(y)), x[:, active]])
    expected = np.zeros(1 + x.shape[1])
    expected[np.r_[0, active + 1]] = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(result.params, expected, atol=1e-9)
    if published_method == result_method and published_target == 'active_set_refit_coefficient':
        return
    assert published_method is None and published_target is None, (
        'Only absent provenance fields belong to issue #215, not a wrong populated label'
    )
    raise _MissingPostSelectionProvenance('Successful post-selection result omitted shared provenance')



@pytest.mark.parametrize('weighted', [False, True])
def test_non_gaussian_hc1_uses_observation_count_and_includes_intercept(weighted):
    from statgpu.linear_model import PenalizedPoissonRegression

    rng = np.random.default_rng(72)
    X = rng.normal(size=(60, 2))
    y = rng.poisson(np.exp(0.2 + 0.3 * X[:, 0] - 0.2 * X[:, 1]))
    weights = np.linspace(0.4, 2.0, len(y)) if weighted else None
    options = {'penalty': 'l2', 'alpha': 0.2, 'solver': 'lbfgs',
               'compute_inference': True, 'inference_method': 'auto', 'device': 'cpu'}
    hc0 = PenalizedPoissonRegression(cov_type='hc0', **options).fit(
        X, y, sample_weight=weights)
    hc1 = PenalizedPoissonRegression(cov_type='hc1', **options).fit(
        X, y, sample_weight=weights)
    np.testing.assert_allclose(hc0.coef_, hc1.coef_)
    n, k = len(y), X.shape[1] + 1
    np.testing.assert_allclose(hc1._bse ** 2, hc0._bse ** 2 * n / (n - k),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_penalized_guide_separates_hc0_formula_from_hc1_correction(language):
    text = (ROOT / f'docs/{language}/guides/penalized-glm-inference.md').read_text()
    assert '$n/(n-k)$' in text
    assert r'$n\le k$' in text
    assert ('the HC0 covariance is' in text if language == 'en'
            else '则 HC0 协方差为' in text)
