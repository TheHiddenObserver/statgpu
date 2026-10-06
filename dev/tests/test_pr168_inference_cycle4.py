"""Fresh PR168 inference documentation checks and unfixed-runtime regressions.

CPU execution only. Strict expected failures state intended contracts instead
of turning existing numerical/model-reconstruction bugs into passing behavior.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from statgpu.inference import combine_pvalues
from statgpu.linear_model import (
    GammaRegression,
    LogisticRegression,
    NegativeBinomialRegression,
    PenalizedGeneralizedLinearModel,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
    PoissonRegression,
    TweedieRegression,
)

ROOT = Path(__file__).resolve().parents[2]


class _WeightScaleChangedCombination(AssertionError):
    """A common finite weight scale changed the combination result."""


class _PenalizedOracleRefit(AssertionError):
    """Oracle inference differs from the unpenalized active-set reference."""


class _LostOracleFamilyParameter(AssertionError):
    """Oracle reconstruction replaced a requested family-specific setting."""


def _example(language, page, marker):
    text = (ROOT / f'docs/{language}/guides/{page}.md').read_text()
    match = re.search(r'<!-- ' + re.escape(marker) + r' -->\s*```python\n(.*?)```',
                      text, flags=re.DOTALL)
    assert match is not None, marker
    scope = {'__name__': '__doc_example__'}
    exec(compile(match[1], f'{language}/{page}:{marker}', 'exec'), scope)  # noqa: S102
    return scope


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_scaled_weight_example_and_rejected_invalid_vectors(language):
    scope = _example(language, 'multiple-testing-combine-pvalues',
                     'safety-example: scaled-combination-weights')
    normalize = scope['relative_weights']
    np.testing.assert_array_equal(normalize([1e308, 1e308]), [1, 1])
    np.testing.assert_array_equal(normalize([0, 1e308]), [0, 1])
    for bad in ([], [[1, 2]], 1, [0, 0], [-1, 2], [np.nan, 1], [np.inf, 1]):
        with pytest.raises(ValueError):
            normalize(bad)
    p = scope['p']
    scaled = scope['weights']
    z = stats.norm.isf(p).sum() / np.sqrt(2)
    expected = {
        'stouffer': (z, stats.norm.sf(z)),
        'cauchy': (np.mean(np.tan(np.pi * (.5 - p))),
                   .5 - np.arctan(np.mean(np.tan(np.pi * (.5 - p)))) / np.pi),
    }
    for method, reference in expected.items():
        np.testing.assert_allclose(
            combine_pvalues(p, method=method, weights=scaled, backend='numpy'),
            reference, rtol=1e-13, atol=1e-15,
        )


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_explicit_logistic_diagnostic_refit_preserves_unpenalized_score(language):
    scope = _example(language, 'penalized-glm-inference',
                     'inference-example: explicit-logistic-diagnostic')
    refit = scope['refit']
    np.testing.assert_array_equal(scope['active'], [0, 1])
    params = np.r_[refit.intercept_, refit.coef_]
    np.testing.assert_allclose(params, [.6262858766, 2.2952593207, -1.0373981120],
                               atol=1e-8, rtol=1e-8)
    design = np.column_stack([np.ones(len(scope['y'])), scope['X'][:, scope['active']]])
    probability = 1 / (1 + np.exp(-(design @ params)))
    np.testing.assert_allclose(design.T @ (probability - scope['y']), 0, atol=1e-7)
    assert refit.get_params()['C'] == 0
    assert scope['selection'] is not refit


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_weighted_poisson_example_is_self_contained(language):
    scope = _example(language, 'penalized-glm-inference',
                     'inference-example: weighted-poisson-m-estimation')
    model = scope['model']
    assert model.inference_requested_method_ == 'auto'
    assert model.inference_resolved_method_ == 'm_estimation'
    assert model.inference_target_ == 'penalized_estimating_equation'
    assert model._inference_result.params.shape == (3,)
    assert np.isfinite(model._inference_result.conf_int).all()
    assert model._inference_result.cov_type == 'hc0'


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_bilingual_oracle_and_bootstrap_boundaries(language):
    page = (ROOT / f'docs/{language}/guides/penalized-glm-inference.md').read_text()
    modes = (ROOT / f'docs/{language}/guides/inference-modes.md').read_text()
    for token in ('`C=1`', '`alpha=1`', '`1.5`', '`device="auto"`', '`NaN`',
                  '`ddof=1`', 'explicit-logistic-diagnostic'):
        assert token in page
    assert 'current-non-gaussian-oracle-limitation' in modes
    if language == 'en':
        assert 'no plus-one correction' in page
        assert 'null-imposed' in page
        assert 'not available for L2' in page
    else:
        assert '不采用加一修正' in page
        assert '强制原假设' in page
        assert '不适用于 L2' in page
        assert '解析权重' not in modes


def test_combine_help_warns_about_finite_weight_overflow():
    help_text = inspect.getdoc(combine_pvalues)
    assert 'overflow' in help_text
    assert 'largest positive entry' in help_text


def test_bootstrap_reporting_matches_draws_and_not_normal_tail(monkeypatch):
    draws = []
    original = PenalizedLinearRegression.fit

    def record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        draws.append(self._params.copy())
        return result

    monkeypatch.setattr(PenalizedLinearRegression, 'fit', record)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 2))
    y = .5 + x @ np.array([.6, -.2]) + rng.normal(scale=.4, size=40)
    model = PenalizedGeneralizedLinearModel(
        loss='squared_error', penalty='l1', alpha=.1, device='cpu',
        solver='fista', compute_inference=True, inference_method='bootstrap',
    )
    # Bootstrap controls are attributes on the generic estimator; typed Lasso
    # exposes them as constructor parameters. Keep this probe lightweight.
    model.n_bootstrap = 9
    model.bootstrap_random_state = 7
    model.fit(x, y)
    boot = np.asarray(draws)
    assert boot.shape == (9, 3)
    result = model._inference_result
    np.testing.assert_allclose(result.bse, boot.std(axis=0, ddof=1))
    np.testing.assert_allclose(result.conf_int, np.quantile(boot, [.025, .975], axis=0).T)
    expected_p = np.minimum(1, 2 * np.minimum((boot <= 0).mean(0), (boot >= 0).mean(0)))
    np.testing.assert_allclose(result.pvalues, expected_p)
    np.testing.assert_allclose(result.statistic, model._params / (result.bse + 1e-30))


@pytest.mark.xfail(strict=True, raises=_WeightScaleChangedCombination,
                   reason='Issue #228: finite combination weights overflow their normalization sum')
@pytest.mark.parametrize('method', ['cauchy', 'stouffer'])
@pytest.mark.parametrize('axis', [None, 1])
def test_combination_is_invariant_to_large_common_weight_scale(method, axis):
    from statgpu.inference import _multiple_testing as implementation

    p = np.array([.01, .1]) if axis is None else np.array([[.01, .1], [.04, .2]])
    normalized = []
    original = implementation._validate_weights

    def record(weights, m, backend):
        result = original(weights, m, backend)
        normalized.append((np.asarray(weights).copy(), np.asarray(result).copy()))
        return result

    with patch.object(implementation, '_validate_weights', side_effect=record), np.errstate(over='ignore', invalid='ignore'):
        actual = combine_pvalues(p, method=method, weights=[1e308, 1e308], axis=axis)
    if method == 'cauchy':
        statistic = np.mean(np.tan(np.pi * (.5 - p)), axis=axis)
        expected = (statistic, .5 - np.arctan(statistic) / np.pi)
    else:
        statistic = np.sum(stats.norm.isf(p), axis=axis) / np.sqrt(2)
        expected = (statistic, stats.norm.sf(statistic))
    assert all(np.isfinite(value).all() for value in expected)
    assert all(np.shape(observed) == np.shape(reference)
               for observed, reference in zip(actual, expected))
    if all(np.allclose(observed, reference, rtol=1e-12, atol=1e-14)
           for observed, reference in zip(actual, expected)):
        return

    # NaN Stouffer output alone is not enough: the known mechanism must have
    # executed, with finite huge inputs normalized to exactly zero weights.
    assert len(normalized) == 1
    for supplied, resolved in normalized:
        np.testing.assert_array_equal(supplied, [1e308, 1e308])
        np.testing.assert_array_equal(resolved, [0., 0.])
    signature = (0., .5) if method == 'cauchy' else (np.nan, np.nan)
    for observed, value in zip(actual, signature):
        np.testing.assert_array_equal(observed, np.full(np.shape(observed), value))
    raise _WeightScaleChangedCombination(
        f'{method} followed the verified overflowing-sum normalization path'
    )


@pytest.mark.xfail(strict=True, raises=_PenalizedOracleRefit,
                   reason='Issue #227: oracle reconstruction retains LogisticRegression default C=1')
@pytest.mark.parametrize('penalty', ['scad', 'mcp'])
def test_logistic_oracle_targets_unpenalized_active_set(penalty):
    rng = np.random.default_rng(48)
    x = rng.normal(size=(200, 2))
    y = rng.binomial(1, 1 / (1 + np.exp(-(.6 + x @ np.array([2., -1.])))))
    model = PenalizedLogisticRegression(
        penalty=penalty, alpha=.01, solver='fista', device='cpu',
        compute_inference=True, inference_method='oracle', max_iter=10000, tol=1e-10,
    ).fit(x, y)
    active = np.flatnonzero(np.asarray(model._inference_result.metadata['active_set']))
    reference = LogisticRegression(C=0, device='cpu', max_iter=10000, tol=1e-10).fit(x[:, active], y)
    observed = model._inference_result.params[np.r_[0, active + 1]]
    expected = np.r_[reference.intercept_, reference.coef_]
    assert np.shape(observed) == np.shape(expected)
    assert np.isfinite(observed).all()
    if np.allclose(observed, expected, rtol=1e-6, atol=1e-7):
        return
    default_child = LogisticRegression(C=1, device='cpu').fit(x[:, active], y)
    known_bad = np.r_[default_child.intercept_, default_child.coef_]
    np.testing.assert_allclose(observed, known_bad, rtol=1e-6, atol=1e-7,
                               err_msg='A different oracle corruption is not issue #227')
    raise _PenalizedOracleRefit('logistic oracle exactly retained the default C=1 child')


@pytest.mark.xfail(strict=True, raises=_PenalizedOracleRefit,
                   reason='Issue #227: oracle reconstruction drops explicit Newton and runs penalized IRLS')
@pytest.mark.parametrize('penalty', ['scad', 'mcp'])
def test_poisson_oracle_targets_unpenalized_active_set(penalty):
    rng = np.random.default_rng(59)
    x = rng.normal(size=(100, 2))
    y = rng.poisson(np.exp(.2 + x @ np.array([.3, -.2])))
    model = PenalizedPoissonRegression(
        penalty=penalty, alpha=.001, solver='fista', device='cpu',
        compute_inference=True, inference_method='oracle', max_iter=10000, tol=1e-10,
    ).fit(x, y)
    reference = PoissonRegression(solver='newton', device='cpu', max_iter=1000, tol=1e-8).fit(x, y)
    observed = model._inference_result.params
    expected = np.r_[reference.intercept_, reference.coef_]
    assert np.shape(observed) == np.shape(expected)
    assert np.isfinite(observed).all()
    if np.allclose(observed, expected, rtol=1e-6, atol=1e-7):
        return
    default_child = PoissonRegression(C=1, solver='auto', device='cpu').fit(x, y)
    known_bad = np.r_[default_child.intercept_, default_child.coef_]
    np.testing.assert_allclose(observed, known_bad, rtol=1e-6, atol=1e-7,
                               err_msg='A different oracle corruption is not issue #227')
    raise _PenalizedOracleRefit('Poisson oracle exactly used default positive-C IRLS')


@pytest.mark.xfail(strict=True, raises=_LostOracleFamilyParameter,
                   reason='Issue #227: oracle constructor introspection loses family-specific settings')
@pytest.mark.parametrize('penalty', ['scad', 'mcp'])
@pytest.mark.parametrize('loss,cls,settings,parameter', [
    ('negative_binomial', NegativeBinomialRegression, {'alpha': .25}, 'alpha'),
    ('tweedie', TweedieRegression, {'power': 1.2}, 'power'),
    ('gamma', GammaRegression, {'link': 'inverse_power'}, 'link'),
])
def test_oracle_refit_preserves_family_parameters(monkeypatch, penalty, loss, cls, settings, parameter):
    rng = np.random.default_rng(61)
    x = rng.uniform(.5, 1.5, size=(100, 2))
    y = np.exp(x @ np.array([.4, .2])) * rng.lognormal(0, .2, 100)
    if loss == 'gamma':
        y = 1 / (x @ np.array([.4, .2])) * rng.lognormal(0, .1, 100)
    else:
        x = x - .5
    if loss == 'negative_binomial':
        y = rng.poisson(y)
    seen = []
    original = cls.fit

    def record(self, *args, **kwargs):
        seen.append(self.get_params())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, 'fit', record)
    PenalizedGeneralizedLinearModel(
        loss=loss, penalty=penalty, alpha=.0001, loss_kwargs=settings,
        fit_intercept=False, compute_inference=True, inference_method='oracle',
        device='cpu', solver='fista', max_iter=3000, tol=1e-8,
    ).fit(x, y)
    assert len(seen) == 1
    observed = seen[0][parameter]
    expected = settings[parameter]
    if observed != expected:
        default = inspect.signature(cls).parameters[parameter].default
        assert observed == default, 'An arbitrary family setting is not the default-reset bug'
        raise _LostOracleFamilyParameter(f'oracle {parameter}={observed!r}, requested {expected!r}')
