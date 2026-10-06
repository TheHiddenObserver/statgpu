"""Fresh fifth-pass GLM public API and documented usable workflows.

The two outstanding runtime defects are represented by narrow strict xfails;
passing tests validate the intended contracts and safe reporting alternatives.
"""
from __future__ import annotations

import inspect
import re
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from statgpu import (
    GeneralizedLinearModel,
    PenalizedGeneralizedLinearModel,
    PenalizedGLM_CV,
    PoissonRegression,
)

ROOT = Path(__file__).resolve().parents[2]
GLM_CLASSES = [GeneralizedLinearModel, PenalizedGeneralizedLinearModel, PenalizedGLM_CV]


class _MixedFailedRefit(AssertionError):
    """A rejected fit changed outputs without invalidating the estimator."""


class _BrokenSummaryDelegation(AssertionError):
    """The public summary delegates to an absent final-estimator method."""


def _page(language, directory, name):
    return (ROOT / 'docs' / language / directory / f'{name}.md').read_text()


def _signature(cls):
    signature = inspect.signature(cls)
    parameters = []
    for parameter in signature.parameters.values():
        default = parameter.default
        if isinstance(default, Enum):
            default = default.value
        parameters.append(parameter.replace(annotation=inspect.Parameter.empty, default=default))
    return cls.__name__ + str(signature.replace(parameters=parameters, return_annotation=inspect.Signature.empty))


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('cls', GLM_CLASSES)
def test_generic_glm_reference_has_its_own_complete_live_constructor(language, cls):
    reference = _page(language, 'reference', 'linear-model-api')
    section = reference.split(f'## {cls.__name__}\n', 1)[1].split('\n## ', 1)[0]
    assert _signature(cls) in section
    for parameter in inspect.signature(cls).parameters:
        assert f'| `{parameter}` |' in section
    for name in dir(cls):
        if name.startswith('_'):
            continue
        member = getattr(cls, name)
        if callable(member) or isinstance(inspect.getattr_static(cls, name), property):
            assert name in section + _page(language, 'reference', 'estimator-api'), name


@pytest.mark.parametrize('cls', GLM_CLASSES)
def test_generic_glm_installed_help_covers_all_constructor_parameters(cls):
    doc = inspect.getdoc(cls)
    for name in inspect.signature(cls).parameters:
        assert re.search(r'^' + name + r'\s*:', doc, flags=re.MULTILINE), name
    for method in ['fit(', 'predict(', 'get_params(', 'set_params(', 'adjust_pvalues',
                   'combine_pvalues', 'bootstrap_statistic', 'permutation_test']:
        assert method in doc


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('label', ['glm-poisson', 'glm-formula', 'glm-cv-inference'])
def test_glm_learner_examples_are_independently_executable(language, label):
    if label == 'glm-formula':
        pytest.importorskip('pandas')
        pytest.importorskip('patsy')
    text = _page(language, 'models', 'generalized-linear-model')
    match = re.search(r'<!-- learner-example: ' + label + r' -->\s*```python\n(.*?)```', text, re.DOTALL)
    assert match
    ns = {}
    exec(compile(match.group(1), f'{language}/{label}', 'exec'), ns)  # noqa: S102
    model = ns['model']
    if label == 'glm-poisson':
        assert model._conf_int.shape == (3, 2)
        np.testing.assert_allclose(model.coef_, [.402, -.327], atol=.0005)
        assert ns['heldout_loss'] == pytest.approx(.556, abs=.0005)
        weights, X, y = ns['weights'], ns['X'][:180], ns['y'][:180]
        residual = weights * (model.predict(X) - y) / weights.sum()
        np.testing.assert_allclose(X.T @ residual, 0, atol=1e-7)
        assert abs(residual.sum()) < 1e-7
    elif label == 'glm-cv-inference':
        assert ns['report']['method'] == 'm_estimation'
        assert np.isfinite(model.estimator_._bse).all()
        assert model.estimator_._bse.shape == (3,)
    else:
        assert ns['prediction'].shape == (5,)
        assert np.isfinite(ns['prediction']).all()


def _problem():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(30, 2))
    y = rng.poisson(np.exp(.7 + .3 * X[:, 0]))
    return X, y, np.linspace(.5, 2., len(y))


def _model(cls, solver):
    kwargs = {'family': 'poisson'} if cls is GeneralizedLinearModel else {}
    return cls(**kwargs, C=0, device='cpu', solver=solver)


def _snapshot(model, X):
    return (model.predict(X).copy(), model.loglikelihood, model.aic, model.bic)


def _assert_atomic_failure(model, X, before, known_bad=None):
    if not model._fitted:
        with pytest.raises(RuntimeError):
            model.predict(X)
        return  # Explicitly invalidating a rejected fit is an acceptable policy.
    after = _snapshot(model, X)
    for actual, expected in zip(after, before):
        assert np.shape(actual) == np.shape(expected)
        assert np.isfinite(actual).all()
    if all(np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
           for actual, expected in zip(after, before)):
        return
    assert known_bad is not None, 'A new failed-refit behavior is not the known mixed-state defect'
    for actual, expected in zip(after, known_bad):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12,
                                   err_msg='Only the documented metadata mixture is issue #237')
    raise _MixedFailedRefit('failed fit left the verified prediction/row-count mixture')


@pytest.mark.xfail(strict=True, raises=_MixedFailedRefit, reason='Issue #237: ordinary non-smooth-wrapper refits retain mixed state')
@pytest.mark.parametrize('cls', [GeneralizedLinearModel, PoissonRegression])
@pytest.mark.parametrize('solver', ['auto', 'irls', 'fista'])
@pytest.mark.parametrize('failure', ['row_count', 'formula'])
def test_failed_ordinary_glm_refit_should_preserve_or_invalidate_complete_state(cls, solver, failure):
    X, y, weights = _problem()
    model = _model(cls, solver).fit(X, y, sample_weight=weights)
    before = _snapshot(model, X)
    previous_intercept = model.intercept_
    previous_params = model._params.copy()
    if failure == 'formula':
        pd = pytest.importorskip('pandas')
        pytest.importorskip('patsy')
        frame = pd.DataFrame({'y': np.r_[-1., y[1:]], 'x1': X[:, 0], 'x2': X[:, 1]})
        with pytest.raises(ValueError, match='poisson response requires'):
            model.fit(formula='y ~ 0 + x1 + x2', data=frame)
        known_bad = (before[0] * np.exp(-previous_intercept), *before[1:])
    else:
        with pytest.raises(ValueError, match='Response length must match'):
            model.fit(X[:5], y[:4])
        mixed_ll = before[1] * 5 / len(y)
        k = len(previous_params)
        known_bad = (before[0], mixed_ll, -2*mixed_ll + 2*k,
                     -2*mixed_ll + k*np.log(5))
    if model._fitted:
        np.testing.assert_array_equal(model._params, previous_params)
    _assert_atomic_failure(model, X, before, known_bad)


@pytest.mark.parametrize('cls', [GeneralizedLinearModel, PoissonRegression])
@pytest.mark.parametrize('solver', ['newton', 'lbfgs'])
def test_existing_smooth_glm_failed_refit_preserves_prior_outputs(cls, solver):
    X, y, weights = _problem()
    model = _model(cls, solver).fit(X, y, sample_weight=weights)
    before = _snapshot(model, X)
    with pytest.raises(ValueError, match='Response length must match'):
        model.fit(X[:5], y[:4])
    _assert_atomic_failure(model, X, before)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_failed_fit_warning_and_fresh_estimator_workflow(language):
    X, y, weights = _problem()
    old = _model(PoissonRegression, 'auto').fit(X, y, sample_weight=weights)
    with pytest.raises(ValueError, match='Response length must match'):
        old.fit(X[:5], y[:4])
    fresh = _model(PoissonRegression, 'auto').fit(X, y, sample_weight=weights)
    assert fresh._nobs == len(y)
    assert fresh.loglikelihood == pytest.approx(-8.498153093088963, abs=1e-9)
    assert fresh.aic == pytest.approx(22.996306186177925, abs=1e-9)
    text = _page(language, 'reference', 'linear-model-api')
    assert 'failed-ordinary-glm-refits' in _page(language, 'models', 'generalized-linear-model')
    assert ('new row counts' in text if language == 'en' else '新的观测数' in text)
    assert 'Failed auto/IRLS/FISTA refits' in inspect.getdoc(GeneralizedLinearModel)


@pytest.fixture
def inferred_cv():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 2))
    y = rng.poisson(np.exp(.2 + .3 * X[:, 0]))
    return PenalizedGLM_CV(loss='poisson', penalty='l2', alpha_grid=[.05, .2],
                          cv=2, device='cpu', compute_inference=True).fit(X, y)


@pytest.mark.xfail(strict=True, raises=_BrokenSummaryDelegation, reason='Issue #154: generic CV summary delegates to missing final method')
def test_successful_generic_cv_inference_should_have_usable_summary(inferred_cv):
    # Setup/fit failures remain fixture errors; only the known delegation is xfail.
    assert inferred_cv._inference_result.method == 'm_estimation'
    assert np.isfinite(inferred_cv._bse).all()
    try:
        inferred_cv.summary()
    except AttributeError as error:
        if str(error) != "'PenalizedGeneralizedLinearModel' object has no attribute 'summary'":
            raise
        raise _BrokenSummaryDelegation(str(error)) from error


def test_generic_glm_prediction_and_score_semantics_and_summary_return(capsys):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2))
    y = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    ordinary = GeneralizedLinearModel(family='binomial', C=0, device='cpu').fit(X, y)
    probability = ordinary.predict(X)
    assert np.all((probability > 0) & (probability < 1))
    report = ordinary.summary()
    assert isinstance(report, str) and 'inference not computed' in report
    assert capsys.readouterr().out == ''
    generic = PenalizedGeneralizedLinearModel(loss='logistic', penalty='l2', alpha=.1,
                                             device='cpu').fit(X, y)
    labels = generic.predict(X)
    assert set(np.unique(labels)) <= {0., 1.}
    expected = 1 - np.sum((y-labels)**2) / np.sum((y-y.mean())**2)
    assert generic.score(X, y) == pytest.approx(expected)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_elasticnet_diagnostic_example_enables_the_inference_it_interprets(language):
    text = _page(language, 'models', 'elastic-net')
    match = re.search(r'<!-- learner-example: elasticnet-weighted-score -->\s*```python\n(.*?)```', text, re.DOTALL)
    ns = {}
    exec(compile(match.group(1), 'elasticnet-diagnostics', 'exec'), ns)  # noqa: S102
    assert ns['model'].compute_inference is True
    assert ns['model'].rsquared is not None
    assert ns['weighted_r2'] == pytest.approx(.1798437899836577, abs=1e-9)
