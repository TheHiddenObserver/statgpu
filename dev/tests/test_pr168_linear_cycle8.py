"""Eighth-pass typed GLM/API examples and narrow known-defect guards.

CPU checks verify observable documentation contracts. Torch checks execute on
CPU and do not establish physical CUDA correctness or performance.
"""
from __future__ import annotations

import doctest
import importlib.util
import inspect
import re
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from statgpu import linear_model as lm

ROOT = Path(__file__).resolve().parents[2]
TYPED = [lm.GammaRegression, lm.InverseGaussianRegression,
         lm.NegativeBinomialRegression, lm.TweedieRegression,
         lm.PenalizedLinearRegression, lm.PenalizedLogisticRegression,
         lm.PenalizedPoissonRegression]


def _page(language, directory, name):
    return (ROOT / 'docs' / language / directory / f'{name}.md').read_text()


def _signature(cls):
    signature = inspect.signature(cls)
    parameters = [p.replace(annotation=inspect.Parameter.empty,
                            default=p.default.value if isinstance(p.default, Enum) else p.default)
                  for p in signature.parameters.values()]
    return cls.__name__ + str(signature.replace(
        parameters=parameters, return_annotation=inspect.Signature.empty))


def _load_regression(name):
    path = ROOT / 'dev' / 'tests' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'cycle8_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('cls', TYPED)
def test_typed_glm_reference_covers_actual_constructor_and_all_public_members(language, cls):
    reference = _page(language, 'reference', 'linear-model-api')
    section = reference.split(f'## {cls.__name__}\n', 1)[1].split('\n## ', 1)[0]
    assert _signature(cls) in section
    generic = 'PenalizedGeneralizedLinearModel' if cls.__name__.startswith('Penalized') else 'GeneralizedLinearModel'
    generic_section = reference.split(f'## {generic}\n', 1)[1].split('\n## ', 1)[0]
    assert f'#{generic.lower()}' in section
    help_text = inspect.getdoc(cls)
    for parameter in inspect.signature(cls).parameters:
        assert re.search(r'^' + parameter + r'\s*:', help_text, re.MULTILINE), parameter
        assert parameter in generic_section + section
    shared = _page(language, 'reference', 'estimator-api')
    for name in dir(cls):
        if name.startswith('_'):
            continue
        member = inspect.getattr_static(cls, name)
        if callable(member) or isinstance(member, property):
            assert name in section + generic_section + shared, name
            assert name in help_text, name
    assert 'typed-glm-constructors' in _page(language, 'models', 'generalized-linear-model')


@pytest.mark.parametrize('cls', [lm.ElasticNetCV, lm.LogisticRegressionCV])
def test_cv_installed_help_examples_execute_on_cpu_and_cover_methods(cls):
    document = inspect.getdoc(cls)
    example = doctest.DocTestParser().get_doctest(document, {}, cls.__name__, None, None)
    runner = doctest.DocTestRunner()
    runner.run(example)
    assert runner.failures == 0
    assert runner.tries >= 8
    for parameter in inspect.signature(cls).parameters:
        assert re.search(r'^' + parameter + r'\s*:', document, re.MULTILINE)
    for name in dir(cls):
        if not name.startswith('_') and callable(getattr(cls, name)):
            assert name in document, name
    for name in ['estimator_', 'cv_selected_device_', 'n_iter_', 'cv_results_', 'best_score_']:
        assert name in document


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('label', ['poisson-unpenalized', 'poisson-formula'])
def test_poisson_learner_examples_are_self_contained_and_interpretable(language, label):
    if label == 'poisson-formula':
        pytest.importorskip('pandas')
        pytest.importorskip('patsy')
    page = _page(language, 'models', 'poisson-regression')
    block = re.search(r'<!-- learner-example: ' + label + r' -->\s*```python\n(.*?)```', page, re.DOTALL)
    assert block is not None
    scope = {}
    exec(compile(block[1], f'{language}/{label}', 'exec'), scope)  # noqa: S102
    model = scope['model']
    if label == 'poisson-formula':
        assert scope['prediction'].shape == (5,)
        assert np.isfinite(scope['prediction']).all()
    else:
        assert scope['mean_prediction'].shape == (50,)
        np.testing.assert_allclose(scope['mean_prediction'],
                                   np.exp(model.intercept_ + scope['X_test'] @ model.coef_))
        assert scope['heldout_loss'] == pytest.approx(.963, abs=.0005)
        np.testing.assert_allclose(np.exp(model.coef_), [1.459, .857], atol=.0005)
        assert model._conf_int.shape == (3, 2)
        assert model._nobs == 200
    assert 'y_count' not in page and 'df_new' not in page


@pytest.mark.parametrize('weighted', [False, True])
def test_poisson_sklearn_regularization_mapping_matches_average_objective(weighted):
    sklearn = pytest.importorskip('sklearn.linear_model')
    rng = np.random.default_rng(28)
    X = rng.normal(size=(80, 2))
    y = rng.poisson(np.exp(.2 + X @ [.3, -.2]))
    weight = np.linspace(.5, 2., len(y)) if weighted else None
    C = .7
    model = lm.PoissonRegression(C=C, solver='irls', device='cpu',
                                max_iter=1000, tol=1e-11).fit(X, y, sample_weight=weight)
    reference = sklearn.PoissonRegressor(alpha=1/(2*C), max_iter=1000, tol=1e-11).fit(
        X, y, sample_weight=weight)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-8)
    assert model.intercept_ == pytest.approx(reference.intercept_, abs=1e-8)


@pytest.mark.parametrize('cls', TYPED)
def test_typed_formula_interactions_missing_rows_and_weights_match_array_design(cls):
    pd = pytest.importorskip('pandas')
    patsy = pytest.importorskip('patsy')
    rng = np.random.default_rng(811)
    X = rng.normal(size=(60, 2))
    eta = .3 + .15*X[:, 0] - .1*X[:, 1]
    name = cls.__name__
    if 'Logistic' in name:
        y = rng.binomial(1, 1/(1+np.exp(-eta)))
    elif 'Poisson' in name or 'Binomial' in name:
        y = rng.poisson(np.exp(eta))
    elif cls is lm.PenalizedLinearRegression:
        y = eta + rng.normal(scale=.2, size=len(X))
    else:
        y = np.exp(eta) * rng.lognormal(0, .15, len(X))
    frame = pd.DataFrame({'y': y, 'x': X[:, 0], 'z': X[:, 1], 'group': ['a', 'b']*30})
    frame.loc[[2, 7], 'x'] = np.nan
    weight = np.linspace(.5, 2., len(frame))
    formula = 'y ~ 0 + x + C(group) + x:z'
    response, design = patsy.dmatrices(formula, frame, return_type='dataframe')
    opts = {'device': 'cpu', 'compute_inference': False, 'max_iter': 5000, 'tol': 1e-8}
    if name.startswith('Penalized'):
        opts.update(penalty='l2', alpha=.1, solver='fista' if cls is lm.PenalizedLinearRegression else 'newton')
    else:
        opts.update(C=0, solver='newton')
    model = cls(**opts).fit(formula=formula, data=frame, sample_weight=weight)
    reference = cls(fit_intercept=False, **opts).fit(
        design.to_numpy(), response.to_numpy().ravel(), sample_weight=weight[design.index])
    assert model.intercept_ == reference.intercept_ == 0
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-8)
    np.testing.assert_allclose(model.predict(frame.loc[design.index]), reference.predict(design), atol=1e-8)
    if name.startswith('Penalized'):
        with pytest.raises(ValueError, match='missing values'):
            model.predict(frame)


def test_typed_family_controls_and_penalized_prediction_distinctions():
    assert lm.GammaRegression(link='inverse_power')._get_loss_kwargs() == {'link': 'inverse_power'}
    assert lm.NegativeBinomialRegression(alpha=.25)._get_loss_kwargs() == {'alpha': .25}
    assert lm.TweedieRegression(power=1.2)._get_loss_kwargs() == {'power': 1.2}
    assert lm.PenalizedLinearRegression().penalty == 'l1'
    for cls in [lm.PenalizedLogisticRegression, lm.PenalizedPoissonRegression]:
        assert cls().penalty == 'l2'
        assert 'loss' not in inspect.signature(cls).parameters
        assert 'nodewise_alpha' not in inspect.signature(cls).parameters
        with pytest.raises(TypeError, match='nodewise_alpha'):
            cls(nodewise_alpha=.1)
    X = np.array([[-1.], [-1.], [1.], [1.]])
    y = np.array([0., 1., 0., 1.])
    model = lm.PenalizedLogisticRegression(alpha=.1, device='cpu').fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X), np.full((4, 2), .5))
    np.testing.assert_array_equal(model.predict(X), np.zeros(4))
    assert model.score(X, y) == pytest.approx(-1.)
    assert not hasattr(model, 'summary')
    with pytest.raises(TypeError, match='return_cpu'):
        model.predict_proba(X, return_cpu=True)


@pytest.mark.parametrize('method', ['cauchy', 'stouffer'])
@pytest.mark.parametrize('value', [np.nan, .42])
def test_combination_guard_rejects_unrelated_outputs(monkeypatch, method, value):
    module = _load_regression('test_pr168_inference_cycle4')
    monkeypatch.setattr(module, 'combine_pvalues', lambda *args, **kw: (value, value))
    with pytest.raises(AssertionError) as caught:
        module.test_combination_is_invariant_to_large_common_weight_scale(method, None)
    assert not isinstance(caught.value, module._WeightScaleChangedCombination)


@pytest.mark.parametrize('method', ['cauchy', 'stouffer'])
@pytest.mark.parametrize('axis', [None, 1])
def test_combination_guard_recognizes_actual_normalization_repair(monkeypatch, method, axis):
    from statgpu.inference import _multiple_testing
    module = _load_regression('test_pr168_inference_cycle4')
    original = _multiple_testing._validate_weights

    def repaired(weights, m, backend):
        weights = np.asarray(weights, dtype=float)
        return original(weights/weights.max(), m, backend)

    monkeypatch.setattr(_multiple_testing, '_validate_weights', repaired)
    module.test_combination_is_invariant_to_large_common_weight_scale(method, axis)


@pytest.mark.parametrize('value', [np.nan, 42.])
def test_weighted_r2_guard_rejects_unrelated_outputs(monkeypatch, value):
    module = _load_regression('test_pr168_linear_feature_cycle4')
    monkeypatch.setattr(lm.ElasticNet, 'rsquared', property(lambda self: value))
    with pytest.raises(AssertionError) as caught:
        module.test_sparse_training_r2_should_match_original_weighted_observations(lm.ElasticNet)
    assert not isinstance(caught.value, module._WorkingResponseR2)


def test_weighted_r2_guard_recognizes_original_data_repair(monkeypatch):
    module = _load_regression('test_pr168_linear_feature_cycle4')
    X, y, weights = module._weighted_problem()
    monkeypatch.setattr(lm.ElasticNet, 'rsquared', property(
        lambda self: module._weighted_r2(y, self.predict(X), weights)))
    module.test_sparse_training_r2_should_match_original_weighted_observations(lm.ElasticNet)


@pytest.mark.parametrize('value', [np.nan, 42.])
def test_kkt_guard_rejects_unrelated_residuals(monkeypatch, value):
    module = _load_regression('test_pr168_linear_independent_cycle4')
    monkeypatch.setattr(module, '_kkt_residual', lambda *args: value)
    with pytest.raises(AssertionError) as caught:
        module.test_requested_direct_kkt_stopping_should_certify_objective_residual(lm.Lasso, 1.)
    assert not isinstance(caught.value, module._UncertifiedKKT)


@pytest.mark.parametrize('kind', ['logistic', 'poisson'])
@pytest.mark.parametrize('value', [np.nan, 42.])
def test_oracle_guard_rejects_unrelated_inference_params(monkeypatch, kind, value):
    module = _load_regression('test_pr168_inference_cycle4')
    cls = lm.PenalizedLogisticRegression if kind == 'logistic' else lm.PenalizedPoissonRegression
    original = cls.fit

    def corrupted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self._inference_result.params[:] = value
        return result

    monkeypatch.setattr(cls, 'fit', corrupted)
    test = getattr(module, f'test_{kind}_oracle_targets_unpenalized_active_set')
    with pytest.raises(AssertionError) as caught:
        test('scad')
    assert not isinstance(caught.value, module._PenalizedOracleRefit)


class _DroppedFormulaPredictionRows(AssertionError):
    """Successful ordinary formula prediction silently removed query rows."""


ORDINARY_FORMULA = [lm.LinearRegression, lm.GeneralizedLinearModel, lm.PoissonRegression,
                    lm.GammaRegression, lm.InverseGaussianRegression,
                    lm.NegativeBinomialRegression, lm.TweedieRegression]


def _ordinary_formula_model(cls):
    pd = pytest.importorskip('pandas')
    pytest.importorskip('patsy')
    rng = np.random.default_rng(811)
    frame = pd.DataFrame({'x': rng.normal(size=40), 'z': rng.normal(size=40)})
    frame['y'] = rng.poisson(np.exp(.3 + .2*frame.x - .1*frame.z)) + 1
    options = {} if cls is lm.LinearRegression else {'C': 0, 'solver': 'newton', 'max_iter': 1000, 'tol': 1e-8}
    if cls is lm.GeneralizedLinearModel:
        options['family'] = 'poisson'
    model = cls(device='cpu', compute_inference=False, **options).fit(formula='y ~ x + z', data=frame)
    query = frame.iloc[:5].copy()
    query.loc[query.index[2], 'x'] = np.nan
    return model, query


def _assert_missing_query_rejected(model, query):
    complete = query.dropna(subset=['x', 'z'])
    expected = model.predict(complete).copy()
    coefficients = model.coef_.copy()
    try:
        actual = model.predict(query)
    except ValueError as error:
        assert re.search(r'missing|nan|non.?finite', str(error), re.IGNORECASE), str(error)
        np.testing.assert_array_equal(model.coef_, coefficients)
        np.testing.assert_allclose(model.predict(complete), expected, atol=1e-12)
        return
    assert actual.shape == expected.shape == (len(query)-1,)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12,
                               err_msg='A different numerical error is not silent formula row dropping')
    np.testing.assert_array_equal(model.coef_, coefficients)
    np.testing.assert_allclose(model.predict(complete), expected, atol=1e-12)
    raise _DroppedFormulaPredictionRows('Prediction exactly returned complete cases without their row index')


@pytest.mark.xfail(strict=True, raises=_DroppedFormulaPredictionRows,
                   reason='Issue #244: ordinary LinearRegression/GLM formula prediction silently drops missing query rows')
@pytest.mark.parametrize('cls', ORDINARY_FORMULA)
def test_ordinary_formula_prediction_should_reject_missing_query_rows(cls):
    model, query = _ordinary_formula_model(cls)
    _assert_missing_query_rejected(model, query)


@pytest.mark.xfail(strict=True, raises=_DroppedFormulaPredictionRows,
                   reason='Issue #244: LinearRegression.score can broadcast a formula-shortened prediction over the full response')
def test_linear_formula_score_should_reject_dropped_rows_before_broadcasting():
    pd = pytest.importorskip('pandas')
    pytest.importorskip('patsy')
    frame = pd.DataFrame({'x': np.linspace(-1., 1., 30)})
    frame['y'] = 1 + .2*frame.x
    model = lm.LinearRegression(device='cpu').fit(formula='y ~ x', data=frame)
    query = pd.DataFrame({'x': [.2, np.nan]})
    y_eval = np.array([1., 3.])
    try:
        score = model.score(query, y_eval)
    except ValueError as error:
        assert re.search(r'missing|nan|non.?finite|shape|length|sample', str(error), re.IGNORECASE)
        return
    actual = model.predict(query)
    np.testing.assert_allclose(actual, [1.04], atol=1e-12)
    assert np.shape(actual) == (1,)
    assert np.isfinite(score)
    broadcast_score = 1 - np.sum((y_eval-actual[0])**2)/np.sum((y_eval-y_eval.mean())**2)
    assert score == pytest.approx(broadcast_score, abs=1e-12)
    assert score == pytest.approx(-.9216, abs=1e-12)
    raise _DroppedFormulaPredictionRows('A single complete-case prediction was broadcast over two observations')


@pytest.mark.parametrize('fault', ['nan', 'wrong_finite', 'runtime_error', 'unrelated_value_error'])
def test_missing_query_guard_rejects_unrelated_failures(monkeypatch, fault):
    model, query = _ordinary_formula_model(lm.PoissonRegression)
    original = model.predict

    def corrupted(values):
        if values.isna().any().any():
            if fault == 'runtime_error':
                raise RuntimeError('unrelated prediction error')
            if fault == 'unrelated_value_error':
                raise ValueError('unrelated feature issue')
            return np.full(len(values)-1, np.nan if fault == 'nan' else 42.)
        return original(values)

    monkeypatch.setattr(model, 'predict', corrupted)
    with pytest.raises((AssertionError, RuntimeError)) as caught:
        _assert_missing_query_rejected(model, query)
    assert not isinstance(caught.value, _DroppedFormulaPredictionRows)


def test_missing_query_guard_recognizes_rejection_repair(monkeypatch):
    model, query = _ordinary_formula_model(lm.PoissonRegression)
    original = model.predict

    def repaired(values):
        if values.isna().any().any():
            raise ValueError('Prediction predictors contain missing values')
        return original(values)

    monkeypatch.setattr(model, 'predict', repaired)
    _assert_missing_query_rejected(model, query)


@pytest.mark.parametrize('kind', ['logistic', 'poisson'])
def test_oracle_guard_recognizes_unpenalized_refit_repair(monkeypatch, kind):
    module = _load_regression('test_pr168_inference_cycle4')
    cls = lm.PenalizedLogisticRegression if kind == 'logistic' else lm.PenalizedPoissonRegression
    direct = lm.LogisticRegression if kind == 'logistic' else lm.PoissonRegression
    original = cls.fit

    def repaired(self, X, y, **kwargs):
        result = original(self, X, y, **kwargs)
        active = np.flatnonzero(np.asarray(self._inference_result.metadata['active_set']))
        options = {'solver': 'newton'} if kind == 'poisson' else {}
        tolerance = 1e-8 if kind == 'poisson' else 1e-10
        refit = direct(C=0, device='cpu', max_iter=10000, tol=tolerance, **options).fit(X[:, active], y)
        self._inference_result.params[np.r_[0, active+1]] = np.r_[refit.intercept_, refit.coef_]
        return result

    monkeypatch.setattr(cls, 'fit', repaired)
    getattr(module, f'test_{kind}_oracle_targets_unpenalized_active_set')('scad')


@pytest.mark.parametrize('fault', ['nan', 'wrong_finite', 'unrelated_exception'])
def test_failed_refit_guard_rejects_unrelated_state_corruption(fault):
    module = _load_regression('test_pr168_linear_feature_cycle5')
    before = (np.array([1., 2.]), -10., 26., 30.)
    known_bad = (before[0], -2., 10., 12.)
    if fault == 'unrelated_exception':
        def prediction(X):
            raise RuntimeError('unrelated prediction failure')
    else:
        def prediction(X):
            return np.full(2, np.nan if fault == 'nan' else 42.)
    model = SimpleNamespace(_fitted=True, predict=prediction, loglikelihood=-2., aic=10., bic=12.)
    with pytest.raises((AssertionError, RuntimeError)) as caught:
        module._assert_atomic_failure(model, np.zeros((2, 1)), before, known_bad)
    assert not isinstance(caught.value, module._MixedFailedRefit)


def test_failed_refit_guard_recognizes_preserved_state_repair():
    module = _load_regression('test_pr168_linear_feature_cycle5')
    before = (np.array([1., 2.]), -10., 26., 30.)
    model = SimpleNamespace(_fitted=True, predict=lambda X: before[0],
                            loglikelihood=before[1], aic=before[2], bic=before[3])
    module._assert_atomic_failure(model, np.zeros((2, 1)), before)


@pytest.mark.parametrize('method,target', [('wrong_method', None), (None, 'wrong_target')])
def test_provenance_guard_rejects_wrong_populated_labels(monkeypatch, method, target):
    module = _load_regression('test_pr168_inference_cycle3')
    original = lm.Lasso.fit

    def corrupted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.inference_method_, self.inference_target_ = method, target
        return result

    monkeypatch.setattr(lm.Lasso, 'fit', corrupted)
    with pytest.raises(AssertionError) as caught:
        module.test_post_selection_publishes_the_same_method_and_target_as_result(lm.Lasso)
    assert not isinstance(caught.value, module._MissingPostSelectionProvenance)


def test_provenance_guard_recognizes_matching_publication_repair(monkeypatch):
    module = _load_regression('test_pr168_inference_cycle3')
    original = lm.Lasso.fit

    def repaired(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.inference_method_ = self._inference_result.method
        self.inference_target_ = 'active_set_refit_coefficient'
        return result

    monkeypatch.setattr(lm.Lasso, 'fit', repaired)
    module.test_post_selection_publishes_the_same_method_and_target_as_result(lm.Lasso)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_missing_formula_row_caution_is_visible_at_each_ordinary_entrypoint(language):
    reference = _page(language, 'reference', 'linear-model-api')
    assert 'missing-prediction-rows-in-ordinary-glms' in reference
    for cls in ORDINARY_FORMULA:
        assert cls.__name__ in reference.split('missing-prediction-rows-in-ordinary-glms', 1)[1]
        assert 'shorter unlabelled array' in ' '.join(inspect.getdoc(cls).split())
    for name in ['linear-regression', 'generalized-linear-model', 'poisson-regression']:
        assert 'missing-prediction-rows-in-ordinary-glms' in _page(language, 'models', name)
    if language == 'en':
        assert 'Unknown levels or missing values that would drop prediction rows raise' not in reference
    else:
        assert '未知水平或导致预测删行的缺失值会报错' not in reference
    assert 'LinearRegression.score' in reference


@pytest.mark.parametrize('cls', [lm.PenalizedLinearRegression, lm.PenalizedLogisticRegression,
                                lm.PenalizedPoissonRegression])
def test_fixed_penalized_loss_help_does_not_advertise_other_family_kwargs(cls):
    doc = inspect.getdoc(cls)
    section = doc.split('loss_kwargs :', 1)[1].split('\n\n', 1)[0]
    assert 'None or an empty dictionary' in section
    assert 'e.g.' not in section
    model = cls(device='cpu', compute_inference=False, loss_kwargs={})
    X = np.arange(8.)[:, None]
    y = np.arange(8.) % 2 if cls is lm.PenalizedLogisticRegression else np.arange(8.) + 1
    assert model.fit(X, y) is model
    with pytest.raises(TypeError, match=r'(SquaredErrorLoss|LogisticLoss|PoissonLoss)\(\) takes no arguments'):
        cls(device='cpu', compute_inference=False, loss_kwargs={'power': 1.2}).fit(X, y)
