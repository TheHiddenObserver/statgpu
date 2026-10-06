"""Public workflows and intended controls from the independent fourth review.

Strict xfails describe outstanding numerical/API repairs. Passing tests validate
safe workflows without requiring any implementation defect to persist.
"""
from __future__ import annotations

import inspect
import re
import warnings
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from statgpu import ElasticNet, Lasso, LassoCV
from statgpu.feature_selection import (
    FixedXKnockoffSelector,
    KnockoffSelector,
    fixed_x_knockoff_filter,
    knockoff_filter,
    model_x_knockoff_filter,
)
from statgpu.feature_selection._knockoff_utils import (
    _build_fixed_x_knockoffs,
    _standardize_design,
)

ROOT = Path(__file__).resolve().parents[2]


class _IgnoredADMMRho(AssertionError):
    """The final ADMM solver did not receive the requested initial rho."""


class _UncenteredFixedXGram(AssertionError):
    """Generated knockoffs fail the response-projected Gram identity."""


class _UncertifiedKKT(AssertionError):
    """A returned fit neither meets KKT tolerance nor reports exhausted work."""


class _MissingQValidation(AssertionError):
    """A knockoff consumer returned successfully for an invalid target rate."""


NONCONVERGENCE = (
    r'(?:did not|failed to|without) converg|convergence (?:failed|not achieved)|'
    r'maximum (?:number of )?iterations (?:reached|exceeded)'
)

KNOCKOFF_ROUTES = [
    pytest.param(fixed_x_knockoff_filter, {}, id='fixed-function'),
    pytest.param(model_x_knockoff_filter, {}, id='model-function'),
    pytest.param(knockoff_filter, {'knockoff_type': 'fixed_x'}, id='unified-fixed'),
    pytest.param(knockoff_filter, {'knockoff_type': 'model_x'}, id='unified-model'),
    pytest.param(FixedXKnockoffSelector, {}, id='fixed-selector'),
    pytest.param(KnockoffSelector, {'knockoff_type': 'fixed_x'}, id='selector-fixed'),
    pytest.param(KnockoffSelector, {'knockoff_type': 'model_x'}, id='selector-model'),
]


def _page(language, directory, name):
    return (ROOT / 'docs' / language / directory / f'{name}.md').read_text()


def _execute(code, label):
    namespace = {}
    exec(compile(code, label, 'exec'), namespace)  # noqa: S102 - repository example
    return namespace


def _marked(text, marker):
    match = re.search(r'<!-- ' + re.escape(marker) + r' -->\s*```python\n(.*?)```',
                      text, flags=re.DOTALL)
    assert match is not None
    return _execute(match.group(1), marker)


def _problem():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(180, 3))
    y = 1 + X @ np.array([2., -1., .4]) + rng.normal(scale=.3, size=180)
    return X, y


def _kkt_residual(X, y, coef, intercept, alpha, ratio, weights=None):
    """Subgradient distance for normalized weighted squared loss + L1/L2."""
    w = np.ones(len(y)) if weights is None else weights
    residual = intercept + X @ coef - y
    gradient = X.T @ (w * residual) / w.sum() + alpha * (1-ratio) * coef
    violation = np.maximum(np.abs(gradient) - alpha * ratio, 0)
    active = coef != 0
    violation[active] = np.abs(gradient[active] + alpha * ratio * np.sign(coef[active]))
    return float(violation.max())


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_lasso_first_example_is_independent_prediction_workflow(language):
    text = _page(language, 'models', 'lasso')
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.DOTALL)
    ns = _execute(blocks[0], f'{language}/lasso-first')
    model = ns['model']
    assert model.compute_inference is False
    assert ns['X_train'].shape == (180, 6)
    assert ns['X_test'].shape == (60, 6)
    assert ns['prediction'].shape == (60,)
    np.testing.assert_allclose(model.coef_, [1.887, -.880, 0, 0, 0, 0], atol=.0005)
    assert model.score(ns['X_test'], ns['y_test']) == pytest.approx(.980, abs=.0005)
    assert 'linear-model-api.md#lasso' in text
    assert r'\min_{b,\beta}' in text
    assert r'\sum_{i=1}^n w_i' in text


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_lasso_simultaneous_example_is_independently_executable(language):
    ns = _marked(_page(language, 'models', 'lasso'), 'learner-example: lasso-simultaneous')
    assert ns['ci_marginal'].shape == (7, 2)
    assert ns['ci_simul'].shape == (7, 2)
    assert np.isfinite(ns['ci_simul']).all()
    assert ns['m_sim'].simultaneous_include_intercept


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_lasso_complete_reference_covers_its_own_live_inventory(language):
    text = _page(language, 'reference', 'linear-model-api')
    section = text.split('## Lasso\n', 1)[1].split('## ElasticNet\n', 1)[0]
    model_page = _page(language, 'models', 'lasso')
    shared = _page(language, 'reference', 'estimator-api')
    sig = inspect.signature(Lasso)
    params = []
    for name, parameter in sig.parameters.items():
        assert f'| `{name}` |' in section
        assert f'| `{name}` |' in model_page
        default = parameter.default
        if isinstance(default, Enum):
            default = default.value
        params.append(parameter.replace(annotation=inspect.Parameter.empty, default=default))
    expected = 'Lasso' + str(sig.replace(parameters=params, return_annotation=inspect.Signature.empty))
    assert expected in section
    for name in dir(Lasso):
        if name.startswith('_'):
            continue
        member = getattr(Lasso, name)
        if callable(member):
            assert name in section, name
            for parameter in inspect.signature(member).parameters:
                if parameter != 'self':
                    assert parameter in section + shared, (name, parameter)
        elif isinstance(inspect.getattr_static(Lasso, name), property):
            assert name in section, name
    for shape in ['(n,p)', '(n,)', '(m,p)', '(m,)', '(p,)', '(k,)', '(k,2)']:
        assert shape in section
    for extra in ['l1_ratio', 'cov_type', 'hac_maxlags', 'initial_coef']:
        assert f'| `{extra}` |' not in section


def test_lasso_installed_help_covers_parameters_methods_and_limits():
    doc = inspect.getdoc(Lasso)
    for name in inspect.signature(Lasso).parameters:
        assert re.search(r'^' + name + r'\s*:', doc, flags=re.MULTILINE), name
    for name in ['fit(X=None', 'predict(X, return_cpu=True)', 'score(X, y, sample_weight=None)',
                 'summary()', 'get_params(deep=True)', 'set_params(**params)',
                 'adjust_pvalues', 'combine_pvalues', 'bootstrap_statistic', 'permutation_test']:
        assert name in doc
    for phrase in ['currently ignored', 'rho=1.0', 'NumPy host helper', 'empirical residuals']:
        assert phrase in doc


@pytest.mark.parametrize('fit_intercept', [True, False])
def test_lasso_both_inference_parameter_layouts(fit_intercept):
    X, y = _problem()
    model = Lasso(alpha=.1, fit_intercept=fit_intercept, device='cpu',
                  solver='coordinate_descent', max_iter=5000, tol=1e-10).fit(X, y)
    k = X.shape[1] + int(fit_intercept)
    assert model.coef_.shape == (3,)
    for field in ['_params', '_bse', '_pvalues', '_zvalues']:
        assert getattr(model, field).shape == (k,)
    assert model._conf_int.shape == (k, 2)
    np.testing.assert_allclose(model._params, model._inference_result.params)
    if fit_intercept:
        assert model._params[0] == pytest.approx(y.mean() - X.mean(0) @ model._params[1:])
    else:
        assert model.intercept_ == 0
        assert model._params.shape == model.coef_.shape


def test_lasso_parameter_lifecycle_and_prediction_return_cpu():
    clone = pytest.importorskip('sklearn.base').clone
    X, y = _problem()
    model = Lasso(alpha=.1, device='cpu', compute_inference=False).fit(X, y)
    np.testing.assert_allclose(model.predict(X), model.predict(X, return_cpu=False))
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    assert model.set_params(alpha=.2) is model
    with pytest.raises((ValueError, RuntimeError)):
        model.predict(X)
    model.fit(X, y)
    assert model.predict(X[:4]).shape == (4,)


@pytest.mark.parametrize('fit_intercept', [True, False])
def test_lasso_weighted_objective_matches_aligned_sklearn(fit_intercept):
    sk_lasso = pytest.importorskip('sklearn.linear_model').Lasso
    X, y = _problem()
    w = np.linspace(.2, 3., len(y))
    options = {'alpha': .1, 'fit_intercept': fit_intercept, 'max_iter': 10000, 'tol': 1e-11}
    model = Lasso(**options, solver='coordinate_descent', device='cpu',
                  compute_inference=False).fit(X, y, sample_weight=w)
    ref = sk_lasso(**options).fit(X, y, sample_weight=w)
    np.testing.assert_allclose(model.coef_, ref.coef_, atol=1e-9)
    assert model.intercept_ == pytest.approx(ref.intercept_, abs=1e-9)
    assert _kkt_residual(X, y, model.coef_, model.intercept_, .1, 1., w) < 1e-8
    weighted_r2 = 1 - np.sum(w * (y-model.predict(X))**2) / np.sum(w * (y-np.average(y, weights=w))**2)
    assert model.score(X, y, sample_weight=w) == pytest.approx(weighted_r2)


@pytest.mark.parametrize('cls,ratio', [(Lasso, 1.), (ElasticNet, .5)])
def test_coordinate_descent_recovers_scaled_diagonal_analytic_solution(cls, ratio):
    X = np.diag([1000., 1.])
    y = np.array([1000., 1.])
    options = {'l1_ratio': ratio} if cls is ElasticNet else {}
    model = cls(alpha=.1, fit_intercept=False, solver='coordinate_descent',
                stopping='coef_delta', device='cpu', tol=1e-10, max_iter=1000,
                compute_inference=False, **options).fit(X, y)
    expected = (X.T @ y / 2 - .1 * ratio) / (np.diag(X.T @ X) / 2 + .1*(1-ratio))
    np.testing.assert_allclose(model.coef_, expected, atol=1e-9)
    assert _kkt_residual(X, y, model.coef_, 0., .1, ratio) < 1e-8


@pytest.mark.xfail(strict=True, raises=_UncertifiedKKT,
                   reason='Issue #233: direct Gaussian stopping=kkt is ignored')
@pytest.mark.parametrize('cls,ratio', [(Lasso, 1.), (ElasticNet, .5)])
def test_requested_direct_kkt_stopping_should_certify_objective_residual(cls, ratio):
    X = np.diag([1000., 1.])
    y = np.array([1000., 1.])
    options = {'l1_ratio': ratio} if cls is ElasticNet else {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = cls(alpha=.1, fit_intercept=False, solver='fista', stopping='kkt',
                        device='cpu', tol=1e-4, max_iter=1000, compute_inference=False,
                        **options).fit(X, y)
    except ValueError as error:
        # Clear rejection of unsupported KKT stopping is an acceptable repair.
        if re.search(r"(?:kkt|stopping).*(?:unsupported|not supported|not implemented)|"
                     r"(?:unsupported|not supported|not implemented).*(?:kkt|stopping)", str(error), re.IGNORECASE):
            return
        raise  # Do not swallow unrelated input or programming failures.
    except RuntimeError as error:
        if re.search(NONCONVERGENCE,
                     str(error), re.IGNORECASE):
            return
        raise
    assert np.asarray(model.coef_).shape == (2,)
    assert np.isfinite(model.coef_).all()
    residual = _kkt_residual(X, y, model.coef_, 0., .1, ratio)
    assert np.isfinite(residual), "Nonfinite KKT evaluation is a different failure"
    assert residual >= 0
    diagonal = np.diag(X)**2 / len(y) + .1*(1-ratio)
    smooth_gradient = diagonal*model.coef_ - np.diag(X)*y/len(y)
    independent = np.maximum(np.abs(smooth_gradient)-.1*ratio, 0.)
    active = np.asarray(model.coef_) != 0
    independent[active] = np.abs(smooth_gradient[active] + .1*ratio*np.sign(model.coef_[active]))
    assert residual == pytest.approx(float(independent.max()), abs=1e-10)
    if residual <= model.tol:
        return  # Convergence is valid even on the final allowed iteration.
    reported_failure = getattr(model, 'converged_', None) is False or any(
        re.search(NONCONVERGENCE,
                  str(w.message), re.IGNORECASE) or w.category.__name__ == 'ConvergenceWarning'
        for w in caught
    )
    if model.n_iter_ >= model.max_iter and reported_failure:
        return  # An honestly exhausted finite budget is not false convergence.
    # The reported early exit must be the specific two-step coefficient-delta
    # result on this separable diagonal fixture. Do not xfail arbitrary solver
    # corruption merely because its KKT residual is large.
    curvature = np.diag(X.T @ X) / len(y) + .1 * (1-ratio)
    linear_score = X.T @ y / len(y) - .1 * ratio
    step = 1 / curvature.max()
    first = step * linear_score
    second = first + step * (linear_score - curvature * first)
    np.testing.assert_allclose(model.coef_, second, atol=1e-12, rtol=1e-10)
    assert model.n_iter_ == 2
    analytic_residual = np.max(np.abs(curvature * model.coef_ - linear_score))
    assert residual == pytest.approx(analytic_residual, abs=1e-12)
    raise _UncertifiedKKT(f'KKT residual {residual} exceeds tolerance {model.tol}')


@pytest.mark.parametrize('outcome', ['converged-at-budget', 'reported-exhaustion', 'explicit-rejection'])
def test_kkt_regression_does_not_mask_repaired_stopping(outcome):
    class RepairedEstimator:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            if outcome == 'explicit-rejection':
                raise ValueError("stopping='kkt' is not supported")
            if outcome == 'converged-at-budget':
                coef = (X.T @ y / len(y) - self.params['alpha']) / (np.diag(X.T @ X) / len(y))
            else:
                coef = np.zeros(X.shape[1])
            return SimpleNamespace(coef_=coef, tol=self.params['tol'],
                                   n_iter_=self.params['max_iter'], max_iter=self.params['max_iter'],
                                   converged_=outcome == 'converged-at-budget')

    # Direct invocation avoids pytest's xfail marker: every repaired policy must pass.
    test_requested_direct_kkt_stopping_should_certify_objective_residual(RepairedEstimator, 1.)


def test_kkt_regression_does_not_swallow_unrelated_errors():
    class BrokenEstimator:
        def __init__(self, **params):
            pass

        def fit(self, X, y):
            raise ValueError('unrelated shape mismatch')

    with pytest.raises(ValueError, match='unrelated shape mismatch'):
        test_requested_direct_kkt_stopping_should_certify_objective_residual(BrokenEstimator, 1.)


@pytest.mark.xfail(strict=True, raises=_IgnoredADMMRho,
                   reason='Issue #234: unified ADMM hardcodes initial rho=1.0')
@pytest.mark.parametrize('requested', [.1, 10.])
@pytest.mark.parametrize('use_cv', [False, True])
def test_lasso_admm_rho_should_reach_direct_and_cv_final_solver(monkeypatch, requested, use_cv):
    from statgpu import solvers

    original = solvers.admm_solver
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs['rho'])
        return original(*args, **kwargs)

    monkeypatch.setattr(solvers, 'admm_solver', capture)
    X, y = _problem()
    options = {'solver': 'admm', 'admm_rho': requested, 'device': 'cpu',
               'max_iter': 2000, 'tol': 1e-8, 'compute_inference': False}
    if use_cv:
        model = LassoCV(alphas=[.1, .2], cv=2, random_state=7, **options).fit(X, y)
        assert model.estimator_.admm_rho == requested
    else:
        model = Lasso(alpha=.1, **options).fit(X, y)
        assert model.admm_rho == requested
    assert calls, 'actual ADMM solver was not reached'
    if calls[-1] != requested:
        assert calls[-1] == 1.0, 'Only the hardcoded initial rho belongs to issue #234'
        raise _IgnoredADMMRho(f'final solver received rho={calls[-1]!r}, requested {requested!r}')


@pytest.mark.xfail(strict=True, raises=_UncenteredFixedXGram,
                   reason='Issue #232: generated fixed-X Xk is not centered')
@pytest.mark.parametrize('n,p', [(2, 1), (30, 6)])
def test_generated_fixedx_should_preserve_gram_after_response_projection(n, p):
    X = np.array([[-1.], [1.]]) if n == 2 else np.random.default_rng(123).normal(size=(n, p))
    X = _standardize_design(X, np)
    Xk = _build_fixed_x_knockoffs(X, 7, np)
    np.testing.assert_allclose(X.T @ X, Xk.T @ Xk, atol=1e-12)
    P = np.eye(n) - np.ones((n, n))/n
    # This is a covariance identity, not a measured population-FDR assertion.
    expected = X.T @ P @ X
    observed = Xk.T @ P @ Xk
    assert observed.shape == expected.shape
    if not np.allclose(expected, observed, rtol=1e-7, atol=1e-12, equal_nan=True):
        raise _UncenteredFixedXGram(f'projected knockoff Gram {observed!r} differs from {expected!r}')


@pytest.mark.parametrize('language', ['en', 'cn'])
@pytest.mark.parametrize('marker', ['api-example: knockoff-fresh-inputs', 'api-example: knockoff-selector'])
def test_documented_supplied_pairs_have_valid_projected_geometry(language, marker, monkeypatch):
    from collections import OrderedDict

    from statgpu.feature_selection import _knockoff_utils as utilities
    from statgpu.linear_model.wrappers import _lasso

    # Each independently executed example starts with the documented fresh-process
    # precondition. Throwaway EN/CN namespaces must not reuse another example's
    # pointer-keyed cache after its input references have been released (#211).
    monkeypatch.setattr(utilities, '_LASSO_DIFF_CACHE', OrderedDict())
    monkeypatch.setattr(_lasso, '_LASSO_CV_ALPHA_CACHE', OrderedDict())
    ns = _marked(_page(language, 'reference', 'feature-selection-api'), marker)
    X, Xk = ns['X'], ns['Xk']
    n, p = X.shape
    assert n >= 2*p + 1
    P = np.eye(n) - np.ones((n, n))/n
    np.testing.assert_allclose(X.mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(Xk.mean(0), 0, atol=1e-14)
    np.testing.assert_allclose(X.T @ P @ X, Xk.T @ P @ Xk, atol=1e-14)
    np.testing.assert_allclose(X.T @ P @ Xk, np.zeros((p, p)), atol=1e-14)
    # Swapping a valid supplied pair reverses corr_diff featurewise.
    y = np.random.default_rng(11).normal(size=n)
    result = fixed_x_knockoff_filter(X, y, Xk=Xk, backend='numpy')
    swapped = fixed_x_knockoff_filter(Xk, y, Xk=X, backend='numpy')
    np.testing.assert_allclose(result.W, -swapped.W, atol=1e-14)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_new_control_geometry_and_inference_boundaries_are_explicit(language):
    lasso = _page(language, 'models', 'lasso')
    enet = _page(language, 'models', 'elastic-net')
    linear = _page(language, 'reference', 'linear-model-api')
    for text in [lasso, enet, linear]:
        assert 'stopping' in text and 'KKT' in text
        assert ('ignored' in text or 'ignores' in text) if language == 'en' else '忽略' in text
    assert 'rho=1.0' in lasso and 'rho=1.0' in linear
    for directory, page in [('models', 'knockoff'), ('reference', 'feature-selection-api')]:
        text = _page(language, directory, page)
        for token in ['Xk', 'n≥2p+1', 'Model-X']:
            assert token in text
        assert ('homoskedastic' in text) if language == 'en' else '同方差' in text
        assert ('arbitrary' in text) if language == 'en' else '任意' in text
        assert ('projection' in text or 'projected' in text) if language == 'en' else '投影' in text
    assert 'fit_intercept=False' in lasso
    assert ('host' in lasso) if language == 'en' else '主机' in lasso
    for stale in ['failing closed', 'Maintained', '_selected_backend_name',
                  '## strict/approx difference', 'Physical CUDA validators']:
        assert stale not in lasso
    if language == 'cn':
        for stale in ['新的 Gaussian 响应变量', '解析权重']:
            assert stale not in lasso
        assert '拟合值加重采样残差' in lasso
        assert 'Formula 输入' not in _page(language, 'models', 'linear-regression')
        assert 'penalized solver API 迁移指南' not in enet
        assert 'S-matrix 方法' not in _page(language, 'models', 'knockoff').replace('`', '')
        assert 'StepwiseSelector 消费' not in _page(language, 'reference', 'feature-selection-api')


@pytest.mark.parametrize('requested', [None, 'auto', 'debiased'])
def test_lasso_default_and_ordinary_auto_report_debiased_inference(requested):
    X, y = _problem()
    options = {} if requested is None else {'inference_method': requested}
    model = Lasso(alpha=.1, device='cpu', solver='coordinate_descent',
                  max_iter=5000, tol=1e-10, **options).fit(X, y)
    assert inspect.signature(Lasso).parameters['inference_method'].default == 'debiased'
    assert model.get_params()['inference_method'] == (requested or 'debiased')
    assert model._inference_result.method == 'debiased'
    assert np.isfinite(model._conf_int).all()


def test_lasso_simultaneous_constructor_requires_explicit_debiased_spelling():
    with pytest.raises(ValueError, match="Simultaneous inference requires inference_method='debiased'"):
        Lasso(inference_method='auto', enable_simultaneous_inference=True, device='cpu')
    model = Lasso(inference_method='debiased', enable_simultaneous_inference=True, device='cpu')
    assert model.enable_simultaneous_inference and model.compute_inference


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_lasso_auto_resolution_and_path_dependent_admm_are_documented(language):
    for directory in ['models', 'reference']:
        text = _page(language, directory, 'lasso' if directory == 'models' else 'linear-model-api')
        section = text if directory == 'models' else text.split('## Lasso\n')[1].split('## ElasticNet\n')[0]
        row = next(line for line in section.splitlines() if line.startswith('| `inference_method` |'))
        assert '`auto`' in row and '`debiased`' in row
        assert '`enable_simultaneous_inference=True`' in row
        assert 'Cholesky' in section
        assert ('keeps rho fixed' in section if language == 'en' else '保持 rho 不变' in section)
    doc = inspect.getdoc(Lasso)
    assert "ordinary 'auto'" in doc and "constructor currently rejects 'auto'" in doc
    assert 'Cholesky solve keeps rho fixed' in doc


def _centered_knockoff_pair():
    rng = np.random.default_rng(42)
    n, p = 80, 4
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
    X, Xk = Q[:, 1:p+1], Q[:, p+1:2*p+1]
    y = 5*X[:, 0] + rng.normal(size=n)
    return X, y, Xk


def _call_knockoff(entrypoint, branch, X, y, Xk, **options):
    if inspect.isclass(entrypoint):
        selector = entrypoint(**branch, **options).fit(X, y, Xk=Xk)
        result = selector.result_
        np.testing.assert_array_equal(np.flatnonzero(selector.get_support()), result.selected_features)
        return result
    return entrypoint(X, y, Xk=Xk, **branch, **options)


@pytest.mark.parametrize('entrypoint,branch', KNOCKOFF_ROUTES)
@pytest.mark.parametrize('fdr_control', ['knockoff_plus', 'knockoff'])
@pytest.mark.parametrize('q', [
    pytest.param(np.nan, marks=pytest.mark.xfail(
        strict=True, raises=_MissingQValidation,
        reason='Issue #235: knockoff q validation accepts NaN and can produce empty selection',
    ), id='nan'),
    pytest.param(np.inf, id='positive-inf'),
    pytest.param(-np.inf, id='negative-inf'),
    pytest.param(0., id='zero'),
    pytest.param(1., id='one'),
    pytest.param(-.1, id='negative'),
    pytest.param(1.1, id='above-one'),
])
def test_knockoff_invalid_target_must_raise_q_value_error(entrypoint, branch, fdr_control, q):
    X, y, Xk = _centered_knockoff_pair()
    try:
        _call_knockoff(entrypoint, branch, X, y, Xk, q=q,
                       fdr_control=fdr_control, backend='numpy')
    except ValueError as error:
        if re.search(r"\bq\b", str(error), re.IGNORECASE):
            return
        raise  # An unrelated error is not successful q validation.
    raise _MissingQValidation('invalid q was accepted instead of raising ValueError')


@pytest.mark.parametrize('entrypoint,branch', KNOCKOFF_ROUTES)
@pytest.mark.parametrize('fdr_control', ['knockoff_plus', 'knockoff'])
@pytest.mark.parametrize('q', [.1, .5])
def test_knockoff_finite_target_with_valid_supplied_pair(entrypoint, branch, fdr_control, q):
    X, y, Xk = _centered_knockoff_pair()
    assert np.isfinite(q) and 0 < q < 1
    result = _call_knockoff(entrypoint, branch, X, y, Xk, q=q,
                            fdr_control=fdr_control, backend='numpy')
    assert result.q == q
    assert result.fdr_control == fdr_control
    assert np.isfinite(result.W).all()
    np.testing.assert_array_equal(result.selected_features, np.flatnonzero(result.W >= result.threshold))
    assert np.isfinite(result.estimated_fdr)


@pytest.mark.parametrize('entrypoint', [
    fixed_x_knockoff_filter, model_x_knockoff_filter, knockoff_filter,
    FixedXKnockoffSelector, KnockoffSelector,
])
def test_knockoff_q_defaults_and_installed_help_disclose_nan(entrypoint):
    assert inspect.signature(entrypoint).parameters['q'].default == .1
    assert inspect.signature(entrypoint).parameters['backend'].default == 'auto'
    doc = inspect.getdoc(entrypoint)
    assert 'np.isfinite(q) and 0 < q < 1' in doc
    assert 'NaN' in doc and 'estimated_fdr=0.0' in doc


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_knockoff_q_checks_are_executable_in_both_documented_workflows(language):
    for directory, page, marker in [
        ('models', 'knockoff', 'learner-example: knockoff-selection'),
        ('reference', 'feature-selection-api', 'api-example: knockoff-selector'),
    ]:
        text = _page(language, directory, page)
        namespace = _marked(text, marker)
        q = namespace['q']
        assert np.isfinite(q) and 0 < q < 1
        result = namespace['result'] if 'result' in namespace else namespace['selector'].result_
        assert result.q == q
        assert 'np.isfinite(q) and 0 < q < 1' in text
        assert 'q=np.nan' in text and 'estimated_fdr=0.0' in text
        match = re.search(r'<!-- ' + re.escape(marker) + r' -->\s*```python\n(.*?)```',
                          text, flags=re.DOTALL)
        for invalid in ['np.nan', 'np.inf', '-np.inf', '0.', '1.']:
            invalid_code, count = re.subn(r'^q = [^\n]+$', f'q = {invalid}',
                                          match.group(1), flags=re.MULTILINE)
            assert count == 1
            with pytest.raises(ValueError, match='q must be finite and strictly between 0 and 1'):
                _execute(invalid_code, marker + '-invalid-q')


@pytest.mark.parametrize('entrypoint,branch', KNOCKOFF_ROUTES)
@pytest.mark.parametrize('input_kind', ['numpy', 'torch-cpu'])
def test_knockoff_torch_library_selection_preserves_cpu_inputs(monkeypatch, entrypoint, branch, input_kind):
    torch = pytest.importorskip('torch')
    from statgpu.feature_selection import _knockoff

    X, y, Xk = _centered_knockoff_pair()
    if input_kind == 'torch-cpu':
        X, y, Xk = [torch.as_tensor(a, device='cpu') for a in (X, y, Xk)]
    original = _knockoff._compute_w_statistics
    devices = []

    def observe_inputs(x, xk, response, **kwargs):
        devices.append(tuple(a.device.type for a in (x, xk, response)))
        return original(x, xk, response, **kwargs)

    monkeypatch.setattr(_knockoff, '_compute_w_statistics', observe_inputs)
    result = _call_knockoff(entrypoint, branch, X, y, Xk, q=.1, backend='torch')
    assert devices and all(row == ('cpu', 'cpu', 'cpu') for row in devices)
    assert result.backend == 'torch'
    assert isinstance(result.W, np.ndarray) and np.isfinite(result.W).all()


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_knockoff_torch_placement_and_two_row_normalization_are_explicit(language):
    learner = _page(language, 'models', 'knockoff')
    reference = _page(language, 'reference', 'feature-selection-api')
    for text in [learner, reference]:
        assert 'Torch CPU' in text and 'CUDA' in text and 'device="torch"' in text
    assert 'X_work=[−1,1]ᵀ/√2' in learner
    assert ('input X=[−1,1]ᵀ has Gram 2' in learner if language == 'en' else '输入 X=[−1,1]ᵀ 的 Gram 为 2' in learner)
    X = np.array([[-1.], [1.]])
    assert (X.T @ X).item() == 2
    work = _standardize_design(X, np)
    np.testing.assert_allclose(work, X/np.sqrt(2))
    assert (work.T @ work).item() == pytest.approx(1.)
    doc = inspect.getdoc(fixed_x_knockoff_filter)
    assert 'Torch library, not CUDA placement' in doc
    assert 'Torch CPU inputs run on CPU' in doc


def test_logistic_learner_has_one_language_switch():
    text = _page('en', 'models', 'logistic-regression')
    assert text.count('[Chinese](../../cn/models/logistic-regression.md)') == 1
