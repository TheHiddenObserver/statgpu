"""Executable workflows and exact API boundaries from the fourth PR168 review.

Known runtime limitations are observed conditionally, not required to persist.
Correctness assertions cover the documented safe workflows and public results.
"""
import linecache
import re
from pathlib import Path

import numpy as np
import pytest

from statgpu import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PenalizedLinearRegression,
    StepwiseSelector,
)
from statgpu.feature_selection import (
    FixedXKnockoffSelector,
    KnockoffSelector,
    fixed_x_knockoff_filter,
)

ROOT = Path(__file__).resolve().parents[2]


class _WorkingResponseR2(AssertionError):
    """Sparse training R-squared differs from original weighted observations."""


class _MissingSmallCVDetails(AssertionError):
    """The small-sample CV selector omitted std_mse at result publication."""


def _page(language, directory, name):
    return (ROOT / 'docs' / language / directory / (name + '.md')).read_text()


def _weighted_problem():
    rng = np.random.default_rng(25)
    X = rng.normal(size=(20, 2))
    y = np.arange(20.) + 2 * X[:, 0]
    return X, y, np.r_[np.ones(19), 1000.]


def _weighted_r2(y, prediction, weights):
    mean = np.average(y, weights=weights)
    return 1 - np.sum(weights * (y - prediction)**2) / np.sum(weights * (y - mean)**2)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_weighted_training_example_matches_original_observation_definition(language):
    text = _page(language, 'models', 'elastic-net')
    block = re.search(r'<!-- learner-example: elasticnet-weighted-score -->\s*```python\n(.*?)```', text, re.DOTALL)
    assert block
    namespace = {}
    exec(compile(block.group(1), 'elasticnet-weighted-score', 'exec'), namespace)  # noqa: S102
    m, X, y, w = [namespace[name] for name in ['model', 'X', 'y', 'weights']]
    expected = _weighted_r2(y, m.predict(X), w)
    assert namespace['weighted_r2'] == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(.1798437899836577, abs=1e-9)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_weighted_reporting_limit_has_safe_scoring_alternative(language):
    X, y, w = _weighted_problem()
    options = {'alpha': .3, 'device': 'cpu', 'max_iter': 5000, 'tol': 1e-8}
    fit = ElasticNet(compute_inference=True, **options).fit(X, y, sample_weight=w)
    plain = ElasticNet(compute_inference=False, **options).fit(X, y, sample_weight=w)
    np.testing.assert_allclose(fit.predict(X), plain.predict(X), atol=1e-12)
    expected = _weighted_r2(y, fit.predict(X), w)
    assert fit.score(X, y, sample_weight=w) == pytest.approx(expected, abs=1e-12)
    assert fit.score(X, y, sample_weight=10*w) == pytest.approx(expected, abs=1e-12)
    if not np.isclose(fit.rsquared, expected):
        for model in ['elastic-net', 'lasso']:
            text = _page(language, 'models', model)
            assert 'rsquared_adj' in text
            assert 'score(X, y, sample_weight=weights)' in text
            assert ('working response' in ' '.join(text.split()) if language == 'en' else '工作响应' in text)
        reference = _page(language, 'reference', 'linear-model-api')
        assert ('re-centered working response' in reference if language == 'en' else '再次中心化的工作响应' in reference)
        assert 're-centered working response' in ElasticNet.__doc__


@pytest.fixture
def logistic_model():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(60, 2))
    y = rng.binomial(1, 1/(1+np.exp(-X[:, 0])))
    return LogisticRegression(device='cpu', compute_inference=False).fit(X, y), X[:8]


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_precision_recall_one_class_boundary_is_explicit(language, logistic_model):
    model, X = logistic_model
    zeros, ones = np.zeros(len(X)), np.ones(len(X))
    for method in ['precision_recall_curve', 'average_precision_score']:
        with pytest.raises(ValueError, match='no positive class'):
            getattr(model, method)(X, zeros)
    precision, recall, thresholds = model.precision_recall_curve(X, ones)
    assert len(precision) == len(recall) == len(thresholds)
    assert np.isinf(thresholds[0])
    assert precision[0] == 1 and recall[0] == 0
    assert model.average_precision_score(X, ones) == pytest.approx(1.)
    for y in [zeros, ones]:
        with pytest.raises(ValueError, match='only one class'):
            model.roc_curve(X, y)
        assert np.isfinite(model.classification_table(X, y)['accuracy'])
    for directory, page in [('models', 'logistic-regression'), ('reference', 'linear-model-api')]:
        text = _page(language, directory, page)
        assert ('all-zero' in text if language == 'en' else '全零' in text)
        assert ('all-one' in text.lower() if language == 'en' else '全一' in text)
    assert 'all-zero y raises' in model.precision_recall_curve.__doc__


def test_lasso_profile_discloses_changes_to_statistical_tuning():
    rng = np.random.default_rng(23)
    n, p = 80, 4
    basis, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
    X, Xk = basis[:, 1:p+1].copy(), basis[:, p+1:2*p+1].copy()
    y = 5 * X[:, 0] + rng.normal(scale=.1, size=n)
    # Keep both input snapshots alive to avoid address-based seeded cache reuse.
    snapshots = [tuple(a.copy() for a in (X, y, Xk)) for _ in range(2)]
    results = []
    for arrays, profile in zip(snapshots, ['off', 'aggressive']):
        x, response, xk = arrays
        results.append(fixed_x_knockoff_filter(
            x, response, Xk=xk, method='lasso_coef_diff', random_state=17,
            backend='numpy', lasso_cv_impl='statgpu', lasso_fast_profile=profile,
        ))
    assert all(np.isfinite(result.W).all() for result in results)
    for language in ['en', 'cn']:
        for directory, name in [('models', 'knockoff'), ('reference', 'feature-selection-api')]:
            text = _page(language, directory, name)
            assert 'lasso_fast_profile' in text
            assert ('candidate penalties' in text if language == 'en' else '候选惩罚' in text)
            assert ('selected features' in text or 'selected\nfeatures' in text if language == 'en' else '入选特征' in text)
    assert 'not an\n        output-preserving speed switch' in fixed_x_knockoff_filter.__doc__


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_stepwise_original_layout_matches_direct_selected_model(language):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 4))
    y = 2*X[:, 0] + rng.normal(scale=.1, size=60)
    selector = StepwiseSelector(LinearRegression, device='cpu', compute_inference=False,
                                max_features=1, criterion='bic').fit(X, y)
    selected = selector.transform(X)
    np.testing.assert_allclose(selector.predict(X), selector.best_model_.predict(selected))
    assert selector.selected_features_ == [0]
    try:
        changed_width = selector.predict(X[:, :1])
    except ValueError:
        return  # A future width validator satisfies the stronger contract.
    np.testing.assert_allclose(changed_width, selector.predict(X))
    text = _page(language, 'models', 'feature-selection')
    assert ('do not verify the original width' in text if language == 'en' else '不检查原始列数' in text)


@pytest.mark.parametrize('selector_class', [KnockoffSelector, FixedXKnockoffSelector])
def test_failed_knockoff_refit_discloses_stale_selection(selector_class):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] + rng.normal(size=50)
    selector = selector_class(backend='numpy', random_state=7).fit(X, y)
    old = selector.result_
    with pytest.raises(ValueError):
        selector.fit(X, y[:-1])
    if selector.result_ is old:
        for language in ['en', 'cn']:
            text = _page(language, 'reference', 'feature-selection-api')
            assert ('failed `fit`' in text if language == 'en' else '失败的 `fit`' in text)


def test_small_elasticnet_cv_boundary_is_documented():
    X, y, _ = _weighted_problem()
    try:
        ElasticNetCV(alphas=[.1], cv=2, device='cpu').fit(X[:3], y[:3])
    except KeyError as error:
        assert str(error) == "'std_mse'"
        for language in ['en', 'cn']:
            text = _page(language, 'reference', 'linear-model-api')
            assert ('below four rows' in text if language == 'en' else '少于四行' in text)
    except ValueError:
        pass  # A clear future domain validation is an acceptable repair.


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_shared_pvalue_helpers_reference_validated_controls(language):
    model = LinearRegression(device='cpu', compute_inference=False)
    pvalues = [.01, .04, .5]
    alpha = .05
    assert np.isfinite(alpha) and 0 < alpha < 1
    assert model.adjust_pvalues(pvalues, method='holm', alpha=alpha)['reject'].tolist() == [True, False, False]
    for method in ['cauchy', 'stouffer']:
        ordinary = model.combine_pvalues(pvalues, method=method, weights=[1., 2., 3.])
        large = np.array([1., 2., 3.])*1e200
        scaled = model.combine_pvalues(pvalues, method=method, weights=large/large.max())
        assert scaled['pvalue'] == pytest.approx(ordinary['pvalue'], abs=1e-12)
    text = _page(language, 'reference', 'estimator-api')
    assert 'NaN' in text and 'a-complete-axis-and-weight-example' in text
    assert 'validate-and-rescale-combination-weights' in text


@pytest.mark.xfail(strict=True, raises=_WorkingResponseR2, reason="Issue #229: sparse weighted diagnostic totals use working y")
@pytest.mark.parametrize("estimator_class", [ElasticNet, Lasso, PenalizedLinearRegression])
def test_sparse_training_r2_should_match_original_weighted_observations(estimator_class):
    X, y, weights = _weighted_problem()
    model = estimator_class(alpha=.3, device="cpu", compute_inference=True,
                            max_iter=5000, tol=1e-8).fit(X, y, sample_weight=weights)
    expected = _weighted_r2(y, model.predict(X), weights)
    observed = model.rsquared
    assert np.isfinite(observed)
    assert np.isfinite(expected)
    if observed == pytest.approx(expected, abs=1e-10):
        return
    # Identify the exact second-centering error from raw data, rather than
    # accepting arbitrary R² mismatches (including finite garbage) as #229.
    row_scale = np.sqrt(weights / weights.mean())
    working_y = row_scale * (y - np.average(y, weights=weights))
    working_residual = row_scale * (y - model.predict(X))
    np.testing.assert_allclose(model._y, working_y, atol=1e-12)
    np.testing.assert_allclose(model._resid, working_residual, atol=1e-12)
    known_bad = 1 - np.sum(working_residual**2) / np.sum((working_y-working_y.mean())**2)
    assert observed == pytest.approx(known_bad, abs=1e-10)
    raise _WorkingResponseR2(f'training R-squared uses re-centered working response: {observed!r}')


@pytest.mark.xfail(strict=True, raises=_MissingSmallCVDetails, reason="Issue #230: small ElasticNetCV result omits required details")
@pytest.mark.parametrize("n", [2, 3])
def test_small_elasticnet_cv_should_reject_clearly_or_return_coherent_result(n):
    X, y, _ = _weighted_problem()
    model = ElasticNetCV(alphas=[.1], cv=2, device="cpu")
    X_small, y_small = X[:n], y[:n]
    try:
        model.fit(X_small, y_small)
    except KeyError as error:
        origin = error.__traceback__
        while origin.tb_next is not None:
            origin = origin.tb_next
        source = linecache.getline(origin.tb_frame.f_code.co_filename, origin.tb_lineno).strip()
        if (error.args == ("std_mse",)
                and origin.tb_frame.f_code is ElasticNetCV._fit_cv.__code__
                and source == '"std_mse": details["std_mse"],'):
            raise _MissingSmallCVDetails('small-sample CV result omitted std_mse') from error
        raise  # Fixture, solver, and other KeyErrors must remain real failures.
    except ValueError:
        return  # Explicit domain rejection is an acceptable small-sample policy.
    assert np.isfinite(model.predict(X_small)).all()
    assert "std_mse" in model.cv_results_
