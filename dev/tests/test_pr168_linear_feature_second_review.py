"""Safe workflows and disclosures for the second linear/selection doc review.

Known runtime limitations are observed, not asserted as desired behavior. When
those algorithms are repaired, their issue regressions should enforce rejection
or correctness directly; these tests ensure current docs do not hide them.
"""
from pathlib import Path

import numpy as np
import pytest

from statgpu import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LinearRegression,
    Ridge,
    StepwiseSelector,
)

ROOT = Path(__file__).resolve().parents[2]


def _page(language, directory, name):
    return (ROOT / 'docs' / language / directory / (name + '.md')).read_text()


@pytest.mark.parametrize('direction', ['forward', 'backward', 'both'])
def test_finite_stepwise_scores_respect_cap_and_parameter_lifecycle(direction):
    rng = np.random.default_rng(22)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] + rng.normal(scale=0.1, size=50)
    selector = StepwiseSelector(LinearRegression, direction=direction, max_features=1,
                                device='cpu', compute_inference=False).fit(X, y)
    assert len(selector.selected_features_) <= 1
    assert np.isfinite(selector.aic_history_).all()
    assert np.isfinite(selector.bic_history_).all()
    predictions = selector.predict(X).copy()
    with pytest.raises(ValueError):
        selector.set_params(direction='invalid')
    np.testing.assert_array_equal(selector.predict(X), predictions)
    assert selector.set_params() is selector
    assert selector.set_params(criterion='bic') is selector
    with pytest.raises(RuntimeError):
        selector.predict(X)
    assert selector.get_params(deep=True) == selector.get_params(deep=False)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_nonfinite_stepwise_score_outcomes_have_explicit_caveats(language):
    class Criteria:
        def fit(self, X, y):
            self.aic = self.bic = np.inf if X.shape[1] == 0 else 1.
            return self

    class NoCriteria:
        def fit(self, X, y):
            self.aic = self.bic = np.inf
            return self

    X = np.arange(12.).reshape(6, 2)
    y = X[:, 0]
    forward = StepwiseSelector(Criteria, direction='forward').fit(X, y)
    backward = StepwiseSelector(NoCriteria, direction='backward', max_features=1).fit(X, y)
    if not np.isfinite(forward.aic_history_).all() or len(backward.selected_features_) > 1:
        text = _page(language, 'models', 'feature-selection')
        reference = _page(language, 'reference', 'feature-selection-api')
        assert 'nonfinite-score-limitations' in reference
        assert 'max_features' in text
        assert ('infinite starting' in StepwiseSelector.__doc__)
        assert ('infinite' in text if language == 'en' else '无穷大' in text)
        assert ('discard' in text if language == 'en' else '弃用' in text)


def test_stepwise_model_keywords_are_validated_at_fit_not_set_params():
    selector = StepwiseSelector(LinearRegression, device='cpu', compute_inference=False)
    assert selector.set_params(unknown_model_parameter=1) is selector
    assert selector.get_params()['unknown_model_parameter'] == 1
    with pytest.raises(TypeError, match='unknown_model_parameter'):
        selector.fit(np.arange(6.)[:, None], np.arange(6.))


@pytest.mark.parametrize('fit_intercept', [False, True])
def test_explicit_ridge_grid_matches_selected_direct_objective(fit_intercept):
    rng = np.random.default_rng(15)
    X = rng.normal(size=(120, 2))
    y = 2 * X[:, 0] + rng.normal(scale=.1, size=120)
    model = ElasticNetCV(l1_ratio=0., alphas=[.0001, .001, .01, .1, 1., 10.],
                         cv=3, random_state=2, device='cpu', tol=1e-8,
                         max_iter=3000, fit_intercept=fit_intercept).fit(X[:90], y[:90])
    direct = ElasticNet(l1_ratio=0., alpha=model.alpha_, device='cpu', tol=1e-8,
                        max_iter=3000, fit_intercept=fit_intercept).fit(X[:90], y[:90])
    np.testing.assert_allclose(model.predict(X), direct.predict(X), atol=1e-10)
    assert model.score(X[90:], y[90:]) > .99
    assert model.best_score_ == pytest.approx(-np.min(model.cv_results_['mean_mse']))


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_automatic_ridge_grid_limit_is_disclosed(language):
    for directory, page in [('models', 'elastic-net'), ('reference', 'linear-model-api')]:
        text = _page(language, directory, page)
        assert 'max(l1_ratio, 1e-6)' in text
        assert 'fit_intercept' in text
        assert 'alphas' in text
    assert 'excessively large Ridge penalties' in ElasticNetCV.__doc__


@pytest.mark.parametrize('cls', [ElasticNet, Lasso, Ridge])
def test_valid_weighted_r2_matches_definition_and_is_weight_scale_invariant(cls):
    X = np.arange(6.)[:, None]
    y = 2 + 3 * X[:, 0]
    model = cls(alpha=.1, device='cpu', compute_inference=False).fit(X, y)
    weights = np.array([0., .5, 1., 1.5, 2., 3.])
    residual = y - model.predict(X)
    expected = 1 - np.sum(weights * residual**2) / np.sum(weights * (y - np.average(y, weights=weights))**2)
    assert model.score(X, y, sample_weight=weights) == pytest.approx(expected)
    assert model.score(X, y, sample_weight=7 * weights) == pytest.approx(expected)


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_negative_score_weight_limitation_is_not_hidden(language):
    X = np.arange(3.)[:, None]
    model = ElasticNet(alpha=0, device='cpu', compute_inference=False).fit(X, X[:, 0])
    try:
        score = model.score(X, np.array([2., 1., 2.]), sample_weight=[-.1, 1., 1.])
    except ValueError:
        return  # A future fail-closed validation fix is welcome.
    if score > 1:
        for directory, page in [('models', 'elastic-net'), ('reference', 'linear-model-api')]:
            text = _page(language, directory, page)
            assert ('negative weights' in text if language == 'en' else '负权重' in text)
        assert 'invalid R² above 1' in model.score.__doc__


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_ols_rank_formula_uses_weighted_design_with_zero_weights(language):
    X = np.column_stack([np.arange(6.), [0., 0., 0., 0., 0., 1.]])
    weights = np.array([1., 1., 1., 1., 1., 0.])
    design = np.column_stack([np.ones(6), X])
    model = LinearRegression(device='cpu', compute_inference=False).fit(
        X, np.arange(6.), sample_weight=weights)
    assert np.linalg.matrix_rank(design) == 3
    assert model.rank_ == np.linalg.matrix_rank(np.sqrt(weights)[:, None] * design) == 2
    text = _page(language, 'models', 'linear-regression')
    assert r'\operatorname{rank}(W^{1/2}D)' in text


def test_residual_bootstrap_refits_full_design_with_fixed_tuning(monkeypatch):
    from statgpu.linear_model import (
        _gaussian_residual_bootstrap_backend_contract as contract,
    )

    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 5))
    y = x @ np.array([1.0, 0.2, 0.0, 0.0, 0.0]) + rng.normal(size=50)
    children = []
    original = contract._make_child_refit

    def capture(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(contract, "_make_child_refit", capture)
    model = ElasticNet(
        alpha=0.2, l1_ratio=0.8, device="cpu", compute_inference=True,
        inference_method="bootstrap", tol=1e-7, max_iter=3000,
    )
    model.n_bootstrap = 20
    model.bootstrap_random_state = 7
    model.fit(x, y)
    assert len(children) == model.n_bootstrap
    parent_support = model.coef_ != 0
    for child in children:
        assert child.coef_.shape == (x.shape[1],)
        assert child.alpha == model.alpha
        assert child.l1_ratio == model.l1_ratio
    assert any(np.any((child.coef_ != 0) != parent_support) for child in children)


def test_knockpy_compatibility_fallbacks_have_observable_metadata(monkeypatch):
    import sys

    from statgpu.feature_selection import model_x_knockoff_filter

    rng = np.random.default_rng(44)
    x = rng.normal(size=(80, 4))
    y = x[:, 0] + rng.normal(size=80)
    monkeypatch.setitem(sys.modules, "knockpy", None)
    monkeypatch.setitem(sys.modules, "sklearn", None)
    result = model_x_knockoff_filter(
        x, y, backend="numpy", compat_mode="knockpy", method="corr_diff",
        modelx_draws=1, random_state=7, modelx_smatrix_method="mvr",
    )
    assert result.metadata["modelx_smatrix_method"] == "mvr"
    assert result.metadata["modelx_smatrix_source"] == "equicorrelated_fallback"
    assert result.metadata["modelx_covariance_estimator"] == "mle_fallback_no_sklearn"
    assert np.isfinite(result.W).all()
