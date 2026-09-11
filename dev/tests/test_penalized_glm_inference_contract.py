"""Contract tests for the 0.2.6-targeted penalized-GLM inference repair."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from statgpu.linear_model import (
    ElasticNet,
    Lasso,
    PenalizedGLM_CV,
    PenalizedGammaRegression,
    PenalizedGeneralizedLinearModel,
    PenalizedInverseGaussianRegression,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedNegativeBinomialRegression,
    PenalizedPoissonRegression,
    PenalizedTweedieRegression,
    Ridge,
)


def _logistic_data(seed=137, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([0.65, -0.45, 0.30])[:p]
    eta = -0.15 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)
    y[0] = 0.0
    y[1] = 1.0
    return X, y


def _poisson_data(seed=211, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.55, size=(n, p))
    beta = np.array([0.22, -0.17, 0.13])[:p]
    mu = np.exp(0.20 + X @ beta)
    y = rng.poisson(mu).astype(float)
    return X, y


def test_public_defaults_are_auto_without_changing_sparse_wrapper_defaults():
    auto_classes = (
        PenalizedGeneralizedLinearModel,
        PenalizedLinearRegression,
        PenalizedLogisticRegression,
        PenalizedPoissonRegression,
        PenalizedGammaRegression,
        PenalizedInverseGaussianRegression,
        PenalizedNegativeBinomialRegression,
        PenalizedTweedieRegression,
    )
    for cls in auto_classes:
        assert inspect.signature(cls).parameters["inference_method"].default == "auto"

    assert inspect.signature(Lasso).parameters["inference_method"].default == "debiased"
    assert inspect.signature(ElasticNet).parameters["inference_method"].default == "debiased"

    cv_sig = inspect.signature(PenalizedGLM_CV)
    assert cv_sig.parameters["compute_inference"].default is False
    assert cv_sig.parameters["inference_method"].default == "auto"
    assert cv_sig.parameters["cov_type"].default == "nonrobust"
    assert cv_sig.parameters["hac_maxlags"].default is None


def test_logistic_l2_auto_resolves_and_reports_m_estimation():
    X, y = _logistic_data()
    model = PenalizedLogisticRegression(
        penalty="l2",
        alpha=0.08,
        solver="irls",
        device="cpu",
        compute_inference=True,
        cov_type="hc0",
    ).fit(X, y)

    assert model.inference_method == "auto"
    assert model.inference_requested_method_ == "auto"
    assert model.inference_resolved_method_ == "m_estimation"
    assert model.inference_method_ == "m_estimation"
    assert model._inference_result.method == "m_estimation"
    assert model.inference_target_ == "penalized_estimating_equation"
    assert model.penalty_conditioning_ == "fixed_penalty"
    assert model.penalty_selection_adjusted_ is None
    assert np.all(np.isfinite(np.asarray(model._bse)))

    metadata = model._inference_result.metadata
    assert metadata["inference_requested_method"] == "auto"
    assert metadata["inference_resolved_method"] == "m_estimation"
    assert metadata["inference_target"] == "penalized_estimating_equation"
    assert metadata["numerical_backend"] == "numpy"
    assert metadata["numerical_device"] == "cpu"
    assert metadata["reporting_backend"] == "numpy"


def test_explicit_m_estimation_matches_auto_for_logistic_l2():
    X, y = _logistic_data(seed=141)
    common = dict(
        penalty="l2",
        alpha=0.05,
        solver="irls",
        device="cpu",
        compute_inference=True,
        cov_type="hc1",
    )
    auto = PenalizedLogisticRegression(**common).fit(X, y)
    explicit = PenalizedLogisticRegression(
        **common, inference_method="m_estimation"
    ).fit(X, y)

    np.testing.assert_allclose(auto.coef_, explicit.coef_, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(auto._bse, explicit._bse, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(auto._pvalues, explicit._pvalues, rtol=1e-10, atol=1e-12)
    assert explicit.inference_requested_method_ == "m_estimation"
    assert explicit.inference_resolved_method_ == "m_estimation"


def test_historical_debiased_l2_spelling_warns_and_resolves_truthfully():
    X, y = _logistic_data(seed=151)
    model = PenalizedLogisticRegression(
        penalty="l2",
        alpha=0.05,
        solver="irls",
        device="cpu",
        compute_inference=True,
        inference_method="debiased",
    )
    with pytest.warns(FutureWarning, match="L2 inference was not debiased-Lasso"):
        model.fit(X, y)

    assert model.inference_method == "debiased"
    assert model.inference_requested_method_ == "debiased"
    assert model.inference_resolved_method_ == "m_estimation"
    assert model.inference_method_ == "m_estimation"
    assert model._inference_result.metadata["legacy_inference_alias"] == "debiased"


@pytest.mark.parametrize("penalty", ["l1", "elasticnet"])
def test_non_gaussian_sparse_inference_fails_closed(penalty):
    X, y = _logistic_data(seed=163)
    model = PenalizedLogisticRegression(
        penalty=penalty,
        alpha=0.06,
        l1_ratio=0.4,
        device="cpu",
        compute_inference=True,
        inference_method="auto",
        max_iter=300,
    )
    with pytest.raises(NotImplementedError, match="no statistically defined automatic inference"):
        model.fit(X, y)
    assert not getattr(model, "_fitted", False)
    assert model._inference_result is None


def test_failed_refit_clears_prior_successful_inference_state():
    X, y = _logistic_data(seed=167)
    model = PenalizedLogisticRegression(
        penalty="l2",
        alpha=0.05,
        device="cpu",
        solver="irls",
        compute_inference=True,
        inference_method="auto",
    ).fit(X, y)
    assert model._fitted and model._inference_result is not None

    model.set_params(penalty="l1", inference_method="auto")
    with pytest.raises(NotImplementedError):
        model.fit(X, y)

    assert not model._fitted
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._inference_result is None
    assert model.inference_method_ is None
    assert model.inference_resolved_method_ is None


def test_non_gaussian_bootstrap_is_rejected_before_resampling():
    X, y = _poisson_data()
    model = PenalizedPoissonRegression(
        penalty="l1",
        alpha=0.04,
        device="cpu",
        compute_inference=True,
        inference_method="bootstrap",
        max_iter=300,
    )
    with pytest.raises(NotImplementedError, match="Gaussian residual bootstrap only"):
        model.fit(X, y)
    assert model._inference_result is None


@pytest.mark.parametrize("cov_type", ["hc2", "hc3", "hac"])
def test_non_gaussian_m_estimation_covariance_scope_fails_closed(cov_type):
    X, y = _logistic_data(seed=173)
    model = PenalizedLogisticRegression(
        penalty="l2",
        alpha=0.05,
        device="cpu",
        solver="irls",
        compute_inference=True,
        inference_method="m_estimation",
        cov_type=cov_type,
    )
    with pytest.raises(NotImplementedError, match="nonrobust/HC0/HC1|nonrobust.*hc0.*hc1"):
        model.fit(X, y)


def test_weighted_poisson_l2_m_estimation_is_supported():
    X, y = _poisson_data(seed=181)
    weights = np.linspace(0.4, 1.8, X.shape[0])
    model = PenalizedPoissonRegression(
        penalty="l2",
        alpha=0.03,
        device="cpu",
        solver="irls",
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y, sample_weight=weights)

    assert model.inference_method_ == "m_estimation"
    assert np.all(np.isfinite(np.asarray(model._bse)))
    assert np.all(np.isfinite(np.asarray(model._pvalues)))


def test_logistic_l2_hc0_matches_independent_sandwich_algebra():
    X, y = _logistic_data(seed=191, n=128)
    alpha = 0.07
    model = PenalizedLogisticRegression(
        penalty="l2",
        alpha=alpha,
        device="cpu",
        solver="irls",
        compute_inference=True,
        inference_method="m_estimation",
        cov_type="hc0",
    ).fit(X, y)

    Xd = np.column_stack([np.ones(X.shape[0]), X])
    params = np.concatenate([[model.intercept_], np.asarray(model.coef_)])
    loss = model._loss
    H = np.asarray(loss.hessian(Xd, y, params), dtype=float)
    curvature = np.zeros(params.shape[0], dtype=float)
    curvature[1:] = alpha
    bread = np.linalg.inv(H + np.diag(curvature))
    J = np.asarray(loss.score_outer(Xd, y, params), dtype=float) / X.shape[0]
    cov = bread @ J @ bread / X.shape[0]
    expected_bse = np.sqrt(np.maximum(np.diag(cov), 0.0))

    np.testing.assert_allclose(model._bse, expected_bse, rtol=5e-9, atol=5e-11)


def test_gaussian_elasticnet_bootstrap_preserves_refit_penalty(monkeypatch):
    rng = np.random.default_rng(223)
    X = rng.normal(size=(72, 3))
    y = 0.4 + X @ np.array([0.8, -0.5, 0.25]) + rng.normal(scale=0.35, size=72)

    observed = []
    original_fit = PenalizedLinearRegression.fit

    def recording_fit(self, *args, **kwargs):
        observed.append(str(getattr(self.penalty, "name", self.penalty)).lower())
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(PenalizedLinearRegression, "fit", recording_fit)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="elasticnet",
        alpha=0.05,
        l1_ratio=0.35,
        device="cpu",
        compute_inference=True,
        inference_method="bootstrap",
        cov_type="nonrobust",
        max_iter=500,
    )
    model.n_bootstrap = 6
    model.bootstrap_random_state = 7
    model.fit(X, y)

    assert observed and all(name in ("elasticnet", "en") for name in observed)
    assert model.inference_resolved_method_ == "residual_bootstrap"
    assert model._inference_result.metadata["refit_penalty"] in ("elasticnet", "en")


def test_weighted_gaussian_bootstrap_fails_closed_and_invalidates_fit():
    rng = np.random.default_rng(227)
    X = rng.normal(size=(64, 3))
    y = X @ np.array([0.6, -0.3, 0.2]) + rng.normal(scale=0.3, size=64)
    weights = np.linspace(0.5, 1.5, 64)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l1",
        alpha=0.04,
        device="cpu",
        compute_inference=True,
        inference_method="bootstrap",
        max_iter=400,
    )
    model.n_bootstrap = 4
    with pytest.raises(NotImplementedError, match="Weighted Gaussian residual-bootstrap"):
        model.fit(X, y, sample_weight=weights)
    assert not getattr(model, "_fitted", False)
    assert model.coef_ is None
    assert model._inference_result is None


def test_penalized_glm_cv_runs_inference_only_on_selected_final_refit(monkeypatch):
    X, y = _logistic_data(seed=233, n=84)
    inference_fit_calls = []
    original_fit = PenalizedGeneralizedLinearModel.fit

    def recording_fit(self, *args, **kwargs):
        if bool(getattr(self, "compute_inference", False)):
            inference_fit_calls.append((float(self.alpha), str(self.inference_method)))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(PenalizedGeneralizedLinearModel, "fit", recording_fit)
    cv = PenalizedGLM_CV(
        loss="logistic",
        penalty="l2",
        alpha_grid=np.array([0.12, 0.05]),
        cv=2,
        random_state=3,
        device="cpu",
        solver="irls",
        max_iter=300,
        tol=1e-6,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y)

    assert len(inference_fit_calls) == 1
    assert inference_fit_calls[0][0] == pytest.approx(cv.alpha_)
    assert cv._inference_result is cv.estimator_._inference_result
    assert cv.inference_method_ == "m_estimation"
    assert cv.penalty_conditioning_ == "cv_selected_penalty"
    assert cv.penalty_selection_adjusted_ is False
    assert cv.estimator_.penalty_conditioning_ == "cv_selected_penalty"
    assert cv.estimator_.penalty_selection_adjusted_ is False
    assert cv._inference_result.metadata["selected_alpha"] == pytest.approx(cv.alpha_)


def test_cv_clone_surface_includes_new_inference_controls():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = PenalizedGLM_CV(
        loss="poisson",
        penalty="l2",
        alpha_grid=np.array([0.2, 0.1]),
        compute_inference=True,
        inference_method="m_estimation",
        cov_type="hc1",
    )
    cloned = clone(model)
    params = cloned.get_params(deep=False)
    assert params["compute_inference"] is True
    assert params["inference_method"] == "m_estimation"
    assert params["cov_type"] == "hc1"


def test_formula_and_array_routes_match_for_logistic_l2_inference():
    pd = pytest.importorskip("pandas")
    X, y = _logistic_data(seed=239, n=90)
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    frame["y"] = y

    array_model = PenalizedGeneralizedLinearModel(
        loss="logistic",
        penalty="l2",
        alpha=0.06,
        device="cpu",
        solver="irls",
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y)
    formula_model = PenalizedGeneralizedLinearModel(
        loss="logistic",
        penalty="l2",
        alpha=0.06,
        device="cpu",
        solver="irls",
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(formula="y ~ x1 + x2 + x3", data=frame)

    np.testing.assert_allclose(array_model._params, formula_model._params, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(array_model._bse, formula_model._bse, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(array_model._pvalues, formula_model._pvalues, rtol=1e-8, atol=1e-10)
    assert formula_model.inference_requested_method_ == "auto"
    assert formula_model.inference_resolved_method_ == "m_estimation"


def test_ridge_lasso_elasticnet_specialized_surfaces_remain_usable():
    rng = np.random.default_rng(241)
    X = rng.normal(size=(80, 3))
    y = 0.2 + X @ np.array([0.7, -0.4, 0.2]) + rng.normal(scale=0.4, size=80)

    ridge = Ridge(alpha=0.1, device="cpu", compute_inference=True).fit(X, y)
    assert ridge._inference_result is not None

    lasso = Lasso(alpha=0.05, device="cpu", compute_inference=True, max_iter=500).fit(X, y)
    elastic = ElasticNet(
        alpha=0.05,
        l1_ratio=0.4,
        device="cpu",
        compute_inference=True,
        max_iter=500,
    ).fit(X, y)
    assert lasso.inference_method == "debiased"
    assert elastic.inference_method == "debiased"
    assert lasso._inference_result.method == "debiased"
    assert elastic._inference_result.method == "debiased"
