"""Shared-estimator regressions for weighted GLM L-BFGS."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV, PenalizedGeneralizedLinearModel
from statgpu.solvers._convergence import ConvergenceWarning


def _count_data(seed=15201, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.4, size=(n, p)).astype(np.float64)
    beta = np.array([0.22, -0.16, 0.1])[:p]
    mu = np.exp(0.08 + X @ beta)
    y = rng.poisson(mu).astype(np.float64)
    return X, y


def _positive_data(seed=15202, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.3, size=(n, p)).astype(np.float64)
    beta = np.array([0.18, -0.12, 0.08])[:p]
    mu = np.exp(0.15 + X @ beta)
    y = mu * rng.lognormal(mean=0.0, sigma=0.12, size=n)
    return X, y.astype(np.float64)


_CV_CASES = (
    ("negative_binomial", _count_data, None),
    ("gamma", _positive_data, {"link": "log"}),
    ("inverse_gaussian", _positive_data, None),
)


def _fit_cv(loss, loss_kwargs, X, y, weights, *, compute_inference=False):
    kwargs = {}
    if compute_inference:
        kwargs.update(
            compute_inference=True,
            inference_method="auto",
            cov_type="hc0",
        )
    model = PenalizedGLM_CV(
        loss=loss,
        loss_kwargs=loss_kwargs,
        penalty="l2",
        alpha_grid=np.array([0.08, 0.035]),
        cv=2,
        random_state=17,
        solver="auto",
        device="cpu",
        max_iter=400,
        tol=1e-8,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.filterwarnings(
            "error",
            message="lbfgs_solver: line search failed.*",
            category=RuntimeWarning,
        )
        return model.fit(X, y, sample_weight=weights)


@pytest.mark.parametrize("loss,data_factory,loss_kwargs", _CV_CASES)
def test_weighted_penalized_explicit_lbfgs_supported_smooth_glm_consumers(
    loss, data_factory, loss_kwargs
):
    X, y = data_factory()
    weights = np.linspace(0.45, 1.75, X.shape[0], dtype=np.float64)

    model = PenalizedGeneralizedLinearModel(
        loss=loss,
        loss_kwargs=loss_kwargs,
        penalty="l2",
        alpha=0.03,
        solver="lbfgs",
        device="cpu",
        max_iter=600,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    assert model._selected_solver == "lbfgs"
    assert model._selected_backend_name == "numpy"
    assert np.all(np.isfinite(np.asarray(model.coef_)))
    assert np.isfinite(float(model.intercept_))


@pytest.mark.parametrize("loss", ["negative_binomial", "gamma", "inverse_gaussian"])
def test_weighted_penalized_lbfgs_global_weight_rescaling_invariance(loss):
    data_factory = _count_data if loss == "negative_binomial" else _positive_data
    X, y = data_factory(seed=15203)
    weights = np.linspace(0.5, 1.6, X.shape[0], dtype=np.float64)
    loss_kwargs = {"link": "log"} if loss == "gamma" else None

    def fit(w):
        return PenalizedGeneralizedLinearModel(
            loss=loss,
            loss_kwargs=loss_kwargs,
            penalty="l2",
            alpha=0.025,
            solver="lbfgs",
            device="cpu",
            max_iter=600,
            tol=1e-9,
        ).fit(X, y, sample_weight=w)

    a = fit(weights)
    b = fit(6.0 * weights)
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=3e-7, atol=3e-8)
    np.testing.assert_allclose(a.intercept_, b.intercept_, rtol=3e-7, atol=3e-8)


@pytest.mark.parametrize("loss,data_factory,loss_kwargs", _CV_CASES)
def test_weighted_smooth_glm_cv_auto_uses_lbfgs_for_cv_and_final_refit(
    loss, data_factory, loss_kwargs
):
    X, y = data_factory(seed=15204, n=84)
    weights = np.linspace(0.55, 1.55, X.shape[0], dtype=np.float64)

    cv = _fit_cv(loss, loss_kwargs, X, y, weights)

    assert cv.get_params(deep=False)["solver"] == "auto"
    assert cv._solver_for_cv("cpu", X=X) == "lbfgs"
    assert cv.estimator_._selected_solver == "lbfgs"
    assert cv.estimator_._selected_backend_name == "numpy"
    assert np.all(np.isfinite(np.asarray(cv.estimator_.coef_)))
    assert np.isfinite(float(cv.estimator_.intercept_))


@pytest.mark.parametrize("loss,data_factory,loss_kwargs", _CV_CASES)
def test_weighted_smooth_glm_cv_global_weight_rescaling_preserves_selection_and_refit(
    loss, data_factory, loss_kwargs
):
    X, y = data_factory(seed=15205, n=84)
    weights = np.linspace(0.5, 1.6, X.shape[0], dtype=np.float64)

    a = _fit_cv(loss, loss_kwargs, X, y, weights)
    b = _fit_cv(loss, loss_kwargs, X, y, 7.0 * weights)

    assert a.alpha_ == b.alpha_
    assert a.estimator_._selected_solver == b.estimator_._selected_solver == "lbfgs"
    np.testing.assert_allclose(
        a.estimator_.coef_, b.estimator_.coef_, rtol=5e-6, atol=5e-7
    )
    np.testing.assert_allclose(
        a.estimator_.intercept_, b.estimator_.intercept_, rtol=5e-6, atol=5e-7
    )


def test_weighted_negative_binomial_cv_inference_remains_final_refit_only():
    X, y = _count_data(seed=15206, n=84)
    weights = np.linspace(0.55, 1.55, X.shape[0], dtype=np.float64)

    cv = _fit_cv(
        "negative_binomial",
        None,
        X,
        y,
        weights,
        compute_inference=True,
    )

    assert cv.estimator_._selected_solver == "lbfgs"
    assert cv.inference_method_ == "m_estimation"
    assert cv.penalty_conditioning_ == "cv_selected_penalty"
    assert cv.penalty_selection_adjusted_ is False
    assert np.all(np.isfinite(np.asarray(cv._bse)))
