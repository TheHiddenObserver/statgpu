"""PR151 inverse-power Gamma smooth-domain closure regressions."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import get_glm_loss
from statgpu.linear_model import GammaRegression, PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_gamma import PenalizedGammaRegression
from statgpu.solvers import lbfgs_solver, newton_solver


def _data(seed=15157, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.06, size=(n, p))
    X[:, 0] = rng.uniform(0.85, 1.15, size=n)
    beta = np.array([0.92, 0.04, -0.03])
    eta = X @ beta
    y = (1.0 / eta) * rng.lognormal(0.0, 0.025, size=n)
    weights = np.linspace(0.6, 1.8, n)
    weights[::19] = 0.0
    return X.astype(np.float64), y.astype(np.float64), weights.astype(np.float64)


@pytest.mark.parametrize("solver_fn", [newton_solver, lbfgs_solver])
def test_direct_invalid_explicit_init_fails_before_objective(monkeypatch, solver_fn):
    X, y, weights = _data(n=24, p=2)
    loss = get_glm_loss("gamma", link="inverse_power")

    def forbidden(*args, **kwargs):
        raise AssertionError("objective primitive must not run for invalid explicit init")

    monkeypatch.setattr(loss, "fused_value_and_gradient", forbidden)
    monkeypatch.setattr(loss, "gradient", forbidden)
    monkeypatch.setattr(loss, "hessian", forbidden)

    with pytest.raises(ValueError, match="Explicit init_coef"):
        solver_fn(
            loss, None, X, y,
            init_coef=np.zeros(X.shape[1]),
            sample_weight=weights,
            max_iter=10,
        )


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_penalized_inverse_gamma_no_intercept_uses_shared_domain_start(solver):
    X, y, weights = _data()
    model = PenalizedGammaRegression(
        link="inverse_power",
        penalty="l2",
        alpha=0.02,
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=600,
        tol=1e-9,
        compute_inference=False,
    ).fit(X, y, sample_weight=weights)

    loss = get_glm_loss("gamma", link="inverse_power")
    lo, hi = loss._loss_domain_bounds(X)
    eta = X @ model.coef_
    assert np.all(eta > lo)
    assert np.all(eta < hi)
    assert model.intercept_ == 0.0


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_penalized_inverse_gamma_intercept_does_not_use_log_mean_start(solver):
    X, y, weights = _data(seed=15158)
    # Force mean(y) below one so the historical log-link warm start is negative
    # and therefore outside the inverse-link Gamma training domain.
    y = 0.55 * y / np.mean(y)
    assert np.log(np.mean(y)) < 0.0

    model = PenalizedGammaRegression(
        link="inverse_power",
        penalty="l2",
        alpha=0.01,
        fit_intercept=True,
        solver=solver,
        device="cpu",
        max_iter=600,
        tol=1e-9,
        compute_inference=False,
    ).fit(X, y, sample_weight=weights)

    eta = X @ model.coef_ + model.intercept_
    loss = get_glm_loss("gamma", link="inverse_power")
    lo, hi = loss._loss_domain_bounds(np.column_stack([X, np.ones(X.shape[0])]))
    active = weights > 0
    assert np.all(eta[active] > lo)
    assert np.all(eta[active] < hi)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_invalid_framework_warm_start_is_discarded_and_reseeded(solver):
    X, y, weights = _data(seed=15159)
    model = PenalizedGeneralizedLinearModel(
        loss="gamma",
        loss_kwargs={"link": "inverse_power"},
        penalty="l2",
        alpha=0.02,
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=600,
        tol=1e-9,
        compute_inference=False,
    )
    model._init_coef = -np.ones(X.shape[1])
    model.fit(X, y, sample_weight=weights)
    assert np.all(np.isfinite(model.coef_))
    assert np.all((X[weights > 0] @ model.coef_) > 0.0)


def test_inverse_gamma_cv_validation_uses_declared_link_not_log_link():
    X, y, weights = _data(seed=15160, n=20, p=2)
    cv = PenalizedGLM_CV(
        loss="gamma",
        loss_kwargs={"link": "inverse_power"},
        penalty="l2",
        alpha_grid=[0.1],
        cv=2,
        device="cpu",
        max_iter=100,
        tol=1e-8,
    )

    class Model:
        fit_intercept = True
        coef_ = np.array([0.82, 0.03])
        intercept_ = 0.08

    observed = cv._evaluate_single(
        Model(), X, y, sample_weight=weights
    )
    loss = get_glm_loss("gamma", link="inverse_power")
    design = np.column_stack([X, np.ones(X.shape[0])])
    params = np.concatenate([Model.coef_, [Model.intercept_]])
    expected = loss.value(design, y, params, sample_weight=weights)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)

    eta = X @ Model.coef_ + Model.intercept_
    log_link_value = np.average(eta + y * np.exp(-eta), weights=weights)
    assert abs(observed - log_link_value) > 1e-4


def test_inverse_gamma_smooth_l2_cv_preserves_link_through_selected_refit():
    X, y, weights = _data(seed=15161, n=60, p=2)
    cv = PenalizedGLM_CV(
        loss="gamma",
        loss_kwargs={"link": "inverse_power"},
        penalty="l2",
        alpha_grid=np.array([0.08, 0.02], dtype=np.float64),
        cv=3,
        random_state=151,
        device="cpu",
        max_iter=500,
        tol=1e-8,
    ).fit(X, y, sample_weight=weights)

    assert cv.alpha_ in {0.08, 0.02}
    assert getattr(cv.estimator_._loss, "link", None) == "inverse_power"
    eta = X @ cv.estimator_.coef_ + cv.estimator_.intercept_
    assert np.all(np.isfinite(eta))


def test_inverse_gamma_log_link_preservation():
    X, y, weights = _data(seed=15162, n=50, p=2)
    model = PenalizedGammaRegression(
        link="log", penalty="l2", alpha=0.02,
        fit_intercept=True, solver="lbfgs", device="cpu",
        max_iter=400, tol=1e-8, compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)


def test_ordinary_failed_domain_refit_does_not_publish_attempted_provenance():
    X, y, weights = _data(seed=15163, n=40, p=1)
    model = GammaRegression(
        link="inverse_power", fit_intercept=False, solver="lbfgs",
        device="cpu", max_iter=400, tol=1e-9,
    ).fit(X, y, sample_weight=weights)
    prior_solver = model._selected_solver
    prior_backend = model._selected_backend_name
    prior_device = model._selected_backend_device

    X_bad = np.array([[1.0], [-1.0]], dtype=np.float64)
    y_bad = np.ones(2, dtype=np.float64)
    with pytest.raises(RuntimeError, match="numerically certified smooth-domain start"):
        model.fit(X_bad, y_bad)

    assert model._selected_solver == prior_solver
    assert model._selected_backend_name == prior_backend
    assert model._selected_backend_device == prior_device
