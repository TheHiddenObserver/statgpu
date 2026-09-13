"""PR151 inverse-power Gamma smooth-domain closure regressions."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import GammaLoss, get_glm_loss
from statgpu.linear_model import GammaRegression, PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized._penalized_gamma import PenalizedGammaRegression
from statgpu.solvers import lbfgs_solver, newton_solver
from statgpu.solvers._smooth_domain import _LossDomainError


def _data(seed=15157, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.06, size=(n, p))
    X[:, 0] = rng.uniform(0.85, 1.15, size=n)
    beta = np.zeros(p, dtype=np.float64)
    beta[0] = 0.92
    if p > 1:
        beta[1] = 0.04
    if p > 2:
        beta[2] = -0.03
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

    assert model.link == "inverse_power"
    assert model.loss_kwargs is None
    assert getattr(model._loss, "link", None) == "inverse_power"
    assert model._selected_solver == solver
    loss = get_glm_loss("gamma", link="inverse_power")
    lo, hi = loss._loss_domain_bounds(X)
    eta = X @ model.coef_
    assert np.all(eta > lo)
    assert np.all(eta < hi)
    assert model.intercept_ == 0.0


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_penalized_inverse_gamma_intercept_does_not_use_log_mean_start(solver):
    X, y, weights = _data(seed=15158)
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

    assert model.link == "inverse_power"
    assert model.loss_kwargs is None
    assert getattr(model._loss, "link", None) == "inverse_power"
    assert model._selected_solver == solver
    assert getattr(model._fit_loss_backend, "_statgpu_inverse_gamma_domain_penalized", False)

    eta = X @ model.coef_ + model.intercept_
    loss = get_glm_loss("gamma", link="inverse_power")
    lo, hi = loss._loss_domain_bounds(np.column_stack([X, np.ones(X.shape[0])]))
    active = weights > 0
    assert np.all(eta[active] > lo), (
        model.coef_, model.intercept_, model._params, eta[active].min()
    )
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

    observed = cv._evaluate_single(Model(), X, y, sample_weight=weights)
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


def test_typed_penalized_gamma_clone_preserves_public_and_internal_link_contract():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = PenalizedGammaRegression(
        link="inverse_power",
        loss_kwargs=None,
        penalty="l2",
        alpha=0.02,
        fit_intercept=False,
        solver="lbfgs",
        device="cpu",
        compute_inference=False,
    )
    params = model.get_params(deep=False)
    assert params["link"] == "inverse_power"
    assert params["loss_kwargs"] is None
    assert model._resolve_loss().link == "inverse_power"

    cloned = clone(model)
    cloned_params = cloned.get_params(deep=False)
    assert cloned_params["link"] == "inverse_power"
    assert cloned_params["loss_kwargs"] is None
    assert cloned._resolve_loss().link == "inverse_power"


def test_typed_penalized_gamma_preserves_legacy_loss_kwargs_link_precedence():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    supplied = {"link": "inverse_power"}
    model = PenalizedGammaRegression(loss_kwargs=supplied)
    assert model.link == "log"
    assert model.loss_kwargs is supplied
    assert model._resolve_loss().link == "inverse_power"
    cloned = clone(model)
    assert cloned.link == "log"
    assert cloned.loss_kwargs == supplied
    assert cloned._resolve_loss().link == "inverse_power"

    explicit_kwargs_win = PenalizedGammaRegression(
        link="inverse_power", loss_kwargs={"link": "log"}
    )
    assert explicit_kwargs_win._resolve_loss().link == "log"


def test_penalized_inverse_gamma_weighted_m_estimation_inference():
    X, y, weights = _data(seed=15164, n=80, p=2)
    model = PenalizedGammaRegression(
        link="inverse_power",
        penalty="l2",
        alpha=0.025,
        fit_intercept=True,
        solver="newton",
        device="cpu",
        max_iter=600,
        tol=1e-9,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y, sample_weight=weights)

    assert model.inference_resolved_method_ == "m_estimation"
    assert model._inference_result.method == "m_estimation"
    assert getattr(model._loss, "link", None) == "inverse_power"
    assert np.all(np.isfinite(np.asarray(model._bse)))
    assert np.all(np.isfinite(np.asarray(model._pvalues)))


def test_inverse_gamma_cv_inference_runs_on_selected_final_refit_only(monkeypatch):
    X, y, weights = _data(seed=15165, n=72, p=2)
    calls = []
    original_fit = PenalizedGeneralizedLinearModel.fit

    def recording_fit(self, *args, **kwargs):
        if bool(getattr(self, "compute_inference", False)):
            calls.append((float(self.alpha), getattr(self, "loss_kwargs", None)))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(PenalizedGeneralizedLinearModel, "fit", recording_fit)
    cv = PenalizedGLM_CV(
        loss="gamma",
        loss_kwargs={"link": "inverse_power"},
        penalty="l2",
        alpha_grid=np.array([0.08, 0.03]),
        cv=2,
        random_state=151,
        device="cpu",
        max_iter=500,
        tol=1e-8,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y, sample_weight=weights)

    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(cv.alpha_)
    assert calls[0][1]["link"] == "inverse_power"
    assert cv._inference_result is cv.estimator_._inference_result
    assert cv.inference_method_ == "m_estimation"
    assert cv._inference_result.metadata["selected_alpha"] == pytest.approx(cv.alpha_)


def test_ordinary_inverse_gamma_no_intercept_formula_matches_array_route():
    pd = pytest.importorskip("pandas")
    X, y, weights = _data(seed=15166, n=64, p=2)
    direct = GammaRegression(
        link="inverse_power", fit_intercept=False, solver="lbfgs",
        device="cpu", max_iter=500, tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    data = pd.DataFrame({"y": y, "x0": X[:, 0], "x1": X[:, 1]})
    formula = GammaRegression(
        link="inverse_power", fit_intercept=True, solver="lbfgs",
        device="cpu", max_iter=500, tol=1e-9,
    ).fit(formula="y ~ 0 + x0 + x1", data=data, sample_weight=weights)

    assert formula.intercept_ == 0.0
    np.testing.assert_allclose(formula.coef_, direct.coef_, rtol=2e-8, atol=2e-10)


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


def test_penalized_domain_failure_invalidates_prior_fit_and_inference():
    X, y, weights = _data(seed=15167, n=60, p=1)
    model = PenalizedGammaRegression(
        link="inverse_power", penalty="l2", alpha=0.02,
        fit_intercept=False, solver="newton", device="cpu",
        max_iter=500, tol=1e-9, compute_inference=True,
        inference_method="auto", cov_type="hc0",
    ).fit(X, y, sample_weight=weights)
    assert model._fitted and model._inference_result is not None

    X_bad = np.array([[1.0], [-1.0]], dtype=np.float64)
    y_bad = np.ones(2, dtype=np.float64)
    with pytest.raises(RuntimeError, match="numerically certified smooth-domain start"):
        model.fit(X_bad, y_bad)

    assert not model._fitted
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._inference_result is None
    assert model._selected_solver is None
    assert model._selected_backend_name is None


def test_inverse_gamma_cv_domain_failure_resets_selection_state(monkeypatch):
    X, y, weights = _data(seed=15168, n=60, p=2)
    cv = PenalizedGLM_CV(
        loss="gamma", loss_kwargs={"link": "inverse_power"}, penalty="l2",
        alpha_grid=np.array([0.08, 0.03]), cv=2, random_state=4,
        device="cpu", max_iter=400, tol=1e-8,
    ).fit(X, y, sample_weight=weights)
    assert cv._fitted and cv.alpha_ is not None and cv.estimator_ is not None

    def fail_domain(self, X_arg, y_arg, sample_weight=None):
        raise _LossDomainError("no numerically certified smooth-domain start")

    monkeypatch.setattr(GammaLoss, "_loss_domain_initial_point", fail_domain)
    with pytest.raises(RuntimeError, match="numerically certified smooth-domain start"):
        cv.fit(X, y, sample_weight=weights)

    assert not cv._fitted
    assert cv.alpha_ is None
    assert cv.best_score_ is None
    assert cv.cv_results_ is None
    assert cv.estimator_ is None
    assert cv.coef_ is None
    assert cv.intercept_ is None
