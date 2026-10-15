"""Analytic/external references for PR151 inverse-power Gamma closure."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statgpu.glm_core import get_glm_loss
from statgpu.linear_model import GammaRegression


def _fixture(seed=15171, n=120, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.07, size=(n, p)).astype(np.float64)
    X[:, 0] = rng.uniform(0.85, 1.15, size=n)
    beta = np.array([0.95, 0.04, -0.025], dtype=np.float64)[:p]
    eta = X @ beta
    y = (1.0 / eta) * rng.lognormal(0.0, 0.03, size=n)
    w = np.linspace(0.6, 1.7, n, dtype=np.float64)
    w[::23] = 0.0
    return X, y.astype(np.float64), w


def test_inverse_gamma_weighted_value_gradient_hessian_are_one_smooth_objective():
    X, y, w = _fixture(n=40, p=3)
    loss = get_glm_loss("gamma", link="inverse_power")
    beta = np.array([0.92, 0.03, -0.02], dtype=np.float64)
    assert loss._loss_domain_is_feasible(X, beta, sample_weight=w)

    analytic_g = np.asarray(loss.gradient(X, y, beta, sample_weight=w))
    analytic_h = np.asarray(loss.hessian(X, y, beta, sample_weight=w))

    eps_g = 2.0e-6
    numeric_g = np.empty_like(beta)
    for j in range(beta.size):
        step = np.zeros_like(beta)
        step[j] = eps_g
        numeric_g[j] = (
            loss.value(X, y, beta + step, sample_weight=w)
            - loss.value(X, y, beta - step, sample_weight=w)
        ) / (2.0 * eps_g)
    np.testing.assert_allclose(analytic_g, numeric_g, rtol=2e-6, atol=2e-8)

    eps_h = 4.0e-6
    numeric_h = np.empty_like(analytic_h)
    for j in range(beta.size):
        step = np.zeros_like(beta)
        step[j] = eps_h
        numeric_h[:, j] = (
            loss.gradient(X, y, beta + step, sample_weight=w)
            - loss.gradient(X, y, beta - step, sample_weight=w)
        ) / (2.0 * eps_h)
    np.testing.assert_allclose(analytic_h, numeric_h, rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(analytic_h, analytic_h.T, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_inverse_gamma_solver_never_evaluates_outside_smooth_domain(monkeypatch, solver):
    X, y, w = _fixture(seed=15172, n=60, p=3)
    loss = get_glm_loss("gamma", link="inverse_power")

    checked = {"fvg": 0, "grad": 0, "hess": 0}
    original_fvg = loss.fused_value_and_gradient
    original_grad = loss.gradient
    original_hess = loss.hessian

    def _assert_feasible(coef):
        assert loss._loss_domain_is_feasible(X, coef, sample_weight=w)

    def guarded_fvg(X_arg, y_arg, coef, sample_weight=None):
        _assert_feasible(coef)
        checked["fvg"] += 1
        return original_fvg(X_arg, y_arg, coef, sample_weight=sample_weight)

    def guarded_grad(X_arg, y_arg, coef, sample_weight=None):
        _assert_feasible(coef)
        checked["grad"] += 1
        return original_grad(X_arg, y_arg, coef, sample_weight=sample_weight)

    def guarded_hess(X_arg, y_arg, coef, sample_weight=None):
        _assert_feasible(coef)
        checked["hess"] += 1
        return original_hess(X_arg, y_arg, coef, sample_weight=sample_weight)

    monkeypatch.setattr(loss, "fused_value_and_gradient", guarded_fvg)
    monkeypatch.setattr(loss, "gradient", guarded_grad)
    monkeypatch.setattr(loss, "hessian", guarded_hess)

    from statgpu.solvers import lbfgs_solver, newton_solver
    solver_fn = newton_solver if solver == "newton" else lbfgs_solver
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coef, _ = solver_fn(
            loss, None, X, y,
            max_iter=500, tol=1e-9, sample_weight=w,
        )
    assert loss._loss_domain_is_feasible(X, coef, sample_weight=w)
    assert checked["fvg"] > 0
    if solver == "newton":
        assert checked["grad"] > 0
        assert checked["hess"] > 0


@pytest.mark.parametrize("weighted", [False, True])
def test_inverse_gamma_no_intercept_matches_statsmodels_reference(weighted):
    sm = pytest.importorskip("statsmodels.api")
    X, y, w = _fixture(seed=15173, n=100, p=3)

    links = sm.families.links
    inverse_cls = getattr(links, "InversePower", None)
    if inverse_cls is None:
        inverse_cls = getattr(links, "inverse_power", None)
    if inverse_cls is None:
        pytest.skip("statsmodels does not expose an inverse-power Gamma link")

    family = sm.families.Gamma(link=inverse_cls())
    kwargs = {}
    sample_weight = None
    if weighted:
        kwargs["freq_weights"] = w
        sample_weight = w

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sm.GLM(y, X, family=family, **kwargs).fit(
            maxiter=500, tol=1e-10, disp=0
        )

    ours = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver="newton",
        device="cpu",
        max_iter=500,
        tol=1e-10,
    ).fit(X, y, sample_weight=sample_weight)

    np.testing.assert_allclose(
        np.asarray(ours.coef_, dtype=np.float64),
        np.asarray(ref.params, dtype=np.float64),
        rtol=3e-5,
        atol=3e-6,
    )
