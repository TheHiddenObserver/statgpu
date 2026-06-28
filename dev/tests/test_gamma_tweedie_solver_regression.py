"""Regression tests for Gamma/Tweedie Newton and IRLS correctness."""

import warnings

import numpy as np
import pytest

from statgpu.glm_core._family import Gamma, LogLink
from statgpu.glm_core._gamma import GammaLoss
from statgpu.glm_core._irls import irls_solver
from statgpu.glm_core._tweedie import TweedieLoss
from statgpu.penalties._l2 import L2Penalty
from statgpu.linear_model.penalized._base import SelectivePenalty
from statgpu.solvers._convergence import ConvergenceWarning
from statgpu.solvers._lbfgs import lbfgs_solver
from statgpu.solvers._newton import newton_solver
from statgpu.solvers._utils import _smooth_penalty_value_dev


def _data(seed=123, n=600, p=6):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.4, size=(n, p))
    coef = np.linspace(0.35, -0.2, p)
    mu = np.exp(X @ coef)
    y_gamma = rng.gamma(shape=2.0, scale=mu / 2.0)
    y_tweedie = np.maximum(mu + rng.normal(scale=0.3, size=n), 0.0)
    return X, y_gamma, y_tweedie


@pytest.mark.parametrize(
    "loss,y_index",
    [(GammaLoss(), 1), (TweedieLoss(power=1.5), 2)],
)
def test_observed_hessian_matches_finite_difference(loss, y_index):
    data = _data(n=300, p=5)
    X, y = data[0], data[y_index]
    coef = np.linspace(-0.1, 0.15, X.shape[1])
    eps = 1e-6
    numerical = np.column_stack([
        (
            loss.gradient(X, y, coef + eps * np.eye(X.shape[1])[j])
            - loss.gradient(X, y, coef - eps * np.eye(X.shape[1])[j])
        ) / (2.0 * eps)
        for j in range(X.shape[1])
    ])
    analytic = loss.hessian(X, y, coef)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-7)
    assert np.linalg.eigvalsh(0.5 * (analytic + analytic.T)).min() > 0.0


@pytest.mark.parametrize("loss,y_index", [(GammaLoss(), 1), (TweedieLoss(1.5), 2)])
def test_fused_value_and_gradient_match_base_formulas(loss, y_index):
    data = _data(n=200, p=4)
    X, y = data[0], data[y_index]
    coef = np.linspace(-0.15, 0.1, X.shape[1])
    fused_value, fused_gradient = loss.fused_value_and_gradient(X, y, coef)
    np.testing.assert_allclose(fused_value, loss.value(X, y, coef), rtol=1e-13)
    np.testing.assert_allclose(
        fused_gradient, loss.gradient(X, y, coef), rtol=1e-13, atol=1e-13
    )


def test_gamma_log_irls_uses_unit_fisher_weights():
    mu = np.array([0.2, 1.0, 7.5])
    weights = Gamma(LogLink()).irls_weights(mu, np.array([0.3, 1.2, 6.0]))
    np.testing.assert_allclose(weights, np.ones_like(mu))


def test_selective_l2_penalty_is_included_in_device_objective():
    penalty = SelectivePenalty()
    penalty.configure(L2Penalty(alpha=0.2), p=2, backend="numpy")
    coef = np.array([1.0, 2.0, 100.0])
    assert _smooth_penalty_value_dev(penalty, coef) == pytest.approx(0.5)


@pytest.mark.parametrize("loss,y_index", [(GammaLoss(), 1), (TweedieLoss(1.5), 2)])
def test_newton_converges_to_lbfgs_solution(loss, y_index):
    data = _data()
    X, y = data[0], data[y_index]
    penalty = L2Penalty(alpha=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef_newton, n_iter = newton_solver(
            loss, penalty, X, y, max_iter=50, tol=1e-7
        )
        coef_lbfgs, _ = lbfgs_solver(
            loss, penalty, X, y, max_iter=200, tol=1e-10
        )
    assert n_iter < 10
    np.testing.assert_allclose(coef_newton, coef_lbfgs, rtol=2e-5, atol=2e-7)
    gradient = loss.gradient(X, y, coef_newton) + penalty.gradient(coef_newton)
    assert np.linalg.norm(gradient) <= 1e-7


def test_gamma_irls_matches_penalized_newton():
    X, y, _ = _data()
    alpha = 0.05
    penalty = L2Penalty(alpha=alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef_irls, irls_iter = irls_solver(
            Gamma(LogLink()),
            X,
            y,
            max_iter=50,
            tol=1e-7,
            ridge_alpha=X.shape[0] * alpha,
            ridge_penalize_intercept=True,
        )
        coef_newton, _ = newton_solver(
            GammaLoss(), penalty, X, y, max_iter=50, tol=1e-8
        )
    assert irls_iter < 20
    np.testing.assert_allclose(coef_irls, coef_newton, rtol=2e-5, atol=2e-7)


def test_positive_response_validation():
    X = np.ones((3, 1))
    with pytest.raises(ValueError, match="strictly positive"):
        GammaLoss().preprocess(X, np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="non-negative"):
        TweedieLoss().preprocess(X, np.array([1.0, -0.1, 2.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        irls_solver(Gamma(LogLink()), X, np.array([1.0, 0.0, 2.0]))
