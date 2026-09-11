"""Regression tests for analytic sample_weight support in Newton."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core._logistic import LogisticLoss
from statgpu.linear_model import PenalizedGLM_CV, PenalizedLogisticRegression
from statgpu.losses import CoxPartialLikelihoodLoss
from statgpu.penalties import L2Penalty
from statgpu.solvers import newton_solver


def _logistic_data(seed=14271, n=96, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([0.55, -0.38, 0.24, -0.12])[:p]
    eta = -0.18 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    return X.astype(np.float64), y


def _solve_logistic(X, y, *, weights=None, alpha=0.04):
    return newton_solver(
        LogisticLoss(),
        L2Penalty(alpha),
        X,
        y,
        max_iter=100,
        tol=1e-11,
        sample_weight=weights,
    )[0]


def test_weighted_newton_integer_weights_equal_literal_row_replication():
    X, y = _logistic_data(n=48)
    weights = np.tile(np.array([1, 2, 3, 1], dtype=np.int64), 12)

    weighted = np.asarray(_solve_logistic(X, y, weights=weights))
    X_rep = np.repeat(X, weights, axis=0)
    y_rep = np.repeat(y, weights, axis=0)
    replicated = np.asarray(_solve_logistic(X_rep, y_rep))

    np.testing.assert_allclose(weighted, replicated, rtol=2e-9, atol=2e-10)


def test_weighted_newton_is_invariant_to_global_weight_rescaling():
    X, y = _logistic_data(seed=14272)
    weights = np.linspace(0.35, 1.85, X.shape[0], dtype=np.float64)

    base = np.asarray(_solve_logistic(X, y, weights=weights))
    scaled = np.asarray(_solve_logistic(X, y, weights=7.25 * weights))

    np.testing.assert_allclose(base, scaled, rtol=2e-10, atol=2e-11)


def test_uniform_weights_preserve_historical_unweighted_newton_result():
    X, y = _logistic_data(seed=14273)

    unweighted = np.asarray(_solve_logistic(X, y))
    uniform = np.asarray(
        _solve_logistic(X, y, weights=np.full(X.shape[0], 3.5, dtype=np.float64))
    )
    almost_uniform_weights = np.full(X.shape[0], 3.5, dtype=np.float64)
    almost_uniform_weights[-1] += 1e-8
    almost_uniform = np.asarray(
        _solve_logistic(X, y, weights=almost_uniform_weights)
    )

    np.testing.assert_allclose(unweighted, uniform, rtol=0.0, atol=1e-13)
    # The historical Newton gate used allclose() for floating-point uniformity.
    np.testing.assert_allclose(unweighted, almost_uniform, rtol=0.0, atol=1e-13)


def test_weighted_newton_torch_cpu_matches_numpy_when_available():
    torch = pytest.importorskip("torch")
    X, y = _logistic_data(seed=14278)
    weights = np.linspace(0.4, 1.9, X.shape[0], dtype=np.float64)

    expected = np.asarray(_solve_logistic(X, y, weights=weights))
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    w_t = torch.as_tensor(weights, dtype=torch.float64)
    actual = _solve_logistic(X_t, y_t, weights=w_t)

    np.testing.assert_allclose(
        expected,
        actual.detach().cpu().numpy(),
        rtol=2e-9,
        atol=2e-10,
    )


@pytest.mark.parametrize(
    "weights, match",
    [
        (np.ones(8), "length n_samples"),
        (np.r_[np.ones(95), -1.0], "non-negative"),
        (np.zeros(96), "positive sum"),
        (np.r_[np.ones(95), np.nan], "finite"),
    ],
)
def test_weighted_newton_rejects_invalid_weights_before_optimization(weights, match):
    X, y = _logistic_data(seed=14274)
    with pytest.raises(ValueError, match=match):
        _solve_logistic(X, y, weights=weights)


def test_nonuniform_cox_weights_remain_explicitly_unsupported():
    rng = np.random.default_rng(14275)
    n = 40
    X = rng.normal(size=(n, 3))
    time = rng.exponential(scale=2.0, size=n) + 0.1
    event = np.ones(n, dtype=np.float64)
    y = {"time": time, "event": event}
    weights = np.linspace(0.5, 1.5, n)

    with pytest.raises((ValueError, NotImplementedError), match="sample_weight|weight"):
        newton_solver(
            CoxPartialLikelihoodLoss(ties="breslow"),
            L2Penalty(0.0),
            X,
            y,
            max_iter=20,
            tol=1e-8,
            sample_weight=weights,
        )


def test_weighted_logistic_explicit_newton_works_without_inference():
    X, y = _logistic_data(seed=14279, n=120)
    weights = np.linspace(0.45, 1.75, X.shape[0], dtype=np.float64)

    model = PenalizedLogisticRegression(
        penalty="l2",
        alpha=0.035,
        solver="newton",
        device="cpu",
        compute_inference=False,
        max_iter=400,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    assert model.solver == "newton"
    assert model._selected_solver == "newton"
    assert np.all(np.isfinite(np.asarray(model.coef_)))
    assert np.isfinite(float(model.intercept_))


def test_weighted_logistic_auto_uses_newton_and_keeps_public_request():
    X, y = _logistic_data(seed=14276, n=120)
    weights = np.linspace(0.45, 1.75, X.shape[0], dtype=np.float64)

    model = PenalizedLogisticRegression(
        penalty="l2",
        alpha=0.035,
        solver="auto",
        device="cpu",
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
        max_iter=400,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    assert model.solver == "auto"
    assert model._selected_solver == "newton"
    assert model.inference_requested_method_ == "auto"
    assert model.inference_resolved_method_ == "m_estimation"
    assert model.inference_method_ == "m_estimation"
    assert np.all(np.isfinite(np.asarray(model.coef_)))
    assert np.all(np.isfinite(np.asarray(model._bse)))
    assert np.all(np.isfinite(np.asarray(model._pvalues)))


def test_weighted_logistic_cv_auto_keeps_newton_for_selected_final_refit():
    X, y = _logistic_data(seed=14277, n=90)
    weights = np.linspace(0.5, 1.6, X.shape[0], dtype=np.float64)

    cv = PenalizedGLM_CV(
        loss="logistic",
        penalty="l2",
        alpha_grid=np.array([0.09, 0.04]),
        cv=2,
        random_state=11,
        device="cpu",
        solver="auto",
        max_iter=500,
        tol=1e-8,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y, sample_weight=weights)

    assert cv.get_params(deep=False)["solver"] == "auto"
    assert getattr(cv, "_solver", None) == "auto"
    assert cv.estimator_._selected_solver == "newton"
    assert cv.inference_method_ == "m_estimation"
    assert cv.penalty_conditioning_ == "cv_selected_penalty"
    assert cv.penalty_selection_adjusted_ is False
