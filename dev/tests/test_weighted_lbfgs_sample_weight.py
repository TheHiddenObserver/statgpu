"""Regression tests for genuine non-uniform GLM sample_weight support in L-BFGS."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import GLMLoss
from statgpu.glm_core._logistic import LogisticLoss
from statgpu.glm_core._poisson import PoissonLoss
from statgpu.glm_core._squared import SquaredErrorLoss
from statgpu.losses import HuberLoss, QuantileLoss
from statgpu.solvers import lbfgs_solver


def _continuous_data(seed=15001, n=72, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.7, size=(n, p)).astype(np.float64)
    beta = np.array([0.65, -0.35, 0.2])[:p]
    y = (X @ beta + rng.normal(scale=0.12, size=n)).astype(np.float64)
    return X, y


def _logistic_data(seed=15002, n=96, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    beta = np.array([0.55, -0.38, 0.24, -0.12])[:p]
    eta = -0.15 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    return X, y


def _poisson_data(seed=15003, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.45, size=(n, p)).astype(np.float64)
    beta = np.array([0.25, -0.18, 0.12])[:p]
    mu = np.exp(X @ beta)
    y = rng.poisson(mu).astype(np.float64)
    return X, y


def _solve(loss, X, y, *, weights=None, tol=1e-11):
    return lbfgs_solver(
        loss,
        None,
        X,
        y,
        max_iter=600,
        tol=tol,
        sample_weight=weights,
    )[0]


@pytest.mark.parametrize(
    "loss,data_factory",
    [
        (SquaredErrorLoss(), _continuous_data),
        (LogisticLoss(), _logistic_data),
        (PoissonLoss(), _poisson_data),
    ],
)
def test_weighted_lbfgs_integer_weights_equal_literal_row_replication(loss, data_factory):
    X, y = data_factory()
    weights = np.resize(np.array([1, 2, 3, 1], dtype=np.int64), X.shape[0])

    weighted = np.asarray(_solve(loss, X, y, weights=weights))
    X_rep = np.repeat(X, weights, axis=0)
    y_rep = np.repeat(y, weights, axis=0)
    replicated = np.asarray(_solve(loss, X_rep, y_rep))

    np.testing.assert_allclose(weighted, replicated, rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize(
    "loss,data_factory",
    [
        (SquaredErrorLoss(), _continuous_data),
        (LogisticLoss(), _logistic_data),
        (PoissonLoss(), _poisson_data),
    ],
)
def test_weighted_lbfgs_is_invariant_to_positive_global_weight_rescaling(loss, data_factory):
    X, y = data_factory()
    weights = np.linspace(0.35, 1.85, X.shape[0], dtype=np.float64)

    base = np.asarray(_solve(loss, X, y, weights=weights))
    scaled = np.asarray(_solve(loss, X, y, weights=7.25 * weights))

    np.testing.assert_allclose(base, scaled, rtol=2e-8, atol=2e-9)


def test_weighted_lbfgs_zero_weight_rows_equal_dropping_those_rows():
    X, y = _logistic_data(seed=15004)
    weights = np.linspace(0.4, 1.6, X.shape[0], dtype=np.float64)
    weights[::7] = 0.0
    keep = weights > 0.0

    weighted = np.asarray(_solve(LogisticLoss(), X, y, weights=weights))
    dropped = np.asarray(
        _solve(LogisticLoss(), X[keep], y[keep], weights=weights[keep])
    )

    np.testing.assert_allclose(weighted, dropped, rtol=2e-8, atol=2e-9)


def test_uniform_and_historically_almost_uniform_weights_use_unweighted_path():
    X, y = _logistic_data(seed=15005)
    unweighted = np.asarray(_solve(LogisticLoss(), X, y))

    uniform = np.full(X.shape[0], 3.5, dtype=np.float64)
    almost_uniform = uniform.copy()
    almost_uniform[-1] += 1e-8

    np.testing.assert_allclose(
        unweighted,
        np.asarray(_solve(LogisticLoss(), X, y, weights=uniform)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        unweighted,
        np.asarray(_solve(LogisticLoss(), X, y, weights=almost_uniform)),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "weights,match",
    [
        (np.ones(8), "length n_samples"),
        (np.r_[np.ones(95), -1.0], "non-negative"),
        (np.zeros(96), "positive sum"),
        (np.r_[np.ones(95), np.nan], "finite"),
    ],
)
def test_weighted_lbfgs_rejects_invalid_weights_before_iteration(weights, match):
    X, y = _logistic_data(seed=15006)
    with pytest.raises(ValueError, match=match):
        _solve(LogisticLoss(), X, y, weights=weights)


@pytest.mark.parametrize("loss", [HuberLoss(), QuantileLoss(quantile=0.5)])
def test_non_glm_direct_lbfgs_nonuniform_weights_remain_fail_closed(loss):
    X, y = _continuous_data(seed=15007)
    weights = np.linspace(0.5, 1.5, X.shape[0], dtype=np.float64)

    with pytest.raises(ValueError, match="does not support non-uniform sample_weight"):
        _solve(loss, X, y, weights=weights)


def test_custom_glm_loss_inherits_weighted_lbfgs_capability():
    class CustomSquaredLoss(GLMLoss):
        name = "custom_squared"

        def per_sample_value(self, eta, y):
            resid = eta - y
            return 0.5 * resid * resid

        def per_sample_gradient(self, eta, y):
            return eta - y

    X, y = _continuous_data(seed=15008)
    weights = np.linspace(0.4, 1.8, X.shape[0], dtype=np.float64)

    actual = np.asarray(_solve(CustomSquaredLoss(), X, y, weights=weights))
    expected = np.asarray(_solve(SquaredErrorLoss(), X, y, weights=weights))
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)


def test_weighted_lbfgs_passes_weights_to_every_glm_objective_evaluation():
    class CountingSquaredLoss(SquaredErrorLoss):
        def __init__(self):
            self.weight_presence = []

        def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
            self.weight_presence.append(sample_weight is not None)
            return super().fused_value_and_gradient(
                X, y, coef, sample_weight=sample_weight
            )

    X, y = _continuous_data(seed=15009)
    weights = np.linspace(0.45, 1.7, X.shape[0], dtype=np.float64)
    loss = CountingSquaredLoss()

    _solve(loss, X, y, weights=weights)

    assert len(loss.weight_presence) >= 3
    assert all(loss.weight_presence)


def test_weighted_lbfgs_torch_cpu_matches_numpy_when_available():
    torch = pytest.importorskip("torch")
    X, y = _logistic_data(seed=15010)
    weights = np.linspace(0.4, 1.9, X.shape[0], dtype=np.float64)

    expected = np.asarray(_solve(LogisticLoss(), X, y, weights=weights))
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    w_t = torch.as_tensor(weights, dtype=torch.float64)
    actual = _solve(LogisticLoss(), X_t, y_t, weights=w_t)

    np.testing.assert_allclose(
        expected,
        actual.detach().cpu().numpy(),
        rtol=2e-7,
        atol=2e-8,
    )
