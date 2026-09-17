"""Weighted Quantile step-scale and LLA surrogate contracts for PR #166."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.solvers import _fista_lla_group_contract as lla_contract


def test_quantile_weighted_step_scale_matches_normalized_weighted_gram():
    X = np.asarray(
        [
            [4.0, 0.0],
            [0.0, 1.0],
            [2.0, 3.0],
            [1.0, -2.0],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([0.2, 3.0, 0.7, 5.0], dtype=np.float64)
    tau = 0.35
    loss = QuantileLoss(tau)

    actual = loss.lipschitz(
        X,
        np.zeros(X.shape[1], dtype=np.float64),
        sample_weight=weights,
    )
    gram = X.T @ (X * weights[:, None]) / float(np.sum(weights))
    expected = max(tau, 1.0 - tau) * float(np.linalg.eigvalsh(gram)[-1])

    # Production uses the shared 20-step power iteration with tol=1e-8 rather
    # than a full eigendecomposition.
    assert actual == pytest.approx(expected, rel=1e-8, abs=1e-10)


def test_quantile_weighted_step_scale_is_weight_rescaling_invariant():
    rng = np.random.default_rng(166501)
    X = rng.normal(size=(18, 4))
    weights = np.linspace(0.3, 2.1, X.shape[0], dtype=np.float64)
    coef = np.zeros(X.shape[1], dtype=np.float64)
    loss = QuantileLoss(0.6)

    base = loss.lipschitz(X, coef, sample_weight=weights)
    scaled = loss.lipschitz(X, coef, sample_weight=weights * 17.0)

    assert scaled == pytest.approx(base, rel=2e-12, abs=2e-14)


def test_quantile_equal_weights_recover_unweighted_step_scale():
    rng = np.random.default_rng(166502)
    X = rng.normal(size=(15, 3))
    coef = np.zeros(X.shape[1], dtype=np.float64)
    weights = np.full(X.shape[0], 4.25, dtype=np.float64)
    loss = QuantileLoss(0.4)

    unweighted = loss.lipschitz(X, coef)
    weighted = loss.lipschitz(X, coef, sample_weight=weights)

    assert weighted == pytest.approx(unweighted, rel=2e-12, abs=2e-14)


def test_scalar_lla_surrogate_preserves_raw_derivative_scale_and_free_intercept():
    """Scalar SCAD/MCP LLA derivatives are not Adaptive-Lasso normalized."""
    derivatives = np.asarray([0.3, 0.1, 0.0, 0.2, 0.0], dtype=np.float64)
    penalty = lla_contract._scalar_surrogate_factory(derivatives)

    assert penalty.name == "adaptive_l1"
    assert penalty.alpha == pytest.approx(1.0)
    assert penalty.normalize is False
    np.testing.assert_array_equal(penalty._weights, derivatives)
    # The final derivative is the augmented intercept coordinate and remains
    # exactly unpenalized in the convex WLS surrogate.
    assert penalty._weights[-1] == 0.0


def test_quantile_irls_observation_weights_include_normalized_analytic_weights():
    """The WLS majorizer must use the same normalized analytic weights as Quantile IRLS."""
    loss = QuantileLoss(0.35)
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.5, -0.25, 0.8, -0.4], dtype=np.float64)
    params = np.zeros(4, dtype=np.float64)
    raw = np.asarray([0.4, 0.8, 1.2, 1.6], dtype=np.float64)
    normalized = raw * (4.0 / float(np.sum(raw)))

    actual = lla_contract._quantile_irls_weights(
        loss,
        X,
        y,
        params,
        normalized,
        np,
        "numpy",
    )
    residual = y.copy()
    asym = np.where(residual < 0.0, 1.0 - 0.35, 0.35)
    expected = normalized * asym / np.maximum(np.abs(residual), 1e-8)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
