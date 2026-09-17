"""Weighted first-order step-scale contracts for PR #166 Quantile group LLA."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.solvers._fista_lla_group_contract import _GroupFISTALossProxy


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

    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-14)


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


def test_group_fista_loss_proxy_retains_weight_for_periodic_refresh():
    class RecordingLoss:
        def __init__(self):
            self.weights = []

        def lipschitz(self, X, coef, y=None, sample_weight=None):
            self.weights.append(sample_weight)
            return 1.0

    base = RecordingLoss()
    proxy = _GroupFISTALossProxy(base)
    X = np.eye(3, dtype=np.float64)
    coef = np.zeros(3, dtype=np.float64)
    weights = np.asarray([0.5, 1.0, 2.0], dtype=np.float64)

    assert proxy.lipschitz(X, coef, sample_weight=weights) == 1.0
    assert proxy.lipschitz(X, coef) == 1.0

    assert base.weights[0] is weights
    assert base.weights[1] is weights
