"""Weighted first-order step-scale contracts for PR #166 Quantile FISTA-LLA."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import GroupSCADPenalty, SCADPenalty
from statgpu.solvers import _fista_lla_group_contract as group_contract
from statgpu.solvers._fista_lla_group_contract import (
    _QuantileWeightedStepScaleProxy,
)


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
    # than a full eigendecomposition, so compare the implemented step scale to
    # the analytic eigenvalue at that declared numerical accuracy.
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


def test_quantile_step_proxy_retains_weight_for_periodic_refresh():
    class RecordingLoss:
        name = "quantile"

        def __init__(self):
            self.weights = []

        def lipschitz(self, X, coef, y=None, sample_weight=None):
            self.weights.append(sample_weight)
            return 1.0

    base = RecordingLoss()
    proxy = _QuantileWeightedStepScaleProxy(base)
    X = np.eye(3, dtype=np.float64)
    coef = np.zeros(3, dtype=np.float64)
    weights = np.asarray([0.5, 1.0, 2.0], dtype=np.float64)

    assert proxy.lipschitz(X, coef, sample_weight=weights) == 1.0
    assert proxy.lipschitz(X, coef) == 1.0

    assert base.weights[0] is weights
    assert base.weights[1] is weights


def _recording_quantile_loss():
    class RecordingQuantileLoss:
        name = "quantile"
        has_hessian = False

        def __init__(self):
            self.weights = []

        def lipschitz(self, X, coef, y=None, sample_weight=None):
            self.weights.append(sample_weight)
            return 1.0

    return RecordingQuantileLoss()


def test_public_group_lla_installs_quantile_weight_proxy(monkeypatch):
    """Direct public group calls get the same weighted refresh contract."""
    loss = _recording_quantile_loss()
    penalty = GroupSCADPenalty(alpha=0.1, a=3.7, groups=[[0, 1], [2, 3]])
    weights = np.asarray([0.4, 0.8, 1.2, 1.6], dtype=np.float64)
    X = np.eye(4, dtype=np.float64)
    y = np.zeros(4, dtype=np.float64)
    captured = {}

    def fake_base(loss_proxy, *args, **kwargs):
        captured["loss_proxy"] = loss_proxy
        assert loss_proxy.lipschitz(
            X, np.zeros(4), sample_weight=weights
        ) == 1.0
        assert loss_proxy.lipschitz(X, np.zeros(4)) == 1.0
        return np.zeros(4), 0.0, 2

    monkeypatch.setattr(group_contract, "_base_fista_lla_path", fake_base)
    result = group_contract.fista_lla_path(
        loss,
        penalty,
        X,
        y,
        alpha_path=[0.1],
        fit_intercept=False,
        sample_weight=weights,
    )

    assert result[2] == 2
    assert isinstance(
        captured["loss_proxy"]._loss,
        _QuantileWeightedStepScaleProxy,
    )
    assert loss.weights[0] is weights
    assert loss.weights[1] is weights


def test_public_scalar_lla_installs_quantile_weight_proxy(monkeypatch):
    """Scalar direct public Quantile LLA calls retain weighted step state too."""
    loss = _recording_quantile_loss()
    penalty = SCADPenalty(alpha=0.1, a=3.7)
    weights = np.asarray([0.4, 0.8, 1.2, 1.6], dtype=np.float64)
    X = np.eye(4, dtype=np.float64)
    y = np.zeros(4, dtype=np.float64)
    captured = {}

    def fake_base(loss_proxy, *args, **kwargs):
        captured["loss_proxy"] = loss_proxy
        assert loss_proxy.lipschitz(
            X, np.zeros(4), sample_weight=weights
        ) == 1.0
        assert loss_proxy.lipschitz(X, np.zeros(4)) == 1.0
        return np.zeros(4), 0.0, 2

    monkeypatch.setattr(group_contract, "_base_fista_lla_path", fake_base)
    result = group_contract.fista_lla_path(
        loss,
        penalty,
        X,
        y,
        alpha_path=[0.1],
        fit_intercept=False,
        sample_weight=weights,
    )

    assert result[2] == 2
    assert isinstance(captured["loss_proxy"], _QuantileWeightedStepScaleProxy)
    assert loss.weights[0] is weights
    assert loss.weights[1] is weights
