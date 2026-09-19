"""Weighted Quantile step-scale and Group IRLS contracts for PR #166."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import GroupSCADPenalty, SCADPenalty
from statgpu.solvers import _fista_lla as fista_lla_base
from statgpu.solvers import _fista_lla_group_contract as fista_lla_contract
from statgpu.solvers import _quantile_group_proximal_irls_lla as group_solver
from statgpu.solvers._fista_lla_group_contract import (
    _QuantileWeightedStepScaleProxy,
)


def test_quantile_weighted_step_scale_uses_safe_normalized_weighted_gram_bound():
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
    exact = float(np.linalg.eigvalsh(gram)[-1])
    safe_bound = min(
        float(np.max(np.sum(np.abs(gram), axis=1))),
        float(np.sqrt(np.sum(gram * gram))),
    )
    scale = max(tau, 1.0 - tau)

    assert actual == pytest.approx(scale * safe_bound, rel=1e-12, abs=1e-14)
    assert actual >= scale * exact


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


def test_quantile_fista_lla_normalizes_response_weight_and_warm_start(monkeypatch):
    loss = QuantileLoss(0.35)
    penalty = SCADPenalty(alpha=0.05)
    X = np.eye(3, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.7], dtype=np.float64)
    weights = np.asarray([0.5, 1.0, 1.5], dtype=np.float64)
    init = np.asarray([0.1, -0.05, 0.02], dtype=np.float64)
    original = fista_lla_base._xp_asarray
    calls = []

    def recording_asarray(value, dtype, ref):
        calls.append((value, ref))
        return original(value, dtype, ref)

    monkeypatch.setattr(fista_lla_base, "_xp_asarray", recording_asarray)

    fista_lla_base.fista_lla_path(
        loss,
        penalty,
        X,
        y,
        alpha_path=[0.05],
        max_lla_per_step=1,
        max_iter=1,
        fit_intercept=False,
        sample_weight=weights,
        init_coef=init,
    )

    assert any(value is y and ref is X for value, ref in calls)
    assert any(value is weights and ref is X for value, ref in calls)
    assert any(value is init for value, _ in calls)


def test_low_level_quantile_fista_lla_proxy_retains_periodic_weight():
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


def test_public_group_fista_lla_stays_fista_and_installs_quantile_weight_proxy(monkeypatch):
    """Low-level fista_lla_path keeps FISTA semantics after the auto-route repair."""
    class RecordingQuantileLoss:
        name = "quantile"
        has_hessian = False

        def __init__(self):
            self.weights = []

        def lipschitz(self, X, coef, y=None, sample_weight=None):
            self.weights.append(sample_weight)
            return 1.0

    loss = RecordingQuantileLoss()
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

    monkeypatch.setattr(fista_lla_contract, "_base_fista_lla_path", fake_base)
    result = fista_lla_contract.fista_lla_path(
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


def test_group_proximal_irls_observation_weights_include_normalized_analytic_weights():
    loss = QuantileLoss(0.35)
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.5, -0.25, 0.8, -0.4], dtype=np.float64)
    params = np.zeros(4, dtype=np.float64)
    raw = np.asarray([0.4, 0.8, 1.2, 1.6], dtype=np.float64)
    normalized = raw * (4.0 / float(np.sum(raw)))

    actual = group_solver._quantile_irls_weights(
        loss,
        X,
        y,
        params,
        normalized,
        np,
        "numpy",
    )
    asym = np.where(y < 0.0, 1.0 - 0.35, 0.35)
    expected = normalized * asym / np.maximum(np.abs(y), 1e-8)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
