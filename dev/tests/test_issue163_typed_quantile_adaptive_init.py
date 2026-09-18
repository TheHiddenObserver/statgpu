"""Adaptive-initializer propagation for typed Quantile Issue #163 closure."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)
from statgpu.linear_model.penalized import _fit_mixin
from statgpu.penalties import AdaptiveL1Penalty


def test_typed_quantile_adaptive_initializer_receives_resolved_quantile(monkeypatch):
    captured = {}

    def fake_init(
        X,
        y,
        loss_name,
        alpha=0.01,
        max_iter=100,
        tol=1e-4,
        loss_kwargs=None,
        sample_weight=None,
    ):
        captured["loss_name"] = loss_name
        captured["loss_kwargs"] = dict(loss_kwargs or {})
        captured["sample_weight"] = sample_weight
        return np.full(X.shape[1], 0.25, dtype=np.float64)

    monkeypatch.setattr(_fit_mixin, "_irls_ridge_init", fake_init)

    model = PenalizedQuantileRegression(
        quantile=0.2,
        penalty="adaptive_l1",
        alpha=0.04,
        solver="auto",
        device="cpu",
    )
    model._penalty_kwargs = model.penalty_kwargs or {}
    model._loss_kwargs = model.loss_kwargs or {}
    model._penalty = model._resolve_penalty()
    model._loss = model._resolve_loss()

    X = np.arange(30, dtype=np.float64).reshape(10, 3) / 10.0
    y = np.linspace(-0.5, 0.8, 10)
    initial = model._fit_initial(X, y, backend_name="numpy")

    np.testing.assert_array_equal(initial, np.full(3, 0.25))
    assert captured == {
        "loss_name": "quantile",
        "loss_kwargs": {"quantile": pytest.approx(0.2)},
        "sample_weight": None,
    }
    assert model.loss_kwargs is None
    assert model._loss_kwargs["quantile"] == pytest.approx(0.2)


def test_typed_quantile_adaptive_initializer_receives_analytic_weights(monkeypatch):
    captured = {}

    def fake_init(
        X,
        y,
        loss_name,
        alpha=0.01,
        max_iter=100,
        tol=1e-4,
        loss_kwargs=None,
        sample_weight=None,
    ):
        captured["loss_name"] = loss_name
        captured["loss_kwargs"] = dict(loss_kwargs or {})
        captured["sample_weight"] = np.asarray(sample_weight, dtype=np.float64).copy()
        return np.full(X.shape[1], 0.3, dtype=np.float64)

    monkeypatch.setattr(_fit_mixin, "_irls_ridge_init", fake_init)

    rng = np.random.default_rng(16356)
    X = rng.normal(size=(36, 3))
    y = 0.2 + X @ np.array([0.5, -0.25, 0.1])
    weights = np.linspace(0.4, 1.8, X.shape[0], dtype=np.float64)

    PenalizedQuantileRegression(
        quantile=0.2,
        penalty="adaptive_l1",
        alpha=0.04,
        solver="auto",
        device="cpu",
        max_iter=80,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    assert captured["loss_name"] == "quantile"
    assert captured["loss_kwargs"]["quantile"] == pytest.approx(0.2)
    np.testing.assert_array_equal(captured["sample_weight"], weights)


def test_quantile_adaptive_cv_initializer_uses_fold_local_and_full_weights(monkeypatch):
    captured = []

    def fake_init(
        X,
        y,
        loss_name,
        alpha=0.01,
        max_iter=100,
        tol=1e-4,
        loss_kwargs=None,
        sample_weight=None,
    ):
        captured.append(np.asarray(sample_weight, dtype=np.float64).copy())
        return np.full(X.shape[1], 0.25, dtype=np.float64)

    monkeypatch.setattr(_fit_mixin, "_irls_ridge_init", fake_init)

    rng = np.random.default_rng(16357)
    X = rng.normal(size=(40, 3))
    y = 0.1 + X @ np.array([0.45, -0.2, 0.15])
    weights = np.linspace(0.5, 1.7, X.shape[0], dtype=np.float64)
    idx = np.arange(X.shape[0])
    folds = [(idx[:20], idx[20:]), (idx[20:], idx[:20])]

    PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty="adaptive_l1",
        alpha_grid=np.asarray([0.04], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=80,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    assert len(captured) == 3
    np.testing.assert_array_equal(captured[0], weights[folds[0][0]])
    np.testing.assert_array_equal(captured[1], weights[folds[1][0]])
    np.testing.assert_array_equal(captured[2], weights)


def test_quantile_direct_adaptive_object_is_fit_local_and_reinitialized(monkeypatch):
    calls = {"count": 0}

    def fake_init(
        X,
        y,
        loss_name,
        alpha=0.01,
        max_iter=100,
        tol=1e-4,
        loss_kwargs=None,
        sample_weight=None,
    ):
        calls["count"] += 1
        if calls["count"] == 1:
            return np.asarray([1.0, 2.0], dtype=np.float64)
        return np.asarray([2.0, 1.0], dtype=np.float64)

    monkeypatch.setattr(_fit_mixin, "_irls_ridge_init", fake_init)

    rng = np.random.default_rng(16358)
    X = rng.normal(size=(36, 2))
    y1 = 0.2 + X @ np.asarray([0.5, -0.2])
    y2 = -0.1 + X @ np.asarray([-0.3, 0.6])
    penalty = AdaptiveL1Penalty(
        alpha=0.04,
        nu=1.0,
        eps=1e-6,
        normalize=False,
    )
    model = PenalizedQuantileRegression(
        quantile=0.3,
        penalty=penalty,
        alpha=0.04,
        solver="auto",
        device="cpu",
        max_iter=80,
        tol=1e-6,
    )

    model.fit(X, y1)
    first_weights = np.asarray(model._penalty._weights).copy()
    assert penalty._weights is None
    assert model._penalty is not penalty

    model.fit(X, y2)
    second_weights = np.asarray(model._penalty._weights).copy()

    assert calls["count"] == 2
    assert penalty._weights is None
    assert model.penalty is penalty
    assert model._penalty is not penalty
    assert not np.array_equal(first_weights, second_weights)


def test_quantile_cv_adaptive_object_preserves_fixed_state_and_skips_init(monkeypatch):
    fixed_weights = np.asarray([0.5, 1.25, 2.0], dtype=np.float64)
    penalty = AdaptiveL1Penalty(
        alpha=0.9,
        nu=2.0,
        eps=2e-5,
        init_method="ridge",
        normalize=False,
        weights=fixed_weights,
    )

    def forbidden_init(*args, **kwargs):
        raise AssertionError("fixed Adaptive L1 weights must not run initialization")

    monkeypatch.setattr(
        _fit_mixin._PenalizedFitMixin,
        "_fit_initial",
        forbidden_init,
    )

    rng = np.random.default_rng(16359)
    X = rng.normal(size=(40, 3))
    y = 0.1 + X @ np.asarray([0.4, -0.25, 0.15])
    idx = np.arange(X.shape[0])
    folds = [(idx[:20], idx[20:]), (idx[20:], idx[:20])]

    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.3},
        penalty=penalty,
        alpha_grid=np.asarray([0.04], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=80,
        tol=1e-6,
    ).fit(X, y)

    assert model.penalty is penalty
    assert penalty.alpha == pytest.approx(0.9)
    np.testing.assert_array_equal(penalty._weights, fixed_weights)

    resolved = model.estimator_._penalty
    assert resolved is not penalty
    assert resolved.alpha == pytest.approx(0.04)
    assert resolved.nu == pytest.approx(2.0)
    assert resolved.eps == pytest.approx(2e-5)
    assert resolved.init_method == "ridge"
    assert resolved.normalize is False
    np.testing.assert_array_equal(resolved._weights, fixed_weights)
    assert not hasattr(resolved, "_statgpu_quantile_scalar_cv_alpha_from_estimator")
