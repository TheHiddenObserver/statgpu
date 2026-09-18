"""Adaptive-initializer propagation for typed Quantile Issue #163 closure."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedQuantileRegression
from statgpu.linear_model.penalized import _fit_mixin


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
