"""Adaptive-initializer propagation for typed Quantile Issue #163 closure."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedQuantileRegression
from statgpu.linear_model.penalized import _fit_mixin


def test_typed_quantile_adaptive_initializer_receives_resolved_quantile(monkeypatch):
    captured = {}

    def fake_init(X, y, loss_name, alpha=0.01, max_iter=100, tol=1e-4, loss_kwargs=None):
        captured["loss_name"] = loss_name
        captured["loss_kwargs"] = dict(loss_kwargs or {})
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
    }
    assert model.loss_kwargs is None
    assert model._loss_kwargs["quantile"] == pytest.approx(0.2)
