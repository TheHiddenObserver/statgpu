"""Typed Quantile constructor-capture regressions for Issue #163."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedQuantileRegression


def test_typed_quantile_resolves_private_loss_kwargs_without_mutating_public_state():
    model = PenalizedQuantileRegression(quantile=0.2, penalty="l2", alpha=0.04)

    # BaseEstimator intentionally preserves the exact outer public constructor
    # argument for sklearn clone compatibility.
    assert model.loss_kwargs is None
    assert model.quantile == pytest.approx(0.2)

    loss = model._resolve_loss()
    assert loss._tau == pytest.approx(0.2)
    assert model._loss_kwargs == {"quantile": pytest.approx(0.2)}
    assert model.loss_kwargs is None


def test_typed_quantile_explicit_loss_kwargs_keeps_historical_precedence():
    model = PenalizedQuantileRegression(
        quantile=0.2,
        penalty="l2",
        alpha=0.04,
        loss_kwargs={"quantile": 0.35},
    )
    loss = model._resolve_loss()
    assert loss._tau == pytest.approx(0.35)
    assert model._loss_kwargs["quantile"] == pytest.approx(0.35)
    assert model.quantile == pytest.approx(0.2)
    assert model.loss_kwargs == {"quantile": 0.35}


def test_typed_quantile_sklearn_clone_preserves_public_quantile_when_available():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = PenalizedQuantileRegression(
        quantile=0.2,
        penalty="l2",
        alpha=0.04,
        solver="irls",
        device="cpu",
    )
    cloned = clone(model)
    assert cloned.quantile == pytest.approx(0.2)
    assert cloned.loss_kwargs is None
    assert cloned.get_params(deep=False)["quantile"] == pytest.approx(0.2)

    rng = np.random.default_rng(16331)
    X = rng.normal(size=(48, 2))
    y = 0.1 + X @ np.array([0.5, -0.25]) + rng.laplace(scale=0.1, size=48)
    fitted = cloned.fit(X, y)
    assert fitted._loss._tau == pytest.approx(0.2)
