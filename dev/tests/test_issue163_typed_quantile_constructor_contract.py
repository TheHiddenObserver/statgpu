"""Typed Quantile constructor-capture regressions for Issue #163."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)


@pytest.mark.parametrize("bad_quantile", ["0.2", True, np.nan])
def test_typed_quantile_rejects_invalid_quantile_with_value_error(bad_quantile):
    with pytest.raises(ValueError, match="finite real number in"):
        PenalizedQuantileRegression(quantile=bad_quantile)


def test_generic_quantile_loss_kwargs_reject_invalid_quantile_on_fit():
    X = np.arange(24, dtype=np.float64).reshape(12, 2)
    y = np.linspace(-0.4, 0.7, 12)
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": "0.2"},
        penalty="l2",
        alpha=0.02,
        device="cpu",
    )
    with pytest.raises(ValueError, match="finite real number in"):
        model.fit(X, y)


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


def test_typed_quantile_get_params_and_clone_preserve_lla_controls_when_available():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = PenalizedQuantileRegression(
        quantile=0.2,
        penalty="scad",
        alpha=0.04,
        solver="auto",
        device="cpu",
        lla=True,
        max_lla_iters=3,
        lla_tol=2e-5,
    )
    params = model.get_params(deep=False)
    assert params["lla"] is True
    assert params["max_lla_iters"] == 3
    assert params["lla_tol"] == pytest.approx(2e-5)

    cloned = clone(model)
    assert cloned.lla is True
    assert cloned.max_lla_iters == 3
    assert cloned.lla_tol == pytest.approx(2e-5)
    assert cloned._lla_enabled is True
    assert cloned._max_lla_iters == 3
    assert cloned._lla_tol == pytest.approx(2e-5)


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


def test_typed_quantile_score_rejects_invalid_weights_before_prediction(monkeypatch):
    X = np.array(
        [[-1.0, 0.2], [0.0, -0.1], [0.5, 0.4], [1.0, -0.3]],
        dtype=np.float64,
    )
    y = np.array([-0.4, 0.1, 0.35, 0.8], dtype=np.float64)
    model = PenalizedQuantileRegression(
        quantile=0.3,
        penalty="l2",
        alpha=0.02,
        solver="irls",
        device="cpu",
        max_iter=200,
        tol=1e-8,
    ).fit(X, y)

    def forbidden_predict(*args, **kwargs):
        raise AssertionError("invalid score weights must fail before prediction")

    monkeypatch.setattr(model, "predict", forbidden_predict)
    with pytest.raises(ValueError, match="sample_weight"):
        model.score(
            X,
            y,
            sample_weight=np.array([1.0, -0.1, 1.0, 1.0]),
        )


@pytest.mark.parametrize(
    "sample_weight",
    [
        np.array([1.0, -0.1, 1.0, 1.0]),
        np.array([1.0, np.nan, 1.0, 1.0]),
        np.zeros(4, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    ],
)
def test_typed_quantile_score_rejects_invalid_sample_weight(sample_weight):
    X = np.array(
        [[-1.0, 0.2], [0.0, -0.1], [0.5, 0.4], [1.0, -0.3]],
        dtype=np.float64,
    )
    y = np.array([-0.4, 0.1, 0.35, 0.8], dtype=np.float64)
    model = PenalizedQuantileRegression(
        quantile=0.3,
        penalty="l2",
        alpha=0.02,
        solver="irls",
        device="cpu",
        max_iter=200,
        tol=1e-8,
    ).fit(X, y)

    with pytest.raises(ValueError, match="sample_weight"):
        model.score(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize(
    "loss_kwargs, expected_q",
    [
        (None, 0.2),
        ({"quantile": 0.35}, 0.35),
    ],
)
def test_typed_quantile_score_uses_effective_quantile(loss_kwargs, expected_q):
    rng = np.random.default_rng(16332)
    X = rng.normal(size=(64, 2))
    y = 0.15 + X @ np.array([0.55, -0.2]) + rng.laplace(scale=0.12, size=64)
    weights = np.linspace(0.5, 1.5, X.shape[0])

    model = PenalizedQuantileRegression(
        quantile=0.2,
        penalty="l2",
        alpha=0.025,
        solver="irls",
        device="cpu",
        max_iter=400,
        tol=1e-9,
        loss_kwargs=loss_kwargs,
    ).fit(X, y)

    pred = model.predict(X)
    resid = y - pred
    per_sample = np.where(
        resid >= 0.0,
        expected_q * resid,
        (expected_q - 1.0) * resid,
    )
    expected = -float(np.average(per_sample, weights=weights))
    assert model.score(X, y, sample_weight=weights) == pytest.approx(
        expected, rel=0.0, abs=1e-12
    )
