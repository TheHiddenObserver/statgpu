"""Low-level validation contracts for Quantile fista_lla_path in PR #166."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import L1Penalty, L2Penalty, MCPPenalty, SCADPenalty
from statgpu.solvers import fista_lla_path


@pytest.mark.parametrize(
    ("X_transform", "y_transform", "message"),
    [
        (lambda X: X[:, 0], lambda y: y, "X must be two-dimensional"),
        (lambda X: X, lambda y: y[:, None], "y must be one-dimensional"),
        (
            lambda X: X,
            lambda y: y[:1],
            "same number of observations as X",
        ),
    ],
)
def test_quantile_fista_lla_rejects_invalid_xy_shapes_before_loss_work(
    monkeypatch, X_transform, y_transform, message
):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = SCADPenalty(alpha=0.04, a=3.7)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    monkeypatch.setattr(loss, "lipschitz", forbidden)

    with pytest.raises(ValueError, match=message):
        fista_lla_path(
            loss,
            penalty,
            X_transform(X),
            y_transform(y),
            alpha_path=np.asarray([0.04], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=20,
            tol=1e-6,
            fit_intercept=False,
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("X", "X must contain finite values"),
        ("y", "y must contain finite values"),
    ],
)
def test_quantile_fista_lla_rejects_nonfinite_xy_before_loss_work(
    monkeypatch, target, message
):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    X_bad = X.copy()
    y_bad = y.copy()
    if target == "X":
        X_bad[0, 0] = np.nan
    else:
        y_bad[0] = np.inf

    loss = QuantileLoss(0.35)
    penalty = SCADPenalty(alpha=0.04, a=3.7)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    monkeypatch.setattr(loss, "lipschitz", forbidden)

    with pytest.raises(ValueError, match=message):
        fista_lla_path(
            loss,
            penalty,
            X_bad,
            y_bad,
            alpha_path=np.asarray([0.04], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=20,
            tol=1e-6,
            fit_intercept=False,
        )


@pytest.mark.parametrize(
    ("factory", "attr", "bad_value", "message"),
    [
        (
            lambda: SCADPenalty(alpha=0.04, a=3.7),
            "a",
            np.nan,
            "SCAD penalty a must be",
        ),
        (
            lambda: SCADPenalty(alpha=0.04, a=3.7),
            "a",
            2.0,
            "SCAD penalty a must be",
        ),
        (
            lambda: MCPPenalty(alpha=0.04, gamma=3.0),
            "gamma",
            np.nan,
            "MCP penalty gamma must be",
        ),
        (
            lambda: MCPPenalty(alpha=0.04, gamma=3.0),
            "gamma",
            1.0,
            "MCP penalty gamma must be",
        ),
    ],
)
def test_quantile_fista_lla_rejects_mutated_penalty_shape_before_loss_work(
    monkeypatch, factory, attr, bad_value, message
):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = factory()
    setattr(penalty, attr, bad_value)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid penalty shape must fail before loss work")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    monkeypatch.setattr(loss, "lipschitz", forbidden)

    with pytest.raises(ValueError, match=message):
        fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.04], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=20,
            tol=1e-6,
            fit_intercept=False,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"init_coef": np.zeros((4, 1), dtype=np.float64)},
            "init_coef must be one-dimensional",
        ),
        (
            {"init_coef": np.zeros(3, dtype=np.float64)},
            "init_coef must have length n_features",
        ),
        (
            {"init_coef": np.asarray([0.0, 0.0, np.nan, 0.0])},
            "init_coef must contain finite values",
        ),
        (
            {"init_coef": np.asarray([0.0, 0.0, 1.0j, 0.0])},
            "init_coef must contain real numeric values",
        ),
        (
            {"init_intercept": np.asarray([0.1])},
            "init_intercept must be a scalar",
        ),
        (
            {"init_intercept": np.nan},
            "init_intercept must contain finite values",
        ),
        (
            {"init_intercept": 1.0j},
            "init_intercept must contain real numeric values",
        ),
        (
            {"init_intercept": True},
            "init_intercept must be a finite real scalar",
        ),
    ],
)
def test_quantile_fista_lla_rejects_invalid_warm_start_before_loss_work(
    monkeypatch, kwargs, message
):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = SCADPenalty(alpha=0.04, a=3.7)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    monkeypatch.setattr(loss, "lipschitz", forbidden)

    with pytest.raises(ValueError, match=message):
        fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.04], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=20,
            tol=1e-6,
            fit_intercept=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fit_intercept": "False"}, "fit_intercept must be boolean"),
        ({"max_lla_per_step": 0}, "max_lla_per_step must be a positive integer"),
        ({"max_lla_per_step": True}, "max_lla_per_step must be a positive integer"),
        ({"tol": 0.0}, "tol must be a finite positive number"),
        ({"tol": "1e-6"}, "tol must be a finite positive number"),
        ({"lla_tol": False}, "lla_tol must be a finite positive number"),
        ({"max_iter": 0}, "max_iter must be a positive integer or sequence"),
        ({"max_iter": True}, "max_iter must be a positive integer or sequence"),
        ({"max_iter": [0]}, "max_iter sequence must contain only positive integers"),
        ({"max_iter": [2, 2]}, "one positive integer per alpha_path step"),
        ({"alpha_path": []}, "alpha_path must be a non-empty"),
        ({"alpha_path": [0.04, 0.0]}, "alpha_path must contain finite positive"),
        ({"alpha_path": ["0.04"]}, "alpha_path must contain finite positive"),
        ({"alpha_path": [[0.08, 0.04]]}, "one-dimensional"),
        (
            {"alpha_path": [0.04, 0.08]},
            "alpha_path must be non-increasing",
        ),
    ],
)
def test_quantile_fista_lla_rejects_invalid_public_controls_before_loss_work(
    monkeypatch, kwargs, message
):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = SCADPenalty(alpha=0.04, a=3.7)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    monkeypatch.setattr(loss, "lipschitz", forbidden)

    call_kwargs = dict(
        alpha_path=np.asarray([0.04], dtype=np.float64),
        max_lla_per_step=1,
        max_iter=20,
        tol=1e-6,
        lla_tol=1e-6,
        fit_intercept=False,
    )
    call_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        fista_lla_path(loss, penalty, X, y, **call_kwargs)


@pytest.mark.parametrize(
    "penalty",
    [
        L1Penalty(alpha=0.04),
        L2Penalty(alpha=0.04),
    ],
)
def test_quantile_fista_lla_rejects_non_lla_penalty_before_loss_work(
    monkeypatch, penalty
):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    loss = QuantileLoss(0.35)

    def forbidden(*args, **kwargs):
        raise AssertionError("loss numerical work must not start")

    monkeypatch.setattr(loss, "preprocess", forbidden)
    monkeypatch.setattr(loss, "lipschitz", forbidden)

    with pytest.raises(ValueError, match="requires SCAD/MCP"):
        fista_lla_path(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.04], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=20,
            tol=1e-6,
            fit_intercept=False,
        )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.asarray([1.0, -0.2, 1.0, 1.0]), "sample_weight must be non-negative"),
        (np.asarray([1.0, np.nan, 1.0, 1.0]), "sample_weight must contain only finite values"),
        (np.asarray([1.0, 1.0, 1.0]), "sample_weight must be 1D with length n_samples"),
    ],
)
def test_quantile_fista_lla_preserves_shared_sample_weight_validation(weights, message):
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.4, -0.2, 0.6, -0.1], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = SCADPenalty(alpha=0.04, a=3.7)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid weights must fail before loss numerical work")

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(loss, "preprocess", forbidden)
        monkeypatch.setattr(loss, "lipschitz", forbidden)
        with pytest.raises(ValueError, match=message):
            fista_lla_path(
                loss,
                penalty,
                X,
                y,
                alpha_path=np.asarray([0.04], dtype=np.float64),
                max_lla_per_step=1,
                max_iter=20,
                tol=1e-6,
                fit_intercept=False,
                sample_weight=weights,
            )
    finally:
        monkeypatch.undo()
