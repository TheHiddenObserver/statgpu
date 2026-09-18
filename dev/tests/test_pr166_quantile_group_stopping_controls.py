"""Stopping-control validation for PR #166 Quantile Group auto routes."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


GROUPS = [[0, 1], [2, 3]]


def _data():
    X = np.eye(8, 4, dtype=np.float64)
    y = np.asarray([0.8, -0.5, 0.4, -0.3, 0.7, -0.2, 0.3, -0.1])
    idx = np.arange(X.shape[0])
    folds = [(idx[:4], idx[4:]), (idx[4:], idx[:4])]
    return X, y, folds


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("max_iter", 1.5, "max_iter must be a positive integer"),
        ("max_iter", True, "max_iter must be a positive integer"),
        ("max_lla_iters", 0, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", 2.5, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", False, "max_lla_iters must be a positive integer"),
        ("tol", 0.0, "tol must be a finite positive number"),
        ("tol", np.inf, "tol must be a finite positive number"),
        ("lla_tol", 0.0, "lla_tol must be a finite positive number"),
        ("lla_tol", np.nan, "lla_tol must be a finite positive number"),
    ],
)
def test_direct_quantile_group_auto_rejects_invalid_stopping_controls(
    name, value, message
):
    X, y, _ = _data()
    kwargs = dict(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha=0.04,
        solver="auto",
        device="cpu",
        compute_inference=False,
        max_iter=20,
        tol=1e-6,
        max_lla_iters=6,
        lla_tol=1e-6,
    )
    kwargs[name] = value
    model = PenalizedGeneralizedLinearModel(**kwargs)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("max_iter", 1.5, "max_iter must be a positive integer"),
        ("max_iter", True, "max_iter must be a positive integer"),
        ("tol", 0.0, "tol must be a finite positive number"),
        ("tol", np.inf, "tol must be a finite positive number"),
    ],
)
def test_quantile_group_cv_rejects_invalid_stopping_controls_as_user_errors(
    name, value, message
):
    X, y, folds = _data()
    kwargs = dict(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha_grid=np.asarray([0.05, 0.03], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=20,
        tol=1e-6,
    )
    kwargs[name] = value
    model = PenalizedGLM_CV(**kwargs)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)

    assert model.alpha_ is None
    assert model.estimator_ is None



@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("max_lla_iters", 0, "max_lla_iters must be a positive integer"),
        ("tol", 0.0, "tol must be a finite positive number"),
        ("lla_tol", np.nan, "lla_tol must be a finite positive number"),
    ],
)
def test_direct_public_stopping_replacement_is_validated_at_refit(
    name, value, message
):
    X, y, _ = _data()
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha=0.04,
        solver="auto",
        device="cpu",
        compute_inference=False,
        max_iter=20,
        tol=1e-6,
        max_lla_iters=6,
        lla_tol=1e-6,
    )
    setattr(model, name, value)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_direct_public_stopping_replacement_reaches_group_solver(monkeypatch):
    from statgpu.solvers import _quantile_group_proximal_irls_lla as solver_mod

    X, y, _ = _data()
    seen = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen["max_iter"] = list(kwargs["max_iter"])
        seen["max_lla_per_step"] = kwargs["max_lla_per_step"]
        seen["tol"] = kwargs["tol"]
        seen["lla_tol"] = kwargs["lla_tol"]
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        solver_mod, "quantile_group_proximal_irls_lla_solver", fake_solver
    )
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha=0.04,
        solver="auto",
        device="cpu",
        compute_inference=False,
        max_iter=20,
        tol=1e-6,
        max_lla_iters=6,
        lla_tol=1e-6,
    )
    model.max_iter = 7
    model.max_lla_iters = 9
    model.tol = 2e-5
    model.lla_tol = 3e-5
    model.fit(X, y)

    assert seen["max_iter"][-1] == 7
    assert seen["max_lla_per_step"] == 3
    assert seen["tol"] == pytest.approx(2e-5)
    assert seen["lla_tol"] == pytest.approx(3e-5)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("tol", 0.0, "tol must be a finite positive number"),
    ],
)
def test_cv_public_stopping_replacement_is_validated_at_refit(
    name, value, message
):
    X, y, folds = _data()
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha_grid=np.asarray([0.05, 0.03], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        solver="auto",
        device="cpu",
        max_iter=20,
        tol=1e-6,
    )
    setattr(model, name, value)

    with pytest.raises(ValueError, match=message):
        model.fit(X, y)

    assert model.alpha_ is None
    assert model.estimator_ is None
