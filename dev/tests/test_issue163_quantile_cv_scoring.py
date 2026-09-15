"""Issue #163 regressions for Quantile CV scoring at the requested tau."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu._config import Device
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


def _data(seed=16321, n=64):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2)).astype(np.float64)
    noise = rng.exponential(scale=0.22, size=n) - 0.22
    y = (0.3 + X @ np.array([0.7, -0.25]) + noise).astype(np.float64)
    folds = [
        (np.arange(0, n // 2), np.arange(n // 2, n)),
        (np.arange(n // 2, n), np.arange(0, n // 2)),
    ]
    return X, y, folds


def _pinball(y, eta, tau, sample_weight=None):
    u = np.asarray(y) - np.asarray(eta)
    values = np.where(u >= 0.0, tau * u, (tau - 1.0) * u)
    if sample_weight is None:
        return float(np.mean(values))
    return float(np.average(values, weights=np.asarray(sample_weight)))


@pytest.mark.parametrize("weighted", [False, True])
def test_quantile_cv_general_scores_use_requested_tau(weighted):
    X, y, folds = _data()
    tau = 0.2
    alpha = 0.035
    weights = (
        np.linspace(0.45, 1.8, X.shape[0], dtype=np.float64)
        if weighted
        else None
    )

    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": tau},
        penalty="l2",
        alpha_grid=np.array([alpha], dtype=np.float64),
        cv=folds,
        solver="auto",
        device="cpu",
        max_iter=500,
        tol=1e-9,
    )
    observed = cv._compute_cv_scores(
        X,
        y,
        np.array([alpha], dtype=np.float64),
        Device.CPU,
        folds,
        sample_weight=weights,
        max_iter=500,
        tol=1e-9,
        strict=True,
    )

    expected = []
    median_scores = []
    for train_idx, val_idx in folds:
        train_weight = weights[train_idx] if weights is not None else None
        val_weight = weights[val_idx] if weights is not None else None
        model = PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": tau},
            penalty="l2",
            alpha=alpha,
            solver="irls",
            device="cpu",
            max_iter=500,
            tol=1e-9,
        ).fit(X[train_idx], y[train_idx], sample_weight=train_weight)
        eta = X[val_idx] @ model.coef_ + model.intercept_
        expected.append(_pinball(y[val_idx], eta, tau, val_weight))
        median_scores.append(_pinball(y[val_idx], eta, 0.5, val_weight))

    np.testing.assert_allclose(
        observed[:, 0], np.asarray(expected), rtol=2e-8, atol=2e-10
    )
    assert float(np.max(np.abs(observed[:, 0] - np.asarray(median_scores)))) > 1e-4

    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    assert contract._QUANTILE_CV_LEVEL.get() is None


def test_quantile_eval_dispatch_uses_call_local_tau_without_registering_fast_residual():
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod
    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    # Validation gets a backend-native registry helper, but there is still no
    # Quantile residual registration. Therefore this PR does not silently
    # enable the incomplete fold-batched sparse solver.
    assert "quantile" not in cv_mod._LOSS_RESIDUAL_FNS
    assert "quantile" in cv_mod._LOSS_VALLOSS_FNS

    eta = np.array([-0.2, 0.1, 0.8], dtype=np.float64)
    y = np.array([0.4, -0.1, 1.2], dtype=np.float64)
    tau = 0.23

    token = contract._QUANTILE_CV_LEVEL.set(tau)
    try:
        eval_fn, _ = cv_mod._LOSS_EVAL_DISPATCH["quantile"]
        dispatch_loss = eval_fn(eta, y)
        registry_loss = cv_mod._LOSS_VALLOSS_FNS["quantile"](eta, y)
    finally:
        contract._QUANTILE_CV_LEVEL.reset(token)

    u = y - eta
    expected = np.where(u >= 0.0, tau * u, (tau - 1.0) * u)
    np.testing.assert_allclose(dispatch_loss, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(registry_loss, expected, rtol=0.0, atol=0.0)
    assert contract._QUANTILE_CV_LEVEL.get() is None


def test_quantile_sparse_fold_batch_is_fail_safe_to_general_path():
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod

    # The guard returns before touching backend arrays. This proves that the
    # incomplete private fold-batch Quantile route is not accidentally promoted
    # into a new numerical capability by the provenance/scoring repair.
    result = cv_mod._glm_sparse_cv_folds(
        None,
        None,
        [],
        np.array([0.03]),
        "l1",
        1.0,
        5,
        1e-4,
        "quantile",
        "torch",
    )
    assert result is None


def test_quantile_scad_weighted_validation_uses_requested_tau():
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod
    from statgpu.linear_model.penalized import _quantile_solver_contract as contract

    X, y, _ = _data(seed=16322, n=48)
    train_idx = np.arange(0, 32)
    val_idx = np.arange(32, 48)
    tau = 0.2
    alpha = 0.025
    val_weight = np.linspace(0.5, 1.7, val_idx.size, dtype=np.float64)

    token = contract._QUANTILE_CV_LEVEL.set(tau)
    try:
        path = cv_mod._scad_mcp_cv_path(
            "quantile",
            X[train_idx],
            y[train_idx],
            np.array([alpha], dtype=np.float64),
            "scad",
            1.0,
            80,
            1e-6,
            Device.CPU,
            X_val=X[val_idx],
            y_val=y[val_idx],
            val_sample_weight=val_weight,
            return_path=True,
        )
    finally:
        contract._QUANTILE_CV_LEVEL.reset(token)

    assert path is not None
    assert path["scores"].shape == (1,)
    eta = X[val_idx] @ path["coef"][0] + path["intercept"][0]
    expected = _pinball(y[val_idx], eta, tau, val_weight)
    median = _pinball(y[val_idx], eta, 0.5, val_weight)
    assert path["scores"][0] == pytest.approx(expected, rel=2e-8, abs=2e-10)
    assert abs(path["scores"][0] - median) > 1e-4
    assert contract._QUANTILE_CV_LEVEL.get() is None


def test_quantile_eval_preserves_historical_default_outside_cv_context():
    from statgpu.linear_model.penalized import _penalized_cv as cv_mod

    eta = np.array([0.0, 0.5], dtype=np.float64)
    y = np.array([1.0, 0.0], dtype=np.float64)
    eval_fn, _ = cv_mod._LOSS_EVAL_DISPATCH["quantile"]
    dispatch_loss = eval_fn(eta, y)
    registry_loss = cv_mod._LOSS_VALLOSS_FNS["quantile"](eta, y)
    expected = np.array([0.5, 0.25], dtype=np.float64)
    np.testing.assert_allclose(dispatch_loss, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(registry_loss, expected, rtol=0.0, atol=0.0)
