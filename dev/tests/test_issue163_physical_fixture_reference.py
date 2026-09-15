"""Hosted reproduction of the PR164 physical validator's CPU reference contract."""

from __future__ import annotations

import numpy as np

from dev.benchmarks import validate_quantile_solver_provenance_gpu as gate
from statgpu._config import Device
from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)


def _pinball(y, eta, q, sample_weight):
    y = np.asarray(y, dtype=np.float64).ravel()
    eta = np.asarray(eta, dtype=np.float64).ravel()
    u = y - eta
    values = np.where(u >= 0.0, q * u, (q - 1.0) * u)
    return float(np.average(values, weights=np.asarray(sample_weight, dtype=np.float64)))


def test_physical_fixture_public_fit_matches_direct_cv_scores():
    """Public fit must not reinterpret fixed folds or requested Quantile scoring."""
    X, y, weights, folds = gate._data()
    alphas = np.asarray([0.045, 0.02], dtype=np.float64)

    direct_owner = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": gate.Q},
        penalty="l2",
        alpha_grid=alphas,
        cv=2,
        cv_splits=folds,
        random_state=164,
        solver="auto",
        device="cpu",
        cv_strategy="strict",
        max_iter=600,
        tol=1e-9,
    )
    direct = direct_owner._compute_cv_scores(
        X,
        y,
        alphas,
        Device.CPU,
        folds,
        sample_weight=weights,
        max_iter=600,
        tol=1e-9,
        strict=True,
    )

    public = gate._cv_fit(
        X,
        y,
        weights,
        folds,
        penalty="l2",
        alpha_grid=alphas,
        device="cpu",
        cv_strategy="strict",
        max_iter=600,
        tol=1e-9,
    )
    public_scores = np.asarray(public.cv_results_["all_scores"], dtype=np.float64)
    np.testing.assert_allclose(public_scores, direct, rtol=0.0, atol=0.0)


def test_physical_fixture_generic_and_typed_irls_children_are_equivalent():
    """Typed Quantile and generic Quantile children must solve the same fold problem."""
    X, y, weights, folds = gate._data()
    alpha = 0.045

    for train_idx, val_idx in folds:
        generic = PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": gate.Q},
            penalty="l2",
            alpha=alpha,
            solver="irls",
            device="cpu",
            max_iter=600,
            tol=1e-9,
        ).fit(
            X[train_idx],
            y[train_idx],
            sample_weight=weights[train_idx],
        )
        typed = PenalizedQuantileRegression(
            quantile=gate.Q,
            penalty="l2",
            alpha=alpha,
            solver="irls",
            device="cpu",
            max_iter=600,
            tol=1e-9,
        ).fit(
            X[train_idx],
            y[train_idx],
            sample_weight=weights[train_idx],
        )

        np.testing.assert_allclose(typed.coef_, generic.coef_, rtol=0.0, atol=0.0)
        assert float(typed.intercept_) == float(generic.intercept_)

        generic_score = _pinball(
            y[val_idx],
            X[val_idx] @ np.asarray(generic.coef_) + float(generic.intercept_),
            gate.Q,
            weights[val_idx],
        )
        typed_score = _pinball(
            y[val_idx],
            X[val_idx] @ np.asarray(typed.coef_) + float(typed.intercept_),
            gate.Q,
            weights[val_idx],
        )
        assert typed_score == generic_score


def test_physical_fixture_direct_cv_scores_match_generic_manual_oracle():
    """The existing hosted q-scoring oracle is exercised on the exact P100 fixture."""
    X, y, weights, folds = gate._data()
    alphas = np.asarray([0.045, 0.02], dtype=np.float64)
    owner = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": gate.Q},
        penalty="l2",
        alpha_grid=alphas,
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=600,
        tol=1e-9,
    )
    observed = owner._compute_cv_scores(
        X,
        y,
        alphas,
        Device.CPU,
        folds,
        sample_weight=weights,
        max_iter=600,
        tol=1e-9,
        strict=True,
    )

    expected = np.empty_like(observed)
    median = np.empty_like(observed)
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        for alpha_index, alpha in enumerate(alphas):
            child = PenalizedGeneralizedLinearModel(
                loss="quantile",
                loss_kwargs={"quantile": gate.Q},
                penalty="l2",
                alpha=float(alpha),
                solver="irls",
                device="cpu",
                max_iter=600,
                tol=1e-9,
            ).fit(
                X[train_idx],
                y[train_idx],
                sample_weight=weights[train_idx],
            )
            eta = X[val_idx] @ np.asarray(child.coef_) + float(child.intercept_)
            expected[fold_index, alpha_index] = _pinball(
                y[val_idx], eta, gate.Q, weights[val_idx]
            )
            median[fold_index, alpha_index] = _pinball(
                y[val_idx], eta, 0.5, weights[val_idx]
            )

    np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=2e-10)
    assert float(np.max(np.abs(observed - median))) > 1e-4
