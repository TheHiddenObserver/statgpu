"""Correlated-design correctness for the Group Lasso composite objective."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import get_glm_loss
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import GroupLassoPenalty
from statgpu.solvers import fista_solver


_GROUPS = [[0, 3], [1, 2]]


def _correlated_data(seed=10401):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=180)
    z1 = rng.normal(size=180)
    X = np.column_stack(
        [
            z0 + 0.05 * rng.normal(size=180),
            z1 + 0.08 * rng.normal(size=180),
            0.85 * z1 + 0.12 * rng.normal(size=180),
            0.9 * z0 + 0.1 * rng.normal(size=180),
        ]
    )
    y = 0.35 + X @ np.array([0.7, -0.45, 0.25, 0.55])
    y += rng.normal(scale=0.07, size=X.shape[0])
    return X, y


def _kkt_residual(model, X, y):
    prediction = model.predict(X)
    gradient = X.T @ (prediction - y) / X.shape[0]
    intercept_gradient = float(np.mean(prediction - y))
    residuals = [abs(intercept_gradient)]
    for group in _GROUPS:
        idx = np.asarray(group, dtype=np.int64)
        beta_g = np.asarray(model.coef_)[idx]
        grad_g = gradient[idx]
        norm = np.linalg.norm(beta_g)
        threshold = model.alpha * np.sqrt(idx.size)
        if norm > 1e-9:
            residuals.append(
                np.linalg.norm(grad_g + threshold * beta_g / norm)
            )
        else:
            residuals.append(max(np.linalg.norm(grad_g) - threshold, 0.0))
    return float(max(residuals))


def test_correlated_group_lasso_matches_centered_fista_reference_and_kkt():
    X, y = _correlated_data()
    alpha = 0.08
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": _GROUPS},
        alpha=alpha,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_iter=5000,
        tol=1e-10,
    ).fit(X, y)

    X_mean = np.mean(X, axis=0)
    y_mean = float(np.mean(y))
    X_centered = X - X_mean
    y_centered = y - y_mean
    reference_penalty = GroupLassoPenalty(alpha=alpha, groups=_GROUPS)
    reference_coef, _ = fista_solver(
        get_glm_loss("squared_error"),
        reference_penalty,
        X_centered,
        y_centered,
        max_iter=10000,
        tol=1e-12,
    )
    reference_coef = np.asarray(reference_coef)
    reference_intercept = y_mean - X_mean @ reference_coef

    assert model._selected_solver == "fista"
    # The composite KKT residual is the primary correctness gate. The two
    # independently stopped accelerated paths can differ by a few 1e-6 on a
    # highly collinear design while satisfying the same optimum conditions.
    assert _kkt_residual(model, X, y) < 2e-5
    np.testing.assert_allclose(
        model.coef_, reference_coef, rtol=1e-4, atol=1.2e-5
    )
    assert model.intercept_ == pytest.approx(
        reference_intercept, rel=1e-4, abs=1.2e-5
    )
