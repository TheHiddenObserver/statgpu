"""Post-merge review regressions for the Quantile × ADMM boundary."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import admm_solver as glm_core_admm_solver
from statgpu.linear_model.penalized import PenalizedGLM_CV, PenalizedQuantileRegression
from statgpu.losses import QuantileLoss
from statgpu.penalties import L1Penalty
from statgpu.solvers import admm_solver


def _data(seed=16491):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(48, 3))
    y = 0.15 + X @ np.array([0.55, -0.3, 0.12])
    y = y + rng.exponential(scale=0.18, size=X.shape[0]) - 0.18
    return X, y


def test_direct_quantile_admm_fails_before_backend_work(monkeypatch):
    X, y = _data()
    model = PenalizedQuantileRegression(
        quantile=0.2,
        penalty="l1",
        alpha=0.04,
        solver="admm",
        device="cpu",
    )

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("backend numerical work must not start")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(ValueError, match="not a maintained Quantile route"):
        model.fit(X, y)


@pytest.mark.parametrize("cv_strategy", ["strict", "two_stage"])
def test_cv_quantile_admm_fails_before_alpha_grid(monkeypatch, cv_strategy):
    X, y = _data(seed=16492)
    model = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.2},
        penalty="l1",
        cv=2,
        solver="admm",
        cv_strategy=cv_strategy,
        acknowledge_approx=(cv_strategy == "two_stage"),
        device="cpu",
    )

    def forbidden_grid(*args, **kwargs):
        raise AssertionError("alpha-grid numerical work must not start")

    monkeypatch.setattr(model, "_generate_alpha_grid", forbidden_grid)
    with pytest.raises(ValueError, match="not a maintained Quantile route"):
        model.fit(X, y)


@pytest.mark.parametrize("solver_fn", [admm_solver, glm_core_admm_solver])
def test_public_admm_solver_rejects_quantile_before_gradient_work(
    monkeypatch, solver_fn
):
    X, y = _data(seed=16493)
    loss = QuantileLoss(quantile=0.2)
    penalty = L1Penalty(alpha=0.04)

    def forbidden_gradient(*args, **kwargs):
        raise AssertionError("ADMM gradient work must not start")

    monkeypatch.setattr(loss, "gradient", forbidden_gradient)
    with pytest.raises(ValueError, match="requires a smooth loss gradient"):
        solver_fn(loss, penalty, X, y, max_iter=5)
