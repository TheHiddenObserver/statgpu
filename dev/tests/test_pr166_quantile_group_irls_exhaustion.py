"""Convergence-failure contract for PR #166 Quantile group IRLS-LLA."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import GroupSCADPenalty
from statgpu.solvers._convergence import ConvergenceWarning
from statgpu.solvers import _quantile_group_proximal_irls_lla as solver_mod


GROUPS = [[0, 1], [2, 3]]


def test_active_quantile_group_irls_exhaustion_fails_closed(monkeypatch):
    """An unconverged IRLS surrogate must not feed an approximate point to LLA."""
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.8, -0.5, 0.4, -0.3], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = GroupSCADPenalty(alpha=0.3, a=3.7, groups=GROUPS)
    calls = {"value": 0}

    def drifting_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        # Deliberately keep the outer IRLS iterate moving by a visible amount.
        return current + 0.05, 1

    monkeypatch.setattr(solver_mod, "admm_solver", drifting_admm)

    with pytest.raises(
        ConvergenceWarning,
        match="did not converge within 2 IRLS iterations",
    ):
        solver_mod.quantile_group_proximal_irls_lla_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=2,
            tol=1e-12,
            fit_intercept=False,
        )

    assert calls["value"] == 2
