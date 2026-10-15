"""Flat-to-active convergence ownership for PR #166 Quantile Group LLA."""

from __future__ import annotations

import warnings

import numpy as np

from statgpu.losses import QuantileLoss
from statgpu.penalties import GroupSCADPenalty
from statgpu.solvers import _quantile_group_proximal_irls_lla as solver_mod
from statgpu.solvers._convergence import ConvergenceWarning


GROUPS = [[0, 1], [2, 3]]


def test_exhausted_flat_solve_does_not_poison_later_active_convergence(monkeypatch):
    """A later converged active surrogate owns the final target verdict."""
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.8, -0.5, 0.4, -0.3], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = GroupSCADPenalty(alpha=0.3, a=3.7, groups=GROUPS)
    calls = {"irls": 0, "admm": 0}

    def exhausted_flat_irls(
        X_arg,
        y_arg,
        penalty=None,
        max_iter=100,
        tol=1e-6,
        init_coef=None,
        eps=1e-8,
        sample_weight=None,
        fit_intercept=False,
    ):
        calls["irls"] += 1
        if init_coef is None:
            # The accepted flat solve leaves the flat region and consumes the
            # declared budget.
            return np.zeros(X_arg.shape[1], dtype=np.float64), int(max_iter)
        # The one-step diagnostic probe still moves materially, proving the
        # preceding flat solve was genuinely exhausted rather than converged
        # exactly on its last allowed iteration. This probe is not accepted.
        return np.asarray(init_coef, dtype=np.float64) + 0.1, int(max_iter)

    def converged_active_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["admm"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        return current.copy(), 1

    monkeypatch.setattr(loss, "irls", exhausted_flat_irls)
    monkeypatch.setattr(solver_mod, "admm_solver", converged_active_admm)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=2,
            max_iter=2,
            tol=1e-12,
            lla_tol=1e-12,
            fit_intercept=False,
            init_coef=np.full(4, 2.0, dtype=np.float64),
            fail_on_target_nonconvergence=True,
        )

    assert calls["irls"] == 2  # flat solve + diagnostic boundary probe
    assert calls["admm"] == 1
    assert n_iter == 3
    np.testing.assert_array_equal(coef, np.zeros(4, dtype=np.float64))
    assert intercept == 0.0
