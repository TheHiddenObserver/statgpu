"""Convergence-reporting contract for PR #166 Quantile group IRLS-LLA."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import GroupSCADPenalty
from statgpu.solvers._convergence import ConvergenceWarning
from statgpu.solvers import _quantile_group_proximal_irls_lla as solver_mod


GROUPS = [[0, 1], [2, 3]]


def _drifting_problem():
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.8, -0.5, 0.4, -0.3], dtype=np.float64)
    loss = QuantileLoss(0.35)
    penalty = GroupSCADPenalty(alpha=0.3, a=3.7, groups=GROUPS)
    return X, y, loss, penalty


def test_active_quantile_group_target_lla_exhaustion_warns_at_external_callsite(
    monkeypatch,
):
    """An exhausted direct target is visible and points to the caller."""
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def drifting_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        return current + 0.05, 1

    monkeypatch.setattr(solver_mod, "admm_solver", drifting_admm)

    with pytest.warns(
        ConvergenceWarning,
        match="max_lla_per_step=1 at the target",
    ) as caught:
        coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=2,
            tol=1e-12,
            lla_tol=1e-12,
            fit_intercept=False,
        )

    assert caught[0].filename == __file__
    assert calls["value"] == 2
    assert n_iter == 2
    np.testing.assert_allclose(coef, np.full(4, 0.1), rtol=0.0, atol=0.0)
    assert intercept == 0.0


def test_active_quantile_group_target_lla_exhaustion_fails_in_strict_cv_mode(
    monkeypatch,
):
    """A CV candidate may not be scored after target LLA exhaustion."""
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def drifting_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        return current + 0.05, 1

    monkeypatch.setattr(solver_mod, "admm_solver", drifting_admm)

    with pytest.raises(
        FloatingPointError,
        match="max_lla_per_step=1 at the target",
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
            lla_tol=1e-12,
            fit_intercept=False,
            fail_on_target_nonconvergence=True,
        )

    assert calls["value"] == 2


def test_intermediate_quantile_group_lla_exhaustion_is_only_a_warm_path(monkeypatch):
    """An inexact intermediate alpha is allowed when the final target LLA closes."""
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def warm_then_converged_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        if calls["value"] <= 2:
            return current + 0.05, 1
        return current.copy(), 1

    monkeypatch.setattr(solver_mod, "admm_solver", warm_then_converged_admm)

    coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
        loss,
        penalty,
        X,
        y,
        alpha_path=np.asarray([0.4, 0.3], dtype=np.float64),
        max_lla_per_step=1,
        max_iter=[2, 2],
        tol=1e-12,
        lla_tol=1e-12,
        fit_intercept=False,
        fail_on_target_nonconvergence=True,
    )

    assert calls["value"] == 3
    assert n_iter == 3
    np.testing.assert_allclose(coef, np.full(4, 0.1), rtol=0.0, atol=0.0)
    assert intercept == 0.0


def test_flat_quantile_group_target_irls_budget_exhaustion_warns_at_external_callsite(
    monkeypatch,
):
    """A flat direct target reports exhaustion and points to the caller."""
    X, y, loss, penalty = _drifting_problem()

    def exhausted_irls(
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
        return np.zeros(X_arg.shape[1], dtype=np.float64), int(max_iter)

    monkeypatch.setattr(loss, "irls", exhausted_irls)

    with pytest.warns(
        ConvergenceWarning,
        match="flat target reached max_iter=2 in Quantile IRLS",
    ) as caught:
        coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1,
            max_iter=2,
            tol=1e-12,
            lla_tol=1e-12,
            fit_intercept=False,
            init_coef=np.full(4, 2.0, dtype=np.float64),
        )

    assert caught[0].filename == __file__
    assert n_iter == 2
    np.testing.assert_array_equal(coef, np.zeros(4, dtype=np.float64))
    assert intercept == 0.0


def test_flat_quantile_group_target_irls_budget_exhaustion_fails_in_strict_cv_mode(
    monkeypatch,
):
    """A flat CV candidate may not be scored after Quantile IRLS exhaustion."""
    X, y, loss, penalty = _drifting_problem()

    def exhausted_irls(
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
        return np.zeros(X_arg.shape[1], dtype=np.float64), int(max_iter)

    monkeypatch.setattr(loss, "irls", exhausted_irls)

    with pytest.raises(
        FloatingPointError,
        match="flat target reached max_iter=2 in Quantile IRLS",
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
            lla_tol=1e-12,
            fit_intercept=False,
            init_coef=np.full(4, 2.0, dtype=np.float64),
            fail_on_target_nonconvergence=True,
        )