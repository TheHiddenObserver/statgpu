"""Convergence-reporting contract for PR #166 Quantile group IRLS-LLA."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
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


def test_active_quantile_group_target_lla_exhaustion_warns_at_external_callsite(monkeypatch):
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def drifting_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        if calls["value"] == 1:
            return current + 0.05, 1
        return current.copy(), 1

    monkeypatch.setattr(solver_mod, "admm_solver", drifting_admm)

    with pytest.warns(ConvergenceWarning, match="max_lla_per_step=1 at the target") as caught:
        coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
            loss, penalty, X, y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1, max_iter=2, tol=1e-12, lla_tol=1e-12,
            fit_intercept=False,
        )

    assert caught[0].filename == __file__
    assert calls["value"] == 2
    assert n_iter == 2
    np.testing.assert_allclose(coef, np.full(4, 0.05), rtol=0.0, atol=0.0)
    assert intercept == 0.0


def test_estimator_target_warning_points_to_external_fit_callsite(monkeypatch):
    X, y, _, _ = _drifting_problem()

    def drifting_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        return current + 0.001, 1

    monkeypatch.setattr(solver_mod, "admm_solver", drifting_admm)
    model = PenalizedGeneralizedLinearModel(
        loss="quantile", loss_kwargs={"quantile": 0.35},
        penalty="group_scad", penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha=0.3, solver="auto", device="cpu", compute_inference=False,
        max_iter=2, tol=1e-12, lla_tol=1e-12,
    )

    with pytest.warns(ConvergenceWarning) as caught:
        model.fit(X, y)

    assert caught[-1].filename == __file__
    assert model._selected_solver == "group_proximal_irls_lla"


def test_active_quantile_group_target_lla_exhaustion_fails_in_strict_cv_mode(monkeypatch):
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def drifting_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        if calls["value"] == 1:
            return current + 0.05, 1
        return current.copy(), 1

    monkeypatch.setattr(solver_mod, "admm_solver", drifting_admm)

    with pytest.raises(FloatingPointError, match="max_lla_per_step=1 at the target"):
        solver_mod.quantile_group_proximal_irls_lla_solver(
            loss, penalty, X, y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1, max_iter=2, tol=1e-12, lla_tol=1e-12,
            fit_intercept=False, fail_on_target_nonconvergence=True,
        )

    assert calls["value"] == 2


def test_active_quantile_group_target_irls_budget_exhaustion_warns_even_when_lla_delta_is_small(monkeypatch):
    """Inner IRLS exhaustion must outrank a looser outer LLA tolerance."""
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def slowly_drifting_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        return current + 6e-4, 1

    monkeypatch.setattr(solver_mod, "admm_solver", slowly_drifting_admm)

    with pytest.warns(
        ConvergenceWarning,
        match="active target reached max_iter=2 in Quantile IRLS",
    ) as caught:
        coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
            loss, penalty, X, y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1, max_iter=2, tol=5e-4, lla_tol=1e-2,
            fit_intercept=False,
        )

    assert caught[0].filename == __file__
    assert calls["value"] == 2
    assert n_iter == 2
    np.testing.assert_allclose(coef, np.full(4, 1.2e-3), rtol=0.0, atol=1e-15)
    assert intercept == 0.0


def test_active_quantile_group_target_irls_budget_exhaustion_fails_in_strict_cv_mode(monkeypatch):
    """Strict CV must not score a target whose active IRLS loop exhausted."""
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def slowly_drifting_admm(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        return current + 6e-4, 1

    monkeypatch.setattr(solver_mod, "admm_solver", slowly_drifting_admm)

    with pytest.raises(
        FloatingPointError,
        match="active target reached max_iter=2 in Quantile IRLS",
    ):
        solver_mod.quantile_group_proximal_irls_lla_solver(
            loss, penalty, X, y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1, max_iter=2, tol=5e-4, lla_tol=1e-2,
            fit_intercept=False, fail_on_target_nonconvergence=True,
        )

    assert calls["value"] == 2


def test_later_converged_active_irls_clears_prior_active_exhaustion(monkeypatch):
    """A later active surrogate that truly converges owns the final verdict."""
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def exhaust_then_converge(loss_arg, penalty_arg, X_arg, y_arg, **kwargs):
        calls["value"] += 1
        current = np.asarray(kwargs["init_coef"], dtype=np.float64)
        if calls["value"] <= 2:
            return current + 6e-4, 1
        return current.copy(), 1

    monkeypatch.setattr(solver_mod, "admm_solver", exhaust_then_converge)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
            loss, penalty, X, y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=2, max_iter=2, tol=5e-4, lla_tol=1e-3,
            fit_intercept=False, fail_on_target_nonconvergence=True,
        )

    assert calls["value"] == 3
    assert n_iter == 3
    np.testing.assert_allclose(coef, np.full(4, 1.2e-3), rtol=0.0, atol=1e-15)
    assert intercept == 0.0


def test_intermediate_quantile_group_lla_exhaustion_is_only_a_warm_path(monkeypatch):
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
        loss, penalty, X, y,
        alpha_path=np.asarray([0.4, 0.3], dtype=np.float64),
        max_lla_per_step=1, max_iter=[2, 2], tol=1e-12, lla_tol=1e-12,
        fit_intercept=False, fail_on_target_nonconvergence=True,
    )

    assert calls["value"] == 3
    assert n_iter == 3
    np.testing.assert_allclose(coef, np.full(4, 0.1), rtol=0.0, atol=0.0)
    assert intercept == 0.0


def _install_genuinely_exhausted_flat_irls(monkeypatch, loss):
    calls = {"value": 0}

    def exhausted_irls(
        X_arg, y_arg, penalty=None, max_iter=100, tol=1e-6,
        init_coef=None, eps=1e-8, sample_weight=None, fit_intercept=False,
    ):
        calls["value"] += 1
        if init_coef is None:
            return np.zeros(X_arg.shape[1], dtype=np.float64), int(max_iter)
        # The boundary probe still sees a material next-step move, so this is
        # a true exhausted state rather than last-iteration convergence.
        return np.asarray(init_coef, dtype=np.float64) + 0.1, int(max_iter)

    monkeypatch.setattr(loss, "irls", exhausted_irls)
    return calls


def test_flat_quantile_group_target_irls_budget_exhaustion_warns_at_external_callsite(monkeypatch):
    X, y, loss, penalty = _drifting_problem()
    calls = _install_genuinely_exhausted_flat_irls(monkeypatch, loss)

    with pytest.warns(ConvergenceWarning, match="flat target reached max_iter=2 in Quantile IRLS") as caught:
        coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
            loss, penalty, X, y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1, max_iter=2, tol=1e-12, lla_tol=1e-12,
            fit_intercept=False, init_coef=np.full(4, 2.0, dtype=np.float64),
        )

    assert calls["value"] == 2  # solve + diagnostic one-step boundary probe
    assert caught[0].filename == __file__
    assert n_iter == 2
    np.testing.assert_array_equal(coef, np.zeros(4, dtype=np.float64))
    assert intercept == 0.0


def test_flat_quantile_group_target_irls_budget_exhaustion_fails_in_strict_cv_mode(monkeypatch):
    X, y, loss, penalty = _drifting_problem()
    calls = _install_genuinely_exhausted_flat_irls(monkeypatch, loss)

    with pytest.raises(FloatingPointError, match="flat target reached max_iter=2 in Quantile IRLS"):
        solver_mod.quantile_group_proximal_irls_lla_solver(
            loss, penalty, X, y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1, max_iter=2, tol=1e-12, lla_tol=1e-12,
            fit_intercept=False, init_coef=np.full(4, 2.0, dtype=np.float64),
            fail_on_target_nonconvergence=True,
        )

    assert calls["value"] == 2


def test_flat_quantile_group_last_iteration_convergence_is_not_false_exhaustion(monkeypatch):
    """n_iter == max_iter is ambiguous; a stable boundary probe must win."""
    X, y, loss, penalty = _drifting_problem()
    calls = {"value": 0}

    def converged_on_boundary(
        X_arg, y_arg, penalty=None, max_iter=100, tol=1e-6,
        init_coef=None, eps=1e-8, sample_weight=None, fit_intercept=False,
    ):
        calls["value"] += 1
        point = np.full(X_arg.shape[1], 2.0, dtype=np.float64)
        return point, int(max_iter)

    monkeypatch.setattr(loss, "irls", converged_on_boundary)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        coef, intercept, n_iter = solver_mod.quantile_group_proximal_irls_lla_solver(
            loss, penalty, X, y,
            alpha_path=np.asarray([0.3], dtype=np.float64),
            max_lla_per_step=1, max_iter=2, tol=1e-12, lla_tol=1e-12,
            fit_intercept=False, init_coef=np.full(4, 2.0, dtype=np.float64),
            fail_on_target_nonconvergence=True,
        )

    assert calls["value"] == 2  # solve + probe; probe is not accepted/countable
    assert n_iter == 2
    np.testing.assert_array_equal(coef, np.full(4, 2.0, dtype=np.float64))
    assert intercept == 0.0
