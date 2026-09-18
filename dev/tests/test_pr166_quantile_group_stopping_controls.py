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
        ("max_lla_iters", 2.5, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", "6", "max_lla_iters must be a positive integer"),
        ("lla_tol", False, "lla_tol must be a finite positive number"),
        ("lla_tol", "1e-6", "lla_tol must be a finite positive number"),
    ],
)
def test_explicit_group_fista_rejects_coerced_lla_controls_before_numerics(
    monkeypatch, name, value, message
):
    X, y, _ = _data()
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha=0.04,
        solver="fista",
        device="cpu",
        compute_inference=False,
        max_iter=20,
        tol=1e-6,
        max_lla_iters=6,
        lla_tol=1e-6,
    )
    setattr(model, name, value)

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("backend numerical work must not start")

    monkeypatch.setattr(model, "_get_backend", forbidden_backend)
    with pytest.raises(ValueError, match=message):
        model.fit(X, y)


def test_direct_group_public_solver_replacement_from_fista_to_auto_uses_auto_route(
    monkeypatch,
):
    from statgpu.solvers import _quantile_group_proximal_irls_lla as solver_mod

    X, y, _ = _data()
    seen = {"group": 0}

    def fake_group_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen["group"] += 1
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        solver_mod, "quantile_group_proximal_irls_lla_solver", fake_group_solver
    )
    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha=0.04,
        solver="fista",
        device="cpu",
        compute_inference=False,
        max_iter=20,
        tol=1e-6,
        max_lla_iters=6,
        lla_tol=1e-6,
    )
    model.solver = "auto"
    model.fit(X, y)

    assert model._solver == "auto"
    assert model._selected_solver == "group_proximal_irls_lla"
    assert seen["group"] == 1


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("max_iter", 0, "max_iter must be a positive integer"),
        ("max_iter", 1.5, "max_iter must be a positive integer"),
        ("max_iter", True, "max_iter must be a positive integer"),
        ("max_lla_iters", 0, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", 2, "max_lla_iters must be at least 3"),
        ("max_lla_iters", 2.5, "max_lla_iters must be a positive integer"),
        ("max_lla_iters", False, "max_lla_iters must be a positive integer"),
        ("tol", 0.0, "tol must be a finite positive number"),
        ("tol", np.inf, "tol must be a finite positive number"),
        ("tol", True, "tol must be a finite positive number"),
        ("tol", "1e-6", "tol must be a finite positive number"),
        ("lla_tol", 0.0, "lla_tol must be a finite positive number"),
        ("lla_tol", np.nan, "lla_tol must be a finite positive number"),
        ("lla_tol", False, "lla_tol must be a finite positive number"),
        ("lla_tol", "1e-6", "lla_tol must be a finite positive number"),
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
        ("tol", True, "tol must be a finite positive number"),
        ("tol", "1e-6", "tol must be a finite positive number"),
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
        ("tol", True, "tol must be a finite positive number"),
        ("tol", "1e-6", "tol must be a finite positive number"),
        ("lla_tol", np.nan, "lla_tol must be a finite positive number"),
        ("lla_tol", False, "lla_tol must be a finite positive number"),
        ("lla_tol", "1e-6", "lla_tol must be a finite positive number"),
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


def test_group_quantile_three_lla_budget_maps_to_one_update_per_continuation(
    monkeypatch,
):
    from statgpu.solvers import _quantile_group_proximal_irls_lla as solver_mod

    X, y, _ = _data()
    seen = {}

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen["n_steps"] = len(alpha_path)
        seen["max_lla_per_step"] = kwargs["max_lla_per_step"]
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
        max_lla_iters=3,
        lla_tol=1e-6,
    ).fit(X, y)

    assert seen == {"n_steps": 3, "max_lla_per_step": 1}


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


def test_two_stage_public_stopping_replacement_updates_screening_and_refinement(
    monkeypatch,
):
    """Two-stage budgets must derive from current public max_iter/tol values."""
    from types import SimpleNamespace

    X, y, folds = _data()
    calls = []

    def fake_scores(
        self,
        X_arg,
        y_arg,
        alpha_grid,
        cv_device,
        folds_arg,
        **kwargs,
    ):
        calls.append(
            {
                "max_iter": kwargs["max_iter"],
                "tol": kwargs["tol"],
                "strict": kwargs["strict"],
                "n_alphas": len(alpha_grid),
            }
        )
        return np.zeros((len(folds_arg), len(alpha_grid)), dtype=np.float64)

    def fake_refit(self, X_arg, y_arg, best_alpha, sample_weight=None):
        return SimpleNamespace(
            coef_=np.zeros(X_arg.shape[1], dtype=np.float64),
            intercept_=0.0,
        )

    monkeypatch.setattr(PenalizedGLM_CV, "_compute_cv_scores", fake_scores)
    monkeypatch.setattr(PenalizedGLM_CV, "_refit_best", fake_refit)

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
        cv_strategy="two_stage",
        acknowledge_approx=True,
        refine_top_k=1,
        max_iter=20,
        tol=1e-6,
    )
    model.max_iter = 7
    model.tol = 2e-5
    model.fit(X, y)

    assert len(calls) == 2
    assert calls[0]["strict"] is False
    assert calls[0]["max_iter"] == 7
    assert calls[0]["tol"] == pytest.approx(2e-4)
    assert calls[1]["strict"] is True
    assert calls[1]["max_iter"] == 7
    assert calls[1]["tol"] == pytest.approx(2e-5)


def test_quantile_group_cv_installer_reload_is_idempotent_for_fit_wrapper():
    import importlib

    from statgpu.linear_model.penalized import _quantile_group_lla_contract as contract

    before = (
        PenalizedGLM_CV.fit,
        PenalizedGLM_CV._cv_fold_general,
        PenalizedGLM_CV._refit_best,
        PenalizedGLM_CV._compute_cv_scores,
    )
    importlib.reload(contract)
    after = (
        PenalizedGLM_CV.fit,
        PenalizedGLM_CV._cv_fold_general,
        PenalizedGLM_CV._refit_best,
        PenalizedGLM_CV._compute_cv_scores,
    )

    assert after == before


def test_invalid_cv_control_refit_clears_prior_selection_state():
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

    # Simulate a prior successful fit. A rejected refit must never leave these
    # results visible as though the new request had succeeded.
    model._fitted = True
    model.alpha_ = 0.03
    model.alpha_grid_ = np.asarray([0.05, 0.03], dtype=np.float64)
    model.best_score_ = -0.1
    model.cv_results_ = {"mean_score": np.asarray([0.2, 0.1])}
    model.estimator_ = object()
    model.coef_ = np.ones(X.shape[1], dtype=np.float64)
    model.intercept_ = 0.25
    model.cv_strategy_ = "strict"
    model.cv_selected_device_ = "cpu"

    model.max_iter = 0
    with pytest.raises(ValueError, match="max_iter must be a positive integer"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.alpha_ is None
    assert model.alpha_grid_ is None
    assert model.best_score_ is None
    assert model.cv_results_ is None
    assert model.estimator_ is None
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model.cv_strategy_ is None
    assert model.cv_selected_device_ is None
