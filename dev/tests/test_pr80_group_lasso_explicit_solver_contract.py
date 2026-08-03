"""Explicit Group Lasso solver choices must not be silently rewritten."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


_GROUPS = [[0, 3], [1, 2]]


def _data():
    rng = np.random.default_rng(10701)
    X = rng.normal(size=(100, 4))
    y = 0.2 + X @ np.array([0.8, -0.45, 0.25, 0.65])
    y += rng.normal(scale=0.08, size=X.shape[0])
    return X, y


def _objective(model, X, y):
    X_work = np.column_stack([X, np.ones(X.shape[0])])
    params = np.append(np.asarray(model.coef_), float(model.intercept_))
    return float(model._loss.value(X_work, y, params)) + float(
        model._penalty.value(np.asarray(model.coef_))
    )


@pytest.mark.parametrize("solver", ["fista", "fista_bb", "admm"])
def test_explicit_group_lasso_solver_is_preserved_and_improves_objective(solver):
    X, y = _data()
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": _GROUPS},
        alpha=0.08,
        solver=solver,
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_iter=3000,
        tol=1e-8,
    ).fit(X, y)

    assert model._selected_solver == solver
    assert model._penalty.name == "group_lasso"
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model.intercept_)
    fitted = _objective(model, X, y)
    zero = 0.5 * float(np.mean(y**2))
    assert fitted < zero - 1e-3


def test_auto_group_lasso_still_resolves_to_fista():
    X, y = _data()
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": _GROUPS},
        alpha=0.08,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_iter=3000,
        tol=1e-8,
    ).fit(X, y)

    assert model._selected_solver == "fista"
