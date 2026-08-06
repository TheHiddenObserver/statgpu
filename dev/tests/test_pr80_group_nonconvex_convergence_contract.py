"""Convergence evidence for Group MCP/SCAD after the FISTA routing fix."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


_GROUPS = [[0, 3], [1, 2]]


def _data():
    rng = np.random.default_rng(9951)
    X = rng.normal(size=(128, 4))
    y = 0.3 + X @ np.array([0.85, -0.45, 0.25, 0.7])
    y += rng.normal(scale=0.08, size=X.shape[0])
    return X, y


def _kwargs(kind):
    kwargs = {"groups": _GROUPS}
    if kind == "group_mcp":
        kwargs["gamma"] = 3.0
    else:
        kwargs["a"] = 3.7
    return kwargs


def _huber_value(y, prediction, delta=1.0):
    residual = np.asarray(y) - np.asarray(prediction)
    absolute = np.abs(residual)
    per_sample = np.where(
        absolute <= delta,
        0.5 * residual**2,
        delta * (absolute - 0.5 * delta),
    )
    return float(np.mean(per_sample))


def _objective(model, X, y):
    prediction = model.predict(X)
    return _huber_value(y, prediction) + float(model._penalty.value(model.coef_))


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_group_nonconvex_huber_fit_updates_and_improves_objective_without_newton_failure(
    kind,
):
    X, y = _data()
    model = PenalizedGeneralizedLinearModel(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=_kwargs(kind),
        alpha=0.16,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_iter=800,
        tol=1e-9,
        max_lla_iters=30,
        lla_tol=1e-8,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)

    failure_messages = [
        str(item.message)
        for item in caught
        if "proximal_newton line search failed" in str(item.message)
    ]
    assert failure_messages == []
    assert model.n_iter_ > 0
    assert np.all(np.isfinite(model.coef_))
    assert np.linalg.norm(model.coef_) > 1e-3

    fitted_objective = _objective(model, X, y)
    zero_objective = _huber_value(y, np.zeros_like(y))
    assert np.isfinite(fitted_objective)
    assert fitted_objective < zero_objective - 1e-3


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_group_nonconvex_huber_solution_is_stable_under_tighter_tolerance(kind):
    X, y = _data()
    common = dict(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=_kwargs(kind),
        alpha=0.16,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_lla_iters=30,
        lla_tol=1e-8,
    )
    ordinary = PenalizedGeneralizedLinearModel(
        max_iter=500,
        tol=1e-7,
        **common,
    ).fit(X, y)
    tight = PenalizedGeneralizedLinearModel(
        max_iter=1200,
        tol=1e-10,
        **common,
    ).fit(X, y)

    np.testing.assert_allclose(
        ordinary.coef_, tight.coef_, rtol=2e-4, atol=2e-5
    )
    assert ordinary.intercept_ == pytest.approx(
        tight.intercept_, rel=2e-4, abs=2e-5
    )
    assert _objective(ordinary, X, y) == pytest.approx(
        _objective(tight, X, y), rel=2e-5, abs=2e-7
    )
