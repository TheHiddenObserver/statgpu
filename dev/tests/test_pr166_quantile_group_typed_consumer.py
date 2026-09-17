"""Typed-wrapper consumer coverage for the PR #166 Quantile group LLA repair."""

from __future__ import annotations

import numpy as np

from statgpu.linear_model.penalized import PenalizedQuantileRegression


GROUPS = [[0, 1], [2, 3]]


def test_typed_penalized_quantile_group_scad_reaches_group_lla(monkeypatch):
    rng = np.random.default_rng(166311)
    X = rng.normal(size=(18, 4))
    y = 0.2 + X @ np.array([0.7, -0.4, 0.25, 0.1])
    y += rng.laplace(scale=0.15, size=X.shape[0])
    weights = np.linspace(0.6, 1.7, X.shape[0])

    import statgpu.solvers as solvers

    seen = {}

    def fake_lla(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen["loss"] = getattr(loss, "name", None)
        seen["penalty"] = getattr(penalty, "name", None)
        seen["weights"] = np.asarray(kwargs["sample_weight"], dtype=np.float64)
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(solvers, "fista_lla_path", fake_lla)

    model = PenalizedQuantileRegression(
        quantile=0.35,
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha=0.04,
        solver="auto",
        device="cpu",
        max_iter=40,
    ).fit(X, y, sample_weight=weights)

    assert model._selected_solver == "fista"
    assert seen["loss"] == "quantile"
    assert seen["penalty"] == "group_scad"
    np.testing.assert_array_equal(seen["weights"], weights)
