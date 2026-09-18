"""Direct lla=False ownership for PR #166 Quantile Group auto routing."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.solvers import _quantile_group_proximal_irls_lla as group_solver


GROUPS = [[0, 1], [2, 3]]


@pytest.mark.parametrize(
    ("penalty", "penalty_kwargs"),
    [
        ("group_scad", {"groups": GROUPS, "a": 3.7}),
        ("group_mcp", {"groups": GROUPS, "gamma": 3.0}),
    ],
)
def test_quantile_group_auto_lla_false_uses_ordinary_fista(
    monkeypatch, penalty, penalty_kwargs
):
    import statgpu.solvers as solvers

    rng = np.random.default_rng(166901)
    X = rng.normal(size=(12, 4))
    y = 0.2 + X @ np.asarray([0.7, -0.4, 0.25, 0.1])
    calls = {"fista": 0}

    def forbidden_group_solver(*args, **kwargs):
        raise AssertionError("lla=False must not enter Group Proximal IRLS-LLA")

    def fake_fista(loss, penalty_obj, X_work, y_work, **kwargs):
        calls["fista"] += 1
        assert getattr(loss, "name", None) == "quantile"
        assert getattr(penalty_obj, "name", None) in {"group_scad", "group_mcp"}
        return np.zeros(int(X_work.shape[1]), dtype=np.float64), 1

    monkeypatch.setattr(
        group_solver,
        "quantile_group_proximal_irls_lla_solver",
        forbidden_group_solver,
    )
    monkeypatch.setattr(solvers, "fista_solver", fake_fista)

    model = PenalizedGeneralizedLinearModel(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty=penalty,
        penalty_kwargs=penalty_kwargs,
        alpha=0.04,
        solver="auto",
        lla=False,
        device="cpu",
        compute_inference=False,
        max_iter=20,
        tol=1e-6,
    ).fit(X, y)

    assert calls["fista"] == 1
    assert model._selected_solver == "fista"
    np.testing.assert_array_equal(model.coef_, np.zeros(X.shape[1]))
    assert model.intercept_ == 0.0
