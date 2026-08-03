"""Capability boundaries for Group MCP/SCAD after the PR #80 LLA fixes."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(9801)
    X = rng.normal(size=(40, 4))
    y = X @ np.array([0.8, -0.4, 0.2, 0.6]) + rng.normal(
        scale=0.1, size=X.shape[0]
    )
    return X, y


def _kwargs(kind):
    result = {"groups": [[0, 3], [1, 2]]}
    if kind == "group_mcp":
        result["gamma"] = 3.0
    else:
        result["a"] = 3.7
    return result


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_default_group_nonconvex_inference_fails_before_solver_work(
    monkeypatch,
    regression_data,
    kind,
):
    X, y = regression_data
    solver_called = False

    def forbidden_solver(*args, **kwargs):
        nonlocal solver_called
        solver_called = True
        raise AssertionError("solver work must not start")

    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_fit_loss_backend",
        forbidden_solver,
    )
    model = PenalizedGeneralizedLinearModel(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=_kwargs(kind),
        alpha=0.18,
        solver="auto",
        compute_inference=True,
        inference_method="debiased",
        device="cpu",
    )

    with pytest.raises(NotImplementedError, match="Inference not supported"):
        model.fit(X, y)
    assert solver_called is False


@pytest.mark.parametrize("kind", ["group_mcp", "group_scad"])
def test_smooth_only_solver_rejects_group_nonconvex_penalty_before_fit(
    monkeypatch,
    regression_data,
    kind,
):
    X, y = regression_data
    solver_called = False

    def forbidden_solver(*args, **kwargs):
        nonlocal solver_called
        solver_called = True
        raise AssertionError("solver work must not start")

    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_fit_loss_backend",
        forbidden_solver,
    )
    model = PenalizedGeneralizedLinearModel(
        loss="huber",
        loss_kwargs={"delta": 1.0},
        penalty=kind,
        penalty_kwargs=_kwargs(kind),
        alpha=0.18,
        solver="newton",
        compute_inference=False,
        device="cpu",
    )

    with pytest.raises(ValueError, match="only supports smooth objectives"):
        model.fit(X, y)
    assert solver_called is False
