"""All public group penalties are estimation-only until inference preserves groups."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


@pytest.fixture
def data():
    rng = np.random.default_rng(10801)
    X = rng.normal(size=(48, 4))
    y = 0.2 + X @ np.array([0.8, -0.4, 0.25, 0.6])
    y += rng.normal(scale=0.08, size=X.shape[0])
    return X, y


def _kwargs(kind):
    result = {"groups": [[0, 3], [1, 2]]}
    if kind == "group_mcp":
        result["gamma"] = 3.0
    elif kind == "group_scad":
        result["a"] = 3.7
    return result


@pytest.mark.parametrize("kind", ["group_lasso", "group_mcp", "group_scad"])
@pytest.mark.parametrize("loss", ["squared_error", "huber"])
@pytest.mark.parametrize("method", ["bootstrap", "debiased", "oracle"])
def test_group_inference_requests_fail_before_solver_or_bootstrap_refit(
    monkeypatch,
    data,
    kind,
    loss,
    method,
):
    X, y = data
    solver_called = False
    bootstrap_called = False

    def forbidden_solver(*args, **kwargs):
        nonlocal solver_called
        solver_called = True
        raise AssertionError("solver work must not start")

    def forbidden_bootstrap(*args, **kwargs):
        nonlocal bootstrap_called
        bootstrap_called = True
        raise AssertionError("bootstrap refits must not start")

    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_fit_loss_backend",
        forbidden_solver,
    )
    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_compute_post_fit_bootstrap_inference",
        forbidden_bootstrap,
    )
    model = PenalizedGeneralizedLinearModel(
        loss=loss,
        loss_kwargs={"delta": 1.0} if loss == "huber" else None,
        penalty=kind,
        penalty_kwargs=_kwargs(kind),
        alpha=0.12,
        solver="auto",
        device="cpu",
        compute_inference=True,
        inference_method=method,
    )

    with pytest.raises(NotImplementedError, match="estimation-only"):
        model.fit(X, y)
    assert solver_called is False
    assert bootstrap_called is False
    assert model.coef_ is None
    assert model._inference_result is None
