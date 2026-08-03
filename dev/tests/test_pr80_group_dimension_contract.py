"""Exact coefficient dimensions for public group penalty numerical APIs."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import get_glm_loss
from statgpu.penalties import (
    AdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
    GroupMCPPenalty,
    GroupSCADPenalty,
)
from statgpu.solvers import fista_solver


_PENALTIES = [
    pytest.param(
        GroupLassoPenalty(alpha=0.1, groups=[[0, 1], [2, 3]]),
        id="group-lasso",
    ),
    pytest.param(
        AdaptiveGroupLassoPenalty(
            alpha=0.1,
            groups=[[0, 1], [2, 3]],
            weights=[0.5, 1.5],
        ),
        id="adaptive-group-lasso",
    ),
    pytest.param(
        GroupMCPPenalty(
            alpha=0.1, gamma=3.0, groups=[[0, 1], [2, 3]]
        ),
        id="group-mcp",
    ),
    pytest.param(
        GroupSCADPenalty(
            alpha=0.1, a=3.7, groups=[[0, 1], [2, 3]]
        ),
        id="group-scad",
    ),
]


@pytest.mark.parametrize("penalty", _PENALTIES)
@pytest.mark.parametrize("operation", ["value", "gradient", "proximal", "lla_weights"])
@pytest.mark.parametrize(
    "coef",
    [
        pytest.param(np.zeros(3), id="too-short"),
        pytest.param(np.zeros(5), id="too-long"),
        pytest.param(np.zeros((4, 1)), id="column-vector"),
        pytest.param(np.zeros((1, 4)), id="row-vector"),
    ],
)
def test_group_penalty_numeric_methods_reject_dimension_mismatch(
    penalty,
    operation,
    coef,
):
    method = getattr(penalty, operation)
    with pytest.raises(ValueError, match="requires a one-dimensional|expected 4"):
        if operation == "proximal":
            method(coef, 0.2, backend="numpy")
        else:
            method(coef)


@pytest.mark.parametrize("penalty", _PENALTIES)
def test_group_penalty_numeric_methods_accept_exact_feature_vector(penalty):
    coef = np.array([0.5, -0.2, 0.3, 0.1])
    assert np.isfinite(penalty.value(coef))
    assert penalty.gradient(coef).shape == coef.shape
    assert penalty.proximal(coef, 0.2, backend="numpy").shape == coef.shape
    assert penalty.lla_weights(coef).shape == coef.shape


def test_direct_fista_solver_rejects_uncovered_trailing_coordinate():
    rng = np.random.default_rng(10201)
    X = rng.normal(size=(40, 5))
    y = rng.normal(size=40)
    penalty = GroupLassoPenalty(
        alpha=0.1,
        groups=[[0, 1], [2, 3]],
    )
    loss = get_glm_loss("squared_error")

    with pytest.raises(
        ValueError,
        match=r"expected 4 feature coefficients(?: from groups)?, got 5",
    ):
        fista_solver(loss, penalty, X, y, max_iter=20, tol=1e-6)
