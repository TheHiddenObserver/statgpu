"""Group penalty objects must evaluate the actual CV alpha grid."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.penalties import (
    GroupLassoPenalty,
    GroupMCPPenalty,
    GroupSCADPenalty,
)


GROUPS = [[0, 3], [1, 2]]
ALPHAS = [0.32, 0.04]


def _data(seed=10801):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(84, 4))
    y = 0.25 + X @ np.array([0.9, -0.5, 0.3, 0.65])
    y += rng.normal(scale=0.08, size=X.shape[0])
    return X, y


def _cases():
    return [
        pytest.param(
            "group_lasso",
            GroupLassoPenalty(alpha=1.0, groups=GROUPS),
            {"groups": GROUPS},
            "squared_error",
            None,
            id="group-lasso",
        ),
        pytest.param(
            "group_mcp",
            GroupMCPPenalty(alpha=1.0, gamma=3.0, groups=GROUPS),
            {"groups": GROUPS, "gamma": 3.0},
            "huber",
            {"delta": 1.0},
            id="group-mcp",
        ),
        pytest.param(
            "group_scad",
            GroupSCADPenalty(alpha=1.0, a=3.7, groups=GROUPS),
            {"groups": GROUPS, "a": 3.7},
            "huber",
            {"delta": 1.0},
            id="group-scad",
        ),
    ]


@pytest.mark.parametrize(
    "name,penalty_object,penalty_kwargs,loss,loss_kwargs",
    _cases(),
)
def test_object_penalty_cv_matches_string_grid_and_selected_refit(
    name,
    penalty_object,
    penalty_kwargs,
    loss,
    loss_kwargs,
):
    X, y = _data()
    common = dict(
        loss=loss,
        loss_kwargs=loss_kwargs,
        alpha_grid=ALPHAS,
        cv=2,
        random_state=43,
        device="cpu",
        max_iter=1000,
        tol=1e-8,
    )
    object_cv = PenalizedGLM_CV(
        penalty=penalty_object,
        **common,
    ).fit(X, y)
    string_cv = PenalizedGLM_CV(
        penalty=name,
        penalty_kwargs=penalty_kwargs,
        **common,
    ).fit(X, y)

    # Distinct grid columns prove the candidate alpha reached the resolved
    # object penalty rather than every candidate silently using alpha=1.
    object_scores = np.asarray(object_cv.cv_results_["all_scores"])
    assert not np.allclose(object_scores[:, 0], object_scores[:, 1])
    np.testing.assert_allclose(
        object_scores,
        string_cv.cv_results_["all_scores"],
        rtol=3e-6,
        atol=3e-8,
    )
    assert object_cv.alpha_ == pytest.approx(string_cv.alpha_)
    np.testing.assert_allclose(
        object_cv.coef_, string_cv.coef_, rtol=3e-6, atol=3e-7
    )
    assert object_cv.estimator_._penalty.alpha == pytest.approx(object_cv.alpha_)

    # The user's constructor object remains untouched and is restored on CV.
    assert object_cv.penalty is penalty_object
    assert penalty_object.alpha == pytest.approx(1.0)
