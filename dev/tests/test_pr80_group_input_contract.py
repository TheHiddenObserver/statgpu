"""Strict public input and design-width contracts for all group penalties."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import (
    AdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
    GroupMCPPenalty,
    GroupSCADPenalty,
)


_GROUP_CLASSES = [
    pytest.param(
        lambda groups: GroupLassoPenalty(alpha=0.1, groups=groups),
        id="group-lasso",
    ),
    pytest.param(
        lambda groups: GroupMCPPenalty(
            alpha=0.1, gamma=3.0, groups=groups
        ),
        id="group-mcp",
    ),
    pytest.param(
        lambda groups: GroupSCADPenalty(alpha=0.1, a=3.7, groups=groups),
        id="group-scad",
    ),
]


@pytest.mark.parametrize("factory", _GROUP_CLASSES)
@pytest.mark.parametrize(
    "groups,error_type,match",
    [
        (np.array([[0, 0], [1, 1]]), ValueError, "one-dimensional"),
        ([[0, True], [1, 2]], TypeError, "boolean"),
        ([[0, 0.25], [1, 2]], ValueError, "integer-valued"),
        ([[0, np.nan], [1, 2]], ValueError, "finite"),
        ([[0, "3"], [1, 2]], TypeError, "integer-valued numeric"),
        ([[0, -1], [1, 2]], ValueError, "non-negative"),
        ([[0, 1], []], ValueError, "empty groups"),
        ([[0, 1], 2], TypeError, "either a flat"),
        ([[0, 1], [1, 2]], ValueError, "duplicate"),
        ([0, 0, 2, 2], ValueError, "contiguous and start at zero"),
        ([False, False, True], TypeError, "boolean"),
        (["0", "0", "1"], TypeError, "integer-valued numeric"),
    ],
)
def test_group_inputs_reject_lossy_or_ambiguous_values(
    factory,
    groups,
    error_type,
    match,
):
    with pytest.raises(error_type, match=match):
        factory(groups)


@pytest.mark.parametrize(
    "alpha,error_type,match",
    [
        (True, TypeError, "numeric scalar"),
        (-0.1, ValueError, "non-negative"),
        (np.nan, ValueError, "non-negative"),
        (np.inf, ValueError, "non-negative"),
        ("bad", TypeError, "numeric scalar"),
    ],
)
def test_group_lasso_alpha_is_validated_before_numerical_use(
    alpha,
    error_type,
    match,
):
    with pytest.raises(error_type, match=match):
        GroupLassoPenalty(alpha=alpha, groups=[[0, 1], [2, 3]])


def test_exact_integer_valued_float_indices_are_accepted_without_truncation():
    penalty = GroupLassoPenalty(
        alpha=0.1,
        groups=[[3.0, 0.0], [2.0, 1.0]],
    )
    assert penalty.groups == ((0, 3), (1, 2))
    np.testing.assert_array_equal(penalty._flat_indices, np.array([0, 3, 1, 2]))


def _data(seed=10001):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(90, 3))
    y = 0.2 + X @ np.array([0.8, -0.45, 0.6])
    y += rng.normal(scale=0.06, size=X.shape[0])
    return X, y


def _model(kind, groups):
    kwargs = {"groups": groups}
    loss = "squared_error" if kind == "group_lasso" else "huber"
    loss_kwargs = None if kind == "group_lasso" else {"delta": 1.0}
    if kind == "group_mcp":
        kwargs["gamma"] = 3.0
    elif kind == "group_scad":
        kwargs["a"] = 3.7
    return PenalizedGeneralizedLinearModel(
        loss=loss,
        loss_kwargs=loss_kwargs,
        penalty=kind,
        penalty_kwargs=kwargs,
        alpha=0.12,
        solver="auto",
        device="cpu",
        fit_intercept=True,
        compute_inference=False,
        max_iter=600,
        tol=1e-8,
        max_lla_iters=20,
        lla_tol=1e-8,
    )


@pytest.mark.parametrize("kind", ["group_lasso", "group_mcp", "group_scad"])
def test_trailing_uncovered_feature_is_completed_consistently_before_fit(kind):
    X, y = _data()
    with pytest.warns(UserWarning, match="Auto-adding 1 single-feature"):
        incomplete = _model(kind, [[0, 1]]).fit(X, y)
    explicit = _model(kind, [[0, 1], [2]]).fit(X, y)

    assert incomplete._penalty.groups == ((0, 1), (2,))
    np.testing.assert_allclose(
        incomplete.coef_, explicit.coef_, rtol=2e-7, atol=2e-8
    )
    assert incomplete.intercept_ == pytest.approx(
        explicit.intercept_, rel=2e-7, abs=2e-8
    )
    np.testing.assert_allclose(
        incomplete.predict(X), explicit.predict(X), rtol=2e-7, atol=2e-8
    )


def test_adaptive_group_weights_require_complete_design_coverage():
    X, y = _data()
    penalty = AdaptiveGroupLassoPenalty(
        groups=[[0, 1]],
        alpha=0.12,
        weights=[1.0],
    )
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty=penalty,
        alpha=0.12,
        solver="auto",
        device="cpu",
        compute_inference=False,
    )

    with pytest.raises(ValueError, match="adaptive group weights require"):
        model.fit(X, y)


def test_out_of_range_group_index_fails_before_solver_selection(monkeypatch):
    X, y = _data()
    solver_called = False

    def forbidden_solver(*args, **kwargs):
        nonlocal solver_called
        solver_called = True
        raise AssertionError("solver selection must not run")

    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_select_solver",
        forbidden_solver,
    )
    model = _model("group_lasso", [[0, 3], [1, 2]])

    with pytest.raises(ValueError, match="outside the design matrix"):
        model.fit(X, y)
    assert solver_called is False


def test_cv_group_validation_runs_before_alpha_grid_or_candidate_work(monkeypatch):
    X, y = _data()
    work_started = False

    def forbidden_standard(*args, **kwargs):
        nonlocal work_started
        work_started = True
        raise AssertionError("CV work must not start")

    monkeypatch.setattr(PenalizedGLM_CV, "_fit_standard", forbidden_standard)
    cv = PenalizedGLM_CV(
        loss="squared_error",
        penalty="group_lasso",
        penalty_kwargs={"groups": [[0, 3], [1, 2]]},
        alpha_grid=[0.2, 0.1],
        cv=2,
        device="cpu",
    )

    with pytest.raises(ValueError, match="outside the design matrix"):
        cv.fit(X, y)
    assert work_started is False


@pytest.mark.parametrize("kind", ["group_lasso", "group_mcp", "group_scad"])
def test_cv_trailing_group_completion_reaches_scores_selection_and_refit(kind):
    X, y = _data(seed=10002)
    loss = "squared_error" if kind == "group_lasso" else "huber"
    loss_kwargs = None if kind == "group_lasso" else {"delta": 1.0}
    incomplete_kwargs = {"groups": [[0, 1]]}
    explicit_kwargs = {"groups": [[0, 1], [2]]}
    if kind == "group_mcp":
        incomplete_kwargs["gamma"] = explicit_kwargs["gamma"] = 3.0
    elif kind == "group_scad":
        incomplete_kwargs["a"] = explicit_kwargs["a"] = 3.7
    common = dict(
        loss=loss,
        loss_kwargs=loss_kwargs,
        penalty=kind,
        alpha_grid=[0.18, 0.09],
        cv=2,
        random_state=13,
        device="cpu",
        max_iter=500,
        tol=1e-7,
    )

    with pytest.warns(UserWarning, match="Auto-adding 1 single-feature"):
        incomplete = PenalizedGLM_CV(
            penalty_kwargs=incomplete_kwargs, **common
        ).fit(X, y)
    explicit = PenalizedGLM_CV(
        penalty_kwargs=explicit_kwargs, **common
    ).fit(X, y)

    assert incomplete._penalty_kwargs["groups"] == ((0, 1), (2,))
    np.testing.assert_allclose(
        incomplete.cv_results_["all_scores"],
        explicit.cv_results_["all_scores"],
        rtol=3e-6,
        atol=3e-8,
    )
    assert incomplete.alpha_ == pytest.approx(explicit.alpha_)
    assert incomplete.estimator_.alpha == pytest.approx(incomplete.alpha_)
    np.testing.assert_allclose(
        incomplete.coef_, explicit.coef_, rtol=3e-6, atol=3e-7
    )
