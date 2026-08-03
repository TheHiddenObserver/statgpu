"""Public estimator/CV capability for AdaptiveGroupLassoPenalty."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.penalties import (
    AdaptiveGroupLassoPenalty,
    GroupLassoPenalty,
)


GROUPS = [[0, 3], [1, 2]]


def _data(seed=11101):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(88, 4))
    y = 0.3 + X @ np.array([0.8, -0.5, 0.25, 0.65])
    y += rng.normal(scale=0.07, size=X.shape[0])
    return X, y


def _model(penalty, *, solver="auto", compute_inference=False):
    return PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty=penalty,
        alpha=0.09,
        solver=solver,
        device="cpu",
        fit_intercept=True,
        compute_inference=compute_inference,
        inference_method="bootstrap",
        max_iter=3000,
        tol=1e-9,
    )


def test_uniform_adaptive_group_lasso_matches_group_lasso_objective():
    X, y = _data()
    adaptive_parameter = AdaptiveGroupLassoPenalty(
        groups=GROUPS,
        alpha=0.09,
        weights=[1.0, 1.0],
    )
    group_parameter = GroupLassoPenalty(alpha=0.09, groups=GROUPS)

    adaptive = _model(adaptive_parameter).fit(X, y)
    group = _model(group_parameter).fit(X, y)

    assert adaptive._selected_solver == "fista"
    assert adaptive._penalty is not adaptive_parameter
    assert adaptive._penalty._group_weights == (1.0, 1.0)
    np.testing.assert_allclose(
        adaptive.coef_, group.coef_, rtol=1e-5, atol=2e-6
    )
    assert adaptive.intercept_ == pytest.approx(
        group.intercept_, rel=1e-5, abs=2e-6
    )
    np.testing.assert_allclose(
        adaptive.predict(X), group.predict(X), rtol=1e-5, atol=3e-6
    )
    assert adaptive._penalty.value(adaptive.coef_) == pytest.approx(
        group._penalty.value(group.coef_), rel=1e-5, abs=2e-6
    )


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_smooth_solver_rejects_adaptive_group_lasso_before_solver_work(
    monkeypatch,
    solver,
):
    X, y = _data(seed=11102)
    work_started = False

    def forbidden(*args, **kwargs):
        nonlocal work_started
        work_started = True
        raise AssertionError("numerical solver work must not start")

    monkeypatch.setattr(
        PenalizedGeneralizedLinearModel,
        "_fit_loss_backend",
        forbidden,
    )
    penalty = AdaptiveGroupLassoPenalty(
        groups=GROUPS,
        alpha=0.09,
        weights=[0.8, 1.3],
    )
    with pytest.raises(ValueError, match="only supports smooth objectives"):
        _model(penalty, solver=solver).fit(X, y)
    assert work_started is False


def test_adaptive_group_lasso_bootstrap_inference_is_rejected_before_fit():
    X, y = _data(seed=11103)
    penalty = AdaptiveGroupLassoPenalty(
        groups=GROUPS,
        alpha=0.09,
        weights=[0.8, 1.3],
    )
    with pytest.raises(
        NotImplementedError,
        match="Adaptive Group Lasso.*estimation-only",
    ):
        _model(penalty, compute_inference=True).fit(X, y)


def test_adaptive_group_object_cv_uses_grid_alpha_not_template_alpha():
    X, y = _data(seed=11104)
    alphas = [0.3, 0.04]
    first_parameter = AdaptiveGroupLassoPenalty(
        groups=GROUPS,
        alpha=1.0,
        weights=[0.7, 1.4],
    )
    second_parameter = AdaptiveGroupLassoPenalty(
        groups=GROUPS,
        alpha=2.0,
        weights=[0.7, 1.4],
    )
    common = dict(
        loss="squared_error",
        alpha_grid=alphas,
        cv=2,
        random_state=53,
        device="cpu",
        max_iter=1500,
        tol=1e-8,
    )
    first = PenalizedGLM_CV(penalty=first_parameter, **common).fit(X, y)
    second = PenalizedGLM_CV(penalty=second_parameter, **common).fit(X, y)

    first_scores = np.asarray(first.cv_results_["all_scores"])
    assert not np.allclose(first_scores[:, 0], first_scores[:, 1])
    np.testing.assert_allclose(
        first_scores,
        second.cv_results_["all_scores"],
        rtol=3e-6,
        atol=3e-8,
    )
    assert first.alpha_ == pytest.approx(second.alpha_)
    np.testing.assert_allclose(first.coef_, second.coef_, rtol=3e-6, atol=3e-7)
    assert first.estimator_._penalty.alpha == pytest.approx(first.alpha_)
    assert first.estimator_.penalty.alpha == pytest.approx(first.alpha_)
    assert first.estimator_.penalty._group_weights == (0.7, 1.4)
    assert not hasattr(
        first.estimator_.penalty, "_statgpu_cv_alpha_from_estimator"
    )

    assert first.penalty is first_parameter
    assert first_parameter.alpha == pytest.approx(1.0)
    assert second.penalty is second_parameter
    assert second_parameter.alpha == pytest.approx(2.0)
