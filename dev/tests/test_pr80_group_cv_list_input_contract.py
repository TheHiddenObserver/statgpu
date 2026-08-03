"""Group CV validation must cover public list-like designs transactionally."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV


def test_list_design_group_validation_runs_before_cv_work(monkeypatch):
    rng = np.random.default_rng(10003)
    X = rng.normal(size=(40, 3)).tolist()
    y = rng.normal(size=40).tolist()
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


def test_list_design_trailing_group_completion_reaches_final_refit():
    rng = np.random.default_rng(10004)
    X_array = rng.normal(size=(60, 3))
    y_array = 0.2 + X_array @ np.array([0.8, -0.4, 0.6])
    y_array += rng.normal(scale=0.05, size=60)
    common = dict(
        loss="squared_error",
        penalty="group_lasso",
        alpha_grid=[0.18, 0.09],
        cv=2,
        random_state=7,
        device="cpu",
        max_iter=1000,
        tol=1e-8,
    )
    actual_kwargs = {"groups": [[0, 1]]}

    with pytest.warns(UserWarning, match="Auto-adding 1 single-feature"):
        actual = PenalizedGLM_CV(
            penalty_kwargs=actual_kwargs, **common
        ).fit(X_array.tolist(), y_array.tolist())
    expected = PenalizedGLM_CV(
        penalty_kwargs={"groups": [[0, 1], [2]]}, **common
    ).fit(X_array, y_array)

    assert actual._penalty_kwargs is actual_kwargs
    assert actual_kwargs == {"groups": [[0, 1]]}
    assert actual.estimator_._penalty.groups == ((0, 1), (2,))
    np.testing.assert_allclose(
        actual.cv_results_["all_scores"],
        expected.cv_results_["all_scores"],
        rtol=2e-6,
        atol=2e-8,
    )
    assert actual.alpha_ == pytest.approx(expected.alpha_)
    assert actual.estimator_.alpha == pytest.approx(actual.alpha_)
    np.testing.assert_allclose(
        actual.coef_, expected.coef_, rtol=2e-6, atol=2e-7
    )
