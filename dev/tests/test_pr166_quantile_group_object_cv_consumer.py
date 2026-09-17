"""Penalty-object CV consumer coverage for PR #166 Quantile Group LLA."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.penalties import GroupSCADPenalty


GROUPS = [[0, 1], [2, 3]]


def _data():
    rng = np.random.default_rng(166701)
    X = rng.normal(size=(22, 4)).astype(np.float64)
    y = 0.25 + X @ np.array([0.75, -0.4, 0.28, 0.18])
    y += rng.laplace(scale=0.16, size=X.shape[0])
    weights = np.linspace(0.45, 1.75, X.shape[0], dtype=np.float64)
    rng.shuffle(weights)
    idx = np.arange(X.shape[0])
    folds = [(idx[11:], idx[:11]), (idx[:11], idx[11:])]
    return X, y, weights, folds


def _cv(penalty, folds, penalty_kwargs=None):
    return PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": 0.35},
        penalty=penalty,
        penalty_kwargs=penalty_kwargs,
        alpha_grid=np.asarray([0.05, 0.03], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        random_state=166,
        solver="auto",
        device="cpu",
        max_iter=80,
        tol=1e-5,
    )


def test_quantile_group_scad_penalty_object_matches_string_cv_and_refit():
    X, y, weights, folds = _data()
    penalty_object = GroupSCADPenalty(alpha=0.9, a=3.7, groups=GROUPS)

    object_cv = _cv(penalty_object, folds).fit(
        X, y, sample_weight=weights
    )
    string_cv = _cv(
        "group_scad",
        folds,
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
    ).fit(X, y, sample_weight=weights)

    np.testing.assert_allclose(
        object_cv.cv_results_["all_scores"],
        string_cv.cv_results_["all_scores"],
        rtol=3e-7,
        atol=3e-9,
    )
    assert object_cv.alpha_ == pytest.approx(string_cv.alpha_)
    np.testing.assert_allclose(
        object_cv.coef_, string_cv.coef_, rtol=3e-7, atol=3e-9
    )
    assert object_cv.intercept_ == pytest.approx(
        string_cv.intercept_, rel=3e-7, abs=3e-9
    )

    assert penalty_object.alpha == pytest.approx(0.9)
    assert object_cv.penalty is penalty_object
    assert object_cv.estimator_.penalty is not penalty_object
    assert object_cv.estimator_.penalty.alpha == pytest.approx(object_cv.alpha_)
    assert object_cv.estimator_._selected_solver == "group_proximal_irls_lla"
