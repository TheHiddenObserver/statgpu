"""Penalty-object CV consumer coverage for PR #166 Quantile Group LLA."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.penalties import GroupSCADPenalty


GROUPS = [[0, 1], [2, 3]]


def _data():
    # Reuse the maintained physical-acceptance fixture so this consumer test
    # compares two genuinely converged CV paths rather than two equally
    # exhausted low-budget paths.
    rng = np.random.default_rng(166401)
    X = rng.normal(size=(64, 4)).astype(np.float64)
    y = 0.25 + X @ np.array([0.85, -0.42, 0.28, 0.16])
    y += rng.laplace(scale=0.16, size=X.shape[0])
    weights = np.linspace(0.45, 1.85, X.shape[0], dtype=np.float64)
    rng.shuffle(weights)
    idx = np.arange(X.shape[0])
    folds = [(idx[:32], idx[32:]), (idx[32:], idx[:32])]
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
        max_iter=400,
        tol=1e-6,
    )


def test_quantile_group_cv_public_penalty_kwargs_replacement_reaches_children(
    monkeypatch,
):
    from statgpu.solvers import _quantile_group_proximal_irls_lla as solver_mod

    X, y, weights, folds = _data()
    seen = []

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        seen.append(
            {
                "a": float(getattr(penalty, "a", np.nan)),
                "groups": [list(group) for group in penalty.groups],
            }
        )
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        solver_mod, "quantile_group_proximal_irls_lla_solver", fake_solver
    )
    cv = _cv(
        "group_scad",
        folds,
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
    )
    replacement_groups = [[0, 2], [1, 3]]
    cv.penalty_kwargs = {"groups": replacement_groups, "a": 4.2}
    cv.fit(X, y, sample_weight=weights)

    assert cv._penalty_kwargs["a"] == pytest.approx(4.2)
    assert cv._penalty_kwargs["groups"] == replacement_groups
    assert seen
    assert all(item["a"] == pytest.approx(4.2) for item in seen)
    assert all(item["groups"] == replacement_groups for item in seen)


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