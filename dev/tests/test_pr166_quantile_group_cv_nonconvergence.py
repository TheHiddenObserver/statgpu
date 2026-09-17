"""CV convergence ownership for PR #166 Quantile Group Proximal IRLS-LLA."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.solvers import _quantile_group_proximal_irls_lla as group_solver


GROUPS = [[0, 1], [2, 3]]
Q = 0.35


def test_nonconverged_quantile_group_candidate_is_nan_and_cannot_be_selected(
    monkeypatch,
):
    """CV candidates use strict convergence while the selected refit does not."""
    rng = np.random.default_rng(166901)
    X = rng.normal(size=(16, 4)).astype(np.float64)
    y = (0.2 + X @ np.array([0.7, -0.35, 0.2, 0.1])).astype(np.float64)
    weights = np.linspace(0.5, 1.5, X.shape[0], dtype=np.float64)
    idx = np.arange(X.shape[0])
    folds = [(idx[8:], idx[:8]), (idx[:8], idx[8:])]
    seen = []

    def fake_solver(loss, penalty, X_fit, y_fit, alpha_path, **kwargs):
        alpha = float(penalty.alpha)
        strict = bool(kwargs.get("fail_on_target_nonconvergence"))
        seen.append((strict, alpha, int(X_fit.shape[0])))
        if strict and alpha == pytest.approx(0.05):
            raise FloatingPointError("sentinel target nonconvergence")
        return np.zeros(X_fit.shape[1], dtype=np.float64), 0.0, 1

    monkeypatch.setattr(
        group_solver, "quantile_group_proximal_irls_lla_solver", fake_solver
    )

    cv = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": Q},
        penalty="group_scad",
        penalty_kwargs={"groups": GROUPS, "a": 3.7},
        alpha_grid=np.asarray([0.05, 0.03], dtype=np.float64),
        cv=2,
        cv_splits=folds,
        random_state=166,
        solver="auto",
        device="cpu",
        max_iter=60,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    scores = np.asarray(cv.cv_results_["all_scores"], dtype=np.float64)
    assert scores.shape == (2, 2)
    assert np.all(np.isnan(scores[:, 0]))
    assert np.all(np.isfinite(scores[:, 1]))
    assert cv.alpha_ == pytest.approx(0.03)
    assert cv.estimator_._selected_solver == "group_proximal_irls_lla"

    strict_calls = [item for item in seen if item[0]]
    refit_calls = [item for item in seen if not item[0]]
    assert len(strict_calls) == 4
    assert {round(item[1], 8) for item in strict_calls} == {0.05, 0.03}
    assert len(refit_calls) == 1
    assert refit_calls[0][1] == pytest.approx(0.03)
    assert refit_calls[0][2] == X.shape[0]
