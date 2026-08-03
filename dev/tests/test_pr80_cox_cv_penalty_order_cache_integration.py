"""Real CoxPHCV cache hits must remain custom-grid order invariant."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv as cox_cv


def _data():
    rng = np.random.default_rng(12141)
    X = rng.normal(size=(54, 2))
    beta = np.array([0.4, -0.25])
    raw = rng.exponential(scale=4.5, size=X.shape[0])
    time = 0.3 + raw * np.exp(-0.2 * (X @ beta))
    time += np.arange(X.shape[0], dtype=np.float64) * 1e-7
    event = (np.arange(X.shape[0]) % 3 != 1).astype(np.float64)
    return X, time, event


def _fit(X, time, event, penalties):
    return CoxPHCV(
        penalties=penalties,
        cv=2,
        random_state=29,
        ties="efron",
        max_iter=300,
        tol=1e-8,
        device="cpu",
        compute_inference=False,
    ).fit(X, time, event)


def _scores(model):
    return {
        float(penalty): float(score)
        for penalty, score in zip(
            model.penalties_, model.cv_results_["mean_pl"]
        )
    }


def test_permuted_custom_grid_hits_canonical_cache_and_restores_public_order():
    X, time, event = _data()
    sorted_grid = np.array([0.25, 0.08, 0.02])
    permuted_grid = np.array([0.02, 0.25, 0.08])

    cox_cv._COXPH_CV_CACHE.clear()
    first = _fit(X, time, event, sorted_grid)
    second = _fit(X, time, event, permuted_grid)

    assert first.cv_results_["selection_cache_hit"] is False
    assert second.cv_results_["selection_cache_hit"] is True
    assert first.penalty_ == pytest.approx(second.penalty_)
    np.testing.assert_array_equal(first.penalties_, sorted_grid)
    np.testing.assert_array_equal(second.penalties_, permuted_grid)
    np.testing.assert_array_equal(
        second.cv_results_["penalty_evaluation_order"], sorted_grid
    )
    assert second.cv_results_["penalty_input_order_preserved"] is True

    first_scores = _scores(first)
    second_scores = _scores(second)
    assert first_scores.keys() == second_scores.keys()
    for penalty in first_scores:
        assert first_scores[penalty] == pytest.approx(
            second_scores[penalty], rel=0.0, abs=0.0
        )
    np.testing.assert_allclose(
        first.coef_, second.coef_, rtol=0.0, atol=0.0
    )
