"""CPU integration gate for order-invariant public CoxPHCV grids."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv as cox_cv


def _data():
    rng = np.random.default_rng(12121)
    X = rng.normal(size=(60, 2))
    beta = np.array([0.45, -0.3])
    baseline = rng.exponential(scale=5.0, size=X.shape[0])
    time = 0.2 + baseline * np.exp(-0.2 * (X @ beta))
    time += np.arange(X.shape[0], dtype=np.float64) * 1e-7
    event = (np.arange(X.shape[0]) % 3 != 0).astype(np.float64)
    return X, time, event


def _fit(X, time, event, penalties):
    return CoxPHCV(
        penalties=penalties,
        cv=2,
        random_state=19,
        ties="efron",
        max_iter=300,
        tol=1e-8,
        device="cpu",
        compute_inference=False,
    ).fit(X, time, event)


def _by_penalty(model):
    return {
        float(penalty): (float(score), bool(complete))
        for penalty, score, complete in zip(
            model.penalties_,
            model.cv_results_["mean_pl"],
            model.cv_results_["candidate_complete"],
        )
    }


def test_public_coxphcv_is_invariant_to_custom_grid_permutation():
    X, time, event = _data()
    sorted_grid = np.array([0.3, 0.1, 0.03])
    permuted_grid = np.array([0.03, 0.3, 0.1])

    cox_cv._COXPH_CV_CACHE.clear()
    sorted_model = _fit(X, time, event, sorted_grid)
    assert sorted_model.cv_results_["selection_cache_hit"] is False

    cox_cv._COXPH_CV_CACHE.clear()
    permuted_model = _fit(X, time, event, permuted_grid)
    assert permuted_model.cv_results_["selection_cache_hit"] is False

    assert sorted_model.penalty_ == pytest.approx(permuted_model.penalty_)
    np.testing.assert_array_equal(sorted_model.penalties_, sorted_grid)
    np.testing.assert_array_equal(permuted_model.penalties_, permuted_grid)
    np.testing.assert_array_equal(
        sorted_model.cv_results_["penalty_evaluation_order"], sorted_grid
    )
    np.testing.assert_array_equal(
        permuted_model.cv_results_["penalty_evaluation_order"], sorted_grid
    )

    sorted_results = _by_penalty(sorted_model)
    permuted_results = _by_penalty(permuted_model)
    assert sorted_results.keys() == permuted_results.keys()
    for penalty in sorted_results:
        sorted_score, sorted_complete = sorted_results[penalty]
        permuted_score, permuted_complete = permuted_results[penalty]
        assert sorted_complete is permuted_complete
        assert sorted_score == pytest.approx(
            permuted_score, rel=2e-10, abs=2e-10
        )

    np.testing.assert_allclose(
        sorted_model.coef_,
        permuted_model.coef_,
        rtol=2e-9,
        atol=2e-9,
    )
