"""Tests for CoxPHCV cross-validation behavior."""

import numpy as np
import pytest

from statgpu.survival import CoxPHCV
from statgpu.survival._cox_cv import _select_coxph_penalty_cv, _env_int, _env_float


def _make_survival_data(n_samples=180, n_features=5, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.array([0.6, -0.4, 0.5, -0.2, 0.3], dtype=np.float64)[:n_features]
    hazard = np.exp(X @ beta)
    time = rng.exponential(1.0 / hazard)
    event = rng.binomial(1, 0.7, size=n_samples).astype(np.int32)
    return X.astype(np.float64), time.astype(np.float64), event


def test_coxphcv_rejects_nonzero_delayed_entry_penalties_on_cpu():
    """CPU delayed-entry CV must not silently fit a different objective."""
    X, time, event = _make_survival_data(seed=77)
    entry = np.zeros_like(time, dtype=np.float64)
    entry[:60] = np.minimum(time[:60] * 0.25, time[:60] * 0.95)
    cluster = np.repeat(np.arange(30), 6)

    model = CoxPHCV(
        device="cpu",
        cv=3,
        n_penalties=8,
        max_iter=50,
        tol=1e-8,
        compute_inference=False,
        random_state=11,
    )
    with pytest.raises(NotImplementedError, match='nonzero penalties'):
        model.fit(X, time, event, entry=entry, cluster=cluster)


def test_coxphcv_allows_explicit_unpenalized_delayed_entry_cpu():
    X, time, event = _make_survival_data(seed=78)
    entry = np.minimum(time * 0.2, time * 0.95)
    model = CoxPHCV(
        penalties=[0.0], device='cpu', cv=3, max_iter=50, tol=1e-7,
        compute_inference=False, random_state=11,
    ).fit(X, time, event, entry=entry)

    assert model.penalty_ == 0.0
    assert model.coef_ is not None
    assert np.all(np.isfinite(model.coef_))
    assert model.cv_results_ is not None
    assert model.cv_results_["pl_path"].shape[0] == model.penalties_.shape[0]
    assert model.termination_reason_ == model.estimator_.termination_reason_


def test_coxphcv_env_toggles_do_not_change_cpu_penalty_selection(monkeypatch):
    """Two-stage/halving env toggles should not affect CPU full-grid selection."""
    invalid_value = "invalid"
    X, time, event = _make_survival_data(seed=2026)
    penalties = np.geomspace(1.5, 0.01, 12)

    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE", "0")
    monkeypatch.setenv("STATGPU_COXPHCV_SUCCESSIVE_HALVING", "0")
    best_full, details_full = _select_coxph_penalty_cv(
        X,
        time,
        event,
        penalties=penalties,
        cv_folds=4,
        random_state=5,
        device="cpu",
        max_iter=40,
        tol=1e-7,
        return_details=True,
    )

    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_SUCCESSIVE_HALVING", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE_COARSE", invalid_value)
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE_WINDOW", invalid_value)
    monkeypatch.setenv("STATGPU_COXPHCV_HALVING_TOPK", invalid_value)
    monkeypatch.setenv("STATGPU_COXPHCV_HALVING_FAST_ITER", invalid_value)
    monkeypatch.setenv("STATGPU_COXPHCV_HALVING_FAST_TOL", invalid_value)
    assert _env_int("STATGPU_COXPHCV_TWO_STAGE_COARSE", 6, min_value=3, max_value=12) == 6
    assert _env_int("STATGPU_COXPHCV_TWO_STAGE_WINDOW", 2, min_value=1) == 2
    assert _env_int("STATGPU_COXPHCV_HALVING_TOPK", 3, min_value=1, max_value=12) == 3
    assert _env_int("STATGPU_COXPHCV_HALVING_FAST_ITER", 30, min_value=5, max_value=40) == 30
    assert np.isclose(_env_float("STATGPU_COXPHCV_HALVING_FAST_TOL", 1e-6, min_value=1e-7), 1e-6)

    best_tuned, details_tuned = _select_coxph_penalty_cv(
        X,
        time,
        event,
        penalties=penalties,
        cv_folds=4,
        random_state=5,
        device="cpu",
        max_iter=40,
        tol=1e-7,
        return_details=True,
    )

    assert np.isclose(best_full, best_tuned)
    assert np.allclose(details_full["mean_pl"], details_tuned["mean_pl"], equal_nan=True)
