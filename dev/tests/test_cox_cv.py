"""Tests for CoxPHCV cross-validation behavior."""

import numpy as np

from statgpu.survival import CoxPHCV
from statgpu.survival._cox_cv import _select_coxph_penalty_cv


def _make_survival_data(n_samples=180, n_features=5, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.array([0.6, -0.4, 0.5, -0.2, 0.3], dtype=np.float64)[:n_features]
    hazard = np.exp(X @ beta)
    time = rng.exponential(1.0 / hazard)
    event = rng.binomial(1, 0.7, size=n_samples).astype(np.int32)
    return X.astype(np.float64), time.astype(np.float64), event


def test_coxphcv_supports_entry_and_cluster_cpu():
    """CoxPHCV should fit on CPU with entry/cluster passthrough enabled."""
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
    model.fit(X, time, event, entry=entry, cluster=cluster)

    assert model.penalty_ is not None
    assert np.isfinite(model.penalty_)
    assert model.coef_ is not None
    assert np.all(np.isfinite(model.coef_))
    assert model.cv_results_ is not None
    assert model.cv_results_["pl_path"].shape[0] == model.penalties_.shape[0]


def test_coxphcv_env_toggles_do_not_change_cpu_penalty_selection(monkeypatch):
    """Two-stage/halving env toggles should not affect CPU full-grid selection."""
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
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE_COARSE", "invalid")
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE_WINDOW", "invalid")
    monkeypatch.setenv("STATGPU_COXPHCV_HALVING_TOPK", "invalid")
    monkeypatch.setenv("STATGPU_COXPHCV_HALVING_FAST_ITER", "invalid")
    monkeypatch.setenv("STATGPU_COXPHCV_HALVING_FAST_TOL", "invalid")

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
