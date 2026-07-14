"""Tests for CoxPHCV cross-validation behavior."""

import sys
import types

import numpy as np
import pytest

from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv as cox_cv_module
from statgpu.survival._cox_cv import (
    _COXPH_CV_CACHE,
    _compute_partial_likelihood,
    _env_float,
    _env_int,
    _select_coxph_penalty_cv,
)


def _make_survival_data(n_samples=180, n_features=5, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.array([0.6, -0.4, 0.5, -0.2, 0.3], dtype=np.float64)[:n_features]
    hazard = np.exp(X @ beta)
    time = rng.exponential(1.0 / hazard)
    event = rng.binomial(1, 0.7, size=n_samples).astype(np.int32)
    return X.astype(np.float64), time.astype(np.float64), event


def _make_counting_process_data(n_subjects=30, n_features=2, seed=910):
    rng = np.random.default_rng(seed)
    subject_X = rng.normal(size=(n_subjects, n_features))
    beta = np.linspace(0.4, -0.2, n_features)
    duration = rng.exponential(scale=np.exp(-(subject_X @ beta))) + 0.2
    X = np.repeat(subject_X, 2, axis=0)
    start = np.tile(np.array([0.0, 1.0]), n_subjects)
    stop = np.empty(2 * n_subjects, dtype=np.float64)
    stop[0::2] = 1.0
    stop[1::2] = 1.0 + duration
    event = np.zeros(2 * n_subjects, dtype=np.int32)
    event[1::2] = 1
    strata = np.repeat(np.where(np.arange(n_subjects) % 2 == 0, "A", "B"), 2)
    subject_id = np.repeat(
        np.asarray([f"subject-{idx}" for idx in range(n_subjects)]), 2
    )
    return X, stop, event, start, strata, subject_id


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
    assert (
        _env_int("STATGPU_COXPHCV_TWO_STAGE_COARSE", 6, min_value=3, max_value=12) == 6
    )
    assert _env_int("STATGPU_COXPHCV_TWO_STAGE_WINDOW", 2, min_value=1) == 2
    assert _env_int("STATGPU_COXPHCV_HALVING_TOPK", 3, min_value=1, max_value=12) == 3
    assert (
        _env_int("STATGPU_COXPHCV_HALVING_FAST_ITER", 30, min_value=5, max_value=40)
        == 30
    )
    assert np.isclose(
        _env_float("STATGPU_COXPHCV_HALVING_FAST_TOL", 1e-6, min_value=1e-7), 1e-6
    )

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
    assert np.allclose(
        details_full["mean_pl"], details_tuned["mean_pl"], equal_nan=True
    )


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (None, -(2.0 * np.log(4.0) + np.log(2.0))),
        (np.array([0.0, 0.0, 1.5, 2.5]), -2.0 * np.log(2.0)),
    ],
)
def test_breslow_heldout_partial_likelihood_ties_zero_coef(entry, expected):
    """Tied failures must share one Breslow denominator, including beta=0."""
    X = np.arange(4, dtype=np.float64).reshape(-1, 1)
    time = np.array([1.0, 1.0, 2.0, 3.0])
    event = np.array([1, 1, 1, 0], dtype=np.int32)

    actual = _compute_partial_likelihood(
        X, time, event, np.zeros(1), entry=entry, ties="breslow"
    )
    actual_none = _compute_partial_likelihood(
        X, time, event, None, entry=entry, ties="breslow"
    )

    assert actual == pytest.approx(expected)
    assert actual_none == pytest.approx(expected)


def test_breslow_heldout_partial_likelihood_ties_nonzero_hand_calculation():
    """Breslow numerator and tied denominator agree with a direct formula."""
    X = np.arange(4, dtype=np.float64).reshape(-1, 1)
    time = np.array([1.0, 1.0, 2.0, 3.0])
    event = np.array([1, 1, 1, 0], dtype=np.int32)
    coef = np.array([0.2])
    eta = X[:, 0] * coef[0]
    expected = (
        eta[0]
        + eta[1]
        - 2.0 * np.log(np.sum(np.exp(eta)))
        + eta[2]
        - np.log(np.sum(np.exp(eta[2:])))
    )

    actual = _compute_partial_likelihood(X, time, event, coef, ties="breslow")

    assert actual == pytest.approx(expected)


def test_heldout_partial_likelihood_uses_open_left_start_boundary():
    """Rows with start equal to a failure time are not yet in its risk set."""
    X = np.zeros((3, 1), dtype=np.float64)
    stop = np.array([1.0, 2.0, 2.0])
    event = np.array([1, 0, 0], dtype=np.int32)
    start = np.array([0.0, 1.0, 0.0])

    actual = _compute_partial_likelihood(
        X, stop, event, np.zeros(1), entry=start, ties="breslow"
    )

    assert actual == pytest.approx(-np.log(2.0))


def test_heldout_partial_likelihood_uses_independent_strata_risk_sets():
    """Stratified held-out likelihood is the sum of per-stratum terms."""
    X = np.zeros((4, 1), dtype=np.float64)
    stop = np.array([1.0, 2.0, 1.0, 2.0])
    event = np.array([1, 0, 1, 0], dtype=np.int32)
    strata = np.array(["A", "A", "B", "B"])

    actual = _compute_partial_likelihood(
        X,
        stop,
        event,
        np.zeros(1),
        strata=strata,
        ties="breslow",
    )

    assert actual == pytest.approx(-2.0 * np.log(2.0))


def test_heldout_partial_likelihood_supports_exact_ties():
    X = np.zeros((3, 1), dtype=np.float64)
    stop = np.array([1.0, 1.0, 2.0])
    event = np.array([1, 1, 0], dtype=np.int32)

    actual = _compute_partial_likelihood(X, stop, event, np.zeros(1), ties="exact")

    assert actual == pytest.approx(-np.log(3.0))


def test_coxphcv_fit_exception_is_not_swallowed(monkeypatch):
    """A candidate fit error must retain its original type and propagate."""

    class CandidateFitError(RuntimeError):
        pass

    class BrokenCoxPH:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, *args, **kwargs):
            raise CandidateFitError("candidate failed")

    X, time, event = _make_survival_data(n_samples=30, seed=81)
    monkeypatch.setattr(cox_cv_module, "CoxPH", BrokenCoxPH)

    with pytest.raises(CandidateFitError, match="candidate failed"):
        _select_coxph_penalty_cv(
            X,
            time,
            event,
            penalties=np.array([1.0, 0.1]),
            cv_folds=3,
            random_state=3,
            device="cpu",
            cache_key="fit-exception-is-not-swallowed",
        )


def test_coxphcv_all_candidates_invalid_raise(monkeypatch):
    """Finite shared-fold evidence is required; no first-penalty fallback."""

    class NonFiniteCoxPH:
        def __init__(self, **kwargs):
            self._converged = False
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            self.coef_ = np.full(X.shape[1], np.nan)
            return self

    X, time, event = _make_survival_data(n_samples=30, seed=82)
    monkeypatch.setattr(cox_cv_module, "CoxPH", NonFiniteCoxPH)

    with pytest.raises(RuntimeError, match="All CoxPHCV penalty candidates failed"):
        _select_coxph_penalty_cv(
            X,
            time,
            event,
            penalties=np.array([1.0, 0.1]),
            cv_folds=3,
            random_state=3,
            device="cpu",
            cache_key="all-candidates-invalid",
        )


def test_coxphcv_candidates_use_same_effective_folds_and_report_status():
    """Candidate means use a shared fold set and expose convergence metadata."""
    X, time, event = _make_survival_data(n_samples=90, seed=83)
    penalties = np.array([1.0, 0.1, 0.01])

    _, details = _select_coxph_penalty_cv(
        X,
        time,
        event,
        penalties=penalties,
        cv_folds=3,
        random_state=7,
        device="cpu",
        max_iter=60,
        tol=1e-7,
        return_details=True,
        cache_key="shared-effective-folds-and-status",
    )

    complete = details["candidate_complete"]
    assert np.any(complete)
    assert np.all(
        details["effective_fold_counts"][complete] == details["effective_n_folds"]
    )
    assert details["converged_path"].shape == details["pl_path"].shape
    assert details["failure_path"].shape == details["pl_path"].shape
    assert len(details["fold_indices"]) == 3
    assert np.array_equal(details["fold"], np.arange(3))
    assert details["effective_device"] == "cpu"


def test_coxphcv_counting_process_cpu_groups_subjects_and_refits():
    """start/strata/subject_id reach every fold and the final NumPy refit."""
    X, stop, event, start, strata, subject_id = _make_counting_process_data()
    model = CoxPHCV(
        penalties=[0.1, 0.01],
        cv=3,
        ties="breslow",
        device="cpu",
        compute_inference=False,
        max_iter=80,
        tol=1e-8,
        random_state=12,
    ).fit(
        X,
        stop,
        event,
        start=start,
        strata=strata,
        subject_id=subject_id,
    )

    assert model.cv_results_["grouped_by_subject"] is True
    assert model.cv_results_["uses_start"] is True
    assert model.cv_results_["uses_strata"] is True
    assert model.cv_results_["uses_subject_id"] is True
    for train_idx, test_idx in model.cv_results_["fold_indices"]:
        assert set(subject_id[train_idx]).isdisjoint(subject_id[test_idx])
    assert np.array_equal(model.estimator_._entry, start)
    assert set(model.estimator_._strata_labels.tolist()) == {"A", "B"}
    assert model.estimator_._subject_id.shape == subject_id.shape
    score = model.score(
        X,
        stop,
        event,
        start=start,
        strata=strata,
        subject_id=subject_id,
    )
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_coxphcv_passes_counting_arrays_to_fold_fits_and_refit(monkeypatch):
    """Guard both candidate-fit and final-refit keyword propagation."""
    fit_records = []

    class RecordingCoxPH:
        def __init__(self, **kwargs):
            self._converged = True
            self._iterations = 1

        def fit(
            self,
            X,
            time,
            event,
            entry=None,
            cluster=None,
            init_coef=None,
            *,
            start=None,
            strata=None,
            subject_id=None,
        ):
            fit_records.append(
                {
                    "n": int(X.shape[0]),
                    "entry": entry,
                    "start": None if start is None else np.asarray(start).copy(),
                    "strata": None if strata is None else np.asarray(strata).copy(),
                    "subject_id": (
                        None if subject_id is None else np.asarray(subject_id).copy()
                    ),
                }
            )
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
            self.hazard_ratios_ = np.ones(X.shape[1], dtype=np.float64)
            return self

    X, stop, event, start, strata, subject_id = _make_counting_process_data(
        n_subjects=8, seed=912
    )
    monkeypatch.setattr(cox_cv_module, "CoxPH", RecordingCoxPH)
    model = CoxPHCV(
        penalties=[0.1],
        cv=2,
        device="cpu",
        compute_inference=False,
        random_state=2,
    ).fit(
        X,
        stop,
        event,
        start=start,
        strata=strata,
        subject_id=subject_id,
    )

    assert len(fit_records) == 3
    for record in fit_records[:-1]:
        assert record["entry"] is None
        assert record["start"].shape == (record["n"],)
        assert record["strata"].shape == (record["n"],)
        assert record["subject_id"].shape == (record["n"],)
    final_record = fit_records[-1]
    assert final_record["n"] == X.shape[0]
    assert np.array_equal(final_record["start"], start)
    assert np.array_equal(final_record["strata"], strata)
    assert np.array_equal(final_record["subject_id"], subject_id)
    assert model.estimator_ is not None


def test_coxphcv_custom_splits_reject_subject_leakage():
    """Explicit folds cannot split time-varying rows from one subject."""
    X, stop, event, start, strata, subject_id = _make_counting_process_data(
        n_subjects=6
    )
    test_idx = np.array([0, 3, 5], dtype=np.int64)
    train_idx = np.setdiff1d(np.arange(X.shape[0]), test_idx)

    with pytest.raises(ValueError, match="subject leakage"):
        _select_coxph_penalty_cv(
            X,
            stop,
            event,
            start=start,
            strata=strata,
            subject_id=subject_id,
            penalties=np.array([0.1]),
            cv_splits=[(train_idx, test_idx)],
            device="cpu",
        )


def test_coxphcv_rejects_entry_and_start_together():
    X, stop, event, start, _, _ = _make_counting_process_data(n_subjects=6)
    with pytest.raises(ValueError, match="only one of entry and start"):
        CoxPHCV(penalties=[0.1], cv=2, device="cpu", compute_inference=False).fit(
            X, stop, event, entry=start, start=start
        )


@pytest.mark.parametrize("device", ["cuda", "torch"])
def test_coxphcv_counting_process_gpu_passthrough(device):
    """GPU counting-process CV keeps its requested backend when available."""
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CuPy CUDA device unavailable")
        except Exception as exc:
            pytest.skip(f"CuPy CUDA unavailable: {exc}")
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device unavailable")

    X, stop, event, start, strata, subject_id = _make_counting_process_data(
        n_subjects=12, seed=911
    )
    model = CoxPHCV(
        penalties=[0.05],
        cv=2,
        device=device,
        compute_inference=False,
        max_iter=60,
        tol=1e-7,
        random_state=3,
    ).fit(
        X,
        stop,
        event,
        start=start,
        strata=strata,
        subject_id=subject_id,
    )

    assert model.effective_device_ == device
    assert model.cv_results_["grouped_by_subject"] is True
    assert np.all(np.isfinite(model.coef_))


def test_coxphcv_excludes_candidate_missing_one_effective_fold(monkeypatch):
    """A high partial mean cannot win by silently dropping a failed fold."""

    class PenaltyEncodedCoxPH:
        def __init__(self, *, penalty, **kwargs):
            self.penalty = penalty
            self._converged = True
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            self.coef_ = np.full(X.shape[1], self.penalty, dtype=np.float64)
            return self

    high_penalty_calls = 0

    def controlled_score(X, time, event, coef, **kwargs):
        nonlocal high_penalty_calls
        if np.isclose(coef[0], 1.0):
            high_penalty_calls += 1
            return np.nan if high_penalty_calls == 1 else 100.0
        return 0.0

    X, time, event = _make_survival_data(n_samples=30, seed=831)
    monkeypatch.setattr(cox_cv_module, "CoxPH", PenaltyEncodedCoxPH)
    monkeypatch.setattr(cox_cv_module, "_compute_partial_likelihood", controlled_score)

    best, details = _select_coxph_penalty_cv(
        X,
        time,
        event,
        penalties=np.array([1.0, 0.0]),
        cv_folds=3,
        random_state=7,
        device="cpu",
        return_details=True,
        cache_key="exclude-incomplete-candidate",
    )

    assert best == pytest.approx(0.0)
    assert np.array_equal(details["candidate_complete"], [False, True])
    assert np.array_equal(details["effective_fold_counts"], [2, 3])
    assert np.isnan(details["mean_pl"][0])
    assert details["mean_pl"][1] == pytest.approx(0.0)


def test_coxphcv_explicit_cuda_never_bridges_to_torch(monkeypatch):
    """The removed bridge env var cannot change an explicit CUDA request."""
    observed_devices = []
    observed_compute_cindex = []
    observed_compute_derivatives = []

    fake_cupy = types.SimpleNamespace(
        float64=np.float64,
        int32=np.int32,
        int64=np.int64,
        asarray=lambda value, dtype=None: np.asarray(value, dtype=dtype),
    )

    class RecordingCoxPH:
        def __init__(self, *, device, compute_cindex, **kwargs):
            observed_devices.append(device)
            observed_compute_cindex.append(compute_cindex)
            self._converged = True
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
            return self

    rng = np.random.default_rng(84)
    X = rng.normal(size=(1500, 40))
    time = np.linspace(0.1, 10.0, X.shape[0])
    event = np.ones(X.shape[0], dtype=np.int32)
    real_objective = cox_cv_module.cox_counting_process_objective

    def recording_objective(*args, **kwargs):
        observed_compute_derivatives.append(kwargs.get("compute_derivatives"))
        return real_objective(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(cox_cv_module, "CoxPH", RecordingCoxPH)
    monkeypatch.setattr(
        cox_cv_module, "cox_counting_process_objective", recording_objective
    )
    monkeypatch.setenv("STATGPU_COXPHCV_CUDA_TORCH_BRIDGE", "1")

    _, details = _select_coxph_penalty_cv(
        X,
        time,
        event,
        penalties=np.array([1.0, 0.1]),
        cv_folds=2,
        random_state=1,
        device="cuda",
        return_details=True,
        cache_key="explicit-cuda-device-purity",
    )

    assert observed_devices
    assert set(observed_devices) == {"cuda"}
    assert observed_compute_cindex
    assert set(observed_compute_cindex) == {False}
    assert observed_compute_derivatives
    assert set(observed_compute_derivatives) == {False}
    assert details["effective_device"] == "cuda"


def test_coxphcv_backend_preparation_error_is_not_swallowed(monkeypatch):
    """Explicit CUDA array preparation errors surface immediately."""

    class BackendPreparationError(RuntimeError):
        pass

    def fail_asarray(value, dtype=None):
        raise BackendPreparationError("cannot prepare CUDA fold")

    fake_cupy = types.SimpleNamespace(
        float64=np.float64,
        int32=np.int32,
        int64=np.int64,
        asarray=fail_asarray,
    )
    X, time, event = _make_survival_data(n_samples=30, seed=85)
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    with pytest.raises(BackendPreparationError, match="cannot prepare CUDA fold"):
        _select_coxph_penalty_cv(
            X,
            time,
            event,
            penalties=np.array([1.0]),
            cv_folds=3,
            random_state=1,
            device="cuda",
            cache_key="backend-preparation-error",
        )


def test_coxphcv_cache_reuses_complete_diagnostics(monkeypatch):
    """A cache hit reuses selection and its fold/convergence diagnostics."""
    fit_calls = 0

    class CountingCoxPH:
        def __init__(self, **kwargs):
            self._converged = True
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
            return self

    X, time, event = _make_survival_data(n_samples=30, seed=86)
    key = "complete-diagnostics-cache"
    _COXPH_CV_CACHE.pop(key, None)
    monkeypatch.setattr(cox_cv_module, "CoxPH", CountingCoxPH)
    kwargs = dict(
        penalties=np.array([1.0, 0.1]),
        cv_folds=3,
        random_state=1,
        device="cpu",
        return_details=True,
        cache_key=key,
    )

    first_penalty, first = _select_coxph_penalty_cv(X, time, event, **kwargs)
    calls_after_first = fit_calls
    second_penalty, second = _select_coxph_penalty_cv(X, time, event, **kwargs)

    assert calls_after_first == 6
    assert fit_calls == calls_after_first
    assert second_penalty == first_penalty
    assert np.array_equal(second["converged_path"], first["converged_path"])
    assert second["effective_device"] == first["effective_device"]


def test_coxphcv_auto_cache_distinguishes_strata_and_subject_id(monkeypatch):
    """Risk-set and grouping arrays are part of the automatic cache identity."""
    fit_calls = 0

    class CountingCoxPH:
        def __init__(self, **kwargs):
            self._converged = True
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
            return self

    X, time, event = _make_survival_data(n_samples=24, seed=861)
    indices = np.arange(X.shape[0])
    splits = [
        (indices[12:], indices[:12]),
        (indices[:12], indices[12:]),
    ]
    strata_a = np.repeat("A", X.shape[0])
    strata_b = np.where(indices % 2 == 0, "A", "B")
    subjects_a = np.asarray([f"row-{idx}" for idx in indices])
    subjects_b = np.asarray([f"alternate-{idx}" for idx in indices])
    _COXPH_CV_CACHE.clear()
    monkeypatch.setattr(cox_cv_module, "CoxPH", CountingCoxPH)
    common = dict(
        penalties=np.array([1.0, 0.1]),
        cv_splits=splits,
        device="cpu",
        return_details=True,
    )

    _select_coxph_penalty_cv(
        X, time, event, strata=strata_a, subject_id=subjects_a, **common
    )
    _select_coxph_penalty_cv(
        X, time, event, strata=strata_b, subject_id=subjects_a, **common
    )
    _select_coxph_penalty_cv(
        X, time, event, strata=strata_b, subject_id=subjects_b, **common
    )

    assert fit_calls == 12


def test_coxphcv_refit_predict_score_and_cleanup_timing(monkeypatch):
    """Refit uses the winner; fit retains caches while public reads clean up."""
    details = {
        "penalty": 0.1,
        "penalties": np.array([1.0, 0.1]),
        "pl_path": np.array([[-3.0, -3.0], [-2.0, -2.0]]),
        "mean_pl": np.array([-3.0, -2.0]),
        "best_pl": -2.0,
        "effective_device": "cpu",
        "fold": np.arange(2),
        "fold_indices": [],
        "fold_metadata": [],
        "converged_path": np.ones((2, 2), dtype=bool),
        "failure_path": np.full((2, 2), None, dtype=object),
        "effective_fold_counts": np.array([2, 2]),
        "effective_n_folds": 2,
    }

    def fake_select(*args, **kwargs):
        return 0.1, details

    class FinalCoxPH:
        def __init__(self, *, penalty, device, **kwargs):
            self.penalty = penalty
            self.device = device

        def fit(self, X, *args, **kwargs):
            self.coef_ = np.array([1.0])
            self.hazard_ratios_ = np.exp(self.coef_)
            return self

        def predict(self, X):
            return np.exp(np.asarray(X)[:, 0])

        def predict_risk_score(self, X):
            return np.asarray(X)[:, 0]

        def score(self, X, time, event, **kwargs):
            return 1.0

    monkeypatch.setattr(cox_cv_module, "_select_coxph_penalty_cv", fake_select)
    monkeypatch.setattr(cox_cv_module, "CoxPH", FinalCoxPH)
    model = CoxPHCV(
        penalties=[1.0, 0.1],
        cv=2,
        device="cpu",
        compute_inference=False,
        gpu_memory_cleanup=True,
    )
    cleanup_calls = {"cuda": 0, "torch": 0}
    monkeypatch.setattr(
        model,
        "_cleanup_cuda_memory",
        lambda: cleanup_calls.__setitem__("cuda", cleanup_calls["cuda"] + 1),
    )
    monkeypatch.setattr(
        model,
        "_cleanup_torch_memory",
        lambda: cleanup_calls.__setitem__("torch", cleanup_calls["torch"] + 1),
    )
    X = np.array([[3.0], [2.0], [1.0]])
    time = np.array([1.0, 2.0, 3.0])
    event = np.ones(3, dtype=np.int32)

    model.fit(X, time, event)

    assert cleanup_calls == {"cuda": 0, "torch": 0}
    assert model.estimator_.penalty == pytest.approx(0.1)
    assert model.estimator_.device == "cpu"
    assert model.effective_device_ == "cpu"
    assert np.allclose(model.predict(X), np.exp(X[:, 0]))
    assert np.allclose(model.predict_risk_score(X), X[:, 0])
    assert model.score(X, time, event) == pytest.approx(1.0)
    assert cleanup_calls == {"cuda": 3, "torch": 3}


def test_coxphcv_rejects_fractional_events_before_integer_cast():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([0.0, 0.5, 1.0, 0.0])
    with pytest.raises(ValueError, match="event"):
        _compute_partial_likelihood(X, time, event, np.zeros(1))
    with pytest.raises(ValueError, match="event"):
        CoxPHCV(penalties=[0.0], cv=2, device="cpu", compute_inference=False).fit(
            X, time, event
        )

    X_valid, time_valid, event_valid = _make_survival_data(
        n_samples=48, n_features=2, seed=872
    )
    model = CoxPHCV(
        penalties=[0.0],
        cv=2,
        device="cpu",
        compute_inference=False,
        max_iter=60,
        random_state=0,
    ).fit(X_valid, time_valid, event_valid)
    invalid_event = event_valid.astype(np.float64)
    invalid_event[0] = 0.5
    with pytest.raises(ValueError, match="event must contain only 0/1"):
        model.score(X_valid, time_valid, invalid_event)


def test_cox_estimators_are_cloneable_and_coxphcv_grid_search_smoke():
    sklearn_base = pytest.importorskip("sklearn.base")
    sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
    from statgpu.survival import CoxPH

    cox_model = CoxPH(
        device="cpu", compute_inference=False, compute_cindex=False, max_iter=60
    )
    assert isinstance(sklearn_base.clone(cox_model), CoxPH)
    cv_model = CoxPHCV(
        penalties=[0.1],
        cv=2,
        device="cpu",
        compute_inference=False,
        max_iter=60,
    )
    assert isinstance(sklearn_base.clone(cv_model), CoxPHCV)
    X, time, event = _make_survival_data(n_samples=48, n_features=2, seed=871)
    y = np.column_stack([time, event])
    cox_search = sklearn_model_selection.GridSearchCV(
        cox_model,
        {"penalty": [0.0, 0.1]},
        cv=2,
        error_score="raise",
    ).fit(X, y)
    assert cox_search.best_estimator_.coef_ is not None
    search = sklearn_model_selection.GridSearchCV(
        cv_model,
        {"tol": [1e-7]},
        cv=2,
        error_score="raise",
    ).fit(X, y)
    assert search.best_estimator_.estimator_ is not None


def test_coxphcv_failed_refit_clears_previous_model_state():
    X, time, event = _make_survival_data(n_samples=60, n_features=2, seed=872)
    model = CoxPHCV(
        penalties=[0.1],
        cv=2,
        device="cpu",
        compute_inference=False,
        max_iter=60,
    ).fit(X, time, event)
    assert model.estimator_ is not None
    invalid_event = event.astype(np.float64)
    invalid_event[0] = 0.5
    with pytest.raises(ValueError, match="event"):
        model.fit(X, time, invalid_event)
    assert model.estimator_ is None
    assert model.coef_ is None
    assert model.cv_results_ is None
    assert model._fitted is False
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(X[:2])


@pytest.mark.parametrize(
    "penalties",
    [[], [np.nan], [-0.1]],
)
def test_coxphcv_rejects_invalid_explicit_penalty_grids(penalties):
    X, time, event = _make_survival_data(n_samples=30, n_features=2, seed=873)
    with pytest.raises(ValueError, match="penalties"):
        _select_coxph_penalty_cv(
            X,
            time,
            event,
            penalties=penalties,
            cv_folds=2,
            device="cpu",
        )


def test_coxphcv_rejects_overlapping_custom_train_test_indices():
    X, time, event = _make_survival_data(n_samples=30, n_features=2, seed=874)
    with pytest.raises(ValueError, match="disjoint"):
        _select_coxph_penalty_cv(
            X,
            time,
            event,
            penalties=[0.1],
            cv_splits=[(np.arange(20), np.arange(10, 30))],
            device="cpu",
        )


@pytest.mark.parametrize("side", ["train", "test"])
@pytest.mark.parametrize("invalid_kind", ["fractional", "boolean"])
def test_coxphcv_validates_custom_indices_before_integer_conversion(
    side, invalid_kind
):
    X, time, event = _make_survival_data(n_samples=30, n_features=2, seed=876)
    train_idx = np.arange(20)
    test_idx = np.arange(20, 30)
    if invalid_kind == "fractional":
        invalid_idx = (
            np.array([0.0, 1.5])
            if side == "train"
            else np.array([20.0, 21.5])
        )
    else:
        invalid_idx = [0, True] if side == "train" else [20, True]
    if side == "train":
        train_idx = invalid_idx
    else:
        test_idx = invalid_idx

    with pytest.raises(ValueError, match=rf"{side} indices must contain integers"):
        _select_coxph_penalty_cv(
            X,
            time,
            event,
            penalties=[0.1],
            cv_splits=[(train_idx, test_idx)],
            device="cpu",
        )


def test_coxphcv_nonconverged_high_score_candidate_is_ineligible(monkeypatch):
    class ConvergenceEncodedCoxPH:
        def __init__(self, *, penalty, **kwargs):
            self.penalty = float(penalty)
            self._converged = not np.isclose(self.penalty, 1.0)
            self._iterations = 2

        def fit(self, X, *args, **kwargs):
            self.coef_ = np.array([self.penalty, 0.0])
            return self

    def score_from_penalty(X, time, event, coef, **kwargs):
        return 100.0 if np.isclose(coef[0], 1.0) else 0.0

    X, time, event = _make_survival_data(n_samples=36, n_features=2, seed=875)
    monkeypatch.setattr(cox_cv_module, "CoxPH", ConvergenceEncodedCoxPH)
    monkeypatch.setattr(
        cox_cv_module, "_compute_partial_likelihood", score_from_penalty
    )
    best, details = _select_coxph_penalty_cv(
        X,
        time,
        event,
        penalties=[1.0, 0.0],
        cv_folds=3,
        random_state=2,
        device="cpu",
        return_details=True,
        cache_key="nonconverged-high-score-is-ineligible",
    )
    assert best == pytest.approx(0.0)
    assert np.array_equal(details["candidate_complete"], [False, True])
    assert np.all(details["failure_path"][0] == "did_not_converge")


def test_coxphcv_auto_cache_hashes_middle_rows_not_only_edges(monkeypatch):
    fit_calls = 0

    class CountingCoxPH:
        def __init__(self, **kwargs):
            self._converged = True
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            self.coef_ = np.zeros(X.shape[1])
            return self

    X, time, event = _make_survival_data(n_samples=40, n_features=2, seed=876)
    indices = np.arange(len(time))
    splits = [
        (indices[20:], indices[:20]),
        (indices[:20], indices[20:]),
    ]
    monkeypatch.setattr(cox_cv_module, "CoxPH", CountingCoxPH)
    _COXPH_CV_CACHE.clear()
    kwargs = dict(
        penalties=[0.1],
        cv_splits=splits,
        device="cpu",
        return_details=True,
    )
    _select_coxph_penalty_cv(X, time, event, **kwargs)
    first_calls = fit_calls
    X_changed = X.copy()
    X_changed[len(X) // 2, 0] += 1.0
    _select_coxph_penalty_cv(X_changed, time, event, **kwargs)
    assert first_calls == 2
    assert fit_calls == 4


def test_coxphcv_accepts_torch_cpu_penalty_array():
    torch = pytest.importorskip("torch")
    X, time, event = _make_survival_data(n_samples=42, n_features=2, seed=877)
    model = CoxPHCV(
        penalties=torch.tensor([0.1], dtype=torch.float64),
        cv=2,
        device="cpu",
        compute_inference=False,
        max_iter=60,
    ).fit(X, time, event)
    assert np.array_equal(model.penalties_, np.array([0.1]))
    assert model.cv_results_["scoring_device"] == "cpu"
    assert model.cv_results_["orchestration_device"] == "cpu"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_penalties": 0}, "n_penalties"),
        ({"n_penalties": 2.5}, "n_penalties"),
        ({"penalty_min_ratio": 0.0}, "penalty_min_ratio"),
        ({"penalty_min_ratio": np.nan}, "penalty_min_ratio"),
        ({"penalty_min_ratio": 1.1}, "less than or equal to 1"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": np.inf}, "tol"),
    ],
)
def test_coxphcv_rejects_invalid_grid_and_solver_controls(kwargs, match):
    X, time, event = _make_survival_data(n_samples=36, n_features=2, seed=878)
    with pytest.raises(ValueError, match=match):
        _select_coxph_penalty_cv(
            X,
            time,
            event,
            cv_folds=3,
            device="cpu",
            cache_key=f"invalid-controls-{repr(kwargs)}",
            **kwargs,
        )


def test_coxphcv_direct_auto_device_resolves_before_backend_dispatch(monkeypatch):
    X, time, event = _make_survival_data(n_samples=42, n_features=2, seed=879)
    monkeypatch.setattr(cox_cv_module, "get_device", lambda: cox_cv_module.Device.CPU)
    _, details = _select_coxph_penalty_cv(
        X,
        time,
        event,
        penalties=[0.1],
        cv_folds=2,
        device="auto",
        max_iter=60,
        return_details=True,
        cache_key="direct-auto-resolves-to-cpu",
    )
    assert details["effective_device"] == "cpu"
    assert details["scoring_device"] == "cpu"


def test_coxphcv_cache_results_are_isolated_from_caller_mutation():
    X, time, event = _make_survival_data(n_samples=42, n_features=2, seed=880)
    _COXPH_CV_CACHE.clear()
    kwargs = dict(
        penalties=[0.1],
        cv_folds=2,
        random_state=7,
        device="cpu",
        max_iter=60,
        return_details=True,
        cache_key="cache-mutation-isolation",
    )
    _, first = _select_coxph_penalty_cv(X, time, event, **kwargs)
    expected = first["pl_path"].copy()
    first["pl_path"][:] = 12345.0
    first["fold_metadata"][0]["n_train"] = -1

    _, second = _select_coxph_penalty_cv(X, time, event, **kwargs)
    assert np.array_equal(second["pl_path"], expected)
    assert second["fold_metadata"][0]["n_train"] >= 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_penalties": 2.5}, "n_penalties"),
        ({"cv": 2.5}, "cv"),
        ({"max_iter": 2.5}, "max_iter"),
        ({"penalty_min_ratio": np.nan}, "penalty_min_ratio"),
    ],
)
def test_coxphcv_public_fit_does_not_coerce_invalid_controls(kwargs, match):
    X, time, event = _make_survival_data(n_samples=36, n_features=2, seed=881)
    model_kwargs = {
        "device": "cpu",
        "compute_inference": False,
        "n_penalties": 3,
        "cv": 2,
        "max_iter": 40,
    }
    model_kwargs.update(kwargs)
    model = CoxPHCV(**model_kwargs)
    with pytest.raises(ValueError, match=match):
        model.fit(X, time, event)
    assert model.coef_ is None
    assert model.cv_results_ is None


def test_coxphcv_exact_robust_covariance_rejected_before_cv(monkeypatch):
    X, time, event = _make_survival_data(n_samples=40, n_features=2, seed=882)

    def fail_if_selected(*args, **kwargs):
        pytest.fail("penalty selection ran before validating exact robust inference")

    monkeypatch.setattr(cox_cv_module, "_select_coxph_penalty_cv", fail_if_selected)
    model = CoxPHCV(
        ties="exact",
        cov_type="hc0",
        compute_inference=True,
        penalties=[0.1],
        cv=2,
        device="cpu",
    )
    with pytest.raises(NotImplementedError, match="robust covariance"):
        model.fit(X, time, event)
    assert model.coef_ is None
    assert model.cv_results_ is None


def test_coxphcv_exact_allows_irrelevant_cov_type_without_inference():
    X, time, event = _make_survival_data(n_samples=50, n_features=2, seed=883)
    model = CoxPHCV(
        ties="exact",
        cov_type="hc0",
        compute_inference=False,
        penalties=[0.1],
        cv=2,
        max_iter=80,
        device="cpu",
        random_state=4,
    ).fit(X, time, event)
    assert model._fitted
    assert model.estimator_._bse is None
