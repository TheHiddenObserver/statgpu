"""Regression contracts for safe CoxPHCV staged-screening fallback."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import threading
import time
import warnings

import numpy as np
import pytest

from statgpu.survival import CoxPHCV
from statgpu.survival import _cox_cv as cox_cv
from statgpu.survival import _cox_cv_staged_safety_contract as staged


def _sample():
    rng = np.random.default_rng(14001)
    X = rng.normal(size=(30, 2))
    time_values = np.linspace(1.0, 30.0, 30)
    event = np.tile(np.array([1.0, 0.0, 1.0]), 10)
    return X, time_values, event


def test_staged_request_is_explicit_exhaustive_fallback(monkeypatch):
    observed = []

    def fake_selector(*args, **kwargs):
        observed.append(
            (
                cox_cv._env_flag("STATGPU_COXPHCV_TWO_STAGE", False),
                cox_cv._env_flag(
                    "STATGPU_COXPHCV_SUCCESSIVE_HALVING", False
                ),
                bool(kwargs.get("return_details", False)),
            )
        )
        details = {
            "penalty": 1.0,
            "penalties": np.array([1.0, 0.5, 0.1]),
            "mean_pl": np.array([3.0, 2.0, 1.0]),
        }
        return 1.0, details

    monkeypatch.setattr(staged, "_ORIGINAL_SELECT_COXPH_PENALTY_CV", fake_selector)
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_SUCCESSIVE_HALVING", "1")

    with pytest.warns(RuntimeWarning, match="exhaustive full-precision"):
        best, details = cox_cv._select_coxph_penalty_cv(
            np.zeros((4, 1)),
            np.arange(1.0, 5.0),
            np.array([1.0, 0.0, 1.0, 0.0]),
            penalties=[1.0, 0.5, 0.1],
            return_details=True,
        )

    assert best == pytest.approx(1.0)
    assert observed == [(False, False, True)]
    assert details["two_stage_requested"] is True
    assert details["two_stage_enabled"] is False
    assert details["successive_halving_requested"] is True
    assert details["successive_halving_enabled"] is False
    assert details["staged_execution_mode"] == "exhaustive_safety_fallback"
    assert details["staged_safety_strategy"] == "single_pass_exhaustive"
    np.testing.assert_array_equal(
        details["fast_pass_candidate_mask"], np.zeros(3, dtype=bool)
    )
    np.testing.assert_array_equal(
        details["full_precision_candidate_mask"], np.ones(3, dtype=bool)
    )
    np.testing.assert_array_equal(
        details["screened_out_candidate_mask"], np.zeros(3, dtype=bool)
    )


def test_explicit_cupy_request_does_not_patch_staged_budget_readers(monkeypatch):
    """CuPy must not retain the old full-grid staged machinery."""
    original_env_int = cox_cv._env_int
    original_env_float = cox_cv._env_float
    calls = []

    def fake_selector(*args, **kwargs):
        calls.append(kwargs.get("device"))
        assert cox_cv._env_flag("STATGPU_COXPHCV_TWO_STAGE", False) is False
        assert (
            cox_cv._env_flag("STATGPU_COXPHCV_SUCCESSIVE_HALVING", False)
            is False
        )
        assert cox_cv._env_int is original_env_int
        assert cox_cv._env_float is original_env_float
        return 0.5, {
            "penalty": 0.5,
            "penalties": np.array([1.0, 0.5]),
            "mean_pl": np.array([1.0, 2.0]),
        }

    monkeypatch.setattr(staged, "_ORIGINAL_SELECT_COXPH_PENALTY_CV", fake_selector)
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_SUCCESSIVE_HALVING", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_HALVING_TOPK", "1")

    with pytest.warns(RuntimeWarning):
        best, details = cox_cv._select_coxph_penalty_cv(
            np.zeros((4, 1)),
            np.arange(1.0, 5.0),
            np.array([1.0, 0.0, 1.0, 0.0]),
            penalties=[1.0, 0.5],
            device="cuda",
            return_details=True,
        )

    assert best == pytest.approx(0.5)
    assert calls == ["cuda"]
    assert details["staged_safety_strategy"] == "single_pass_exhaustive"


def test_staged_scalar_and_detailed_calls_share_selected_penalty(monkeypatch):
    def fake_selector(*args, **kwargs):
        details = {
            "penalty": 0.5,
            "penalties": np.array([1.0, 0.5]),
            "mean_pl": np.array([1.0, 2.0]),
        }
        return 0.5, details

    monkeypatch.setattr(staged, "_ORIGINAL_SELECT_COXPH_PENALTY_CV", fake_selector)
    monkeypatch.setenv("STATGPU_COXPHCV_SUCCESSIVE_HALVING", "1")

    with pytest.warns(RuntimeWarning):
        scalar = cox_cv._select_coxph_penalty_cv(
            np.zeros((4, 1)),
            np.arange(1.0, 5.0),
            np.array([1.0, 0.0, 1.0, 0.0]),
            penalties=[1.0, 0.5],
            return_details=False,
        )
    with pytest.warns(RuntimeWarning):
        detailed, details = cox_cv._select_coxph_penalty_cv(
            np.zeros((4, 1)),
            np.arange(1.0, 5.0),
            np.array([1.0, 0.0, 1.0, 0.0]),
            penalties=[1.0, 0.5],
            return_details=True,
        )

    assert scalar == pytest.approx(detailed)
    assert details["successive_halving_requested"] is True


def test_staged_fallback_restores_env_reader_after_failure(monkeypatch):
    original_env_flag = cox_cv._env_flag

    def broken_selector(*args, **kwargs):
        assert cox_cv._env_flag("STATGPU_COXPHCV_TWO_STAGE", False) is False
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(staged, "_ORIGINAL_SELECT_COXPH_PENALTY_CV", broken_selector)
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE", "1")

    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="candidate failure"):
            cox_cv._select_coxph_penalty_cv(
                np.zeros((4, 1)),
                np.arange(1.0, 5.0),
                np.array([1.0, 0.0, 1.0, 0.0]),
                penalties=[1.0],
                return_details=True,
            )
    assert cox_cv._env_flag is original_env_flag


def test_staged_fallback_serializes_concurrent_requests(monkeypatch):
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    def fake_selector(*args, **kwargs):
        nonlocal active, maximum_active
        assert cox_cv._env_flag("STATGPU_COXPHCV_TWO_STAGE", False) is False
        assert (
            cox_cv._env_flag("STATGPU_COXPHCV_SUCCESSIVE_HALVING", False)
            is False
        )
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.05)
            return 1.0, {
                "penalty": 1.0,
                "penalties": np.array([1.0, 0.5]),
                "mean_pl": np.array([2.0, 1.0]),
            }
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(staged, "_ORIGINAL_SELECT_COXPH_PENALTY_CV", fake_selector)
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_SUCCESSIVE_HALVING", "1")

    def invoke():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return cox_cv._select_coxph_penalty_cv(
                np.zeros((4, 1)),
                np.arange(1.0, 5.0),
                np.array([1.0, 0.0, 1.0, 0.0]),
                penalties=[1.0, 0.5],
                return_details=True,
            )[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        selected = list(executor.map(lambda _: invoke(), range(2)))

    assert selected == [1.0, 1.0]
    assert maximum_active == 1


def test_real_selector_evaluates_every_candidate_when_halving_requested(
    monkeypatch,
):
    class DeterministicCoxPH:
        def __init__(self, *, penalty, **kwargs):
            self.penalty = float(penalty)
            self._converged = True
            self._iterations = 1

        def fit(self, X, *args, **kwargs):
            self.coef_ = np.array([self.penalty, 0.0], dtype=np.float64)
            return self

    def score_from_penalty(X, time_values, event, coef, **kwargs):
        return float(coef[0])

    X, time_values, event = _sample()
    penalties = np.geomspace(1.0, 0.01, 8)
    cox_cv._COXPH_CV_CACHE.clear()
    monkeypatch.setattr(cox_cv, "CoxPH", DeterministicCoxPH)
    monkeypatch.setattr(cox_cv, "_compute_partial_likelihood", score_from_penalty)
    monkeypatch.setenv("STATGPU_COXPHCV_TWO_STAGE", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_SUCCESSIVE_HALVING", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_HALVING_TOPK", "1")
    monkeypatch.setenv("STATGPU_COXPHCV_FOLD_CACHE_MAX_BYTES", "0")

    with pytest.warns(RuntimeWarning, match="exhaustive full-precision"):
        best, details = cox_cv._select_coxph_penalty_cv(
            X,
            time_values,
            event,
            penalties=penalties,
            cv_folds=3,
            random_state=4,
            device="cpu",
            return_details=True,
            cache_key="staged-safety-evaluates-all",
        )

    assert best == pytest.approx(penalties[0])
    assert np.all(details["attempted_path"])
    assert np.all(details["candidate_complete"])
    assert np.all(details["full_precision_candidate_mask"])
    assert not np.any(details["fast_pass_candidate_mask"])
    assert not np.any(details["screened_out_candidate_mask"])
    assert details["staged_safety_strategy"] == "single_pass_exhaustive"
    assert (
        details["fold_backend_preparation_count_this_call"]
        == details["effective_n_folds"]
    )


def test_coxphcv_docstring_discloses_staged_safety_fallback():
    documentation = inspect.getdoc(CoxPHCV)
    assert documentation is not None
    assert "Experimental screening safety" in documentation
    assert "one exhaustive full-precision CV pass" in documentation
