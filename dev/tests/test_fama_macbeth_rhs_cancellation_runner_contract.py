"""Hosted contract checks for the focused Fama-MacBeth RHS CUDA validator."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


_RUNNER = Path("dev/benchmarks/validate_fama_macbeth_rhs_cancellation_gpu.py")
_SPEC = importlib.util.spec_from_file_location("fmb_rhs_gpu_runner", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_fmb_rhs_runner_intercept_tail_fixture_is_well_conditioned():
    X, y, time, amplitude = _MOD._intercept_tail_fixture()
    assert X.shape == (9, 1)
    assert y.shape == (9,)
    assert time.shape == (9,)
    assert amplitude == float(2.0**55)
    design = np.column_stack([np.ones(3), X[:3, 0]])
    assert np.linalg.cond(design) < 2.0
    np.testing.assert_array_equal(
        y[:3],
        np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64),
    )


def test_fmb_rhs_runner_precision_controls_are_condition_one():
    X_bad, y_bad, _time_bad = _MOD._ambiguous_nonconstant_rhs_fixture()
    bad_design = np.column_stack([np.ones(4), X_bad[:4, 0]])
    assert np.linalg.cond(bad_design) == 1.0
    # Ordinary BLAS can erase the nonconstant -1 tail on this maintained input.
    assert (bad_design.T @ y_bad[:4])[1] == 0.0

    X_drift, y_drift, _time_drift = _MOD._nonzero_rhs_svd_drift_fixture()
    drift_design = np.column_stack([np.ones(4), X_drift[:4, 0]])
    assert np.linalg.cond(drift_design) == 1.0
    np.testing.assert_array_equal(
        drift_design.T @ y_drift[:4],
        np.asarray([-32.0, -32.0], dtype=np.float64),
    )

    X_zero, y_zero, _time_zero = _MOD._genuine_zero_rhs_fixture()
    zero_design = np.column_stack([np.ones(4), X_zero[:4, 0]])
    assert np.linalg.cond(zero_design) == 1.0
    amplitude = float(2.0**55)
    np.testing.assert_array_equal(
        y_zero[:4],
        np.asarray([amplitude, -amplitude, -amplitude, amplitude], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        zero_design.T @ y_zero[:4],
        np.zeros(2, dtype=np.float64),
    )


def test_fmb_rhs_runner_records_backend_provenance_and_all_four_outcomes():
    source = inspect.getsource(_MOD.run)
    trace_source = inspect.getsource(_MOD._trace_public_fallback)
    for token in (
        "_git_clean()",
        '"schema_version": 3',
        '"git_sha": _git_sha()',
        '"clean_worktree": True',
        '"requested_backend": backend',
        '"executed_backend"',
        '"execution_device"',
        "cp.__version__",
        "torch.__version__",
        '"intercept_tail"',
        '"lost_nonconstant_rhs_tail_fail_closed"',
        '"lost_nonconstant_rhs_trace"',
        '"nonzero_rhs_svd_drift_fail_closed"',
        '"nonzero_rhs_svd_drift_trace"',
        '"genuine_zero_rhs"',
        '"gram-certified"',
        "_period_svd_fallbacks",
        "_assert_success_backend",
        "_expected_precision_failure",
    ):
        assert token in source

    for token in (
        "panel_lstsq_deferred_rank",
        "panel_lstsq_batched",
        '"params_native"',
        '"rank_native"',
        '"params_device"',
        '"rank_device"',
    ):
        assert token in trace_source

    backend_source = inspect.getsource(_MOD._backend_arrays)
    assert 'backend == "cupy"' in backend_source
    assert 'backend == "torch"' in backend_source


def test_fmb_rhs_runner_trace_validator_rejects_silent_fallback_evidence():
    _MOD._validate_public_trace(
        "cupy",
        [
            {
                "namespace": "cupy",
                "x_native": True,
                "y_native": True,
                "params_native": True,
                "rank_native": True,
                "x_device": 0,
                "y_device": 0,
                "params_device": 0,
                "rank_device": 0,
            }
        ],
        0,
    )
    _MOD._validate_public_trace(
        "torch",
        [
            {
                "namespace": "torch",
                "x_native": True,
                "y_native": True,
                "params_native": True,
                "rank_native": True,
                "x_device": "cuda:0",
                "y_device": "cuda:0",
                "params_device": "cuda:0",
                "rank_device": "cuda:0",
                "x_is_cuda": True,
                "y_is_cuda": True,
                "params_is_cuda": True,
                "rank_is_cuda": True,
            }
        ],
        "cuda:0",
    )

    with pytest.raises(AssertionError, match="namespace mismatch"):
        _MOD._validate_public_trace(
            "cupy",
            [
                {
                    "namespace": "numpy",
                    "x_native": False,
                    "y_native": False,
                    "params_native": False,
                    "rank_native": False,
                    "x_device": None,
                    "y_device": None,
                    "params_device": None,
                    "rank_device": None,
                }
            ],
            0,
        )

    with pytest.raises(AssertionError, match="non-native"):
        _MOD._validate_public_trace(
            "cupy",
            [
                {
                    "namespace": "cupy",
                    "x_native": True,
                    "y_native": True,
                    "params_native": False,
                    "rank_native": False,
                    "x_device": 0,
                    "y_device": 0,
                    "params_device": None,
                    "rank_device": None,
                }
            ],
            0,
        )

    with pytest.raises(AssertionError, match="inconsistent fallback device"):
        _MOD._validate_public_trace(
            "torch",
            [
                {
                    "namespace": "torch",
                    "x_native": True,
                    "y_native": True,
                    "params_native": True,
                    "rank_native": True,
                    "x_device": "cuda:0",
                    "y_device": "cuda:0",
                    "params_device": "cuda:1",
                    "rank_device": "cuda:1",
                    "x_is_cuda": True,
                    "y_is_cuda": True,
                    "params_is_cuda": True,
                    "rank_is_cuda": True,
                }
            ],
            "cuda:0",
        )


def test_fmb_rhs_runner_cli_requires_one_physical_backend():
    source = inspect.getsource(_MOD.main)
    assert 'choices=("cupy", "torch")' in source
    assert '"--backend"' in source
    assert '"--output"' in source
