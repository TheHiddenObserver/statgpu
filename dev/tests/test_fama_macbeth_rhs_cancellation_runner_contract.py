"""Hosted contract checks for the focused Fama-MacBeth RHS CUDA validator."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np


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


def test_fmb_rhs_runner_lost_tail_and_genuine_zero_controls_are_condition_one():
    X_bad, y_bad, _time_bad = _MOD._ambiguous_nonconstant_rhs_fixture()
    bad_design = np.column_stack([np.ones(4), X_bad[:4, 0]])
    assert np.linalg.cond(bad_design) == 1.0
    # Ordinary BLAS can erase the nonconstant -1 tail on this maintained input.
    assert (bad_design.T @ y_bad[:4])[1] == 0.0

    X_zero, y_zero, _time_zero = _MOD._genuine_zero_rhs_fixture()
    zero_design = np.column_stack([np.ones(4), X_zero[:4, 0]])
    assert np.linalg.cond(zero_design) == 1.0
    np.testing.assert_array_equal(
        zero_design.T @ y_zero[:4],
        np.zeros(2, dtype=np.float64),
    )


def test_fmb_rhs_runner_records_backend_provenance_and_all_three_outcomes():
    source = inspect.getsource(_MOD.run)
    for token in (
        "_git_clean()",
        '"git_sha": _git_sha()',
        '"requested_backend": backend',
        '"executed_backend"',
        '"intercept_tail"',
        '"lost_nonconstant_rhs_tail_fail_closed"',
        '"genuine_zero_rhs"',
        '"gram-certified"',
        "_period_svd_fallbacks",
    ):
        assert token in source

    backend_source = inspect.getsource(_MOD._backend_arrays)
    assert 'backend == "cupy"' in backend_source
    assert 'backend == "torch"' in backend_source
    assert 'device="cuda"' not in source  # device comes from the explicit backend helper


def test_fmb_rhs_runner_cli_requires_one_physical_backend():
    source = inspect.getsource(_MOD.main)
    assert 'choices=("cupy", "torch")' in source
    assert '"--backend"' in source
    assert '"--output"' in source
