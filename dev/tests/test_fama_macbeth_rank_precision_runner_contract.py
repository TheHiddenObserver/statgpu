"""Hosted contract for the Fama-MacBeth rank/precision physical GPU gate."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


_RUNNER = Path("dev/benchmarks/validate_fama_macbeth_rank_precision_precedence_gpu.py")
_SPEC = importlib.util.spec_from_file_location("fmb_rank_precision_gpu", _RUNNER)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_rank_precision_runner_fixture_keeps_rank_two_and_erases_raw_intercept_tail():
    X, y, time, design, y_period = _MOD._fixture()
    assert X.shape == (9, 2)
    assert y.shape == (9,)
    assert time.shape == (9,)
    assert design.shape == (3, 3)
    assert np.linalg.matrix_rank(design) == 2
    assert (design.T @ y_period)[0] == 0.0
    assert y_period[1] == 1.0


def test_rank_precision_runner_locks_backend_and_failure_precedence_contract():
    source = inspect.getsource(_MOD.run)
    for token in (
        'backend == "cupy"',
        'backend == "torch"',
        "panel_lstsq_deferred_rank",
        "panel_lstsq_batched",
        "_trace_public_fallback(backend)",
        "_validate_public_trace(backend, trace)",
        'executed_backend = str(trace[0]["namespace"])',
        "cp.__version__",
        "torch.__version__",
        "except FloatingPointError",
        "except ValueError",
        '"rank=2, columns=3"',
        '"fallback_svd_rank"',
        '"public_rank_deficiency_value_error"',
        '"precision_failure_misclassification": False',
        '"requested_backend": backend',
        '"executed_backend": executed_backend',
        '"public_fallback_trace": trace',
        '"public_failure_state_clean": True',
        '"inference_state_published": False',
        '"git_sha": _git_sha()',
        '"clean_worktree": True',
    ):
        assert token in source


def test_rank_precision_runner_trace_validator_rejects_silent_fallback_evidence():
    _MOD._validate_public_trace(
        "cupy",
        [
            {
                "namespace": "cupy",
                "x_native": True,
                "y_native": True,
                "x_device": 0,
                "y_device": 0,
            }
        ],
    )
    _MOD._validate_public_trace(
        "torch",
        [
            {
                "namespace": "torch",
                "x_native": True,
                "y_native": True,
                "x_device": "cuda:0",
                "y_device": "cuda:0",
                "x_is_cuda": True,
                "y_is_cuda": True,
            }
        ],
    )

    with pytest.raises(AssertionError, match="namespace"):
        _MOD._validate_public_trace(
            "cupy",
            [
                {
                    "namespace": "numpy",
                    "x_native": False,
                    "y_native": False,
                    "x_device": None,
                    "y_device": None,
                }
            ],
        )
    with pytest.raises(AssertionError, match="left CUDA"):
        _MOD._validate_public_trace(
            "torch",
            [
                {
                    "namespace": "torch",
                    "x_native": True,
                    "y_native": True,
                    "x_device": "cpu",
                    "y_device": "cpu",
                    "x_is_cuda": False,
                    "y_is_cuda": False,
                }
            ],
        )


def test_rank_precision_runner_cli_requires_one_physical_backend():
    source = inspect.getsource(_MOD.main)
    assert 'choices=("cupy", "torch")' in source
    assert '"--backend"' in source
    assert '"--output"' in source
