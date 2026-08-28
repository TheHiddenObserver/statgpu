"""Hosted contract checks for the issue #127 physical GPU validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


RUNNER = Path(__file__).parents[1] / "benchmarks" / "validate_gaussian_inference_backend_native_gpu.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("gaussian_inference_gpu_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_requires_exact_two_backend_matrix():
    runner = _load_runner()
    assert runner._validate_backend_list("cupy,torch") == ("cupy", "torch")
    for invalid in ("cupy", "torch", "torch,cupy", "cupy,torch,numpy", ""):
        with pytest.raises(ValueError, match="exactly 'cupy,torch'"):
            runner._validate_backend_list(invalid)


def test_runner_schema_and_validation_tiers_are_frozen():
    runner = _load_runner()
    assert runner.SCHEMA_VERSION == 2
    assert runner._REQUIRED_BACKENDS == ("cupy", "torch")
    assert runner._VALIDATION_TIERS == (
        "local-minimal",
        "local-full",
        "remote-full",
    )


def test_runner_source_contains_exact_sha_clean_tree_and_provenance_gates():
    source = RUNNER.read_text()
    for required in (
        '"--expected-sha", required=True',
        '"--validation-tier", required=True',
        '"--backends", required=True',
        '"working_tree_clean_before"',
        '"working_tree_clean_after_checks"',
        '"executed_inference_backend"',
        '"executed_inference_device"',
        '"reporting_boundary"',
        '"no_silent_fallback"',
        '"ridgecv_final_refit_inference"',
        '"functional_rank_and_multitarget"',
        '"student_t_df2_extreme_tail"',
        '"covariance"',
        '"statistic"',
    ):
        assert required in source


def test_runner_small_df_reference_is_nonzero_at_extreme_float64_tail():
    runner = _load_runner()
    statistic = 1.0e154
    expected = runner._expected_t2_two_sided(statistic)
    assert expected > 0.0
    assert expected < 1e-300
