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
    assert runner.SCHEMA_VERSION == 5
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
        '"host_transfer_provenance"',
        '"linear_regression_nonrobust"',
        '"linear_regression_hc3"',
        '"linear_regression_hac"',
        '"cupy_nonrank_failure_fail_closed"',
        "_cupy_nonrank_failure_case(concrete_device)",
        '"distribution_backends"',
        "_linear_regression_case(backend, concrete_device, cov_type)",
        '"pre_reporting_host_transfers"',
        '"reference_distribution_completed_on_backend"',
        "_host_transfer_case(backend, concrete_device)",
        '"ridgecv_final_refit_inference"',
        '"functional_rank_and_multitarget"',
        '"student_t_df2_extreme_tail"',
        '"covariance"',
        '"statistic"',
    ):
        assert required in source


def test_runner_host_transfer_guard_restores_instrumented_functions():
    source = RUNNER.read_text()
    assert "finally:" in source
    assert "_gi_module._to_numpy = real_gi_to_numpy" in source
    assert "_pglm_base_module._to_numpy = real_pglm_to_numpy" in source
    assert "_gi_module.two_sided_reference_inference = real_reference" in source


def test_runner_small_df_reference_is_nonzero_at_extreme_float64_tail():
    runner = _load_runner()
    statistic = 1.0e154
    expected = runner._expected_t2_two_sided(statistic)
    assert expected > 0.0
    assert expected < 1e-300


def test_linear_regression_gpu_scalar_critical_values_are_backend_explicit():
    linear_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "wrappers"
        / "_linear.py"
    ).read_text()
    cupy_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "backends"
        / "_gpu_inference_cupy.py"
    ).read_text()
    assert 'norm.ppf(0.975, backend="cupy")' in linear_source
    assert 'norm.ppf(0.975, backend="torch", device=torch_device)' in linear_source
    assert 'self._selected_backend_device = str(X_arr.device)' in linear_source
    assert 'self._selected_backend_device = f"cuda:{int(X_arr.device.id)}"' in linear_source
    assert 'getattr(self, "_selected_backend_device", "")' in linear_source
    assert 't.two_sided_critical_value(alpha, df=df_resid, backend="cupy")' in cupy_source



def test_linear_regression_gpu_device_authority_and_cupy_fail_closed_are_static():
    linear_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "wrappers"
        / "_linear.py"
    ).read_text()
    cupy_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "backends"
        / "_gpu_inference_cupy.py"
    ).read_text()
    workflow_source = (
        Path(__file__).parents[2]
        / ".github"
        / "workflows"
        / "gaussian-inference.yml"
    ).read_text()

    assert "str(X.device) if isinstance(X, torch.Tensor)" in linear_source
    assert "torch_device = str(X.device)" in linear_source
    assert "elif str(y.device) != torch_device" in linear_source
    assert "with cp.cuda.Device(cupy_device_id)" in linear_source
    assert "_linalg_exception_is_rank_failure(exc)" in cupy_source
    assert "with cp.cuda.Device(device_id)" in cupy_source
    assert workflow_source.count("statgpu/backends/_gpu_inference_cupy.py") >= 3
