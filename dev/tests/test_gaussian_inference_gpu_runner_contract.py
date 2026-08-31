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
        'for cov_type in ("nonrobust", "hc3", "hac"):',
        '"cupy_nonrank_failure_fail_closed"',
        "_cupy_nonrank_failure_case(concrete_device)",
        '"distribution_backends"',
        "_linear_regression_case(backend, concrete_device, cov_type)",
        '"pre_reporting_host_transfers"',
        '"reference_distribution_completed_on_backend"',
        "_host_transfer_case(backend, concrete_device)",
        "PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference = guarded_post_fit",
        "model.fit(X_native, y_native)",
        '"native_fit_state_observed_before_inference"',
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
    assert "PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference = real_post_fit" in source
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
    assert 'self._selected_backend_device = f"cuda:{cupy_device_id}"' in linear_source
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
    assert "_cupy_asarray_on_device(y_arr, cupy_device_id)" in linear_source
    assert "sample_weight = _cupy_asarray_on_device(" in linear_source
    assert "sample_weight, cupy_device_id" in linear_source
    assert "_cupy_asarray_on_device(y, cupy_device_id)" in linear_source
    assert "from cupyx.scipy.linalg import solve_triangular" in linear_source
    assert "cp.linalg.solve_triangular" not in linear_source
    assert "compute_aic_bic_gpu(" in linear_source
    assert "compute_f_stat_gpu(" in linear_source
    assert "_linalg_exception_is_rank_failure(exc)" in cupy_source
    assert "with cp.cuda.Device(device_id)" in cupy_source
    assert workflow_source.count("statgpu/backends/_gpu_inference_cupy.py") >= 3


def test_shared_gaussian_cupy_allocations_follow_reference_device():
    shared_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "_gaussian_inference.py"
    ).read_text()
    utils_source = (
        Path(__file__).parents[2] / "statgpu" / "backends" / "_utils.py"
    ).read_text()

    assert 'return f"cuda:{int(value.device.id)}"' in shared_source
    assert 'return "cuda"' not in shared_source
    assert 'elif device_label.startswith("cuda:")' in shared_source
    assert 'target_device = int(cp.cuda.runtime.getDevice())' in shared_source
    assert 'return _cupy_asarray_on_device(' in shared_source
    assert 'def _cupy_asarray_on_device' in utils_source
    assert 'with cp.cuda.Device(target_device):' in utils_source
    assert 'value = cp.copy(value)' in utils_source
    assert 'device_id = int(X_arr.device.id)' in shared_source
    assert 'with cp.cuda.Device(device_id):' in shared_source
    assert 'with xp.cuda.Device(int(X.device.id)):' in shared_source


def test_exact_l2_precomputed_gpu_inference_has_concrete_device_contract():
    source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "penalized"
        / "_inference_mixin.py"
    ).read_text()

    assert "device_id = int(X.device.id)" in source
    assert "with cp.cuda.Device(device_id)" in source
    assert 'backend="cupy"' in source
    assert '"numerical_backend": "cupy"' in source
    assert '"numerical_device": f"cuda:{device_id}"' in source
    assert '"numerical_backend": "torch"' in source
    assert '"numerical_device": str(X.device)' in source
    assert source.count('"reporting_boundary": "post_numerical_inference"') >= 2


def test_penalized_gaussian_router_uses_fit_time_concrete_device_authority():
    base_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "penalized"
        / "_base.py"
    ).read_text()
    fit_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "penalized"
        / "_fit_mixin.py"
    ).read_text()

    assert "self._selected_backend_device = None" in base_source
    assert 'getattr(self, "_selected_backend_device", None)' in base_source
    assert "device_label = str(selected_device or \"\")" in base_source
    assert 'elif backend_name == "cupy":' in base_source
    assert 'device_label != "cpu" and not device_label.startswith("cuda:")' in base_source
    assert "device=selected_device" in base_source
    assert "missing concrete executed-device provenance" in base_source
    assert 'self._selected_backend_device = str(X_arr.device)' in fit_source
    assert "cupy_device_id = int(X_arr.device.id)" in fit_source
    assert 'self._selected_backend_device = f"cuda:{cupy_device_id}"' in fit_source
    assert "with cp.cuda.Device(int(X.device.id))" in fit_source


def test_cupy_public_consumers_hold_concrete_device_context_across_numerics():
    linear_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "wrappers"
        / "_linear.py"
    ).read_text()
    fit_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "penalized"
        / "_fit_mixin.py"
    ).read_text()
    base_source = (
        Path(__file__).parents[2]
        / "statgpu"
        / "linear_model"
        / "penalized"
        / "_base.py"
    ).read_text()

    assert "with cp.cuda.Device(int(X_arr.device.id))" in linear_source
    assert "cupy_device_id = int(X_arr.device.id)" in fit_source
    assert "_cupy_asarray_on_device(y_arr, cupy_device_id)" in fit_source
    assert "_cupy_asarray_on_device(_sw_arr, cupy_device_id)" in fit_source
    assert "with cp.cuda.Device(cupy_device_id)" in fit_source
    assert "def _run_gaussian_inference_on_fit_device" in base_source
    assert "with cp.cuda.Device(cupy_device_id)" in base_source
