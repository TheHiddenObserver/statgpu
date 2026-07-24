"""Contract tests for the PR79 full Cox physical-GPU evidence matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import dev.benchmarks.pr79.diagnose_cox_pen as diagnostic
from dev.benchmarks.pr79.aggregate_results import expected_checks
from dev.benchmarks.pr79.diagnose_cox_pen import (
    _add_backend_parity_checks,
    _maximum_objective_decrease,
    _new_model,
    expand_physical_gpu_matrix,
    prepare_physical_gpu_case,
)


MANIFEST_PATH = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "pr79"
    / "configs"
    / "expected_accuracy_manifest.json"
)


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_full_manifest_declares_complete_cox_physical_gpu_axes():
    matrix = _manifest()["configurations"]["full"]["cox_physical_gpu_matrix"]
    axes = matrix["axes"]

    assert matrix["matrix_schema_version"] == "pr79-cox-gpu-matrix-1.0"
    assert matrix["reference_backend"] == "numpy"
    assert matrix["physical_gpu_backends"] == ["cupy", "torch"]
    assert axes["penalty"] == [0.01, 0.1, 1.0]
    assert axes["ties"] == ["breslow", "efron"]
    assert axes["entry"] == [False, True]
    assert axes["tie_pattern"] == ["no_ties", "small_ties", "heavy_ties"]
    assert axes["compute_inference"] == [False, True]
    assert axes["inference_mode"] == ["strict", "approx"]
    assert axes["row_order"] == ["canonical", "permuted"]
    assert matrix["execution"] == {
        "driver": "dev/benchmarks/pr79/diagnose_cox_pen.py",
        "enumerate_cli": "--print-full-matrix",
        "single_case_cli": "--matrix-case-id CASE_ID",
        "full_matrix_cli": "--run-full-matrix --matrix-backend BACKEND",
        "cov_type_policy": "hc0_when_no_entry_and_inference_else_nonrobust",
        "stable_sort_required_after_permutation": True,
        "gpu_synchronize_for_timing": True,
        "final_state_recomputation": True,
        "peak_memory_recording": True,
    }


def test_expanded_cox_gpu_matrix_is_complete_unique_and_thresholded():
    matrix = _manifest()["configurations"]["full"]["cox_physical_gpu_matrix"]
    cases = expand_physical_gpu_matrix(matrix)
    expected_count = 2 * 3 * 2 * 2 * 3 * 2 * 2 * 2

    assert len(cases) == expected_count
    assert len({case["case_id"] for case in cases}) == expected_count
    assert all(case["thresholds"] == matrix["thresholds"] for case in cases)
    assert all(
        case["cov_type"]
        == ("hc0" if case["compute_inference"] and not case["entry"] else "nonrobust")
        for case in cases
    )
    assert any(
        case["backend"] == "cupy"
        and case["penalty"] == 1.0
        and case["ties"] == "efron"
        and case["entry"] is True
        and case["tie_pattern"] == "heavy_ties"
        and case["compute_inference"] is True
        and case["inference_mode"] == "strict"
        and case["row_order"] == "permuted"
        for case in cases
    )


def test_full_cox_gate_thresholds_are_not_hidden_by_unified_1e_minus_5():
    manifest = _manifest()
    checks = expected_checks(manifest, "full")
    cox_case_ids = {
        case["case_id"]
        for case in manifest["cases"].values()
        if case["model_id"] == "CoxPH"
    }
    cox_final = [
        check
        for check in checks
        if check["case_id"] in cox_case_ids
        and check["metric"] == "final_state_contract"
    ]
    cox_loglik = [
        check
        for check in checks
        if check["case_id"] in cox_case_ids
        and check["metric"] == "loglik_rel_error"
    ]

    assert cox_final and all(check["threshold"] == 1e-9 for check in cox_final)
    assert cox_loglik and all(check["threshold"] == 1e-9 for check in cox_loglik)
    assert (
        manifest["configurations"]["smoke"]["expected_checks"][1]["threshold"]
        == 1e-9
    )

    thresholds = manifest["configurations"]["full"]["cox_physical_gpu_matrix"][
        "thresholds"
    ]
    assert thresholds == {
        "coefficient_rel_l2_error": 1e-6,
        "unpenalized_log_likelihood_rel_error": 1e-9,
        "penalized_objective_rel_error": 1e-9,
        "normalized_final_kkt": 1e-7,
        "hessian_rel_fro_error": 1e-6,
        "covariance_rel_fro_error": 1e-5,
        "bse_rel_error": 1e-5,
        "objective_decrease_tolerance": 1e-10,
    }


def test_diagnostic_uses_reported_fitted_parity_thresholds():
    fixed = {
        "unpenalized_log_likelihood": -10.0,
        "penalized_objective": -10.1,
        "gradient": np.zeros(2),
        "unpenalized_hessian": -np.eye(2),
        "penalized_hessian": -1.2 * np.eye(2),
        "covariance": np.eye(2),
        "bse": np.ones(2),
    }
    fitted = {
        "coefficients": np.array([0.2, -0.1]),
        "unpenalized_log_likelihood": -10.0,
        "penalized_objective": -10.1,
        "final_kkt_normalized": 1e-10,
        "bse": np.ones(2),
        "converged": True,
        "termination_reason": "kkt_converged",
        "iterations": 4,
        "fixed_beta_bse_at_solution": np.ones(2),
        "objective_history": [-11.0, -10.5, -10.1],
    }
    checks = []
    _add_backend_parity_checks(
        checks, "cupy", fixed, fixed, fitted, dict(fitted)
    )
    tolerances = {check["name"]: check.get("tolerance") for check in checks}

    assert tolerances["fitted_coefficients_parity"] == 1e-6
    assert tolerances["fitted_unpenalized_log_likelihood_parity"] == 1e-9
    assert tolerances["fitted_penalized_objective_parity"] == 1e-9
    assert tolerances["fitted_final_normalized_kkt"] == 1e-7
    assert tolerances["fitted_bse_parity"] == 1e-5
    assert tolerances["fitted_objective_maximum_decrease"] == 1e-10
    assert _maximum_objective_decrease(
        [-2.0, -1.0, -1.0 - 5e-11]
    ) == pytest.approx(5e-11)


def test_matrix_case_preparation_applies_ties_entry_and_row_permutation():
    base = {
        "penalty": 0.1,
        "ties": "efron",
        "entry": True,
        "tie_pattern": "heavy_ties",
        "compute_inference": True,
        "inference_mode": "approx",
        "row_order": "canonical",
    }
    canonical = prepare_physical_gpu_case(base, n=48, p=4)
    permuted = prepare_physical_gpu_case(
        {**base, "row_order": "permuted"}, n=48, p=4
    )

    assert canonical["entry"] is not None
    assert np.all(canonical["entry"] <= canonical["time"])
    _, counts = np.unique(canonical["time"], return_counts=True)
    assert counts.max() == 12
    assert not np.array_equal(canonical["time"], permuted["time"])
    assert np.array_equal(
        np.sort(canonical["time"]), np.sort(permuted["time"])
    )


def test_matrix_model_options_are_passed_to_cox_constructor(monkeypatch):
    captured = {}

    class FakeCoxPH:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import statgpu.survival

    monkeypatch.setattr(statgpu.survival, "CoxPH", FakeCoxPH)
    _new_model(
        "cupy",
        compute_inference=False,
        penalty=1.0,
        ties="breslow",
        tol=1e-6,
        max_iter=30,
        inference_mode="approx",
        cov_type="hc0",
    )

    assert captured["device"] == "cuda"
    assert captured["compute_inference"] is False
    assert captured["penalty"] == 1.0
    assert captured["ties"] == "breslow"
    assert captured["inference_mode"] == "approx"
    assert captured["cov_type"] == "hc0"


def test_matrix_runner_forwards_every_case_parameter(monkeypatch):
    matrix = _manifest()["configurations"]["full"]["cox_physical_gpu_matrix"]
    case = next(
        item
        for item in expand_physical_gpu_matrix(matrix)
        if item["backend"] == "cupy"
        and item["penalty"] == 1.0
        and item["ties"] == "efron"
        and item["entry"] is True
        and item["tie_pattern"] == "heavy_ties"
        and item["compute_inference"] is False
        and item["inference_mode"] == "approx"
        and item["row_order"] == "permuted"
    )
    entry = np.array([0.1, 0.2])
    monkeypatch.setattr(
        diagnostic,
        "prepare_physical_gpu_case",
        lambda selected: {
            "X": np.eye(2),
            "time": np.array([1.0, 2.0]),
            "event": np.ones(2, dtype=np.int32),
            "entry": entry,
            "fixed_beta": np.zeros(2),
        },
    )
    monkeypatch.setattr(diagnostic, "_require_backend", lambda backend: None)
    monkeypatch.setattr(diagnostic, "_start_gpu_memory_tracking", lambda backend: None)
    monkeypatch.setattr(diagnostic, "_synchronize_backend", lambda backend: None)
    monkeypatch.setattr(diagnostic, "_peak_gpu_memory_bytes", lambda backend: 123)
    fixed_calls = []
    fit_calls = []
    fixed = {
        "unpenalized_log_likelihood": -2.0,
        "penalized_objective": -2.0,
        "gradient": np.zeros(2),
        "unpenalized_hessian": -np.eye(2),
    }
    fitted = {
        "coefficients": np.zeros(2),
        "unpenalized_log_likelihood": -2.0,
        "reported_unpenalized_log_likelihood": -2.0,
        "penalized_objective": -2.0,
        "reported_penalized_objective": -2.0,
        "final_kkt_normalized": 0.0,
        "objective_history": [-3.0, -2.0],
        "converged": True,
        "termination_reason": "kkt_converged",
    }

    def fake_fixed(backend, **kwargs):
        fixed_calls.append((backend, kwargs))
        return dict(fixed)

    def fake_fit(backend, **kwargs):
        fit_calls.append((backend, kwargs))
        return dict(fitted)

    monkeypatch.setattr(diagnostic, "evaluate_fixed_beta", fake_fixed)
    monkeypatch.setattr(diagnostic, "_fit_backend", fake_fit)
    report = diagnostic.run_physical_gpu_matrix_case(case)

    assert report["status"] == "pass"
    assert [backend for backend, _ in fixed_calls] == ["numpy", "cupy"]
    assert [backend for backend, _ in fit_calls] == ["cupy"]
    forwarded = fit_calls[0][1]
    assert forwarded["penalty"] == 1.0
    assert forwarded["ties"] == "efron"
    assert forwarded["entry"] is entry
    assert forwarded["compute_inference"] is False
    assert forwarded["inference_mode"] == "approx"
    assert forwarded["cov_type"] == "nonrobust"
    assert report["peak_gpu_memory_bytes"] == 123


def test_full_matrix_runner_checks_canonical_vs_permuted_results(monkeypatch):
    thresholds = {
        "coefficient_rel_l2_error": 1e-6,
        "unpenalized_log_likelihood_rel_error": 1e-9,
        "penalized_objective_rel_error": 1e-9,
    }
    cases = [
        {
            "case_id": "canonical-id",
            "backend": "cupy",
            "penalty": 0.1,
            "ties": "efron",
            "entry": False,
            "tie_pattern": "small_ties",
            "compute_inference": True,
            "inference_mode": "strict",
            "cov_type": "hc0",
            "row_order": "canonical",
            "thresholds": thresholds,
        },
        {
            "case_id": "permuted-id",
            "backend": "cupy",
            "penalty": 0.1,
            "ties": "efron",
            "entry": False,
            "tie_pattern": "small_ties",
            "compute_inference": True,
            "inference_mode": "strict",
            "cov_type": "hc0",
            "row_order": "permuted",
            "thresholds": thresholds,
        },
    ]

    def fake_run(case):
        shift = 2e-4 if case["row_order"] == "permuted" else 0.0
        return {
            "case": case,
            "status": "pass",
            "results": {
                "fitted_gpu": {
                    "coefficients": [0.2 + shift, -0.1],
                    "unpenalized_log_likelihood": -10.0,
                    "penalized_objective": -10.1,
                }
            },
        }

    monkeypatch.setattr(diagnostic, "run_physical_gpu_matrix_case", fake_run)
    report = diagnostic.run_physical_gpu_matrix(cases)

    assert report["status"] == "fail"
    assert len(report["permutation_checks"]) == 1
    assert report["permutation_checks"][0]["status"] == "fail"
    assert (
        report["permutation_checks"][0]["metrics"]["coefficient_rel_l2_error"]
        > thresholds["coefficient_rel_l2_error"]
    )
