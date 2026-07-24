"""Behavior tests for the PR79 canonical accuracy evidence pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from dev.benchmarks.pr79 import aggregate_results as aggregate_module
from dev.benchmarks.pr79.aggregate_results import (
    AggregationError,
    aggregate_results,
    expected_checks,
    load_json_strict,
)
from dev.benchmarks.pr79.emit_final_report import (
    ReportValidationError,
    emit_report,
    render_markdown,
    validate_aggregated_report,
)
from dev.benchmarks.pr79.runners.common import make_raw_run, safe_run
from dev.benchmarks.pr79.validators.numerical import (
    NumericalValidationError,
    bse_rel_error,
    recompute_cox_final_state,
    validate_cox_final_state,
)


TEST_SHA = "a" * 40


def _clean_snapshot() -> dict:
    return {
        "git_sha": TEST_SHA,
        "worktree_clean": True,
        "dirty_entries": [],
        "inspection_error": None,
    }


@pytest.fixture(autouse=True)
def _canonical_aggregation_snapshot(monkeypatch):
    monkeypatch.setattr(aggregate_module, "_git_snapshot", _clean_snapshot)


def _manifest() -> dict:
    return {
        "manifest_schema_version": "pr79-accuracy-manifest-1.0",
        "allowed_classifications": [
            "meaningful_parity",
            "contract",
            "not_comparable",
        ],
        "cases": {
            "linear": {
                "case_id": "case-linear",
                "model_id": "LinearRegression",
                "rank_deficient": False,
            }
        },
        "configurations": {
            "smoke": {
                "cases": ["linear"],
                "backends": ["numpy"],
                "iterations": 1,
                "expected_runs": [
                    {
                        "run_key": "linear-numpy-0",
                        "case_id": "case-linear",
                        "case_label": "linear",
                        "backend": "numpy",
                        "model_id": "LinearRegression",
                    }
                ],
                "expected_checks": [
                    {
                        "check_id": "linear-final",
                        "run_key": "linear-numpy-0",
                        "reference_run_key": "linear-numpy-0",
                        "case_id": "case-linear",
                        "backend": "numpy",
                        "reference_backend": "numpy",
                        "metric": "final_state_contract",
                        "expected_class": "contract",
                        "threshold": 1e-12,
                    }
                ],
            }
        },
    }


def _raw() -> dict:
    run = make_raw_run(
        "linear-numpy-0",
        "case-linear",
        "method-linear",
        "LinearRegression",
        "statgpu",
        "numpy",
        {"backend": "numpy", "iteration": 0},
        {"fit_warm_s": 0.01},
        {
            "coef_": [2.0],
            "intercept_": 0.0,
            "predictions": [2.0, 4.0],
            "residual_sum_squares": 0.0,
        },
    )
    return {
        "source_schema_version": "pr79-benchmark-source-2.1",
        "benchmark_session_id": "test-session",
        "git_sha": TEST_SHA,
        "repository_provenance": {
            "schema_version": "pr79-repository-provenance-1.0",
            "inspection": "git-status-porcelain-v1",
            "allow_dirty_requested": False,
            "sha_unchanged_during_collection": True,
            "canonical_eligible": True,
            "initial": _clean_snapshot(),
            "final": _clean_snapshot(),
        },
        "configuration": "smoke",
        "selected_backends": ["numpy"],
        "environment": {"python_version": "test"},
        "cases": {
            "case-linear": {
                "case_label": "linear",
                "case_id": "case-linear",
                "model_id": "LinearRegression",
                "parameters": {},
                "inputs": {"X": [[1.0], [2.0]], "y": [2.0, 4.0]},
            }
        },
        "runs": [run],
    }


def test_safe_run_retains_structured_failure_evidence():
    def explode():
        raise RuntimeError("retained failure")

    result, error = safe_run(explode)
    assert result is None
    assert error["error_type"] == "RuntimeError"
    assert error["error"] == "retained failure"
    assert "explode" in error["traceback"]

    record = make_raw_run(
        "failed-run",
        "case",
        "method",
        "Model",
        "statgpu",
        "numpy",
        {"backend": "numpy"},
        None,
        None,
        status="error",
        error=error["error"],
        error_type=error["error_type"],
        traceback_text=error["traceback"],
    )
    assert record["status"] == "error"
    assert record["timing"] is None
    assert record["results"] is None
    assert record["error_type"] == "RuntimeError"
    assert record["traceback"]


def test_non_finite_bse_is_a_hard_numerical_failure():
    with pytest.raises(NumericalValidationError, match="NaN or Inf"):
        bse_rel_error(np.array([np.nan]), np.array([np.nan]))
    with pytest.raises(NumericalValidationError, match="NaN or Inf"):
        bse_rel_error(np.array([1.0]), np.array([np.inf]))


def test_strict_json_loader_rejects_nan(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"metric": NaN}', encoding="utf-8")
    with pytest.raises(AggregationError, match="non-standard/non-finite"):
        load_json_strict(path)


def test_valid_evidence_aggregates_and_summary_is_recomputable():
    report = aggregate_results(
        _raw(), _manifest(), config_name="smoke", expected_sha=TEST_SHA
    )
    assert report["status"] == "pass"
    assert report["summary"]["total_checks"] == len(report["checks"]) == 1
    assert report["summary"]["final_state_contracts_passed"] == 1
    assert report["summary"]["unresolved"] == 0
    validate_aggregated_report(report)


def test_missing_expected_raw_run_hard_fails():
    raw = _raw()
    raw["runs"] = []
    with pytest.raises(AggregationError, match="completeness mismatch"):
        aggregate_results(raw, _manifest(), config_name="smoke", expected_sha=TEST_SHA)


def test_duplicate_run_key_hard_fails():
    raw = _raw()
    raw["runs"].append(copy.deepcopy(raw["runs"][0]))
    with pytest.raises(AggregationError, match="duplicate raw run_key"):
        aggregate_results(raw, _manifest(), config_name="smoke", expected_sha=TEST_SHA)


def test_failed_status_hard_fails_even_when_failure_is_retained():
    raw = _raw()
    raw["runs"][0].update(
        {
            "status": "error",
            "timing": None,
            "results": None,
            "error_type": "RuntimeError",
            "error": "boom",
            "traceback": "trace",
        }
    )
    with pytest.raises(AggregationError, match="status is not success"):
        aggregate_results(raw, _manifest(), config_name="smoke", expected_sha=TEST_SHA)


def test_non_finite_metric_hard_fails():
    raw = _raw()
    raw["runs"][0]["results"]["predictions"][0] = float("nan")
    with pytest.raises(AggregationError, match="non-finite numerical evidence"):
        aggregate_results(raw, _manifest(), config_name="smoke", expected_sha=TEST_SHA)


def test_wrong_validated_sha_hard_fails():
    with pytest.raises(AggregationError, match="validated SHA mismatch"):
        aggregate_results(
            _raw(), _manifest(), config_name="smoke", expected_sha="b" * 40
        )


def test_incomplete_result_schema_hard_fails():
    raw = _raw()
    del raw["runs"][0]["results"]["predictions"]
    with pytest.raises(AggregationError, match="result schema missing"):
        aggregate_results(raw, _manifest(), config_name="smoke", expected_sha=TEST_SHA)


def test_missing_threshold_and_unknown_classification_hard_fail():
    manifest = _manifest()
    del manifest["configurations"]["smoke"]["expected_checks"][0]["threshold"]
    with pytest.raises(AggregationError, match="missing: threshold"):
        aggregate_results(_raw(), manifest, config_name="smoke", expected_sha=TEST_SHA)

    manifest = _manifest()
    manifest["configurations"]["smoke"]["expected_checks"][0][
        "expected_class"
    ] = "runtime_guess"
    with pytest.raises(AggregationError, match="unknown classification"):
        aggregate_results(_raw(), manifest, config_name="smoke", expected_sha=TEST_SHA)


def _cox_case_and_run():
    case = {
        "model_id": "CoxPH",
        "parameters": {"ties": "efron", "penalty": 0.1},
        "inputs": {
            "X": [[0.0], [0.0], [0.0]],
            "time": [1.0, 2.0, 3.0],
            "event": [1, 1, 1],
            "entry": None,
        },
    }
    run = {
        "run_key": "cox-numpy-0",
        "case_id": "case-cox",
        "model_id": "CoxPH",
        "backend": "numpy",
        "status": "success",
        "parameters": {
            "backend": "numpy",
            "iteration": 0,
            "ties": "efron",
            "penalty": 0.1,
        },
        "results": {"coef_": [0.0]},
    }
    recomputed = recompute_cox_final_state(run, case)
    run["results"].update(
        {
            "_log_likelihood": recomputed["log_likelihood"],
            "_penalized_objective": recomputed["penalized_objective"],
            "_final_kkt_inf": recomputed["kkt_inf"],
            "_final_kkt_normalized": recomputed["kkt_normalized"],
            "_var_matrix": recomputed["covariance"].tolist(),
            "_bse": recomputed["bse"].tolist(),
        }
    )
    return case, run


def test_cox_final_state_is_recomputed_at_stored_beta():
    case, run = _cox_case_and_run()
    validation = validate_cox_final_state(run, case, threshold=1e-12)
    assert validation["status"] == "pass"
    names = {check["check"] for check in validation["checks"]}
    assert {
        "cox_log_likelihood_final",
        "cox_penalized_objective_final",
        "cox_kkt_inf_final",
        "cox_hessian_symmetry",
        "cox_covariance_final",
        "cox_bse_final",
    } <= names

    run["results"]["_log_likelihood"] += 1.0
    validation = validate_cox_final_state(run, case, threshold=1e-12)
    assert validation["status"] == "fail"
    assert not next(
        check
        for check in validation["checks"]
        if check["check"] == "cox_log_likelihood_final"
    )["passed"]


def test_renderer_uses_intact_validated_aggregator_object(tmp_path: Path):
    report = aggregate_results(
        _raw(), _manifest(), config_name="smoke", expected_sha=TEST_SHA
    )
    output_json = tmp_path / "final.json"
    output_markdown = tmp_path / "final.md"
    emit_report(report, output_json, output_markdown)
    assert json.loads(output_json.read_text(encoding="utf-8")) == report
    markdown = output_markdown.read_text(encoding="utf-8")
    assert TEST_SHA in markdown
    assert "1 | 1" in markdown
    assert markdown == render_markdown(report)

    tampered = copy.deepcopy(report)
    tampered["validated_git_sha"] = "b" * 40
    with pytest.raises(ReportValidationError, match="SHA-256 mismatch"):
        validate_aggregated_report(tampered)

    tampered = copy.deepcopy(report)
    tampered["summary"]["passed"] = 99
    with pytest.raises(ReportValidationError, match="do not derive"):
        validate_aggregated_report(tampered)


def test_repository_manifest_expands_documented_rank_deficient_checks():
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "pr79"
        / "configs"
        / "expected_accuracy_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks = expected_checks(manifest, "full")
    exclusions = [
        check for check in checks if check["expected_class"] == "not_comparable"
    ]
    assert exclusions
    assert all(check["reason"] for check in exclusions)
    assert all(check["still_comparable"] for check in exclusions)
