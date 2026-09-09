import json

import pytest

from dev.benchmarks import validate_post_selection_ols_gpu as physical


def _payload():
    return {
        "schema_version": 7,
        "issue": 137,
        "head_sha": "deadbeef",
        "worktree_clean": True,
        "timestamp_utc": "2026-09-09T00:00:00+00:00",
        "python": "3.9.16",
        "platform": "test-platform",
        "cases": [
            {
                "model": "lasso",
                "case": "unweighted",
                "backend": "cupy",
                "status": "success",
            }
        ],
        "closure_cases": [],
    }


def test_record_case_failure_writes_partial_artifact_and_reraises(tmp_path):
    payload = _payload()
    output = tmp_path / "physical_failure.json"
    original = AssertionError(
        "rank-deficient torch pvalue error 2.538e-06 exceeds 2.000e-06"
    )

    def fail():
        raise original

    with pytest.raises(AssertionError) as captured:
        physical._record_case(
            payload,
            "closure_cases",
            backend="torch",
            model="elasticnet",
            case="weighted_rank_deficient_active_refit",
            output=output,
            factory=fail,
        )

    assert captured.value is original
    assert output.is_file()
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["schema_version"] == 7
    assert artifact["head_sha"] == "deadbeef"
    assert artifact["worktree_clean"] is True
    assert artifact["status"] == "failure"
    assert artifact["cases"] == payload["cases"]
    assert artifact["closure_cases"] == []
    assert artifact["failure"] == {
        "section": "closure_cases",
        "backend": "torch",
        "model": "elasticnet",
        "case": "weighted_rank_deficient_active_refit",
        "error_type": "AssertionError",
        "error": str(original),
        "completed_cases": 1,
        "completed_closure_cases": 0,
    }


def test_record_case_success_preserves_success_result_without_early_artifact(tmp_path):
    payload = _payload()
    output = tmp_path / "physical_success.json"
    result = {
        "model": "elasticnet",
        "case": "weighted_rank_deficient_active_refit",
        "backend": "torch",
        "status": "success",
    }

    returned = physical._record_case(
        payload,
        "closure_cases",
        backend="torch",
        model="elasticnet",
        case="weighted_rank_deficient_active_refit",
        output=output,
        factory=lambda: result,
    )

    assert returned is result
    assert payload["closure_cases"] == [result]
    assert "status" not in payload
    assert "failure" not in payload
    assert not output.exists()
