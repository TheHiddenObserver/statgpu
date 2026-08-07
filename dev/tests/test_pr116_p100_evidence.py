from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_IMPLEMENTATION_SHA = "e6e4846b06604ed53e65fc9afd9054bd5777098f"
EXPECTED_CANONICAL_SHA256 = "bd8d512adced442b066de20fb2c31f3f50b19f271d6d0de6bd0d82e9b4cd9be8"
EXPECTED_FOCUSED_SHA256 = "70d0b094cf66555fcba1b3967b80f92cc6471b136f15683dbc9c9c3cfa69415c"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pr116_canonical_p100_source_is_exact_head_and_all_success() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    path = REPO_ROOT / "results" / "pr116_p100" / "cv_benchmark_pr116_p100.json"
    assert _sha256(path) == EXPECTED_CANONICAL_SHA256

    source = _load(path)
    assert validate_cv_source(source) == []
    assert source["git_sha"] == EXPECTED_IMPLEMENTATION_SHA
    assert source["environment"]["env_id"] == "remote-p100-pr116-20260807"
    assert source["environment"]["gpu"] == "Tesla P100-SXM2-16GB"
    assert source["environment"]["python"] == "3.9.16"
    assert source["environment"]["packages"]["torch"] == "2.0.0"
    assert source["environment"]["packages"]["cupy"] == "13.6.0"
    assert set(source["environment"]["available_backends"]) == {"numpy", "cupy", "torch"}

    statgpu_runs = [
        run
        for case in source["cases"]
        for run in case["runs"]
        if run["framework"] == "statgpu"
    ]
    assert len(statgpu_runs) == 18
    assert all(run["status"] == "success" for run in statgpu_runs)
    assert all(run["reason"] in (None, "") for run in statgpu_runs)
    assert all(run["convergence"]["failed_candidates"] == 0 for run in statgpu_runs)
    assert all(run["convergence"]["failed_folds"] == 0 for run in statgpu_runs)
    assert all(run["convergence"]["final_refit_converged"] is True for run in statgpu_runs)

    logistic = next(case for case in source["cases"] if case["model_id"] == "LogisticRegressionCV")
    logistic_statgpu = {
        run["backend"]: run
        for run in logistic["runs"]
        if run["framework"] == "statgpu"
    }
    assert set(logistic_statgpu) == {"numpy", "cupy", "torch"}
    assert all(run["status"] == "success" for run in logistic_statgpu.values())
    assert all(run["selected_parameters"] == {"C": 0.1} for run in logistic_statgpu.values())
    assert logistic_statgpu["torch"]["device"] == "torch"


def test_pr116_focused_p100_evidence_covers_repaired_branches() -> None:
    path = REPO_ROOT / "results" / "pr116_p100" / "focused_validation.json"
    assert _sha256(path) == EXPECTED_FOCUSED_SHA256

    focused = _load(path)
    assert focused["git_sha"] == EXPECTED_IMPLEMENTATION_SHA
    assert focused["status"] == "success"
    assert focused["torch_version"] == "2.0.0+cu117"
    assert focused["torch_cuda_version"] == "11.7"
    assert focused["gpu"] == "Tesla P100-SXM2-16GB"
    assert focused["cuda_capability"] == [6, 0]

    cases = {case["name"]: case for case in focused["cases"]}
    assert set(cases) == {
        "mixed_precision_float32_unweighted_intercept",
        "float64_unweighted_intercept",
        "mixed_precision_float32_weighted_intercept",
        "mixed_precision_float32_no_intercept",
    }
    assert all(case["gpu_cv_selected_device"] == "torch" for case in cases.values())
    assert all(case["selected_same"] is True for case in cases.values())
    assert all(case["near_tie"] is False for case in cases.values())
    assert all(case["cpu_C"] == case["gpu_C"] == 0.2 for case in cases.values())
    assert max(case["max_abs_loss_diff"] for case in cases.values()) < 1e-6
