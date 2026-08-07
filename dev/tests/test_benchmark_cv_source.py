from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MODELS = (
    ("RidgeCV", "regression"),
    ("LassoCV", "regression"),
    ("ElasticNetCV", "regression"),
    ("LogisticRegressionCV", "classification"),
    ("PenalizedGLM_CV", "regression"),
    ("CoxPHCV", "survival"),
)


def _success(framework: str, backend: str | None) -> dict:
    return {
        "framework": framework,
        "backend": backend,
        "device": "cpu",
        "status": "success",
        "reason": None,
        "timing": {
            "cv_evaluation_ms": 10.0,
            "final_refit_ms": 2.0,
            "total_fit_ms": 12.5,
            "peak_memory_bytes": None,
        },
        "selected_parameters": {"alpha": 0.1},
        "scores": {"validation_score": 0.25, "final_score": 0.8},
        "convergence": {
            "candidate_count": 3,
            "fold_count": 3,
            "failed_candidates": 0,
            "failed_folds": 0,
            "final_refit_converged": True,
            "n_iter": 4,
        },
        "repeat_samples": [
            {
                "seed": 42,
                "cv_evaluation_ms": 10.0,
                "final_refit_ms": 2.0,
                "total_fit_ms": 12.5,
            }
        ],
    }


def _unavailable(backend: str) -> dict:
    return {
        "framework": "statgpu",
        "backend": backend,
        "device": "cuda",
        "status": "unavailable",
        "reason": f"{backend} CUDA runtime is not installed in this environment",
        "timing": None,
        "selected_parameters": None,
        "scores": None,
        "convergence": None,
        "repeat_samples": [],
    }


def _source() -> dict:
    cases = []
    for model, task in MODELS:
        cases.append(
            {
                "case_id": f"cv-{model.lower().replace('_', '-')}",
                "model_id": model,
                "task": task,
                "dataset": {
                    "generator": "synthetic-contract-fixture",
                    "n_samples": 60,
                    "n_features": 8,
                    "n_test_samples": 20,
                    "parameters": {"seed": 42},
                },
                "cv": {
                    "fold_count": 3,
                    "split_strategy": "deterministic-kfold",
                    "shuffle": True,
                    "subject_preserving": task == "survival",
                },
                "grid": {
                    "candidate_count": 3,
                    "identity": "sha256:fixture-grid",
                    "parameters": {"alpha": [1.0, 0.1, 0.01]},
                },
                "scoring": {
                    "name": "held-out-objective",
                    "direction": "minimize",
                    "normalization": "per-validation-observation",
                },
                "runs": [
                    _success("statgpu", "numpy"),
                    _unavailable("cupy"),
                    _unavailable("torch"),
                    _success("sklearn", None),
                ],
            }
        )
    return {
        "source_schema_version": "1.0",
        "source_date": "2026-08-07",
        "generated_at": "2026-08-07T00:00:00Z",
        "git_sha": "69a3237ac471b55e5614527b9d522dd5ec77b847",
        "environment": {
            "env_id": "contract-fixture-cpu",
            "host": "github-actions",
            "cpu": "x86_64",
            "gpu": None,
            "python": "3.11",
            "packages": {"statgpu": "0.2.4", "numpy": "2.x"},
            "available_backends": ["numpy"],
        },
        "protocol": {
            "seeds": [42],
            "warmup": 0,
            "repeats": 1,
            "dtype": "float64",
            "synchronization": "CPU wall-clock; CUDA backends explicitly unavailable",
            "timing_scope": "cross-validation evaluation plus final full-data refit",
            "transfer_policy": "host-resident input; no device transfer measured",
            "failure_policy": "retain_explicit_disposition",
        },
        "cases": cases,
    }


def test_valid_source_contract() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    assert validate_cv_source(_source()) == []


def test_pre_june_source_is_not_canonical() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    source = _source()
    source["source_date"] = "2026-04-09"
    errors = validate_cv_source(source)
    assert any("predates canonical minimum" in error for error in errors)


def test_all_statgpu_backends_require_explicit_disposition() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    source = _source()
    source["cases"][0]["runs"] = [
        run for run in source["cases"][0]["runs"] if run["backend"] != "torch"
    ]
    errors = validate_cv_source(source)
    assert any("missing explicit statgpu backend disposition for torch" in error for error in errors)


def test_gpu_success_cannot_be_claimed_on_cpu_environment() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    source = _source()
    source["cases"][0]["runs"][1] = _success("statgpu", "cupy")
    source["cases"][0]["runs"][1]["device"] = "cuda"
    errors = validate_cv_source(source)
    assert any("GPU success claimed without a GPU environment" in error for error in errors)
    assert any("successful backend is absent from available_backends" in error for error in errors)


def test_failed_rows_cannot_publish_partial_measurements() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    source = _source()
    broken = copy.deepcopy(source["cases"][0]["runs"][1])
    broken["timing"] = {
        "cv_evaluation_ms": 1.0,
        "final_refit_ms": 1.0,
        "total_fit_ms": 2.0,
        "peak_memory_bytes": None,
    }
    source["cases"][0]["runs"][1] = broken
    errors = validate_cv_source(source)
    assert any("must not publish measurements" in error for error in errors)


def test_non_finite_measurements_are_rejected() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    source = _source()
    source["cases"][0]["runs"][0]["scores"]["validation_score"] = float("nan")
    source["cases"][1]["runs"][0]["timing"]["total_fit_ms"] = float("inf")
    errors = validate_cv_source(source)
    assert any("validation_score: non-finite numeric value nan" in error for error in errors)
    assert any("total_fit_ms: non-finite numeric value inf" in error for error in errors)


def test_six_initial_models_are_required_exactly_once() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    source = _source()
    source["cases"].pop()
    errors = validate_cv_source(source)
    assert any("missing required CV model cases: CoxPHCV" in error for error in errors)


def test_registry_exposes_new_parser_without_removing_historical_parser() -> None:
    from dev.benchmarks.frontend_data.parsers import parse_cv_benchmark, parse_lassocv_combined
    from dev.benchmarks.frontend_data.registry import PARSER_FUNCTIONS

    assert PARSER_FUNCTIONS["cv_benchmark"] is parse_cv_benchmark
    assert PARSER_FUNCTIONS["lassocv_combined"] is parse_lassocv_combined


def test_canonical_parser_emits_normalized_cv_metrics(tmp_path: Path) -> None:
    from jsonschema import Draft202012Validator

    from dev.benchmarks.frontend_data.parsers.cv_package import parse_cv_benchmark

    source_path = tmp_path / "cv_source.json"
    source_path.write_text(json.dumps(_source()), encoding="utf-8")
    runs, models, warnings = parse_cv_benchmark(source_path, "contract-fixture-cpu")

    assert len(runs) == 24
    assert len(models) == 6
    assert warnings == []
    assert {run["model_id"] for run in runs} == {model for model, _ in MODELS}
    assert {run["backend"] for run in runs} == {"numpy", "cupy", "torch", None}
    assert all(model["supports_penalty"] is True for model in models)
    assert all(run["parameters"]["metric_scope"] == "cross_validation" for run in runs)
    assert all("cv_evaluation_ms" not in run["parameters"] for run in runs)
    assert all(run["case_id"].startswith("case-") for run in runs)
    assert all(run["method_config_id"].startswith("method-") for run in runs)

    successful = [run for run in runs if run["metrics"]["cross_validation"]["status"] == "success"]
    unavailable = [run for run in runs if run["metrics"]["cross_validation"]["status"] == "unavailable"]
    assert len(successful) == 12
    assert len(unavailable) == 12
    assert all(run["metrics"]["timing"]["fit_time_ms"] == 12.5 for run in successful)
    assert all(run["metrics"]["cross_validation"]["cv_evaluation_ms"] == 10.0 for run in successful)
    assert all(run["metrics"]["cross_validation"]["final_refit_ms"] == 2.0 for run in successful)
    assert all(run["metrics"]["cross_validation"]["selected_parameters"] == {"alpha": 0.1} for run in successful)
    assert all("timing" not in run["metrics"] for run in unavailable)
    assert all(run["metrics"]["cross_validation"]["reason"] for run in unavailable)
    assert all("cv_evaluation_ms" not in run["metrics"]["cross_validation"] for run in unavailable)

    dashboard_schema = json.loads(
        (REPO_ROOT / "dev" / "benchmarks" / "benchmark_frontend_schema.json").read_text()
    )
    cv_schema = dashboard_schema["properties"]["runs"]["items"]["properties"]["metrics"]["properties"]["cross_validation"]
    validator = Draft202012Validator(cv_schema)
    for run in runs:
        assert list(validator.iter_errors(run["metrics"]["cross_validation"])) == []


def test_repository_p100_source_is_immutable_and_registered() -> None:
    from dev.benchmarks.cv_source import validate_cv_source
    from dev.benchmarks.frontend_data.canonical import source_sha256
    from dev.benchmarks.frontend_data.parsers.cv_package import parse_cv_benchmark
    from dev.benchmarks.frontend_data.registry import load_manifest

    raw_path = REPO_ROOT / "results" / "cv_benchmark_candidate.json"
    canonical_path = REPO_ROOT / "results" / "benchmark_frontend_sources" / "cv_benchmark_20260807.json"
    expected_sha = "1347184c988d0f9648c8477d64752b646249282978cf28f65c165b391839bad2"
    assert raw_path.read_bytes() == canonical_path.read_bytes()
    assert source_sha256(raw_path) == expected_sha
    assert source_sha256(canonical_path) == expected_sha

    source = json.loads(raw_path.read_text(encoding="utf-8"))
    assert validate_cv_source(source) == []
    assert source["git_sha"] == "unknown"
    assert source["environment"]["gpu"] == "Tesla P100-SXM2-16GB"

    runs, models, warnings = parse_cv_benchmark(canonical_path, "remote-p100-cv-20260807")
    assert len(models) == 6
    assert len(runs) == 22
    assert warnings == []
    failed = [run for run in runs if run["metrics"]["cross_validation"]["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["model_id"] == "LogisticRegressionCV"
    assert failed[0]["backend"] == "torch"
    assert "CPU fallback is disabled" in failed[0]["metrics"]["cross_validation"]["reason"]
    assert "timing" not in failed[0]["metrics"]

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None
    entry = next(source for source in manifest["sources"] if source["source_id"] == "cv-benchmark-20260807-1347184c988d")
    assert entry["sha256"] == expected_sha
    assert entry["measurement_git_sha"] == "ad2cf88d1d443a53eeb5207c33c4ee4f25de2400"
    assert entry["raw_git_sha"] == "unknown"
