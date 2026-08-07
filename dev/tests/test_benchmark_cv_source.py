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


def test_six_initial_models_are_required_exactly_once() -> None:
    from dev.benchmarks.cv_source import validate_cv_source

    source = _source()
    source["cases"].pop()
    errors = validate_cv_source(source)
    assert any("missing required CV model cases: CoxPHCV" in error for error in errors)


def test_canonical_parser_emits_success_rows_and_explicit_warnings(tmp_path: Path) -> None:
    from dev.benchmarks.frontend_data.parsers.cv_package import parse_cv_benchmark

    source_path = tmp_path / "cv_source.json"
    source_path.write_text(json.dumps(_source()), encoding="utf-8")
    runs, models, warnings = parse_cv_benchmark(source_path, "contract-fixture-cpu")

    assert len(runs) == 12
    assert len(models) == 6
    assert len(warnings) == 12
    assert {run["model_id"] for run in runs} == {model for model, _ in MODELS}
    assert {run["backend"] for run in runs} == {"numpy", None}
    assert all(run["parameters"]["metric_scope"] == "cross_validation" for run in runs)
    assert all(run["parameters"]["cv_evaluation_ms"] == 10.0 for run in runs)
    assert all(run["parameters"]["final_refit_ms"] == 2.0 for run in runs)
    assert all(run["metrics"]["timing"]["fit_time_ms"] == 12.5 for run in runs)
    assert all(run["case_id"].startswith("case-") for run in runs)
    assert all(run["method_config_id"].startswith("method-") for run in runs)
