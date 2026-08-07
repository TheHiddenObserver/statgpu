from __future__ import annotations

"""Parser for the six-family canonical CV benchmark source contract."""

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from dev.benchmarks.cv_source import validate_cv_source

from ..canonical import make_scale_key, make_scale_label


_PRIMARY_CATEGORY = {
    "RidgeCV": "linear_models",
    "LassoCV": "linear_models",
    "ElasticNetCV": "linear_models",
    "LogisticRegressionCV": "linear_models",
    "PenalizedGLM_CV": "penalized_glm",
    "CoxPHCV": "survival",
}


def _digest_id(prefix: str, value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16]}"


def _category_ids(model_id: str) -> list[str]:
    primary = _PRIMARY_CATEGORY[model_id]
    if model_id in {"RidgeCV", "LassoCV", "ElasticNetCV", "LogisticRegressionCV"}:
        return ["linear_models", "penalized_glm"]
    return [primary]


def _penalty(model_id: str, case: dict[str, Any]) -> str | None:
    if "penalty" in case["grid"]["parameters"]:
        raw = case["grid"]["parameters"]["penalty"]
        return None if raw is None else str(raw)
    return {
        "RidgeCV": "l2",
        "LassoCV": "l1",
        "ElasticNetCV": "elasticnet",
        "LogisticRegressionCV": "l2",
        "PenalizedGLM_CV": None,
        "CoxPHCV": None,
    }[model_id]


def parse_cv_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse a validated canonical six-family CV source.

    Successful runs expose measured timing, selection, score, and convergence
    fields. Explicit non-success backend dispositions are retained as dashboard
    rows with status/reason metadata and without fabricated measurements.
    """
    with filepath.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    errors = validate_cv_source(data)
    if errors:
        raise ValueError(
            f"{filepath.name}: invalid canonical CV source:\n- " + "\n- ".join(errors)
        )

    source = {
        "file": filepath.name,
        "date": data["source_date"],
        "parser": "parse_cv_benchmark_v1_1",
        "parser_version": "1.1",
    }
    protocol = data["protocol"]
    runs: list[dict] = []
    warnings: list[str] = []
    models: list[dict] = []

    for case in data["cases"]:
        model_id = case["model_id"]
        categories = _category_ids(model_id)
        source_case_id = case["case_id"]
        case_id = _digest_id(
            "case",
            {
                "source_case_id": source_case_id,
                "model_id": model_id,
                "task": case["task"],
                "dataset": case["dataset"],
                "cv": case["cv"],
                "grid": case["grid"],
                "scoring": case["scoring"],
            },
        )
        dataset = case["dataset"]
        scale = {
            "scale_key": make_scale_key(dataset["n_samples"], dataset["n_features"]),
            "n_samples": dataset["n_samples"],
            "n_features": dataset["n_features"],
            "label": make_scale_label(dataset["n_samples"], dataset["n_features"]),
        }

        models.append(
            {
                "model_id": model_id,
                "primary_category_id": _PRIMARY_CATEGORY[model_id],
                "category_ids": categories,
                "supports_penalty": True,
                "supports_inference": False,
            }
        )

        for raw_run in case["runs"]:
            framework = raw_run["framework"]
            backend = raw_run["backend"]
            selected = raw_run["selected_parameters"]
            method_config_id = _digest_id(
                "method",
                {
                    "framework": framework,
                    "backend": backend,
                    "grid": case["grid"],
                    "selected_parameters": selected,
                    "device": raw_run["device"],
                },
            )
            parameters = {
                "metric_scope": "cross_validation",
                "source_case_id": source_case_id,
                "task": case["task"],
                "dataset": dataset,
                "cv": case["cv"],
                "grid": case["grid"],
                "scoring": case["scoring"],
                "protocol": protocol,
                "device": raw_run["device"],
            }
            run: dict[str, Any] = {
                "run_id": "",
                "env_id": env_id,
                "category_ids": categories,
                "model_id": model_id,
                "loss": str(case["grid"]["parameters"].get("loss", case["task"])),
                "penalty": _penalty(model_id, case),
                "solver": "cv",
                "solver_display": "Cross-validation",
                "solver_kind": "internal",
                "case_id": case_id,
                "method_config_id": method_config_id,
                "variant": case["scoring"]["name"],
                "implementation": raw_run["device"],
                "parameters": parameters,
                "framework": framework,
                "backend": backend,
                "scale": scale,
                "source": dict(source),
            }

            status = raw_run["status"]
            if status != "success":
                run["metrics"] = {
                    "cross_validation": {
                        "status": status,
                        "reason": raw_run["reason"],
                        "scoring_name": case["scoring"]["name"],
                        "scoring_direction": case["scoring"]["direction"],
                        "candidate_count": case["grid"]["candidate_count"],
                        "fold_count": case["cv"]["fold_count"],
                        "quality": "reported",
                        "source_file": filepath.name,
                    }
                }
                runs.append(run)
                continue

            repeat_samples = raw_run["repeat_samples"]
            total_samples = [sample["total_fit_ms"] for sample in repeat_samples]
            timing = raw_run["timing"]
            convergence = raw_run["convergence"]
            scores = raw_run["scores"]
            assert timing is not None
            assert selected is not None
            assert convergence is not None
            assert scores is not None

            std_ms = statistics.pstdev(total_samples) if len(total_samples) > 1 else 0.0
            converged_rate = 1.0 if convergence["final_refit_converged"] else 0.0
            validation_status = (
                "pass"
                if convergence["failed_candidates"] == 0
                and convergence["failed_folds"] == 0
                and convergence["final_refit_converged"]
                else "warn"
            )

            run["replicate"] = {
                "n_runs": len(repeat_samples),
                "seed_count": len({sample["seed"] for sample in repeat_samples}),
                "n_failed": convergence["failed_candidates"] + convergence["failed_folds"],
            }
            run["metrics"] = {
                "timing": {
                    "fit_time_ms": timing["total_fit_ms"],
                    "std_ms": std_ms,
                    "min_ms": min(total_samples),
                    "max_ms": max(total_samples),
                    "sample_count": len(total_samples),
                    "std_ddof": 0,
                    "std_scope": "replicates",
                    "quality": "measured",
                    "source_file": filepath.name,
                },
                "cross_validation": {
                    "status": "success",
                    "reason": None,
                    "cv_evaluation_ms": timing["cv_evaluation_ms"],
                    "final_refit_ms": timing["final_refit_ms"],
                    "total_fit_ms": timing["total_fit_ms"],
                    "selected_parameters": selected,
                    "validation_score": scores["validation_score"],
                    "final_score": scores["final_score"],
                    "scoring_name": case["scoring"]["name"],
                    "scoring_direction": case["scoring"]["direction"],
                    "candidate_count": convergence["candidate_count"],
                    "fold_count": convergence["fold_count"],
                    "failed_candidates": convergence["failed_candidates"],
                    "failed_folds": convergence["failed_folds"],
                    "final_refit_converged": convergence["final_refit_converged"],
                    "quality": "measured",
                    "source_file": filepath.name,
                },
                "convergence": {
                    "n_iter_mean": convergence.get("n_iter", 0) or 0,
                    "converged_rate": converged_rate,
                    "quality": "reported",
                    "source_file": filepath.name,
                },
                "validation": {
                    "status": validation_status,
                    "checks": [
                        {
                            "metric": "validation_score",
                            "status": "pass",
                            "value": scores["validation_score"],
                            "reference": case["scoring"]["name"],
                        },
                        {
                            "metric": "final_score",
                            "status": "pass",
                            "value": scores["final_score"],
                            "reference": case["scoring"]["name"],
                        },
                    ],
                    "quality": "measured",
                    "source_file": filepath.name,
                },
            }
            runs.append(run)

    return runs, models, warnings
