#!/usr/bin/env python3
"""Validation helpers for canonical cross-validation benchmark sources.

The JSON Schema enforces shape. This module enforces cross-row semantics that
cannot be expressed cleanly in JSON Schema: complete model coverage, explicit
backend dispositions, timing decomposition, finite measurements, and truthful
environment claims.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "dev" / "benchmarks" / "cv_source_schema.json"
MINIMUM_SOURCE_DATE = date(2026, 6, 1)
REQUIRED_MODELS = {
    "RidgeCV",
    "LassoCV",
    "ElasticNetCV",
    "LogisticRegressionCV",
    "PenalizedGLM_CV",
    "CoxPHCV",
}
STATGPU_BACKENDS = {"numpy", "cupy", "torch"}
SUCCESS = "success"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_against_schema(source: dict[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - dependency is installed in CI
        raise RuntimeError("jsonschema is required to validate CV sources") from exc

    schema = load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(source), key=lambda item: list(item.absolute_path))
    ]


def _successful_runs(case: dict[str, Any]) -> Iterable[dict[str, Any]]:
    return (run for run in case.get("runs", []) if run.get("status") == SUCCESS)


def _nonfinite_paths(value: Any, path: str = "<root>") -> list[str]:
    errors: list[str] = []
    if isinstance(value, bool):
        return errors
    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}: non-finite numeric value {value!r}"]
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = str(key) if path == "<root>" else f"{path}.{key}"
            errors.extend(_nonfinite_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_nonfinite_paths(child, f"{path}[{index}]"))
    return errors


def validate_cv_source(source: dict[str, Any]) -> list[str]:
    errors = validate_against_schema(source)
    if errors:
        return errors

    errors.extend(_nonfinite_paths(source))

    source_date = date.fromisoformat(source["source_date"])
    if source_date < MINIMUM_SOURCE_DATE:
        errors.append(
            f"source_date {source_date.isoformat()} predates canonical minimum "
            f"{MINIMUM_SOURCE_DATE.isoformat()}"
        )

    cases = source["cases"]
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        errors.append("case_id values must be unique")

    models = {case["model_id"] for case in cases}
    missing = sorted(REQUIRED_MODELS - models)
    if missing:
        errors.append(f"missing required CV model cases: {', '.join(missing)}")

    duplicate_models = sorted(
        model for model in REQUIRED_MODELS
        if sum(case["model_id"] == model for case in cases) != 1
    )
    if duplicate_models:
        errors.append(
            "the initial canonical package requires exactly one representative "
            f"case per model: {', '.join(duplicate_models)}"
        )

    available_backends = set(source["environment"]["available_backends"])
    gpu_declared = source["environment"]["gpu"] is not None

    for case in cases:
        prefix = f"{case['case_id']} ({case['model_id']})"
        statgpu_runs = [run for run in case["runs"] if run["framework"] == "statgpu"]
        disposition = {run["backend"] for run in statgpu_runs}
        missing_backends = sorted(STATGPU_BACKENDS - disposition)
        if missing_backends:
            errors.append(
                f"{prefix}: missing explicit statgpu backend disposition for "
                f"{', '.join(missing_backends)}"
            )

        seen_pairs: set[tuple[str, str | None]] = set()
        for run in case["runs"]:
            pair = (run["framework"], run["backend"])
            if pair in seen_pairs:
                errors.append(f"{prefix}: duplicate framework/backend disposition {pair}")
            seen_pairs.add(pair)

            status = run["status"]
            reason = run["reason"]
            if status == SUCCESS:
                if reason not in (None, ""):
                    errors.append(f"{prefix} {pair}: successful runs must not carry a failure reason")
                timing = run["timing"]
                selected = run["selected_parameters"]
                scores = run["scores"]
                convergence = run["convergence"]
                if timing is None or selected is None or scores is None or convergence is None:
                    errors.append(f"{prefix} {pair}: successful run lacks required measured outputs")
                    continue
                if not selected:
                    errors.append(f"{prefix} {pair}: selected_parameters must be non-empty")
                if timing["total_fit_ms"] < max(
                    timing["cv_evaluation_ms"], timing["final_refit_ms"]
                ):
                    errors.append(
                        f"{prefix} {pair}: total_fit_ms cannot be smaller than a component"
                    )
                if convergence["candidate_count"] != case["grid"]["candidate_count"]:
                    errors.append(f"{prefix} {pair}: candidate_count disagrees with case grid")
                if convergence["fold_count"] != case["cv"]["fold_count"]:
                    errors.append(f"{prefix} {pair}: fold_count disagrees with case CV contract")
                if not run["repeat_samples"]:
                    errors.append(f"{prefix} {pair}: successful run has no raw repeat samples")
                if run["backend"] in {"cupy", "torch"} and not gpu_declared:
                    errors.append(f"{prefix} {pair}: GPU success claimed without a GPU environment")
                if run["backend"] is not None and run["backend"] not in available_backends:
                    errors.append(
                        f"{prefix} {pair}: successful backend is absent from available_backends"
                    )
            else:
                if not reason:
                    errors.append(f"{prefix} {pair}: non-success disposition requires a reason")
                if any(
                    run[field] not in (None, [])
                    for field in ("timing", "selected_parameters", "scores", "convergence", "repeat_samples")
                ):
                    errors.append(
                        f"{prefix} {pair}: unavailable/unsupported/failed run must not publish measurements"
                    )

        if not list(_successful_runs(case)):
            errors.append(f"{prefix}: representative case has no successful implementation")

    return errors


def check_file(path: Path) -> None:
    source = load_json(path)
    errors = validate_cv_source(source)
    if errors:
        raise ValueError("invalid CV benchmark source:\n- " + "\n- ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    check_file(args.source)
    print(f"OK — canonical CV source contract: {args.source}")


if __name__ == "__main__":
    main()
