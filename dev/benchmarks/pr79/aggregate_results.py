#!/usr/bin/env python3
"""Validate raw PR79 evidence and emit the only canonical Gate object."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dev.benchmarks.pr79.validators.numerical import (
    NumericalValidationError,
    bse_rel_error,
    coef_max_abs_error,
    coef_rel_l2_error,
    covariance_rel_fro_error,
    objective_rel_error,
    prediction_rel_error,
    validate_run_final_state,
)


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "configs" / "expected_accuracy_manifest.json"
ALLOWED_METRICS = {
    "final_state_contract",
    "coef_max_abs_error",
    "coef_rel_l2_error",
    "prediction_rel_error",
    "bse_rel_error",
    "covariance_rel_fro_error",
    "loglik_rel_error",
}
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class AggregationError(RuntimeError):
    """A hard Gate failure, optionally carrying the failed canonical object."""

    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.report = report


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard/non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot load strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"JSON root must be an object: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _git_snapshot() -> Dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_project_root,
            text=True,
            timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=_project_root,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "git_sha": "unknown",
            "worktree_clean": False,
            "dirty_entries": [],
            "inspection_error": f"{type(exc).__name__}: {exc}",
        }
    dirty_entries = [line for line in status.splitlines() if line]
    return {
        "git_sha": sha,
        "worktree_clean": not dirty_entries,
        "dirty_entries": dirty_entries,
        "inspection_error": None,
    }


def _provenance_noncanonical_reasons(provenance: Any) -> List[str]:
    if not isinstance(provenance, Mapping):
        return ["raw repository_provenance is missing"]
    reasons: List[str] = []
    if provenance.get("schema_version") != "pr79-repository-provenance-1.0":
        reasons.append("unsupported raw repository provenance schema")
    if provenance.get("allow_dirty_requested") is not False:
        reasons.append("raw evidence was collected with --allow-dirty")
    if provenance.get("sha_unchanged_during_collection") is not True:
        reasons.append("raw HEAD was not stable during collection")
    if provenance.get("canonical_eligible") is not True:
        reasons.append("raw evidence is marked non-canonical")
    for phase in ("initial", "final"):
        snapshot = provenance.get(phase)
        if not isinstance(snapshot, Mapping):
            reasons.append(f"raw {phase} Git snapshot is missing")
            continue
        if snapshot.get("worktree_clean") is not True:
            reasons.append(f"raw repository was not clean at {phase} snapshot")
        if snapshot.get("inspection_error") is not None:
            reasons.append(f"raw {phase} Git inspection failed")
        if snapshot.get("dirty_entries") != []:
            reasons.append(f"raw {phase} Git snapshot contains dirty entries")
    return reasons


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregationError(message)


def _finite_number(value: Any, name: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AggregationError(f"{name} must be finite")
    if non_negative and number < 0.0:
        raise AggregationError(f"{name} must be non-negative")
    return number


def _assert_finite_tree(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float, np.number)):
        if not math.isfinite(float(value)):
            raise AggregationError(f"non-finite numerical evidence at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
        return
    raise AggregationError(f"unsupported evidence value at {path}: {type(value).__name__}")


def _configuration(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    _require(
        manifest.get("manifest_schema_version") == "pr79-accuracy-manifest-1.0",
        "unsupported accuracy manifest schema",
    )
    configurations = manifest.get("configurations")
    _require(isinstance(configurations, Mapping), "manifest configurations are missing")
    _require(name in configurations, f"manifest configuration {name!r} is missing")
    config = configurations[name]
    _require(isinstance(config, Mapping), f"manifest configuration {name!r} is invalid")
    return config


def expected_runs(
    manifest: Mapping[str, Any], config_name: str
) -> List[Dict[str, Any]]:
    config = _configuration(manifest, config_name)
    explicit = config.get("expected_runs")
    if explicit is not None:
        _require(isinstance(explicit, list), "expected_runs must be a list")
        runs = [dict(item) for item in explicit]
    else:
        cases = manifest.get("cases", {})
        runs = []
        for label in config.get("cases", []):
            case = cases.get(label)
            _require(isinstance(case, Mapping), f"manifest case {label!r} is missing")
            for backend in config.get("backends", []):
                for iteration in range(int(config.get("iterations", 0))):
                    runs.append({
                        "run_key": f"{label}-{backend}-{iteration}",
                        "case_id": case.get("case_id"),
                        "case_label": label,
                        "backend": backend,
                        "model_id": case.get("model_id"),
                    })
    seen = set()
    for item in runs:
        for field in ("run_key", "case_id", "case_label", "backend", "model_id"):
            _require(item.get(field) not in (None, ""), f"expected run is missing {field}")
        _require(item["run_key"] not in seen, f"duplicate expected run_key: {item['run_key']}")
        seen.add(item["run_key"])
    return runs


def _template_cases(
    manifest: Mapping[str, Any], config: Mapping[str, Any], scope: Any
) -> List[str]:
    configured = list(config.get("cases", []))
    cases = manifest.get("cases", {})
    if scope == "all":
        return configured
    if scope == "full_rank":
        return [label for label in configured if not cases[label].get("rank_deficient")]
    if scope == "rank_deficient":
        return [label for label in configured if cases[label].get("rank_deficient")]
    if scope == "cox":
        return [label for label in configured if cases[label].get("model_id") == "CoxPH"]
    _require(isinstance(scope, list), "check template cases scope is invalid")
    unknown = sorted(set(scope) - set(configured))
    _require(not unknown, "check template references unconfigured cases: " + ", ".join(unknown))
    return list(scope)


def expected_checks(
    manifest: Mapping[str, Any], config_name: str
) -> List[Dict[str, Any]]:
    config = _configuration(manifest, config_name)
    explicit = config.get("expected_checks")
    if explicit is not None:
        _require(isinstance(explicit, list), "expected_checks must be a list")
        checks = [dict(item) for item in explicit]
    else:
        checks = []
        cases = manifest.get("cases", {})
        for template in config.get("check_templates", []):
            _require(isinstance(template, Mapping), "check template must be an object")
            labels = _template_cases(manifest, config, template.get("cases"))
            backends = (
                list(config.get("backends", []))
                if template.get("backends") == "all"
                else list(template.get("backends", []))
            )
            for label in labels:
                case = cases[label]
                for backend in backends:
                    for iteration in range(int(config.get("iterations", 0))):
                        run_key = f"{label}-{backend}-{iteration}"
                        reference_key = f"{label}-numpy-{iteration}"
                        item = {
                            "check_id": f"{template.get('template_id')}-{run_key}",
                            "run_key": run_key,
                            "reference_run_key": reference_key,
                            "case_id": case.get("case_id"),
                            "case_label": label,
                            "backend": backend,
                            "reference_backend": "numpy",
                            "metric": template.get("metric"),
                            "expected_class": template.get("expected_class"),
                            "threshold": template.get("threshold"),
                        }
                        if item["expected_class"] == "not_comparable":
                            reason_field = template.get("reason_field")
                            comparable_field = template.get("still_comparable_field")
                            item["reason"] = case.get(reason_field) if reason_field else None
                            item["still_comparable"] = (
                                case.get(comparable_field) if comparable_field else None
                            )
                        checks.append(item)

    allowed = set(manifest.get("allowed_classifications", []))
    _require(allowed, "manifest allowed_classifications are missing")
    seen = set()
    for item in checks:
        required = (
            "check_id",
            "run_key",
            "reference_run_key",
            "case_id",
            "backend",
            "reference_backend",
            "metric",
            "expected_class",
            "threshold",
        )
        missing = [field for field in required if field not in item]
        _require(not missing, "expected check is missing: " + ", ".join(missing))
        _require(item["check_id"] not in seen, f"duplicate check_id: {item['check_id']}")
        seen.add(item["check_id"])
        _require(
            item["expected_class"] in allowed,
            f"unknown classification: {item['expected_class']!r}",
        )
        _require(item["metric"] in ALLOWED_METRICS, f"unknown metric: {item['metric']!r}")
        _finite_number(item["threshold"], f"threshold for {item['check_id']}", non_negative=True)
        if item["expected_class"] == "not_comparable":
            _require(bool(item.get("reason")), f"{item['check_id']} lacks exclusion reason")
            _require(
                isinstance(item.get("still_comparable"), list) and item["still_comparable"],
                f"{item['check_id']} lacks still-comparable quantities",
            )
    return checks


def _validate_raw_schema(
    raw: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config_name: str,
    expected_sha: str,
) -> Dict[str, Dict[str, Any]]:
    _require(
        raw.get("source_schema_version") == "pr79-benchmark-source-2.1",
        "unsupported raw source schema",
    )
    _require(raw.get("configuration") == config_name, "raw configuration mismatch")
    _require(
        isinstance(raw.get("benchmark_session_id"), str)
        and bool(raw["benchmark_session_id"]),
        "raw benchmark_session_id is missing",
    )
    raw_sha = raw.get("git_sha")
    _require(isinstance(raw_sha, str) and SHA_PATTERN.match(raw_sha) is not None, "raw git_sha is invalid")
    _require(SHA_PATTERN.match(expected_sha) is not None, "expected git SHA is invalid")
    _require(raw_sha == expected_sha, f"validated SHA mismatch: raw {raw_sha}, expected {expected_sha}")
    provenance = raw.get("repository_provenance")
    _require(
        isinstance(provenance, Mapping),
        "raw repository_provenance is missing",
    )
    for phase in ("initial", "final"):
        snapshot = provenance.get(phase)
        if isinstance(snapshot, Mapping):
            _require(
                snapshot.get("git_sha") == raw_sha,
                f"raw {phase} provenance SHA mismatch",
            )
    _require(isinstance(raw.get("environment"), Mapping), "raw environment is missing")
    _require(isinstance(raw.get("cases"), Mapping), "raw cases are missing")
    _require(isinstance(raw.get("runs"), list), "raw runs must be a list")

    expected = expected_runs(manifest, config_name)
    config = _configuration(manifest, config_name)
    _require(
        raw.get("selected_backends") == list(config.get("backends", [])),
        "raw backend selection is incomplete or out of manifest order",
    )
    expected_case_ids = {item["case_id"] for item in expected}
    actual_case_ids = set(raw["cases"])
    _require(
        actual_case_ids == expected_case_ids,
        f"raw case completeness mismatch: missing={sorted(expected_case_ids - actual_case_ids)}, "
        f"unexpected={sorted(actual_case_ids - expected_case_ids)}",
    )
    for case_id, case in raw["cases"].items():
        _require(isinstance(case, Mapping), f"case {case_id} is not an object")
        _require(case.get("case_id") == case_id, f"case {case_id} identity mismatch")
        _require(isinstance(case.get("model_id"), str), f"case {case_id} model_id is missing")
        _require(isinstance(case.get("inputs"), Mapping), f"case {case_id} inputs are missing")
        _assert_finite_tree(case["inputs"], f"cases.{case_id}.inputs")

    by_key: Dict[str, Dict[str, Any]] = {}
    for index, run in enumerate(raw["runs"]):
        _require(isinstance(run, dict), f"raw run {index} is not an object")
        required_fields = {
            "run_key",
            "case_id",
            "method_config_id",
            "model_id",
            "framework",
            "backend",
            "parameters",
            "status",
            "timing",
            "results",
            "resources",
            "error",
        }
        missing_fields = sorted(required_fields - set(run))
        _require(
            not missing_fields,
            f"raw run {index} schema missing: {', '.join(missing_fields)}",
        )
        run_key = run.get("run_key")
        _require(isinstance(run_key, str) and run_key, f"raw run {index} lacks run_key")
        _require(run_key not in by_key, f"duplicate raw run_key: {run_key}")
        by_key[run_key] = run

    expected_by_key = {item["run_key"]: item for item in expected}
    actual_keys = set(by_key)
    expected_keys = set(expected_by_key)
    _require(
        actual_keys == expected_keys,
        f"raw run completeness mismatch: missing={sorted(expected_keys - actual_keys)}, "
        f"unexpected={sorted(actual_keys - expected_keys)}",
    )

    for run_key, expected_run in expected_by_key.items():
        run = by_key[run_key]
        for field in ("case_id", "backend", "model_id"):
            _require(
                run.get(field) == expected_run[field],
                f"{run_key} {field} mismatch: {run.get(field)!r} != {expected_run[field]!r}",
            )
        _require(isinstance(run.get("parameters"), Mapping), f"{run_key} parameters are missing")
        _require(
            isinstance(run.get("method_config_id"), str) and bool(run["method_config_id"]),
            f"{run_key} method_config_id is missing",
        )
        _require(run.get("framework") == "statgpu", f"{run_key} framework mismatch")
        _require(isinstance(run.get("resources"), Mapping), f"{run_key} resources are invalid")
        _require(run["parameters"].get("backend") == run["backend"], f"{run_key} backend parameter mismatch")
        _require(run.get("status") == "success", f"{run_key} status is not success")
        _require(run.get("error") is None, f"{run_key} success record contains an error")
        _require(isinstance(run.get("timing"), Mapping), f"{run_key} timing is missing")
        _finite_number(run["timing"].get("fit_warm_s"), f"{run_key} fit timing", non_negative=True)
        results = run.get("results")
        _require(isinstance(results, Mapping), f"{run_key} results are missing")
        common_required = ("coef_", "predictions")
        model_required = {
            "CoxPH": (
                "_log_likelihood",
                "_penalized_objective",
                "_final_kkt_inf",
                "_final_kkt_normalized",
                "_var_matrix",
                "_bse",
            ),
            "LinearRegression": ("residual_sum_squares",),
            "PooledOLS": ("residual_sum_squares",),
        }
        _require(run["model_id"] in model_required, f"{run_key} has unsupported model_id")
        missing = [
            field
            for field in common_required + model_required[run["model_id"]]
            if field not in results or results[field] is None
        ]
        _require(not missing, f"{run_key} result schema missing: {', '.join(missing)}")
        _assert_finite_tree(results, f"runs.{run_key}.results")
    return by_key


def _metric_value(
    metric: str, run: Mapping[str, Any], reference: Mapping[str, Any]
) -> float:
    actual = run["results"]
    expected = reference["results"]
    if metric == "coef_max_abs_error":
        return coef_max_abs_error(actual["coef_"], expected["coef_"])
    if metric == "coef_rel_l2_error":
        return coef_rel_l2_error(actual["coef_"], expected["coef_"])
    if metric == "prediction_rel_error":
        return prediction_rel_error(actual["predictions"], expected["predictions"])
    if metric == "bse_rel_error":
        return bse_rel_error(actual["_bse"], expected["_bse"])
    if metric == "covariance_rel_fro_error":
        return covariance_rel_fro_error(actual["_var_matrix"], expected["_var_matrix"])
    if metric == "loglik_rel_error":
        return objective_rel_error(actual["_log_likelihood"], expected["_log_likelihood"])
    raise NumericalValidationError(f"unsupported comparison metric: {metric}")


def _evaluate_checks(
    definitions: Iterable[Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, Any]],
    cases: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for definition in definitions:
        run = runs[definition["run_key"]]
        reference = runs[definition["reference_run_key"]]
        threshold = float(definition["threshold"])
        try:
            if definition["metric"] == "final_state_contract":
                validation = validate_run_final_state(
                    run, cases[definition["case_id"]], threshold
                )
                passed = bool(validation["passed"])
                value = max(
                    (float(check["value"]) for check in validation["checks"]),
                    default=0.0,
                )
                details: Any = validation
            else:
                value = float(_metric_value(definition["metric"], run, reference))
                passed = bool(value <= threshold)
                details = None
            if not math.isfinite(value):
                raise NumericalValidationError("metric value is NaN or Inf")
            record = {
                "check_id": definition["check_id"],
                "case_id": definition["case_id"],
                "run_key": definition["run_key"],
                "backend": definition["backend"],
                "reference_backend": definition["reference_backend"],
                "metric": definition["metric"],
                "classification": definition["expected_class"],
                "threshold": threshold,
                "value": value,
                "status": "pass" if passed else "fail",
                "passed": passed,
            }
            if details is not None:
                record["details"] = details
            if definition["expected_class"] == "not_comparable":
                record["reason"] = definition["reason"]
                record["still_comparable"] = definition["still_comparable"]
                record["projected_or_estimable_space_passed"] = passed
            records.append(record)
        except (KeyError, TypeError, ValueError, NumericalValidationError, np.linalg.LinAlgError) as exc:
            records.append({
                "check_id": definition["check_id"],
                "case_id": definition["case_id"],
                "run_key": definition["run_key"],
                "backend": definition["backend"],
                "reference_backend": definition["reference_backend"],
                "metric": definition["metric"],
                "classification": definition["expected_class"],
                "threshold": threshold,
                "value": None,
                "status": "fail",
                "passed": False,
                "reason": f"numerical_validation_error: {exc}",
            })
    return records


def _summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def count(classification: str, passed: Optional[bool] = None) -> int:
        selected = [
            record for record in records if record["classification"] == classification
        ]
        if passed is not None:
            selected = [record for record in selected if record["passed"] is passed]
        return len(selected)

    passed = sum(1 for record in records if record["passed"])
    failed = len(records) - passed
    not_comparable = count("not_comparable")
    return {
        "total_checks": len(records),
        "passed": passed,
        "failed": failed,
        "meaningful_parity_checks": count("meaningful_parity"),
        "meaningful_parity_passed": count("meaningful_parity", True),
        "final_state_contracts": count("contract"),
        "final_state_contracts_passed": count("contract", True),
        "rank_def_non_identifiable": not_comparable,
        "rank_def_estimable_space_passed": count("not_comparable", True),
        "unresolved": failed,
        "gate_verdict": (
            "FAIL"
            if failed
            else "PASS_WITH_DOCUMENTED_NOT_COMPARABLE"
            if not_comparable
            else "PASS"
        ),
    }


def aggregate_results(
    raw: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    config_name: str,
    expected_sha: Optional[str] = None,
    allow_dirty: bool = False,
) -> Dict[str, Any]:
    validation_initial_snapshot = _git_snapshot()
    validated_sha = expected_sha or str(
        validation_initial_snapshot.get("git_sha", "unknown")
    )
    runs = _validate_raw_schema(raw, manifest, config_name, validated_sha)
    noncanonical_reasons = _provenance_noncanonical_reasons(
        raw.get("repository_provenance")
    )
    if validation_initial_snapshot.get("worktree_clean") is not True:
        noncanonical_reasons.append(
            "aggregation repository is not clean at initial snapshot"
        )
    if validation_initial_snapshot.get("inspection_error") is not None:
        noncanonical_reasons.append("initial aggregation Git inspection failed")
    if validation_initial_snapshot.get("git_sha") != validated_sha:
        noncanonical_reasons.append(
            "initial aggregation HEAD does not match validated Git SHA"
        )
    if allow_dirty:
        noncanonical_reasons.append("aggregation used --allow-dirty")
    definitions = expected_checks(manifest, config_name)
    run_keys = set(runs)
    for definition in definitions:
        _require(definition["run_key"] in run_keys, f"check run missing: {definition['run_key']}")
        _require(
            definition["reference_run_key"] in run_keys,
            f"check reference missing: {definition['reference_run_key']}",
        )
        _require(
            runs[definition["run_key"]]["case_id"] == definition["case_id"],
            f"check case_id mismatch: {definition['check_id']}",
        )
        _require(
            runs[definition["run_key"]]["backend"] == definition["backend"],
            f"check backend mismatch: {definition['check_id']}",
        )
        _require(
            runs[definition["reference_run_key"]]["backend"]
            == definition["reference_backend"],
            f"check reference backend mismatch: {definition['check_id']}",
        )
        _require(
            runs[definition["reference_run_key"]]["case_id"]
            == definition["case_id"],
            f"check reference case mismatch: {definition['check_id']}",
        )
    records = _evaluate_checks(definitions, runs, raw["cases"])
    summary = _summary(records)
    validation_final_snapshot = _git_snapshot()
    if validation_final_snapshot.get("worktree_clean") is not True:
        noncanonical_reasons.append(
            "aggregation repository is not clean at final snapshot"
        )
    if validation_final_snapshot.get("inspection_error") is not None:
        noncanonical_reasons.append("final aggregation Git inspection failed")
    if validation_final_snapshot.get("git_sha") != validated_sha:
        noncanonical_reasons.append(
            "final aggregation HEAD does not match validated Git SHA"
        )
    if (
        validation_initial_snapshot.get("git_sha")
        != validation_final_snapshot.get("git_sha")
    ):
        noncanonical_reasons.append("aggregation HEAD changed during validation")
    noncanonical_reasons = list(dict.fromkeys(noncanonical_reasons))
    if noncanonical_reasons:
        summary = dict(summary)
        summary["gate_verdict"] = "NONCANONICAL_FAIL"
    report = {
        "validated_schema_version": "pr79-validated-accuracy-1.0",
        "status": (
            "pass" if summary["failed"] == 0 and not noncanonical_reasons else "fail"
        ),
        "canonical_eligible": not noncanonical_reasons,
        "configuration": config_name,
        "validated_git_sha": validated_sha,
        "repository_provenance": {
            "raw": raw.get("repository_provenance"),
            "aggregation": {
                "schema_version": "pr79-aggregation-provenance-1.0",
                "allow_dirty_requested": bool(allow_dirty),
                "canonical_eligible": not noncanonical_reasons,
                "sha_unchanged_during_aggregation": (
                    validation_initial_snapshot.get("git_sha")
                    == validation_final_snapshot.get("git_sha")
                    and validation_initial_snapshot.get("git_sha") != "unknown"
                ),
                "initial": validation_initial_snapshot,
                "final": validation_final_snapshot,
            },
            "noncanonical_reasons": noncanonical_reasons,
        },
        "benchmark_session_id": raw.get("benchmark_session_id"),
        "manifest_sha256": canonical_sha256(manifest),
        "raw_evidence_sha256": canonical_sha256(raw),
        "environment": raw.get("environment"),
        "summary": summary,
        "checks": records,
    }
    report["canonical_sha256"] = canonical_sha256(report)
    if noncanonical_reasons:
        raise AggregationError(
            "accuracy evidence is non-canonical: "
            + "; ".join(noncanonical_reasons),
            report,
        )
    if report["status"] != "pass":
        raise AggregationError(
            f"accuracy gate failed with {summary['failed']} failed check(s)", report
        )
    return report


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", default="full")
    parser.add_argument("--expected-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "inspect non-canonical local evidence from a dirty tree; output remains "
            "status=fail and cannot be a canonical PASS"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    artifact_dir = Path("results/pr79/accuracy")
    raw_path = args.raw or artifact_dir / f"{args.config}_accuracy_results.json"
    output_path = args.output or artifact_dir / f"{args.config}_validated_results.json"
    try:
        raw = load_json_strict(raw_path)
        manifest = load_json_strict(args.manifest)
        report = aggregate_results(
            raw,
            manifest,
            config_name=args.config,
            expected_sha=args.expected_sha,
            allow_dirty=args.allow_dirty,
        )
    except AggregationError as exc:
        failed = exc.report or {
            "validated_schema_version": "pr79-validated-accuracy-1.0",
            "status": "fail",
            "configuration": args.config,
            "errors": [str(exc)],
        }
        _write_json(output_path, failed)
        print(f"PR79 accuracy aggregation failed: {exc}", file=sys.stderr)
        return 1
    _write_json(output_path, report)
    print(f"Validated {report['summary']['total_checks']} checks: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
