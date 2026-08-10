from __future__ import annotations
"""Canonical parsers for PR #126 Panel Stage-C P100 evidence."""

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from ..canonical import make_scale_key, make_scale_label

_SOURCE_DATE = "2026-08-10"
_MEASUREMENT_SHA = "aad53587c9611da0e71a676e86ef32d9f6403f5c"
_VALIDATION_PARSER = "parse_panel_stage_c_physical_validation_v1"
_PERFORMANCE_PARSER = "parse_panel_stage_c_performance_v1"
_PARSER_VERSION = "1.0"

_EXPECTED_CASES = {
    "pooled_hc0", "pooled_hc2", "pooled_hc3",
    "pooled_cluster_one_way", "pooled_cluster_two_way_group_debias",
    "pooled_dk_bartlett", "pooled_dk_qs", "pooled_legacy_hac",
    "panel_entity_hc0", "panel_entity_hc2", "panel_entity_hc3",
    "panel_two_way_hc3", "panel_two_way_cluster_group_debias", "panel_two_way_dk",
    "random_effects_explicit_constant_robust", "random_effects_explicit_constant_hc0",
    "random_effects_explicit_constant_hc2", "random_effects_explicit_constant_hc3",
    "random_effects_cluster_two_way", "random_effects_dk",
    "between_hc0", "between_hc2", "between_hc3",
    "first_difference_hc0", "first_difference_hc2", "first_difference_hc3",
}
_EXPECTED_PRIMITIVES = {
    "cluster_group_debias", "driscoll_kraay_qs",
    "ill_conditioned_hc0", "ill_conditioned_hc2",
    "ill_conditioned_hc3", "ill_conditioned_dk",
}
_BASE_CASES = {
    "pooled_nonrobust", "pooled_hc3", "pooled_cluster_two_way", "pooled_dk_qs",
    "panel_entity_nonrobust", "panel_entity_hc3", "panel_entity_dk",
    "random_effects_nonrobust", "random_effects_hc3",
}
_BASE_SCALES = {(10000, 2, 20), (100000, 2, 20), (100000, 10, 20)}
_HIGH_T_CASES = {"pooled_dk_qs", "panel_entity_dk_qs"}


def _stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{kind}-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _model_id(case: str) -> str:
    if case.startswith("pooled_"):
        return "PooledOLS"
    if case.startswith("panel_"):
        return "PanelOLS"
    if case.startswith("random_effects_"):
        return "RandomEffects"
    if case.startswith("between_"):
        return "BetweenOLS"
    if case.startswith("first_difference_"):
        return "FirstDifferenceOLS"
    raise ValueError(f"unknown Stage-C case identity: {case!r}")


def _scale(
    n_samples: int,
    n_features: int,
    *,
    suffix: str | None = None,
    key_suffix: str | None = None,
) -> dict[str, Any]:
    label = make_scale_label(int(n_samples), int(n_features))
    if suffix:
        label = f"{label} · {suffix}"
    scale_key = make_scale_key(int(n_samples), int(n_features))
    if key_suffix:
        scale_key = f"{scale_key}_{key_suffix}"
    return {
        "scale_key": scale_key,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "label": label,
    }


def _models(model_ids: set[str]) -> list[dict]:
    return [
        {
            "model_id": model_id,
            "primary_category_id": "panel",
            "category_ids": ["panel"],
            "supports_penalty": False,
            "supports_inference": True,
        }
        for model_id in sorted(model_ids)
    ]


def _source(filepath: Path, parser: str) -> dict[str, str]:
    return {
        "file": filepath.name,
        "date": _SOURCE_DATE,
        "parser": parser,
        "parser_version": _PARSER_VERSION,
    }


def _validation(ok: bool, filepath: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pass" if ok else "fail",
        "checks": checks,
        "quality": "reported",
        "source_file": filepath.name,
    }


def _bool_check(metric: str, ok: bool, **extra: Any) -> dict[str, Any]:
    return {"metric": metric, "status": "pass" if ok else "fail", **extra}


def _finite_diff_map(value: Any) -> bool:
    """Validate stored difference diagnostics; runner success owns rtol+atol parity."""
    if not isinstance(value, dict) or not value:
        return False
    for item in value.values():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        item = float(item)
        if not math.isfinite(item) or item < 0.0:
            return False
    return True


def parse_panel_stage_c_physical_validation(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    warnings: list[str] = []
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("PR126 Stage-C validation source requires schema_version=1")
    if data.get("git_sha") != _MEASUREMENT_SHA:
        raise ValueError("PR126 Stage-C validation source measurement SHA drifted")
    if int(data.get("case_count_per_backend", -1)) != len(_EXPECTED_CASES):
        raise ValueError("PR126 Stage-C validation estimator case count drifted")
    if int(data.get("public_primitive_count_per_backend", -1)) != len(_EXPECTED_PRIMITIVES):
        raise ValueError("PR126 Stage-C public primitive count drifted")

    dataset = data.get("dataset", {})
    scale = _scale(dataset.get("nobs", 0), dataset.get("n_features", 0))
    source_ok = data.get("status") == "success" and data.get("working_tree_clean") is True
    runs: list[dict] = []
    model_ids: set[str] = set()

    backends = data.get("backends", {})
    if set(backends) != {"cupy", "torch"}:
        raise ValueError("PR126 Stage-C validation requires exactly CuPy and Torch backends")

    for backend in ("cupy", "torch"):
        result = backends[backend]
        cases = result.get("cases", {})
        primitives = result.get("public_primitives", {})
        if set(cases) != _EXPECTED_CASES:
            raise ValueError(f"{backend}: PR126 estimator case identity drifted")
        if set(primitives) != _EXPECTED_PRIMITIVES:
            raise ValueError(f"{backend}: PR126 public primitive identity drifted")
        backend_ok = (
            result.get("status") == "success"
            and result.get("requested_backend") == backend
        )

        for case_name in sorted(_EXPECTED_CASES):
            case = cases[case_name]
            diff_ok = _finite_diff_map(case.get("max_abs_differences"))
            executed_ok = case.get("executed_backend") == backend
            case_ok = case.get("status") == "success"
            ok = source_ok and backend_ok and case_ok and executed_ok and diff_ok
            checks = [
                _bool_check("source_status_success", source_ok),
                _bool_check("backend_status_success", backend_ok),
                _bool_check("case_status_success", case_ok),
                _bool_check("executed_backend_matches_requested", executed_ok),
                _bool_check("recorded_numpy_difference_finite", diff_ok),
            ]
            model_id = _model_id(case_name)
            model_ids.add(model_id)
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-validation",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": model_id,
                    "case_id": _stable_id("case", "stage-c", case_name, scale["scale_key"]),
                    "method_config_id": _stable_id("method", "stage-c-physical", case_name),
                    "variant": case_name.replace("_", "-"),
                    "penalty": None,
                    "solver": "physical_validation",
                    "solver_display": "Physical validation",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": dict(scale),
                    "parameters": {
                        "metric_scope": "physical_validation",
                        "measurement_git_sha": data.get("git_sha"),
                        "working_tree_clean": bool(data.get("working_tree_clean")),
                        "executed_backend": case.get("executed_backend"),
                        "covariance_metadata": case.get("covariance_metadata", {}),
                    },
                    "source": _source(filepath, _VALIDATION_PARSER),
                    "metrics": {
                        "validation": _validation(ok, filepath, checks),
                        "inference": {
                            "ok": ok,
                            "quality": "reported",
                            "source_file": filepath.name,
                        },
                    },
                }
            )

        for primitive_name in sorted(_EXPECTED_PRIMITIVES):
            item = primitives[primitive_name]
            try:
                diff = float(item.get("max_abs_difference"))
                diff_ok = math.isfinite(diff) and diff >= 0.0
            except (TypeError, ValueError):
                diff_ok = False
            executed_ok = item.get("executed_backend") == backend
            item_ok = item.get("status") == "success"
            ok = source_ok and backend_ok and item_ok and executed_ok and diff_ok
            checks = [
                _bool_check("source_status_success", source_ok),
                _bool_check("backend_status_success", backend_ok),
                _bool_check("primitive_status_success", item_ok),
                _bool_check("executed_backend_matches_requested", executed_ok),
                _bool_check("recorded_numpy_difference_finite", diff_ok),
            ]
            model_ids.add("PanelCovariancePrimitive")
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-validation",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": "PanelCovariancePrimitive",
                    "case_id": _stable_id("case", "stage-c-primitive", primitive_name, scale["scale_key"]),
                    "method_config_id": _stable_id("method", "stage-c-public-primitive", primitive_name),
                    "variant": f"public-{primitive_name.replace('_', '-')}",
                    "penalty": None,
                    "solver": "physical_validation",
                    "solver_display": "Physical validation",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": dict(scale),
                    "parameters": {
                        "metric_scope": "public_primitive_physical_validation",
                        "measurement_git_sha": data.get("git_sha"),
                        "working_tree_clean": bool(data.get("working_tree_clean")),
                        "executed_backend": item.get("executed_backend"),
                    },
                    "source": _source(filepath, _VALIDATION_PARSER),
                    "metrics": {"validation": _validation(ok, filepath, checks)},
                }
            )

    if len(runs) != 64:
        raise ValueError(f"PR126 validation parser expected 64 rows, got {len(runs)}")
    return runs, _models(model_ids), warnings


def parse_panel_stage_c_performance(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 2:
        raise ValueError("PR126 Stage-C performance source requires schema_version=2")
    if data.get("git_sha") != _MEASUREMENT_SHA:
        raise ValueError("PR126 Stage-C performance source measurement SHA drifted")
    if data.get("working_tree_clean") is not True:
        raise ValueError("PR126 Stage-C performance source requires a clean measurement tree")
    if data.get("benchmark") != "panel_stage_c_covariance_fit_overhead":
        raise ValueError("PR126 Stage-C performance benchmark identity drifted")
    if data.get("timing_scope") != "synchronized end-to-end estimator fit":
        raise ValueError("PR126 Stage-C performance timing scope drifted")
    if data.get("high_t_scale") != "10000x2x200":
        raise ValueError("PR126 Stage-C high-T scale drifted")

    rows = data.get("rows", [])
    if len(rows) != 58:
        raise ValueError(f"PR126 Stage-C performance requires 58 rows, got {len(rows)}")
    if {row.get("backend") for row in rows} != {"cupy", "torch"}:
        raise ValueError("PR126 Stage-C performance requires CuPy and Torch rows")

    base_rows = [row for row in rows if row.get("scenario") == "base"]
    expected_base = {
        (backend, case_name, n_samples, n_features, n_times)
        for backend in ("cupy", "torch")
        for case_name in _BASE_CASES
        for n_samples, n_features, n_times in _BASE_SCALES
    }
    actual_base = [
        (
            row.get("backend"),
            row.get("case"),
            int(row.get("n_samples", 0)),
            int(row.get("n_features", 0)),
            int(row.get("n_times", 0)),
        )
        for row in base_rows
    ]
    if len(actual_base) != len(expected_base) or set(actual_base) != expected_base:
        raise ValueError("PR126 Stage-C performance base matrix drifted")

    high_t = [row for row in rows if row.get("scenario") == "high_t_qs"]
    expected_high_t = {
        (backend, case_name, 10000, 2, 200)
        for backend in ("cupy", "torch")
        for case_name in _HIGH_T_CASES
    }
    actual_high_t = [
        (
            row.get("backend"),
            row.get("case"),
            int(row.get("n_samples", 0)),
            int(row.get("n_features", 0)),
            int(row.get("n_times", 0)),
        )
        for row in high_t
    ]
    if len(actual_high_t) != len(expected_high_t) or set(actual_high_t) != expected_high_t:
        raise ValueError("PR126 Stage-C performance high-T QS matrix drifted")

    output: list[dict] = []
    model_ids: set[str] = set()
    for row in rows:
        backend = row.get("backend")
        case_name = str(row.get("case"))
        scenario = str(row.get("scenario"))
        if scenario not in {"base", "high_t_qs"}:
            raise ValueError(f"unknown PR126 Stage-C performance scenario: {scenario!r}")
        repeats = int(row.get("repeats", 0))
        samples = row.get("samples_seconds")
        if repeats <= 0 or not isinstance(samples, list) or len(samples) != repeats:
            raise ValueError("PR126 Stage-C timing samples/repeats contract failed")
        numeric_samples = [float(value) for value in samples]
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric_samples):
            raise ValueError("PR126 Stage-C timing samples must be finite and positive")
        median = float(row.get("median_seconds"))
        if not math.isfinite(median) or median <= 0.0:
            raise ValueError("PR126 Stage-C timing median must be finite and positive")
        expected_median = float(statistics.median(numeric_samples))
        if not math.isclose(median, expected_median, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("PR126 Stage-C reported median does not match raw samples")

        model_id = _model_id(case_name)
        model_ids.add(model_id)
        n_samples = int(row["n_samples"])
        n_features = int(row["n_features"])
        n_times = int(row["n_times"])
        scale = _scale(
            n_samples,
            n_features,
            suffix=f"T={n_times}",
            key_suffix=f"t{n_times}",
        )
        output.append(
            {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-performance",
                "env_id": env_id,
                "category_ids": ["panel"],
                "model_id": model_id,
                "case_id": _stable_id(
                    "case", "stage-c-performance", case_name, scenario,
                    n_samples, n_features, n_times,
                ),
                "method_config_id": _stable_id("method", "stage-c-performance", case_name),
                "variant": f"{case_name.replace('_', '-')}-{scenario.replace('_', '-')}",
                "penalty": None,
                "solver": "covariance_fit",
                "solver_display": "Covariance fit",
                "solver_kind": "internal",
                "framework": "statgpu",
                "backend": backend,
                "scale": scale,
                "parameters": {
                    "scenario": scenario,
                    "n_times": n_times,
                    "repeats": repeats,
                    "timing_scope": data.get("timing_scope"),
                    "input_residency": data.get("input_residency"),
                    "measurement_git_sha": data.get("git_sha"),
                    "working_tree_clean": True,
                },
                "source": _source(filepath, _PERFORMANCE_PARSER),
                "metrics": {
                    "timing": {
                        "fit_time_ms": round(median * 1000.0, 6),
                        "quality": "measured",
                        "source_file": filepath.name,
                    },
                    "validation": {
                        "status": "pass",
                        "checks": [
                            {"metric": "synchronized_timing", "status": "pass"},
                            {"metric": "raw_samples_finite_positive", "status": "pass"},
                            {"metric": "median_matches_raw_samples", "status": "pass"},
                        ],
                        "quality": "reported",
                        "source_file": filepath.name,
                    },
                },
            }
        )

    return output, _models(model_ids), []
