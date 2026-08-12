from __future__ import annotations
"""Canonical v4 parsers for PR #126 post-identifiability P100 evidence.

The v1 Stage-C parser remains frozen to the historical ``ec511f53`` source.
These parser identities intentionally use distinct benchmark sessions and stable
ID namespaces so the fresh ``a99726e1`` evidence can coexist with the
historical v1/v2/v3 sources without overwriting or colliding with them.
"""

import json
import math
import statistics
from pathlib import Path
from typing import Any

from .panel_stage_c import (
    _bool_check,
    _finite_diff_map,
    _model_id,
    _models,
    _scale,
    _stable_id,
    _validation,
)

_SOURCE_DATE = "2026-08-12"
_MEASUREMENT_SHA = "a99726e19c535dfcd0a94711bbc8be6aac437584"
_VALIDATION_PARSER = "parse_panel_stage_c_identifiability_physical_validation_v4"
_PERFORMANCE_PARSER = "parse_panel_stage_c_identifiability_performance_v4"
_PARSER_VERSION = "4.0"
_EXPECTED_GPU = "Tesla P100-SXM2-16GB"
_EXPECTED_CUPY_VERSION = "13.6.0"
_EXPECTED_TORCH_VERSION = "2.0.0"

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
    "panel_rank_boundary_dk",
    "panel_entity_rank_deficient_nonrobust",
    "panel_entity_rank_deficient_robust",
    "between_rank_deficient_nonrobust",
    "between_rank_deficient_robust",
    "first_difference_rank_deficient_nonrobust",
    "first_difference_rank_deficient_robust",
    "random_effects_rank_deficient_nonrobust",
    "random_effects_rank_deficient_robust",
}
_EXPECTED_PRIMITIVES = {
    "cluster_group_debias", "driscoll_kraay_qs",
    "ill_conditioned_hc0", "ill_conditioned_hc2",
    "ill_conditioned_hc3", "ill_conditioned_dk",
    "rank_boundary_nonrobust", "rank_boundary_hc0", "rank_boundary_hc2",
    "rank_boundary_hc3", "rank_boundary_cluster", "rank_boundary_dk",
}
_EXPECTED_RANK_DEFICIENT_CASES = {
    "panel_entity_rank_deficient_nonrobust",
    "panel_entity_rank_deficient_robust",
    "between_rank_deficient_nonrobust",
    "between_rank_deficient_robust",
    "first_difference_rank_deficient_nonrobust",
    "first_difference_rank_deficient_robust",
    "random_effects_rank_deficient_nonrobust",
    "random_effects_rank_deficient_robust",
}
_EXPECTED_IDENTIFIABILITY_CASES = _EXPECTED_RANK_DEFICIENT_CASES | {
    "panel_rank_boundary_dk"
}
_PREDICTION_BACKEND_CASES = {
    "panel_entity_hc0",
    "random_effects_explicit_constant_hc0",
}

_BASE_CASES = {
    "pooled_nonrobust", "pooled_hc3", "pooled_cluster_two_way", "pooled_dk_qs",
    "panel_entity_nonrobust", "panel_entity_hc3", "panel_entity_dk",
    "random_effects_nonrobust", "random_effects_hc3",
}
_BASE_SCALES = {(10000, 2, 20), (100000, 2, 20), (100000, 10, 20)}
_HIGH_T_CASES = {"pooled_dk_qs", "panel_entity_dk_qs"}


def _source(filepath: Path, parser: str) -> dict[str, str]:
    return {
        "file": filepath.name,
        "date": _SOURCE_DATE,
        "parser": parser,
        "parser_version": _PARSER_VERSION,
    }


def parse_panel_stage_c_identifiability_physical_validation(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    warnings: list[str] = []
    if int(data.get("schema_version", -1)) != 2:
        raise ValueError("PR126 identifiability Stage-C validation requires schema_version=2")
    if data.get("git_sha") != _MEASUREMENT_SHA:
        raise ValueError("PR126 identifiability Stage-C validation measurement SHA drifted")
    if data.get("working_tree_clean") is not True:
        raise ValueError("PR126 identifiability Stage-C validation requires a clean measurement tree")
    if data.get("status") != "success":
        raise ValueError("PR126 identifiability Stage-C validation source did not succeed")
    environment = data.get("environment", {})
    if environment.get("gpu") != _EXPECTED_GPU:
        raise ValueError("PR126 identifiability Stage-C validation GPU identity drifted")
    packages = environment.get("packages", {})
    if packages.get("cupy") != _EXPECTED_CUPY_VERSION:
        raise ValueError("PR126 identifiability Stage-C CuPy provenance drifted")
    if packages.get("torch") != _EXPECTED_TORCH_VERSION:
        raise ValueError("PR126 identifiability Stage-C Torch provenance drifted")
    if int(data.get("case_count_per_backend", -1)) != len(_EXPECTED_CASES):
        raise ValueError("PR126 identifiability Stage-C estimator case count drifted")
    if int(data.get("public_primitive_count_per_backend", -1)) != len(_EXPECTED_PRIMITIVES):
        raise ValueError("PR126 identifiability Stage-C public primitive count drifted")

    dataset = data.get("dataset", {})
    scale = _scale(dataset.get("nobs", 0), dataset.get("n_features", 0))
    source_ok = True
    runs: list[dict] = []
    model_ids: set[str] = set()

    backends = data.get("backends", {})
    if set(backends) != {"cupy", "torch"}:
        raise ValueError("PR126 identifiability Stage-C validation requires exactly CuPy and Torch")

    for backend in ("cupy", "torch"):
        result = backends[backend]
        cases = result.get("cases", {})
        primitives = result.get("public_primitives", {})
        if set(cases) != _EXPECTED_CASES:
            raise ValueError(f"{backend}: PR126 identifiability estimator case identity drifted")
        if set(primitives) != _EXPECTED_PRIMITIVES:
            raise ValueError(f"{backend}: PR126 identifiability public primitive identity drifted")
        backend_ok = result.get("status") == "success" and result.get("requested_backend") == backend

        boundary = cases["panel_rank_boundary_dk"].get("covariance_metadata", {})
        boundary_contract_ok = (
            boundary.get("design_rank") == 2
            and boundary.get("design_columns") == 3
            and boundary.get("rank_deficient_extension") is True
            and boundary.get("coefficient_inference_applicable") is False
            and isinstance(boundary.get("coefficient_inference_reason"), str)
            and "rank deficient" in boundary.get("coefficient_inference_reason", "").lower()
        )
        if not boundary_contract_ok:
            raise ValueError(f"{backend}: PanelOLS rank-boundary covariance metadata drifted")

        rank_deficient_contract_ok: dict[str, bool] = {}
        for rank_case_name in sorted(_EXPECTED_IDENTIFIABILITY_CASES):
            rank_case = cases[rank_case_name]
            fit_rank = rank_case.get("fit_rank")
            parameter_count = rank_case.get("parameter_count")
            contract_ok = (
                isinstance(fit_rank, int)
                and not isinstance(fit_rank, bool)
                and isinstance(parameter_count, int)
                and not isinstance(parameter_count, bool)
                and 0 < fit_rank < parameter_count
                and rank_case.get("coefficient_inference_applicable") is False
                and isinstance(rank_case.get("coefficient_inference_reason"), str)
                and "rank deficient" in rank_case.get("coefficient_inference_reason", "").lower()
            )
            if not contract_ok:
                raise ValueError(
                    f"{backend}: {rank_case_name} identified-rank contract drifted"
                )
            rank_deficient_contract_ok[rank_case_name] = True

        for case_name in sorted(_EXPECTED_CASES):
            case = cases[case_name]
            identified_case = case_name in _EXPECTED_IDENTIFIABILITY_CASES
            inference_contract_ok = (
                case.get("coefficient_inference_applicable") is (not identified_case)
                and (
                    (
                        identified_case
                        and isinstance(case.get("coefficient_inference_reason"), str)
                        and "rank deficient"
                        in case.get("coefficient_inference_reason", "").lower()
                    )
                    or (
                        not identified_case
                        and case.get("coefficient_inference_reason") is None
                    )
                )
            )
            if not inference_contract_ok:
                raise ValueError(
                    f"{backend}: {case_name} coefficient-inference applicability drifted"
                )
            prediction_backend_ok = (
                case_name not in _PREDICTION_BACKEND_CASES
                or case.get("prediction_backend") == backend
            )
            if not prediction_backend_ok:
                raise ValueError(
                    f"{backend}: {case_name} prediction backend provenance drifted"
                )
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
                _bool_check("coefficient_inference_applicability_contract", inference_contract_ok),
            ]
            if case_name in _PREDICTION_BACKEND_CASES:
                checks.append(
                    _bool_check("prediction_backend_matches_requested", prediction_backend_ok)
                )
            if case_name == "panel_rank_boundary_dk":
                checks.append(_bool_check("rank_boundary_identified_subspace", boundary_contract_ok))
            if case_name in _EXPECTED_IDENTIFIABILITY_CASES:
                checks.append(
                    _bool_check(
                        "identified_rank_less_than_parameter_count",
                        rank_deficient_contract_ok[case_name],
                    )
                )
            model_id = _model_id(case_name)
            model_ids.add(model_id)
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-identifiability-validation",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": model_id,
                    "case_id": _stable_id(
                        "case", "stage-c-identifiability", case_name, scale["scale_key"]
                    ),
                    "method_config_id": _stable_id(
                        "method", "stage-c-identifiability-physical", case_name
                    ),
                    "variant": f"identifiability-{case_name.replace('_', '-')}",
                    "penalty": None,
                    "solver": "physical_validation",
                    "solver_display": "Physical validation",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": dict(scale),
                    "parameters": {
                        "metric_scope": "identifiability_physical_validation",
                        "measurement_git_sha": data.get("git_sha"),
                        "working_tree_clean": True,
                        "executed_backend": case.get("executed_backend"),
                        "covariance_metadata": case.get("covariance_metadata", {}),
                        "fit_rank": case.get("fit_rank"),
                        "parameter_count": case.get("parameter_count"),
                        "coefficient_inference_applicable": case.get(
                            "coefficient_inference_applicable"
                        ),
                        "coefficient_inference_reason": case.get(
                            "coefficient_inference_reason"
                        ),
                        "prediction_backend": case.get("prediction_backend"),
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
                    "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-identifiability-validation",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": "PanelCovariancePrimitive",
                    "case_id": _stable_id(
                        "case", "stage-c-identifiability-primitive", primitive_name, scale["scale_key"]
                    ),
                    "method_config_id": _stable_id(
                        "method", "stage-c-identifiability-public-primitive", primitive_name
                    ),
                    "variant": f"identifiability-public-{primitive_name.replace('_', '-')}",
                    "penalty": None,
                    "solver": "physical_validation",
                    "solver_display": "Physical validation",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": dict(scale),
                    "parameters": {
                        "metric_scope": "identifiability_public_primitive_physical_validation",
                        "measurement_git_sha": data.get("git_sha"),
                        "working_tree_clean": True,
                        "executed_backend": item.get("executed_backend"),
                    },
                    "source": _source(filepath, _VALIDATION_PARSER),
                    "metrics": {"validation": _validation(ok, filepath, checks)},
                }
            )

    expected_rows = 2 * (len(_EXPECTED_CASES) + len(_EXPECTED_PRIMITIVES))
    if len(runs) != expected_rows:
        raise ValueError(
            f"PR126 identifiability validation parser expected {expected_rows} rows, got {len(runs)}"
        )
    return runs, _models(model_ids), warnings


def parse_panel_stage_c_identifiability_performance(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 3:
        raise ValueError("PR126 identifiability Stage-C performance requires schema_version=3")
    if data.get("git_sha") != _MEASUREMENT_SHA:
        raise ValueError("PR126 identifiability Stage-C performance measurement SHA drifted")
    if data.get("working_tree_clean") is not True:
        raise ValueError("PR126 identifiability Stage-C performance requires a clean measurement tree")
    if data.get("benchmark") != "panel_stage_c_covariance_fit_overhead":
        raise ValueError("PR126 identifiability Stage-C performance benchmark identity drifted")
    if data.get("timing_scope") != "synchronized end-to-end estimator fit":
        raise ValueError("PR126 identifiability Stage-C performance timing scope drifted")
    if data.get("high_t_scale") != "10000x2x200":
        raise ValueError("PR126 identifiability Stage-C high-T scale drifted")
    if data.get("two_way_unbalanced_scale") != "10000x2x20":
        raise ValueError("PR126 identifiability Stage-C two-way scale drifted")
    performance_environment = data.get("environment", {})
    gpu_by_backend = performance_environment.get("gpu_by_backend", {})
    if gpu_by_backend != {"cupy": _EXPECTED_GPU, "torch": _EXPECTED_GPU}:
        raise ValueError("PR126 identifiability Stage-C performance GPU provenance drifted")
    performance_packages = performance_environment.get("packages", {})
    if performance_packages.get("cupy") != _EXPECTED_CUPY_VERSION:
        raise ValueError("PR126 identifiability Stage-C performance CuPy provenance drifted")
    if performance_packages.get("torch") != _EXPECTED_TORCH_VERSION:
        raise ValueError("PR126 identifiability Stage-C performance Torch provenance drifted")

    rows = data.get("rows", [])
    if len(rows) != 60:
        raise ValueError(f"PR126 identifiability Stage-C performance requires 60 rows, got {len(rows)}")
    if {row.get("backend") for row in rows} != {"cupy", "torch"}:
        raise ValueError("PR126 identifiability Stage-C performance requires CuPy and Torch rows")

    base_rows = [row for row in rows if row.get("scenario") == "base"]
    expected_base = {
        (backend, case_name, n_samples, n_features, n_times)
        for backend in ("cupy", "torch")
        for case_name in _BASE_CASES
        for n_samples, n_features, n_times in _BASE_SCALES
    }
    actual_base = [
        (
            row.get("backend"), row.get("case"), int(row.get("n_samples", 0)),
            int(row.get("n_features", 0)), int(row.get("n_times", 0)),
        )
        for row in base_rows
    ]
    if len(actual_base) != len(expected_base) or set(actual_base) != expected_base:
        raise ValueError("PR126 identifiability Stage-C performance base matrix drifted")

    high_t = [row for row in rows if row.get("scenario") == "high_t_qs"]
    expected_high_t = {
        (backend, case_name, 10000, 2, 200)
        for backend in ("cupy", "torch")
        for case_name in _HIGH_T_CASES
    }
    actual_high_t = [
        (
            row.get("backend"), row.get("case"), int(row.get("n_samples", 0)),
            int(row.get("n_features", 0)), int(row.get("n_times", 0)),
        )
        for row in high_t
    ]
    if len(actual_high_t) != len(expected_high_t) or set(actual_high_t) != expected_high_t:
        raise ValueError("PR126 identifiability Stage-C performance high-T QS matrix drifted")

    two_way = [row for row in rows if row.get("scenario") == "two_way_unbalanced"]
    expected_two_way = {
        (backend, "panel_two_way_nonrobust", 10000, 2, 20)
        for backend in ("cupy", "torch")
    }
    actual_two_way = [
        (
            row.get("backend"), row.get("case"), int(row.get("n_samples", 0)),
            int(row.get("n_features", 0)), int(row.get("n_times", 0)),
        )
        for row in two_way
    ]
    if len(actual_two_way) != len(expected_two_way) or set(actual_two_way) != expected_two_way:
        raise ValueError(
            "PR126 identifiability Stage-C performance two-way matrix drifted"
        )

    output: list[dict] = []
    model_ids: set[str] = set()
    for row in rows:
        backend = row.get("backend")
        case_name = str(row.get("case"))
        scenario = str(row.get("scenario"))
        if scenario not in {"base", "high_t_qs", "two_way_unbalanced"}:
            raise ValueError(f"unknown PR126 identifiability Stage-C scenario: {scenario!r}")
        repeats = int(row.get("repeats", 0))
        samples = row.get("samples_seconds")
        if repeats != 3 or not isinstance(samples, list) or len(samples) != 3:
            raise ValueError(
                "PR126 identifiability Stage-C timing requires exactly three raw samples"
            )
        numeric_samples = [float(value) for value in samples]
        if any(not math.isfinite(value) or value <= 0.0 for value in numeric_samples):
            raise ValueError("PR126 identifiability Stage-C timing samples must be finite and positive")
        median = float(row.get("median_seconds"))
        if not math.isfinite(median) or median <= 0.0:
            raise ValueError("PR126 identifiability Stage-C timing median must be finite and positive")
        expected_median = float(statistics.median(numeric_samples))
        if median != expected_median:
            raise ValueError(
                "PR126 identifiability Stage-C reported median must exactly match raw samples"
            )

        model_id = _model_id(case_name)
        model_ids.add(model_id)
        n_samples = int(row["n_samples"])
        n_features = int(row["n_features"])
        n_times = int(row["n_times"])
        scale = _scale(
            n_samples, n_features, suffix=f"T={n_times}", key_suffix=f"t{n_times}"
        )
        output.append(
            {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-panel-stage-c-pr126-identifiability-performance",
                "env_id": env_id,
                "category_ids": ["panel"],
                "model_id": model_id,
                "case_id": _stable_id(
                    "case", "stage-c-identifiability-performance", case_name, scenario,
                    n_samples, n_features, n_times,
                ),
                "method_config_id": _stable_id(
                    "method", "stage-c-identifiability-performance", case_name
                ),
                "variant": (
                    f"identifiability-{case_name.replace('_', '-')}-{scenario.replace('_', '-')}"
                ),
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
                            {"metric": "exactly_three_raw_samples", "status": "pass"},
                            {"metric": "raw_samples_finite_positive", "status": "pass"},
                            {"metric": "median_exactly_matches_raw_samples", "status": "pass"},
                        ],
                        "quality": "reported",
                        "source_file": filepath.name,
                    },
                },
            }
        )

    return output, _models(model_ids), []
