from __future__ import annotations
"""Parse PR #122 Panel Stage-B physical GPU validation evidence."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..canonical import make_scale_key, make_scale_label

_PARSER_NAME = "parse_panel_stage_b_physical_validation_v1"
_PARSER_VERSION = "1.0"


def _stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{kind}-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _scale(case: dict[str, Any]) -> dict[str, Any]:
    n_samples = int(case["n_samples"])
    n_features = int(case["n_features"])
    return {
        "scale_key": make_scale_key(n_samples, n_features),
        "n_samples": n_samples,
        "n_features": n_features,
        "label": make_scale_label(n_samples, n_features),
    }


def _validation(checks: list[str], status: str, filepath: Path) -> dict[str, Any]:
    normalized = "pass" if status == "success" else "fail"
    return {
        "status": normalized,
        "checks": [{"metric": metric, "status": normalized} for metric in checks],
        "quality": "reported",
        "source_file": filepath.name,
    }


def _apply_aggregate_validation_contract(
    validation: dict[str, Any],
    *,
    schema_ok: bool,
    source_ok: bool,
    backend_ok: bool,
    executed_backend_ok: bool,
) -> dict[str, Any]:
    """Prevent a failed aggregate physical run from emitting passing rows."""
    failed_metrics: list[str] = []
    if not schema_ok:
        failed_metrics.append("source_schema_status_ok")
    if not source_ok:
        failed_metrics.append("source_validation_status_success")
    if not backend_ok:
        failed_metrics.append("backend_validation_status_success")
    if not executed_backend_ok:
        failed_metrics.append("executed_backend_matches_requested")

    if failed_metrics:
        validation["status"] = "fail"
        validation["checks"].extend(
            {"metric": metric, "status": "fail"} for metric in failed_metrics
        )
    return validation


def parse_panel_stage_b_physical_validation(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Emit validation-only Panel runs; this source contains no timings."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    warnings: list[str] = []

    schema_ok = data.get("schema_status") == "ok"
    source_ok = data.get("status") == "success"
    if not schema_ok:
        warnings.append(f"{filepath.name}: source schema_status is not ok")
    if not source_ok:
        warnings.append(f"{filepath.name}: physical validation status is not success")
    if data.get("protocol", {}).get("timing_collected") is not False:
        warnings.append(
            f"{filepath.name}: PR122 physical source must remain validation-only"
        )

    source = {
        "file": filepath.name,
        "date": data.get("source_date", ""),
        "parser": _PARSER_NAME,
        "parser_version": _PARSER_VERSION,
    }
    case_catalog = {
        case["case_id"]: case for case in data.get("case_catalog", [])
    }
    runs: list[dict] = []
    model_ids: set[str] = set()

    for backend in ("cupy", "torch"):
        backend_result = data.get("backend_results", {}).get(backend, {})
        backend_ok = backend_result.get("status") == "success"
        executed_backend = backend_result.get("executed_backend")
        executed_backend_ok = executed_backend == backend
        if not backend_ok:
            warnings.append(
                f"{filepath.name}: {backend} backend validation status is not success"
            )
        if not executed_backend_ok:
            warnings.append(
                f"{filepath.name}: requested {backend} but executed {executed_backend!r}"
            )

        for case_id, status in backend_result.get("model_cases", {}).items():
            case = case_catalog.get(case_id)
            if case is None:
                warnings.append(f"{filepath.name}: unknown case {case_id!r}")
                continue
            model_id = str(case["model_id"])
            model_ids.add(model_id)
            scale = _scale(case)
            validation = _apply_aggregate_validation_contract(
                _validation(case.get("checks", []), str(status), filepath),
                schema_ok=schema_ok,
                source_ok=source_ok,
                backend_ok=backend_ok,
                executed_backend_ok=executed_backend_ok,
            )

            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-panel-stage-b-pr122",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": model_id,
                    "case_id": _stable_id(
                        "case", case_id, case.get("variant"), scale["scale_key"]
                    ),
                    "method_config_id": _stable_id(
                        "method",
                        "panel-stage-b-physical-validation",
                        model_id,
                        case.get("variant"),
                    ),
                    "variant": str(case.get("variant") or case_id),
                    "penalty": None,
                    "solver": "physical_validation",
                    "solver_display": "Physical validation",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": scale,
                    "parameters": {
                        "metric_scope": "physical_validation",
                        "measurement_git_sha": data.get("git_sha"),
                        "working_tree_clean": bool(data.get("working_tree_clean")),
                    },
                    "source": dict(source),
                    "metrics": {
                        "validation": validation,
                        "inference": {
                            "ok": validation["status"] == "pass",
                            "quality": "reported",
                            "source_file": filepath.name,
                        },
                    },
                }
            )

        for diagnostic_id, diagnostic in backend_result.get("diagnostics", {}).items():
            balance = "unbalanced" if diagnostic_id.endswith("unbalanced") else "balanced"
            explicit_re_constant = diagnostic_id.startswith(
                "hausman_explicit_re_constant_"
            )
            parameterization = (
                "re-explicit-constant" if explicit_re_constant else "standard"
            )
            variant = (
                f"hausman-re-explicit-constant-{balance}"
                if explicit_re_constant
                else f"hausman-{balance}"
            )
            n_samples = 49 if balance == "unbalanced" else 54
            scale = {
                "scale_key": make_scale_key(n_samples, 2),
                "n_samples": n_samples,
                "n_features": 2,
                "label": make_scale_label(n_samples, 2),
            }
            status = str(diagnostic.get("status", "failed"))
            validation = _apply_aggregate_validation_contract(
                _validation(
                    ["hausman_backend_consistency", "backend_provenance"],
                    status,
                    filepath,
                ),
                schema_ok=schema_ok,
                source_ok=source_ok,
                backend_ok=backend_ok,
                executed_backend_ok=executed_backend_ok,
            )
            method_parts: list[object] = [
                "panel-stage-b-physical-validation",
                "hausman",
                balance,
            ]
            if explicit_re_constant:
                method_parts.append(parameterization)
            model_ids.add("PanelOLS")
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-panel-stage-b-pr122",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": "PanelOLS",
                    "case_id": _stable_id("case", diagnostic_id, scale["scale_key"]),
                    "method_config_id": _stable_id("method", *method_parts),
                    "variant": variant,
                    "penalty": None,
                    "solver": "physical_validation",
                    "solver_display": "Physical validation",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": scale,
                    "parameters": {
                        "metric_scope": "physical_validation",
                        "diagnostic": "hausman",
                        "parameterization": parameterization,
                        "applicable": bool(diagnostic.get("applicable")),
                        "measurement_git_sha": data.get("git_sha"),
                        "working_tree_clean": bool(data.get("working_tree_clean")),
                    },
                    "source": dict(source),
                    "metrics": {"validation": validation},
                }
            )

    models = [
        {
            "model_id": model_id,
            "primary_category_id": "panel",
            "category_ids": ["panel"],
            "supports_penalty": False,
            "supports_inference": True,
        }
        for model_id in sorted(model_ids)
    ]
    return runs, models, warnings
