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


def parse_panel_stage_b_physical_validation(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Emit validation-only Panel runs; this source contains no timings."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    warnings: list[str] = []

    if data.get("schema_status") != "ok":
        warnings.append(f"{filepath.name}: source schema_status is not ok")
    if data.get("status") != "success":
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
        executed_backend = backend_result.get("executed_backend")
        if executed_backend != backend:
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
            validation = _validation(case.get("checks", []), str(status), filepath)
            if executed_backend != backend:
                validation["status"] = "fail"
                validation["checks"].append(
                    {
                        "metric": "executed_backend_matches_requested",
                        "status": "fail",
                    }
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
            n_samples = 49 if balance == "unbalanced" else 54
            scale = {
                "scale_key": make_scale_key(n_samples, 2),
                "n_samples": n_samples,
                "n_features": 2,
                "label": make_scale_label(n_samples, 2),
            }
            status = str(diagnostic.get("status", "failed"))
            validation = _validation(
                ["hausman_backend_consistency", "backend_provenance"], status, filepath
            )
            if executed_backend != backend:
                validation["status"] = "fail"
                validation["checks"].append(
                    {
                        "metric": "executed_backend_matches_requested",
                        "status": "fail",
                    }
                )
            model_ids.add("PanelOLS")
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-panel-stage-b-pr122",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": "PanelOLS",
                    "case_id": _stable_id("case", diagnostic_id, scale["scale_key"]),
                    "method_config_id": _stable_id(
                        "method", "panel-stage-b-physical-validation", "hausman", balance
                    ),
                    "variant": f"hausman-{balance}",
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
