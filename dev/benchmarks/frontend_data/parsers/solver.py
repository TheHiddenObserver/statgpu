from __future__ import annotations
"""Parse glm_solver_benchmark_*.json — GLM solver benchmark with speedup data."""

import json
import re
from pathlib import Path

from ..canonical import (
    SCALE_CONFIG, BACKEND_MAP, FAMILY_MODEL_MAP,
    SOLVER_DISPLAY_MAP, SOLVER_KIND_MAP,
    make_scale_key, make_run_id, _short_hash, parse_family_penalty_solver,
    SPEEDUP_REFERENCE_BY_SOURCE,
)


def parse_glm_solver_benchmark(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """
    Parse glm_solver_benchmark_*.json.
    Structure: performance[scale][family_penalty][backend] = {best_solver, best_speedup, solvers: {name: speedup}}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = set()
    warnings = []
    source_date = data.get("date", "")

    ref_info = SPEEDUP_REFERENCE_BY_SOURCE.get(
        filepath.name,
        {"reference_backend": "numpy", "reference_framework": "statgpu", "reported_semantics": "reported_by_runner"}
    )

    env_info = data.get("environment", {})
    env_scale = env_info.get("scale", "100K x 50")

    perf = data.get("performance", {})
    for scale_name, scale_entries in perf.items():
        if scale_name in SCALE_CONFIG:
            scale_cfg = SCALE_CONFIG[scale_name]
        else:
            m = re.match(r"(\d+)K\s*[x×]\s*(\d+)", scale_name) or re.match(r"(\d+)K\s*[x×]\s*(\d+)", env_scale)
            if m:
                n_samples = int(m.group(1)) * 1000
                n_features = int(m.group(2))
                scale_cfg = {"n_samples": n_samples, "n_features": n_features, "label": env_scale}
            else:
                warnings.append(f"{filepath.name}: cannot determine scale for '{scale_name}', skipping")
                continue

        scale = {
            "scale_key": make_scale_key(scale_cfg["n_samples"], scale_cfg["n_features"]),
            "n_samples": scale_cfg["n_samples"],
            "n_features": scale_cfg["n_features"],
            "label": scale_cfg["label"],
        }

        for model_key, backends in scale_entries.items():
            family, penalty, _ = parse_family_penalty_solver(model_key)
            model_id = FAMILY_MODEL_MAP.get(family, f"Penalized{family.replace('_', ' ').title().replace(' ', '')}Regression")
            models.add(model_id)

            for bk_name, bk_data in backends.items():
                bk_canon = BACKEND_MAP.get(bk_name, bk_name)
                if bk_canon not in ("numpy", "cupy", "torch"):
                    warnings.append(f"{filepath.name}: unknown backend '{bk_name}' in {model_key}")
                    continue

                session_id = f"{env_id}-glm-solver-{source_date}"

                # best-solver (auto/dispatch) run
                best_solver = bk_data.get("best_solver", "auto")
                best_speedup = bk_data.get("best_speedup", 1.0)
                source_hash = _short_hash(f"{filepath.name}:{scale_name}:{model_key}:{bk_name}:auto")
                auto_run_id = make_run_id(
                    model_id, family, penalty, "auto", bk_canon,
                    "statgpu", scale["scale_key"], env_id, session_id, source_hash,
                )

                category_ids = ["penalized_glm"]
                if penalty == "none":
                    category_ids.append("glm")

                source = {
                    "file": filepath.name,
                    "date": source_date,
                    "parser": "parse_glm_solver_benchmark_v1",
                    "parser_version": "1.0",
                }

                auto_run = {
                    "run_id": auto_run_id,
                    "benchmark_session_id": session_id,
                    "env_id": env_id,
                    "category_ids": category_ids,
                    "model_id": model_id,
                    "loss": family,
                    "penalty": penalty,
                    "solver": "auto",
                    "solver_display": "Auto (best)",
                    "solver_kind": "dispatch",
                    "framework": "statgpu",
                    "backend": bk_canon,
                    "scale": scale,
                    "source": source,
                    "metrics": {
                        "speedup": {
                            "value": round(best_speedup, 4),
                            "reference_backend": ref_info["reference_backend"],
                            "reference_framework": ref_info["reference_framework"],
                            "reported_semantics": ref_info["reported_semantics"],
                            "quality": "reported",
                            "source_file": filepath.name,
                        }
                    },
                }
                if best_speedup > 0:
                    runs.append(auto_run)
                else:
                    warnings.append(
                        f"{filepath.name}: best solver unavailable for {model_key}/{bk_name} "
                        f"(non-positive speedup)"
                    )

                # per-solver runs
                solvers = bk_data.get("solvers", {})
                for solver_name, speedup_val in solvers.items():
                    if speedup_val <= 0:
                        warnings.append(
                            f"{filepath.name}: solver '{solver_name}' unavailable for "
                            f"{model_key}/{bk_name} (non-positive speedup)"
                        )
                        continue
                    source_hash = _short_hash(f"{filepath.name}:{scale_name}:{model_key}:{bk_name}:{solver_name}")
                    solver_run_id = make_run_id(
                        model_id, family, penalty, solver_name, bk_canon,
                        "statgpu", scale["scale_key"], env_id, session_id, source_hash,
                    )

                    solver_run = {
                        "run_id": solver_run_id,
                        "benchmark_session_id": session_id,
                        "env_id": env_id,
                        "category_ids": category_ids,
                        "model_id": model_id,
                        "loss": family,
                        "penalty": penalty,
                        "solver": solver_name,
                        "solver_display": SOLVER_DISPLAY_MAP.get(solver_name, solver_name),
                        "solver_kind": SOLVER_KIND_MAP.get(solver_name, "manual"),
                        "framework": "statgpu",
                        "backend": bk_canon,
                        "scale": scale,
                        "source": source,
                        "metrics": {
                            "speedup": {
                                "value": round(speedup_val, 4),
                                "reference_backend": ref_info["reference_backend"],
                                "reference_framework": ref_info["reference_framework"],
                                "reported_semantics": ref_info["reported_semantics"],
                                "quality": "reported",
                                "source_file": filepath.name,
                            }
                        },
                    }
                    runs.append(solver_run)

    model_entries = [
        {
            "model_id": m,
            "primary_category_id": "penalized_glm",
            "category_ids": ["penalized_glm", "glm"],
            "supports_penalty": True,
            "supports_inference": False,
        }
        for m in sorted(models)
    ]
    return runs, model_entries, warnings
