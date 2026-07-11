from __future__ import annotations
"""Parse penalized_glm_bench_perf_*.json — Penalized GLM performance benchmark."""

import json
from pathlib import Path

from ..canonical import (
    SCALE_CONFIG, BACKEND_MAP, FAMILY_MODEL_MAP,
    SOLVER_DISPLAY_MAP, SOLVER_KIND_MAP,
    make_scale_key, make_run_id, _short_hash, parse_family_penalty_solver,
)


def parse_penalized_glm_bench_perf(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """
    Parse penalized_glm_bench_perf_*.json.
    Structure: performance[scale][family_penalty_auto][backend] = {mean_ms, std_ms, min_ms, max_ms}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = set()
    warnings = []
    source_date = data.get("date", "")

    perf = data.get("performance", {})
    for scale_name, scale_entries in perf.items():
        scale_cfg = SCALE_CONFIG.get(scale_name)
        if scale_cfg is None:
            warnings.append(f"{filepath.name}: unknown scale '{scale_name}', skipping")
            continue

        scale = {
            "scale_key": make_scale_key(scale_cfg["n_samples"], scale_cfg["n_features"]),
            "n_samples": scale_cfg["n_samples"],
            "n_features": scale_cfg["n_features"],
            "label": scale_cfg["label"],
        }

        for model_key, backends in scale_entries.items():
            family, penalty, solver = parse_family_penalty_solver(model_key)
            model_id = FAMILY_MODEL_MAP.get(family, f"Penalized{family.replace('_', ' ').title().replace(' ', '')}Regression")

            models.add(model_id)

            numpy_time = None
            backend_entries = []
            for bk_name, bk_data in backends.items():
                bk_canon = BACKEND_MAP.get(bk_name, bk_name)
                if bk_canon not in ("numpy", "cupy", "torch"):
                    warnings.append(f"{filepath.name}: unknown backend '{bk_name}' in {model_key}")
                    continue
                backend_entries.append((bk_canon, bk_data))
                if bk_canon == "numpy":
                    numpy_time = bk_data.get("mean_ms") or 0

            session_id = f"{env_id}-glm-{source_date}"
            source = {
                "file": filepath.name,
                "date": source_date,
                "parser": "parse_penalized_glm_bench_perf_v1",
                "parser_version": "1.0",
            }
            category_ids = ["penalized_glm"]
            if penalty == "none":
                category_ids.append("glm")

            for bk_canon, bk_data in backend_entries:
                source_hash = _short_hash(f"{filepath.name}:{scale_name}:{model_key}:{bk_canon}")
                run_id = make_run_id(
                    model_id, family, penalty, solver, bk_canon,
                    "statgpu", scale["scale_key"], env_id, session_id, source_hash,
                )

                std_val = bk_data.get("std_ms")
                min_val = bk_data.get("min_ms")
                max_val = bk_data.get("max_ms")
                timing = {
                    "fit_time_ms": bk_data["mean_ms"],
                    "quality": "measured",
                    "source_file": filepath.name,
                }
                if std_val is not None:
                    timing["std_ms"] = std_val
                if min_val is not None:
                    timing["min_ms"] = min_val
                else:
                    timing["min_ms"] = bk_data["mean_ms"]
                if max_val is not None:
                    timing["max_ms"] = max_val
                else:
                    timing["max_ms"] = bk_data["mean_ms"]

                metrics: dict = {"timing": timing}

                if numpy_time and numpy_time > 0 and bk_canon != "numpy":
                    speedup_val = numpy_time / bk_data["mean_ms"]
                    metrics["speedup"] = {
                        "value": round(speedup_val, 4),
                        "reference_backend": "numpy",
                        "reference_framework": "statgpu",
                        "reported_semantics": "computed",
                        "quality": "computed",
                        "source_file": filepath.name,
                    }

                run = {
                    "run_id": run_id,
                    "benchmark_session_id": session_id,
                    "env_id": env_id,
                    "category_ids": category_ids,
                    "model_id": model_id,
                    "loss": family,
                    "penalty": penalty,
                    "solver": solver,
                    "solver_display": SOLVER_DISPLAY_MAP.get(solver, solver),
                    "solver_kind": SOLVER_KIND_MAP.get(solver, "manual"),
                    "framework": "statgpu",
                    "backend": bk_canon,
                    "scale": scale,
                    "source": source,
                    "metrics": metrics,
                }
                runs.append(run)

    model_entries = [
        {
            "model_id": m,
            "primary_category_id": "penalized_glm",
            "category_ids": ["penalized_glm", "glm"] if m in FAMILY_MODEL_MAP.values() else ["penalized_glm"],
            "supports_penalty": True,
            "supports_inference": True,
        }
        for m in sorted(models)
    ]
    return runs, model_entries, warnings
