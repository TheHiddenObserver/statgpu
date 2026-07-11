"""Parse ElasticNet benchmark files (statgpu + glmnet)."""

import json
from pathlib import Path

from ..canonical import (
    BACKEND_MAP, SOLVER_DISPLAY_MAP,
    make_scale_key, make_scale_label, make_run_id, _short_hash,
)


def parse_elasticnet_benchmark_full(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """
    Parse benchmark_statgpu_all.json and benchmark_glmnet_all.json.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = set()
    warnings = []
    source_date = data.get("timestamp", "")[:10]

    is_glmnet = data.get("backend") == "glmnet"
    is_statgpu = "results" in data and isinstance(data["results"], list)

    if is_glmnet:
        glmnet_results = data.get("results", {})
        for dataset_name, entry in glmnet_results.items():
            if not isinstance(entry, dict):
                continue

            n_samples = entry.get("n_samples", 0)
            n_features = entry.get("n_features", 0)
            if n_samples == 0:
                continue

            scale = {
                "scale_key": make_scale_key(n_samples, n_features),
                "n_samples": n_samples,
                "n_features": n_features,
                "label": make_scale_label(n_samples, n_features),
            }

            model_id = "Lasso"
            models.add(model_id)

            source_hash = _short_hash(f"{filepath.name}:{dataset_name}")
            run_id = make_run_id(
                model_id, "squared_error", "l1", "auto",
                None, "glmnet", scale["scale_key"], env_id, "", source_hash,
            )

            run = {
                "run_id": run_id,
                "env_id": env_id,
                "category_ids": ["linear_models", "penalized_glm"],
                "model_id": model_id,
                "loss": "squared_error",
                "penalty": "l1",
                "solver": "auto",
                "solver_display": "Auto (best)",
                "solver_kind": "dispatch",
                "framework": "glmnet",
                "backend": None,
                "scale": scale,
                "source": {
                    "file": filepath.name,
                    "date": source_date,
                    "parser": "parse_elasticnet_benchmark_full_v1",
                    "parser_version": "1.0",
                },
                "metrics": {
                    "timing": {
                        "fit_time_ms": entry.get("fit_time_ms", 0),
                        "quality": "measured",
                        "source_file": filepath.name,
                    },
                    "convergence": {
                        "n_iter_mean": float(entry.get("n_iterations", entry.get("n_iter", 0))),
                        "n_iter_std": 0.0,
                        "converged_rate": 1.0,
                        "quality": "reported",
                        "source_file": filepath.name,
                    },
                },
            }
            runs.append(run)

    elif is_statgpu:
        for entry in data["results"]:
            n_samples = entry.get("n_samples", 0)
            n_features = entry.get("n_features", 0)
            if n_samples == 0:
                continue

            entry_runs = []

            scale = {
                "scale_key": make_scale_key(n_samples, n_features),
                "n_samples": n_samples,
                "n_features": n_features,
                "label": make_scale_label(n_samples, n_features),
            }

            model_id = "Lasso"
            models.add(model_id)
            numpy_time = None

            backends = entry.get("backends", {})
            for bk_name, bk_data in backends.items():
                if "error" in bk_data:
                    warnings.append(f"{filepath.name}: error in {entry.get('name', '?')}/{bk_name}: {bk_data['error']}")
                    continue

                bk_canon = BACKEND_MAP.get(bk_name)
                if bk_canon is None:
                    warnings.append(f"{filepath.name}: unknown backend '{bk_name}'")
                    continue

                framework = "statgpu" if "statgpu" in bk_name else "sklearn"
                session_id = f"{env_id}-elasticnet-{source_date}"
                source_hash = _short_hash(f"{filepath.name}:{entry.get('name', '')}:{bk_name}")

                actual_backend = bk_canon if framework == "statgpu" else None
                run_id = make_run_id(
                    model_id, "squared_error", "l1", "auto",
                    actual_backend, framework, scale["scale_key"], env_id, session_id, source_hash,
                )

                timing = {
                    "fit_time_ms": bk_data.get("fit_time_ms", 0),
                    "quality": "measured",
                    "source_file": filepath.name,
                }
                metrics: dict = {"timing": timing}

                if bk_canon == "numpy":
                    numpy_time = bk_data.get("fit_time_ms")

                if "n_iter" in bk_data:
                    metrics["convergence"] = {
                        "n_iter_mean": float(bk_data["n_iter"]),
                        "n_iter_std": 0.0,
                        "converged_rate": 1.0,
                        "quality": "reported",
                        "source_file": filepath.name,
                    }

                run = {
                    "run_id": run_id,
                    "benchmark_session_id": session_id,
                    "env_id": env_id,
                    "category_ids": ["linear_models", "penalized_glm"],
                    "model_id": model_id,
                    "loss": "squared_error",
                    "penalty": "l1",
                    "solver": "auto",
                    "solver_display": "Auto (best)",
                    "solver_kind": "dispatch",
                    "framework": framework,
                    "backend": actual_backend,
                    "scale": scale,
                    "source": {
                        "file": filepath.name,
                        "date": source_date,
                        "parser": "parse_elasticnet_benchmark_full_v1",
                        "parser_version": "1.0",
                    },
                    "metrics": metrics,
                }
                entry_runs.append(run)

            if numpy_time and numpy_time > 0:
                for run in entry_runs:
                    if run["framework"] == "statgpu" and run["backend"] in ("cupy", "torch"):
                        run_ftime = run["metrics"]["timing"]["fit_time_ms"]
                        if run_ftime > 0:
                            run["metrics"]["speedup"] = {
                                "value": round(numpy_time / run_ftime, 4),
                                "reference_backend": "numpy",
                                "reference_framework": "statgpu",
                                "reported_semantics": "computed",
                                "quality": "computed",
                                "source_file": filepath.name,
                            }
            runs.extend(entry_runs)

    model_entries = [
        {
            "model_id": m,
            "primary_category_id": "linear_models",
            "category_ids": ["linear_models", "penalized_glm"],
            "supports_penalty": True,
            "supports_inference": False,
        }
        for m in sorted(models)
    ]
    return runs, model_entries, warnings
