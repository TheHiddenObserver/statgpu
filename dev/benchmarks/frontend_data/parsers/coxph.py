"""Parse coxph_efron_bench_*.json — CoxPH survival analysis benchmark."""

import json
from pathlib import Path

from ..canonical import make_scale_key, make_scale_label


BACKEND_MAP_COXPH = {
    "cpu": ("numpy", "statgpu", None),
    "torch_gpu": ("torch", "statgpu", None),
    "cupy_gpu": ("cupy", "statgpu", None),
    "cpu_numba": ("numpy", "statgpu", "numba"),
    "statsmodels": (None, "statsmodels", None),
}


def parse_coxph_efron_bench(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """Parse coxph_efron_bench_*.json."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = set()
    warnings = []
    source_date = data.get("date", "")
    model_id = "CoxPH"
    models.add(model_id)

    source = {
        "file": filepath.name,
        "date": source_date,
        "parser": "parse_coxph_efron_bench_v1",
        "parser_version": "1.0",
    }

    # --- Precision runs (validation only) ---
    precision = data.get("precision", {})
    precision_ref = precision.get("reference", "statsmodels PHReg")
    for result in precision.get("results", []):
        n = result.get("n", 0)
        p = result.get("p", 0)
        if n == 0:
            continue

        bk_raw = result["backend"]
        bk_info = BACKEND_MAP_COXPH.get(bk_raw)
        if bk_info is None:
            warnings.append(f"{filepath.name}: unknown backend '{bk_raw}' in precision")
            continue
        backend, framework, implementation = bk_info

        scale_key = make_scale_key(n, p)
        scale = {
            "scale_key": scale_key,
            "n_samples": n,
            "n_features": p,
            "label": make_scale_label(n, p),
        }

        status = result.get("status", "PASS").lower()
        if status not in ("pass", "warn", "fail"):
            status = "pass"

        run = {
            "run_id": "",
            "env_id": env_id,
            "category_ids": ["survival"],
            "model_id": model_id,
            "loss": "coxph",
            "penalty": None,
            "solver": "newton",
            "solver_display": "Newton",
            "solver_kind": "manual",
            "variant": "efron_precision",
            "implementation": implementation,
            "framework": framework,
            "backend": backend,
            "scale": scale,
            "source": dict(source),
            "metrics": {
                "validation": {
                    "status": status,
                    "checks": [{
                        "metric": "max_abs_error",
                        "status": status,
                        "value": result.get("max_abs_error"),
                        "reference": precision_ref,
                    }],
                    "quality": "measured",
                    "source_file": filepath.name,
                }
            },
        }
        runs.append(run)

    # --- Performance runs (timing + speedup) ---
    performance = data.get("performance", {})
    speedups = data.get("speedups", {})

    for ties_variant in ("light_ties", "heavy_ties"):
        perf_data = performance.get(ties_variant, {})
        sp_data = speedups.get(ties_variant, {})

        variant = f"efron_{ties_variant}"
        session_id = f"{env_id}-coxph-{ties_variant}-{source_date}"

        # First pass: collect all runs, track cpu timing for speedup computation
        perf_runs_by_key: dict[tuple, dict] = {}

        for scale_group, scale_entries in perf_data.items():
            for scale_name, backends in scale_entries.items():
                # Parse scale from scale_name like "n1000_p10" or "n1000_p10_uft52"
                import re
                m = re.match(r"n(\d+)_p(\d+)", scale_name)
                if not m:
                    warnings.append(f"{filepath.name}: cannot parse scale '{scale_name}'")
                    continue
                n_samples = int(m.group(1))
                n_features = int(m.group(2))
                scale_key = make_scale_key(n_samples, n_features)
                scale = {
                    "scale_key": scale_key,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "label": make_scale_label(n_samples, n_features),
                }

                for bk_raw, bk_data in backends.items():
                    bk_info = BACKEND_MAP_COXPH.get(bk_raw)
                    if bk_info is None:
                        warnings.append(f"{filepath.name}: unknown backend '{bk_raw}' in {ties_variant}/{scale_name}")
                        continue
                    backend, framework, implementation = bk_info

                    key = (scale_key, framework, backend, implementation or "default")
                    run = {
                        "run_id": "",
                        "benchmark_session_id": session_id,
                        "env_id": env_id,
                        "category_ids": ["survival"],
                        "model_id": model_id,
                        "loss": "coxph",
                        "penalty": None,
                        "solver": "newton",
                        "solver_display": "Newton",
                        "solver_kind": "manual",
                        "variant": variant,
                        "implementation": implementation,
                        "framework": framework,
                        "backend": backend,
                        "scale": scale,
                        "source": dict(source),
                        "metrics": {
                            "timing": {
                                "fit_time_ms": bk_data["mean_ms"],
                                "quality": "measured",
                                "source_file": filepath.name,
                            }
                        },
                    }
                    perf_runs_by_key[key] = run

        # Second pass: attach speedups (source reports speedup vs statsmodels)
        for scale_group, scale_sp in sp_data.items():
            for scale_name, backends_sp in scale_sp.items():
                m = re.match(r"n(\d+)_p(\d+)", scale_name)
                if not m:
                    continue
                n_samples = int(m.group(1))
                n_features = int(m.group(2))
                scale_key = make_scale_key(n_samples, n_features)

                for bk_raw, speedup_val in backends_sp.items():
                    bk_info = BACKEND_MAP_COXPH.get(bk_raw)
                    if bk_info is None:
                        continue
                    backend, framework, implementation = bk_info
                    key = (scale_key, framework, backend, implementation or "default")
                    if key in perf_runs_by_key and speedup_val > 0:
                        run = perf_runs_by_key[key]
                        run["metrics"]["speedup"] = {
                            "value": round(speedup_val, 4),
                            "reference_backend": None,
                            "reference_framework": "statsmodels",
                            "reported_semantics": "reported_by_runner",
                            "quality": "reported",
                            "source_file": filepath.name,
                        }

        runs.extend(perf_runs_by_key.values())

    model_entries = [
        {
            "model_id": model_id,
            "primary_category_id": "survival",
            "category_ids": ["survival"],
            "supports_penalty": False,
            "supports_inference": True,
        }
    ]
    return runs, model_entries, warnings
