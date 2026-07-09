#!/usr/bin/env python3
"""
Generate unified benchmark data JSON for the statgpu frontend dashboard.

Usage:
    python dev/benchmarks/generate_benchmark_data.py \
        --out frontend/public/data/benchmark_data.json \
        --report frontend/public/data/parse_report.json

    python dev/benchmarks/generate_benchmark_data.py --check  # validate only
"""

import json
import os
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

KNOWN_FAMILIES = [
    "squared_error", "logistic", "poisson", "gamma",
    "inverse_gaussian", "negative_binomial", "tweedie",
]

KNOWN_PENALTIES = [
    "none", "l1", "l2", "elasticnet", "scad", "mcp", "adaptive_l1",
    "group_lasso", "group_mcp", "group_scad",
]

BACKEND_MAP = {
    "numpy": "numpy", "cpu": "numpy", "statgpu_cpu": "numpy",
    "numPy": "numpy", "NumPy": "numpy", "CPU/NumPy": "numpy",
    "cupy": "cupy", "cuda": "cupy", "statgpu_gpu_cupy": "cupy",
    "CuPy": "cupy", "cupy_gpu": "cupy",
    "torch": "torch", "statgpu_gpu_torch": "torch",
    "Torch": "torch", "torch_cuda": "torch",
}

FRAMEWORK_MAP = {
    "statgpu": "statgpu", "statgpu_cpu": "statgpu",
    "statgpu_gpu_cupy": "statgpu", "statgpu_gpu_torch": "statgpu",
    "sklearn": "sklearn", "scikit-learn": "sklearn",
    "sklearn.linear_model": "sklearn",
    "statsmodels": "statsmodels", "sm": "statsmodels",
    "glmnet": "glmnet", "R": "r", "r": "r",
}

# scale_key to n,p mapping (derived from benchmark configs)
SCALE_CONFIG = {
    "small_5k": {"n_samples": 5000, "n_features": 500, "label": "5K×500"},
    "medium_100k": {"n_samples": 100000, "n_features": 50, "label": "100K×50"},
}

# scale_key from (n_samples, n_features)
def make_scale_key(n_samples: int, n_features: int) -> str:
    return f"n{n_samples}_p{n_features}"

def make_scale_label(n_samples: int, n_features: int) -> str:
    if n_samples >= 1000:
        ns = f"{n_samples//1000}K" if n_samples % 1000 == 0 else f"{n_samples/1000:.0f}K"
    else:
        ns = str(n_samples)
    return f"{ns}×{n_features}"

# solver -> solver_kind mapping
SOLVER_KIND_MAP = {
    "auto": "dispatch",
    "exact": "manual", "newton": "manual", "irls": "manual",
    "lbfgs": "manual", "fista": "manual", "fista_bb": "manual",
    "admm": "manual", "irls_cd": "manual",
    "proximal_irls_cd": "manual", "proximal_newton": "manual",
}

SOLVER_DISPLAY_MAP = {
    "auto": "Auto (best)", "exact": "Exact", "newton": "Newton",
    "irls": "IRLS", "lbfgs": "L-BFGS", "fista": "FISTA",
    "fista_bb": "FISTA-BB", "admm": "ADMM",
}

# Family to model_id mapping (for PenalizedGLM subclasses)
FAMILY_MODEL_MAP = {
    "squared_error": "PenalizedLinearRegression",
    "logistic": "PenalizedLogisticRegression",
    "poisson": "PenalizedPoissonRegression",
    "gamma": "PenalizedGammaRegression",
    "inverse_gaussian": "PenalizedInverseGaussianRegression",
    "negative_binomial": "PenalizedNegativeBinomialRegression",
    "tweedie": "PenalizedTweedieRegression",
}

# Speedup reference by source file
SPEEDUP_REFERENCE_BY_SOURCE = {
    "glm_solver_benchmark_2026-06-23.json": {
        "reference_backend": "numpy",
        "reference_framework": "statgpu",
        "reported_semantics": "reported_by_runner",
    }
}

# Category definitions (from plan)
CATEGORIES = [
    {"category_id": "penalized_glm", "name_zh": "惩罚GLM", "name_en": "Penalized GLM"},
    {"category_id": "linear_models", "name_zh": "线性模型", "name_en": "Linear Models"},
    {"category_id": "glm", "name_zh": "GLM", "name_en": "GLM"},
    {"category_id": "survival", "name_zh": "生存分析", "name_en": "Survival Analysis"},
    {"category_id": "robust_quantile", "name_zh": "稳健/分位数", "name_en": "Robust/Quantile"},
    {"category_id": "unsupervised", "name_zh": "无监督学习", "name_en": "Unsupervised"},
    {"category_id": "ordered", "name_zh": "有序模型", "name_en": "Ordered Models"},
    {"category_id": "nonparametric", "name_zh": "非参数", "name_en": "Nonparametric"},
    {"category_id": "panel", "name_zh": "面板数据", "name_en": "Panel Data"},
    {"category_id": "covariance", "name_zh": "协方差估计", "name_en": "Covariance"},
    {"category_id": "feature_selection", "name_zh": "特征选择", "name_en": "Feature Selection"},
    {"category_id": "anova", "name_zh": "ANOVA", "name_en": "ANOVA"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_hash(s: str, length: int = 6) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:length]

def make_run_id(
    model_id: str, loss: str, penalty: str, solver: str,
    backend: str, framework: str, scale_key: str,
    env_id: str, session_id: str, source_hash: str,
) -> str:
    parts = [model_id, loss, penalty or "none", solver, backend or "ext",
             framework, scale_key, env_id, session_id[:8] if session_id else "", source_hash]
    return "-".join(str(p) for p in parts).lower().replace(" ", "")


def parse_family_penalty_solver(key: str):
    """Parse a key like 'squared_error_l1_auto' or 'logistic_none_auto' into (family, penalty, solver)."""
    parts = key.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in SOLVER_KIND_MAP:
        solver = parts[1]
        prefix = parts[0]
    else:
        solver = "auto"
        prefix = key

    for fam in sorted(KNOWN_FAMILIES, key=len, reverse=True):
        if prefix.startswith(fam):
            family = fam
            penalty = prefix[len(fam) + 1:]
            if penalty not in KNOWN_PENALTIES:
                penalty = "none"
            return family, penalty, solver

    for pen in sorted(KNOWN_PENALTIES, key=len, reverse=True):
        if prefix.endswith("_" + pen):
            penalty = pen
            family = prefix[:-(len(pen) + 1)]
            return family, penalty, solver

    return prefix, "none", solver


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

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
            for bk_name, bk_data in backends.items():
                bk_canon = BACKEND_MAP.get(bk_name, bk_name)
                if bk_canon not in ("numpy", "cupy", "torch"):
                    warnings.append(f"{filepath.name}: unknown backend '{bk_name}' in {model_key}")
                    continue

                if bk_canon == "numpy":
                    numpy_time = bk_data.get("mean_ms")

                session_id = f"{env_id}-glm-{source_date}"
                source_hash = _short_hash(f"{filepath.name}:{scale_name}:{model_key}:{bk_name}")
                run_id = make_run_id(
                    model_id, family, penalty, solver, bk_canon,
                    "statgpu", scale["scale_key"], env_id, session_id, source_hash,
                )

                category_ids = ["penalized_glm"]
                if penalty == "none":
                    category_ids.append("glm")

                timing = {
                    "fit_time_ms": bk_data["mean_ms"],
                    "std_ms": bk_data.get("std_ms", 0),
                    "min_ms": bk_data.get("min_ms", bk_data["mean_ms"]),
                    "max_ms": bk_data.get("max_ms", bk_data["mean_ms"]),
                    "quality": "measured",
                    "source_file": filepath.name,
                }

                metrics: dict = {"timing": timing}

                if numpy_time is not None and bk_canon != "numpy" and numpy_time > 0:
                    speedup_val = numpy_time / bk_data["mean_ms"]
                    metrics["speedup"] = {
                        "value": round(speedup_val, 4),
                        "reference_backend": "numpy",
                        "reference_framework": "statgpu",
                        "reported_semantics": "computed",
                        "quality": "computed",
                        "source_file": filepath.name,
                    }

                source = {
                    "file": filepath.name,
                    "date": source_date,
                    "parser": "parse_penalized_glm_bench_perf_v1",
                    "parser_version": "1.0",
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
        # Derive n,p from scale name or environment
        if scale_name in SCALE_CONFIG:
            scale_cfg = SCALE_CONFIG[scale_name]
        else:
            # Try to parse from env_scale like "100K x 50"
            import re
            m = re.match(r"(\d+)K\s*[x×]\s*(\d+)", env_scale)
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

                # Generate best-solver (auto/dispatch) run
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
                runs.append(auto_run)

                # Generate per-solver runs
                solvers = bk_data.get("solvers", {})
                for solver_name, speedup_val in solvers.items():
                    if solver_name == best_solver and speedup_val == best_speedup:
                        continue  # Already added as auto

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


def parse_elasticnet_benchmark_full(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """
    Parse benchmark_statgpu_all.json and benchmark_glmnet_all.json.
    Structure: results[name, n_samples, n_features, backends{backend_name: {fit_time_ms, n_iter, coef_norm}}]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = set()
    warnings = []
    source_date = data.get("timestamp", "")[:10]

    # Detect file type
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
                        "n_iter": entry.get("n_iterations", entry.get("n_iter", 0)),
                        "converged": True,
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
                        "n_iter": bk_data["n_iter"],
                        "converged": True,
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
                runs.append(run)

            # Add speedup for GPU runs
            if numpy_time and numpy_time > 0:
                for run in runs:
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


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

PARSER_REGISTRY: dict[str, dict] = {
    "penalized_glm_bench_perf_2026-06-22.json": {
        "parser": parse_penalized_glm_bench_perf,
        "env_id": "remote-p100",
    },
    "glm_solver_benchmark_2026-06-23.json": {
        "parser": parse_glm_solver_benchmark,
        "env_id": "remote-p100",
    },
    "benchmark_full/benchmark_statgpu_all.json": {
        "parser": parse_elasticnet_benchmark_full,
        "env_id": "remote-p100",
    },
    "benchmark_full/benchmark_glmnet_all.json": {
        "parser": parse_elasticnet_benchmark_full,
        "env_id": "remote-p100",
    },
}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def get_git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def generate(results_dir: Path, check_only: bool = False) -> tuple[dict, dict]:
    """Generate unified benchmark_data.json from results_dir."""
    all_runs: list[dict] = []
    all_models: dict[str, dict] = {}
    all_warnings: list[dict] = []
    files_seen = 0
    files_parsed = 0

    for filename, config in PARSER_REGISTRY.items():
        filepath = results_dir / filename
        files_seen += 1
        if not filepath.exists():
            all_warnings.append({"file": filename, "reason": "file not found"})
            continue

        parser_fn = config["parser"]
        env_id = config["env_id"]
        try:
            runs, model_entries, warns = parser_fn(filepath, env_id)
            all_runs.extend(runs)
            for m in model_entries:
                mid = m["model_id"]
                if mid not in all_models:
                    all_models[mid] = m
                else:
                    # Merge category_ids
                    existing = set(all_models[mid].get("category_ids", []))
                    existing.update(m.get("category_ids", []))
                    all_models[mid]["category_ids"] = sorted(existing)
            for w in warns:
                all_warnings.append({"file": filename, "reason": w})
            files_parsed += 1
        except Exception as e:
            all_warnings.append({"file": filename, "reason": f"parse error: {e}"})

    # Build environments
    environments = [
        {
            "env_id": "remote-p100",
            "label": "Tesla P100 + Xeon 8163",
            "gpu": "NVIDIA Tesla P100-16GB",
            "cpu": "12× Intel Xeon Platinum 8163 @ 2.50GHz",
            "host": "hz-4.matpool.com",
        }
    ]

    output = {
        "schema_version": "1.0.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "generator": "dev/benchmarks/generate_benchmark_data.py",
            "git_sha": get_git_sha(),
        },
        "environments": environments,
        "categories": CATEGORIES,
        "models": sorted(all_models.values(), key=lambda x: x["model_id"]),
        "runs": all_runs,
    }

    parse_report = {
        "files_seen": files_seen,
        "files_parsed": files_parsed,
        "files_skipped": files_seen - files_parsed,
        "runs_generated": len(all_runs),
        "warnings": all_warnings,
    }

    return output, parse_report


def validate_output(output: dict) -> list[str]:
    """Basic validation checks (full JSON Schema validation done separately)."""
    errors = []
    runs = output.get("runs", [])
    seen_ids = set()

    for i, run in enumerate(runs):
        rid = run.get("run_id", f"<index {i}>")
        if rid in seen_ids:
            errors.append(f"Duplicate run_id: {rid}")
        seen_ids.add(rid)

        # Required fields
        for field in ["run_id", "env_id", "category_ids", "model_id", "framework", "backend", "scale", "source"]:
            if field not in run:
                errors.append(f"{rid}: missing required field '{field}'")

        # framework/backend consistency
        fw = run.get("framework")
        bk = run.get("backend")
        if fw == "statgpu" and bk not in ("numpy", "cupy", "torch"):
            errors.append(f"{rid}: statgpu run has invalid backend '{bk}'")
        if fw != "statgpu" and bk is not None:
            errors.append(f"{rid}: external framework '{fw}' has non-null backend '{bk}'")

        # Metrics validation
        metrics = run.get("metrics", {})
        if "timing" in metrics:
            t = metrics["timing"]
            ft = t.get("fit_time_ms", -1)
            if ft < 0:
                errors.append(f"{rid}: timing.fit_time_ms < 0 ({ft})")
            if "std_ms" in t and t["std_ms"] < 0:
                errors.append(f"{rid}: timing.std_ms < 0")
            if "min_ms" in t and "max_ms" in t and t["min_ms"] > t["max_ms"]:
                errors.append(f"{rid}: timing.min_ms > max_ms")

        if "speedup" in metrics:
            s = metrics["speedup"]
            if s.get("value", -1) < 0:
                errors.append(f"{rid}: speedup.value < 0")
            for sf in ["reference_backend", "reference_framework", "reported_semantics"]:
                if sf not in s:
                    errors.append(f"{rid}: speedup missing '{sf}'")

        # No NaN/Inf
        def check_nan(obj, path=""):
            if isinstance(obj, float):
                import math
                if math.isnan(obj) or math.isinf(obj):
                    errors.append(f"{rid}: NaN/Inf at {path}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_nan(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for j, v in enumerate(obj):
                    check_nan(v, f"{path}[{j}]")

        check_nan(run, rid)

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark frontend data")
    parser.add_argument("--out", help="Output path for benchmark_data.json")
    parser.add_argument("--report", help="Output path for parse_report.json")
    parser.add_argument("--check", action="store_true", help="Validate only, don't write output")
    parser.add_argument("--results-dir", default="results", help="Directory containing benchmark result files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        # Relative to project root
        repo_root = Path(__file__).resolve().parents[2]
        results_dir = repo_root / args.results_dir

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning results from: {results_dir}")
    output, parse_report = generate(results_dir, check_only=args.check)

    errors = validate_output(output)
    if errors:
        print(f"VALIDATION ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        if args.check:
            sys.exit(1)

    print(f"Runs generated: {len(output['runs'])}")
    print(f"Models: {len(output['models'])}")
    print(f"Validation errors: {len(errors)}")
    print(f"Parse warnings: {len(parse_report['warnings'])}")
    if parse_report["warnings"]:
        for w in parse_report["warnings"][:5]:
            print(f"  - {w['file']}: {w['reason']}")

    if args.check:
        if errors:
            sys.exit(1)
        print("OK — validation passed")
        return

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {out_path}")

    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(parse_report, f, indent=2)
        print(f"Wrote: {report_path}")

    if not args.out and not args.report:
        # Print summary to stdout when no output files specified
        print(json.dumps(parse_report, indent=2))


if __name__ == "__main__":
    main()
