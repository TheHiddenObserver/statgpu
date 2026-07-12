from __future__ import annotations
"""Parsers for benchmark domains that were previously listed but had no runs."""

import json
import re
from pathlib import Path
from typing import Any

from ..canonical import make_scale_key, make_scale_label


_BACKENDS = {
    "cpu": "numpy",
    "numpy": "numpy",
    "NumPy": "numpy",
    "cuda": "cupy",
    "cupy": "cupy",
    "CuPy": "cupy",
    "torch": "torch",
    "Torch": "torch",
}


def _scale(n_samples: int, n_features: int) -> dict[str, Any]:
    return {
        "scale_key": make_scale_key(n_samples, n_features),
        "n_samples": n_samples,
        "n_features": n_features,
        "label": make_scale_label(n_samples, n_features),
    }


def _source(filepath: Path, date: str, parser: str) -> dict[str, str]:
    return {
        "file": filepath.name,
        "date": date,
        "parser": parser,
        "parser_version": "1.0",
    }


def _timing_ms(value_ms: float, filepath: Path, quality: str = "measured") -> dict[str, Any]:
    return {
        "fit_time_ms": round(float(value_ms), 6),
        "quality": quality,
        "source_file": filepath.name,
    }


def _reported_speedup(
    value: float,
    filepath: Path,
    reference_framework: str,
    reference_backend: str | None = None,
) -> dict[str, Any]:
    return {
        "value": round(float(value), 6),
        "reference_backend": reference_backend,
        "reference_framework": reference_framework,
        "reported_semantics": "reported_by_runner",
        "quality": "reported",
        "source_file": filepath.name,
    }


def _computed_speedup(value: float, filepath: Path) -> dict[str, Any]:
    return {
        "value": round(float(value), 6),
        "reference_backend": "numpy",
        "reference_framework": "statgpu",
        "reported_semantics": "computed",
        "quality": "computed",
        "source_file": filepath.name,
    }


def parse_loss_functions_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse the 2026-06-23 robust/quantile performance benchmark."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    date = data.get("date", "")
    source = _source(filepath, date, "parse_loss_functions_benchmark_v1")
    warnings: list[str] = []
    runs: list[dict] = []

    validation_rows: dict[tuple[str, int, int, str], dict] = {}
    for row in data.get("precision", {}).get("results", []):
        loss = row.get("loss")
        backend = _BACKENDS.get(str(row.get("backend")))
        if loss in {"quantile", "huber"} and backend:
            validation_rows[(loss, int(row["n"]), int(row["p"]), backend)] = row

    specs = {
        "quantile": ("QuantileRegression", True),
        "huber": ("RobustRegression", False),
    }
    for loss, (model_id, _supports_inference) in specs.items():
        for scale_name, methods in data.get("performance", {}).get(loss, {}).items():
            match = re.fullmatch(r"n(\d+)_p(\d+)", scale_name)
            if not match:
                warnings.append(f"{filepath.name}: invalid {loss} scale '{scale_name}'")
                continue
            n_samples, n_features = map(int, match.groups())
            scale = _scale(n_samples, n_features)
            case_id = f"{loss}-{scale['scale_key']}"

            for method_name, entry in methods.items():
                mean_ms = entry.get("mean_ms")
                if mean_ms is None:
                    continue
                framework = "statgpu" if method_name == "cpu" else "sklearn"
                backend = "numpy" if framework == "statgpu" else None
                metrics: dict[str, Any] = {
                    "timing": _timing_ms(mean_ms, filepath),
                }

                if framework == "statgpu":
                    speedup = (
                        data.get("speedups", {})
                        .get(loss, {})
                        .get(scale_name, {})
                        .get("cpu")
                    )
                    if speedup and speedup > 0:
                        metrics["speedup"] = _reported_speedup(
                            speedup, filepath, "sklearn", None
                        )
                    validation = validation_rows.get((loss, n_samples, n_features, "numpy"))
                    if validation:
                        status = str(validation.get("status", "PASS")).lower()
                        checks = []
                        for key in ("objective", "ref_objective", "grad_norm"):
                            if validation.get(key) is not None:
                                checks.append(
                                    {
                                        "metric": key,
                                        "status": status,
                                        "value": float(validation[key]),
                                    }
                                )
                        metrics["validation"] = {
                            "status": status,
                            "checks": checks,
                            "quality": "reported",
                            "source_file": filepath.name,
                        }

                runs.append(
                    {
                        "run_id": "",
                        "benchmark_session_id": f"{env_id}-loss-functions-{date}",
                        "env_id": env_id,
                        "category_ids": ["robust_quantile"],
                        "model_id": model_id,
                        "case_id": case_id,
                        "method_config_id": "irls",
                        "loss": loss,
                        "penalty": None,
                        "solver": "irls",
                        "solver_display": "IRLS",
                        "solver_kind": "manual",
                        "framework": framework,
                        "backend": backend,
                        "scale": scale,
                        "source": dict(source),
                        "metrics": metrics,
                    }
                )

    models = [
        {
            "model_id": "QuantileRegression",
            "primary_category_id": "robust_quantile",
            "category_ids": ["robust_quantile"],
            "supports_penalty": False,
            "supports_inference": True,
        },
        {
            "model_id": "RobustRegression",
            "primary_category_id": "robust_quantile",
            "category_ids": ["robust_quantile"],
            "supports_penalty": False,
            "supports_inference": False,
        },
    ]
    return runs, models, warnings


def parse_ordered_inference_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse ordered-logit/probit rows from PR #74 inference validation."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    date = ""
    source = _source(filepath, date, "parse_ordered_inference_benchmark_v1")
    runs: list[dict] = []
    warnings: list[str] = []
    model_map = {
        "ordered_logit": "OrderedLogit",
        "ordered_probit": "OrderedProbit",
    }

    for key, backend_rows in data.items():
        match = re.fullmatch(r"(ordered_logit|ordered_probit)_n(\d+)_p(\d+)", key)
        if not match:
            continue
        family, n_raw, p_raw = match.groups()
        n_samples, n_features = int(n_raw), int(p_raw)
        model_id = model_map[family]
        scale = _scale(n_samples, n_features)
        numpy_time = backend_rows.get("NumPy", {}).get("time")

        for backend_name, entry in backend_rows.items():
            backend = _BACKENDS.get(backend_name)
            if backend is None or not entry.get("ok", True):
                continue
            time_s = entry.get("time")
            if time_s is None:
                warnings.append(f"{filepath.name}: {key}/{backend_name} has no timing")
                continue
            metrics: dict[str, Any] = {
                "timing": _timing_ms(float(time_s) * 1000.0, filepath),
                "inference": {
                    "bse": float(entry["bse0"]),
                    "ok": bool(entry.get("ok", True)),
                    "quality": "measured",
                    "source_file": filepath.name,
                },
            }
            if entry.get("wald_stat") is not None:
                metrics["inference"]["wald_stat"] = float(entry["wald_stat"])
            if entry.get("wald_pval") is not None:
                metrics["inference"]["p_value"] = float(entry["wald_pval"])
            if entry.get("iter") is not None:
                metrics["convergence"] = {
                    "n_iter_mean": float(entry["iter"]),
                    "converged_rate": 1.0,
                    "quality": "measured",
                    "source_file": filepath.name,
                }
            if backend != "numpy" and numpy_time and time_s > 0:
                metrics["speedup"] = _computed_speedup(
                    float(numpy_time) / float(time_s), filepath
                )

            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-ordered-pr74",
                    "env_id": env_id,
                    "category_ids": ["ordered"],
                    "model_id": model_id,
                    "case_id": scale["scale_key"],
                    "method_config_id": family,
                    "loss": family,
                    "penalty": None,
                    "solver": "newton",
                    "solver_display": "Newton",
                    "solver_kind": "manual",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": scale,
                    "source": dict(source),
                    "metrics": metrics,
                }
            )

    models = [
        {
            "model_id": model_id,
            "primary_category_id": "ordered",
            "category_ids": ["ordered"],
            "supports_penalty": False,
            "supports_inference": True,
        }
        for model_id in model_map.values()
    ]
    return runs, models, warnings


_UNSUPERVISED_CASES = {
    "pca_large": ("PCA", 100000, 100, None),
    "kmeans_large": ("KMeans", 100000, 100, None),
    "gmm_large": ("GaussianMixture", 100000, 100, None),
    "nmf_large": ("NMF", 100000, 100, None),
    "tsvd_large": ("TruncatedSVD", 100000, 100, None),
    "ipca_large": ("IncrementalPCA", 100000, 100, None),
    "agglo_medium": ("AgglomerativeClustering", 10000, 50, None),
    "dbscan_10d_large": ("DBSCAN", 100000, 10, "10d"),
    "dbscan_50d_large": ("DBSCAN", 100000, 50, "50d"),
    "umap_medium": ("UMAP", 100000, 100, None),
    "tsne_small": ("TSNE", 1000, 20, None),
    "mbkmeans_large": ("MiniBatchKMeans", 100000, 100, None),
    "mbnmf_large": ("MiniBatchNMF", 100000, 100, None),
}


def parse_unsupervised_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse the latest 2026-06-27 unsupervised summary benchmark."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    date = "2026-06-27"
    source = _source(filepath, date, "parse_unsupervised_benchmark_v1")
    runs: list[dict] = []
    warnings: list[str] = []
    model_ids: set[str] = set()

    for prefix, (model_id, n_samples, n_features, variant) in _UNSUPERVISED_CASES.items():
        scale = _scale(n_samples, n_features)
        entries: dict[str, dict] = {}
        for backend in ("numpy", "cupy", "torch"):
            entry = data.get(f"{prefix}_{backend}")
            if entry and entry.get("time") is not None:
                entries[backend] = entry
        if not entries:
            warnings.append(f"{filepath.name}: no rows for {prefix}")
            continue
        model_ids.add(model_id)
        numpy_time = entries.get("numpy", {}).get("time")
        external_time = next(
            (entry.get("external") for entry in entries.values() if entry.get("external") is not None),
            None,
        )
        case_id = f"{prefix}-{scale['scale_key']}"

        for backend, entry in entries.items():
            time_s = float(entry["time"])
            metrics: dict[str, Any] = {
                "timing": _timing_ms(time_s * 1000.0, filepath),
            }
            if entry.get("speedup") and external_time:
                metrics["speedup"] = _reported_speedup(
                    float(entry["speedup"]), filepath, "sklearn", None
                )
            elif backend != "numpy" and numpy_time and time_s > 0:
                metrics["speedup"] = _computed_speedup(
                    float(numpy_time) / time_s, filepath
                )

            run = {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-unsupervised-{date}",
                "env_id": env_id,
                "category_ids": ["unsupervised"],
                "model_id": model_id,
                "case_id": case_id,
                "method_config_id": model_id.lower(),
                "penalty": None,
                "solver": "auto",
                "solver_display": "Auto",
                "solver_kind": "dispatch",
                "framework": "statgpu",
                "backend": backend,
                "scale": scale,
                "source": dict(source),
                "metrics": metrics,
            }
            if variant:
                run["variant"] = variant
            runs.append(run)

        if external_time and float(external_time) > 0:
            external_run = {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-unsupervised-{date}",
                "env_id": env_id,
                "category_ids": ["unsupervised"],
                "model_id": model_id,
                "case_id": case_id,
                "method_config_id": model_id.lower(),
                "penalty": None,
                "solver": "auto",
                "solver_display": "Auto",
                "solver_kind": "dispatch",
                "framework": "sklearn",
                "backend": None,
                "scale": scale,
                "source": dict(source),
                "metrics": {
                    "timing": _timing_ms(float(external_time) * 1000.0, filepath, "reported")
                },
            }
            if variant:
                external_run["variant"] = variant
            runs.append(external_run)

    models = [
        {
            "model_id": model_id,
            "primary_category_id": "unsupervised",
            "category_ids": ["unsupervised"],
            "supports_penalty": False,
            "supports_inference": False,
        }
        for model_id in sorted(model_ids)
    ]
    return runs, models, warnings


def parse_new_modules_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse the latest panel and aligned-GAM comparisons from 2026-06-24."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    date = data.get("date", "")
    source = _source(filepath, date, "parse_new_modules_benchmark_v1")
    runs: list[dict] = []
    warnings: list[str] = []
    models: list[dict] = []

    panel_rows = data.get("modules", {}).get("panel", {}).get("external_comparison", {})
    panel_specs = {"PanelOLS": "PanelOLS", "RE": "RandomEffects"}
    panel_scale = _scale(100000, 20)
    for token, model_id in panel_specs.items():
        selected = {
            backend: panel_rows.get(f"panel_large_{token}_{backend}")
            for backend in ("numpy", "cupy", "torch")
        }
        selected = {key: value for key, value in selected.items() if value}
        if not selected:
            warnings.append(f"{filepath.name}: no panel_large rows for {token}")
            continue
        external_time = next(
            float(value["external_time"])
            for value in selected.values()
            if value.get("external_time") is not None
        )
        case_id = f"panel-large-{token.lower()}"
        for backend, entry in selected.items():
            metrics: dict[str, Any] = {
                "timing": _timing_ms(float(entry["statgpu_time"]) * 1000.0, filepath),
                "speedup": _reported_speedup(
                    float(entry["speedup"]), filepath, "linearmodels", None
                ),
            }
            if entry.get("coef_rel_diff") is not None:
                metrics["accuracy"] = {
                    "coef_l2_rel_error": float(entry["coef_rel_diff"]),
                    "reference": "linearmodels",
                    "quality": "computed",
                    "source_file": filepath.name,
                }
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-new-modules-{date}",
                    "env_id": env_id,
                    "category_ids": ["panel"],
                    "model_id": model_id,
                    "case_id": case_id,
                    "method_config_id": token.lower(),
                    "penalty": None,
                    "solver": "closed_form",
                    "solver_display": "Closed form",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": panel_scale,
                    "source": dict(source),
                    "metrics": metrics,
                }
            )
        runs.append(
            {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-new-modules-{date}",
                "env_id": env_id,
                "category_ids": ["panel"],
                "model_id": model_id,
                "case_id": case_id,
                "method_config_id": token.lower(),
                "penalty": None,
                "solver": "closed_form",
                "solver_display": "Closed form",
                "solver_kind": "internal",
                "framework": "linearmodels",
                "backend": None,
                "scale": panel_scale,
                "source": dict(source),
                "metrics": {
                    "timing": _timing_ms(external_time * 1000.0, filepath, "reported")
                },
            }
        )
        models.append(
            {
                "model_id": model_id,
                "primary_category_id": "panel",
                "category_ids": ["panel"],
                "supports_penalty": False,
                "supports_inference": True,
            }
        )

    gam_rows = data.get("modules", {}).get("gam", {}).get("precision_aligned", {})
    gam_scale = _scale(100000, 10)
    selected_gam = {
        backend: gam_rows.get(f"gam_fixed_large_{backend}")
        for backend in ("numpy", "cupy", "torch")
    }
    selected_gam = {key: value for key, value in selected_gam.items() if value}
    if selected_gam:
        external_time = next(
            float(value["pygam_time"])
            for value in selected_gam.values()
            if value.get("pygam_time") is not None
        )
        for backend, entry in selected_gam.items():
            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-new-modules-{date}",
                    "env_id": env_id,
                    "category_ids": ["nonparametric"],
                    "model_id": "GAM",
                    "case_id": "gam-aligned-large",
                    "method_config_id": "aligned-pygam",
                    "penalty": None,
                    "solver": "gcv",
                    "solver_display": "GCV",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": gam_scale,
                    "source": dict(source),
                    "metrics": {
                        "timing": _timing_ms(float(entry["statgpu_time"]) * 1000.0, filepath),
                        "speedup": _reported_speedup(
                            float(entry["speedup"]), filepath, "pygam", None
                        ),
                        "validation": {
                            "status": "pass" if float(entry["pred_rel_diff"]) <= 0.05 else "warn",
                            "checks": [
                                {
                                    "metric": "prediction_relative_difference",
                                    "operator": "le",
                                    "status": "pass" if float(entry["pred_rel_diff"]) <= 0.05 else "warn",
                                    "value": float(entry["pred_rel_diff"]),
                                    "tolerance": 0.05,
                                    "reference": "pygam",
                                }
                            ],
                            "quality": "computed",
                            "source_file": filepath.name,
                        },
                    },
                }
            )
        runs.append(
            {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-new-modules-{date}",
                "env_id": env_id,
                "category_ids": ["nonparametric"],
                "model_id": "GAM",
                "case_id": "gam-aligned-large",
                "method_config_id": "aligned-pygam",
                "penalty": None,
                "solver": "gcv",
                "solver_display": "GCV",
                "solver_kind": "internal",
                "framework": "pygam",
                "backend": None,
                "scale": gam_scale,
                "source": dict(source),
                "metrics": {
                    "timing": _timing_ms(external_time * 1000.0, filepath, "reported")
                },
            }
        )
        models.append(
            {
                "model_id": "GAM",
                "primary_category_id": "nonparametric",
                "category_ids": ["nonparametric"],
                "supports_penalty": True,
                "supports_inference": False,
            }
        )

    return runs, models, warnings


def parse_p2_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse covariance and nonparametric benchmark sections from P2 results."""
    data = json.loads(filepath.read_text(encoding="utf-8"))
    timestamp = data.get("meta", {}).get("timestamp", "")
    date = timestamp[:10] if timestamp else ""
    source = _source(filepath, date, "parse_p2_benchmark_v1")
    runs: list[dict] = []
    warnings: list[str] = []
    model_entries: dict[str, dict] = {}

    section_specs = {
        "covariance": ("EmpiricalCovariance", "covariance"),
        "nystroem": ("Nystroem", "nonparametric"),
        "rbf_kernel": ("RBFKernel", "nonparametric"),
        "splines": ("BSplineBasis", "nonparametric"),
    }
    benchmarks = data.get("benchmarks", {})
    for section, (model_id, category_id) in section_specs.items():
        for scale_name, entry in benchmarks.get(section, {}).items():
            match = re.fullmatch(r"n=(\d+)p=(\d+)", scale_name)
            if not match:
                warnings.append(f"{filepath.name}: invalid {section} scale '{scale_name}'")
                continue
            n_samples, n_features = map(int, match.groups())
            scale = _scale(n_samples, n_features)
            case_id = f"{section}-{scale['scale_key']}"
            external_ms = entry.get("sklearn_ms")

            for backend in ("numpy", "cupy", "torch"):
                time_ms = entry.get(f"{backend}_ms")
                if time_ms is None:
                    continue
                metrics: dict[str, Any] = {
                    "timing": _timing_ms(float(time_ms), filepath),
                }
                speedup = entry.get(f"{backend}_speedup")
                if speedup and external_ms:
                    metrics["speedup"] = _reported_speedup(
                        float(speedup), filepath, "sklearn", None
                    )
                runs.append(
                    {
                        "run_id": "",
                        "benchmark_session_id": f"{env_id}-p2-{date}",
                        "env_id": env_id,
                        "category_ids": [category_id],
                        "model_id": model_id,
                        "case_id": case_id,
                        "method_config_id": section,
                        "penalty": None,
                        "solver": "auto",
                        "solver_display": "Auto",
                        "solver_kind": "dispatch",
                        "framework": "statgpu",
                        "backend": backend,
                        "scale": scale,
                        "source": dict(source),
                        "metrics": metrics,
                    }
                )

            if external_ms is not None:
                runs.append(
                    {
                        "run_id": "",
                        "benchmark_session_id": f"{env_id}-p2-{date}",
                        "env_id": env_id,
                        "category_ids": [category_id],
                        "model_id": model_id,
                        "case_id": case_id,
                        "method_config_id": section,
                        "penalty": None,
                        "solver": "auto",
                        "solver_display": "Auto",
                        "solver_kind": "dispatch",
                        "framework": "sklearn",
                        "backend": None,
                        "scale": scale,
                        "source": dict(source),
                        "metrics": {
                            "timing": _timing_ms(float(external_ms), filepath, "reported")
                        },
                    }
                )

            model_entries[model_id] = {
                "model_id": model_id,
                "primary_category_id": category_id,
                "category_ids": [category_id],
                "supports_penalty": False,
                "supports_inference": category_id == "covariance",
            }

    return runs, list(model_entries.values()), warnings
