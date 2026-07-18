from __future__ import annotations
"""Complete parser for the June 2026 unsupervised benchmark matrix."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..canonical import make_scale_key, make_scale_label


_PARSER_NAME = "parse_unsupervised_benchmark_v2"
_PARSER_VERSION = "2.0"
_DATE = "2026-06-27"

# The source runner starts from (1K,20), (10K,50), and (100K,100), but
# caps the input feature count at 50 for the estimators below.  Scale labels
# must describe the arrays actually passed to fit, not the uncapped template.
_BASE_SCALES = {
    "small": (1_000, 20),
    "medium": (10_000, 50),
    "large": (100_000, 50),
}

_SIMPLE_MODELS: dict[str, tuple[str, dict[str, Any]]] = {
    "pca": ("PCA", {"n_components": 10}),
    "kmeans": ("KMeans", {"n_clusters": 8, "n_init": 1}),
    "gmm": (
        "GaussianMixture",
        {"n_components": 5, "covariance_type": "diag"},
    ),
    "nmf": ("NMF", {"n_components": 10, "max_iter": 100}),
    "tsvd": ("TruncatedSVD", {"n_components": 10}),
    "ipca": ("IncrementalPCA", {"n_components": 10}),
}


CaseSpec = tuple[str, int, int, str | None, dict[str, Any]]


def _case_specs() -> dict[str, CaseSpec]:
    specs: dict[str, CaseSpec] = {}

    for prefix, (model_id, parameters) in _SIMPLE_MODELS.items():
        for scale_name, (n_samples, n_features) in _BASE_SCALES.items():
            specs[f"{prefix}_{scale_name}"] = (
                model_id,
                n_samples,
                n_features,
                None,
                dict(parameters),
            )

    for scale_name, (n_samples, n_features) in {
        "small": (1_000, 20),
        "medium": (10_000, 50),
    }.items():
        specs[f"agglo_{scale_name}"] = (
            "AgglomerativeClustering",
            n_samples,
            n_features,
            None,
            {"n_clusters": 5},
        )

    for dimension, eps in ((10, 1.0), (50, 3.0)):
        for scale_name, (n_samples, _) in _BASE_SCALES.items():
            specs[f"dbscan_{dimension}d_{scale_name}"] = (
                "DBSCAN",
                n_samples,
                dimension,
                f"{dimension}d",
                {"eps": eps, "min_samples": 5},
            )

    for scale_name, (n_samples, n_features) in {
        "small": (1_000, 20),
        "medium": (10_000, 50),
    }.items():
        specs[f"umap_{scale_name}"] = (
            "UMAP",
            n_samples,
            n_features,
            None,
            {"n_components": 2},
        )

    specs["tsne_small"] = (
        "TSNE",
        1_000,
        20,
        None,
        {"n_components": 2, "max_iter": 250},
    )

    for scale_name, (n_samples, n_features) in _BASE_SCALES.items():
        specs[f"mbkmeans_{scale_name}"] = (
            "MiniBatchKMeans",
            n_samples,
            n_features,
            None,
            {"n_clusters": 8},
        )
        specs[f"mbnmf_{scale_name}"] = (
            "MiniBatchNMF",
            n_samples,
            n_features,
            None,
            {"n_components": 10},
        )

    return specs


def _stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{kind}-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _scale(n_samples: int, n_features: int) -> dict[str, Any]:
    return {
        "scale_key": make_scale_key(n_samples, n_features),
        "n_samples": n_samples,
        "n_features": n_features,
        "label": make_scale_label(n_samples, n_features),
    }


def _source(filepath: Path) -> dict[str, str]:
    return {
        "file": filepath.name,
        "date": _DATE,
        "parser": _PARSER_NAME,
        "parser_version": _PARSER_VERSION,
    }


def _timing_ms(seconds: float, filepath: Path, quality: str = "measured") -> dict[str, Any]:
    return {
        "fit_time_ms": round(float(seconds) * 1000.0, 6),
        "quality": quality,
        "source_file": filepath.name,
    }


def _reported_speedup(value: float, filepath: Path) -> dict[str, Any]:
    return {
        "value": round(float(value), 6),
        "reference_backend": None,
        "reference_framework": "sklearn",
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


def parse_unsupervised_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Expose every scale present in ``unsupervised_20260627.json``.

    The previous parser selected only one representative scale for most models.
    Focused/Full-matrix presentation belongs in the frontend; the canonical
    bundle must retain the complete source matrix.
    """
    data = json.loads(filepath.read_text(encoding="utf-8"))
    runs: list[dict] = []
    warnings: list[str] = []
    model_ids: set[str] = set()

    for prefix, (model_id, n_samples, n_features, variant, parameters) in _case_specs().items():
        entries: dict[str, dict[str, Any]] = {}
        for backend in ("numpy", "cupy", "torch"):
            entry = data.get(f"{prefix}_{backend}")
            if entry and entry.get("time") is not None:
                entries[backend] = entry

        if not entries:
            warnings.append(f"{filepath.name}: no rows for {prefix}")
            continue

        model_ids.add(model_id)
        scale = _scale(n_samples, n_features)
        case_id = _stable_id("case", prefix, scale["scale_key"])
        method_id = _stable_id("method", model_id, variant, parameters)
        numpy_time = entries.get("numpy", {}).get("time")
        external_time = next(
            (
                float(entry["external"])
                for entry in entries.values()
                if entry.get("external") is not None
            ),
            None,
        )

        for backend, entry in entries.items():
            time_s = float(entry["time"])
            metrics: dict[str, Any] = {
                "timing": _timing_ms(time_s, filepath),
            }
            if entry.get("speedup") is not None and external_time is not None:
                metrics["speedup"] = _reported_speedup(
                    float(entry["speedup"]), filepath
                )
            elif backend != "numpy" and numpy_time and time_s > 0:
                metrics["speedup"] = _computed_speedup(
                    float(numpy_time) / time_s, filepath
                )

            run: dict[str, Any] = {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-unsupervised-{_DATE}",
                "env_id": env_id,
                "category_ids": ["unsupervised"],
                "model_id": model_id,
                "case_id": case_id,
                "method_config_id": method_id,
                "penalty": None,
                "solver": "auto",
                "solver_display": "Auto",
                "solver_kind": "dispatch",
                "framework": "statgpu",
                "backend": backend,
                "scale": scale,
                "parameters": dict(parameters),
                "source": _source(filepath),
                "metrics": metrics,
            }
            if variant:
                run["variant"] = variant
            runs.append(run)

        if external_time is not None and external_time > 0:
            external_run: dict[str, Any] = {
                "run_id": "",
                "benchmark_session_id": f"{env_id}-unsupervised-{_DATE}",
                "env_id": env_id,
                "category_ids": ["unsupervised"],
                "model_id": model_id,
                "case_id": case_id,
                "method_config_id": method_id,
                "penalty": None,
                "solver": "auto",
                "solver_display": "Auto",
                "solver_kind": "dispatch",
                "framework": "sklearn",
                "backend": None,
                "scale": scale,
                "parameters": dict(parameters),
                "source": _source(filepath),
                "metrics": {
                    "timing": _timing_ms(
                        external_time, filepath, quality="reported"
                    )
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
