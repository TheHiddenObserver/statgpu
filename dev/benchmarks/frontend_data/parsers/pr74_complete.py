from __future__ import annotations
"""Complete parser for the PR #74 ordered and inference benchmark."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..canonical import make_scale_key, make_scale_label
from .pr74 import parse_pr74_inference_benchmark as _parse_pr74_inference_benchmark


_PARSER_NAME = "parse_pr74_inference_benchmark_v2"
_PARSER_VERSION = "2.0"
_DATE = "2026-07-12"

_METHODS: dict[str, dict[str, Any]] = {
    "sandwich": {
        "model_id": "PenalizedLogisticRegression",
        "category_ids": ["penalized_glm"],
        "loss": "logistic",
        "penalty": "l2",
        "variant": "hc0-sandwich",
        "parameters": {
            "alpha": 0.01,
            "compute_inference": True,
            "inference_method": "sandwich",
            "cov_type": "hc0",
            "timing_scope": "fit_plus_inference",
        },
        "solver_display": "Auto + HC0 inference",
    },
    "oracle": {
        "model_id": "PenalizedLogisticRegression",
        "category_ids": ["penalized_glm"],
        "loss": "logistic",
        "penalty": "scad",
        "variant": "oracle-inference",
        "parameters": {
            "alpha": 0.1,
            "compute_inference": True,
            "inference_method": "oracle",
            "timing_scope": "fit_plus_inference",
        },
        "solver_display": "Auto + oracle inference",
    },
    "bootstrap": {
        "model_id": "PenalizedLinearRegression",
        "category_ids": ["linear_models", "penalized_glm"],
        "loss": "squared_error",
        "penalty": "l1",
        "variant": "bootstrap-inference",
        "parameters": {
            "alpha": 0.05,
            "compute_inference": True,
            "inference_method": "bootstrap",
            "n_bootstrap": 50,
            "timing_scope": "fit_plus_inference",
        },
        "solver_display": "Auto + bootstrap inference",
    },
}

_BACKENDS = {"NumPy": "numpy", "CuPy": "cupy", "Torch": "torch"}


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


def _timing_ms(seconds: float, filepath: Path) -> dict[str, Any]:
    return {
        "fit_time_ms": round(float(seconds) * 1000.0, 6),
        "quality": "measured",
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


def parse_pr74_inference_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse every method emitted by ``bench_pr74_results.py``.

    The previous parser retained Ordered Logit/Probit and Quantile inference but
    silently dropped penalized-logistic sandwich/oracle inference and Lasso
    bootstrap inference.  Those are distinct model configurations and are now
    emitted with explicit fit-plus-inference timing scope.
    """
    runs, models, warnings = _parse_pr74_inference_benchmark(filepath, env_id)
    data = json.loads(filepath.read_text(encoding="utf-8"))

    # Use one truthful parser identity/date for the complete source.
    for run in runs:
        run["source"] = _source(filepath)

    for key, backend_rows in data.items():
        match = re.fullmatch(r"(sandwich|oracle|bootstrap)_n(\d+)_p(\d+)", key)
        if not match:
            continue

        method, n_raw, p_raw = match.groups()
        spec = _METHODS[method]
        n_samples, n_features = int(n_raw), int(p_raw)
        scale = _scale(n_samples, n_features)
        numpy_time = backend_rows.get("NumPy", {}).get("time")
        case_id = _stable_id("case", method, scale["scale_key"])
        method_id = _stable_id(
            "method",
            spec["model_id"],
            spec["loss"],
            spec["penalty"],
            spec["variant"],
            spec["parameters"],
        )

        for backend_name, entry in backend_rows.items():
            backend = _BACKENDS.get(backend_name)
            if backend is None or not entry.get("ok", True):
                if backend is not None:
                    warnings.append(
                        f"{filepath.name}: {key}/{backend_name} did not complete"
                    )
                continue
            time_s = entry.get("time")
            if time_s is None:
                warnings.append(f"{filepath.name}: {key}/{backend_name} has no timing")
                continue

            metrics: dict[str, Any] = {
                "timing": _timing_ms(float(time_s), filepath),
                "inference": {
                    "bse": float(entry["bse0"]),
                    "ok": True,
                    "quality": "measured",
                    "source_file": filepath.name,
                },
            }
            if backend != "numpy" and numpy_time and float(time_s) > 0:
                metrics["speedup"] = _computed_speedup(
                    float(numpy_time) / float(time_s), filepath
                )

            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": f"{env_id}-pr74-inference-{_DATE}",
                    "env_id": env_id,
                    "category_ids": list(spec["category_ids"]),
                    "model_id": spec["model_id"],
                    "case_id": case_id,
                    "method_config_id": method_id,
                    "variant": spec["variant"],
                    "loss": spec["loss"],
                    "penalty": spec["penalty"],
                    "solver": "auto",
                    "solver_display": spec["solver_display"],
                    "solver_kind": "dispatch",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": scale,
                    "parameters": dict(spec["parameters"]),
                    "source": _source(filepath),
                    "metrics": metrics,
                }
            )

    existing = {model["model_id"] for model in models}
    for spec in _METHODS.values():
        model_id = spec["model_id"]
        if model_id in existing:
            continue
        models.append(
            {
                "model_id": model_id,
                "primary_category_id": spec["category_ids"][0],
                "category_ids": list(spec["category_ids"]),
                "supports_penalty": True,
                "supports_inference": True,
            }
        )
        existing.add(model_id)

    return runs, models, warnings
