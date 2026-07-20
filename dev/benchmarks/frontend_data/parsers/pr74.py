from __future__ import annotations
"""Combined ordered and quantile inference parser for PR #74 results."""

import json
import re
from pathlib import Path
from typing import Any

from .domains import (
    _BACKENDS,
    _computed_speedup,
    _scale,
    _source,
    _stable_id,
    _timing_ms,
    parse_ordered_inference_benchmark,
)


def parse_pr74_inference_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse ordered models plus quantile kernel/bootstrap GPU inference."""
    runs, models, warnings = parse_ordered_inference_benchmark(filepath, env_id)
    data = json.loads(filepath.read_text(encoding="utf-8"))
    source = _source(filepath, "", "parse_pr74_inference_benchmark_v1")

    for key, backend_rows in data.items():
        match = re.fullmatch(r"quantile_(kernel|bootstrap)_n(\d+)_p(\d+)", key)
        if not match:
            continue
        method, n_raw, p_raw = match.groups()
        n_samples, n_features = int(n_raw), int(p_raw)
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
            solver = str(
                entry.get("solver")
                or ("kernel_sandwich" if method == "kernel" else "batched_pinball_fista")
            )
            metrics: dict[str, Any] = {
                "timing": _timing_ms(float(time_s) * 1000.0, filepath),
                "inference": {
                    "bse": float(entry["bse0"]),
                    "ok": bool(entry.get("ok", True)),
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
                    "benchmark_session_id": f"{env_id}-pr74-inference",
                    "env_id": env_id,
                    "category_ids": ["robust_quantile"],
                    "model_id": "QuantileRegression",
                    "case_id": _stable_id("case", "quantile", method, scale["scale_key"]),
                    "method_config_id": _stable_id("method", "quantile", method, solver),
                    "variant": method,
                    "loss": "quantile",
                    "penalty": None,
                    "solver": solver,
                    "solver_display": solver.replace("_", " ").title(),
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": scale,
                    "source": dict(source),
                    "metrics": metrics,
                }
            )

    models.append(
        {
            "model_id": "QuantileRegression",
            "primary_category_id": "robust_quantile",
            "category_ids": ["robust_quantile"],
            "supports_penalty": False,
            "supports_inference": True,
        }
    )
    return runs, models, warnings
