from __future__ import annotations
"""Composite parser for the June 2026 loss-function benchmark.

The source contains robust/quantile regression results plus CoxPH Breslow and
Efron rows. Efron is intentionally handled by the dedicated CoxPH benchmark,
which has richer light/heavy-ties coverage. This module augments the existing
robust/quantile parser with the previously omitted Breslow comparison.
"""

import json
import re
from pathlib import Path
from typing import Any

from .domains import (
    _reported_speedup,
    _scale,
    _source,
    _stable_id,
    _timing_ms,
    parse_loss_functions_benchmark as _parse_robust_quantile_benchmark,
)


_COX_BACKENDS: dict[str, tuple[str | None, str]] = {
    "cpu": ("numpy", "statgpu"),
    "cupy_gpu": ("cupy", "statgpu"),
    "torch_gpu": ("torch", "statgpu"),
    "statsmodels": (None, "statsmodels"),
}


def _breslow_validation(row: dict[str, Any], filepath: Path) -> dict[str, Any]:
    status = str(row.get("status", "PASS")).lower()
    return {
        "status": status,
        "checks": [
            {
                "metric": "max_abs_error",
                "status": status,
                "value": float(row["max_abs_error"]),
                "reference": "statsmodels PHReg",
            }
        ],
        "quality": "measured",
        "source_file": filepath.name,
    }


def parse_loss_functions_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse robust/quantile rows and the aligned CoxPH Breslow benchmark."""
    runs, models, warnings = _parse_robust_quantile_benchmark(filepath, env_id)
    data = json.loads(filepath.read_text(encoding="utf-8"))
    date = data.get("date", "")
    source = _source(filepath, date, "parse_loss_functions_benchmark_v2")
    source["parser_version"] = "2.0"

    # The source is now parsed by this composite v2 parser. Keep provenance
    # consistent across the pre-existing robust/quantile rows as well.
    for run in runs:
        run["source"]["parser"] = source["parser"]
        run["source"]["parser_version"] = source["parser_version"]

    precision_rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in data.get("precision", {}).get("results", []):
        if row.get("model") != "CoxPH" or row.get("ties") != "breslow":
            continue
        backend_raw = str(row.get("backend"))
        if backend_raw in _COX_BACKENDS:
            precision_rows[(int(row["n"]), int(row["p"]), backend_raw)] = row

    performance = data.get("performance", {}).get("coxph_breslow", {})
    speedups = data.get("speedups", {}).get("coxph_breslow", {})
    session_id = f"{env_id}-loss-functions-coxph-breslow-{date}"

    for scale_name, methods in performance.items():
        match = re.fullmatch(r"n(\d+)_p(\d+)", scale_name)
        if not match:
            warnings.append(f"{filepath.name}: invalid CoxPH Breslow scale '{scale_name}'")
            continue

        n_samples, n_features = map(int, match.groups())
        scale = _scale(n_samples, n_features)
        case_id = _stable_id("case", "coxph", "breslow", scale["scale_key"])
        method_config_id = _stable_id("method", "coxph", "breslow", "newton")

        for method_name, entry in methods.items():
            mapping = _COX_BACKENDS.get(method_name)
            if mapping is None or not isinstance(entry, dict):
                continue
            mean_ms = entry.get("mean_ms")
            if mean_ms is None:
                continue

            backend, framework = mapping
            metrics: dict[str, Any] = {
                "timing": _timing_ms(float(mean_ms), filepath),
            }

            if framework == "statgpu":
                speedup = speedups.get(scale_name, {}).get(method_name)
                if speedup is not None and float(speedup) > 0:
                    metrics["speedup"] = _reported_speedup(
                        float(speedup), filepath, "statsmodels", None
                    )

                validation_row = precision_rows.get(
                    (n_samples, n_features, method_name)
                )
                if validation_row is not None and validation_row.get("max_abs_error") is not None:
                    metrics["validation"] = _breslow_validation(
                        validation_row, filepath
                    )

            runs.append(
                {
                    "run_id": "",
                    "benchmark_session_id": session_id,
                    "env_id": env_id,
                    "category_ids": ["survival"],
                    "model_id": "CoxPH",
                    "case_id": case_id,
                    "method_config_id": method_config_id,
                    "loss": "coxph",
                    "penalty": None,
                    "solver": "newton",
                    "solver_display": "Newton",
                    "solver_kind": "manual",
                    "variant": "breslow",
                    "framework": framework,
                    "backend": backend,
                    "scale": scale,
                    "source": dict(source),
                    "metrics": metrics,
                }
            )

    models.append(
        {
            "model_id": "CoxPH",
            "primary_category_id": "survival",
            "category_ids": ["survival"],
            "supports_penalty": False,
            "supports_inference": True,
        }
    )
    return runs, models, warnings
