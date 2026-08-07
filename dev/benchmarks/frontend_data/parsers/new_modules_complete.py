from __future__ import annotations
"""Complete Panel/GAM coverage layered on the June module/ANOVA parser."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..canonical import make_scale_key, make_scale_label
from .new_modules import (
    parse_new_modules_with_anova_benchmark as _parse_new_modules_with_anova_benchmark,
)


_PANEL_SCALES = {
    "medium": (10_000, 10),
    "large": (100_000, 20),
}

_PANEL_MODELS = {
    "PanelOLS": "PanelOLS",
    "RE": "RandomEffects",
}

_GAM_SCALES = {
    "small": (1_000, 3, 15),
    "medium": (10_000, 5, 20),
    "large": (100_000, 10, 25),
}

_PARSER_NAME = "parse_new_modules_with_anova_benchmark_v4"
_PARSER_VERSION = "1.4"


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


def _timing_ms(seconds: float, filepath: Path, quality: str = "measured") -> dict[str, Any]:
    return {
        "fit_time_ms": round(float(seconds) * 1000.0, 6),
        "quality": quality,
        "source_file": filepath.name,
    }


def _reported_speedup(
    value: float, filepath: Path, reference_framework: str
) -> dict[str, Any]:
    return {
        "value": round(float(value), 6),
        "reference_backend": None,
        "reference_framework": reference_framework,
        "reported_semantics": "reported_by_runner",
        "quality": "reported",
        "source_file": filepath.name,
    }


def _source(filepath: Path, date: str) -> dict[str, str]:
    return {
        "file": filepath.name,
        "date": date,
        "parser": _PARSER_NAME,
        "parser_version": _PARSER_VERSION,
    }


def _append_all_aligned_panel_rows(
    runs: list[dict],
    models: list[dict],
    warnings: list[str],
    data: dict[str, Any],
    filepath: Path,
    env_id: str,
    date: str,
) -> None:
    """Expose both medium and large aligned linearmodels comparisons."""
    panel_rows = data.get("modules", {}).get("panel", {}).get("external_comparison", {})
    emitted_models: set[str] = set()

    for scale_name, (n_samples, n_features) in _PANEL_SCALES.items():
        scale = _scale(n_samples, n_features)

        for token, model_id in _PANEL_MODELS.items():
            selected = {
                backend: panel_rows.get(f"panel_{scale_name}_{token}_{backend}")
                for backend in ("numpy", "cupy", "torch")
            }
            selected = {backend: row for backend, row in selected.items() if row}
            if not selected:
                warnings.append(
                    f"{filepath.name}: no aligned panel rows for {scale_name}/{token}"
                )
                continue

            external_seconds = next(
                (
                    float(row["external_time"])
                    for row in selected.values()
                    if row.get("external_time") is not None
                ),
                None,
            )
            case_id = _stable_id(
                "case", "panel", scale_name, token, scale["scale_key"]
            )
            method_id = _stable_id("method", "panel", token, "closed_form")

            for backend, row in selected.items():
                statgpu_seconds = row.get("statgpu_time")
                if statgpu_seconds is None:
                    continue

                metrics: dict[str, Any] = {
                    "timing": _timing_ms(float(statgpu_seconds), filepath),
                }
                if row.get("speedup") is not None:
                    metrics["speedup"] = _reported_speedup(
                        float(row["speedup"]), filepath, "linearmodels"
                    )
                if row.get("coef_rel_diff") is not None:
                    metrics["accuracy"] = {
                        "coef_l2_rel_error": float(row["coef_rel_diff"]),
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
                        "method_config_id": method_id,
                        "variant": "aligned-linearmodels",
                        "penalty": None,
                        "solver": "closed_form",
                        "solver_display": "Closed form",
                        "solver_kind": "internal",
                        "framework": "statgpu",
                        "backend": backend,
                        "scale": scale,
                        "source": _source(filepath, date),
                        "metrics": metrics,
                    }
                )

            if external_seconds is not None:
                runs.append(
                    {
                        "run_id": "",
                        "benchmark_session_id": f"{env_id}-new-modules-{date}",
                        "env_id": env_id,
                        "category_ids": ["panel"],
                        "model_id": model_id,
                        "case_id": case_id,
                        "method_config_id": method_id,
                        "variant": "aligned-linearmodels",
                        "penalty": None,
                        "solver": "closed_form",
                        "solver_display": "Closed form",
                        "solver_kind": "internal",
                        "framework": "linearmodels",
                        "backend": None,
                        "scale": scale,
                        "source": _source(filepath, date),
                        "metrics": {
                            "timing": _timing_ms(
                                external_seconds, filepath, quality="reported"
                            )
                        },
                    }
                )

            emitted_models.add(model_id)

    for model_id in sorted(emitted_models):
        models.append(
            {
                "model_id": model_id,
                "primary_category_id": "panel",
                "category_ids": ["panel"],
                "supports_penalty": False,
                "supports_inference": True,
            }
        )


def _append_complete_gam_rows(
    runs: list[dict],
    models: list[dict],
    warnings: list[str],
    data: dict[str, Any],
    filepath: Path,
    env_id: str,
    date: str,
) -> None:
    """Expose both source GAM comparison configurations at every scale.

    ``external_comparison`` represents the ordinary pyGAM comparison, while
    ``precision_aligned`` uses uniform knots, gamma=1.4 and fixed lambda to
    tighten numerical alignment.  They are not duplicate timings and therefore
    require distinct variants and method identities.
    """
    gam = data.get("modules", {}).get("gam", {})
    configurations = (
        {
            "variant": "pygam-comparison",
            "section": gam.get("external_comparison", {}),
            "key_template": "gam_{scale}_{backend}",
            "external_key": "external_time",
            "extra_parameters": {"alignment": "source_default"},
        },
        {
            "variant": "aligned-pygam",
            "section": gam.get("precision_aligned", {}),
            "key_template": "gam_fixed_{scale}_{backend}",
            "external_key": "pygam_time",
            "extra_parameters": {
                "alignment": "uniform_knots",
                "gamma": 1.4,
            },
        },
    )
    emitted = 0

    for config in configurations:
        rows = config["section"]
        for scale_name, (n_samples, n_features, n_splines) in _GAM_SCALES.items():
            selected = {
                backend: rows.get(
                    config["key_template"].format(
                        scale=scale_name, backend=backend
                    )
                )
                for backend in ("numpy", "cupy", "torch")
            }
            selected = {backend: row for backend, row in selected.items() if row}
            if not selected:
                warnings.append(
                    f"{filepath.name}: no GAM rows for {config['variant']}/{scale_name}"
                )
                continue

            external_seconds = next(
                (
                    float(row[config["external_key"]])
                    for row in selected.values()
                    if row.get(config["external_key"]) is not None
                ),
                None,
            )
            scale = _scale(n_samples, n_features)
            parameters = {
                "lam": 1.0,
                "n_splines": n_splines,
                **config["extra_parameters"],
            }
            case_id = _stable_id(
                "case", "gam", config["variant"], scale_name, scale["scale_key"]
            )
            method_id = _stable_id(
                "method", "gam", config["variant"], parameters
            )

            for backend, row in selected.items():
                statgpu_seconds = row.get("statgpu_time")
                if statgpu_seconds is None:
                    continue
                rel_diff = row.get("pred_rel_diff")
                metrics: dict[str, Any] = {
                    "timing": _timing_ms(float(statgpu_seconds), filepath),
                }
                if row.get("speedup") is not None:
                    metrics["speedup"] = _reported_speedup(
                        float(row["speedup"]), filepath, "pygam"
                    )
                if rel_diff is not None:
                    rel_diff = float(rel_diff)
                    status = "pass" if rel_diff <= 0.05 else "warn"
                    metrics["validation"] = {
                        "status": status,
                        "checks": [
                            {
                                "metric": "prediction_relative_difference",
                                "operator": "le",
                                "status": status,
                                "value": rel_diff,
                                "tolerance": 0.05,
                                "reference": "pygam",
                            }
                        ],
                        "quality": "computed",
                        "source_file": filepath.name,
                    }

                runs.append(
                    {
                        "run_id": "",
                        "benchmark_session_id": f"{env_id}-new-modules-{date}",
                        "env_id": env_id,
                        "category_ids": ["nonparametric"],
                        "model_id": "GAM",
                        "case_id": case_id,
                        "method_config_id": method_id,
                        "variant": config["variant"],
                        "penalty": None,
                        "solver": "fixed_lam",
                        "solver_display": "Fixed λ=1.0",
                        "solver_kind": "internal",
                        "framework": "statgpu",
                        "backend": backend,
                        "scale": scale,
                        "parameters": dict(parameters),
                        "source": _source(filepath, date),
                        "metrics": metrics,
                    }
                )
                emitted += 1

            if external_seconds is not None:
                runs.append(
                    {
                        "run_id": "",
                        "benchmark_session_id": f"{env_id}-new-modules-{date}",
                        "env_id": env_id,
                        "category_ids": ["nonparametric"],
                        "model_id": "GAM",
                        "case_id": case_id,
                        "method_config_id": method_id,
                        "variant": config["variant"],
                        "penalty": None,
                        "solver": "fixed_lam",
                        "solver_display": "Fixed λ=1.0",
                        "solver_kind": "internal",
                        "framework": "pygam",
                        "backend": None,
                        "scale": scale,
                        "parameters": dict(parameters),
                        "source": _source(filepath, date),
                        "metrics": {
                            "timing": _timing_ms(
                                external_seconds, filepath, quality="reported"
                            )
                        },
                    }
                )
                emitted += 1

    if emitted:
        models.append(
            {
                "model_id": "GAM",
                "primary_category_id": "nonparametric",
                "category_ids": ["nonparametric"],
                "supports_penalty": True,
                "supports_inference": False,
            }
        )


def parse_new_modules_with_anova_benchmark(
    filepath: Path, env_id: str
) -> tuple[list[dict], list[dict], list[str]]:
    """Parse complete Panel/GAM configurations plus the full ANOVA matrix."""
    runs, models, warnings = _parse_new_modules_with_anova_benchmark(filepath, env_id)
    data = json.loads(filepath.read_text(encoding="utf-8"))
    date = data.get("date", "")

    panel_model_ids = set(_PANEL_MODELS.values())
    runs = [
        run
        for run in runs
        if "panel" not in run.get("category_ids", [])
        and run.get("model_id") != "GAM"
    ]
    models = [
        model
        for model in models
        if model.get("model_id") not in panel_model_ids | {"GAM"}
    ]

    # One canonical parser identity is used for every row emitted from the source.
    for run in runs:
        run["source"]["parser"] = _PARSER_NAME
        run["source"]["parser_version"] = _PARSER_VERSION

    _append_all_aligned_panel_rows(
        runs, models, warnings, data, filepath, env_id, date
    )
    _append_complete_gam_rows(
        runs, models, warnings, data, filepath, env_id, date
    )
    return runs, models, warnings
