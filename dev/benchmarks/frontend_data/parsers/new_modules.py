from __future__ import annotations
"""Combined parser for the June 2026 panel, GAM, and ANOVA bundle."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..canonical import make_scale_key, make_scale_label
from .domains import parse_new_modules_benchmark


_ANOVA_SCALES = {
    "small": (100, 5),
    "medium": (10_000, 10),
    "large": (100_000, 20),
}

_ANOVA_MODELS = {
    "f_oneway": ("OneWayANOVA", "One-way F test", "one-way"),
    "f_twoway": ("TwoWayANOVA", "Two-way F test", "two-way"),
    "f_welch": ("WelchANOVA", "Welch ANOVA", "welch"),
    "tukey_hsd": ("TukeyHSD", "Tukey HSD", "post-hoc"),
    "bonferroni": ("BonferroniCorrection", "Bonferroni", "multiple-testing"),
}

_GAM_SCALES = {
    "small": (1_000, 3),
    "medium": (10_000, 5),
    "large": (100_000, 10),
}

_PARSER_NAME = "parse_new_modules_with_anova_benchmark_v2"
_PARSER_VERSION = "1.2"


def _stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{kind}-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _compact_count(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000}M"
    if value >= 1_000 and value % 1_000 == 0:
        return f"{value // 1_000}K"
    return str(value)


def _anova_scale(func_name: str, n_per_group: int, n_groups: int) -> dict[str, Any]:
    if func_name == "f_twoway":
        n_cells = 12  # benchmark generator uses a 3 x 4 design
        total = n_per_group * n_cells
        return {
            "scale_key": make_scale_key(total, n_cells),
            "n_samples": total,
            "n_features": n_cells,
            "label": f"{_compact_count(n_per_group)}/cell · 3×4 cells",
        }

    total = n_per_group * n_groups
    return {
        "scale_key": make_scale_key(total, n_groups),
        "n_samples": total,
        "n_features": n_groups,
        "label": f"{_compact_count(n_per_group)}/group · {n_groups} groups",
    }


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


def _computed_speedup(value: float, filepath: Path) -> dict[str, Any]:
    return {
        "value": round(float(value), 6),
        "reference_backend": "numpy",
        "reference_framework": "statgpu",
        "reported_semantics": "computed",
        "quality": "computed",
        "source_file": filepath.name,
    }


def _reported_speedup(value: float, filepath: Path, reference: str) -> dict[str, Any]:
    return {
        "value": round(float(value), 6),
        "reference_backend": None,
        "reference_framework": reference,
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


def _append_all_aligned_gam_rows(
    runs: list[dict],
    models: list[dict],
    warnings: list[str],
    data: dict[str, Any],
    filepath: Path,
    env_id: str,
    date: str,
) -> None:
    """Expose every aligned GAM scale instead of only the previous large row."""
    aligned = data.get("modules", {}).get("gam", {}).get("precision_aligned", {})
    emitted = 0

    for scale_name, (n_samples, n_features) in _GAM_SCALES.items():
        selected = {
            backend: aligned.get(f"gam_fixed_{scale_name}_{backend}")
            for backend in ("numpy", "cupy", "torch")
        }
        selected = {backend: row for backend, row in selected.items() if row}
        if not selected:
            warnings.append(f"{filepath.name}: no aligned GAM rows for {scale_name}")
            continue

        external_seconds = next(
            (
                float(row["pygam_time"])
                for row in selected.values()
                if row.get("pygam_time") is not None
            ),
            None,
        )
        scale = _scale(n_samples, n_features)
        case_id = _stable_id("case", "gam", "aligned", scale_name, scale["scale_key"])
        method_id = _stable_id("method", "gam", "aligned-pygam")

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
                    "variant": "aligned-pygam",
                    "penalty": None,
                    "solver": "gcv",
                    "solver_display": "GCV",
                    "solver_kind": "internal",
                    "framework": "statgpu",
                    "backend": backend,
                    "scale": scale,
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
                    "variant": "aligned-pygam",
                    "penalty": None,
                    "solver": "gcv",
                    "solver_display": "GCV",
                    "solver_kind": "internal",
                    "framework": "pygam",
                    "backend": None,
                    "scale": scale,
                    "source": _source(filepath, date),
                    "metrics": {
                        "timing": _timing_ms(external_seconds, filepath, "reported")
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
    """Parse all aligned Panel/GAM scales plus all ANOVA functions."""
    runs, models, warnings = parse_new_modules_benchmark(filepath, env_id)
    data = json.loads(filepath.read_text(encoding="utf-8"))
    date = data.get("date", "")

    # The legacy domain parser emitted only the large aligned GAM row. Replace
    # those rows with the complete small/medium/large aligned matrix.
    runs = [run for run in runs if run.get("model_id") != "GAM"]
    models = [model for model in models if model.get("model_id") != "GAM"]

    # Keep one parser identity for every run emitted from this canonical source.
    for run in runs:
        run["source"]["parser"] = _PARSER_NAME
        run["source"]["parser_version"] = _PARSER_VERSION

    _append_all_aligned_gam_rows(runs, models, warnings, data, filepath, env_id, date)

    anova = data.get("modules", {}).get("anova", {})
    performance = anova.get("performance", {})
    external = anova.get("external_comparison", {})

    for scale_name, (n_per_group, n_groups) in _ANOVA_SCALES.items():
        scale_rows = performance.get(scale_name, {})
        for func_name, (model_id, solver_display, variant) in _ANOVA_MODELS.items():
            backend_rows = scale_rows.get(func_name, {})
            if not backend_rows:
                warnings.append(
                    f"{filepath.name}: no ANOVA rows for {scale_name}/{func_name}"
                )
                continue

            scale = _anova_scale(func_name, n_per_group, n_groups)
            case_id = _stable_id("case", "anova", scale_name, func_name)
            method_id = _stable_id("method", "anova", func_name)

            aligned_rows: dict[str, dict[str, Any]] = {}
            if func_name == "f_oneway":
                for backend in ("numpy", "cupy", "torch"):
                    row = external.get(f"anova_{scale_name}_f_oneway_{backend}")
                    if row and row.get("statgpu_time") is not None:
                        aligned_rows[backend] = row

            numpy_seconds = None
            if aligned_rows:
                numpy_seconds = aligned_rows.get("numpy", {}).get("statgpu_time")
            elif backend_rows.get("numpy", {}).get("time") is not None:
                numpy_seconds = backend_rows["numpy"]["time"]

            for backend in ("numpy", "cupy", "torch"):
                if aligned_rows:
                    entry = aligned_rows.get(backend)
                    seconds = entry.get("statgpu_time") if entry else None
                else:
                    entry = backend_rows.get(backend)
                    seconds = entry.get("time") if entry else None
                if seconds is None:
                    continue

                metrics: dict[str, Any] = {
                    "timing": _timing_ms(float(seconds), filepath),
                }
                if backend != "numpy" and numpy_seconds and float(seconds) > 0:
                    metrics["speedup"] = _computed_speedup(
                        float(numpy_seconds) / float(seconds), filepath
                    )

                if func_name == "f_oneway" and entry and entry.get("f_rel_diff") is not None:
                    rel_diff = float(entry["f_rel_diff"])
                    metrics["validation"] = {
                        "status": "pass" if rel_diff <= 1e-10 else "warn",
                        "checks": [
                            {
                                "metric": "f_statistic_relative_difference",
                                "operator": "le",
                                "status": "pass" if rel_diff <= 1e-10 else "warn",
                                "value": rel_diff,
                                "tolerance": 1e-10,
                                "reference": "scipy",
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
                        "category_ids": ["anova"],
                        "model_id": model_id,
                        "case_id": case_id,
                        "method_config_id": method_id,
                        "variant": variant,
                        "penalty": None,
                        "solver": func_name,
                        "solver_display": solver_display,
                        "solver_kind": "internal",
                        "framework": "statgpu",
                        "backend": backend,
                        "scale": scale,
                        "source": _source(filepath, date),
                        "metrics": metrics,
                    }
                )

            if func_name == "f_oneway" and aligned_rows:
                external_seconds = next(
                    (
                        row.get("external_time")
                        for row in aligned_rows.values()
                        if row.get("external_time") is not None
                    ),
                    None,
                )
                if external_seconds is not None:
                    runs.append(
                        {
                            "run_id": "",
                            "benchmark_session_id": f"{env_id}-new-modules-{date}",
                            "env_id": env_id,
                            "category_ids": ["anova"],
                            "model_id": model_id,
                            "case_id": case_id,
                            "method_config_id": method_id,
                            "variant": variant,
                            "penalty": None,
                            "solver": func_name,
                            "solver_display": solver_display,
                            "solver_kind": "internal",
                            "framework": "scipy",
                            "backend": None,
                            "scale": scale,
                            "source": _source(filepath, date),
                            "metrics": {
                                "timing": _timing_ms(
                                    float(external_seconds), filepath, "reported"
                                )
                            },
                        }
                    )

            models.append(
                {
                    "model_id": model_id,
                    "primary_category_id": "anova",
                    "category_ids": ["anova"],
                    "supports_penalty": False,
                    "supports_inference": func_name
                    in {"f_oneway", "f_twoway", "f_welch"},
                }
            )

    return runs, models, warnings
