"""Parse comprehensive validation and Cox package comparison benchmarks."""

import json
from pathlib import Path

from ..canonical import FAMILY_MODEL_MAP, make_scale_key, make_scale_label


def parse_comprehensive_validation(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """Parse bench_validation_comprehensive.json."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = set()
    warnings = []
    config = data.get("config", {})
    n = config.get("n", 500)
    p = config.get("p", 10)
    source_date = data.get("date", data.get("generated", "2026-04"))

    scale_key = make_scale_key(n, p)
    scale = {
        "scale_key": scale_key, "n_samples": n, "n_features": p,
        "label": make_scale_label(n, p),
    }

    source = {
        "file": filepath.name, "date": str(source_date)[:10],
        "parser": "parse_comprehensive_validation_v1", "parser_version": "1.0",
    }

    # Normalize family names to canonical form
    FAMILY_ALIASES = {
        "poisson": "poisson", "gamma": "gamma", "tweedie": "tweedie",
        "invgaussian": "inverse_gaussian", "negbinom": "negative_binomial",
    }
    ext_val = data.get("external_validation", {})
    for family, result in ext_val.items():
        if not isinstance(result, dict):
            continue
        family_key = family.lower().replace("_", " ").replace(" ", "_")
        family_key = FAMILY_ALIASES.get(family_key, family_key)
        model_id = FAMILY_MODEL_MAP.get(family_key, f"Penalized{family}Regression")
        models.add(model_id)

        status = result.get("status", "PASS").lower()
        if status not in ("pass", "warn", "fail"):
            status = "pass"

        run = {
            "run_id": "",
            "env_id": env_id,
            "category_ids": ["glm"],
            "model_id": model_id,
            "loss": family.lower(),
            "penalty": None,
            "solver": "irls",
            "solver_display": "IRLS",
            "solver_kind": "manual",
            "framework": "statgpu",
            "backend": "numpy",
            "scale": dict(scale),
            "source": dict(source),
            "metrics": {
                "accuracy": {
                    "coef_max_abs_diff": result.get("max_coef_diff"),
                    "bse_max_abs_diff": result.get("max_bse_diff"),
                    "reference": "statsmodels",
                    "quality": "measured",
                    "source_file": filepath.name,
                },
                "validation": {
                    "status": status,
                    "checks": [
                        {"metric": "coef_max_abs_diff", "status": status,
                         "value": result.get("max_coef_diff"),
                         "tolerance": result.get("threshold"),
                         "reference": "statsmodels"},
                    ],
                    "quality": "measured",
                    "source_file": filepath.name,
                },
            },
        }
        runs.append(run)

    model_entries = [
        {"model_id": m, "primary_category_id": "glm", "category_ids": ["glm"],
         "supports_penalty": False, "supports_inference": True}
        for m in sorted(models)
    ]
    return runs, model_entries, warnings


def parse_coxph_package_comparison(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """Parse coxph_package_comparison_*.json."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = {"CoxPH"}
    warnings = []

    source = {
        "file": filepath.name, "date": "2026-04-19",
        "parser": "parse_coxph_package_comparison_v1", "parser_version": "1.0",
    }

    framework_map = {
        "statgpu": ("statgpu", "numpy"),
        "lifelines": ("lifelines", None),
        "scikit_survival": ("scikit_survival", None),
        "statsmodels": ("statsmodels", None),
    }

    for entry in data:
        if not isinstance(entry, dict):
            continue
        n = entry.get("n_samples", 0)
        p = entry.get("n_features", 0)
        if n == 0:
            continue

        scale = {
            "scale_key": make_scale_key(n, p), "n_samples": n, "n_features": p,
            "label": make_scale_label(n, p),
        }

        for pkg_name, pkg_data in entry.items():
            if pkg_name in ("n_samples", "n_features"):
                continue
            if not isinstance(pkg_data, dict):
                continue

            fw_info = framework_map.get(pkg_name)
            if fw_info is None:
                warnings.append(f"{filepath.name}: unknown package '{pkg_name}'")
                continue
            framework, backend = fw_info

            timing = pkg_data.get("time")
            c_index = pkg_data.get("c_index")

            metrics = {}
            if timing is not None:
                metrics["timing"] = {
                    "fit_time_ms": timing,
                    "quality": "measured",
                    "source_file": filepath.name,
                }
            if c_index is not None:
                metrics["prediction"] = {
                    "c_index": c_index,
                    "quality": "measured",
                    "source_file": filepath.name,
                }

            if not metrics:
                continue

            run = {
                "run_id": "",
                "env_id": env_id,
                "category_ids": ["survival"],
                "model_id": "CoxPH",
                "loss": "coxph",
                "penalty": None,
                "solver": "newton",
                "solver_display": "Newton",
                "solver_kind": "manual",
                "framework": framework,
                "backend": backend,
                "scale": dict(scale),
                "source": dict(source),
                "metrics": metrics,
            }
            runs.append(run)

    model_entries = [
        {"model_id": "CoxPH", "primary_category_id": "survival", "category_ids": ["survival"],
         "supports_penalty": False, "supports_inference": True}
    ]
    return runs, model_entries, warnings
