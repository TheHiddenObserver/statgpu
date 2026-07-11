"""Parse LassoCV combined benchmark — seed-level aggregation."""

import json
import statistics
from pathlib import Path

from ..canonical import make_scale_key, make_scale_label


def parse_lassocv_combined(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """Parse remote_gpu_lassocv_runtime_compare_combined_*.json."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = set()
    warnings = []
    source_date = data.get("date", "")
    config = data.get("config", {})

    source = {
        "file": filepath.name, "date": source_date,
        "parser": "parse_lassocv_combined_v1", "parser_version": "1.0",
    }

    n_samples = config.get("n_samples", 0)
    n_features = config.get("n_features", 0)
    if n_samples == 0:
        warnings.append(f"{filepath.name}: missing n_samples in config")
        return runs, [], warnings

    scale = {
        "scale_key": make_scale_key(n_samples, n_features),
        "n_samples": n_samples, "n_features": n_features,
        "label": make_scale_label(n_samples, n_features),
    }

    methods = data.get("methods", {})
    for method_name, method_data in methods.items():
        seed_runs = method_data.get("runs", [])
        if not seed_runs:
            continue

        # Framework detection
        if method_name.startswith("sklearn"):
            framework = "sklearn"
            backend = None
        elif method_name.endswith("_cpu"):
            framework = "statgpu"
            backend = "numpy"
        elif method_name.endswith("_gpu"):
            framework = "statgpu"
            backend = "cupy"
        elif "statgpu" in method_name:
            framework = "statgpu"
            if "cupy" in method_name:
                backend = "cupy"
            elif "torch" in method_name:
                backend = "torch"
            else:
                backend = "numpy"
        else:
            framework = "statgpu"
            backend = "numpy"

        model_id = "LassoCV"
        models.add(model_id)
        n_seeds = len(seed_runs)

        # Aggregate timing
        times = [r["time_ms"] for r in seed_runs if "time_ms" in r]
        timing_mean = statistics.mean(times) if times else 0
        timing_std = statistics.stdev(times) if len(times) > 1 else 0.0

        # Aggregate n_iter
        niters = [r["n_iter"] for r in seed_runs if "n_iter" in r]
        niter_mean = statistics.mean(niters) if niters else 0
        niter_std = statistics.stdev(niters) if len(niters) > 1 else 0.0

        # Aggregate alpha
        alphas = [r["alpha"] for r in seed_runs if "alpha" in r]
        alpha_mean = statistics.mean(alphas) if alphas else 0
        alpha_std = statistics.stdev(alphas) if len(alphas) > 1 else 0.0

        # Aggregate MSE
        train_mses = [r["train_mse"] for r in seed_runs if "train_mse" in r]
        test_mses = [r["test_mse"] for r in seed_runs if "test_mse" in r]
        train_mse_mean = statistics.mean(train_mses) if train_mses else None
        train_mse_std = statistics.stdev(train_mses) if len(train_mses) > 1 else 0.0
        test_mse_mean = statistics.mean(test_mses) if test_mses else None
        test_mse_std = statistics.stdev(test_mses) if len(test_mses) > 1 else 0.0

        # coef_l2_rel → coef_l2_rel_error
        coef_errors = [r["coef_l2_rel"] for r in seed_runs if "coef_l2_rel" in r]
        coef_l2_rel_mean = statistics.mean(coef_errors) if coef_errors else None
        coef_l2_rel_std = statistics.stdev(coef_errors) if len(coef_errors) > 1 else 0.0

        metrics = {
            "timing": {
                "fit_time_ms": round(timing_mean, 6),
                "std_ms": round(timing_std, 6),
                "sample_count": n_seeds,
                "quality": "measured",
                "source_file": filepath.name,
            },
            "convergence": {
                "n_iter_mean": round(niter_mean, 2),
                "n_iter_std": round(niter_std, 2),
                "quality": "reported",
                "source_file": filepath.name,
            },
        }

        # Prediction
        pred = {"quality": "measured", "source_file": filepath.name}
        if train_mse_mean is not None:
            pred["train_mse"] = round(train_mse_mean, 6)
            pred["train_mse_std"] = round(train_mse_std, 6)
        if test_mse_mean is not None:
            pred["test_mse"] = round(test_mse_mean, 6)
            pred["test_mse_std"] = round(test_mse_std, 6)
        if alpha_mean:
            pred["alpha_mean"] = round(alpha_mean, 6)
            pred["alpha_std"] = round(alpha_std, 6)
        if len(pred) > 2:
            metrics["prediction"] = pred

        # Accuracy
        if coef_l2_rel_mean is not None:
            metrics["accuracy"] = {
                "coef_l2_rel_error": round(coef_l2_rel_mean, 8),
                "coef_l2_rel_error_std": round(coef_l2_rel_std, 8),
                "quality": "computed",
                "source_file": filepath.name,
            }

        run = {
            "run_id": "",
            "env_id": env_id,
            "category_ids": ["linear_models", "penalized_glm"],
            "model_id": model_id,
            "loss": "squared_error",
            "penalty": "l1",
            "solver": "auto",
            "solver_display": "Auto (best)",
            "solver_kind": "dispatch",
            "framework": framework,
            "backend": backend,
            "scale": dict(scale),
            "source": dict(source),
            "metrics": metrics,
            "replicate": {"n_runs": n_seeds, "seed_count": n_seeds},
        }
        runs.append(run)

    model_entries = [
        {"model_id": "LassoCV", "primary_category_id": "linear_models",
         "category_ids": ["linear_models", "penalized_glm"],
         "supports_penalty": True, "supports_inference": False}
    ]
    return runs, model_entries, warnings
