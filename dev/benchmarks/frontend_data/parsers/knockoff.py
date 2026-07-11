"""Parse knockoff benchmark — feature selection with selection metrics."""

import hashlib
import json
from pathlib import Path

from ..canonical import make_scale_key, make_scale_label


# Method → (model_id, variant, framework, backend)
KNOCKOFF_METHOD_MAP = {
    "knockoff_fixedx_numpy":   ("KnockoffFilter", "fixed_x", "statgpu", "numpy"),
    "knockoff_modelx_numpy":   ("KnockoffFilter", "model_x", "statgpu", "numpy"),
    "knockoff_fixedx_cupy":    ("KnockoffFilter", "fixed_x", "statgpu", "cupy"),
    "knockoff_modelx_cupy":    ("KnockoffFilter", "model_x", "statgpu", "cupy"),
    "marginal_corr_topk":     ("MarginalCorrelationSelector", "top_k", "statgpu", "numpy"),
    "statgpu_lasso_topk":      ("LassoSelector", "top_k", "statgpu", "numpy"),
    "sklearn_lasso_cv":        ("LassoCV", "cv", "sklearn", None),
    "knockpy_gaussian_lasso":  ("KnockoffFilter", "gaussian_lasso", "knockpy", None),
}


def parse_knockoff_benchmark(filepath: Path, env_id: str) -> tuple[list[dict], list[dict], list[str]]:
    """Parse benchmark_knockoff_vs_baselines_*.json."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = []
    models = set()
    warnings = []
    source_date = data.get("date", "")
    config = data.get("config", {})

    n_samples = config.get("n_samples", 0)
    n_features = config.get("n_features", 0)
    if n_samples == 0:
        warnings.append(f"{filepath.name}: missing n_samples")
        return runs, [], warnings

    scale = {
        "scale_key": make_scale_key(n_samples, n_features),
        "n_samples": n_samples, "n_features": n_features,
        "label": make_scale_label(n_samples, n_features),
    }
    source = {
        "file": filepath.name, "date": source_date,
        "parser": "parse_knockoff_benchmark_v1", "parser_version": "1.0",
    }

    # Build canonical case_id from shared benchmark parameters
    config = data.get("config", {})
    identity_params = {k: config[k] for k in ("n_samples", "n_features", "n_signal", "noise_scale", "rho", "q") if k in config}
    if identity_params:
        case_payload = json.dumps(identity_params, sort_keys=True, separators=(",", ":"))
        case_id = "case-" + hashlib.sha256(case_payload.encode()).hexdigest()[:16]
    else:
        case_id = "default"

    methods = data.get("methods", {})
    for method_name, method_data in methods.items():
        if not isinstance(method_data, dict):
            warnings.append(f"{filepath.name}: method '{method_name}' unavailable (null in source)")
            continue  # code: METHOD_UNAVAILABLE — environment capability, not malformed
        method_info = KNOCKOFF_METHOD_MAP.get(method_name)
        if method_info is None:
            warnings.append(f"{filepath.name}: unknown method '{method_name}'")
            continue

        model_id, variant, framework, backend = method_info
        models.add(model_id)

        # Use aggregate data (not per-seed runs)
        aggregate = method_data.get("aggregate", {}) if isinstance(method_data, dict) else {}
        if not aggregate:
            # Fallback: compute from per-seed runs
            seed_runs = method_data.get("runs", [])
            if seed_runs:
                aggregate = _compute_aggregate(seed_runs)
            else:
                continue

        metrics = {}

        # Timing
        time_mean = aggregate.get("time_ms_mean")
        time_std = aggregate.get("time_ms_std", 0)
        if time_mean is not None:
            metrics["timing"] = {
                "fit_time_ms": round(time_mean, 6),
                "std_ms": round(time_std, 6),
                "quality": "measured",
                "source_file": filepath.name,
            }
            n_seeds = aggregate.get("n_runs", 0)
            if n_seeds:
                metrics["timing"]["sample_count"] = n_seeds
                metrics["timing"]["std_ddof"] = 0
                metrics["timing"]["std_scope"] = "replicates"

        # Selection metrics
        sel = {"quality": "measured", "source_file": filepath.name}
        sel_fields = [
            ("precision_mean", "precision"), ("precision_std", "precision_std"),
            ("recall_mean", "recall"), ("recall_std", "recall_std"),
            ("fdp_mean", "fdp"), ("fdp_std", "fdp_std"),
            ("f1_mean", "f1"), ("f1_std", "f1_std"),
            ("n_selected_mean", "n_selected_mean"), ("n_selected_std", "n_selected_std"),
            ("estimated_fdr_mean", "estimated_fdr"), ("estimated_fdr_std", "estimated_fdr_std"),
            ("jaccard_truth_mean", "jaccard_truth"), ("jaccard_truth_std", "jaccard_truth_std"),
        ]
        has_sel = False
        for src_key, dst_key in sel_fields:
            if src_key in aggregate:
                sel[dst_key] = round(aggregate[src_key], 6)
                has_sel = True
        if has_sel:
            # target_fdr from config q
            if "q" in config:
                sel["target_fdr"] = config["q"]
            metrics["selection"] = sel

        # Replicate
        n_seeds = aggregate.get("n_runs", len(method_data.get("runs", [])))
        replicate = {"n_runs": n_seeds, "seed_count": n_seeds}

        run = {
            "run_id": "",
            "env_id": env_id,
            "category_ids": ["feature_selection"],
            "model_id": model_id,
            "case_id": case_id,
            "variant": variant,
            "framework": framework,
            "backend": backend,
            "scale": dict(scale),
            "source": dict(source),
            "metrics": metrics,
            "replicate": replicate,
        }
        runs.append(run)

    model_entries = []
    for m in sorted(models):
        model_entries.append({
            "model_id": m,
            "primary_category_id": "feature_selection",
            "category_ids": ["feature_selection"],
            "supports_penalty": False,
            "supports_inference": False,
        })

    return runs, model_entries, warnings


def _compute_aggregate(seed_runs: list[dict]) -> dict:
    """Fallback: compute aggregate from per-seed runs."""
    import statistics
    n = len(seed_runs)
    numeric_fields = [
        "time_ms", "precision", "recall", "fdp", "f1",
        "n_selected", "estimated_fdr", "jaccard_truth",
    ]
    result = {"n_runs": n}
    for field in numeric_fields:
        values = [r[field] for r in seed_runs if field in r and r[field] is not None]
        if values:
            result[f"{field}_mean"] = statistics.mean(values)
            if len(values) > 1:
                result[f"{field}_std"] = statistics.pstdev(values)
            else:
                result[f"{field}_std"] = 0.0
    return result
