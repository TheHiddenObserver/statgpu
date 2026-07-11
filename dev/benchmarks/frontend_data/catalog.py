"""Catalog and inventory helpers (placeholder — full implementation in A2)."""

import json
import hashlib
from collections import defaultdict
from pathlib import Path

from .canonical import source_sha256, normalize_utf8_bytes


def build_preflight_audit(results_dir: Path, output: dict, parse_report: dict,
                          parser_registry: dict, get_git_sha_fn) -> dict:
    """Build the legacy identity audit fixture for A0."""
    # Source SHA256
    source_hashes = {}
    for filename in parser_registry:
        filepath = results_dir / filename
        if filepath.exists():
            source_hashes[filename] = source_sha256(filepath)

    # Convergence provenance
    conv_prov = {"explicit_converged": 0, "parser_inferred_converged": 0}
    for run in output["runs"]:
        conv = run.get("metrics", {}).get("convergence", {})
        if conv:
            conv_prov["parser_inferred_converged"] += 1

    # Timing provenance
    timing_prov = {}
    for filename in parser_registry:
        short_name = filename.replace("/", "_").replace(".json", "")
        timing_prov[short_name] = {
            "sample_count_known": False,
            "std_ddof": None,
            "std_scope": "unknown",
        }

    # Legacy discriminators
    legacy_disc = {
        "penalized_glm": ["scale_name", "model_key"],
        "glm_solver": ["scale_name", "model_key"],
        "elasticnet_statgpu": ["entry_name"],
        "elasticnet_glmnet": ["dataset_name"],
    }

    # Legacy comparison groups
    legacy_comparison_groups = {
        "penalized_glm_bench_perf_2026-06-22.json": "transitional:penalized-glm-performance",
        "glm_solver_benchmark_2026-06-23.json": "transitional:glm-solver",
        "benchmark_full/benchmark_statgpu_all.json": "transitional:elasticnet-cross-framework",
        "benchmark_full/benchmark_glmnet_all.json": "transitional:elasticnet-cross-framework",
    }

    # Count total JSON files
    catalog_total = 0
    if results_dir.exists():
        catalog_total = len(list(results_dir.rglob("*.json")))

    # Check for duplicate transitional identities
    groups = defaultdict(list)
    for run in output["runs"]:
        key = (
            run["model_id"], run.get("loss", ""), run.get("penalty", ""),
            run.get("solver", ""), run["framework"], str(run["backend"]),
            run["scale"]["scale_key"], run["env_id"],
            run.get("benchmark_session_id", ""),
        )
        groups[key].append(run["run_id"])

    dup_transitional = []
    for key, ids in groups.items():
        if len(ids) > 1:
            dup_transitional.append({"key": str(key), "run_ids": ids})

    # Verify no duplicate run_ids
    seen = set()
    dup_ids = []
    for run in output["runs"]:
        rid = run["run_id"]
        if rid in seen:
            dup_ids.append(rid)
        seen.add(rid)

    return {
        "baseline_git_sha": get_git_sha_fn(),
        "catalog_total": catalog_total,
        "run_count": len(output["runs"]),
        "warning_count": len(parse_report.get("warnings", parse_report.get("issues", []))),
        "duplicate_run_ids": dup_ids,
        "duplicate_transitional_identities": dup_transitional,
        "source_sha256": source_hashes,
        "convergence_provenance": conv_prov,
        "timing_provenance": timing_prov,
        "legacy_discriminators": legacy_disc,
        "legacy_comparison_groups": legacy_comparison_groups,
    }
