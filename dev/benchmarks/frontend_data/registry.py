"""Parser registry — hardcoded in A1, manifest-driven in A2."""

import json
from pathlib import Path
from typing import Optional

from .parsers import (
    parse_penalized_glm_bench_perf,
    parse_glm_solver_benchmark,
    parse_elasticnet_benchmark_full,
    parse_coxph_efron_bench,
    parse_comprehensive_validation,
    parse_coxph_package_comparison,
    parse_lassocv_combined,
    parse_knockoff_benchmark,
)

# Hardcoded fallback (A1 transitional)
PARSER_REGISTRY: dict[str, dict] = {
    "penalized_glm_bench_perf_2026-06-22.json": {
        "parser": parse_penalized_glm_bench_perf,
        "env_id": "remote-p100",
    },
    "glm_solver_benchmark_2026-06-23.json": {
        "parser": parse_glm_solver_benchmark,
        "env_id": "remote-p100",
    },
    "benchmark_full/benchmark_statgpu_all.json": {
        "parser": parse_elasticnet_benchmark_full,
        "env_id": "remote-p100",
    },
    "benchmark_full/benchmark_glmnet_all.json": {
        "parser": parse_elasticnet_benchmark_full,
        "env_id": "remote-p100",
    },
}

# Parser function lookup by name
PARSER_FUNCTIONS = {
    "penalized_glm_bench_perf": parse_penalized_glm_bench_perf,
    "glm_solver_benchmark": parse_glm_solver_benchmark,
    "elasticnet_benchmark_full": parse_elasticnet_benchmark_full,
    "coxph_efron_bench": parse_coxph_efron_bench,
    "comprehensive_validation": parse_comprehensive_validation,
    "coxph_package_comparison": parse_coxph_package_comparison,
    "lassocv_combined": parse_lassocv_combined,
    "knockoff_benchmark": parse_knockoff_benchmark,
}


def load_manifest(repo_root: Path) -> Optional[dict]:
    """Load the frontend_sources.json manifest. Returns None if not found."""
    manifest_path = repo_root / "dev" / "benchmarks" / "frontend_sources.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_registry_from_manifest(manifest: dict) -> dict[str, dict]:
    """Build parser registry from manifest. Keys are canonical paths."""
    registry = {}
    for src in manifest["sources"]:
        parser_name = src["parser"]
        if parser_name not in PARSER_FUNCTIONS:
            raise ValueError(f"Unknown parser: {parser_name}")
        registry[src["path"]] = {
            "parser": PARSER_FUNCTIONS[parser_name],
            "env_id": src["env_id"],
            "source_id": src["source_id"],
            "comparison_id": src.get("comparison_id", src["source_id"]),
            "sha256": src.get("sha256", ""),
            "required": src.get("required", True),
            "allowed_issue_codes": set(src.get("allowed_issue_codes", [])),
        }
    return registry
