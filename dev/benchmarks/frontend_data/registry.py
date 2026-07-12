from __future__ import annotations
"""Parser registry — hardcoded fallback plus manifest-driven canonical mode."""

import json
from datetime import date
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
    parse_loss_functions_benchmark,
    parse_pr74_inference_benchmark,
    parse_unsupervised_benchmark,
    parse_new_modules_benchmark,
    parse_p2_benchmark,
)

MINIMUM_DASHBOARD_SOURCE_DATE = date(2026, 6, 1)

# Hardcoded fallback is intentionally restricted to current June 2026 sources.
PARSER_REGISTRY: dict[str, dict] = {
    "penalized_glm_bench_perf_2026-06-22.json": {
        "parser": parse_penalized_glm_bench_perf,
        "env_id": "remote-p100",
    },
    "glm_solver_benchmark_2026-06-23.json": {
        "parser": parse_glm_solver_benchmark,
        "env_id": "remote-p100",
    },
}

# Parser function lookup by name. Parsers may remain available for historical
# files even when those files are not registered in the dashboard manifest.
PARSER_FUNCTIONS = {
    "penalized_glm_bench_perf": parse_penalized_glm_bench_perf,
    "glm_solver_benchmark": parse_glm_solver_benchmark,
    "elasticnet_benchmark_full": parse_elasticnet_benchmark_full,
    "coxph_efron_bench": parse_coxph_efron_bench,
    "comprehensive_validation": parse_comprehensive_validation,
    "coxph_package_comparison": parse_coxph_package_comparison,
    "lassocv_combined": parse_lassocv_combined,
    "knockoff_benchmark": parse_knockoff_benchmark,
    "loss_functions_benchmark": parse_loss_functions_benchmark,
    "ordered_inference_benchmark": parse_pr74_inference_benchmark,
    "unsupervised_benchmark": parse_unsupervised_benchmark,
    "new_modules_benchmark": parse_new_modules_benchmark,
    "p2_benchmark": parse_p2_benchmark,
}


def validate_manifest_source_dates(manifest: dict) -> None:
    """Reject dashboard manifests containing pre-June-2026 or undated sources."""
    configured_minimum = manifest.get("minimum_source_date")
    if configured_minimum is None:
        raise ValueError("dashboard manifest missing minimum_source_date")

    try:
        minimum = date.fromisoformat(configured_minimum)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid dashboard minimum_source_date: {configured_minimum!r}"
        ) from exc

    if minimum < MINIMUM_DASHBOARD_SOURCE_DATE:
        raise ValueError(
            "dashboard minimum_source_date cannot be earlier than 2026-06-01"
        )

    for source in manifest.get("sources", []):
        raw_date = source.get("source_date")
        source_id = source.get("source_id", source.get("path", "<unknown>"))
        if raw_date is None:
            raise ValueError(f"dashboard source {source_id} missing source_date")
        try:
            source_date = date.fromisoformat(raw_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"dashboard source {source_id} has invalid source_date {raw_date!r}"
            ) from exc
        if source_date < minimum:
            raise ValueError(
                f"dashboard source {source_id} is dated {source_date}; "
                f"minimum allowed date is {minimum}"
            )


def load_manifest(repo_root: Path) -> Optional[dict]:
    """Load and validate the canonical frontend source manifest."""
    manifest_path = repo_root / "dev" / "benchmarks" / "frontend_sources.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest_source_dates(manifest)
    return manifest


def build_registry_from_manifest(manifest: dict) -> dict[str, dict]:
    """Build parser registry from a validated June-or-later manifest."""
    # Enforce the dashboard date policy for every programmatic entry point, not
    # only manifests loaded from disk via ``load_manifest``.
    validate_manifest_source_dates(manifest)

    registry = {}
    for src in manifest["sources"]:
        parser_name = src["parser"]
        if parser_name not in PARSER_FUNCTIONS:
            raise ValueError(f"Unknown parser: {parser_name}")
        config = {
            "parser": PARSER_FUNCTIONS[parser_name],
            "env_id": src["env_id"],
            "source_id": src["source_id"],
            "comparison_id": src.get("comparison_id", src["source_id"]),
            "required": src.get("required", True),
            "allowed_issue_codes": set(src.get("allowed_issue_codes", [])),
        }
        if src.get("sha256"):
            config["sha256"] = src["sha256"]
        registry[src["path"]] = config
    return registry
