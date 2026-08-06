#!/usr/bin/env python3
"""
Generate unified benchmark data JSON for the statgpu frontend dashboard.

Usage:
    python dev/benchmarks/generate_benchmark_data.py \
        --out frontend/public/data/benchmark_data.json \
        --report frontend/public/data/parse_report.json \
        --inventory-out frontend/public/data/source_inventory.json

    python dev/benchmarks/generate_benchmark_data.py --check  # validate only

This wrapper preserves the established parser/generator implementation while
adding the audited source-catalog and coverage contracts used by canonical mode.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path for 'from dev.benchmarks.frontend_data import ...'
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dev.benchmarks.frontend_data import (
    generate as _core_generate,
    validate_output,
    validate_against_schema,
    get_git_sha,
    CATEGORIES,
    BACKEND_MAP,
    FRAMEWORK_MAP,
    SCALE_CONFIG,
    SOLVER_KIND_MAP,
    SOLVER_DISPLAY_MAP,
    FAMILY_MODEL_MAP,
    SPEEDUP_REFERENCE_BY_SOURCE,
    make_scale_key,
    make_scale_label,
    make_run_id,
    _short_hash,
    parse_family_penalty_solver,
    normalize_utf8_bytes,
    source_sha256,
)
from dev.benchmarks.frontend_data.catalog import (
    build_inventory_v2,
    discover_catalog_entries,
    load_coverage_matrix,
    load_source_catalog,
    validate_coverage_matrix,
    validate_source_catalog,
)


class _InventoryV2(dict):
    """Inventory mapping with non-serialized read aliases for old Python callers."""

    _READ_ALIASES = {
        "available_sources": "available_registered_sources",
        "parsed_sources": "parsed_registered_sources",
    }

    def __missing__(self, key):
        target = self._READ_ALIASES.get(key)
        if target is None:
            raise KeyError(key)
        return self[target]


def _canonical_json_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _strip_generation_id(value: dict) -> dict:
    result = deepcopy(value)
    if "meta" in result:
        result["meta"].pop("generation_id", None)
    else:
        result.pop("generation_id", None)
    return result


def _assign_bundle_generation_id(
    output: dict,
    parse_report: dict,
    inventory: dict,
) -> None:
    payload = {
        "benchmark_data": _strip_generation_id(output),
        "parse_report": _strip_generation_id(parse_report),
        "source_inventory": _strip_generation_id(inventory),
    }
    generation_id = _canonical_json_digest(payload)
    output["meta"]["generation_id"] = generation_id
    parse_report["generation_id"] = generation_id
    inventory["generation_id"] = generation_id


def _is_repository_manifest(manifest: Optional[dict]) -> bool:
    if manifest is None:
        return False
    from dev.benchmarks.frontend_data.registry import load_manifest

    repository_manifest = load_manifest(_repo_root)
    if repository_manifest is None:
        return False
    expected = [
        (source.get("source_id"), source.get("path"))
        for source in repository_manifest.get("sources", [])
    ]
    observed = [
        (source.get("source_id"), source.get("path"))
        for source in manifest.get("sources", [])
    ]
    return observed == expected


def generate(
    results_dir: Path,
    deterministic: bool = False,
    manifest: Optional[dict] = None,
    strict_sources: bool = False,
) -> tuple[dict, dict, dict]:
    """Generate the benchmark bundle and apply audited inventory semantics.

    Custom/transitional manifests retain the legacy inventory contract so test
    fixtures and downstream callers are not silently coupled to this repository's
    catalog. The committed canonical manifest emits inventory v2.
    """
    output, parse_report, legacy_inventory = _core_generate(
        results_dir,
        deterministic=deterministic,
        manifest=manifest,
        strict_sources=strict_sources,
    )
    if not _is_repository_manifest(manifest):
        return output, parse_report, legacy_inventory

    assert manifest is not None
    catalog_policy = load_source_catalog(_repo_root)
    coverage_matrix = load_coverage_matrix(_repo_root)
    if coverage_matrix.get("catalog_version") != catalog_policy.get("catalog_version"):
        raise ValueError(
            "coverage matrix catalog_version does not match source catalog"
        )

    entries = discover_catalog_entries(_repo_root, catalog_policy, manifest)
    catalog_errors = validate_source_catalog(catalog_policy, entries, manifest)
    coverage_errors = validate_coverage_matrix(coverage_matrix, manifest)
    contract_errors = catalog_errors + coverage_errors
    if contract_errors:
        preview = "; ".join(contract_errors[:10])
        suffix = "" if len(contract_errors) <= 10 else f"; and {len(contract_errors) - 10} more"
        raise ValueError(f"benchmark catalog/coverage validation failed: {preview}{suffix}")

    available = sum(
        1 for source in manifest.get("sources", [])
        if (_repo_root / source["path"]).exists()
    )
    registered_ids = {source["source_id"] for source in manifest.get("sources", [])}
    parsed = len({
        run.get("source", {}).get("source_id")
        for run in output.get("runs", [])
        if run.get("source", {}).get("source_id") in registered_ids
    })
    catalog_snapshot = {
        "catalog_version": catalog_policy["catalog_version"],
        "minimum_source_date": catalog_policy["minimum_source_date"],
        "entries": entries,
    }
    inventory = _InventoryV2(build_inventory_v2(
        entries,
        manifest,
        available_registered_sources=available,
        parsed_registered_sources=parsed,
        catalog=catalog_snapshot,
        coverage_matrix=coverage_matrix,
    ))
    inventory["catalog_policy_digest"] = _canonical_json_digest(catalog_policy)
    inventory["catalog_entries"] = entries
    inventory["coverage_status_counts"] = dict(sorted(Counter(
        row["status"] for row in coverage_matrix.get("capabilities", [])
    ).items()))
    inventory["generation_id"] = ""
    _assign_bundle_generation_id(output, parse_report, inventory)
    return output, parse_report, inventory


def main() -> None:
    from dev.benchmarks.frontend_data import cli as _cli

    # Route the established CLI through the audited wrapper without duplicating
    # parser, validation, or transactional-write logic.
    _cli.generate = generate
    _cli.main()


__all__ = [
    "generate",
    "validate_output",
    "validate_against_schema",
    "main",
    "get_git_sha",
    "CATEGORIES",
    "BACKEND_MAP",
    "FRAMEWORK_MAP",
    "SCALE_CONFIG",
    "SOLVER_KIND_MAP",
    "SOLVER_DISPLAY_MAP",
    "FAMILY_MODEL_MAP",
    "SPEEDUP_REFERENCE_BY_SOURCE",
    "make_scale_key",
    "make_scale_label",
    "make_run_id",
    "_short_hash",
    "parse_family_penalty_solver",
    "normalize_utf8_bytes",
    "source_sha256",
]

if __name__ == "__main__":
    main()
