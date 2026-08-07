from __future__ import annotations
"""Audited benchmark-source catalog and coverage helpers."""

import fnmatch
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .canonical import source_sha256

CATALOG_PATH = Path("dev/benchmarks/benchmark_source_catalog.json")
COVERAGE_PATH = Path("dev/benchmarks/benchmark_coverage_matrix.json")

CATALOG_CLASSIFICATIONS = {
    "registered_canonical",
    "eligible_unregistered",
    "not_canonical_ready",
    "historical_or_excluded",
    "superseded_or_duplicate",
    "unrelated_json",
    "unclassified",
}
COVERAGE_STATUSES = {
    "canonical_current",
    "partial_canonical",
    "current_evidence_not_canonical_ready",
    "benchmark_data_gap",
    "intentionally_not_benchmarked",
    "not_applicable",
}
_DATE_RE = re.compile(r"(?P<year>20\d{2})[-_]?(?P<month>0[1-9]|1[0-2])[-_]?(?P<day>0[1-9]|[12]\d|3[01])")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_source_catalog(repo_root: Path) -> dict[str, Any]:
    return _load_json(repo_root / CATALOG_PATH)


def load_coverage_matrix(repo_root: Path) -> dict[str, Any]:
    return _load_json(repo_root / COVERAGE_PATH)


def _extract_date(path: str) -> str | None:
    match = _DATE_RE.search(path)
    if not match:
        return None
    return f"{match.group('year')}-{match.group('month')}-{match.group('day')}"


def _matches(rule: dict[str, Any], path: str, source_date: str | None) -> bool:
    if "path" in rule and path != rule["path"]:
        return False
    if "path_glob" in rule and not fnmatch.fnmatch(path, rule["path_glob"]):
        return False
    if "path_regex" in rule and re.search(rule["path_regex"], path) is None:
        return False
    if "date_before" in rule and (source_date is None or source_date >= rule["date_before"]):
        return False
    if "date_on_or_after" in rule and (source_date is None or source_date < rule["date_on_or_after"]):
        return False
    if rule.get("undated") is True and source_date is not None:
        return False
    return True


def _apply_rule(rule: dict[str, Any], path: str, source_date: str | None) -> dict[str, Any]:
    return {
        "path": path,
        "artifact_type": "json",
        "source_date": source_date,
        "classification": rule["classification"],
        "canonical_eligible": bool(rule.get("canonical_eligible", False)),
        "registered": False,
        "source_id": None,
        "parser": rule.get("parser"),
        "parser_version": rule.get("parser_version"),
        "provenance_status": rule.get("provenance_status", "unknown"),
        "timing_protocol_status": rule.get("timing_protocol_status", "unknown"),
        "statistical_alignment_status": rule.get("statistical_alignment_status", "unknown"),
        "reason": rule.get("reason", ""),
        "superseded_by": rule.get("superseded_by"),
        "issue": rule.get("issue"),
        "rule_id": rule["rule_id"],
    }


def discover_catalog_entries(
    repo_root: Path,
    catalog: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Discover and deterministically classify JSON artifacts in declared roots."""
    registered_by_path = {source["path"]: source for source in manifest.get("sources", [])}
    rules = sorted(catalog.get("rules", []), key=lambda item: (item.get("priority", 1000), item["rule_id"]))
    entries: list[dict[str, Any]] = []

    discovered: set[str] = set()
    for root_config in catalog.get("scan_roots", []):
        root_rel = root_config["path"]
        root = repo_root / root_rel
        if not root.exists():
            if root_config.get("required", True):
                raise FileNotFoundError(f"Catalog scan root missing: {root_rel}")
            continue
        pattern = "**/*.json" if root_config.get("recursive", True) else "*.json"
        for artifact in sorted(root.glob(pattern)):
            if not artifact.is_file():
                continue
            rel = artifact.relative_to(repo_root).as_posix()
            if any(fnmatch.fnmatch(rel, excluded) for excluded in root_config.get("exclude_globs", [])):
                continue
            discovered.add(rel)

    for rel in sorted(discovered):
        manifest_source = registered_by_path.get(rel)
        if manifest_source is not None:
            entries.append({
                "path": rel,
                "artifact_type": "json",
                "source_date": manifest_source.get("source_date") or _extract_date(rel),
                "classification": "registered_canonical",
                "canonical_eligible": True,
                "registered": True,
                "source_id": manifest_source["source_id"],
                "parser": manifest_source["parser"],
                "parser_version": manifest_source.get("parser_version"),
                "provenance_status": "complete",
                "timing_protocol_status": "accepted",
                "statistical_alignment_status": "accepted",
                "reason": "Registered in frontend_sources.json and protected by manifest SHA256.",
                "superseded_by": None,
                "issue": None,
                "rule_id": "manifest-registration",
            })
            continue

        source_date = _extract_date(rel)
        matching = next((rule for rule in rules if _matches(rule, rel, source_date)), None)
        if matching is None:
            entries.append({
                "path": rel,
                "artifact_type": "json",
                "source_date": source_date,
                "classification": "unclassified",
                "canonical_eligible": False,
                "registered": False,
                "source_id": None,
                "parser": None,
                "parser_version": None,
                "provenance_status": "unknown",
                "timing_protocol_status": "unknown",
                "statistical_alignment_status": "unknown",
                "reason": "No catalog rule matched this artifact.",
                "superseded_by": None,
                "issue": None,
                "rule_id": None,
            })
        else:
            entries.append(_apply_rule(matching, rel, source_date))

    return entries


def validate_source_catalog(
    catalog: dict[str, Any],
    entries: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    materialized = list(entries)
    paths = [entry["path"] for entry in materialized]
    if len(paths) != len(set(paths)):
        errors.append("catalog contains duplicate artifact paths")

    registered_by_path = {source["path"]: source for source in manifest.get("sources", [])}
    entries_by_path = {entry["path"]: entry for entry in materialized}
    for path, source in registered_by_path.items():
        entry = entries_by_path.get(path)
        if entry is None:
            errors.append(f"registered source missing from catalog: {path}")
            continue
        if entry.get("classification") != "registered_canonical" or not entry.get("registered"):
            errors.append(f"catalog/manifest registration disagreement: {path}")
        if entry.get("source_id") != source.get("source_id"):
            errors.append(f"catalog source_id mismatch: {path}")

    for entry in materialized:
        classification = entry.get("classification")
        if classification not in CATALOG_CLASSIFICATIONS:
            errors.append(f"invalid classification for {entry.get('path')}: {classification}")
        if classification == "unclassified":
            errors.append(f"unclassified artifact: {entry['path']}")
        if classification == "eligible_unregistered" and not entry.get("issue"):
            errors.append(f"eligible unregistered source lacks issue: {entry['path']}")
        if entry.get("canonical_eligible") and classification not in {
            "registered_canonical", "eligible_unregistered"
        }:
            errors.append(f"ineligible classification marked canonical eligible: {entry['path']}")
        if entry.get("registered") and entry["path"] not in registered_by_path:
            errors.append(f"catalog marks unregistered path as registered: {entry['path']}")

    rule_ids = [rule.get("rule_id") for rule in catalog.get("rules", [])]
    if None in rule_ids or len(rule_ids) != len(set(rule_ids)):
        errors.append("catalog rules require unique non-null rule_id values")
    return errors


def validate_coverage_matrix(matrix: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_ids = {source["source_id"] for source in manifest.get("sources", [])}
    seen: set[str] = set()
    for row in matrix.get("capabilities", []):
        capability_id = row.get("capability_id")
        if not capability_id or capability_id in seen:
            errors.append(f"coverage capability_id missing or duplicated: {capability_id!r}")
            continue
        seen.add(capability_id)
        status = row.get("status")
        if status not in COVERAGE_STATUSES:
            errors.append(f"invalid coverage status for {capability_id}: {status}")
        for source_id in row.get("source_ids", []):
            if source_id not in source_ids:
                errors.append(f"{capability_id} references unknown source_id: {source_id}")
        if status in {"benchmark_data_gap", "current_evidence_not_canonical_ready", "partial_canonical"}:
            if not row.get("issue") and not row.get("disposition"):
                errors.append(f"unresolved coverage row lacks issue/disposition: {capability_id}")
        if status == "canonical_current" and not row.get("source_ids"):
            errors.append(f"canonical coverage row lacks source_ids: {capability_id}")
    return errors


def build_inventory_v2(
    entries: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
    available_registered_sources: int,
    parsed_registered_sources: int,
    catalog: dict[str, Any],
    coverage_matrix: dict[str, Any],
) -> dict[str, Any]:
    materialized = list(entries)
    counts = Counter(entry["classification"] for entry in materialized)
    eligible = counts["registered_canonical"] + counts["eligible_unregistered"]
    catalog_digest = hashlib.sha256(
        json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    coverage_digest = hashlib.sha256(
        json.dumps(coverage_matrix, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "inventory_version": "2.0",
        "catalog_version": catalog["catalog_version"],
        "catalog_digest": catalog_digest,
        "coverage_matrix_version": coverage_matrix["coverage_matrix_version"],
        "coverage_matrix_digest": coverage_digest,
        "discovered_json_artifacts": len(materialized),
        "classified_candidate_sources": len(materialized) - counts["unclassified"],
        "eligible_sources": eligible,
        "registered_sources": len(manifest.get("sources", [])),
        "available_registered_sources": available_registered_sources,
        "parsed_registered_sources": parsed_registered_sources,
        "eligible_unregistered_sources": counts["eligible_unregistered"],
        "not_canonical_ready_sources": counts["not_canonical_ready"],
        "historical_or_excluded_sources": (
            counts["historical_or_excluded"]
            + counts["superseded_or_duplicate"]
            + counts["unrelated_json"]
        ),
        "superseded_or_duplicate_sources": counts["superseded_or_duplicate"],
        "unrelated_json_artifacts": counts["unrelated_json"],
        "unclassified_artifacts": counts["unclassified"],
    }


def build_preflight_audit(results_dir: Path, output: dict, parse_report: dict,
                          parser_registry: dict, get_git_sha_fn) -> dict:
    """Build the legacy identity audit fixture for A0."""
    source_hashes = {}
    for filename in parser_registry:
        filepath = results_dir / filename
        if filepath.exists():
            source_hashes[filename] = source_sha256(filepath)

    conv_prov = {"explicit_converged": 0, "parser_inferred_converged": 0}
    for run in output["runs"]:
        if run.get("metrics", {}).get("convergence", {}):
            conv_prov["parser_inferred_converged"] += 1

    timing_prov = {}
    for filename in parser_registry:
        short_name = filename.replace("/", "_").replace(".json", "")
        timing_prov[short_name] = {
            "sample_count_known": False,
            "std_ddof": None,
            "std_scope": "unknown",
        }

    legacy_disc = {
        "penalized_glm": ["scale_name", "model_key"],
        "glm_solver": ["scale_name", "model_key"],
        "elasticnet_statgpu": ["entry_name"],
        "elasticnet_glmnet": ["dataset_name"],
    }
    legacy_comparison_groups = {
        "penalized_glm_bench_perf_2026-06-22.json": "transitional:penalized-glm-performance",
        "glm_solver_benchmark_2026-06-23.json": "transitional:glm-solver",
        "benchmark_full/benchmark_statgpu_all.json": "transitional:elasticnet-cross-framework",
        "benchmark_full/benchmark_glmnet_all.json": "transitional:elasticnet-cross-framework",
    }

    catalog_total = len(list(results_dir.rglob("*.json"))) if results_dir.exists() else 0
    groups = defaultdict(list)
    for run in output["runs"]:
        key = (
            run["model_id"], run.get("loss", ""), run.get("penalty", ""),
            run.get("solver", ""), run["framework"], str(run["backend"]),
            run["scale"]["scale_key"], run["env_id"],
            run.get("benchmark_session_id", ""),
        )
        groups[key].append(run["run_id"])

    dup_transitional = [
        {"key": str(key), "run_ids": ids}
        for key, ids in groups.items() if len(ids) > 1
    ]
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
