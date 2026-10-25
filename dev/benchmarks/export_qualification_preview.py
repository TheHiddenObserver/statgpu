#!/usr/bin/env python3
"""Normalize a diagnostic bridge to a NEW local dashboard bundle, never publish.

The canonical source manifest and measured artifacts are never changed. The
local dashboard can also load the original preview JSON directly and display
all records, including records that cannot be safely normalized here.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from dev.benchmarks.frontend_data.parsers.qualification_preview import _read, normalize_preview, CATEGORY
from dev.benchmarks.frontend_data.cli import validate_output, validate_against_schema, validate_semantic


def build_bundle(path):
    path = Path(path).expanduser().absolute()
    preview = _read(path)
    runs, models, withheld = normalize_preview(preview, path, "diagnostic-preview")
    environments = []
    for env_id in sorted({run["env_id"] for run in runs}):
        environments.append({"env_id": env_id, "label": "Diagnostic evidence environment " + env_id[-12:],
                             "cpu": "See raw evidence; not inferred", "gpu": "See raw evidence; not inferred"})
    if not environments:
        environments = [{"env_id": "diagnostic-preview-empty", "label": "No representable diagnostic rows",
                         "cpu": "Not recorded", "gpu": "Not recorded"}]
    comparisons = [{"comparison_id": cid, "label": "Diagnostic display group (not a speedup comparison)",
                    "env_id": next(run["env_id"] for run in runs if run["comparison_id"] == cid)}
                   for cid in sorted({run["comparison_id"] for run in runs})]
    if not comparisons:
        comparisons = [{"comparison_id": "diagnostic-preview-empty", "label": "No published comparisons",
                        "env_id": environments[0]["env_id"]}]
    data = {"schema_version": "1.2.0", "generated": "1970-01-01T00:00:00+00:00",
            "meta": {"generator": "dev/benchmarks/export_qualification_preview.py", "git_sha": "diagnostic-export", "generation_id": ""},
            "environments": environments, "categories": [CATEGORY], "models": models,
            "frameworks": [{"framework_id": "statgpu", "display_name": "statgpu", "external": False, "backend_policy": "required"}],
            "comparisons": comparisons, "runs": runs}
    report = {"report_version": "2.0", "generation_id": "", "files_seen": 1, "files_parsed": 1,
              "files_skipped": 0, "runs_generated": len(runs), "issues": [
                  {"source_id": "transitional:qualification-preview", "file": path.name, "parser": "qualification_preview",
                   "parser_version": "1.0", "comparison_id": "", "code": row["reason"], "severity": "info",
                   "message": row["record_id"] + ": retained in raw diagnostic preview at " + str(row.get("raw_pointer"))}
                  for row in withheld]}
    inventory = {"inventory_version": "2.0", "catalog_version": "diagnostic-preview-1", "generation_id": "",
                 "discovered_json_artifacts": 1, "classified_candidate_sources": 1, "eligible_sources": 0,
                 "registered_sources": 0, "available_registered_sources": 0, "parsed_registered_sources": 0,
                 "eligible_unregistered_sources": 0, "not_canonical_ready_sources": 1,
                 "historical_or_excluded_sources": 0, "unclassified_artifacts": 0,
                 "catalog_entries": [], "coverage_status_counts": {"diagnostic_only": len(runs)}}
    bundle = {"benchmark_data": data, "parse_report": report, "source_inventory": inventory}
    from dev.benchmarks.generate_benchmark_data import _assign_bundle_generation_id
    _assign_bundle_generation_id(data, report, inventory)
    errors = validate_output(data) + validate_against_schema(data) + validate_semantic(data)
    if errors: raise ValueError("Diagnostic bundle validation failed: " + "; ".join(errors[:10]))
    return bundle


def build_external_manifest(path, bundle, source_date):
    """Create source registration for an explicit external diagnostic bundle only."""
    parsed_date = date.fromisoformat(source_date)
    if parsed_date < date(2026, 6, 1): raise ValueError("Source date predates dashboard minimum")
    from dev.benchmarks.frontend_data.canonical import source_sha256
    path = Path(path).expanduser().absolute()
    sha = source_sha256(path)
    source_id = "qualification-preview-" + parsed_date.strftime("%Y%m%d") + "-" + sha[:12]
    environments = {entry["env_id"]: {key: value for key, value in entry.items() if key != "env_id"}
                    for entry in bundle["benchmark_data"]["environments"]}
    environments["diagnostic-preview"] = {"label": "Diagnostic source container, distinct per-record environments retained",
                                            "cpu": "See raw source", "gpu": "See raw source"}
    return {"manifest_version": "1.0", "minimum_source_date": "2026-06-01", "catalog_total": 1,
            "diagnostic_only": True, "canonical_dashboard_ready": False,
            "source_date_semantics": "Date supplied by operator for this new diagnostic source; never rewrites measured source identity",
            "environments": environments,
            "frameworks": {"statgpu": {"display_name": "statgpu", "external": False, "backend_policy": "required"}},
            "comparisons": {source_id: {"label": "Diagnostic display only; not a performance comparison", "env_id": "diagnostic-preview"}},
            "sources": [{"source_id": source_id, "comparison_id": source_id, "path": str(path),
                         "original_path": str(path), "sha256": sha, "parser": "qualification_preview",
                         "parser_version": "1.0", "env_id": "diagnostic-preview", "required": True,
                         "allowed_issue_codes": ["METHOD_UNAVAILABLE"], "source_date": source_date}]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--manifest-output", type=Path, help="Optional new external manifest, never updates the repository manifest")
    parser.add_argument("--source-date", help="Explicit source date YYYY-MM-DD required with --manifest-output")
    args = parser.parse_args(argv)
    if args.check == bool(args.out_dir): parser.error("Use exactly one of --check or --out-dir")
    if args.manifest_output and (args.check or not args.source_date): parser.error("Manifest output requires --out-dir and explicit --source-date")
    if args.source_date and not args.manifest_output: parser.error("--source-date requires --manifest-output")
    try:
        bundle = build_bundle(args.preview)
        manifest = build_external_manifest(args.preview, bundle, args.source_date) if args.manifest_output else None
        if args.manifest_output:
            target = args.manifest_output.expanduser().resolve()
            if target.exists() or target.is_relative_to(ROOT): raise ValueError("Manifest output must be a fresh path outside checkout")
        if args.out_dir:
            out = args.out_dir.expanduser().resolve()
            if out.exists(): raise ValueError("Refusing existing directory")
            if out.is_relative_to(ROOT): raise ValueError("Diagnostic export must stay outside checkout; canonical assets are untouched")
            if not out.parent.is_dir(): raise ValueError("Output parent must exist")
            out.mkdir()
            for key, document in bundle.items():
                (out / (key + ".json")).write_text(json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            if manifest is not None:
                with target.open("x", encoding="utf-8") as stream:
                    json.dump(manifest, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
                    stream.write("\n")
        print(json.dumps({"normalized_diagnostic_runs": len(bundle["benchmark_data"]["runs"]),
                          "withheld_records": len(bundle["parse_report"]["issues"]),
                          "performance_eligible": False, "canonical_dashboard_ready": False}))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__": raise SystemExit(main())
