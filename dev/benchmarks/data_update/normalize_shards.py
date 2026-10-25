#!/usr/bin/env python3
"""Normalize every indexed diagnostic shard, without executing any workload.

Requires jsonschema in this Python environment and the separately patched
dashboard checkout. Writes only a fresh external directory.
"""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from preview_export import canonical, digest, read_json, safe_path


def normalize(repo, index_path, output, source_date):
    repo = Path(repo).expanduser().resolve()
    index_path = Path(index_path).expanduser().absolute()
    index = read_json(index_path)
    if not isinstance(index, dict): raise ValueError("Shard index must be an object")
    expected = index.get("index_sha256")
    if index.get("schema_version") != "statgpu-preview-shard-index-v1" or expected != hashlib.sha256(
            canonical({key: value for key, value in index.items() if key != "index_sha256"}).encode()).hexdigest():
        raise ValueError("Invalid shard index schema or hash")
    if index.get("status") != "complete": raise ValueError("Shard index is partial; resolve explicitly failed shard exports first")
    if date.fromisoformat(source_date) < date(2026, 6, 1): raise ValueError("Source date predates dashboard minimum")
    output = Path(output).expanduser().resolve()
    if output.exists() or output.is_relative_to(repo) or output.is_relative_to(index_path.parent.resolve()) or not output.parent.is_dir():
        raise ValueError("Output must be fresh, outside the checkout, with an existing parent")
    converter = repo / "dev/benchmarks/export_qualification_preview.py"
    generator = repo / "dev/benchmarks/generate_benchmark_data.py"
    if not converter.is_file() or not generator.is_file(): raise ValueError("Use the patched dashboard checkout, not the pristine benchmark source checkout")
    entries = index.get("shards")
    if not isinstance(entries, list): raise ValueError("Missing shard list")
    sources = []
    seen = set()
    for item in entries:
        if not isinstance(item, dict) or item.get("status") != "complete": raise ValueError("Incomplete shard entry")
        source = safe_path(index_path.parent, item["path"])
        if source in seen: raise ValueError("Duplicate shard path")
        seen.add(source)
        # Reject FIFOs, oversized JSON and malformed files before digest/open.
        source_document = read_json(source)
        if not isinstance(source_document, dict): raise ValueError("Shard preview must be an object")
        if digest(source) != item.get("sha256"): raise ValueError("Shard checksum mismatch: " + str(source))
        sources.append(source)
    output.mkdir()
    report = {"schema_version": "statgpu-normalized-shards-v1", "status": "complete", "input_index_sha256": digest(index_path),
              "performance_eligible": False, "canonical_dashboard_ready": False, "shards": []}
    for number, source in enumerate(sources, 1):
        folder = output / f"campaign-{number:04d}"
        folder.mkdir()
        manifest = folder / "source_manifest.json"
        strict = folder / "strict-generator"
        commands = [[sys.executable, str(converter), "--preview", str(source), "--out-dir", str(folder / "normalized"),
                     "--manifest-output", str(manifest), "--source-date", source_date],
                    [sys.executable, str(generator), "--manifest", str(manifest), "--deterministic", "--strict-sources",
                     "--out", str(strict / "benchmark_data.json"), "--report", str(strict / "parse_report.json"),
                     "--inventory-out", str(strict / "source_inventory.json")]]
        item = {"preview": str(source), "output": folder.name, "status": "complete", "commands": commands}
        report["shards"].append(item)
        for number_in_shard, command in enumerate(commands, 1):
            try:
                process = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            except (OSError, subprocess.SubprocessError) as exc:
                item.update(status="failed", step=number_in_shard, reason=str(exc))
                report["status"] = "partial"
                break
            (folder / f"step-{number_in_shard}.log").write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                item.update(status="failed", step=number_in_shard, exit_code=process.returncode)
                report["status"] = "partial"
                break
        if item["status"] == "complete":
            data, parsed, inventory = (read_json(strict / filename) for filename in
                                      ("benchmark_data.json", "parse_report.json", "source_inventory.json"))
            if not data["meta"]["generation_id"] == parsed["generation_id"] == inventory["generation_id"]:
                raise ValueError("Generated bundle identifiers differ")
            item["generation_id"] = data["meta"]["generation_id"]
            item["normalized_runs"] = len(data["runs"])
            item["withheld_records"] = len(parsed["issues"])
    (output / "normalization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-date", required=True)
    args = parser.parse_args(argv)
    try:
        report = normalize(args.repo, args.index, args.out_dir, args.source_date)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "shards": len(report["shards"])}))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__": raise SystemExit(main())
