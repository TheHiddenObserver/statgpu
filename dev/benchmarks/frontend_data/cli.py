"""CLI for benchmark frontend data generation."""

import argparse
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

from .canonical import CATEGORIES
from .registry import PARSER_REGISTRY
from .catalog import build_preflight_audit
from .models import merge_model_entries


def get_git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def generate(results_dir: Path, deterministic: bool = False) -> tuple[dict, dict]:
    """Generate unified benchmark_data.json from results_dir."""
    all_runs: list[dict] = []
    all_models: dict[str, dict] = {}
    all_warnings: list[dict] = []
    files_seen = 0
    files_parsed = 0

    for filename, config in PARSER_REGISTRY.items():
        filepath = results_dir / filename
        files_seen += 1
        if not filepath.exists():
            all_warnings.append({"file": filename, "reason": "file not found"})
            continue

        parser_fn = config["parser"]
        env_id = config["env_id"]
        try:
            runs, model_entries, warns = parser_fn(filepath, env_id)
            all_runs.extend(runs)
            for m in model_entries:
                mid = m["model_id"]
                all_models[mid] = merge_model_entries(all_models.get(mid, {}), m)
            for w in warns:
                all_warnings.append({"file": filename, "reason": w})
            files_parsed += 1
        except Exception as e:
            all_warnings.append({"file": filename, "reason": f"parse error: {e}"})

    environments = [
        {
            "env_id": "remote-p100",
            "label": "Tesla P100 + Xeon 8163",
            "gpu": "NVIDIA Tesla P100-16GB",
            "cpu": "12× Intel Xeon Platinum 8163 @ 2.50GHz",
            "host": "hz-4.matpool.com",
        }
    ]

    if deterministic:
        generated_ts = "1970-01-01T00:00:00+00:00"
        git_sha = "deterministic"
    else:
        generated_ts = datetime.now(timezone.utc).isoformat()
        git_sha = get_git_sha()

    output = {
        "schema_version": "1.0.0",
        "generated": generated_ts,
        "meta": {
            "generator": "dev/benchmarks/generate_benchmark_data.py",
            "git_sha": git_sha,
        },
        "environments": environments,
        "categories": CATEGORIES,
        "models": sorted(all_models.values(), key=lambda x: x["model_id"]),
        "runs": all_runs,
    }

    parse_report = {
        "files_seen": files_seen,
        "files_parsed": files_parsed,
        "files_skipped": files_seen - files_parsed,
        "runs_generated": len(all_runs),
        "warnings": all_warnings,
    }

    return output, parse_report


# Validation functions (migrated from old module)
def _load_schema() -> dict | None:
    schema_path = Path(__file__).resolve().parent.parent / "benchmark_frontend_schema.json"
    if not schema_path.exists():
        return None
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_jsonschema_validator():
    try:
        from jsonschema import Draft202012Validator
        return Draft202012Validator
    except ImportError:
        return None


def validate_against_schema(output: dict) -> list[str]:
    schema = _load_schema()
    if schema is None:
        return ["Schema file not found, skipping schema validation"]

    Validator = _get_jsonschema_validator()
    if Validator is not None:
        validator = Validator(schema)
        schema_errors = sorted(validator.iter_errors(output), key=lambda e: list(e.path))
        if not schema_errors:
            return []
        return [f"{' → '.join(str(p) for p in e.absolute_path)}: {e.message}" for e in schema_errors[:50]]

    # Fallback
    errors = []
    runs_schema = schema.get("properties", {}).get("runs", {})
    run_props = runs_schema.get("items", {}).get("properties", {})
    run_required = runs_schema.get("items", {}).get("required", [])

    for run in output.get("runs", []):
        rid = run.get("run_id", "?")
        for field in run_required:
            if field not in run:
                errors.append(f"{rid}: missing required field '{field}' (schema)")
        for field_name in ("framework", "backend", "solver_kind"):
            schema_field = run_props.get(field_name, {})
            enum_vals = schema_field.get("enum", [])
            if enum_vals:
                val = run.get(field_name)
                if val not in enum_vals:
                    errors.append(f"{rid}: {field_name}='{val}' not in schema enum {enum_vals}")
    return errors


def validate_output(output: dict) -> list[str]:
    errors = []
    runs = output.get("runs", [])
    seen_ids = set()

    for i, run in enumerate(runs):
        rid = run.get("run_id", f"<index {i}>")
        if rid in seen_ids:
            errors.append(f"Duplicate run_id: {rid}")
        seen_ids.add(rid)

        for field in ["run_id", "env_id", "category_ids", "model_id", "framework", "backend", "scale", "source"]:
            if field not in run:
                errors.append(f"{rid}: missing required field '{field}'")

        fw = run.get("framework")
        bk = run.get("backend")
        if fw == "statgpu" and bk not in ("numpy", "cupy", "torch"):
            errors.append(f"{rid}: statgpu run has invalid backend '{bk}'")
        if fw != "statgpu" and bk is not None:
            errors.append(f"{rid}: external framework '{fw}' has non-null backend '{bk}'")

        metrics = run.get("metrics", {})
        if "timing" in metrics:
            t = metrics["timing"]
            ft = t.get("fit_time_ms", -1)
            if ft < 0:
                errors.append(f"{rid}: timing.fit_time_ms < 0 ({ft})")
            if "std_ms" in t and t["std_ms"] < 0:
                errors.append(f"{rid}: timing.std_ms < 0")
            if "min_ms" in t and "max_ms" in t and t["min_ms"] > t["max_ms"]:
                errors.append(f"{rid}: timing.min_ms > max_ms")

        if "speedup" in metrics:
            s = metrics["speedup"]
            if s.get("value", -1) < 0:
                errors.append(f"{rid}: speedup.value < 0")
            for sf in ["reference_backend", "reference_framework", "reported_semantics"]:
                if sf not in s:
                    errors.append(f"{rid}: speedup missing '{sf}'")

        def check_nan(obj, path=""):
            if isinstance(obj, float):
                import math
                if math.isnan(obj) or math.isinf(obj):
                    errors.append(f"{rid}: NaN/Inf at {path}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_nan(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for j, v in enumerate(obj):
                    check_nan(v, f"{path}[{j}]")

        check_nan(run, rid)

    return errors


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark frontend data")
    parser.add_argument("--out", help="Output path for benchmark_data.json")
    parser.add_argument("--report", help="Output path for parse_report.json")
    parser.add_argument("--check", action="store_true", help="Validate only, don't write output")
    parser.add_argument("--results-dir", default="results", help="Directory containing benchmark result files")
    parser.add_argument("--deterministic", action="store_true",
                        help="Freeze generated/git_sha for CI staleness checks")
    parser.add_argument("--update-preflight-baseline", action="store_true",
                        help="Write legacy identity audit fixture (manual-only; not in CI)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        results_dir = repo_root / args.results_dir

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning results from: {results_dir}")
    output, parse_report = generate(results_dir, deterministic=args.deterministic)

    errors = validate_output(output)
    schema_errors = validate_against_schema(output)
    all_errors = errors + schema_errors
    if all_errors:
        print(f"VALIDATION ERRORS ({len(all_errors)}):")
        for e in all_errors[:20]:
            print(f"  - {e}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
        if args.check:
            sys.exit(1)

    print(f"Runs generated: {len(output['runs'])}")
    print(f"Models: {len(output['models'])}")
    print(f"Validation errors: {len(errors)}")
    print(f"Parse warnings: {len(parse_report['warnings'])}")
    if parse_report["warnings"]:
        for w in parse_report["warnings"][:5]:
            print(f"  - {w['file']}: {w['reason']}")

    if args.check:
        if all_errors:
            sys.exit(1)
        print("OK — validation passed (structural + JSON Schema)")
        return

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {out_path}")

    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = Path.cwd() / report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(parse_report, f, indent=2)
        print(f"Wrote: {report_path}")

    if args.update_preflight_baseline:
        repo_root = Path(__file__).resolve().parents[3]
        fixture_dir = repo_root / "dev" / "tests" / "fixtures" / "benchmark_frontend"
        fixture_dir.mkdir(parents=True, exist_ok=True)

        audit = build_preflight_audit(results_dir, output, parse_report, PARSER_REGISTRY, get_git_sha)
        audit_path = fixture_dir / "legacy_identity_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        print(f"Wrote preflight audit: {audit_path}")

        det_output, det_report = generate(results_dir, deterministic=True)
        det_hashes = {
            "benchmark_data_sha256": hashlib.sha256(
                json.dumps(det_output, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "parse_report_sha256": hashlib.sha256(
                json.dumps(det_report, sort_keys=True, indent=2).encode("utf-8")
            ).hexdigest(),
        }
        det_path = fixture_dir / "legacy_deterministic_output_sha256.json"
        with open(det_path, "w", encoding="utf-8") as f:
            json.dump(det_hashes, f, indent=2)
        print(f"Wrote deterministic output hashes: {det_path}")
        return

    if not args.out and not args.report:
        print(json.dumps(parse_report, indent=2))


if __name__ == "__main__":
    main()
