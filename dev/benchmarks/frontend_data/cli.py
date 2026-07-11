"""CLI for benchmark frontend data generation."""

import argparse
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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


# ---------------------------------------------------------------------------
# Transitional v1.1 field injection (A1b)
# ---------------------------------------------------------------------------

# Legacy comparison groups from A0 audit
LEGACY_COMPARISON_GROUPS = {
    "penalized_glm_bench_perf_2026-06-22.json": "transitional:penalized-glm-performance",
    "glm_solver_benchmark_2026-06-23.json": "transitional:glm-solver",
    "benchmark_full/benchmark_statgpu_all.json": "transitional:elasticnet-cross-framework",
    "benchmark_full/benchmark_glmnet_all.json": "transitional:elasticnet-cross-framework",
}

TRANSITIONAL_FRAMEWORKS = {
    "statgpu":     {"framework_id": "statgpu",     "display_name": "statgpu",      "external": False, "backend_policy": "required"},
    "sklearn":     {"framework_id": "sklearn",     "display_name": "scikit-learn",  "external": True,  "backend_policy": "forbidden"},
    "glmnet":      {"framework_id": "glmnet",      "display_name": "glmnet",        "external": True,  "backend_policy": "forbidden"},
    "statsmodels": {"framework_id": "statsmodels", "display_name": "statsmodels",   "external": True,  "backend_policy": "forbidden"},
}


def _make_transitional_case_id(parser_name: str, raw_case_key: Optional[str]) -> str:
    if not raw_case_key:
        return "default"
    payload = json.dumps({"parser": parser_name, "raw_case_key": raw_case_key},
                         sort_keys=True, separators=(",", ":"))
    return "legacy-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _inject_transitional_fields(runs: list[dict], results_dir: Path) -> None:
    """Inject case_id, method_config_id, comparison_id, and source fields."""
    repo_root = Path(__file__).resolve().parents[3]

    for run in runs:
        src_file = run["source"]["file"]
        parser_name = run["source"]["parser"]

        # comparison_id
        comparison_id = LEGACY_COMPARISON_GROUPS.get(src_file, f"transitional:{src_file.replace('/', '-')}")
        run["comparison_id"] = comparison_id

        # case_id — transitional hash of parser-specific discriminator
        raw_case_key = None
        if "Penalized" in run.get("model_id", ""):
            raw_case_key = f"{run['scale']['scale_key']}:{run.get('loss','')}_{run.get('penalty','')}"
        elif run.get("model_id") == "Lasso":
            raw_case_key = run.get("benchmark_session_id", "") or run["scale"]["scale_key"]
        run["case_id"] = _make_transitional_case_id(parser_name, raw_case_key)

        # method_config_id
        run["method_config_id"] = "default"

        # source fields
        source_path = results_dir / src_file
        repo_rel = source_path.resolve().relative_to(repo_root.resolve()).as_posix() if source_path.exists() else src_file
        run["source"]["source_id"] = f"transitional:{repo_rel}"
        run["source"]["original_path"] = repo_rel
        if source_path.exists():
            from .canonical import source_sha256
            run["source"]["sha256"] = source_sha256(source_path)


def _build_transitional_frameworks(runs: list[dict]) -> list[dict]:
    """Build frameworks[] from transitional registry, only entries referenced by runs."""
    used = sorted(set(r["framework"] for r in runs))
    result = []
    for fw_id in used:
        if fw_id in TRANSITIONAL_FRAMEWORKS:
            result.append(dict(TRANSITIONAL_FRAMEWORKS[fw_id]))
        else:
            result.append({
                "framework_id": fw_id, "display_name": fw_id,
                "external": fw_id != "statgpu", "backend_policy": "forbidden" if fw_id != "statgpu" else "required"
            })
    return result


def _build_transitional_comparisons(runs: list[dict], results_dir: Path) -> list[dict]:
    """Build comparisons[] by deduplicating comparison_ids from runs."""
    seen: dict[str, dict] = {}
    for run in runs:
        cid = run["comparison_id"]
        if cid not in seen:
            label = cid.replace("transitional:", "").replace("-", " ").title()
            seen[cid] = {
                "comparison_id": cid,
                "label": label,
                "env_id": run["env_id"],
            }
    return sorted(seen.values(), key=lambda c: c["comparison_id"])


def _inject_canonical_fields(runs: list[dict], manifest: dict, results_dir: Path) -> None:
    """Inject canonical source_id, case_id, method_config_id, comparison_id (A2+)."""
    from .identity import run_identity

    # Build source lookup by canonical filename AND original path ending
    source_by_canon: dict[str, dict] = {}
    source_by_orig: dict[str, dict] = {}
    for src in manifest.get("sources", []):
        canon_name = Path(src["path"]).name
        source_by_canon[canon_name] = src
        orig = src.get("original_path", "")
        if orig:
            orig_name = Path(orig).name
            source_by_orig[orig_name] = src

    for run in runs:
        src_file = run["source"]["file"]
        # Match by canonical name first, then original name
        manifest_src = source_by_canon.get(src_file) or source_by_orig.get(src_file)
        if not manifest_src:
            # Try original_path ending match
            for orig_name, src in source_by_orig.items():
                if orig_name == src_file:
                    manifest_src = src
                    break

        if manifest_src:
            run["source"]["source_id"] = manifest_src["source_id"]
            run["source"]["original_path"] = manifest_src.get("original_path", "")
            run["comparison_id"] = manifest_src.get("comparison_id", manifest_src["source_id"])
            if manifest_src.get("source_date"):
                run["source"]["date"] = manifest_src["source_date"]
        else:
            # Fallback: transitional source_id
            run["source"]["source_id"] = f"transitional:{src_file}"
            run["comparison_id"] = f"transitional:{src_file.replace('/', '-')}"

        # Set case_id/method_config_id from parser output or fall back to default
        # Parsers should set these fields explicitly; default is a valid choice
        # for benchmarks with no parameter variation
        if run.get("case_id") is None:
            run["case_id"] = "default"
        if run.get("method_config_id") is None:
            run["method_config_id"] = "default"

        # Generate canonical run_id from RunIdentity
        identity = run_identity(run)
        canonical = json.dumps(identity, separators=(",", ":"), ensure_ascii=False)
        run["run_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _build_manifest_frameworks(manifest: dict, runs: list[dict]) -> list[dict]:
    """Build frameworks[] from manifest, only entries referenced by runs."""
    used = sorted(set(r["framework"] for r in runs))
    mf = manifest.get("frameworks", {})
    result = []
    for fw_id in used:
        if fw_id in mf:
            result.append({"framework_id": fw_id, **mf[fw_id]})
        else:
            result.append({"framework_id": fw_id, "display_name": fw_id,
                           "external": fw_id != "statgpu",
                           "backend_policy": "forbidden" if fw_id != "statgpu" else "required"})
    return result


def _build_manifest_comparisons(manifest: dict) -> list[dict]:
    """Build comparisons[] from manifest."""
    mc = manifest.get("comparisons", {})
    return sorted(
        [{"comparison_id": cid, **cdata} for cid, cdata in mc.items()],
        key=lambda c: c["comparison_id"]
    )


def generate(results_dir: Path, deterministic: bool = False,
            manifest: Optional[dict] = None, strict_sources: bool = False) -> tuple[dict, dict, dict]:
    """Generate unified benchmark_data.json from results_dir.

    If manifest is provided, use canonical mode (A2+).
    Otherwise use transitional mode (A1b).
    """
    from .registry import build_registry_from_manifest
    from .identity import run_identity

    all_runs: list[dict] = []
    all_models: dict[str, dict] = {}
    all_issues: list[dict] = []
    files_seen = 0
    files_parsed = 0

    # Build registry
    if manifest is not None:
        registry = build_registry_from_manifest(manifest)
        environments = [
            {"env_id": eid, **edata}
            for eid, edata in manifest.get("environments", {}).items()
        ]
        if not environments:
            environments = [{"env_id": "remote-p100", "label": "Tesla P100 + Xeon 8163",
                             "gpu": "NVIDIA Tesla P100-16GB", "cpu": "Intel Xeon Platinum 8163",
                             "host": "hz-4.matpool.com"}]
    else:
        registry = dict(PARSER_REGISTRY)
        environments = [{"env_id": "remote-p100", "label": "Tesla P100 + Xeon 8163",
                         "gpu": "NVIDIA Tesla P100-16GB", "cpu": "Intel Xeon Platinum 8163",
                         "host": "hz-4.matpool.com"}]

    for src_path, config in registry.items():
        filepath = Path(src_path)
        if not filepath.is_absolute():
            if manifest is not None:
                # Manifest paths are repo-relative
                repo_root = Path(__file__).resolve().parents[3]
                filepath = repo_root / src_path
            else:
                # Hardcoded registry paths are relative to results_dir
                filepath = results_dir / src_path
        path_str = src_path
        files_seen += 1
        if not filepath.exists():
            issue = {"file": path_str, "reason": "file not found",
                     "code": "SOURCE_MISSING", "severity": "error" if config.get("required", True) else "info"}
            all_issues.append(issue)
            if strict_sources and config.get("required", True):
                raise FileNotFoundError(f"Required source missing: {path_str}")
            continue

        # SHA256 check (manifest mode)
        if manifest is not None and strict_sources and config.get("required", True) and not config.get("sha256"):
            raise ValueError(f"Required source missing SHA256: {path_str}")
        if manifest is not None and config.get("sha256"):
            from .canonical import source_sha256
            actual = source_sha256(filepath)
            if actual != config["sha256"]:
                issue = {"file": path_str, "reason": f"SHA256 mismatch: expected {config['sha256'][:12]}..., got {actual[:12]}...",
                         "code": "SHA256_MISMATCH", "severity": "error"}
                all_issues.append(issue)
                if strict_sources:
                    raise ValueError(f"SHA256 mismatch for {path_str}")
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
                code = "PARSE_WARNING"
                severity = "warning"
                if "unavailable" in w.lower() or "not available" in w.lower():
                    code = "METHOD_UNAVAILABLE"
                    severity = "info"
                all_issues.append({"file": path_str, "reason": w, "code": code, "severity": severity})
            files_parsed += 1
        except Exception as e:
            issue = {"file": path_str, "reason": f"parse error: {e}",
                     "code": "PARSE_ERROR", "severity": "error"}
            all_issues.append(issue)
            if strict_sources:
                raise RuntimeError(f"Strict mode: parser failed for {path_str}: {e}") from e

    # Inject fields
    if manifest is not None:
        _inject_canonical_fields(all_runs, manifest, results_dir)
        t_frameworks = _build_manifest_frameworks(manifest, all_runs)
        t_comparisons = _build_manifest_comparisons(manifest)
    else:
        _inject_transitional_fields(all_runs, results_dir)
        t_frameworks = _build_transitional_frameworks(all_runs)
        t_comparisons = _build_transitional_comparisons(all_runs, results_dir)

    if deterministic:
        generated_ts = "1970-01-01T00:00:00+00:00"
        git_sha = "deterministic"
    else:
        generated_ts = datetime.now(timezone.utc).isoformat()
        git_sha = get_git_sha()

    # Build output
    output = {
        "schema_version": "1.1.0",
        "generated": generated_ts,
        "meta": {
            "generator": "dev/benchmarks/generate_benchmark_data.py",
            "git_sha": git_sha,
            "generation_id": "",
        },
        "environments": environments,
        "categories": CATEGORIES,
        "models": sorted(all_models.values(), key=lambda x: x["model_id"]),
        "frameworks": t_frameworks,
        "comparisons": t_comparisons,
        "runs": all_runs,
    }

    # Compute generation_id (bundle hash, without generation_id in payload)
    # Build per-source context for enriching issues
    source_context: dict[str, dict] = {}
    if manifest:
        for src in manifest.get("sources", []):
            source_context[src["path"]] = {
                "source_id": src["source_id"],
                "parser": src["parser"],
                "parser_version": src.get("parser_version", ""),
                "comparison_id": src.get("comparison_id", src["source_id"]),
            }
    raw_issues = []
    for i in all_issues:
        ctx = source_context.get(i.get("file", ""), {})
        raw_issues.append({
            "source_id": i.get("source_id") or ctx.get("source_id", ""),
            "file": i.get("file", ""),
            "parser": i.get("parser") or ctx.get("parser", ""),
            "code": i.get("code", "UNKNOWN"),
            "severity": i.get("severity", "warning"),
            "message": i.get("reason", i.get("message", "")),
            "comparison_id": ctx.get("comparison_id", ""),
            "parser_version": ctx.get("parser_version", ""),
        })
    parse_report = {
        "report_version": "2.0",
        "files_seen": files_seen,
        "files_parsed": files_parsed,
        "files_skipped": files_seen - files_parsed,
        "runs_generated": len(all_runs),
        "generation_id": "",
        "issues": raw_issues,
    }

    # Build inventory
    repo_root = Path(__file__).resolve().parents[3]
    if manifest:
        registered = len(manifest.get("sources", []))
        available = sum(1 for s in manifest["sources"] if (repo_root / s["path"]).exists())
    else:
        registered = len(PARSER_REGISTRY)
        available = sum(1 for p in PARSER_REGISTRY if (results_dir / p).exists())
    parsed = len({r["source"]["source_id"] for r in all_runs})
    catalog_total = manifest.get("catalog_total") if manifest else None
    if catalog_total is None:
        catalog_total = sum(
            1
            for path in results_dir.rglob("*.json")
            if "benchmark_frontend_sources" not in path.relative_to(results_dir).parts
        )
    if not isinstance(catalog_total, int) or catalog_total < registered:
        raise ValueError("catalog_total must be an integer >= registered_sources")
    inventory = {
        "inventory_version": "1.0",
        "catalog_version": "1.0",
        "generation_id": "",
        "catalog_total": catalog_total,
        "eligible_total": registered,
        "registered_sources": registered,
        "available_sources": available,
        "parsed_sources": parsed,
    }

    # Strict mode: check allowed_issue_codes for warnings/errors
    if manifest and strict_sources:
        allowed_by_path: dict[str, set] = {}
        for src in manifest.get("sources", []):
            allowed_by_path[src["path"]] = set(src.get("allowed_issue_codes", []))
        for issue in all_issues:
            if issue.get("severity") in ("warning", "error"):
                path = issue.get("file", "")
                allowed = allowed_by_path.get(path, set())
                code = issue.get("code", "")
                if code not in allowed:
                    raise ValueError(
                        f"Strict mode: unallowed issue [{issue['severity']}] {code} "
                        f"in {path}: {issue.get('reason', '')}"
                    )

    # Two-phase speedup reference resolution (canonical mode — before generation_id)
    if manifest:
        _resolve_speedup_references(all_runs)

    # generation_id from full 3-file bundle
    from copy import deepcopy
    def _strip_gid(obj):
        result = deepcopy(obj)
        if "meta" in result:
            result["meta"].pop("generation_id", None)
        else:
            result.pop("generation_id", None)
        return result

    bundle = {
        "benchmark_data": _strip_gid(output),
        "parse_report": _strip_gid(parse_report),
        "source_inventory": _strip_gid(inventory),
    }
    gid = hashlib.sha256(
        json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    output["meta"]["generation_id"] = gid
    parse_report["generation_id"] = gid
    inventory["generation_id"] = gid

    return output, parse_report, inventory


def _resolve_speedup_references(runs: list[dict]) -> None:
    """Resolve computed speedups to an unambiguous timing run in the same chart cell."""
    from .identity import chart_cell_identity, chart_series_identity, identity_json

    timing_runs_by_cell: dict[str, list[dict]] = {}
    for r in runs:
        timing = r.get("metrics", {}).get("timing")
        if not timing or timing.get("fit_time_ms", 0) <= 0:
            continue
        key = identity_json(
            chart_cell_identity(r, include_session=False) + chart_series_identity(r)
        )
        timing_runs_by_cell.setdefault(key, []).append(r)

    for run in runs:
        sp = run.get("metrics", {}).get("speedup", {})
        if not sp or sp.get("reported_semantics") != "computed":
            continue
        if sp.get("reference_run_id"):
            continue  # already resolved

        group_identity = chart_cell_identity(run, include_session=False)
        reference_framework = sp.get("reference_framework")
        reference_backend = sp.get("reference_backend")
        implementations = [run.get("implementation")]
        if implementations[0] is not None:
            implementations.append(None)

        for implementation in implementations:
            ref_key = identity_json(
                group_identity + [reference_framework, reference_backend, implementation]
            )
            candidates = timing_runs_by_cell.get(ref_key, [])
            if len(candidates) == 1:
                sp["reference_run_id"] = candidates[0]["run_id"]
                break


# Validation functions (migrated from old module)
def _load_schema() -> Optional[dict]:
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


def validate_semantic(output: dict, manifest: Optional[dict] = None, strict_sources: bool = False) -> list[str]:
    """Semantic/referential integrity checks for canonical mode."""
    errors = []
    runs = output.get("runs", [])
    envs = {e["env_id"] for e in output.get("environments", [])}
    models = {m["model_id"] for m in output.get("models", [])}
    frameworks = {f["framework_id"] for f in output.get("frameworks", [])}
    comparisons = {c["comparison_id"] for c in output.get("comparisons", [])}
    fw_policy = {f["framework_id"]: f.get("backend_policy", "forbidden") for f in output.get("frameworks", [])}
    runs_by_id = {run.get("run_id"): run for run in runs}

    for run in runs:
        rid = run.get("run_id", "?")
        fw = run.get("framework", "")
        bk = run.get("backend")

        # Referential integrity
        if run.get("env_id") not in envs:
            errors.append(f"{rid}: env_id '{run.get('env_id')}' not in environments")
        if run.get("model_id") not in models:
            errors.append(f"{rid}: model_id '{run.get('model_id')}' not in models")
        if fw not in frameworks:
            errors.append(f"{rid}: framework '{fw}' not in frameworks")
        if run.get("comparison_id") not in comparisons:
            errors.append(f"{rid}: comparison_id '{run.get('comparison_id')}' not in comparisons")

        # Backend policy
        policy = fw_policy.get(fw)
        if policy == "required" and bk is None:
            errors.append(f"{rid}: framework '{fw}' requires backend, got null")
        if policy == "forbidden" and bk is not None:
            errors.append(f"{rid}: framework '{fw}' forbids backend, got '{bk}'")

        # Category validity
        cats = run.get("category_ids", [])
        if not cats:
            errors.append(f"{rid}: empty category_ids")
        model_entry = next((m for m in output.get("models", []) if m["model_id"] == run.get("model_id")), None)
        if model_entry:
            for cid in cats:
                if cid not in model_entry.get("category_ids", []):
                    errors.append(f"{rid}: category '{cid}' not in model categories")

        # Speedup reference
        sp = run.get("metrics", {}).get("speedup", {})
        ref_fw = sp.get("reference_framework")
        if ref_fw and ref_fw not in frameworks:
            errors.append(f"{rid}: speedup reference_framework '{ref_fw}' not in frameworks")
        if sp.get("reported_semantics") == "computed":
            ref_id = sp.get("reference_run_id")
            if not ref_id:
                errors.append(f"{rid}: computed speedup missing reference_run_id")
            elif ref_id not in runs_by_id:
                errors.append(f"{rid}: speedup reference_run_id '{ref_id}' not found")
            else:
                ref_run = runs_by_id[ref_id]
                if ref_run.get("framework") != ref_fw:
                    errors.append(f"{rid}: speedup reference framework does not match referenced run")
                if ref_run.get("backend") != sp.get("reference_backend"):
                    errors.append(f"{rid}: speedup reference backend does not match referenced run")
                ref_time = ref_run.get("metrics", {}).get("timing", {}).get("fit_time_ms")
                run_time = run.get("metrics", {}).get("timing", {}).get("fit_time_ms")
                if not ref_time or not run_time:
                    errors.append(f"{rid}: computed speedup requires positive timing on both runs")
                else:
                    expected = ref_time / run_time
                    if abs(sp.get("value", 0) - expected) > max(1e-4, abs(expected) * 1e-4):
                        errors.append(f"{rid}: computed speedup value does not match referenced timing")

        # Canonical IDs
        if run.get("source", {}).get("source_id", "").startswith("transitional:"):
            if manifest:
                errors.append(f"{rid}: transitional source_id in canonical mode")
        if run.get("case_id", "").startswith("legacy-"):
            if manifest:
                errors.append(f"{rid}: legacy case_id in canonical mode")

    # Build run_id lookup for speedup reference validation
    run_by_id: dict[str, dict] = {r["run_id"]: r for r in runs}

    # RunIdentity uniqueness + hash verification (canonical mode)
    if manifest:
        from .identity import run_identity, identity_json as _id_json
        seen_identities: dict[str, str] = {}
        for run in runs:
            identity = run_identity(run)
            rid_key = _id_json(identity)
            expected_rid = hashlib.sha256(rid_key.encode("utf-8")).hexdigest()[:16]
            if run["run_id"] != expected_rid:
                errors.append(f"{run['run_id']}: run_id does not match RunIdentity hash (expected {expected_rid})")
            if rid_key in seen_identities:
                errors.append(f"{run['run_id']}: duplicate RunIdentity with {seen_identities[rid_key]}")
            seen_identities[rid_key] = run["run_id"]

    # Speedup reference validation
    for run in runs:
        sp = run.get("metrics", {}).get("speedup", {})
        if not sp:
            continue
        ref_id = sp.get("reference_run_id")
        if sp.get("reported_semantics") == "computed":
            if not ref_id:
                errors.append(f"{run['run_id']}: computed speedup missing reference_run_id")
            elif ref_id == run["run_id"]:
                errors.append(f"{run['run_id']}: speedup self-reference")
            elif ref_id not in run_by_id:
                errors.append(f"{run['run_id']}: speedup reference_run_id '{ref_id}' not found")
            else:
                ref_run = run_by_id[ref_id]
                ref_time = ref_run.get("metrics", {}).get("timing", {}).get("fit_time_ms", 0)
                if ref_time <= 0:
                    errors.append(f"{run['run_id']}: speedup reference run has no positive timing")
                # Cross-check comparison/case compatibility
                if ref_run.get("comparison_id") != run.get("comparison_id"):
                    errors.append(f"{run['run_id']}: speedup reference has different comparison_id")
                if ref_run.get("scale", {}).get("scale_key") != run.get("scale", {}).get("scale_key"):
                    errors.append(f"{run['run_id']}: speedup reference has different scale")

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
            if s.get("value", 0) <= 0:
                errors.append(f"{rid}: speedup.value must be > 0")
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


def _write_transactional(out_path_str, report_path_str, inv_path_str, output, parse_report, inventory):
    """Write three output files atomically: temp → os.replace in order."""
    import os as _os
    import tempfile

    out_p, rpt_p, inv_p = (Path(path) for path in (out_path_str, report_path_str, inv_path_str))
    out_p, rpt_p, inv_p = (
        path if path.is_absolute() else Path.cwd() / path
        for path in (out_p, rpt_p, inv_p)
    )

    for p in (out_p, rpt_p, inv_p):
        p.parent.mkdir(parents=True, exist_ok=True)

    def write_temp(target: Path, data: bytes) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(data)
            handle.flush()
            _os.fsync(handle.fileno())
            return Path(handle.name)

    temp_paths: list[Path] = []
    try:
        tmp_r = write_temp(rpt_p, json.dumps(parse_report, indent=2).encode("utf-8"))
        temp_paths.append(tmp_r)
        tmp_i = write_temp(inv_p, json.dumps(inventory, indent=2).encode("utf-8"))
        temp_paths.append(tmp_i)
        tmp_d = write_temp(out_p, json.dumps(output, indent=2, ensure_ascii=False).encode("utf-8"))
        temp_paths.append(tmp_d)

        _os.replace(tmp_r, rpt_p)
        _os.replace(tmp_i, inv_p)
        _os.replace(tmp_d, out_p)
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
    print(f"Wrote: {out_p}")
    print(f"Wrote: {rpt_p}")
    print(f"Wrote: {inv_p}")


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
    parser.add_argument("--inventory-out", help="Output path for source_inventory.json (A1b+)")
    parser.add_argument("--strict-sources", action="store_true",
                        help="Fail on missing required sources (A2+)")
    args = parser.parse_args()

    # All-or-none for output args (A1b+)
    output_args = [args.out, args.report, args.inventory_out]
    if any(output_args) and not all(output_args):
        parser.error("--out, --report, and --inventory-out must be provided together")
    if args.check and any(output_args):
        parser.error("--check cannot be combined with output paths")
    if args.update_preflight_baseline and (args.check or any(output_args) or args.strict_sources):
        parser.error("--update-preflight-baseline must run alone")

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        results_dir = repo_root / args.results_dir

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load manifest if available (A2+)
    from .registry import load_manifest
    repo_root = Path(__file__).resolve().parents[3]
    manifest = load_manifest(repo_root)
    if manifest:
        print("Using manifest-driven canonical mode")

    print(f"Scanning results from: {results_dir}")
    output, parse_report, source_inventory = generate(results_dir, deterministic=args.deterministic,
                                                       manifest=manifest, strict_sources=args.strict_sources)

    errors = validate_output(output)
    schema_errors = validate_against_schema(output)
    semantic_errors = validate_semantic(output, manifest, args.strict_sources)
    all_errors = errors + schema_errors + semantic_errors
    if all_errors:
        print(f"VALIDATION ERRORS ({len(all_errors)}):")
        for e in all_errors[:20]:
            print(f"  - {e}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
    # Always exit on validation errors — never write invalid output
    if all_errors:
        print(f"ERROR: {len(all_errors)} validation errors, refusing to write", file=sys.stderr)
        sys.exit(1)

    print(f"Runs generated: {len(output['runs'])}")
    print(f"Models: {len(output['models'])}")
    print(f"Validation errors: {len(errors)}")
    issues = parse_report.get("issues", parse_report.get("warnings", []))
    errors_count = sum(1 for i in issues if i.get("severity") == "error")
    warnings_count = sum(1 for i in issues if i.get("severity") == "warning")
    print(f"Issues: {errors_count} errors, {warnings_count} warnings")
    if issues:
        for i in issues[:5]:
            print(f"  - [{i.get('severity','?')}] {i.get('file','')}: {i.get('reason', i.get('message',''))}")

    if args.check:
        print("OK — validation passed (structural + JSON Schema)")
        return

    # Transactional write using inventory from generate() (participated in generation_id hash)
    if args.out:
        _write_transactional(args.out, args.report, args.inventory_out, output, parse_report, source_inventory)

    if args.update_preflight_baseline:
        repo_root = Path(__file__).resolve().parents[3]
        fixture_dir = repo_root / "dev" / "tests" / "fixtures" / "benchmark_frontend"
        fixture_dir.mkdir(parents=True, exist_ok=True)

        audit = build_preflight_audit(results_dir, output, parse_report, PARSER_REGISTRY, get_git_sha)
        audit_path = fixture_dir / "legacy_identity_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        print(f"Wrote preflight audit: {audit_path}")

        det_output, det_report, det_inventory = generate(results_dir, deterministic=True)
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
