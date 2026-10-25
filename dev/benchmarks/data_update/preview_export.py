#!/usr/bin/env python3
"""Read-only evidence export. No benchmark, GPU, publishing, or promotion occurs.

The output is a diagnostic bridge, NOT the dashboard's canonical v1.1 bundle.
Run the separate registered parser to normalize supported records for display.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

SCHEMA = "statgpu-benchmark-update-preview-v1"
MAX_JSON_BYTES = 64 * 1024**2
MAX_CAMPAIGN_INPUT_BYTES = 128 * 1024**2
NATIVE_SCHEMAS = {"statgpu-qualification-campaign-v1": "registered",
                  "statgpu-independent-execution-v1": "independent"}
PASS = {"pass", "passed", "complete", "success"}
CONTEXT_KEYS = {
    "case", "case_id", "operation", "model", "model_id", "backend", "requested_backend",
    "framework", "n", "p", "n_samples", "n_features", "n_times", "shape", "problem",
    "parameters", "scenario", "seed", "dtype", "input_dtype", "weights", "kernel",
    "bandwidth", "dimension", "queries", "resamples", "status", "solver", "penalty",
    "scope", "warmup", "repeats", "statistic", "synchronized_device", "transfer_policy",
    "instrumented", "compute_placement", "implementation", "mode", "cleanup",
    "performance_eligible", "canonical_dashboard_ready",
    "benchmark", "data_identity", "data_id", "selector_policy",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular JSON evidence file")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON evidence exceeds the 64 MiB input budget")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def constant(token):
        raise ValueError("Nonfinite JSON constant: " + token)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=constant)
    def finite(node):
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError("Nonfinite JSON number")
        if isinstance(node, dict):
            for item in node.values(): finite(item)
        elif isinstance(node, list):
            for item in node: finite(item)
    finite(value)
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pointer(parent, key):
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected nonempty relative evidence path")
    target = root / relative
    if ".." in Path(relative).parts or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes its root")
    cursor = target
    while cursor != root:
        if cursor.is_symlink(): raise ValueError("Symlink evidence is not accepted")
        cursor = cursor.parent
    return target


def verify_inventory(root, required=("plan.json", "summary.json", "source-before.json", "source-after.json")):
    """Checksum integrity is tamper detection, not an authenticated signature."""
    errors, hashes = [], {}
    try:
        hashes = read_json(safe_path(root, "SHA256SUMS.json"))
        if not isinstance(hashes, dict) or not hashes: raise ValueError("Empty checksum inventory")
        for name, wanted in hashes.items():
            path = safe_path(root, name)
            if not isinstance(wanted, str) or not re.fullmatch("[0-9a-f]{64}", wanted):
                raise ValueError("Invalid checksum for " + name)
            if not path.is_file() or digest(path) != wanted:
                errors.append("Missing or checksum-mismatched evidence: " + name)
        for name in required:
            if name not in hashes: errors.append("Unsealed required evidence: " + name)
    except (OSError, ValueError, TypeError) as exc:
        errors.append("Checksum inventory unavailable: " + str(exc))
    return hashes, errors


class Validators:
    """Only repository pure read/assessment helpers are called; never execute()."""
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        sys.path.insert(0, str(self.repo))
        self.registered = importlib.import_module("dev.benchmarks.run_qualification_campaign")
        self.independent = importlib.import_module("dev.benchmarks.run_independent_benchmarks")
        for module in (self.registered, self.independent):
            if not Path(module.__file__).resolve().is_relative_to(self.repo):
                raise ValueError("Qualification helper imported from a different checkout")
        self.source = {
            "repo": str(self.repo),
            "helper_sha256": {str(Path(m.__file__).relative_to(self.repo)): digest(m.__file__)
                               for m in (self.registered, self.independent)},
        }
        try:
            self.source["git_commit"] = subprocess.check_output(
                ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            self.source["git_commit"] = None

    def assess(self, kind, job, folder, process, expected_sha):
        if kind == "registered":
            return self.registered.assess(folder / "raw.json", job, process.get("exit_code"), expected_sha)
        return self.independent.assess_artifact(job, folder, process)


def discover_roots(roots):
    result = {}
    def inspect(root, allow_children):
        summary_path = root / "summary.json"
        if not summary_path.exists():
            if allow_children:
                for child in sorted(root.iterdir()):
                    if child.is_dir() and not child.is_symlink(): inspect(child, False)
            return
        document = read_json(summary_path)
        if not isinstance(document, dict): raise ValueError("Summary must be an object")
        schema = document.get("schema_version")
        if schema in NATIVE_SCHEMAS:
            result[root] = NATIVE_SCHEMAS[schema]
        elif schema == "statgpu-batch-attempt-v1":
            if document.get("native_directory") != "native": raise ValueError("Unrecognized batch native directory")
            native = safe_path(root, "native")
            if native.is_dir(): inspect(native, False)
        elif schema == "statgpu-full-benchmark-result-v1":
            for lane in document.get("lanes", {}).values():
                inspect(safe_path(root, lane["evidence_directory"]), False)
        elif allow_children:
            raise ValueError("Unsupported root summary schema: " + str(schema))
    for value in roots:
        root = Path(value).expanduser().resolve()
        if not root.is_dir(): raise ValueError("Evidence root is not a directory: " + str(root))
        inspect(root, True)
    return sorted(result.items(), key=lambda item: str(item[0]))


def batch_context(native):
    parent = native.parent
    path = parent / "summary.json"
    if native.name != "native" or not path.is_file(): return None, []
    summary = read_json(path)
    if summary.get("schema_version") != "statgpu-batch-attempt-v1": return None, []
    hashes, errors = verify_inventory(parent, required=("summary.json", "started.json"))
    for name, field in (("summary.json", "native_summary_sha256"), ("plan.json", "batch_plan_sha256")):
        target = safe_path(native, name)
        if not target.is_file() or digest(target) != summary.get(field):
            errors.append("Batch wrapper does not bind native " + name)
    archive = parent / "native.tar.gz"
    if summary.get("native_archive_sha256") and (not archive.is_file() or digest(archive) != summary["native_archive_sha256"]):
        errors.append("Batch native archive checksum mismatch")
    before, after = summary.get("source_before"), summary.get("source_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        errors.append("Missing batch source before/after objects")
    else:
        errors.extend(source_state(before, after)[2])
        if before != after: errors.append("Batch source identity changed")
        for name in ("source-before.json", "source-after.json"):
            native_source = read_json(native / name)
            if not isinstance(native_source, dict):
                errors.append("Native source snapshot is not an object")
                continue
            if native_source.get("git_commit") != before.get("git_commit"):
                errors.append("Native source commit differs from batch source")
            hashes_seen = native_source.get("files_sha256", {})
            if not hashes_seen or any(before.get("files_sha256", {}).get(k) != v for k, v in hashes_seen.items()):
                errors.append("Native source bytes are not bound to batch source snapshot")
    context = {"batch_id": summary.get("batch_id"), "started_at": summary.get("started_at"),
               "environment_id": summary.get("environment_id"),
               "finished_at": summary.get("finished_at"), "source_commit": before.get("git_commit") if isinstance(before, dict) else None,
               "wrapper_path": str(parent), "wrapper_summary_sha256": digest(path), "operator_attestation": None}
    attest_path = parent / "operator_attestation.json"
    if attest_path.exists():
        if "operator_attestation.json" not in hashes: errors.append("Unsealed batch attestation")
        attestation = read_json(attest_path)
        if not isinstance(attestation, dict) or attestation.get("machine_verified") is not False:
            errors.append("Invalid batch operator attestation")
        else:
            report = attestation.get("report")
            if not isinstance(report, dict): errors.append("Missing operator report")
            else:
                try:
                    validate_attestation(report)
                    if report.get("environment_id") is not None and report["environment_id"] != summary.get("environment_id"):
                        raise ValueError("Attested environment differs from this batch")
                    start = datetime.fromisoformat(str(summary.get("started_at")).replace("Z", "+00:00"))
                    finish = datetime.fromisoformat(str(summary.get("finished_at")).replace("Z", "+00:00"))
                    recorded = datetime.fromisoformat(report["recorded_at"].replace("Z", "+00:00"))
                    if any(t.tzinfo is None for t in (start, finish, recorded)) or not recorded <= start <= finish:
                        raise ValueError("Attestation must precede this completed batch, with timezone-aware timestamps")
                    for field, event in (("started_at", start), ("ended_at", finish)):
                        if field in report:
                            boundary = datetime.fromisoformat(report[field].replace("Z", "+00:00"))
                            if boundary.tzinfo is None or (field == "started_at" and boundary > event) or (field == "ended_at" and boundary < event):
                                raise ValueError("Attestation window does not cover this batch")
                    context["operator_attestation"] = {"kind": "operator_attested", "machine_verified": False,
                        "document": report, "sha256": digest(attest_path), "batch_id": summary.get("batch_id"),
                        "source_commit": context["source_commit"], "applies_to": "this sealed new batch only, never historical evidence",
                        "metadata_binding_verified": not errors, "changes_performance_eligibility": False}
                except (ValueError, TypeError) as exc:
                    errors.append("Operator attestation scope unresolved: " + str(exc))
    return context, errors


def source_state(before, after):
    errors = []
    sha = before.get("git_commit")
    if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{40}", sha):
        errors.append("Missing exact measured source SHA")
    keys = ("git_commit", "working_tree", "content_sha256", "source_content_sha256", "worktree_fingerprint")
    if any(before.get(k) != after.get(k) for k in keys): errors.append("Source changed during campaign")
    if before.get("working_tree") != "clean": errors.append("Measured working tree was not clean")
    fingerprint = next((before.get(k) for k in keys[2:] if isinstance(before.get(k), str)
                        and re.fullmatch("[0-9a-f]{64}", before[k])), None)
    if not fingerprint: errors.append("Missing measured source content fingerprint")
    return sha, fingerprint, errors


def thread_profile(runtime, job, kind):
    requested = job.get("cpu_threads", 1 if kind == "independent" else None)
    snapshots = {key: runtime.get(key, {}) for key in ("before", "after")}
    values, torch_values = [], []
    valid = type(requested) is int and requested > 0
    bound = requested if valid else 0
    for snapshot in snapshots.values():
        pools = snapshot.get("pools") if isinstance(snapshot, dict) else None
        if not isinstance(pools, list) or not pools or snapshot.get("cap_verified") is not True:
            valid = False
            continue
        for pool in pools:
            count = pool.get("num_threads") if isinstance(pool, dict) else None
            values.append(count)
            if type(count) is not int or not 1 <= count <= bound: valid = False
        torch = snapshot.get("torch_threads")
        if job.get("backend") == "torch" and (not isinstance(torch, dict) or not {"intraop", "interop"} <= set(torch)):
            valid = False
        if torch is not None:
            if not isinstance(torch, dict): valid = False
            else:
                for count in torch.values():
                    torch_values.append(count)
                    if type(count) is not int or not 1 <= count <= bound: valid = False
    return {"requested_cap": requested, "observed_pool_threads": values,
            "observed_torch_threads": torch_values, "cap_verified": bool(valid),
            "all_observed_equal_requested": bool(valid and values and all(x == requested for x in values + torch_values)),
            "claim": "Observed library pool bounds before/after workload, not process CPU utilization or exact active thread count"}


def timing_records(document):
    """Lossless timing-field index; no inferred fit scope or cross-record pooling.

    Scalar summaries, samples, and nonnumeric structured timing fields are kept
    separately. Unknown layouts stay pointer-addressable but are not chartable.
    """
    seconds_fields = {"raw_seconds", "samples_seconds", "seconds", "fit_seconds", "median_seconds", "mean_seconds",
                      "min_seconds", "max_seconds", "std_seconds", "stdev_seconds", "iqr_seconds",
                      "cold_seconds", "warm_seconds", "cold_fit_seconds", "warm_fit_seconds",
                      "with_digest_seconds", "without_digest_seconds", "digest_overhead_seconds",
                      "representative_seconds", "first_fit_extra_seconds_vs_eager"}
    ms_fields = {"times_ms", "fit_ms", "fit_ms_all", "fit_ms_mean", "fit_ms_std", "fit_ms_min", "fit_ms_max",
                 "time_ms", "time_ms_mean", "time_ms_std", "time_ms_min", "time_ms_max", "warm_mean_ms",
                 "warm_std_ms", "warm_min_ms", "warm_max_ms", "cold_ms", "predict_ms", "refit_ms",
                 "cv_evaluation_ms", "final_refit_ms", "total_fit_ms"}
    derived_fields = {"digest_overhead_seconds", "first_fit_extra_seconds_vs_eager", "representative_seconds"}
    source_schema = document.get("schema_version") if isinstance(document, dict) else None
    cv_results = document.get("results", []) if source_schema == "statgpu-scoped-cv-candidate-v2" else []
    if not isinstance(cv_results, list):
        cv_results = []
    cv_phases = {"cv_evaluation_ms", "final_refit_ms", "total_fit_ms"}

    def recorded_cv_scope(record, path):
        # This protocol belongs to one CV result, not to arbitrary descendants or
        # its clean_public_fit sibling. Keep the original field and its pointer.
        match = re.fullmatch(r"/results/(\d+)/case/runs/(\d+)/(?:timing|repeat_samples/\d+)/([^/]+)", path)
        if not match or record["timing_key"] not in cv_phases:
            return
        index = int(match.group(1))
        if index >= len(cv_results) or not isinstance(cv_results[index], dict):
            return
        result = cv_results[index]
        protocol = result.get("protocol", {})
        if not isinstance(protocol, dict):
            return
        context = record["context"]
        context["timing_kind"] = "cv_diagnostic_stage"
        context["timing_phase"] = record["timing_key"].removesuffix("_ms")
        if context.get("framework") == "statgpu":
            diagnostic = result.get("diagnostic_timing", {})
            if isinstance(diagnostic, dict) and type(diagnostic.get("instrumented")) is bool:
                context["instrumented"] = diagnostic["instrumented"]
        for name in ("warmup", "repeats", "transfer_policy"):
            if name not in context and name in protocol:
                context[name] = protocol[name]
        scope = protocol.get("timing_scope")
        if not record["timing_scope"] and isinstance(scope, str) and scope.strip():
            record["timing_scope"] = context["scope"] = scope
            context["scope_source_pointer"] = "/results/%d/protocol/timing_scope" % index

    root_timing = document.get("timing", {}) if isinstance(document, dict) else {}
    default_context = {k: v for k, v in root_timing.items() if k in CONTEXT_KEYS} if isinstance(root_timing, dict) else {}
    fit_scope = document.get("timing_scope", {}).get("fit") if isinstance(document.get("timing_scope"), dict) else None
    def walk(value, path="", context=None, row_path="", parent_key=""):
        context = dict(context or {})
        if isinstance(value, dict):
            for key in ("case_id", "model_id"):
                if key in value and key in context and value[key] != context[key]:
                    context["identity_conflicts"] = sorted(set(context.get("identity_conflicts", [])) | {key})
            context.update({key: child for key, child in value.items() if key in CONTEXT_KEYS})
            if any(key in value for key in ("case", "case_id", "model", "model_id", "operation", "problem")):
                row_path = path
            for key, child in value.items():
                lower = key.lower()
                path_to_child = pointer(path, key)
                unit = "milliseconds" if lower in ms_fields else "seconds" if lower in seconds_fields else None
                if parent_key == "raw_fit_seconds" and key in ("disable", "default"):
                    unit = "seconds"
                sample_name = lower in {"with_digest_samples", "without_digest_samples"}
                is_numeric = type(child) in (int, float)
                is_samples = isinstance(child, list) and bool(child) and all(type(x) in (int, float) for x in child)
                if unit or sample_name:
                    central = key not in derived_fields and not any(token in lower for token in ("std", "stdev", "iqr", "min", "max", "elapsed", "timeout", "budget"))
                    record_context = dict(context)
                    if (key == "fit_seconds" or parent_key == "raw_fit_seconds") and isinstance(fit_scope, str):
                        record_context["scope"] = fit_scope
                        record_context["scope_source_pointer"] = "/timing_scope/fit"
                    if parent_key == "raw_fit_seconds": record_context["mode"] = key
                    record = {"raw_pointer": path_to_child, "row_pointer": row_path,
                              "timing_key": key, "timing_scope": record_context.get("scope"),
                              "unit": unit, "raw_value": child, "context": record_context,
                              "timing_disposition": "recorded-timing-field-requires-scope-review"}
                    if source_schema == "statgpu-scoped-cv-candidate-v2":
                        record["source_schema"] = source_schema
                        recorded_cv_scope(record, path_to_child)
                        if re.fullmatch(r"/results/\d+/clean_public_fit/(?:raw_seconds|timing/median_seconds|repeat_samples/\d+/total_fit_ms)", path_to_child):
                            record_context["timing_kind"] = "clean_public_fit"
                            record_context["timing_phase"] = "total_fit"
                    if is_samples:
                        record["sample_count"] = len(child)
                        record["samples_positive_finite"] = all(math.isfinite(x) and x > 0 for x in child)
                        if record["samples_positive_finite"] and unit and central:
                            record["summary_seconds"] = statistics.median(child) / (1000 if unit == "milliseconds" else 1)
                            record["statistic"] = "median_of_recorded_samples"
                    elif is_numeric:
                        record["statistic"] = "recorded_scalar:" + key
                        # Std/min/max are retained but never treated as a central measurement.
                        if key in derived_fields: record["timing_disposition"] = "derived-or-representative-field-not-an-independent-duration"
                        if math.isfinite(child) and child > 0 and unit and central:
                            record["summary_seconds"] = child / (1000 if unit == "milliseconds" else 1)
                    else:
                        record["timing_disposition"] = "missing-empty-or-malformed-timing-field-retained"
                        if isinstance(child, list): record["sample_count"] = len(child)
                    yield record
                if isinstance(child, (dict, list)):
                    yield from walk(child, path_to_child, context, row_path, key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    yield from walk(child, pointer(path, index), context, pointer(path, index) if parent_key in ("rows", "results") else row_path)
    yield from walk(document, context=default_context)


def validate_attestation(doc):
    required = {"operator", "statement", "recorded_at", "scope"}
    allowed = required | {"environment_id", "started_at", "ended_at", "device_uuid"}
    if not isinstance(doc, dict) or not required <= set(doc) or set(doc) - allowed or any(type(v) is not str or not v.strip() or len(v) > 4000 for v in doc.values()):
        raise ValueError("Attestation needs operator, statement, recorded_at, scope; optional environment_id, started_at, ended_at, device_uuid")
    if any("REPLACE_" in value for value in doc.values()): raise ValueError("Attestation still contains placeholder text")
    dates = {}
    for name in ("recorded_at", "started_at", "ended_at"):
        if name in doc:
            dates[name] = datetime.fromisoformat(doc[name].replace("Z", "+00:00"))
            if dates[name].tzinfo is None or dates[name] > datetime.now(timezone.utc):
                raise ValueError("Attestation timestamps must be timezone-aware observed times, not future promises")
    if dates.get("started_at", dates["recorded_at"]) > dates["recorded_at"]:
        raise ValueError("Attestation started_at exceeds recorded_at")
    if "ended_at" in dates and dates.get("started_at", dates["ended_at"]) > dates["ended_at"]:
        raise ValueError("Attestation interval is reversed")


def load_attestation(path):
    if path is None: return None
    doc = read_json(path)
    validate_attestation(doc)
    return {"kind": "operator_attested", "document": doc, "sha256": digest(path),
            "machine_verified": False, "metadata_binding_verified": False, "applies_to": "unbound report only; not retroactively applied to any campaign",
            "changes_performance_eligibility": False}


def comparison_candidates(records):
    """Identify only exact recorded metadata matches, never compute a speedup."""
    groups, withheld = {}, Counter()
    for record in records:
        context = record.get("context", {})
        problem = context.get("problem", {})
        if not isinstance(problem, dict): problem = {}
        data_id = problem.get("data_id", context.get("data_id"))
        env = record.get("environment")
        environment_id = env.get("environment_id") if isinstance(env, dict) else None
        backend = context.get("backend", context.get("requested_backend"))
        if isinstance(backend, dict): backend = backend.get("requested", backend.get("executed"))
        missing = []
        if record.get("validation_status") != "validated_diagnostic": missing.append("correctness_or_provenance")
        if not isinstance(record.get("source_fingerprint"), str) or not re.fullmatch("[0-9a-f]{64}", record["source_fingerprint"]):
            missing.append("exact_source_content_fingerprint")
        if not isinstance(data_id, str) or not re.fullmatch("[0-9a-f]{64}", data_id): missing.append("exact_shared_data_hash")
        if not isinstance(environment_id, str) or not environment_id: missing.append("environment_group")
        if not record.get("timing_scope") or not context.get("transfer_policy"): missing.append("timing_and_transfer_scope")
        if type(record.get("sample_count")) is not int or record.get("sample_count", 0) < 1 or "summary_seconds" not in record:
            missing.append("positive_raw_samples")
        if type(context.get("warmup")) is not int or context.get("repeats") != record.get("sample_count"):
            missing.append("warmup_and_repeat_contract")
        if backend not in ("numpy", "cupy", "torch"): missing.append("backend")
        if backend in ("cupy", "torch") and not re.fullmatch(r"cuda:\d+", str(context.get("synchronized_device", ""))):
            missing.append("concrete_synchronized_device")
        # Matching seeds are not paired bootstrap/RNG protocols.
        if problem.get("resamples") is not None or "bootstrap" in str(context.get("case", "")):
            missing.append("stochastic_protocol_alignment_requires_separate_review")
        if missing:
            withheld.update(set(missing)); continue
        key = canonical([record.get("campaign_id"), record.get("suite_or_lane"), record.get("source_commit"), record.get("source_fingerprint"),
                         environment_id, problem, data_id, record.get("configuration"),
                         context.get("case", context.get("case_id", context.get("operation"))),
                         record["timing_scope"], context["transfer_policy"], context["warmup"], context["repeats"],
                         record.get("cpu_thread_profile"), context.get("selector_policy")])
        groups.setdefault(key, {}).setdefault(backend, []).append(record)
    candidates = []
    for rows in groups.values():
        baseline = rows.get("numpy", [])
        # Duplicate candidate baselines need explicit disambiguation, not first-match pairing.
        if len(baseline) != 1: continue
        for backend in ("cupy", "torch"):
            if len(rows.get(backend, [])) != 1: continue
            other = rows[backend][0]
            candidates.append({"baseline_record_id": baseline[0]["record_id"], "candidate_record_id": other["record_id"],
                               "status": "recorded_metadata_match_requires_promotion_review", "performance_eligible": False,
                               "speedup": None, "remaining_requirements": ["method-specific objective/precision equivalence",
                                   "executed-device and numerical-path evidence review", "operator statement scope review, distinct from machine exclusivity",
                                   "explicit canonical promotion review"]})
    return candidates, dict(withheld)


def export_roots(roots, output_dir, *, repo, coverage_ledger=None, operator_attestation=None, _validators=None):
    out = Path(output_dir).expanduser().resolve()
    if out.exists(): raise ValueError("Refusing existing export destination: " + str(out))
    if not out.parent.is_dir(): raise ValueError("Export parent must already exist")
    if any(out.is_relative_to(Path(root).expanduser().resolve()) for root in roots):
        raise ValueError("Export must remain outside immutable input evidence roots")
    validators = _validators or Validators(repo)
    campaigns, jobs, records = [], [], []
    accumulated_bytes = 0
    discovered = discover_roots(roots)
    if not discovered and not coverage_ledger:
        raise ValueError("No supported native campaign found; supply a coverage ledger to represent not-run/blocked plans")
    for root, kind in discovered:
        hashes, root_errors = verify_inventory(root)
        input_bytes = sum(safe_path(root, name).stat().st_size for name in hashes
                          if Path(name).name in {"raw.json", "qualification.json", "qualified.json", "plan.json"}
                          and safe_path(root, name).is_file())
        if input_bytes > MAX_CAMPAIGN_INPUT_BYTES:
            raise ValueError("Single campaign exceeds the 128 MiB evidence-input budget: " + str(root) +
                             "; retain its original evidence and export a smaller native batch, not a successful truncated preview")
        wrapper, wrapper_errors = batch_context(root)
        root_errors.extend(wrapper_errors)
        def load(relative, errors):
            try:
                if relative not in hashes: errors.append("Unsealed evidence: " + relative)
                value = read_json(safe_path(root, relative))
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                return value
            except (OSError, ValueError, TypeError) as exc:
                errors.append(relative + ": " + str(exc)); return {}
        plan, summary = load("plan.json", root_errors), load("summary.json", root_errors)
        before, after = load("source-before.json", root_errors), load("source-after.json", root_errors)
        sha, fingerprint, source_errors = source_state(before, after)
        root_errors.extend(source_errors)
        if summary.get("source_changed") is True: root_errors.append("Runner reported source change")
        campaign_id = hashlib.sha256(canonical({"path": str(root), "plan_sha256": hashes.get("plan.json")}).encode()).hexdigest()[:16]
        campaign = {"campaign_id": campaign_id, "path": str(root), "kind": kind,
                    "batch_context": wrapper,
                    "reported_status": summary.get("status"), "source_commit": sha,
                    "source_fingerprint": fingerprint, "integrity_errors": root_errors,
                    "capability_exclusions": plan.get("capability_exclusions", []),
                    "checksum_manifest_sha256": digest(root / "SHA256SUMS.json") if (root / "SHA256SUMS.json").is_file() else None}
        campaigns.append(campaign)
        planned = plan.get("jobs", [])
        if not isinstance(planned, list): raise ValueError("Malformed native jobs list")
        summary_rows = summary.get("results", [])
        if not isinstance(summary_rows, list) or any(not isinstance(row, dict) for row in summary_rows):
            raise ValueError("Malformed native summary results")
        for job in planned:
            if not isinstance(job, dict): raise ValueError("Native job must be an object")
            errors = list(root_errors)
            identifier = job.get("id")
            if not isinstance(identifier, str): raise ValueError("Missing native job ID")
            config_id = job.get("config_id")
            native_name = identifier if kind == "registered" else identifier + "-" + str(config_id)[:12]
            folder = safe_path(root, native_name)
            job_id = campaign_id + ":" + native_name
            reported = next((r for r in summary_rows if (r.get("id") == identifier if kind == "registered" else
                            r.get("lane_id") == identifier and r.get("config_id") == config_id)), {})
            item = {"job_id": job_id, "campaign_id": campaign_id, "native_job_id": identifier,
                    "suite_or_lane": job.get("suite", identifier), "configuration_id": config_id,
                    "configuration": job.get("configuration", {}),
                    "expected": job.get("expected_rows", job.get("expected_cells", [])),
                    "reported_status": reported.get("status", "not_run"),
                    "raw_path": str(folder / "raw.json"), "raw_sha256": None,
                    "validation_status": "not_run", "correctness_status": "unavailable",
                    "evidence_errors": errors, "source_commit": sha, "source_fingerprint": fingerprint,
                    "performance_eligible": False, "canonical_dashboard_ready": False}
            jobs.append(item)
            if kind == "independent" and job.get("adapter") is None:
                item.update(validation_status="blocked", reason=job.get("blocked_reason", reported.get("reason")))
                continue
            if not folder.exists():
                item["reason"] = reported.get("reason", "Planned job has no evidence directory")
                continue
            sidecar_name = "qualification.json" if kind == "registered" else "qualified.json"
            sidecar = load(native_name + "/" + sidecar_name, errors)
            item["qualification_path"] = str(folder / sidecar_name)
            process = load(native_name + "/process.json", errors) if kind == "registered" else sidecar.get("process", {})
            if not isinstance(process, dict): process = {}
            if process.get("reason") or type(process.get("exit_code")) is not int or process.get("exit_code") != 0:
                errors.append("Missing/successless process exit or process stop reason")
            runtime = load(native_name + "/runtime.after.json", errors)
            runtime_before = load(native_name + "/runtime.before.json", errors)
            if type(runtime.get("exit_code")) is not int or runtime.get("exit_code") != 0 or runtime.get("error") or runtime.get("resource_guard"):
                errors.append("Runtime finalizer did not record a successful child exit")
            if not runtime.get("before") and runtime_before.get("before"):
                runtime = dict(runtime, before=runtime_before["before"])
            threads = thread_profile(runtime, job, kind)
            item["cpu_thread_profile"] = threads
            if not threads["cap_verified"]: errors.append("Actual numerical thread bounds were not verified")
            job_after = load(native_name + "/source-after.json", errors)
            errors.extend(source_state(before, job_after)[2])
            raw = load(native_name + "/raw.json", errors)
            if not isinstance(raw, dict): errors.append("Raw JSON is not an object"); raw = {}
            raw_file = folder / "raw.json"
            if raw_file.is_file() and not raw_file.is_symlink(): item["raw_sha256"] = digest(raw_file)
            try:
                reassessed = validators.assess(kind, job, folder, process, sha)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                reassessed = {}; errors.append("Reassessment failed: " + str(exc))
            item["reassessment"] = reassessed
            item["recorded_qualification"] = sidecar
            item["raw_performance_eligible"] = raw.get("performance_eligible")
            item["raw_canonical_dashboard_ready"] = raw.get("canonical_dashboard_ready")
            if kind == "registered":
                correct = reassessed.get("correctness", {}).get("status", "unavailable")
                errors.extend(reassessed.get("correctness", {}).get("errors", []))
                errors.extend(reassessed.get("correctness", {}).get("uncovered", []))
                if reassessed.get("expanded_inventory", {}).get("exact_match") is not True:
                    errors.append("Expected expanded row identity multiset not verified")
                if sidecar.get("correctness", {}).get("status") != "complete":
                    errors.append("Stored qualification was not complete")
                valid = correct == "complete"
            else:
                correct = reassessed.get("correctness_status", "unavailable")
                valid = reassessed.get("status") == "complete" and correct in PASS
                if sidecar.get("status") != "complete": errors.append("Stored independent qualification was not complete")
                if reassessed.get("status") != "complete": errors.append(reassessed.get("reason", "Independent protocol did not revalidate"))
            item["correctness_status"] = correct
            item["validation_status"] = "validated_diagnostic" if valid and not errors else "withheld"
            item["environment"] = raw.get("environment", {"environment_id": plan.get("environment_id"), "runtime_packages": runtime.get("packages", {})})
            for measurement in timing_records(raw):
                rec = {key: item.get(key) for key in ("job_id", "campaign_id", "suite_or_lane", "raw_path", "raw_sha256",
                        "correctness_status", "validation_status", "evidence_errors", "cpu_thread_profile",
                        "source_commit", "source_fingerprint", "environment", "performance_eligible", "canonical_dashboard_ready")}
                rec.update(measurement)
                rec["operator_attestation"] = wrapper.get("operator_attestation") if wrapper else None
                # Preserve producer flags as facts, separately from effective policy.
                context = measurement["context"]
                rec["raw_performance_eligible"] = context.get("performance_eligible", raw.get("performance_eligible"))
                rec["raw_canonical_dashboard_ready"] = context.get("canonical_dashboard_ready", raw.get("canonical_dashboard_ready"))
                rec["raw_eligibility_scope"] = "nearest recorded ancestor, then raw document; missing is null"
                rec["configuration"] = item["configuration"]
                rec["record_id"] = hashlib.sha256((job_id + ":" + measurement["raw_pointer"]).encode()).hexdigest()[:24]
                records.append(rec)
                accumulated_bytes += len(canonical(rec).encode("utf-8"))
                if accumulated_bytes > MAX_JSON_BYTES:
                    raise ValueError("Single campaign or selected aggregate exceeds the 64 MiB preview budget: " + str(root) +
                                     "; use --shard-by-campaign for multiple roots; an oversized single campaign remains explicitly failed")
            accumulated_bytes += len(canonical(item).encode("utf-8"))
            if accumulated_bytes > MAX_JSON_BYTES:
                raise ValueError("Single campaign or selected aggregate metadata exceeds the 64 MiB preview budget: " + str(root))
    ledger = None
    if coverage_ledger:
        ledger = {"path": str(Path(coverage_ledger).resolve()), "sha256": digest(coverage_ledger), "document": read_json(coverage_ledger)}
    payload = {"schema_version": SCHEMA, "campaigns": campaigns, "jobs": jobs, "records": records,
               "coverage_ledger": ledger, "coverage_scope": "Supplied native plans only; no claim of all API or full catalog completion",
               "qualification_scope": "Offline stored-artifact contract reassessment, not numerical oracle rerun or authenticated provenance",
               "validator_source": validators.source, "performance_eligible": False, "canonical_dashboard_ready": False,
               "comparisons": [], "exclusivity": {"machine_verified": False, "machine_status": "UNVERIFIED_EXCLUSIVITY",
                   "operator_attestation": load_attestation(operator_attestation)},
               "counts": {"campaigns": len(campaigns), "jobs": len(jobs), "timing_fields": len(records),
                          "job_validation_status": dict(Counter(job["validation_status"] for job in jobs))}}
    candidates, withholding = comparison_candidates(records)
    payload["comparison_candidates"] = candidates
    payload["comparison_withheld_requirement_counts"] = withholding
    payload["export_sha256"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    gate = {"schema_version": "statgpu-dashboard-export-gate-v1", "source_export_sha256": payload["export_sha256"],
            "canonical_dashboard_ready": False, "performance_eligible": False,
            "reason": "Diagnostic display is not canonical performance promotion; no canonical review or machine-verified exclusivity",
            "dashboard_adapter": "qualification_preview", "speedups_emitted": 0,
            "withheld": [{"job_id": job["job_id"], "reason": job.get("reason"), "evidence_errors": job["evidence_errors"]}
                         for job in jobs if job["validation_status"] != "validated_diagnostic"]}
    serialized = {name: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
                  for name, value in (("preview_data.json", payload), ("canonical_export_gate.json", gate))}
    if any(len(value.encode("utf-8")) > MAX_JSON_BYTES for value in serialized.values()):
        raise ValueError("Aggregate preview exceeds the 64 MiB viewer budget; export fewer campaign/batch roots separately (no evidence was written or dropped)")
    out.mkdir()
    for name, value in serialized.items():
        (out / name).write_text(value, encoding="utf-8")
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in (
        record["suite_or_lane"], record["timing_scope"] or "Unspecified scope", record.get("summary_seconds", "Not chartable"),
        record.get("statistic", "Raw field"), record["validation_status"], record["raw_pointer"])) + "</tr>" for record in records)
    page = '<!doctype html><meta charset="utf-8"><title>Benchmark evidence preview</title><style>body{font:15px system-ui;margin:2rem}td,th{padding:.5rem;border:1px solid #ccc;text-align:left}table{border-collapse:collapse}p{max-width:80ch}</style><h1>Diagnostic benchmark evidence</h1><p>No canonical performance promotion, speedup, exclusive-GPU proof, or whole-pipeline native claim. Unknown scope stays unknown. Operator statements are separate from machine evidence.</p><p><a href="preview_data.json">Complete machine-readable evidence index</a> · <a href="canonical_export_gate.json">Export gate</a></p><p>' + html.escape(json.dumps(payload["counts"])) + '</p><table><thead><tr><th>Suite/lane</th><th>Scope</th><th>Seconds</th><th>Statistic</th><th>Qualification</th><th>Raw JSON pointer</th></tr></thead><tbody>' + rows + '</tbody></table>'
    (out / "preview.html").write_text(page, encoding="utf-8")
    return payload


def export_shards(roots, output_dir, *, repo, coverage_ledger=None, operator_attestation=None, _validators=None):
    """One bounded preview per native campaign, with a complete failure index."""
    out = Path(output_dir).expanduser().resolve()
    if out.exists() or not out.parent.is_dir(): raise ValueError("Shard output must be fresh with an existing parent")
    if any(out.is_relative_to(Path(root).expanduser().resolve()) for root in roots):
        raise ValueError("Shard export must remain outside immutable input evidence roots")
    discovered = discover_roots(roots)
    if not discovered and not coverage_ledger:
        raise ValueError("No native campaign or coverage ledger supplied")
    validators = _validators or Validators(repo)
    ledger = read_json(coverage_ledger) if coverage_ledger else None
    if ledger is not None and (not isinstance(ledger, dict) or ledger.get("schema_version") != "statgpu-batch-coverage-ledger-v1"):
        raise ValueError("Unsupported coverage ledger schema")
    index = {"schema_version": "statgpu-preview-shard-index-v1", "status": "complete", "shards": [],
             "performance_eligible": False, "canonical_dashboard_ready": False,
             "counts": {"campaigns": len(discovered), "jobs": 0, "timing_fields": 0},
             "coverage_scope": "All supplied native campaigns; planned not-run/blocked slots are retained in the separate coverage ledger",
             "coverage_ledger": None}
    out.mkdir()
    if coverage_ledger:
        target = out / "coverage_ledger.json"
        target.write_bytes(Path(coverage_ledger).read_bytes())
        index["coverage_ledger"] = {"path": target.name, "sha256": digest(target), "counts": ledger.get("counts", {}),
                                    "plan_sha256": ledger.get("plan_sha256"), "source": ledger.get("source", {})}
    for number, (root, kind) in enumerate(discovered, 1):
        folder_name = f"campaign-{number:04d}"
        entry = {"campaign_root": str(root), "kind": kind, "path": folder_name + "/preview_data.json"}
        try:
            payload = export_roots([root], out / folder_name, repo=repo, operator_attestation=operator_attestation,
                                   _validators=validators)
            entry.update(status="complete", sha256=digest(out / entry["path"]), counts=payload["counts"])
            index["counts"]["jobs"] += payload["counts"]["jobs"]
            index["counts"]["timing_fields"] += payload["counts"]["timing_fields"]
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            index["status"] = "partial"
            entry.update(status="failed", reason=str(exc))
        index["shards"].append(entry)
    index["index_sha256"] = hashlib.sha256(canonical(index).encode()).hexdigest()
    (out / "preview_index.json").write_text(json.dumps(index, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    links = "".join('<li>' + html.escape(entry["status"]) + ': <a href="' + html.escape(entry["path"], quote=True) + '">' + html.escape(entry["path"]) + '</a> ' + html.escape(entry.get("reason", "")) + '</li>' for entry in index["shards"])
    (out / "preview_index.html").write_text('<!doctype html><meta charset="utf-8"><title>Complete diagnostic export index</title><h1>Diagnostic campaign shards</h1><p>Load each preview_data.json in the dashboard Diagnostic preview file picker. No measurement, eligibility, or speedup is invented. A failed shard remains listed here.</p><p>' + html.escape(json.dumps(index["counts"])) + '</p><ul>' + links + '</ul><a href="preview_index.json">Machine-readable index</a>', encoding="utf-8")
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Trusted checkout containing pure qualifier helpers")
    parser.add_argument("--campaign-root", type=Path, action="append", default=[])
    parser.add_argument("--results-root", type=Path, help="Batch results root; inspects direct sealed wrapper directories only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-ledger", type=Path)
    parser.add_argument("--operator-attestation", type=Path)
    parser.add_argument("--shard-by-campaign", action="store_true", help="Export every native campaign into a bounded shard plus a complete index")
    args = parser.parse_args(argv)
    roots = args.campaign_root + ([args.results_root] if args.results_root else [])
    if not roots: parser.error("Supply --results-root or --campaign-root")
    try:
        export = export_shards if args.shard_by_campaign else export_roots
        result = export(roots, args.output_dir, repo=args.repo,
                              coverage_ledger=args.coverage_ledger, operator_attestation=args.operator_attestation)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["counts"], sort_keys=True))
    return 2 if result.get("status") == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
