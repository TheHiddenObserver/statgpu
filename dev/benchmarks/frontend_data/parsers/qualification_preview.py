"""Explicit diagnostic adapter for new campaign evidence, not promotion authority.

All original measurements remain in the bridge preview. This adapter normalizes
only records whose identity and timing scope the dashboard can represent without
inventing a fit boundary, feature dimension, backend, or performance entitlement.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

SCHEMA = "statgpu-benchmark-update-preview-v1"
CATEGORY = {"category_id": "diagnostic_evidence", "name_en": "Diagnostic evidence", "name_zh": "诊断证据"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(prefix, value):
    return prefix + "-" + hashlib.sha256(_canonical(value).encode()).hexdigest()[:16]


def _read(filepath):
    if filepath.is_symlink() or not filepath.is_file() or filepath.stat().st_size > 64 * 1024**2:
        raise ValueError("Preview must be a regular JSON file no larger than 64 MiB")
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError("Duplicate preview JSON key: " + key)
            out[key] = value
        return out
    def invalid(token): raise ValueError("Nonfinite preview value: " + token)
    document = json.loads(filepath.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported qualification preview schema")
    stored = document.get("export_sha256")
    payload = {k: v for k, v in document.items() if k != "export_sha256"}
    if stored != hashlib.sha256(_canonical(payload).encode()).hexdigest():
        raise ValueError("Qualification preview export digest mismatch")
    if document.get("performance_eligible") is not False or document.get("canonical_dashboard_ready") is not False:
        raise ValueError("Diagnostic preview cannot assert canonical/performance eligibility")
    if document.get("comparisons") != []:
        raise ValueError("Diagnostic preview cannot contain published speedup comparisons")
    if not isinstance(document.get("records"), list): raise ValueError("Missing preview records")
    return document


SKLEARN_ADAPTER = "sklearn-scoped-cv-v2"


def _sklearn_identity(record, context):
    """Bind an external timing to its versioned CPU run, never to host backend."""
    if record.get("source_schema") != "statgpu-scoped-cv-candidate-v2":
        return "EXTERNAL_FRAMEWORK_REQUIRES_EXPLICIT_ADAPTER"
    match = re.fullmatch(r"/results/\d+/case/runs/(\d+)/(timing|repeat_samples/(\d+))/(cv_evaluation_ms|final_refit_ms|total_fit_ms)",
                         str(record.get("raw_pointer", "")))
    if not match:
        return "EXTERNAL_FRAMEWORK_REQUIRES_EXPLICIT_ADAPTER"
    case = context.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("runs"), list):
        return "EXTERNAL_FRAMEWORK_REQUIRES_EXPLICIT_ADAPTER"
    case_root = str(record["raw_pointer"]).split("/runs/", 1)[0]
    if (record.get("row_pointer") != case_root or context.get("scope") != record.get("timing_scope")
            or not isinstance(context.get("scope"), str) or not context["scope"].strip()
            or context.get("device", "cpu") != "cpu" or context.get("instrumented", True) is not True
            or context.get("mode") is not None
            or ("scope_source_pointer" in context and context["scope_source_pointer"] !=
                case_root.removesuffix("/case") + "/protocol/timing_scope")):
        return "EXTERNAL_ADAPTER_PROVENANCE_MISMATCH"
    index = int(match.group(1))
    if index >= len(case["runs"]):
        return "EXTERNAL_ADAPTER_PROVENANCE_MISMATCH"
    run = case["runs"][index]
    if (not isinstance(run, dict) or run.get("framework") != "sklearn"
            or run.get("backend") is not None or run.get("device") != "cpu"
            or context.get("backend") is not None or context.get("requested_backend") is not None
            or run.get("status") != "success" or context.get("status") != "success"):
        return "EXTERNAL_ADAPTER_PROVENANCE_MISMATCH"
    key = match.group(4)
    if (record.get("unit") != "milliseconds" or record.get("timing_key") != key
            or context.get("timing_kind") != "cv_diagnostic_stage"
            or context.get("timing_phase") != key.removesuffix("_ms")
            or any(not isinstance(record.get(name), str) or not re.fullmatch(pattern, record[name])
                   for name, pattern in (("source_commit", r"[0-9a-f]{40}"),
                                         ("raw_sha256", r"[0-9a-f]{64}"),
                                         ("source_fingerprint", r"[0-9a-f]{64}")))):
        return "EXTERNAL_ADAPTER_PROVENANCE_MISMATCH"
    dataset = case.get("dataset", {})
    if (not isinstance(dataset, dict) or dataset.get("n_samples") != context.get("n_samples")
            or dataset.get("n_features") != context.get("n_features")):
        return "EXTERNAL_ADAPTER_PROVENANCE_MISMATCH"
    # Reject conflicting aliases even when another field has higher precedence.
    dimension_sources = [context, context.get("problem"), context.get("shape")]
    if isinstance(context.get("problem"), dict): dimension_sources.append(context["problem"].get("shape"))
    for dimensions in dimension_sources:
        if dimensions is None: continue
        if not isinstance(dimensions, dict): return "EXTERNAL_ADAPTER_PROVENANCE_MISMATCH"
        for keys, expected in ((("n", "n_samples"), dataset["n_samples"]),
                               (("p", "n_features", "dimension"), dataset["n_features"])):
            if any(key in dimensions and (type(dimensions[key]) is not int or dimensions[key] != expected) for key in keys):
                return "EXTERNAL_ADAPTER_PROVENANCE_MISMATCH"
    samples, timing = run.get("repeat_samples"), run.get("timing")
    if (not isinstance(samples, list) or not samples or not isinstance(timing, dict)
            or type(context.get("repeats")) is not int or len(samples) != context["repeats"]
            or not all(isinstance(item, dict) and _positive(item.get(key)) for item in samples)
            or not _same_timing(timing.get(key), statistics.median(item[key] for item in samples))):
        return "EXTERNAL_ADAPTER_TIMING_MISMATCH"
    sample_index = int(match.group(3)) if match.group(3) is not None else None
    if sample_index is not None and sample_index >= len(samples):
        return "EXTERNAL_ADAPTER_TIMING_MISMATCH"
    value = timing[key] if sample_index is None else samples[sample_index][key]
    if (not _same_timing(record.get("raw_value"), value)
            or not _same_timing(record.get("summary_seconds"), value / 1000)):
        return "EXTERNAL_ADAPTER_TIMING_MISMATCH"
    return None


def _identity(record):
    context = record.get("context", {})
    if not isinstance(context, dict): return None, "MALFORMED_CONTEXT"
    external = context.get("framework") == "sklearn"
    if external:
        reason = _sklearn_identity(record, context)
        if reason: return None, reason
    elif context.get("framework") not in (None, "statgpu"):
        return None, "EXTERNAL_FRAMEWORK_REQUIRES_EXPLICIT_ADAPTER"
    problem = context.get("problem", {})
    if not isinstance(problem, dict): problem = {}
    shape = problem.get("shape", context.get("shape", problem))
    if not isinstance(shape, dict): shape = {}
    n = shape.get("n", shape.get("n_samples", context.get("n_samples", context.get("n"))))
    p = shape.get("p", shape.get("n_features", shape.get("dimension", context.get("n_features", context.get("p")))))
    if external and (n != context.get("n_samples") or p != context.get("n_features")):
        return None, "EXTERNAL_ADAPTER_PROVENANCE_MISMATCH"
    classification = record.get("suite_or_lane") == "classification-metrics"
    if classification and p is None: p = 0
    if type(n) is not int or n < 1: return None, "MISSING_EXACT_SAMPLE_DIMENSION"
    if type(p) is not int or p < 0 or (p == 0 and not classification):
        return None, "MISSING_OR_UNSUPPORTED_FEATURE_DIMENSION"
    backend = context.get("backend", context.get("requested_backend"))
    if isinstance(backend, dict):
        if backend.get("requested") is not None and backend.get("executed") is not None and backend["requested"] != backend["executed"]:
            return None, "CONTRADICTORY_MEASURED_BACKEND"
        backend = backend.get("requested", backend.get("executed"))
    if not external and backend not in ("numpy", "cupy", "torch"): return None, "UNRESOLVED_MEASURED_BACKEND"
    nested = context.get("case")
    nested = nested if isinstance(nested, dict) else {}
    if context.get("identity_conflicts"):
        return None, "CONFLICTING_CASE_IDENTITY"
    for key in ("case_id", "model_id"):
        if key in context and key in nested and context[key] != nested[key]:
            return None, "CONFLICTING_CASE_IDENTITY"
    # case_id and model_id describe different things, and need not be equal.
    # Prefer the exact case ID; a structured case must not mask a model ID.
    candidates = (context.get("case_id"), nested.get("case_id"), context.get("case"),
                  context.get("model_id"), nested.get("model_id"), nested.get("model_name"),
                  context.get("model"), context.get("operation"))
    case = next((item for item in candidates if isinstance(item, str) and item.strip()), None)
    if not isinstance(case, str):
        parameters = problem.get("parameters", {})
        case = parameters.get("model") if isinstance(parameters, dict) else None
    if not isinstance(case, str) and isinstance(context.get("benchmark"), dict):
        case = context["benchmark"].get("public_api")
    if not isinstance(case, str) or not case: return None, "UNRESOLVED_CASE_IDENTITY"
    return {"n": n, "p": p, "backend": backend, "case": case, "classification": classification,
            **({"framework": "sklearn"} if external else {})}, None


def _cv_observation(record):
    """Only this versioned CV layout proves the cross-parent alias relation."""
    if record.get("source_schema") != "statgpu-scoped-cv-candidate-v2":
        return None
    path = record.get("raw_pointer", "")
    if not isinstance(path, str):
        return None
    clean = re.fullmatch(r"(/results/\d+/clean_public_fit)/(raw_seconds|timing/median_seconds|repeat_samples/(\d+)/total_fit_ms)", path)
    if clean:
        role = "samples" if clean.group(2) == "raw_seconds" else "summary" if clean.group(2) == "timing/median_seconds" else "repeat"
        return clean.group(1), "total_fit", role, int(clean.group(3)) if clean.group(3) else None
    phase = re.fullmatch(r"(/results/\d+/case/runs/\d+)/(timing|repeat_samples/(\d+))/(cv_evaluation_ms|final_refit_ms|total_fit_ms)", path)
    if phase:
        return phase.group(1), phase.group(4), "summary" if phase.group(2) == "timing" else "repeat", int(phase.group(3)) if phase.group(3) else None
    return None


def _alias_context(record):
    context = record.get("context", {})
    if not isinstance(context, dict):
        return None
    # Never de-duplicate different frameworks, modes, or timing boundaries just
    # because numeric values happen to agree or their fields share a parent.
    return [record.get("campaign_id"), record.get("job_id"), record.get("raw_path"),
            _identity(record), record.get("timing_scope"), {key: context.get(key) for key in
                ("framework", "backend", "requested_backend", "instrumented", "mode",
                 "timing_kind", "timing_phase", "case_id", "model_id", "dtype")}]


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _same_timing(left, right):
    return _positive(left) and _positive(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0)


def _alias_dispositions(records):
    """Validate aliases before suppressing display duplicates; keep all raw rows."""
    reasons, groups = {}, {}
    generic_aliases = {"raw_seconds": {"median_seconds", "mean_seconds", "seconds"},
                       "samples_seconds": {"median_seconds", "mean_seconds", "seconds"},
                       "seconds": {"median_seconds", "mean_seconds"},
                       "fit_seconds": {"median_seconds", "mean_seconds"},
                       "times_ms": {"time_ms", "time_ms_mean"},
                       "fit_ms_all": {"fit_ms", "fit_ms_mean"}}
    for record in records:
        if not isinstance(record, dict):
            continue
        cv = _cv_observation(record)
        parent = str(record.get("raw_pointer", "")).rsplit("/", 1)[0]
        key = _canonical([_alias_context(record), list(cv[:2]) if cv else [record.get("row_pointer"), parent]])
        groups.setdefault(key, []).append((record, cv))

    def mark(group, reason):
        for row in group:
            reasons[row["record_id"]] = reason

    for entries in groups.values():
        rows = [row for row, _ in entries]
        if entries[0][1]:
            sample_rows = [row for row, cv in entries if cv[2] == "samples"]
            summaries = [row for row, cv in entries if cv[2] == "summary"]
            repeats = [(row, cv[3]) for row, cv in entries if cv[2] == "repeat"]
            if len(sample_rows) > 1 or len(summaries) > 1 or len({index for _, index in repeats}) != len(repeats):
                mark(rows, "INCONSISTENT_TIMING_ALIASES")
                continue
            canonical_row = sample_rows[0] if sample_rows else summaries[0] if summaries else None
            if canonical_row is None or not _positive(canonical_row.get("summary_seconds")):
                mark(rows, "UNVERIFIABLE_TIMING_ALIASES")
                continue
            identity, identity_reason = _identity(canonical_row)
            if (canonical_row.get("validation_status") != "validated_diagnostic"
                    or canonical_row.get("correctness_status") not in ("complete", "pass", "passed", "success")
                    or canonical_row.get("evidence_errors") or identity_reason):
                mark(rows, "UNQUALIFIED_TIMING_ALIAS_SOURCE")
                continue
            if sample_rows:
                samples = canonical_row.get("raw_value")
                if not isinstance(samples, list) or not samples or not all(_positive(value) for value in samples):
                    mark(rows, "UNVERIFIABLE_TIMING_ALIASES")
                    continue
                factor = 1000 if canonical_row.get("unit") == "milliseconds" else 1
                samples = [value / factor for value in samples]
                valid = canonical_row.get("sample_count") == len(samples)
                valid = valid and _same_timing(canonical_row["summary_seconds"], statistics.median(samples))
                valid = valid and all(index < len(samples) and _same_timing(row.get("summary_seconds"), samples[index]) for row, index in repeats)
            elif repeats:
                repeats.sort(key=lambda pair: pair[1])
                samples = [row.get("summary_seconds") for row, _ in repeats]
                expected = canonical_row.get("context", {}).get("repeats")
                valid = all(_positive(value) for value in samples)
                valid = valid and [index for _, index in repeats] == list(range(len(repeats)))
                valid = valid and (expected is None or type(expected) is int and expected == len(repeats))
                valid = valid and _same_timing(canonical_row["summary_seconds"], statistics.median(samples))
            else:
                # A standalone scalar summary is still a reported observation;
                # repeat-only groups without a canonical aggregate were blocked above.
                continue
            valid = valid and all(_same_timing(row.get("summary_seconds"), statistics.median(samples)) for row in summaries)
            valid = valid and all(_same_timing(row.get("summary_seconds"),
                row["raw_value"] / (1000 if row.get("unit") == "milliseconds" else 1))
                for row in rows if row is not canonical_row or not sample_rows
                if _positive(row.get("raw_value")))
            valid = valid and all(_positive(row.get("raw_value")) for row in rows
                                 if row is not canonical_row or not sample_rows)
            if not valid:
                mark(rows, "INCONSISTENT_TIMING_ALIASES")
            else:
                mark([row for row in rows if row is not canonical_row], "ALIAS_OF_RETAINED_OBSERVATIONS")
            continue

        for raw in rows:
            timing_key = raw.get("timing_key", str(raw.get("raw_pointer", "")).rsplit("/", 1)[-1])
            if type(raw.get("sample_count")) is not int or raw["sample_count"] < 1 or not _positive(raw.get("summary_seconds")):
                continue
            for summary in rows:
                key = summary.get("timing_key", str(summary.get("raw_pointer", "")).rsplit("/", 1)[-1])
                if "sample_count" in summary or key not in generic_aliases.get(timing_key, set()):
                    continue
                expected = raw["summary_seconds"]
                if "mean" in key:
                    values = raw.get("raw_value")
                    if not isinstance(values, list) or not values or not all(_positive(value) for value in values):
                        reasons[summary["record_id"]] = "UNVERIFIABLE_TIMING_ALIASES"
                        continue
                    expected = statistics.mean(values) / (1000 if raw.get("unit") == "milliseconds" else 1)
                if _same_timing(summary.get("summary_seconds"), expected):
                    reasons.setdefault(summary["record_id"], "SUMMARY_ALIAS_OF_RETAINED_RAW_SAMPLES")
                else:
                    mark([raw, summary], "INCONSISTENT_TIMING_ALIASES")
    return reasons


def _external_lineage_reason(document, record):
    """External adapters require one consistent job and campaign provenance."""
    jobs = document.get("jobs", [])
    campaigns = document.get("campaigns", [])
    if not isinstance(jobs, list) or not isinstance(campaigns, list):
        return "EXTERNAL_ADAPTER_LINEAGE_MISMATCH"
    jobs = [item for item in jobs if isinstance(item, dict) and item.get("job_id") == record.get("job_id")]
    campaigns = [item for item in campaigns if isinstance(item, dict) and item.get("campaign_id") == record.get("campaign_id")]
    if len(jobs) != 1 or len(campaigns) != 1:
        return "EXTERNAL_ADAPTER_LINEAGE_MISMATCH"
    job, campaign = jobs[0], campaigns[0]
    fields = ("campaign_id", "source_commit", "source_fingerprint", "raw_path", "raw_sha256",
              "suite_or_lane", "configuration", "cpu_thread_profile", "environment",
              "validation_status", "correctness_status")
    if (any(key not in job or job[key] != record.get(key) for key in fields)
            or any(campaign.get(key) != record.get(key) for key in ("source_commit", "source_fingerprint"))
            or campaign.get("integrity_errors") or job.get("evidence_errors")
            or job.get("performance_eligible") is not False or job.get("canonical_dashboard_ready") is not False):
        return "EXTERNAL_ADAPTER_LINEAGE_MISMATCH"
    return None


def normalize_preview(document, filepath, env_id):
    runs, models, withheld = [], {}, []
    records = document["records"]
    seen = set()
    for record in records:
        if not isinstance(record, dict): raise ValueError("Nonobject diagnostic record")
        rid = record.get("record_id")
        if not isinstance(rid, str) or not rid or rid in seen: raise ValueError("Missing/duplicate diagnostic record ID")
        seen.add(rid)
    alias_reasons = _alias_dispositions(records)
    source_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
    seen = set()
    for record in records:
        if not isinstance(record, dict): raise ValueError("Nonobject diagnostic record")
        rid = record.get("record_id")
        if not isinstance(rid, str) or not rid or rid in seen: raise ValueError("Missing/duplicate diagnostic record ID")
        seen.add(rid)
        reason = None
        if record.get("validation_status") != "validated_diagnostic" or record.get("correctness_status") not in ("complete", "pass", "passed", "success") or record.get("evidence_errors"):
            reason = "CORRECTNESS_OR_EVIDENCE_NOT_QUALIFIED"
        if record.get("performance_eligible") is not False or record.get("canonical_dashboard_ready") is not False:
            raise ValueError("Diagnostic record attempted performance promotion")
        path = record.get("raw_pointer", "")
        if (record.get("suite_or_lane") == "six-family-cv"
                and record.get("source_schema") != "statgpu-scoped-cv-candidate-v2"
                and isinstance(path, str)
                and re.fullmatch(r"/results/\d+/(?:clean_public_fit/(?:raw_seconds|timing/median_seconds|repeat_samples/\d+/total_fit_ms)|case/runs/\d+/(?:timing|repeat_samples/\d+)/(?:cv_evaluation_ms|final_refit_ms|total_fit_ms))", path)):
            reason = reason or "CV_ALIAS_BINDING_REQUIRES_REEXPORT"
        scope = record.get("timing_scope")
        duration = record.get("summary_seconds")
        if not isinstance(scope, str) or not scope.strip(): reason = reason or "TIMING_SCOPE_NOT_RECORDED"
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            reason = reason or "NO_POSITIVE_REPRESENTABLE_TIMING"
        identity, identity_reason = _identity(record)
        lineage_reason = (_external_lineage_reason(document, record)
                          if identity and identity.get("framework", "statgpu") == "sklearn" else None)
        reason = reason or identity_reason or lineage_reason or alias_reasons.get(rid)
        if reason:
            withheld.append({"record_id": rid, "reason": reason, "raw_pointer": record.get("raw_pointer")})
            continue
        context = record["context"]
        observation = _cv_observation(record)
        if observation and "/case/runs/" in record["raw_pointer"]:
            scope += "; recorded phase: " + observation[1].removesuffix("_ms")
        model_id = record["suite_or_lane"] + ":" + identity["case"]
        models[model_id] = {"model_id": model_id, "primary_category_id": CATEGORY["category_id"], "category_ids": [CATEGORY["category_id"]]}
        scale = {"scale_key": f"n{identity['n']}_p{identity['p']}", "n_samples": identity["n"], "n_features": identity["p"],
                 "label": f"{identity['n']} observations" if identity["p"] == 0 else f"{identity['n']}×{identity['p']}"}
        if identity["p"] == 0: scale["feature_dimension_kind"] = "not_applicable"
        cpu = record.get("cpu_thread_profile", {})
        evidence = {"qualification": "diagnostic_only", "correctness_status": record["correctness_status"],
                    "validation_status": record["validation_status"], "performance_eligible": False,
                    "canonical_dashboard_ready": False, "machine_verified_exclusivity": False,
                    "raw_pointer": record["raw_pointer"], "source_commit": record.get("source_commit"),
                    "source_fingerprint": record.get("source_fingerprint"),
                    "cpu_threads_requested": cpu.get("requested_cap"), "cpu_threads_observed": cpu.get("observed_pool_threads"),
                    "raw_performance_eligible": record.get("raw_performance_eligible"), "evidence_errors": []}
        attestation = record.get("operator_attestation")
        if attestation is not None: evidence["operator_attestation"] = attestation
        timing = {"duration_ms": duration * 1000, "scope": scope, "statistic": record.get("statistic", "not_recorded"),
                  "unit": "ms", "quality": "measured" if "sample_count" in record else "reported", "source_file": filepath.name}
        if "sample_count" in record: timing["sample_count"] = record["sample_count"]
        local_env = env_id + "-" + hashlib.sha256(_canonical([record.get("environment"), record.get("source_commit"), cpu]).encode()).hexdigest()[:12]
        run = {"run_id": _identifier("diagnostic", rid), "env_id": local_env, "category_ids": [CATEGORY["category_id"]],
               "model_id": model_id, "framework": identity.get("framework", "statgpu"), "backend": identity["backend"], "scale": scale,
               "benchmark_session_id": record["campaign_id"], "comparison_id": "diagnostic:" + record["job_id"],
               "case_id": _identifier("case", [record["job_id"], identity, record["row_pointer"], context.get("problem")]),
               "method_config_id": _identifier("method", [record["configuration"], cpu, scope, record["raw_pointer"]]),
               "parameters": {"diagnostic_only": True, "scope": scope, "original_context": context,
                              "raw_path": record["raw_path"], "raw_sha256": record.get("raw_sha256")},
               "source": {"source_id": "transitional:qualification-preview:" + source_hash[:16], "file": filepath.name,
                          "original_path": str(filepath), "sha256": source_hash, "date": "unknown",
                          "parser": "qualification_preview", "parser_version": "1.0"},
               "metrics": {"operation_timing": timing}, "evidence": evidence}
        if identity.get("framework", "statgpu") == "sklearn":
            run["parameters"]["external_adapter"] = SKLEARN_ADAPTER
            run["parameters"]["external_adapter_provenance"] = {
                "source_schema": record["source_schema"], "device": "cpu",
                "summary_semantics": "recorded median validated against embedded repeat samples",
                "repeat_sample_count": context["repeats"],
                "framework_version": record.get("environment", {}).get("runtime_packages", {}).get("scikit-learn"),
            }
        runs.append(run)
    return runs, sorted(models.values(), key=lambda row: row["model_id"]), withheld


def parse_qualification_preview(filepath: Path, env_id: str):
    document = _read(filepath)
    runs, models, withheld = normalize_preview(document, filepath, env_id)
    warnings = [f"{row['record_id']}: unavailable normalized display ({row['reason']}); retained in diagnostic preview" for row in withheld]
    return runs, models, warnings
