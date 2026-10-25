"""Synthetic diagnostic exporter/parser contracts. No numerical/GPU workload."""
import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from dev.benchmarks.export_qualification_preview import build_bundle, build_external_manifest, main
from dev.benchmarks.frontend_data.parsers.qualification_preview import parse_qualification_preview
from dev.benchmarks.frontend_data.registry import PARSER_FUNCTIONS


def preview():
    return {"schema_version": "statgpu-benchmark-update-preview-v1", "performance_eligible": False,
            "canonical_dashboard_ready": False, "comparisons": [],
            "exclusivity": {"machine_verified": False, "operator_attestation": None},
            "records": [{"record_id": "raw-sample-record", "campaign_id": "campaign", "job_id": "job",
                "suite_or_lane": "model-families", "raw_path": "raw.json", "raw_sha256": "b" * 64,
                "raw_pointer": "/results/0/timing/raw_seconds", "row_pointer": "/results/0",
                "summary_seconds": .25, "statistic": "median_of_recorded_samples", "sample_count": 3,
                "timing_scope": "predict-only-public-api", "configuration": {"seed": 17},
                "context": {"case": "ols", "backend": "numpy", "problem": {"n": 20, "p": 2, "seed": 17}},
                "environment": {"environment_id": "synthetic"}, "source_commit": "a" * 40,
                "source_fingerprint": "b" * 64, "validation_status": "validated_diagnostic",
                "correctness_status": "complete", "evidence_errors": [], "raw_performance_eligible": None,
                "performance_eligible": False, "canonical_dashboard_ready": False,
                "cpu_thread_profile": {"requested_cap": 4, "observed_pool_threads": [1, 1], "cap_verified": True}}]}


def save(path, document):
    data = copy.deepcopy(document)
    data.pop("export_sha256", None)
    data["export_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(data))
    return path


def test_adapter_registered_without_canonical_source_registration():
    assert PARSER_FUNCTIONS["qualification_preview"] is parse_qualification_preview


def test_explicit_operation_scope_no_fabricated_fit_or_speedup(tmp_path):
    result = build_bundle(save(tmp_path / "preview.json", preview()))
    data = result["benchmark_data"]
    assert data["schema_version"] == "1.2.0"
    row = data["runs"][0]
    assert set(row["metrics"]) == {"operation_timing"}
    assert row["metrics"]["operation_timing"]["scope"] == "predict-only-public-api"
    assert row["metrics"]["operation_timing"]["duration_ms"] == 250
    assert row["evidence"]["performance_eligible"] is False
    assert result["source_inventory"]["registered_sources"] == 0
    assert result["source_inventory"]["not_canonical_ready_sources"] == 1


@pytest.mark.parametrize("change,reason", [
    (lambda r: r.update(timing_scope=None), "TIMING_SCOPE_NOT_RECORDED"),
    (lambda r: r.update(validation_status="withheld"), "CORRECTNESS_OR_EVIDENCE_NOT_QUALIFIED"),
    (lambda r: r.update(summary_seconds=0), "NO_POSITIVE_REPRESENTABLE_TIMING"),
    (lambda r: r["context"]["problem"].pop("p"), "MISSING_OR_UNSUPPORTED_FEATURE_DIMENSION"),
    (lambda r: r["context"].update(backend="unknown"), "UNRESOLVED_MEASURED_BACKEND"),
])
def test_unsupported_records_have_explicit_reason(tmp_path, change, reason):
    doc = preview(); change(doc["records"][0])
    bundle = build_bundle(save(tmp_path / "preview.json", doc))
    assert bundle["benchmark_data"]["runs"] == []
    assert bundle["parse_report"]["issues"][0]["code"] == reason


def test_classification_zero_features_is_explicitly_not_applicable(tmp_path):
    doc = preview(); row = doc["records"][0]
    row["suite_or_lane"] = "classification-metrics"; row["context"]["problem"].pop("p")
    result = build_bundle(save(tmp_path / "preview.json", doc))["benchmark_data"]["runs"][0]
    assert result["scale"]["n_features"] == 0
    assert result["scale"]["feature_dimension_kind"] == "not_applicable"


def test_raw_summary_not_double_counted_as_independent_measurement(tmp_path):
    doc = preview(); row = copy.deepcopy(doc["records"][0]); row.pop("sample_count")
    row.update(record_id="summary-record", raw_pointer="/results/0/timing/median_seconds", statistic="recorded_scalar:median_seconds")
    doc["records"].append(row)
    result = build_bundle(save(tmp_path / "preview.json", doc))
    assert len(result["benchmark_data"]["runs"]) == 1
    assert result["parse_report"]["issues"][0]["code"] == "SUMMARY_ALIAS_OF_RETAINED_RAW_SAMPLES"


def test_distinct_cold_scalar_is_not_removed_as_summary_alias(tmp_path):
    doc = preview(); row = copy.deepcopy(doc["records"][0]); row.pop("sample_count")
    row.update(record_id="cold-record", raw_pointer="/results/0/timing/cold_ms", statistic="recorded_scalar:cold_ms", summary_seconds=.6)
    doc["records"].append(row)
    result = build_bundle(save(tmp_path / "preview.json", doc))
    assert len(result["benchmark_data"]["runs"]) == 2


def test_operator_attestation_never_changes_canonical_qualification(tmp_path):
    doc = preview(); doc["records"][0]["operator_attestation"] = {"kind": "operator_attested", "machine_verified": False}
    row = build_bundle(save(tmp_path / "preview.json", doc))["benchmark_data"]["runs"][0]
    assert row["evidence"]["qualification"] == "diagnostic_only"
    assert row["evidence"]["machine_verified_exclusivity"] is False
    assert row["evidence"]["operator_attestation"]["kind"] == "operator_attested"


def test_unbound_operator_statement_not_applied_to_historical_rows(tmp_path):
    doc = preview(); doc["exclusivity"]["operator_attestation"] = {"kind": "operator_attested", "machine_verified": False}
    row = build_bundle(save(tmp_path / "preview.json", doc))["benchmark_data"]["runs"][0]
    assert "operator_attestation" not in row["evidence"]


@pytest.mark.parametrize("context,reason", [
    ({"framework": "sklearn"}, "EXTERNAL_FRAMEWORK_REQUIRES_EXPLICIT_ADAPTER"),
    ({"backend": {"requested": "cupy", "executed": "numpy"}}, "CONTRADICTORY_MEASURED_BACKEND"),
])
def test_no_reference_or_backend_relabeling(tmp_path, context, reason):
    doc = preview(); doc["records"][0]["context"].update(context)
    bundle = build_bundle(save(tmp_path / "preview.json", doc))
    assert bundle["benchmark_data"]["runs"] == []
    assert bundle["parse_report"]["issues"][0]["code"] == reason


def test_external_manifest_runs_existing_strict_generator(tmp_path):
    from dev.benchmarks.generate_benchmark_data import generate, validate_output, validate_against_schema
    from dev.benchmarks.frontend_data.cli import validate_semantic
    path = save(tmp_path / "preview.json", preview())
    bundle = build_bundle(path)
    manifest = build_external_manifest(path, bundle, "2026-10-25")
    result, report, inventory = generate(tmp_path, deterministic=True, manifest=manifest, strict_sources=True)
    assert result["schema_version"] == "1.2.0"
    assert len(result["runs"]) == 1
    assert validate_output(result) + validate_against_schema(result) + validate_semantic(result, manifest, True) == []
    assert result["runs"][0]["evidence"]["performance_eligible"] is False
    assert inventory["inventory_version"] == "2.0"
    assert inventory["eligible_sources"] == 0
    assert inventory["not_canonical_ready_sources"] == 1
    assert result["meta"]["generation_id"] == report["generation_id"] == inventory["generation_id"]


def _same_basename_manifests(tmp_path, *, cv=False):
    manifests = []
    for index in range(2):
        folder = tmp_path / ("campaign-%d" % index)
        folder.mkdir()
        doc = cv_preview(phase=bool(index)) if cv else preview()
        for row in doc["records"]:
            row.update(campaign_id="campaign-%d" % index, job_id="job-%d" % index)
            if not cv:
                row["summary_seconds"] = (index + 1) / 4
        path = save(folder / "preview_data.json", doc)
        manifests.append(build_external_manifest(path, build_bundle(path), "2026-10-%d" % (24 + index)))
    return manifests


def _merge_manifests(manifests):
    merged = copy.deepcopy(manifests[0])
    for manifest in manifests[1:]:
        merged["sources"].extend(copy.deepcopy(manifest["sources"]))
        merged["comparisons"].update(manifest["comparisons"])
        merged["environments"].update(manifest["environments"])
    merged["catalog_total"] = len(merged["sources"])
    return merged


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("path_form", ["absolute", "repo_relative", "normalized_relative"])
@pytest.mark.parametrize("cv", [False, True])
def test_same_basename_sources_retain_exact_batch_provenance(tmp_path, reverse, path_form, cv):
    from dev.benchmarks.generate_benchmark_data import generate, validate_output, validate_against_schema
    from dev.benchmarks.frontend_data.cli import validate_semantic

    manifests = _same_basename_manifests(tmp_path, cv=cv)
    expected = {}
    # Separate external manifests remain independently reusable, including CV
    # shards with their own alias groups and per-record environment identities.
    for manifest in manifests + manifests[:1]:
        result, _, _ = generate(tmp_path, deterministic=True, manifest=manifest, strict_sources=True)
        row = result["runs"][0]
        expected[row["benchmark_session_id"]] = row
    if reverse:
        manifests.reverse()
    merged = _merge_manifests(manifests)
    repo_root = Path(__file__).resolve().parents[2]
    for source in merged["sources"]:
        if path_form != "absolute":
            relative = Path(os.path.relpath(source["path"], repo_root))
            if path_form == "normalized_relative":
                relative = relative.parent / ".." / relative.parent.name / relative.name
            source["path"] = str(relative)
    original = copy.deepcopy(merged)
    result, report, inventory = generate(tmp_path, deterministic=True, manifest=merged, strict_sources=True)

    assert merged == original
    assert validate_output(result) + validate_against_schema(result) + validate_semantic(result, merged, True) == []
    assert len(result["runs"]) == 2
    assert report["files_parsed"] == 2
    assert inventory["parsed_registered_sources"] == 2
    for row in result["runs"]:
        # Full row equality protects source ID/path/hash/date/comparison plus
        # durations, sessions, environments, raw pointers and display names.
        assert row == expected[row["benchmark_session_id"]]
        assert row["source"]["file"] == "preview_data.json"
        assert row["metrics"]["operation_timing"]["source_file"] == "preview_data.json"
        if not cv:
            index = int(row["benchmark_session_id"].rsplit("-", 1)[1])
            assert row["metrics"]["operation_timing"]["duration_ms"] == (index + 1) * 250
    again = generate(tmp_path, deterministic=True, manifest=merged, strict_sources=True)
    assert (result, report, inventory) == again


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("duplicate", ["identical", "source_id", "path", "normalized_path", "symlink_path"])
def test_repeated_manifest_source_is_rejected_not_overwritten(tmp_path, strict, duplicate):
    from dev.benchmarks.generate_benchmark_data import generate

    merged = _merge_manifests(_same_basename_manifests(tmp_path))
    first, second = merged["sources"]
    if duplicate == "identical":
        merged["sources"][1] = copy.deepcopy(first)
    elif duplicate == "source_id":
        second["source_id"] = first["source_id"]
    else:
        path = Path(first["path"])
        if duplicate == "normalized_path":
            path = path.parent / ".." / path.parent.name / path.name
            path = Path(os.path.relpath(path, Path(__file__).resolve().parents[2]))
        elif duplicate == "symlink_path":
            alias = tmp_path / "alias.json"
            alias.symlink_to(path)
            path = alias
        second["path"] = str(path)
    reason = "source_id" if duplicate in ("identical", "source_id") else "source path"
    with pytest.raises(ValueError, match="Duplicate manifest " + reason):
        generate(tmp_path, deterministic=True, manifest=merged, strict_sources=strict)


@pytest.mark.parametrize("binding", ["ambiguous", "missing", "missing_full_path"])
def test_direct_canonical_injection_rejects_unbound_sources(tmp_path, binding):
    from dev.benchmarks.frontend_data.cli import _inject_canonical_fields

    manifests = _same_basename_manifests(tmp_path)
    merged = _merge_manifests(manifests)
    row = build_bundle(manifests[0]["sources"][0]["path"])["benchmark_data"]["runs"][0]
    row["source"].pop("original_path")
    if binding == "missing":
        row["source"]["file"] = "unregistered.json"
    elif binding == "missing_full_path":
        row["source"]["original_path"] = str(tmp_path / "unregistered" / "preview_data.json")
    reason = "Ambiguous" if binding == "ambiguous" else "Missing"
    original = copy.deepcopy(row)
    with pytest.raises(ValueError, match=reason + " manifest source binding"):
        _inject_canonical_fields([row], merged, tmp_path)
    assert row == original


@pytest.mark.parametrize("binding", ["basename", "original_basename", "absolute", "normalized_relative"])
def test_direct_canonical_injection_preserves_unique_legacy_binding(tmp_path, binding):
    from dev.benchmarks.frontend_data.cli import _inject_canonical_fields

    manifests = _same_basename_manifests(tmp_path)
    manifest = manifests[0]
    source = manifest["sources"][0]
    row = build_bundle(source["path"])["benchmark_data"]["runs"][0]
    if binding in ("basename", "original_basename"):
        row["source"].pop("original_path")
        if binding == "original_basename":
            source["original_path"] = "results/original-preview.json"
            row["source"]["file"] = "original-preview.json"
    else:
        manifest = _merge_manifests(manifests)
        if binding == "normalized_relative":
            path = Path(source["path"])
            path = path.parent / ".." / path.parent.name / path.name
            row["source"]["original_path"] = os.path.relpath(path, Path(__file__).resolve().parents[2])
    _inject_canonical_fields([row], manifest, tmp_path)
    assert row["source"]["source_id"] == source["source_id"]
    assert row["source"]["original_path"] == source["original_path"]
    assert row["source"]["sha256"] == source["sha256"]
    assert row["source"]["date"] == source["source_date"]
    assert row["comparison_id"] == source["comparison_id"]


def test_environment_and_thread_profiles_remain_separate(tmp_path):
    doc = preview(); row = copy.deepcopy(doc["records"][0]); row.update(record_id="other", job_id="other-job")
    row["cpu_thread_profile"]["requested_cap"] = 8; doc["records"].append(row)
    rows = build_bundle(save(tmp_path / "preview.json", doc))["benchmark_data"]["runs"]
    assert rows[0]["env_id"] != rows[1]["env_id"]


def test_tamper_or_eligibility_promotion_rejected(tmp_path):
    path = save(tmp_path / "preview.json", preview())
    doc = json.loads(path.read_text()); doc["records"][0]["summary_seconds"] = 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="digest mismatch"): build_bundle(path)
    doc = preview(); doc["performance_eligible"] = True
    with pytest.raises(ValueError, match="cannot assert"): build_bundle(save(path, doc))


def test_cli_refuses_existing_output(tmp_path):
    path = save(tmp_path / "preview.json", preview())
    out = tmp_path / "bundle"
    assert main(["--preview", str(path), "--out-dir", str(out)]) == 0
    assert (out / "benchmark_data.json").exists()
    with pytest.raises(SystemExit): main(["--preview", str(path), "--out-dir", str(out)])


def cv_preview(*, phase=False):
    doc = preview()
    template = doc["records"][0]
    template.update(source_schema="statgpu-scoped-cv-candidate-v2", unit="seconds")
    template["context"].update(case={"case_id": "cv-ridge", "model_id": "RidgeCV"},
                               model_id="RidgeCV", instrumented=phase, repeats=3)
    samples = [.1, .3, .2]
    root = "/results/0/case/runs/0" if phase else "/results/0/clean_public_fit"
    if phase:
        template["context"].update(framework="statgpu", timing_kind="cv_diagnostic_stage", timing_phase="cv_evaluation")
        template["timing_scope"] = "profiling overhead included; observed CV and final refit"
        template.pop("sample_count")
        template.update(raw_pointer=root + "/timing/cv_evaluation_ms", unit="milliseconds", raw_value=200,
                        summary_seconds=.2, statistic="recorded_scalar:cv_evaluation_ms")
    else:
        template["context"].update(timing_kind="clean_public_fit", timing_phase="total_fit")
        template.update(raw_pointer=root + "/raw_seconds", raw_value=samples, summary_seconds=.2)
    doc["records"] = [template]
    if not phase:
        summary = copy.deepcopy(template); summary.pop("sample_count")
        summary.update(record_id="median", raw_pointer=root + "/timing/median_seconds", raw_value=.2)
        doc["records"].append(summary)
    for index, value in enumerate(samples):
        row = copy.deepcopy(template); row.pop("sample_count", None)
        row.update(record_id="repeat-%d" % index, unit="milliseconds", raw_value=value * 1000,
                   summary_seconds=value, raw_pointer=root + "/repeat_samples/%d/" % index + ("cv_evaluation_ms" if phase else "total_fit_ms"))
        doc["records"].append(row)
    return doc


@pytest.mark.parametrize("case", [
    {"case_id": "cv-ridge", "model_id": "RidgeCV"},
    {"model_id": "RidgeCV"},
    {"model_name": "RidgeCV"},
])
def test_structured_case_does_not_mask_nested_or_outer_identity(tmp_path, case):
    doc = preview(); doc["records"][0]["context"].update(case=case)
    row = build_bundle(save(tmp_path / "preview.json", doc))["benchmark_data"]["runs"][0]
    assert row["model_id"].endswith(":" + case.get("case_id", "RidgeCV"))


@pytest.mark.parametrize("extra", [{"case_id": "different"}, {"model_id": "different"},
                                  {"identity_conflicts": ["model_id"]}])
def test_conflicting_nested_identity_fails_closed(tmp_path, extra):
    doc = cv_preview(); doc["records"] = doc["records"][:1]
    doc["records"][0]["context"].update(extra)
    bundle = build_bundle(save(tmp_path / "preview.json", doc))
    assert not bundle["benchmark_data"]["runs"]
    assert bundle["parse_report"]["issues"][0]["code"] == "CONFLICTING_CASE_IDENTITY"


@pytest.mark.parametrize("phase", [False, True])
def test_cv_cross_parent_aliases_keep_one_observation_aggregate(tmp_path, phase):
    doc = cv_preview(phase=phase)
    bundle = build_bundle(save(tmp_path / "preview.json", doc))
    assert len(bundle["benchmark_data"]["runs"]) == 1
    row = bundle["benchmark_data"]["runs"][0]
    assert row["metrics"]["operation_timing"]["duration_ms"] == 200
    assert set(row["metrics"]) == {"operation_timing"}
    assert row["parameters"]["original_context"]["instrumented"] is phase
    assert len(bundle["parse_report"]["issues"]) == len(doc["records"]) - 1
    assert {issue["code"] for issue in bundle["parse_report"]["issues"]} == {"ALIAS_OF_RETAINED_OBSERVATIONS"}
    assert len(json.loads((tmp_path / "preview.json").read_text())["records"]) == len(doc["records"])


@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_inconsistent_cv_alias_values_never_select_a_preferred_truth(tmp_path, phase, index):
    doc = cv_preview(phase=phase); doc["records"][index]["summary_seconds"] = 9
    bundle = build_bundle(save(tmp_path / "preview.json", doc))
    assert not bundle["benchmark_data"]["runs"]
    assert {issue["code"] for issue in bundle["parse_report"]["issues"]} == {"INCONSISTENT_TIMING_ALIASES"}


def test_inconsistent_generic_summary_is_not_silently_discarded(tmp_path):
    doc = preview(); summary = copy.deepcopy(doc["records"][0]); summary.pop("sample_count")
    summary.update(record_id="summary", raw_pointer="/results/0/timing/median_seconds", summary_seconds=.9)
    doc["records"].append(summary)
    result = build_bundle(save(tmp_path / "preview.json", doc))
    assert not result["benchmark_data"]["runs"]
    assert {issue["code"] for issue in result["parse_report"]["issues"]} == {"INCONSISTENT_TIMING_ALIASES"}


def test_distinct_cv_phases_and_clean_timing_are_not_merged(tmp_path):
    doc = cv_preview(); other = cv_preview(phase=True)
    for row in other["records"]: row["record_id"] += "-phase"
    doc["records"].extend(other["records"])
    for phase in ("final_refit", "total_fit"):
        for row in copy.deepcopy(other["records"]):
            row["record_id"] += "-" + phase
            row["raw_pointer"] = row["raw_pointer"].replace("cv_evaluation_ms", phase + "_ms")
            row["context"]["timing_phase"] = phase
            doc["records"].append(row)
    rows = build_bundle(save(tmp_path / "preview.json", doc))["benchmark_data"]["runs"]
    assert len(rows) == 4
    assert len({(row["parameters"]["original_context"]["timing_kind"], row["parameters"]["original_context"]["timing_phase"]) for row in rows}) == 4
    assert all(row["evidence"]["performance_eligible"] is False for row in rows)


@pytest.mark.parametrize("change", [{"scope": "other-boundary"}, {"instrumented": True},
                                   {"framework": "sklearn"}, {"backend": "torch"}])
def test_alias_group_cannot_cross_scope_framework_or_instrumentation(tmp_path, change):
    doc = preview(); summary = copy.deepcopy(doc["records"][0]); summary.pop("sample_count")
    summary.update(record_id="summary", raw_pointer="/results/0/timing/median_seconds", summary_seconds=8)
    summary["context"].update(change)
    if "scope" in change: summary["timing_scope"] = change["scope"]
    doc["records"].append(summary)
    result = build_bundle(save(tmp_path / "preview.json", doc))
    if change.get("framework") == "sklearn":
        assert len(result["benchmark_data"]["runs"]) == 1
        assert result["parse_report"]["issues"][0]["code"] == "EXTERNAL_FRAMEWORK_REQUIRES_EXPLICIT_ADAPTER"
    else:
        assert len(result["benchmark_data"]["runs"]) == 2
        assert result["parse_report"]["issues"] == []


def test_external_cv_framework_retains_explicit_reason_and_never_inherits_torch(tmp_path):
    doc = cv_preview(phase=True)
    for row in doc["records"]:
        row["context"].update(framework="sklearn", backend=None, requested_backend="torch")
    result = build_bundle(save(tmp_path / "preview.json", doc))
    assert not result["benchmark_data"]["runs"]
    assert {issue["code"] for issue in result["parse_report"]["issues"]} == {"EXTERNAL_FRAMEWORK_REQUIRES_EXPLICIT_ADAPTER"}


@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("duration", [0, -1, None])
def test_invalid_cv_canonical_duration_does_not_promote_its_aliases(tmp_path, phase, duration):
    doc = cv_preview(phase=phase); doc["records"][0]["summary_seconds"] = duration
    result = build_bundle(save(tmp_path / "preview.json", doc))
    assert not result["benchmark_data"]["runs"]
    assert "UNVERIFIABLE_TIMING_ALIASES" in {issue["code"] for issue in result["parse_report"]["issues"]}


def test_cv_repeat_only_group_is_not_an_invented_aggregate(tmp_path):
    doc = cv_preview(phase=True); doc["records"].pop(0)
    result = build_bundle(save(tmp_path / "preview.json", doc))
    assert not result["benchmark_data"]["runs"]
    assert {issue["code"] for issue in result["parse_report"]["issues"]} == {"UNVERIFIABLE_TIMING_ALIASES"}


def test_unqualified_cv_canonical_does_not_suppress_then_promote_aliases(tmp_path):
    doc = cv_preview(); doc["records"][0]["validation_status"] = "withheld"
    result = build_bundle(save(tmp_path / "preview.json", doc))
    assert not result["benchmark_data"]["runs"]
    assert "UNQUALIFIED_TIMING_ALIAS_SOURCE" in {issue["code"] for issue in result["parse_report"]["issues"]}


def test_cv_phase_label_is_explicit_without_rewriting_recorded_protocol(tmp_path):
    doc = cv_preview(phase=True)
    row = build_bundle(save(tmp_path / "preview.json", doc))["benchmark_data"]["runs"][0]
    assert row["metrics"]["operation_timing"]["scope"] == doc["records"][0]["timing_scope"] + "; recorded phase: cv_evaluation"
    assert row["parameters"]["original_context"]["timing_phase"] == "cv_evaluation"


@pytest.mark.parametrize("phase", [False, True])
def test_legacy_cv_preview_requires_reexport_instead_of_duplicate_aliases(tmp_path, phase):
    doc = cv_preview(phase=phase)
    for row in doc["records"]:
        row.pop("source_schema"); row["suite_or_lane"] = "six-family-cv"
    result = build_bundle(save(tmp_path / "preview.json", doc))
    assert not result["benchmark_data"]["runs"]
    assert {issue["code"] for issue in result["parse_report"]["issues"]} == {"CV_ALIAS_BINDING_REQUIRES_REEXPORT"}


def test_legacy_unrelated_timing_remains_compatible(tmp_path):
    result = build_bundle(save(tmp_path / "preview.json", preview()))
    assert len(result["benchmark_data"]["runs"]) == 1


@pytest.mark.parametrize("change", [
    lambda row: row["context"]["case"].update(case_id="different-case"),
    lambda row: row["context"]["problem"].update(n=40),
    lambda row: row["context"]["problem"].update(p=4),
])
def test_cv_alias_binding_includes_resolved_case_and_dimensions(tmp_path, change):
    doc = cv_preview(); change(doc["records"][1])
    result = build_bundle(save(tmp_path / "preview.json", doc))
    rows = result["benchmark_data"]["runs"]
    assert len(rows) == 2
    assert rows[0]["case_id"] != rows[1]["case_id"]
    assert not any(issue["message"].startswith("median:") for issue in result["parse_report"]["issues"])
