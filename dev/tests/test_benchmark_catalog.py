from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


@pytest.fixture(scope="module")
def manifest():
    from dev.benchmarks.frontend_data.registry import load_manifest

    loaded = load_manifest(repo_root)
    assert loaded is not None
    return loaded


@pytest.fixture(scope="module")
def catalog():
    from dev.benchmarks.frontend_data.catalog import load_source_catalog

    return load_source_catalog(repo_root)


@pytest.fixture(scope="module")
def coverage_matrix():
    from dev.benchmarks.frontend_data.catalog import load_coverage_matrix

    return load_coverage_matrix(repo_root)


@pytest.fixture(scope="module")
def entries(catalog, manifest):
    from dev.benchmarks.frontend_data.catalog import discover_catalog_entries

    return discover_catalog_entries(repo_root, catalog, manifest)


def test_catalog_classifies_every_discovered_artifact(catalog, manifest, entries):
    from dev.benchmarks.frontend_data.catalog import validate_source_catalog

    assert validate_source_catalog(catalog, entries, manifest) == []
    assert entries
    assert all(entry["classification"] != "unclassified" for entry in entries)
    assert [entry["path"] for entry in entries] == sorted(entry["path"] for entry in entries)


def test_every_registered_source_is_catalogued_from_manifest(manifest, entries):
    by_path = {entry["path"]: entry for entry in entries}
    for source in manifest["sources"]:
        entry = by_path[source["path"]]
        assert entry["classification"] == "registered_canonical"
        assert entry["canonical_eligible"] is True
        assert entry["registered"] is True
        assert entry["source_id"] == source["source_id"]
        assert entry["parser"] == source["parser"]
        assert entry["parser_version"] == source["parser_version"]


def test_catalog_retains_distinct_noncanonical_dispositions(entries):
    classifications = {entry["classification"] for entry in entries}
    assert "registered_canonical" in classifications
    assert "not_canonical_ready" in classifications
    assert "historical_or_excluded" in classifications
    assert "superseded_or_duplicate" in classifications

    distribution = next(
        entry for entry in entries
        if entry["path"] == "results/distribution_bench_2026-06-21.json"
    )
    assert distribution["classification"] == "not_canonical_ready"
    assert distribution["issue"] == "#101"

    focused = next(
        entry for entry in entries
        if entry["path"] == "results/pr116_p100/focused_validation.json"
    )
    assert focused["classification"] == "not_canonical_ready"
    assert focused["provenance_status"] == "validation_evidence"
    assert focused["issue"] == "#112"

    panel_raw = next(
        entry for entry in entries
        if entry["path"]
        == "results/pr122_p100/panel_stage_b_gpu_validation_faa95ce7.json"
    )
    assert panel_raw["classification"] == "not_canonical_ready"
    assert panel_raw["provenance_status"] == "validation_evidence"
    assert panel_raw["timing_protocol_status"] == "not_applicable"
    assert panel_raw["statistical_alignment_status"] == "accepted"
    assert panel_raw["issue"] == "#93"

    stage_c_validation = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_gpu_validation_ec511f53.json"
    )
    stage_c_performance = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_performance_ec511f53.json"
    )
    assert stage_c_validation["classification"] == "registered_canonical"
    assert stage_c_validation["source_id"] == "panel-stage-c-validation-pr126-20260811-af2227efe3cd"
    assert stage_c_performance["classification"] == "registered_canonical"
    assert stage_c_performance["source_id"] == "panel-stage-c-performance-pr126-20260811-409974070022"


def test_coverage_matrix_is_referentially_complete(coverage_matrix, manifest):
    from dev.benchmarks.frontend_data.catalog import validate_coverage_matrix

    assert validate_coverage_matrix(coverage_matrix, manifest) == []
    rows = {row["capability_id"]: row for row in coverage_matrix["capabilities"]}
    assert rows["ridge-cv"]["status"] == "canonical_current"
    assert rows["ridge-cv"]["source_ids"] == ["cv-benchmark-20260807-1347184c988d"]
    assert rows["logistic-regression-cv"]["status"] == "canonical_current"
    assert rows["logistic-regression-cv"]["source_ids"] == [
        "cv-benchmark-20260807-1347184c988d",
        "cv-benchmark-pr116-20260807-bd8d512adced",
    ]
    assert rows["panel-estimation"]["source_ids"] == [
        "new-modules-20260624-bcbdb676223b",
        "panel-stage-b-pr122-20260809-2056f836bfe2",
        "panel-stage-c-validation-pr126-20260811-af2227efe3cd",
        "panel-stage-c-performance-pr126-20260811-409974070022",
        "panel-stage-c-rank-policy-validation-pr126-20260811-c67ada7ec59f",
        "panel-stage-c-rank-policy-performance-pr126-20260811-f27bef0b7c55",
        "panel-stage-c-rank-df-validation-pr126-20260812-0b4eb5810ad0",
        "panel-stage-c-rank-df-performance-pr126-20260812-09337cc62c94",
        "panel-stage-c-identifiability-validation-pr126-20260812-2d929bccf1c7",
        "panel-stage-c-identifiability-performance-pr126-20260812-2238002d491f",
        "panel-stage-c-final-validation-pr126-20260813-62fbf89e58fb",
        "panel-stage-c-final-performance-pr126-20260813-980ffe9bd392",
    ]
    assert rows["distribution-api"]["issue"] == "#101"
    assert rows["feature-selection-knockoff"]["issue"] == "#103"
    assert rows["penalized-coxph"]["issue"] == "#107"


def test_inventory_v2_reconciles_literal_counts(
    catalog, coverage_matrix, manifest, entries
):
    from dev.benchmarks.frontend_data.catalog import build_inventory_v2

    inventory = build_inventory_v2(
        entries,
        manifest,
        available_registered_sources=len(manifest["sources"]),
        parsed_registered_sources=len(manifest["sources"]),
        catalog=catalog,
        coverage_matrix=coverage_matrix,
    )

    assert inventory["inventory_version"] == "2.0"
    assert inventory["discovered_json_artifacts"] == len(entries)
    assert inventory["classified_candidate_sources"] == len(entries)
    assert inventory["registered_sources"] == len(manifest["sources"]) == 21
    assert inventory["available_registered_sources"] == 21
    assert inventory["parsed_registered_sources"] == 21
    assert inventory["eligible_sources"] == (
        inventory["registered_sources"]
        + inventory["eligible_unregistered_sources"]
    )
    assert inventory["unclassified_artifacts"] == 0
    assert inventory["catalog_digest"]
    assert inventory["coverage_matrix_digest"]


def test_eligible_unregistered_requires_owner(catalog, manifest, entries):
    from dev.benchmarks.frontend_data.catalog import validate_source_catalog

    synthetic = copy.deepcopy(entries)
    synthetic.append(
        {
            "path": "results/synthetic_current.json",
            "artifact_type": "json",
            "source_date": "2026-08-06",
            "classification": "eligible_unregistered",
            "canonical_eligible": True,
            "registered": False,
            "source_id": None,
            "parser": "synthetic",
            "parser_version": "1.0",
            "provenance_status": "complete",
            "timing_protocol_status": "accepted",
            "statistical_alignment_status": "accepted",
            "reason": "Synthetic validation fixture.",
            "superseded_by": None,
            "issue": None,
            "rule_id": "synthetic",
        }
    )
    errors = validate_source_catalog(catalog, synthetic, manifest)
    assert "eligible unregistered source lacks issue: results/synthetic_current.json" in errors


def test_coverage_rejects_unknown_source_id(coverage_matrix, manifest):
    from dev.benchmarks.frontend_data.catalog import validate_coverage_matrix

    broken = copy.deepcopy(coverage_matrix)
    broken["capabilities"][0]["source_ids"] = ["missing-source-id"]
    errors = validate_coverage_matrix(broken, manifest)
    assert any("references unknown source_id" in error for error in errors)


def test_new_unmatched_artifact_is_not_silently_eligible(tmp_path, catalog, manifest):
    from dev.benchmarks.frontend_data.catalog import discover_catalog_entries

    results = tmp_path / "results"
    results.mkdir()
    (results / "new_method_2026-08-06.json").write_text("{}", encoding="utf-8")

    synthetic_catalog = copy.deepcopy(catalog)
    synthetic_catalog["scan_roots"] = [{"path": "results", "recursive": True, "required": True}]
    synthetic_manifest = {**manifest, "sources": []}
    discovered = discover_catalog_entries(tmp_path, synthetic_catalog, synthetic_manifest)
    assert len(discovered) == 1
    assert discovered[0]["classification"] == "not_canonical_ready"
    assert discovered[0]["canonical_eligible"] is False
    assert discovered[0]["issue"] == "#100"


def test_stage_c_superseded_artifacts_remain_historical(entries):
    old_validation = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_gpu_validation_9c0b3050.json"
    )
    old_performance = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_performance_9c0b3050.json"
    )
    assert old_validation["classification"] == "historical_or_excluded"
    assert old_validation["registered"] is False
    assert old_performance["classification"] == "historical_or_excluded"
    assert old_performance["registered"] is False

    prior_validation = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_gpu_validation_c151550a.json"
    )
    prior_performance = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_performance_c151550a.json"
    )
    assert prior_validation["classification"] == "historical_or_excluded"
    assert prior_validation["registered"] is False
    assert prior_performance["classification"] == "historical_or_excluded"
    assert prior_performance["registered"] is False

    aad_validation = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_gpu_validation_aad53587.json"
    )
    aad_performance = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_performance_aad53587.json"
    )
    assert aad_validation["classification"] == "historical_or_excluded"
    assert aad_validation["registered"] is False
    assert aad_performance["classification"] == "historical_or_excluded"
    assert aad_performance["registered"] is False

    five_validation = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_gpu_validation_5ed763be.json"
    )
    five_performance = next(
        entry for entry in entries
        if entry["path"] == "results/pr126_p100/panel_stage_c_performance_5ed763be.json"
    )
    assert five_validation["classification"] == "historical_or_excluded"
    assert five_validation["registered"] is False
    assert five_performance["classification"] == "historical_or_excluded"
    assert five_performance["registered"] is False
