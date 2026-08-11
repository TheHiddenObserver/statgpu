from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_canonical_inventory_v2_publishes_audited_catalog_snapshot() -> None:
    from dev.benchmarks.frontend_data.registry import load_manifest
    from dev.benchmarks.generate_benchmark_data import generate

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None
    output, report, inventory = generate(
        REPO_ROOT / "results",
        deterministic=True,
        manifest=manifest,
        strict_sources=True,
    )

    assert inventory["inventory_version"] == "2.0"
    assert inventory["generation_id"] == output["meta"]["generation_id"]
    assert inventory["generation_id"] == report["generation_id"]
    assert inventory["catalog_digest"]
    assert inventory["catalog_policy_digest"]
    assert inventory["coverage_matrix_digest"]
    assert inventory["discovered_json_artifacts"] == len(inventory["catalog_entries"])
    assert inventory["classified_candidate_sources"] == len(inventory["catalog_entries"])
    assert inventory["unclassified_artifacts"] == 0

    required_entry_fields = {
        "path",
        "artifact_type",
        "source_date",
        "classification",
        "canonical_eligible",
        "registered",
        "source_id",
        "parser",
        "parser_version",
        "provenance_status",
        "timing_protocol_status",
        "statistical_alignment_status",
        "reason",
        "superseded_by",
        "issue",
        "rule_id",
    }
    assert inventory["catalog_entries"]
    assert all(required_entry_fields <= set(entry) for entry in inventory["catalog_entries"])

    registered = [
        entry for entry in inventory["catalog_entries"]
        if entry["classification"] == "registered_canonical"
    ]
    assert len(registered) == inventory["registered_sources"] == 15
    assert all(entry["registered"] for entry in registered)


def test_serialized_inventory_omits_legacy_aliases() -> None:
    from dev.benchmarks.frontend_data.registry import load_manifest
    from dev.benchmarks.generate_benchmark_data import generate

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None
    _, _, inventory = generate(
        REPO_ROOT / "results",
        deterministic=True,
        manifest=manifest,
        strict_sources=True,
    )
    serialized = json.loads(json.dumps(inventory))

    assert "catalog_total" not in serialized
    assert "eligible_total" not in serialized
    assert "available_sources" not in serialized
    assert "parsed_sources" not in serialized
    assert serialized["available_registered_sources"] == 15
    assert serialized["parsed_registered_sources"] == 15

    # Old Python callers can still read the two non-semantic aliases without
    # reintroducing those keys into the published JSON contract.
    assert inventory["available_sources"] == 15
    assert inventory["parsed_sources"] == 15


def test_modified_repository_manifest_retains_legacy_inventory_contract() -> None:
    from dev.benchmarks.frontend_data.registry import load_manifest
    from dev.benchmarks.generate_benchmark_data import generate

    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None
    modified = deepcopy(manifest)
    modified["environments"]["remote-p100"]["label"] = "Synthetic environment"

    output, _, inventory = generate(
        REPO_ROOT / "results",
        deterministic=True,
        manifest=modified,
        strict_sources=True,
    )

    assert inventory["inventory_version"] == "1.0"
    assert "catalog_entries" not in inventory
    assert output["environments"][0]["label"] == "Synthetic environment"
