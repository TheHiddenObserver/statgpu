from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SOURCE_PATH = (
    REPO_ROOT
    / "results"
    / "benchmark_frontend_sources"
    / "panel_stage_b_pr122_p100_20260808.json"
)
RAW_SOURCE_PATH = (
    REPO_ROOT
    / "results"
    / "pr122_p100"
    / "panel_stage_b_gpu_validation_faa95ce7.json"
)
EXPECTED_SHA256 = "b8caffa6f915facfb74965b6834b06d8aa6480cb0e3822640261a77ddd1ec9ba"
RAW_EXPECTED_SHA256 = "c1ba014a3b9bb0d32cbc0ca3d844ccfe767e7149189efb9ba2969f5bc1b94b31"
SOURCE_ID = "panel-stage-b-pr122-20260808-b8caffa6f915"
MEASUREMENT_SHA = "faa95ce7fb5cb204088957fbda5544c20a06fbfc"


def test_pr122_physical_source_contract_and_hash() -> None:
    from dev.benchmarks.frontend_data.canonical import source_sha256

    data = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    assert source_sha256(SOURCE_PATH) == EXPECTED_SHA256
    assert source_sha256(RAW_SOURCE_PATH) == RAW_EXPECTED_SHA256
    assert data["source_schema_version"] == "1.0"
    assert data["source_date"] == "2026-08-08"
    assert data["git_sha"] == MEASUREMENT_SHA
    assert data["working_tree_clean"] is True
    assert data["status"] == "success"
    assert data["schema_status"] == "ok"
    assert data["validation_tier"] == "remote-full"
    assert data["protocol"]["timing_collected"] is False
    assert data["backend_times"] == {"numpy": None, "cupy": None, "torch": None}
    assert data["compatibility_matrix"]["cupy"]["model_cases"] == "17/17"
    assert data["compatibility_matrix"]["torch"]["model_cases"] == "17/17"
    assert data["compatibility_matrix"]["cupy"]["diagnostics"] == "4/4"
    assert data["compatibility_matrix"]["torch"]["diagnostics"] == "4/4"
    assert data["compatibility_matrix"]["cupy"]["cpu_fallback"] is False
    assert data["compatibility_matrix"]["torch"]["cpu_fallback"] is False
    assert data["raw_artifact"]["path"] == str(RAW_SOURCE_PATH.relative_to(REPO_ROOT))
    assert data["raw_artifact"]["sha256"] == RAW_EXPECTED_SHA256
    assert data["environment"]["gpu"] == "Tesla P100-SXM2-16GB"
    assert data["environment"]["packages"]["cupy"] is None


def test_pr122_parser_emits_validation_only_frontend_runs() -> None:
    from dev.benchmarks.frontend_data.parsers import (
        parse_panel_stage_b_physical_validation,
    )

    runs, models, warnings = parse_panel_stage_b_physical_validation(
        SOURCE_PATH, "remote-p100-pr122-20260808"
    )

    assert warnings == []
    assert len(runs) == 42
    assert len(models) == 6
    assert {run["backend"] for run in runs} == {"cupy", "torch"}
    assert {model["model_id"] for model in models} == {
        "PooledOLS",
        "BetweenOLS",
        "FirstDifferenceOLS",
        "PanelOLS",
        "RandomEffects",
        "FamaMacBeth",
    }
    assert all("timing" not in run["metrics"] for run in runs)
    assert all("speedup" not in run["metrics"] for run in runs)
    assert all(run["metrics"]["validation"]["status"] == "pass" for run in runs)
    assert all(
        run["parameters"]["measurement_git_sha"] == MEASUREMENT_SHA for run in runs
    )
    assert all(run["parameters"]["working_tree_clean"] is True for run in runs)

    hausman = [run for run in runs if run["parameters"].get("diagnostic") == "hausman"]
    assert len(hausman) == 8
    assert all(run["parameters"]["applicable"] is False for run in hausman)
    assert all("inference" not in run["metrics"] for run in hausman)
    assert {run["parameters"]["parameterization"] for run in hausman} == {
        "standard",
        "re-explicit-constant",
    }
    assert {run["variant"] for run in hausman} == {
        "hausman-balanced",
        "hausman-unbalanced",
        "hausman-re-explicit-constant-balanced",
        "hausman-re-explicit-constant-unbalanced",
    }

    estimator_runs = [run for run in runs if run not in hausman]
    assert len(estimator_runs) == 34
    assert all(run["metrics"]["inference"]["ok"] is True for run in estimator_runs)

    explicit_re = [
        run
        for run in estimator_runs
        if run["model_id"] == "RandomEffects"
        and run["variant"].startswith("explicit-constant-")
    ]
    assert len(explicit_re) == 4
    assert {run["scale"]["n_features"] for run in explicit_re} == {3}


def test_pr122_parser_is_registered_in_manifest() -> None:
    from dev.benchmarks.frontend_data.parsers import (
        parse_panel_stage_b_physical_validation,
    )
    from dev.benchmarks.frontend_data.registry import PARSER_FUNCTIONS, load_manifest

    assert (
        PARSER_FUNCTIONS["panel_stage_b_physical_validation"]
        is parse_panel_stage_b_physical_validation
    )
    manifest = load_manifest(REPO_ROOT)
    assert manifest is not None
    entry = next(source for source in manifest["sources"] if source["source_id"] == SOURCE_ID)
    assert entry["path"] == (
        "results/benchmark_frontend_sources/panel_stage_b_pr122_p100_20260808.json"
    )
    assert entry["original_path"] == str(RAW_SOURCE_PATH.relative_to(REPO_ROOT))
    assert entry["sha256"] == EXPECTED_SHA256
    assert entry["parser"] == "panel_stage_b_physical_validation"
    assert entry["parser_version"] == "1.0"
    assert entry["measurement_git_sha"] == MEASUREMENT_SHA
    assert entry["raw_git_sha"] == MEASUREMENT_SHA
    assert RAW_EXPECTED_SHA256 in entry["provenance_note"]
