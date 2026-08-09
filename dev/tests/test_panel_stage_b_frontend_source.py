from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SOURCE_PATH = (
    REPO_ROOT
    / "results"
    / "benchmark_frontend_sources"
    / "panel_stage_b_pr122_p100_20260809.json"
)
RAW_SOURCE_PATH = (
    REPO_ROOT
    / "results"
    / "pr122_p100"
    / "panel_stage_b_gpu_validation_a57efcea.json"
)
FOCUSED_SOURCE_PATH = (
    REPO_ROOT
    / "results"
    / "pr122_p100"
    / "panel_stage_b_disconnected_fe_gpu_validation_a57efcea.json"
)
EXPECTED_SHA256 = "a6e47b9dec9c35040d2715b573aa2a93b084d83c574078bdb430861f0a9aa9f9"
RAW_EXPECTED_BLOB_SHA = "254b64776bff4e3b2b642bb4a2ae1eea25f4751c"
FOCUSED_EXPECTED_BLOB_SHA = "3bda0b2040479ba8201e2722eb990ba086c3f3b9"
SOURCE_ID = "panel-stage-b-pr122-20260809-a6e47b9dec9c"
MEASUREMENT_SHA = "a57efcea29b0e87ecb89865c5a6902d5773812c6"
ARTIFACT_COMMIT = "72b3279d2028e8ec2af30e138e123aceb611ae8c"
ENV_ID = "remote-p100-pr122-20260809"


def _git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def test_pr122_physical_source_contract_and_hash() -> None:
    from dev.benchmarks.frontend_data.canonical import source_sha256

    data = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    focused = json.loads(FOCUSED_SOURCE_PATH.read_text(encoding="utf-8"))

    assert source_sha256(SOURCE_PATH) == EXPECTED_SHA256
    assert _git_blob_sha(RAW_SOURCE_PATH) == RAW_EXPECTED_BLOB_SHA
    assert _git_blob_sha(FOCUSED_SOURCE_PATH) == FOCUSED_EXPECTED_BLOB_SHA
    assert data["source_schema_version"] == "1.0"
    assert data["source_date"] == "2026-08-09"
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
    assert data["compatibility_matrix"]["cupy"]["disconnected_fe"] == "pass"
    assert data["compatibility_matrix"]["torch"]["disconnected_fe"] == "pass"

    raw = data["raw_artifact"]
    assert raw["path"] == str(RAW_SOURCE_PATH.relative_to(REPO_ROOT))
    assert raw["repository_commit"] == ARTIFACT_COMMIT
    assert raw["git_blob_sha"] == RAW_EXPECTED_BLOB_SHA

    focused_meta = data["focused_artifact"]
    assert focused_meta["path"] == str(FOCUSED_SOURCE_PATH.relative_to(REPO_ROOT))
    assert focused_meta["repository_commit"] == ARTIFACT_COMMIT
    assert focused_meta["git_blob_sha"] == FOCUSED_EXPECTED_BLOB_SHA
    assert focused_meta["status"] == "success"

    assert focused["git_sha"] == MEASUREMENT_SHA
    assert focused["working_tree_clean"] is True
    assert focused["status"] == "success"
    assert focused["reference"]["legacy_df_resid"] == 0
    assert focused["reference"]["df_resid"] == 1
    assert focused["reference"]["diagnostic_df"]["effect_rank"] == 7
    assert focused["reference"]["diagnostic_df"]["incidence_components"] == 3
    assert focused["backend_results"]["cupy"]["executed_backend"] == "cupy"
    assert focused["backend_results"]["torch"]["executed_backend"] == "torch"
    assert focused["backend_results"]["cupy"]["status"] == "success"
    assert focused["backend_results"]["torch"]["status"] == "success"
    assert focused["backend_results"]["torch"]["differences_vs_numpy"]["conf_int"] < 1e-12
    assert data["environment"]["gpu"] == "Tesla P100-SXM2-16GB"
    assert data["environment"]["packages"]["cupy"] is None


def test_pr122_parser_emits_validation_only_frontend_runs() -> None:
    from dev.benchmarks.frontend_data.parsers import (
        parse_panel_stage_b_physical_validation,
    )

    runs, models, warnings = parse_panel_stage_b_physical_validation(
        SOURCE_PATH, ENV_ID
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
    assert entry["path"] == str(SOURCE_PATH.relative_to(REPO_ROOT))
    assert entry["original_path"] == str(RAW_SOURCE_PATH.relative_to(REPO_ROOT))
    assert entry["sha256"] == EXPECTED_SHA256
    assert entry["parser"] == "panel_stage_b_physical_validation"
    assert entry["parser_version"] == "1.0"
    assert entry["env_id"] == ENV_ID
    assert entry["source_date"] == "2026-08-09"
    assert entry["measurement_git_sha"] == MEASUREMENT_SHA
    assert entry["raw_git_sha"] == MEASUREMENT_SHA
    assert RAW_EXPECTED_BLOB_SHA in entry["provenance_note"]
    assert FOCUSED_EXPECTED_BLOB_SHA in entry["provenance_note"]
