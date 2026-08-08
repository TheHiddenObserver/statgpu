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
EXPECTED_SHA256 = "882892c6e3077fe3b9f6084212647311da795fd05d1ed9f12ec53da1e05d0d4d"
SOURCE_ID = "panel-stage-b-pr122-20260808-882892c6e307"


def test_pr122_physical_source_contract_and_hash() -> None:
    from dev.benchmarks.frontend_data.canonical import source_sha256

    data = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    assert source_sha256(SOURCE_PATH) == EXPECTED_SHA256
    assert data["source_schema_version"] == "1.0"
    assert data["source_date"] == "2026-08-08"
    assert data["git_sha"] == "636988751bcbfad3442d24d3073cdfcd2b3ac637"
    assert data["working_tree_clean"] is True
    assert data["status"] == "success"
    assert data["schema_status"] == "ok"
    assert data["validation_tier"] == "remote-full"
    assert data["protocol"]["timing_collected"] is False
    assert data["backend_times"] == {"numpy": None, "cupy": None, "torch": None}
    assert data["compatibility_matrix"]["cupy"]["model_cases"] == "15/15"
    assert data["compatibility_matrix"]["torch"]["model_cases"] == "15/15"
    assert data["compatibility_matrix"]["cupy"]["cpu_fallback"] is False
    assert data["compatibility_matrix"]["torch"]["cpu_fallback"] is False


def test_pr122_parser_emits_validation_only_frontend_runs() -> None:
    from dev.benchmarks.frontend_data.parsers import (
        parse_panel_stage_b_physical_validation,
    )

    runs, models, warnings = parse_panel_stage_b_physical_validation(
        SOURCE_PATH, "remote-p100-pr122-20260808"
    )

    assert warnings == []
    assert len(runs) == 34
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
        run["parameters"]["measurement_git_sha"]
        == "636988751bcbfad3442d24d3073cdfcd2b3ac637"
        for run in runs
    )
    assert all(run["parameters"]["working_tree_clean"] is True for run in runs)

    hausman = [run for run in runs if run["parameters"].get("diagnostic") == "hausman"]
    assert len(hausman) == 4
    assert all(run["parameters"]["applicable"] is False for run in hausman)
    assert all("inference" not in run["metrics"] for run in hausman)

    estimator_runs = [run for run in runs if run not in hausman]
    assert len(estimator_runs) == 30
    assert all(run["metrics"]["inference"]["ok"] is True for run in estimator_runs)


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
    assert entry["sha256"] == EXPECTED_SHA256
    assert entry["parser"] == "panel_stage_b_physical_validation"
    assert entry["parser_version"] == "1.0"
    assert entry["measurement_git_sha"] == "636988751bcbfad3442d24d3073cdfcd2b3ac637"
