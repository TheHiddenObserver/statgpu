from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from dev.benchmarks.frontend_data.parsers.panel_stage_c_identifiability import (
    parse_panel_stage_c_identifiability_performance,
    parse_panel_stage_c_identifiability_physical_validation,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "results/pr126_p100/panel_stage_c_gpu_validation_a99726e1.json"
PERFORMANCE = ROOT / "results/pr126_p100/panel_stage_c_performance_a99726e1.json"
ENV = "remote-p100-pr126-identifiability-20260812"


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_v4_validation_parser_enforces_identifiability_and_prediction_backend():
    runs, models, warnings = parse_panel_stage_c_identifiability_physical_validation(VALIDATION, ENV)
    assert len(runs) == 94
    assert warnings == []
    assert models
    assert all(run["source"]["parser_version"] == "4.0" for run in runs)
    rank_runs = [
        run for run in runs
        if run["parameters"].get("coefficient_inference_applicable") is False
    ]
    assert len(rank_runs) == 18
    assert all("rank deficient" in run["parameters"]["coefficient_inference_reason"].lower() for run in rank_runs)
    prediction_runs = [run for run in runs if run["parameters"].get("prediction_backend") is not None]
    assert len(prediction_runs) == 4
    assert all(run["parameters"]["prediction_backend"] == run["backend"] for run in prediction_runs)


def test_v4_validation_parser_rejects_inference_or_prediction_provenance_drift(tmp_path):
    payload = json.loads(VALIDATION.read_text())
    broken = copy.deepcopy(payload)
    broken["backends"]["cupy"]["cases"]["panel_rank_boundary_dk"]["coefficient_inference_applicable"] = True
    with pytest.raises(ValueError, match="coefficient-inference|rank-boundary|identified-rank"):
        parse_panel_stage_c_identifiability_physical_validation(
            _write(tmp_path, "bad-inference.json", broken), ENV
        )

    broken = copy.deepcopy(payload)
    broken["backends"]["torch"]["cases"]["panel_entity_hc0"]["prediction_backend"] = "numpy"
    with pytest.raises(ValueError, match="prediction backend"):
        parse_panel_stage_c_identifiability_physical_validation(
            _write(tmp_path, "bad-predict.json", broken), ENV
        )


def test_v4_performance_parser_enforces_60_row_matrix_and_two_way_case():
    runs, models, warnings = parse_panel_stage_c_identifiability_performance(PERFORMANCE, ENV)
    assert len(runs) == 60
    assert warnings == []
    assert models
    two_way = [run for run in runs if run["parameters"]["scenario"] == "two_way_unbalanced"]
    assert len(two_way) == 2
    assert {run["backend"] for run in two_way} == {"cupy", "torch"}
    assert all(run["source"]["parser_version"] == "4.0" for run in runs)


def test_v4_performance_parser_rejects_matrix_repeat_or_median_drift(tmp_path):
    payload = json.loads(PERFORMANCE.read_text())
    broken = copy.deepcopy(payload)
    broken["rows"] = broken["rows"][:-1]
    with pytest.raises(ValueError, match="60 rows"):
        parse_panel_stage_c_identifiability_performance(
            _write(tmp_path, "bad-matrix.json", broken), ENV
        )

    broken = copy.deepcopy(payload)
    broken["rows"][0]["repeats"] = 2
    broken["rows"][0]["samples_seconds"] = broken["rows"][0]["samples_seconds"][:2]
    broken["rows"][0]["median_seconds"] = sum(broken["rows"][0]["samples_seconds"]) / 2.0
    with pytest.raises(ValueError, match="exactly three raw samples"):
        parse_panel_stage_c_identifiability_performance(
            _write(tmp_path, "bad-repeats.json", broken), ENV
        )

    broken = copy.deepcopy(payload)
    current = float(broken["rows"][0]["median_seconds"])
    broken["rows"][0]["median_seconds"] = math.nextafter(current, math.inf)
    with pytest.raises(ValueError, match="exactly match"):
        parse_panel_stage_c_identifiability_performance(
            _write(tmp_path, "bad-median.json", broken), ENV
        )
