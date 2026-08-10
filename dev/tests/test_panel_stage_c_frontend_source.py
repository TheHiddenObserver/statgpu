from __future__ import annotations

import json
from pathlib import Path

import pytest

from dev.benchmarks.frontend_data.parsers.panel_stage_c import (
    parse_panel_stage_c_performance,
    parse_panel_stage_c_physical_validation,
)

ROOT = Path(__file__).resolve().parents[2]
CORRECTNESS = ROOT / "results/pr126_p100/panel_stage_c_gpu_validation_5ed763be.json"
PERFORMANCE = ROOT / "results/pr126_p100/panel_stage_c_performance_5ed763be.json"


def test_stage_c_validation_parser_emits_exact_physical_matrix():
    runs, models, warnings = parse_panel_stage_c_physical_validation(
        CORRECTNESS, "remote-p100-pr126-20260810"
    )
    assert warnings == []
    assert len(runs) == 64
    assert {run["backend"] for run in runs} == {"cupy", "torch"}
    assert sum(run["model_id"] == "PanelCovariancePrimitive" for run in runs) == 12
    assert all(run["metrics"]["validation"]["status"] == "pass" for run in runs)
    assert all("timing" not in run["metrics"] for run in runs)
    assert all("speedup" not in run["metrics"] for run in runs)
    assert {model["model_id"] for model in models} == {
        "PooledOLS", "PanelOLS", "RandomEffects", "BetweenOLS",
        "FirstDifferenceOLS", "PanelCovariancePrimitive",
    }


def test_stage_c_performance_parser_emits_timing_without_speedup():
    runs, _, warnings = parse_panel_stage_c_performance(
        PERFORMANCE, "remote-p100-pr126-20260810"
    )
    assert warnings == []
    assert len(runs) == 58
    assert all(run["metrics"]["timing"]["fit_time_ms"] > 0 for run in runs)
    assert all("speedup" not in run["metrics"] for run in runs)
    base = [run for run in runs if run["parameters"]["scenario"] == "base"]
    high_t = [run for run in runs if run["parameters"]["scenario"] == "high_t_qs"]
    assert len(base) == 54
    assert len(high_t) == 4
    assert {run["backend"] for run in high_t} == {"cupy", "torch"}
    assert {run["parameters"]["n_times"] for run in high_t} == {200}
    assert {run["model_id"] for run in high_t} == {"PooledOLS", "PanelOLS"}
    assert {run["scale"]["scale_key"] for run in base} == {
        "n10000_p2_t20", "n100000_p2_t20", "n100000_p10_t20"
    }
    assert {run["scale"]["scale_key"] for run in high_t} == {"n10000_p2_t200"}
    assert {run["scale"]["label"] for run in high_t} == {"10K×2 · T=200"}


def test_stage_c_performance_parser_fails_closed_on_high_t_contract(tmp_path):
    data = json.loads(PERFORMANCE.read_text(encoding="utf-8"))
    data["high_t_scale"] = "10000x2x20"
    broken = tmp_path / "broken_performance.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="high-T scale drifted"):
        parse_panel_stage_c_performance(broken, "remote-p100-pr126-20260810")


def test_stage_c_validation_parser_fails_closed_on_case_identity(tmp_path):
    data = json.loads(CORRECTNESS.read_text(encoding="utf-8"))
    del data["backends"]["cupy"]["cases"]["pooled_hc0"]
    broken = tmp_path / "broken_validation.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="case identity drifted"):
        parse_panel_stage_c_physical_validation(broken, "remote-p100-pr126-20260810")


def test_stage_c_performance_parser_fails_closed_on_base_matrix_drift(tmp_path):
    data = json.loads(PERFORMANCE.read_text(encoding="utf-8"))
    base_rows = [row for row in data["rows"] if row["scenario"] == "base"]
    assert len(base_rows) == 54
    base_rows[0]["case"] = base_rows[1]["case"]
    broken = tmp_path / "broken_base_matrix.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="base matrix drifted"):
        parse_panel_stage_c_performance(broken, "remote-p100-pr126-20260810")


def test_stage_c_performance_parser_fails_closed_on_high_t_backend_matrix_drift(tmp_path):
    data = json.loads(PERFORMANCE.read_text(encoding="utf-8"))
    high_t = [row for row in data["rows"] if row["scenario"] == "high_t_qs"]
    assert len(high_t) == 4
    high_t[0]["backend"] = "torch"
    broken = tmp_path / "broken_high_t_matrix.json"
    broken.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="high-T QS matrix drifted"):
        parse_panel_stage_c_performance(broken, "remote-p100-pr126-20260810")
