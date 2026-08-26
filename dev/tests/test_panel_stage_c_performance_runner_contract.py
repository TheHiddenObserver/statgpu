"""Hosted contract checks for the Stage-C physical performance runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _runner():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_panel_stage_c_covariance.py"
    spec = importlib.util.spec_from_file_location("panel_stage_c_perf_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_high_t_scenario_is_explicit_and_qs_only():
    mod = _runner()
    assert mod.PERFORMANCE_SCHEMA_VERSION == 3
    n, k, n_times = mod._parse_high_t_scale(mod.DEFAULT_HIGH_T_SCALE)
    assert (n, k, n_times) == (10000, 2, 200)
    assert n_times >= 200
    assert set(mod.HIGH_T_CASES) == {"pooled_dk_qs", "panel_entity_dk_qs"}


def test_two_way_unbalanced_scenario_is_explicit_and_incomplete():
    mod = _runner()
    assert mod.DEFAULT_TWO_WAY_UNBALANCED_SCALE == "10000x2x20"
    assert mod.TWO_WAY_UNBALANCED_CASE == "panel_two_way_nonrobust"
    X, y, entity, time, clusters = mod._unbalanced_two_way_dataset(
        1000, 2, 43, n_times=20
    )
    assert X.shape == (1000, 2)
    assert y.shape == (1000,)
    assert clusters.shape == (1000, 2)
    assert np.unique(clusters, axis=0).shape[0] == 1000
    counts = np.bincount(entity)
    assert np.unique(counts[counts > 0]).size > 1
    assert len(np.unique(time)) == 20


def test_dataset_honors_requested_time_dimension():
    mod = _runner()
    X, y, entity, time, clusters = mod._dataset(1000, 2, 42, n_times=100)
    assert X.shape == (1000, 2)
    assert y.shape == (1000,)
    assert clusters.shape == (1000, 2)
    assert len(np.unique(time)) == 100
    assert len(entity) == 1000
    with pytest.raises(ValueError, match="n>=n_times"):
        mod._dataset(10, 2, 42, n_times=20)


def test_timing_row_schema_records_scenario_and_time_dimension():
    mod = _runner()
    row = mod._timing_row(
        backend="cupy",
        case="pooled_dk_qs",
        scenario="high_t_qs",
        n=10000,
        k=2,
        n_times=200,
        repeats=3,
        samples=[0.3, 0.2, 0.4],
    )
    assert row["scenario"] == "high_t_qs"
    assert row["n_times"] == 200
    assert row["median_seconds"] == pytest.approx(0.3)
    assert row["samples_seconds"] == [0.3, 0.2, 0.4]
    assert mod.PERFORMANCE_SCHEMA_VERSION >= 2



def test_cupy_package_version_accepts_cuda_specific_distribution(monkeypatch):
    mod = _runner()
    versions = {"cupy-cuda12x": "13.6.0"}
    monkeypatch.setattr(mod, "_version", lambda name: versions.get(name))
    assert mod._package_version("cupy") == "13.6.0"
