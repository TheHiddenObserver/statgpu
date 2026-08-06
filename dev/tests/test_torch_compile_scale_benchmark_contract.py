"""Regression tests for the torch.compile scale-crossover benchmark."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "dev"
    / "benchmarks"
    / "benchmark_torch_compile_scale.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_torch_compile_scale", _BENCHMARK_PATH
)
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BENCHMARK)


def test_parse_scales_deduplicates_and_preserves_order():
    assert _BENCHMARK._parse_scales("1024x64, 4096×64,1024x64") == (
        (1024, 64),
        (4096, 64),
    )


@pytest.mark.parametrize("value", ["", "1024", "0x64", "abcx64", "64x-1"])
def test_parse_scales_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _BENCHMARK._parse_scales(value)


def test_parse_cases_validates_and_deduplicates():
    assert _BENCHMARK._parse_cases("lasso,group_scad,lasso") == (
        "lasso",
        "group_scad",
    )
    with pytest.raises(argparse.ArgumentTypeError, match="unknown case"):
        _BENCHMARK._parse_cases("ridge")


def test_standard_preset_separates_n_and_p_scaling():
    scales, cases, repeats = _BENCHMARK._resolve_plan(
        "standard", None, None, None
    )
    assert (1024, 64) in scales
    assert (4096, 64) in scales
    assert (4096, 256) in scales
    assert cases == _BENCHMARK._CASE_NAMES
    assert repeats == 7


def test_scale_benchmark_requires_cold_and_multiple_warm_observations():
    with pytest.raises(ValueError, match="at least 3 repeats"):
        _BENCHMARK._resolve_plan("quick", None, None, 2)


def test_timing_summary_reports_dispersion_speedup_and_break_even():
    summary = _BENCHMARK._summarize_timings(
        eager_values=[10.0, 11.0, 12.0],
        compiled_values=[30.0, 5.0, 6.0, 5.0],
    )
    assert summary["eager"]["median"] == 11.0
    assert summary["eager"]["min"] == 10.0
    assert summary["eager"]["max"] == 12.0
    assert summary["eager"]["iqr"] == pytest.approx(1.0)
    assert summary["compiled_cold"] == 30.0
    assert summary["compiled_warm"]["median"] == 5.0
    assert summary["warm_speedup"] == pytest.approx(2.2)
    assert summary["cold_overhead_ratio"] == pytest.approx(30.0 / 11.0)
    assert summary["break_even_additional_warm_fits"] == 4
    assert summary["break_even_total_fits"] == 5


def test_timing_summary_marks_nonamortizable_warm_slowdown():
    summary = _BENCHMARK._summarize_timings(
        eager_values=[1.0, 1.1, 0.9],
        compiled_values=[4.0, 1.2, 1.3],
    )
    assert summary["warm_speedup"] < 1.0
    assert summary["break_even_additional_warm_fits"] is None
    assert summary["break_even_total_fits"] is None


def test_compile_evidence_distinguishes_eager_and_explicit_compile():
    assert (
        _BENCHMARK._validate_mode_evidence(
            "disable", ({"status": "disabled"},), 0
        )
        == "eager-no-dynamo-graph"
    )
    assert (
        _BENCHMARK._validate_mode_evidence(
            "default", ({"status": "compiled"},), 1
        )
        == "compiled-diagnostic-and-dynamo-graph"
    )
    with pytest.raises(RuntimeError, match="did not create"):
        _BENCHMARK._validate_mode_evidence(
            "default", ({"status": "compiled"},), 0
        )


def test_markdown_summary_exposes_scale_and_break_even(tmp_path):
    report = {
        "preset": "quick",
        "repeats": 3,
        "environment": {"gpu": "GPU", "torch": "2.x", "cuda": "12.x"},
        "results": [
            {
                "axis": "baseline",
                "n_samples": 1024,
                "n_features": 64,
                "case": "lasso",
                "timing_summary": {
                    "eager": {"median": 1.0},
                    "compiled_cold": 4.0,
                    "compiled_warm": {"median": 0.5},
                    "warm_speedup": 2.0,
                    "cold_overhead_ratio": 4.0,
                    "break_even_total_fits": 7,
                },
            }
        ],
    }
    markdown = _BENCHMARK._render_markdown(
        report, tmp_path / "torch_compile_scale.json"
    )
    assert "1024" in markdown
    assert "`lasso`" in markdown
    assert "2.000x" in markdown
    assert "break-even total fits" in markdown
