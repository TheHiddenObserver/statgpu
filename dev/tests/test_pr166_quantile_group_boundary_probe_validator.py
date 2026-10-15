"""Hosted contract for the PR166 physical boundary-probe validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "dev" / "benchmarks" / "validate_quantile_group_boundary_probe_gpu.py"
WRAPPER = ROOT / "dev" / "benchmarks" / "run_quantile_group_lla_gpu_gate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boundary_probe_validator_cpu_case_exercises_ambiguous_budget_state():
    gate = _load(VALIDATOR, "pr166_boundary_validator")
    X = np.eye(4, dtype=np.float64)
    y = np.asarray([0.8, -0.5, 0.4, -0.3], dtype=np.float64)
    weights = np.asarray([0.6, 0.9, 1.2, 1.6], dtype=np.float64)

    result = gate._run_case("numpy", X, y, weights)

    assert result["backend"] == "numpy"
    assert result["device"] == "cpu"
    assert result["parameter_error"] == 0.0
    assert result["n_iter"] == 2
    assert len(result["irls_calls"]) == 2
    assert result["irls_calls"][0]["init_coef"] is None
    assert result["irls_calls"][1]["init_coef"] == ["numpy", "cpu"]


def test_group_wrapper_declares_boundary_probe_as_exact_source_inner_gate():
    wrapper = _load(WRAPPER, "pr166_group_wrapper")
    assert wrapper.BOUNDARY_RUNNER == VALIDATOR
    assert wrapper.BOUNDARY_SCHEMA_VERSION == 1
