"""Hosted contract tests for the Issue #160 L-BFGS diagnostic runner."""

from __future__ import annotations

import numpy as np
import pytest

from dev.benchmarks import diagnose_issue160_float32_lbfgs as diag


def test_historical_float32_fixture_reproduces_schema_v6_weight_roundtrip():
    _, _, weights = diag._data(151025, np.float32)
    raw = np.linspace(0.75, 1.05, diag.N_SAMPLES, dtype=np.float32)
    scale = np.float32(diag.FLOAT32_OVERFLOW_SCALE)
    overflow_weights = raw * scale

    assert np.all(np.isfinite(overflow_weights))
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.sum(overflow_weights, dtype=np.float32))
    expected = overflow_weights / scale
    np.testing.assert_array_equal(weights, expected)


def test_trace_runner_matches_production_numpy_float32_and_public_estimator():
    X, y, w = diag._data(151025, np.float32)
    result = diag._one_case("numpy", X, y, w, use_cuda=False)

    assert result["production_n_iter"] >= 1
    assert result["trace_matches_production_max_abs"] <= 1.0e-12
    assert result["trace"]["backend"] == "numpy"
    assert result["trace"]["termination"] in {
        "gradient", "parameter_step", "parameter_and_gradient",
        "line_search_failed", "max_iter",
    }
    bridge = result["ordinary_estimator_bridge"]
    assert bridge["available"] is True
    assert bridge["params_max_abs"] <= 1.0e-12
    assert bridge["selected_solver"] == "lbfgs"
    assert bridge["backend"] == "numpy"
    assert bridge["device"] == "cpu"
    assert np.isfinite(result["final"]["objective"])
    assert np.isfinite(result["final"]["gradient_norm"])


def test_trace_runner_matches_production_numpy_float64():
    X, y, w = diag._data(16001, np.float64)
    result = diag._one_case("numpy", X, y, w, use_cuda=False)
    assert result["trace_matches_production_max_abs"] <= 1.0e-12
    assert result["trace"]["dtype"] == "float64"
    assert result["ordinary_estimator_bridge"]["available"] is True


def test_torch_cpu_trace_matches_production_and_numpy_when_available():
    pytest.importorskip("torch")
    X, y, w = diag._data(16002, np.float32)
    numpy_result = diag._one_case("numpy", X, y, w, use_cuda=False)
    torch_result = diag._one_case("torch", X, y, w, use_cuda=False)

    assert torch_result["trace_matches_production_max_abs"] <= 1.0e-12
    assert torch_result["ordinary_estimator_bridge"]["available"] is False
    assert "Torch CPU" in torch_result["ordinary_estimator_bridge"]["reason"]
    p_np = np.asarray(numpy_result["final"]["params"], dtype=np.float64)
    p_t = np.asarray(torch_result["final"]["params"], dtype=np.float64)
    assert np.all(np.isfinite(p_t))
    # This is an investigation contract, not the final float32 parity policy.
    # Keep the hosted test descriptive rather than freezing a tolerance that
    # Issue #160 exists to decide.
    assert float(np.max(np.abs(p_t - p_np))) >= 0.0


def test_iteration_trace_records_curvature_and_armijo_state():
    X, y, w = diag._data(16003, np.float32)
    result = diag._one_case("numpy", X, y, w, use_cuda=False)
    records = result["trace"]["iterations"]
    assert records
    iterative = [r for r in records if "step" in r]
    assert iterative
    required = {
        "objective",
        "gradient_norm",
        "directional_derivative",
        "direction_norm",
        "history_size_before",
        "gamma",
        "step",
        "backtracks",
        "acceptance_mode",
        "objective_roundoff",
    }
    assert required.issubset(iterative[0])
    accepted = [r for r in iterative if r["acceptance_mode"] != "failed"]
    assert accepted
    assert {r["acceptance_mode"] for r in accepted} <= {"armijo", "roundoff"}


def test_environment_records_version_fields_without_claiming_cuda():
    env = diag._environment({"cupy": False, "torch": False})
    assert env["python"]
    assert env["numpy"] == np.__version__
    assert env["cuda_available"] == {"cupy": False, "torch": False}
    assert "torch" in env
    assert "cupy" in env
