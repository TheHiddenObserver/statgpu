"""CPU smoke coverage for the PR79 penalized Cox parity diagnostic."""

import builtins

import numpy as np

from dev.benchmarks.pr79.diagnose_cox_pen import (
    build_report,
    stable_sort_risk_set_inputs,
)


def test_cpu_parity_report_executes_and_has_required_schema(monkeypatch):
    real_import = builtins.__import__

    def cpu_only_import(name, *args, **kwargs):
        if name in {"cupy", "torch"} or name.startswith(("cupy.", "torch.")):
            raise ImportError("optional GPU dependency blocked by CPU smoke")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_only_import)
    report = build_report(backend="numpy", include_timing=False)

    required_top_level = {
        "validated_code_sha",
        "case",
        "fixed_beta",
        "fitted",
        "checks",
        "status",
    }
    assert required_top_level <= set(report)
    assert report["status"] == "pass"
    assert report["errors"] == []
    assert report["case"]["risk_set_order"] == "ascending_time_stable"

    fixed_required = {
        "unpenalized_log_likelihood",
        "penalized_objective",
        "gradient",
        "unpenalized_hessian",
        "penalized_hessian",
        "covariance",
        "bse",
    }
    fitted_required = {
        "coefficients",
        "unpenalized_log_likelihood",
        "penalized_objective",
        "final_kkt_inf",
        "final_kkt_normalized",
        "converged",
        "termination_reason",
        "iterations",
        "bse",
    }
    assert fixed_required <= set(report["fixed_beta"]["numpy"])
    assert fitted_required <= set(report["fitted"]["numpy"])
    assert report["checks"]
    assert all(check["status"] == "pass" for check in report["checks"])


def test_stable_sort_keeps_all_risk_set_side_arrays_aligned():
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    times = np.array([2.0, 1.0, 1.0, 3.0])
    event = np.array([0, 1, 0, 1])
    entry = np.array([0.2, 0.1, 0.3, 0.4])
    cluster = np.array([20, 10, 11, 30])

    sorted_inputs = stable_sort_risk_set_inputs(
        X, times, event, entry=entry, cluster=cluster
    )

    expected_order = np.array([1, 2, 0, 3])
    np.testing.assert_array_equal(sorted_inputs["order"], expected_order)
    np.testing.assert_array_equal(sorted_inputs["X"], X[expected_order])
    np.testing.assert_array_equal(sorted_inputs["time"], times[expected_order])
    np.testing.assert_array_equal(sorted_inputs["event"], event[expected_order])
    np.testing.assert_array_equal(sorted_inputs["entry"], entry[expected_order])
    np.testing.assert_array_equal(sorted_inputs["cluster"], cluster[expected_order])
