"""Hosted matrix checks for Issue #160 diagnostics."""

from __future__ import annotations

import numpy as np

from dev.benchmarks import run_issue160_diagnostic_matrix as matrix


def test_numpy_matrix_covers_unweighted_weighted_and_rescaled_float32():
    result = matrix._seed_matrix(
        np.float32,
        151025,
        backend_specs=[("numpy", False)],
    )["numpy"]

    assert set(result) >= {
        "unweighted",
        "weighted",
        "weighted_scaled",
        "analytic_weight_rescale_errors",
    }
    for mode in ("unweighted", "weighted", "weighted_scaled"):
        case = result[mode]
        assert case["trace_matches_production_max_abs"] <= 1.0e-12
        assert case["ordinary_estimator_bridge"]["available"] is True
        assert case["ordinary_estimator_bridge"]["params_max_abs"] <= 1.0e-12
        assert np.isfinite(case["final"]["objective"])
        assert np.isfinite(case["final"]["gradient_norm"])

    scale = result["analytic_weight_rescale_errors"]
    assert scale["params_max_abs"] <= 2.0e-6
    assert scale["objective_abs"] <= 2.0e-6


def test_numpy_matrix_records_float64_reference_without_freezing_cross_backend_tol():
    f32 = matrix._seed_matrix(
        np.float32,
        16001,
        backend_specs=[("numpy", False)],
    )["numpy"]
    f64 = matrix._seed_matrix(
        np.float64,
        16001,
        backend_specs=[("numpy", False)],
    )["numpy"]

    for mode in ("unweighted", "weighted", "weighted_scaled"):
        errors = matrix._errors(f64[mode], f32[mode])
        assert set(errors) == {
            "params_max_abs",
            "objective_abs",
            "gradient_norm_abs",
            "n_iter_abs",
        }
        assert all(np.isfinite(float(value)) for value in errors.values())


def test_scaled_weights_preserve_float32_dtype():
    _, _, weights = matrix.diag._data(151025, np.float32)
    scaled = matrix._scaled_weights(weights)
    assert scaled.dtype == np.float32
    np.testing.assert_allclose(
        scaled,
        weights * np.float32(matrix.WEIGHT_SCALE),
        rtol=0.0,
        atol=0.0,
    )
