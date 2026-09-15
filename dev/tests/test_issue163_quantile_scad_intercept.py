"""Regression coverage for the Quantile SCAD/MCP intercept objective."""

from __future__ import annotations

import numpy as np

from dev.benchmarks import validate_quantile_solver_provenance_gpu as gate
from statgpu.linear_model.penalized import PenalizedQuantileRegression


def _weighted_quantile(values, q, sample_weight):
    """Return the left-continuous weighted empirical quantile."""
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(sample_weight, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = float(q) * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def test_physical_fixture_scad_intercept_minimizes_weighted_pinball_coordinate():
    """The SCAD route must optimize, not mean-reconstruct, the Quantile intercept."""
    X, y, weights, _ = gate._data()
    model = PenalizedQuantileRegression(
        quantile=gate.Q,
        penalty="scad",
        alpha=gate.SCAD_ALPHA,
        solver="auto",
        device="cpu",
        max_iter=220,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)

    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    intercept = float(model.intercept_)
    residual_without_intercept = y - X @ coef
    coordinate_minimizer = _weighted_quantile(
        residual_without_intercept, gate.Q, weights
    )

    observed = gate._scad_objective(
        X, y, weights, coef, intercept, alpha=gate.SCAD_ALPHA
    )
    coordinate_best = gate._scad_objective(
        X,
        y,
        weights,
        coef,
        coordinate_minimizer,
        alpha=gate.SCAD_ALPHA,
    )

    # For fixed feature coefficients, the weighted q-quantile of y-Xb is an
    # exact minimizer of the intercept coordinate. A mean-centering shortcut
    # violates this property badly on the deliberately asymmetric q=.20 fixture.
    assert observed - coordinate_best <= 1e-6
    assert abs(intercept - coordinate_minimizer) <= 5e-4

    assert model._selected_solver == "proximal_irls_cd"
    assert model._selected_backend_name == "numpy"
    assert model._selected_backend_device == "cpu"
