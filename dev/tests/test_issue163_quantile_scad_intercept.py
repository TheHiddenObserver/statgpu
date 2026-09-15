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


def _fit_scad_fixture(X, y, weights):
    return PenalizedQuantileRegression(
        quantile=gate.Q,
        penalty="scad",
        alpha=gate.SCAD_ALPHA,
        solver="auto",
        device="cpu",
        max_iter=220,
        tol=1e-6,
    ).fit(X, y, sample_weight=weights)


def test_physical_fixture_scad_intercept_minimizes_weighted_pinball_coordinate():
    """The SCAD route must optimize, not mean-reconstruct, the Quantile intercept."""
    X, y, weights, _ = gate._data()
    model = _fit_scad_fixture(X, y, weights)

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


def test_physical_fixture_flat_scad_matches_unpenalized_quantile_data_fit():
    """A fully flat final SCAD surrogate should close as ordinary Quantile IRLS."""
    X, y, weights, _ = gate._data()
    scad = _fit_scad_fixture(X, y, weights)
    reference = PenalizedQuantileRegression(
        quantile=gate.Q,
        penalty="l2",
        alpha=0.0,
        solver="irls",
        device="cpu",
        max_iter=600,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)

    scad_coef = np.asarray(scad.coef_, dtype=np.float64).ravel()
    ref_coef = np.asarray(reference.coef_, dtype=np.float64).ravel()

    # On this exact physical fixture every final feature lies beyond the SCAD
    # constant region, so the target-alpha data-fit problem is unpenalized.
    assert float(np.min(np.abs(scad_coef))) > 3.7 * gate.SCAD_ALPHA

    scad_data_fit = gate._pinball(
        y, X @ scad_coef + float(scad.intercept_), gate.Q, weights
    )
    reference_data_fit = gate._pinball(
        y, X @ ref_coef + float(reference.intercept_), gate.Q, weights
    )
    assert abs(scad_data_fit - reference_data_fit) <= 5e-5
