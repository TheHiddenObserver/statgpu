"""Regression contract for Stage-B Hausman covariance degrees of freedom."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import PanelOLS, RandomEffects
from statgpu.panel._diagnostics import _hausman_quadratic


def _panel(seed=1250):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 9, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(np.linspace(-0.6, 0.7, n_entities), n_times)
    y = 0.9 * X[:, 0] - 0.35 * X[:, 1] + alpha + rng.normal(
        scale=0.2, size=entity.size
    )
    return X, y, entity


def test_hausman_and_public_inference_share_standard_fe_covariance():
    X, y, entity = _panel()
    fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
        X, y, entity_ids=entity
    )
    re = RandomEffects().fit(X, y, entity_ids=entity)

    raw_fe = np.asarray(fe._panel_cov_params_raw)
    raw_re = np.asarray(re._panel_cov_params_raw)
    diag_meta = fe.fit_statistics_.metadata["diagnostic_df"]
    legacy_df = fe.fit_statistics_.metadata["legacy_df_resid"]
    standard_df = diag_meta["df_resid"]

    # Public FE inference and diagnostics now use the same full nuisance rank.
    assert standard_df == fe.df_resid
    assert legacy_df == standard_df + 1
    assert fe.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
    assert_allclose(fe.bse_ ** 2, np.diag(raw_fe), rtol=1e-12, atol=1e-14)
    expected_fe_diagnostic = raw_fe
    assert_allclose(
        fe._panel_cov_params,
        expected_fe_diagnostic,
        rtol=0,
        atol=0,
    )

    # RandomEffects likewise exposes the same inference covariance to diagnostics.
    assert_allclose(re._panel_cov_params, raw_re, rtol=0, atol=0)

    # The Hausman entry point must consume the diagnostic matrices.  Whether the
    # finite-sample covariance difference is PSD is data-dependent; this check
    # locks the exact matrix passed to the quadratic-form contract.
    result = fe.hausman_test(re)
    expected_difference = expected_fe_diagnostic - raw_re
    eigvals = np.linalg.eigvalsh(0.5 * (expected_difference + expected_difference.T))
    assert_allclose(
        result.metadata["minimum_eigenvalue"],
        eigvals.min(),
        rtol=1e-10,
        atol=1e-12,
    )
    assert_allclose(
        result.metadata["maximum_eigenvalue"],
        eigvals.max(),
        rtol=1e-10,
        atol=1e-12,
    )


def test_hausman_quadratic_is_scale_safe_at_float64_extremes():
    cases = (
        (1.0e308, np.sqrt(1.0e308)),
        (1.0e-320, np.sqrt(1.0e-320)),
    )
    results = []
    for variance, difference in cases:
        result = _hausman_quadratic(
            np.asarray([difference], dtype=np.float64),
            np.asarray([[variance]], dtype=np.float64),
        )
        assert result.applicable, result.reason
        assert result.df == 1.0
        assert result.metadata["quadratic_evaluation"] == "standardized_eigencoordinates"
        assert_allclose(result.statistic, 1.0, rtol=3e-12, atol=0.0)
        assert np.isfinite(result.pvalue)
        results.append(result)

    assert_allclose(results[0].pvalue, results[1].pvalue, rtol=3e-12, atol=0.0)


def test_hausman_quadratic_rejects_large_finite_nullspace_component():
    result = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e200]),
        np.diag(np.asarray([1.0e308, 0.0])),
    )
    assert result.applicable is False
    assert "outside the identified covariance-difference range" in result.reason
    assert result.metadata["range_comparison_scale"] == 1.0e200
    assert np.isfinite(result.metadata["range_tolerance_normalized"])
    assert np.isfinite(result.metadata["nullspace_component_norm_normalized"])


def test_hausman_quadratic_normalizes_dense_large_covariance_scale():
    result = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e154]),
        np.full((2, 2), 1.0e308, dtype=np.float64),
    )
    assert result.applicable, result.reason
    assert result.df == 1.0
    assert result.metadata["eigen_scale"] == 1.0e308
    assert np.isinf(result.metadata["maximum_eigenvalue"])
    assert_allclose(result.metadata["maximum_eigenvalue_normalized"], 2.0, rtol=0.0, atol=0.0)
    assert result.metadata["quadratic_evaluation"] == "standardized_eigencoordinates"
    assert_allclose(result.statistic, 1.0, rtol=5e-13, atol=0.0)
    assert np.isfinite(result.pvalue)

def test_hausman_quadratic_scales_dense_projection_before_range_check():
    basis = np.full(4, 0.5, dtype=np.float64)
    covariance = 1.0e308 * np.outer(basis, basis)
    difference = np.asarray(
        [
            1.0e308 + 1.0e300,
            1.0e308 - 1.0e300,
            1.0e308,
            1.0e308,
        ],
        dtype=np.float64,
    )
    result = _hausman_quadratic(difference, covariance)
    assert result.applicable is False
    assert (
        "outside the identified covariance-difference range"
        in result.reason
    )
    assert np.isfinite(
        result.metadata["range_tolerance_normalized"]
    )
    assert np.isfinite(
        result.metadata["nullspace_component_norm_normalized"]
    )
