"""Regression contract for Stage-B Hausman covariance degrees of freedom."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import PanelOLS, RandomEffects


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


def test_hausman_uses_standard_fe_covariance_without_changing_legacy_inference():
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

    # Stage A's historical FE inference denominator is intentionally retained.
    assert legacy_df == fe.df_resid
    assert standard_df == legacy_df - 1
    assert_allclose(fe.bse_ ** 2, np.diag(raw_fe), rtol=1e-12, atol=1e-14)

    # Classical Hausman, however, needs the full nuisance-effect model rank.
    # Only the small diagnostic covariance copy is rescaled; public bse/CI above
    # still come from the raw Stage-A covariance.
    expected_fe_diagnostic = raw_fe * (legacy_df / standard_df)
    assert_allclose(
        fe._panel_cov_params,
        expected_fe_diagnostic,
        rtol=1e-12,
        atol=1e-14,
    )

    # RandomEffects has no legacy FE nuisance-df mismatch, so its diagnostic
    # covariance is exactly the inference covariance.
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
