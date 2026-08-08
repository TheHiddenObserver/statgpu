"""Executable external-definition alignment against linearmodels 7.0.

This file is run in a dedicated CI job that installs linearmodels==7.0. It
compares only quantities whose estimator parameterizations and transformed
samples are intentionally aligned in Stage B. RandomEffects coefficients are
excluded because statgpu's existing Swamy-Arora path is a preserved
model-specific contract; for RandomEffects we compare only structural
diagnostic contracts that do not require coefficient equality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip(
    "linearmodels",
    reason="linearmodels is an external Stage-B definition gate, not a base dependency",
)

from linearmodels.panel import (
    BetweenOLS as LMBetweenOLS,
    FirstDifferenceOLS as LMFirstDifferenceOLS,
    PanelOLS as LMPanelOLS,
    PooledOLS as LMPooledOLS,
    RandomEffects as LMRandomEffects,
)

from statgpu.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)


def _panel(seed=1260, *, unbalanced=False):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 9, 6
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(np.linspace(-0.65, 0.75, n_entities), n_times)
    tau = np.tile(np.linspace(-0.2, 0.25, n_times), n_entities)
    y = 0.85 * X[:, 0] - 0.4 * X[:, 1] + alpha + tau + rng.normal(
        scale=0.2, size=entity.size
    )
    if unbalanced:
        keep = np.ones(entity.size, dtype=bool)
        keep[[1, 8, 17, 31, 44]] = False
        X, y, entity, time = X[keep], y[keep], entity[keep], time[keep]
    return X, y, entity, time


def _gap_free_unbalanced_panel(seed=1263):
    """Return an unbalanced panel with contiguous time support per entity."""
    X, y, entity, time = _panel(seed=seed, unbalanced=False)
    keep = np.ones(len(y), dtype=bool)
    keep[(entity == 1) & (time >= 5)] = False
    keep[(entity == 3) & (time >= 4)] = False
    keep[(entity == 7) & (time >= 3)] = False
    return X[keep], y[keep], entity[keep], time[keep]


def _lm_data(X, y, entity, time, *, constant=False):
    index = pd.MultiIndex.from_arrays([entity, time], names=["entity", "time"])
    y_series = pd.Series(y, index=index, name="y")
    X_frame = pd.DataFrame(X, index=index, columns=["x1", "x2"])
    if constant:
        X_frame.insert(0, "const", 1.0)
    return y_series, X_frame


def _assert_r2(actual, expected):
    assert_allclose(
        actual.rsquared_within,
        expected.rsquared_within,
        rtol=2e-10,
        atol=2e-11,
    )
    assert_allclose(
        actual.rsquared_between,
        expected.rsquared_between,
        rtol=2e-10,
        atol=2e-11,
    )
    assert_allclose(
        actual.rsquared_overall,
        expected.rsquared_overall,
        rtol=2e-10,
        atol=2e-11,
    )


def _assert_model_f(actual, expected):
    assert actual.f_statistic is not None
    assert_allclose(
        actual.f_statistic,
        expected.f_statistic.stat,
        rtol=2e-9,
        atol=2e-11,
    )
    assert_allclose(
        actual.f_pvalue,
        expected.f_statistic.pval,
        rtol=2e-8,
        atol=1e-14,
    )
    assert actual.f_df == (
        float(expected.f_statistic.df),
        float(expected.f_statistic.df_denom),
    )


def test_one_way_panelols_matches_linearmodels_parameter_r2_f_pooling_and_diagnostic_covariance():
    X, y, entity, time = _panel(unbalanced=True)
    y_lm, X_lm = _lm_data(X, y, entity, time, constant=False)
    lm = LMPanelOLS(y_lm, X_lm, entity_effects=True).fit(
        cov_type="unadjusted", debiased=True
    )
    sg = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
        X, y, entity_ids=entity
    )

    assert_allclose(sg.coef_, lm.params.to_numpy(), rtol=2e-10, atol=2e-11)
    _assert_r2(sg.fit_statistics_, lm)
    _assert_model_f(sg.fit_statistics_, lm)

    pooled = sg.pooling_f_test()
    assert pooled.applicable
    assert_allclose(pooled.statistic, lm.f_pooled.stat, rtol=2e-9, atol=2e-11)
    assert_allclose(pooled.pvalue, lm.f_pooled.pval, rtol=2e-8, atol=1e-14)
    assert pooled.df == (float(lm.f_pooled.df), float(lm.f_pooled.df_denom))
    assert_allclose(
        sg._panel_cov_params,
        lm.cov.to_numpy(),
        rtol=5e-9,
        atol=5e-11,
    )


def test_two_way_panelols_matches_linearmodels_standard_diagnostics():
    X, y, entity, time = _panel(seed=1261)
    y_lm, X_lm = _lm_data(X, y, entity, time, constant=False)
    lm = LMPanelOLS(
        y_lm,
        X_lm,
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="unadjusted", debiased=True)
    sg = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="nonrobust",
    ).fit(X, y, entity_ids=entity, time_ids=time)

    assert_allclose(sg.coef_, lm.params.to_numpy(), rtol=2e-10, atol=2e-11)
    _assert_r2(sg.fit_statistics_, lm)
    _assert_model_f(sg.fit_statistics_, lm)
    pooled = sg.pooling_f_test()
    assert pooled.applicable
    assert_allclose(pooled.statistic, lm.f_pooled.stat, rtol=2e-9, atol=2e-11)
    assert_allclose(pooled.pvalue, lm.f_pooled.pval, rtol=2e-8, atol=1e-14)
    assert pooled.df == (float(lm.f_pooled.df), float(lm.f_pooled.df_denom))
    assert_allclose(sg._panel_cov_params, lm.cov.to_numpy(), rtol=5e-9, atol=5e-11)


def test_pooled_and_between_match_linearmodels_on_general_unbalanced_panel():
    X, y, entity, time = _panel(seed=1262, unbalanced=True)
    y_lm, X_lm_const = _lm_data(X, y, entity, time, constant=True)

    lm_pool = LMPooledOLS(y_lm, X_lm_const).fit(
        cov_type="unadjusted", debiased=True
    )
    sg_pool = PooledOLS().fit(X, y, entity_ids=entity)
    assert_allclose(sg_pool.coef_, lm_pool.params.to_numpy(), rtol=2e-10, atol=2e-11)
    _assert_r2(sg_pool.fit_statistics_, lm_pool)
    _assert_model_f(sg_pool.fit_statistics_, lm_pool)

    lm_between = LMBetweenOLS(y_lm, X_lm_const).fit(
        cov_type="unadjusted", debiased=True
    )
    sg_between = BetweenOLS().fit(X, y, entity_ids=entity)
    assert_allclose(
        sg_between.coef_, lm_between.params.to_numpy(), rtol=2e-10, atol=2e-11
    )
    _assert_r2(sg_between.fit_statistics_, lm_between)
    _assert_model_f(sg_between.fit_statistics_, lm_between)


def test_random_effects_explicit_constant_matches_linearmodels_f_df_structure():
    X, y, entity, time = _panel(seed=1264, unbalanced=True)
    y_lm, X_lm_const = _lm_data(X, y, entity, time, constant=True)
    lm = LMRandomEffects(y_lm, X_lm_const).fit(
        cov_type="unadjusted", debiased=True
    )

    X_const = np.column_stack([np.ones(X.shape[0]), X])
    sg = RandomEffects().fit(X_const, y, entity_ids=entity)

    assert sg.fit_statistics_.metadata["has_explicit_constant"] is True
    assert sg.fit_statistics_.metadata["model_f"]["rank_restricted"] == 1
    assert sg.fit_statistics_.metadata["model_f"]["restricted_design_supplied"] is True
    assert sg.fit_statistics_.f_df == (
        float(lm.f_statistic.df),
        float(lm.f_statistic.df_denom),
    )


def test_hausman_absorbed_intercept_matches_linearmodels_parameter_structure():
    X, y, entity, time = _panel(seed=1265, unbalanced=True)
    y_lm, X_lm = _lm_data(X, y, entity, time, constant=False)
    _, X_lm_const = _lm_data(X, y, entity, time, constant=True)

    lm_fe = LMPanelOLS(y_lm, X_lm, entity_effects=True).fit(
        cov_type="unadjusted", debiased=True
    )
    lm_re = LMRandomEffects(y_lm, X_lm_const).fit(
        cov_type="unadjusted", debiased=True
    )
    assert tuple(lm_fe.params.index) == ("x1", "x2")
    assert tuple(lm_re.params.index) == ("const", "x1", "x2")

    sg_fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
        X, y, entity_ids=entity
    )
    sg_re = RandomEffects().fit(
        np.column_stack([np.ones(X.shape[0]), X]),
        y,
        entity_ids=entity,
    )
    result = sg_fe.hausman_test(sg_re)

    assert "identity mismatch" not in (result.reason or "")
    assert "no common estimable slope" not in (result.reason or "")
    assert result.metadata["common_features"] == ("x1", "x2")
    assert result.metadata["fe_coefficient_indices"] == (0, 1)
    assert result.metadata["re_coefficient_indices"] == (1, 2)
    assert result.metadata["re_explicit_constant_excluded"] is True


def test_first_difference_matches_linearmodels_when_transformed_sample_is_common():
    X, y, entity, time = _gap_free_unbalanced_panel()
    y_lm, X_lm = _lm_data(X, y, entity, time, constant=False)
    lm_fd = LMFirstDifferenceOLS(y_lm, X_lm).fit(
        cov_type="unadjusted", debiased=True
    )
    sg_fd = FirstDifferenceOLS().fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert_allclose(sg_fd.coef_, lm_fd.params.to_numpy(), rtol=2e-10, atol=2e-11)
    _assert_r2(sg_fd.fit_statistics_, lm_fd)
    _assert_model_f(sg_fd.fit_statistics_, lm_fd)
