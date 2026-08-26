"""Cross-estimator fit-statistics contracts for Panel Tier-1 Stage B."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelFitStatistics,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)


def _panel(seed=1240, unbalanced=False):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 7, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(np.linspace(-0.5, 0.65, n_entities), n_times)
    tau = np.tile(np.linspace(-0.15, 0.2, n_times), n_entities)
    y = 0.75 * X[:, 0] - 0.42 * X[:, 1] + alpha + tau + rng.normal(
        scale=0.18, size=entity.size
    )
    if unbalanced:
        keep = np.ones(entity.size, dtype=bool)
        keep[[1, 7, 18, 29]] = False
        X, y, entity, time = X[keep], y[keep], entity[keep], time[keep]
    return X, y, entity, time


def _assert_standard(result):
    assert isinstance(result, PanelFitStatistics)
    assert result.metadata["r2_definition"] == "parameter-based"
    for value in (
        result.rsquared_within,
        result.rsquared_between,
        result.rsquared_overall,
    ):
        assert value is None or np.isfinite(value)


def test_fit_statistics_populated_for_ols_style_panel_estimators():
    X, y, entity, time = _panel(unbalanced=True)
    models = [
        PooledOLS().fit(X, y, entity_ids=entity),
        BetweenOLS().fit(X, y, entity_ids=entity),
        FirstDifferenceOLS().fit(X, y, entity_ids=entity, time_ids=time),
        PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity),
        RandomEffects().fit(X, y, entity_ids=entity),
    ]
    for model in models:
        _assert_standard(model.fit_statistics_)
        assert model.fit_statistics_.rsquared_adj is not None
        assert np.isfinite(model.fit_statistics_.rsquared_adj)
        assert model.fit_statistics_.f_statistic is not None
        assert np.isfinite(model.fit_statistics_.f_statistic)
        assert model.fit_statistics_.f_pvalue is not None
        assert 0.0 <= model.fit_statistics_.f_pvalue <= 1.0
        assert model.fit_statistics_.f_df is not None


def test_two_way_fe_public_df_uses_standard_effect_rank():
    X, y, entity, time = _panel(seed=1241)
    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X,
        y,
        entity_ids=entity,
        time_ids=time,
    )
    n = len(y)
    N = len(np.unique(entity))
    T = len(np.unique(time))
    diag = model.fit_statistics_.metadata["diagnostic_df"]
    assert diag["effect_rank"] == N + T - 1
    expected_df = n - np.linalg.matrix_rank(
        X
        - np.vstack([X[entity == g].mean(axis=0) for g in entity])
        - np.vstack([X[time == t].mean(axis=0) for t in time])
        + X.mean(axis=0)
    ) - (N + T - 1)
    assert diag["df_resid"] == expected_df
    assert model.df_resid == expected_df
    assert model.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
    assert model.fit_statistics_.metadata["legacy_rsquared_within"] == model.rsquared_within


def test_fama_macbeth_exposes_r2_but_not_residual_ols_f_or_adjusted_r2():
    X, y, entity, time = _panel(seed=1242)
    model = FamaMacBeth(cov_type="newey-west", bandwidth=1).fit(
        X,
        y,
        time_ids=time,
        entity_ids=entity,
    )
    result = model.fit_statistics_
    _assert_standard(result)
    assert result.rsquared_within is not None
    assert result.rsquared_between is not None
    assert result.rsquared_overall is not None
    assert result.rsquared_adj is None
    assert result.f_statistic is None
    assert result.f_pvalue is None
    assert result.f_df is None
    assert "beta-series" in result.metadata["unavailable"]["model_f"]


def test_fama_macbeth_without_entity_ids_leaves_decomposition_unavailable():
    X, y, _, time = _panel(seed=1243)
    result = FamaMacBeth().fit(X, y, time_ids=time).fit_statistics_
    assert result.rsquared_within is None
    assert result.rsquared_between is None
    assert result.rsquared_overall is not None
    assert "within_between_r2" in result.metadata["unavailable"]


def test_between_and_first_difference_leave_existing_legacy_r2_unchanged():
    X, y, entity, time = _panel(seed=1244)
    between = BetweenOLS().fit(X, y, entity_ids=entity)
    fd = FirstDifferenceOLS().fit(X, y, entity_ids=entity, time_ids=time)
    assert_allclose(
        between.fit_statistics_.metadata["legacy_rsquared"],
        between.rsquared,
        rtol=0,
        atol=0,
    )
    assert_allclose(
        fd.fit_statistics_.metadata["legacy_rsquared"],
        fd.rsquared,
        rtol=0,
        atol=0,
    )
