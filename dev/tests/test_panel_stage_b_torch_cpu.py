"""Hosted Torch-CPU parity coverage for Panel Tier-1 Stage B diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import FamaMacBeth, PanelOLS, PooledOLS, RandomEffects
from statgpu.panel._diagnostic_context import explicit_constant_column


torch = pytest.importorskip("torch")


def _panel(seed=1220):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 7, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(np.linspace(-0.5, 0.7, n_entities), n_times)
    y = 0.8 * X[:, 0] - 0.35 * X[:, 1] + alpha + rng.normal(
        scale=0.22, size=entity.size
    )
    return X, y, entity, time


def _torch_arrays(X, y, entity, time):
    return (
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        torch.as_tensor(entity, dtype=torch.int64),
        torch.as_tensor(time, dtype=torch.int64),
    )


def _assert_fit_statistics_close(actual, expected, *, include_adjusted=True):
    for name in ("rsquared_within", "rsquared_between", "rsquared_overall"):
        a = getattr(actual, name)
        e = getattr(expected, name)
        if e is None:
            assert a is None
        else:
            assert_allclose(a, e, rtol=5e-9, atol=5e-10)
    if include_adjusted:
        assert_allclose(
            actual.rsquared_adj,
            expected.rsquared_adj,
            rtol=5e-9,
            atol=5e-10,
        )
        if expected.f_statistic is None:
            assert actual.f_statistic is None
            assert actual.f_pvalue is None
            assert actual.f_df is None
        else:
            assert_allclose(
                actual.f_statistic,
                expected.f_statistic,
                rtol=5e-9,
                atol=5e-10,
            )
            assert_allclose(
                actual.f_pvalue,
                expected.f_pvalue,
                rtol=1e-8,
                atol=1e-14,
            )
            assert_allclose(actual.f_df, expected.f_df, rtol=0, atol=0)


def test_stage_b_pooled_torch_cpu_matches_numpy():
    X, y, entity, time = _panel()
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)

    expected = PooledOLS().fit(X, y, entity_ids=entity)
    actual = PooledOLS().fit(X_t, y_t, entity_ids=entity_t)
    assert_allclose(actual.coef_, expected.coef_, rtol=1e-10, atol=1e-11)
    _assert_fit_statistics_close(actual.fit_statistics_, expected.fit_statistics_)
    bp_expected = expected.breusch_pagan_lm_test()
    bp_actual = actual.breusch_pagan_lm_test()
    assert bp_actual.applicable == bp_expected.applicable
    assert_allclose(bp_actual.statistic, bp_expected.statistic, rtol=5e-9, atol=5e-10)
    assert_allclose(bp_actual.pvalue, bp_expected.pvalue, rtol=1e-8, atol=1e-14)

    # Exercise the metadata-permutation contract on Torch CPU as well.
    scrambled = torch.as_tensor((3 - time + 2 * entity) % 5, dtype=torch.int64)
    hac = PooledOLS(cov_type="hac", bandwidth=1).fit(
        X_t,
        y_t,
        time_index=scrambled,
        entity_ids=entity_t,
    )
    assert_allclose(
        hac.breusch_pagan_lm_test().statistic,
        bp_actual.statistic,
        rtol=5e-9,
        atol=5e-10,
    )


def test_stage_b_panel_fe_torch_cpu_pooling_and_fit_stats_match_numpy():
    X, y, entity, time = _panel(seed=1221)
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)

    expected = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)
    actual = PanelOLS(entity_effects=True).fit(X_t, y_t, entity_ids=entity_t)
    assert_allclose(actual.coef_, expected.coef_, rtol=1e-10, atol=1e-11)
    _assert_fit_statistics_close(actual.fit_statistics_, expected.fit_statistics_)
    pool_expected = expected.pooling_f_test()
    pool_actual = actual.pooling_f_test()
    assert pool_actual.applicable == pool_expected.applicable
    assert_allclose(pool_actual.statistic, pool_expected.statistic, rtol=5e-9, atol=5e-10)
    assert_allclose(pool_actual.pvalue, pool_expected.pvalue, rtol=1e-8, atol=1e-14)
    assert pool_actual.df == pool_expected.df
    # Existing Stage-A df remains the compatibility value on both backends.
    assert actual.df_resid == expected.df_resid
    assert actual.fit_statistics_.metadata["diagnostic_df"] == expected.fit_statistics_.metadata["diagnostic_df"]


def test_stage_b_random_effects_torch_cpu_fit_stats_and_identity_match_numpy():
    X, y, entity, time = _panel(seed=1222)
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)

    expected = RandomEffects().fit(X, y, entity_ids=entity)
    actual = RandomEffects().fit(X_t, y_t, entity_ids=entity_t)
    assert_allclose(actual.coef_, expected.coef_, rtol=1e-9, atol=1e-10)
    _assert_fit_statistics_close(actual.fit_statistics_, expected.fit_statistics_)

    fe_expected = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)
    fe_actual = PanelOLS(entity_effects=True).fit(X_t, y_t, entity_ids=entity_t)
    h_expected = fe_expected.hausman_test(expected)
    h_actual = fe_actual.hausman_test(actual)
    assert h_actual.applicable == h_expected.applicable
    if h_expected.applicable:
        assert_allclose(h_actual.statistic, h_expected.statistic, rtol=1e-7, atol=1e-9)
        assert_allclose(h_actual.pvalue, h_expected.pvalue, rtol=1e-7, atol=1e-12)
        assert h_actual.df == h_expected.df
    else:
        # A finite-sample covariance difference may be indefinite. The important
        # parity contract is that this is reported structurally on both backends,
        # not converted into a solve error or a silent fallback.
        assert h_actual.reason == h_expected.reason


def test_stage_b_random_effects_explicit_constant_torch_cpu_matches_numpy():
    rng = np.random.default_rng(1224)
    counts = np.asarray([5, 4, 3, 5, 4, 3])
    entity = np.repeat(np.arange(counts.size), counts)
    time = np.concatenate([np.arange(count) for count in counts])
    slope = rng.normal(size=entity.size)
    X = np.column_stack([np.ones(entity.size), slope])
    alpha = np.repeat(np.linspace(-0.45, 0.55, counts.size), counts)
    y = 1.4 + 0.72 * slope + alpha + rng.normal(scale=0.14, size=entity.size)
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)

    expected = RandomEffects().fit(X, y, entity_ids=entity)
    actual = RandomEffects().fit(X_t, y_t, entity_ids=entity_t)

    assert_allclose(actual.coef_, expected.coef_, rtol=1e-9, atol=1e-10)
    _assert_fit_statistics_close(actual.fit_statistics_, expected.fit_statistics_)
    assert actual.fit_statistics_.metadata["has_explicit_constant"] is True
    assert actual.fit_statistics_.metadata["constant_column_index"] == 0
    assert actual.fit_statistics_.metadata["model_f"]["rank_restricted"] == 1
    assert actual.fit_statistics_.metadata["model_f"]["restricted_design_supplied"] is True
    assert (
        actual._panel_diagnostic_identity["fingerprint"]["content_digest"]
        == expected._panel_diagnostic_identity["fingerprint"]["content_digest"]
    )


def test_stage_b_torch_constant_detection_is_scale_invariant():
    slope = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    for scale in (1.0, 1e-8, 1e-12):
        X = torch.stack(
            [torch.full_like(slope, scale), slope],
            dim=1,
        )
        assert explicit_constant_column(X, xp=torch) == 0


def test_stage_b_fama_macbeth_torch_cpu_r2_matches_numpy_without_ols_f():
    X, y, entity, time = _panel(seed=1223)
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)

    expected = FamaMacBeth(cov_type="newey-west", bandwidth=1).fit(
        X, y, time_ids=time, entity_ids=entity
    )
    actual = FamaMacBeth(cov_type="newey-west", bandwidth=1).fit(
        X_t, y_t, time_ids=time_t, entity_ids=entity_t
    )
    assert_allclose(
        actual.coef_.detach().cpu().numpy(),
        np.asarray(expected.coef_),
        rtol=1e-9,
        atol=1e-10,
    )
    _assert_fit_statistics_close(
        actual.fit_statistics_, expected.fit_statistics_, include_adjusted=False
    )
    assert actual.fit_statistics_.rsquared_adj is None
    assert actual.fit_statistics_.f_statistic is None
    assert actual.fit_statistics_.f_pvalue is None
    assert actual.fit_statistics_.f_df is None
