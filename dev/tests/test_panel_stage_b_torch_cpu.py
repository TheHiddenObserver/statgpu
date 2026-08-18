"""Hosted Torch-CPU parity coverage for Panel Tier-1 Stage B diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import BetweenOLS, FamaMacBeth, FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects
from statgpu.panel._covariance import (
    _grouped_score_sums,
    clustered_covariance,
    driscoll_kraay_covariance,
    hac_covariance,
    two_way_clustered_covariance,
)
from statgpu.panel._diagnostic_context import explicit_constant_column
from statgpu.panel._diagnostics import (
    _diagnostic_identity,
    _fingerprints_match,
    _hausman_quadratic,
)
from statgpu.panel._utils import _zero_safe_statistic_ratio


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
    assert actual.df_resid == expected.df_resid
    assert actual.fit_statistics_.metadata["diagnostic_df"] == expected.fit_statistics_.metadata["diagnostic_df"]


def test_stage_b_disconnected_two_way_fe_torch_cpu_uses_component_df():
    entity = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int64)
    time = np.asarray([0, 1, 0, 1, 2, 3, 2, 3, 4], dtype=np.int64)
    X = np.asarray([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0]).reshape(-1, 1)
    y = np.asarray([1.0, -1.0, -1.0, 1.0, 2.0, -2.0, -2.0, 2.0, 0.0])
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)

    expected = PanelOLS(entity_effects=True, time_effects=True).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    actual = PanelOLS(entity_effects=True, time_effects=True).fit(
        X_t, y_t, entity_ids=entity_t, time_ids=time_t
    )

    assert_allclose(actual.coef_, expected.coef_, rtol=1e-10, atol=1e-11)
    assert_allclose(actual.bse_, expected.bse_, rtol=1e-10, atol=1e-11)
    assert_allclose(actual.tvalues_, expected.tvalues_, rtol=1e-10, atol=1e-11)
    assert_allclose(actual.pvalues_, expected.pvalues_, rtol=1e-10, atol=1e-12)
    assert_allclose(actual.conf_int_, expected.conf_int_, rtol=1e-10, atol=1e-10)
    assert actual.df_resid == expected.df_resid == 1
    assert actual.fit_statistics_.metadata["legacy_df_resid"] == 0
    assert actual.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
    assert actual.fit_statistics_.metadata["diagnostic_df"] == expected.fit_statistics_.metadata["diagnostic_df"]
    assert actual.fit_statistics_.metadata["diagnostic_df"]["incidence_components"] == 3


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


def test_stage_b_absorbed_intercept_identity_matches_on_torch_cpu():
    rng = np.random.default_rng(1225)
    entity = np.repeat(np.arange(5), 4)
    slopes = rng.normal(size=(entity.size, 2))
    y = slopes @ np.asarray([0.55, -0.2]) + rng.normal(scale=0.1, size=entity.size)
    slopes_t = torch.as_tensor(slopes, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    entity_t = torch.as_tensor(entity, dtype=torch.int64)
    re_t = torch.cat(
        [torch.ones((entity.size, 1), dtype=torch.float64), slopes_t], dim=1
    )

    fe_identity = _diagnostic_identity(
        slopes_t,
        y_t,
        xp=torch,
        entity_codes=entity_t,
        has_constant=False,
    )
    re_identity = _diagnostic_identity(
        re_t,
        y_t,
        xp=torch,
        entity_codes=entity_t,
        has_constant=True,
    )
    matched, reason = _fingerprints_match(fe_identity, re_identity)

    assert matched, reason
    assert fe_identity["coefficient_indices"] == (0, 1)
    assert re_identity["coefficient_indices"] == (1, 2)
    assert fe_identity["feature_names"] == re_identity["feature_names"] == ("x1", "x2")


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

def test_stage_c_torch_cpu_zero_variance_and_response_scale_equivariance():
    tiny = np.nextafter(0.0, 1.0)
    params = torch.tensor([0.0, tiny, -tiny], dtype=torch.float64)
    bse = torch.zeros(3, dtype=torch.float64)
    statistic = _zero_safe_statistic_ratio(params, bse, torch).detach().cpu().numpy()
    assert statistic[0] == 0.0
    assert np.isposinf(statistic[1])
    assert np.isneginf(statistic[2])

    rng = np.random.default_rng(1230)
    n_entities, n_times = 10, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    x = rng.normal(size=entity.size)
    X = x[:, None]
    alpha = np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)
    y = 0.65 * x + alpha + rng.normal(scale=0.16, size=entity.size)
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)
    response_scale = 1.0e-20

    reference_between = BetweenOLS(cov_type="hc0").fit(X_t, y_t, entity_ids=entity_t)
    scaled_between = BetweenOLS(cov_type="hc0").fit(
        X_t, response_scale * y_t, entity_ids=entity_t
    )
    reference_fd = FirstDifferenceOLS(cov_type="hc0").fit(
        X_t, y_t, entity_ids=entity_t, time_ids=time_t
    )
    scaled_fd = FirstDifferenceOLS(cov_type="hc0").fit(
        X_t, response_scale * y_t, entity_ids=entity_t, time_ids=time_t
    )
    for reference, scaled in (
        (reference_between, scaled_between),
        (reference_fd, scaled_fd),
    ):
        assert_allclose(scaled.coef_, response_scale * reference.coef_, rtol=2e-9, atol=0.0)
        assert_allclose(scaled.bse_, response_scale * reference.bse_, rtol=2e-8, atol=0.0)
        assert_allclose(scaled.tvalues_, reference.tvalues_, rtol=2e-8, atol=2e-11)
        assert_allclose(scaled.pvalues_, reference.pvalues_, rtol=2e-8, atol=2e-13)


def test_stage_c_torch_cpu_extreme_covariance_combinations_remain_finite():
    amplitude = 1.4e154
    X = torch.ones((4, 1), dtype=torch.float64)
    resid = torch.tensor([amplitude, amplitude, -amplitude, -amplitude], dtype=torch.float64)
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    expected = np.asarray([[0.5 * amplitude * amplitude]], dtype=np.float64)
    one_way = clustered_covariance(X, resid, groups, xp=torch)
    two_way = two_way_clustered_covariance(X, resid, groups, groups, xp=torch)
    assert_allclose(one_way, expected, rtol=5e-14, atol=0.0)
    assert_allclose(two_way, expected, rtol=6e-14, atol=0.0)

    cancel_groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    score_amplitude = 1.6e308
    scores = torch.tensor(
        [[score_amplitude], [score_amplitude], [-score_amplitude], [-score_amplitude], [1.0], [-1.0]],
        dtype=torch.float64,
    )
    grouped = _grouped_score_sums(scores, cancel_groups, n_groups=2, xp=torch)
    assert_allclose(grouped, np.zeros((2, 1)), rtol=0.0, atol=0.0)

    n = 16
    influence_amplitude = 3.0e153
    X_hac = torch.ones((n, 1), dtype=torch.float64)
    signs = torch.where(
        torch.arange(n) % 2 == 0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(-1.0, dtype=torch.float64),
    )
    resid_hac = (n * influence_amplitude) * signs
    time = np.arange(n, dtype=np.int64)
    expected_hac = np.asarray([[influence_amplitude ** 2]], dtype=np.float64)
    assert_allclose(
        hac_covariance(X_hac, resid_hac, bandwidth=1, xp=torch),
        expected_hac,
        rtol=8e-14,
        atol=0.0,
    )
    assert_allclose(
        driscoll_kraay_covariance(X_hac, resid_hac, time, bandwidth=1, xp=torch),
        expected_hac * (n / (n - 1.0)),
        rtol=8e-14,
        atol=0.0,
    )


def test_stage_c_torch_cpu_lag_accumulator_preserves_finite_hac_and_dk():
    n = 7
    bandwidth = 4
    influence_sq = 2.0e307
    influence_amp = float(np.sqrt(influence_sq))
    signs_np = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    X = torch.ones((n, 1), dtype=torch.float64)
    resid = torch.as_tensor(n * influence_amp * signs_np, dtype=torch.float64)
    time = np.arange(n, dtype=np.int64)
    expected_hac = np.asarray([[5.4 * influence_sq]], dtype=np.float64)
    assert_allclose(
        hac_covariance(X, resid, bandwidth=bandwidth, xp=torch),
        expected_hac,
        rtol=1.2e-13,
        atol=0.0,
    )
    assert_allclose(
        driscoll_kraay_covariance(
            X, resid, time, bandwidth=bandwidth, kernel="bartlett", xp=torch
        ),
        expected_hac * (n / (n - 1.0)),
        rtol=1.5e-13,
        atol=0.0,
    )



def test_stage_c_torch_cpu_pregram_and_two_way_component_cancellation():
    n = 10
    influence_sq = 1.0e308
    influence_amp = float(np.sqrt(influence_sq))
    signs_np = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    X = torch.ones((n, 1), dtype=torch.float64)
    resid = torch.as_tensor(n * influence_amp * signs_np, dtype=torch.float64)
    time = np.arange(n, dtype=np.int64)
    expected_hac = np.asarray([[influence_sq]], dtype=np.float64)
    assert_allclose(
        hac_covariance(X, resid, bandwidth=1, xp=torch),
        expected_hac,
        rtol=2.0e-13,
        atol=0.0,
    )
    assert_allclose(
        driscoll_kraay_covariance(X, resid, time, bandwidth=1, xp=torch),
        expected_hac * (n / (n - 1.0)),
        rtol=2.5e-13,
        atol=0.0,
    )

    n2 = 4
    component_sq = 5.0e307
    component_amp = float(np.sqrt(component_sq))
    X2 = torch.ones((n2, 1), dtype=torch.float64)
    resid2 = torch.as_tensor(
        n2 * component_amp * np.asarray([1.0, -1.0, 1.0, -1.0]),
        dtype=torch.float64,
    )
    unique = np.arange(n2, dtype=np.int64)
    pairs = np.asarray([0, 0, 1, 1], dtype=np.int64)
    reference = clustered_covariance(X2, resid2, pairs, xp=torch)
    actual = two_way_clustered_covariance(X2, resid2, unique, pairs, xp=torch)
    assert_allclose(actual, reference, rtol=3.0e-13, atol=0.0)



def test_stage_c_torch_cpu_delays_tiny_design_scale_until_group_cancellation():
    tiny = 1.0e-320
    X = torch.ones((4, 1), dtype=torch.float64) * tiny
    resid = torch.as_tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    actual = clustered_covariance(X, resid, groups, xp=torch)
    assert_allclose(actual, np.zeros((1, 1)), rtol=0.0, atol=0.0)



def test_stage_c_torch_cpu_preserves_mixed_range_cluster_and_dk_scores():
    X = torch.ones((3, 1), dtype=torch.float64)
    resid = torch.as_tensor([1.5e308, -1.5e308, 3.0e-100], dtype=torch.float64)
    coarse = np.asarray([0, 0, 1], dtype=np.int64)
    unique = np.asarray([0, 1, 2], dtype=np.int64)
    time = np.asarray([0, 0, 1], dtype=np.int64)
    clustered = clustered_covariance(X, resid, coarse, xp=torch)
    two_way = two_way_clustered_covariance(X, resid, unique, coarse, xp=torch)
    dk = driscoll_kraay_covariance(X, resid, time, bandwidth=0, xp=torch)
    assert_allclose(clustered, np.asarray([[1.0e-200]]), rtol=5.0e-13, atol=0.0)
    assert_allclose(two_way, clustered, rtol=5.0e-13, atol=0.0)
    assert_allclose(dk, np.asarray([[1.5e-200]]), rtol=6.0e-13, atol=0.0)



def test_stage_c_torch_cpu_nested_partition_code_permutation_is_exact():
    X = torch.ones((3, 1), dtype=torch.float64)
    resid = torch.as_tensor([1.5e308, -1.5e308, 3.0e-100], dtype=torch.float64)
    coarse = np.asarray([1, 1, 0], dtype=np.int64)
    fine = np.asarray([0, 1, 2], dtype=np.int64)
    reference = clustered_covariance(X, resid, coarse, xp=torch)
    actual = two_way_clustered_covariance(X, resid, coarse, fine, xp=torch)
    assert_allclose(actual, reference, rtol=5e-13, atol=0.0)


def test_stage_b_torch_cpu_hausman_host_quadratic_is_scale_safe():
    # Hausman forms the small covariance-difference quadratic on host after the
    # backend-specific FE/RE fits.  Keep both floating-point scale extremes in
    # the maintained Torch job so GPU-originated results cannot regress here.
    for variance, difference in (
        (1.0e308, np.sqrt(1.0e308)),
        (1.0e-320, np.sqrt(1.0e-320)),
    ):
        result = _hausman_quadratic([difference], [[variance]])
        assert result.applicable, result.reason
        assert_allclose(result.statistic, 1.0, rtol=3e-12, atol=0.0)


def test_stage_c_torch_cpu_two_way_nonnested_structural_cancellation():
    n = 4
    amplitude = 1.0e154
    small = 1.0e-154
    X = torch.full((n, 1), 0.5, dtype=torch.float64)
    target_scores = torch.tensor(
        [-amplitude, small, amplitude, -small], dtype=torch.float64
    )
    resid = 2.0 * target_scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(
        X, resid, cluster1, cluster2, xp=torch
    ).detach().cpu().numpy()
    np.testing.assert_allclose(
        actual, np.asarray([[-4.0 * amplitude * small]]), rtol=2e-12, atol=0.0
    )
