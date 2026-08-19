"""Focused Stage-C covariance correctness and compatibility contracts."""

from __future__ import annotations

import numpy as np
import pytest

import statgpu.panel._covariance as covariance_module
from numpy.testing import assert_allclose

from statgpu.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from statgpu.panel._covariance import (
    _dk_kernel_weights,
    _grouped_score_sums,
    _influence_rows,
    _symmetrize,
    clustered_covariance,
    driscoll_kraay_covariance,
    hac_covariance,
    normalize_covariance_type,
    ols_covariance,
    two_way_clustered_covariance,
)


def _panel(seed=12600, *, unbalanced=False):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 10, 7
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.45, size=n_entities), n_times)
    tau = np.tile(np.linspace(-0.2, 0.25, n_times), n_entities)
    y = 0.75 * X[:, 0] - 0.35 * X[:, 1] + alpha + 0.3 * tau
    y += rng.normal(scale=0.25, size=entity.size)
    if unbalanced:
        keep = np.ones(entity.size, dtype=bool)
        keep[[1, 9, 18, 32, 51]] = False
        X, y, entity, time = X[keep], y[keep], entity[keep], time[keep]
    return X, y, entity, time


def _manual_hc(X, resid, power):
    bread = np.linalg.pinv(X.T @ X)
    h = np.sum((X @ bread) * X, axis=1)
    scale = (1.0 - h) ** (-power)
    meat = X.T @ (X * (resid * resid * scale)[:, None])
    return bread @ meat @ bread


def _manual_dk(X, resid, time, bandwidth, extra_df=0):
    bread = np.linalg.pinv(X.T @ X)
    unique = np.unique(time)
    scores = X * resid[:, None]
    grouped = np.stack([scores[time == value].sum(axis=0) for value in unique])
    meat = grouped.T @ grouped
    for lag in range(1, min(bandwidth, len(unique) - 1) + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = grouped[lag:].T @ grouped[:-lag]
        meat += weight * (gamma + gamma.T)
    rank = np.linalg.matrix_rank(X)
    denom = X.shape[0] - extra_df - rank
    return (X.shape[0] / denom) * bread @ meat @ bread


def test_covariance_aliases_are_canonical_and_backward_compatible():
    assert normalize_covariance_type("HC1") == "robust"
    assert normalize_covariance_type("dk") == "driscoll-kraay"
    assert normalize_covariance_type("kernel") == "driscoll-kraay"
    assert normalize_covariance_type("robust") == "robust"


def test_hc0_hc2_hc3_match_direct_fit_space_sandwich():
    rng = np.random.default_rng(12601)
    X = np.column_stack([np.ones(30), rng.normal(size=(30, 2))])
    y = X @ np.array([0.3, 0.8, -0.4]) + rng.normal(scale=0.2, size=30)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta

    hc0 = ols_covariance(X, resid, cov_type="hc0")
    bread = np.linalg.pinv(X.T @ X)
    expected0 = bread @ (X.T @ (X * (resid * resid)[:, None])) @ bread
    assert_allclose(hc0, expected0, rtol=2e-12, atol=2e-14)

    hc2 = ols_covariance(X, resid, cov_type="hc2")
    hc3 = ols_covariance(X, resid, cov_type="hc3")
    assert_allclose(hc2, _manual_hc(X, resid, 1), rtol=2e-12, atol=2e-14)
    assert_allclose(hc3, _manual_hc(X, resid, 2), rtol=2e-12, atol=2e-14)


def test_hc2_hc3_reject_unit_leverage_instead_of_clipping():
    X = np.eye(4)
    resid = np.ones(4)
    with pytest.raises(ValueError, match="leverage is numerically one"):
        ols_covariance(X, resid, cov_type="hc2")
    with pytest.raises(ValueError, match="leverage is numerically one"):
        ols_covariance(X, resid, cov_type="hc3")


def test_clustered_covariance_rejects_single_cluster():
    rng = np.random.default_rng(126020)
    X = np.column_stack([np.ones(18), rng.normal(size=18)])
    resid = rng.normal(size=18)
    single_cluster = np.zeros(18, dtype=np.int64)
    with pytest.raises(ValueError, match=r"at least two distinct clusters"):
        clustered_covariance(X, resid, single_cluster)


def test_clustered_estimators_reject_single_cluster_inference():
    X, y, entity, _time = _panel(seed=126021, unbalanced=True)
    single_cluster = np.zeros(len(y), dtype=np.int64)

    with pytest.raises(ValueError, match=r"at least two distinct clusters"):
        PooledOLS(cov_type="clustered").fit(X, y, cluster=single_cluster)

    with pytest.raises(ValueError, match=r"at least two distinct clusters"):
        PanelOLS(cov_type="clustered").fit(X, y, cluster=single_cluster)

    X_re = np.column_stack([np.ones(len(y)), X])
    with pytest.raises(ValueError, match=r"at least two distinct clusters"):
        RandomEffects(cov_type="clustered").fit(
            X_re, y, entity_ids=entity, cluster=single_cluster
        )


def test_two_way_clustered_rejects_single_marginal_cluster():
    rng = np.random.default_rng(126022)
    X = np.column_stack([np.ones(24), rng.normal(size=24)])
    resid = rng.normal(size=24)
    single = np.zeros(24, dtype=np.int64)
    second = np.tile(np.arange(6), 4)
    with pytest.raises(ValueError, match=r"at least two distinct clusters"):
        two_way_clustered_covariance(X, resid, single, second)


def test_one_way_group_debias_is_exact_multiplicative_factor():
    rng = np.random.default_rng(12602)
    X = np.column_stack([np.ones(24), rng.normal(size=24)])
    resid = rng.normal(size=24)
    cluster = np.repeat(np.arange(6), 4)
    raw = clustered_covariance(X, resid, cluster)
    corrected = clustered_covariance(X, resid, cluster, group_debias=True)
    factor = (6.0 / 5.0) * (23.0 / 24.0)
    assert_allclose(corrected, raw * factor, rtol=2e-13, atol=2e-15)


def test_two_way_cluster_matches_inclusion_exclusion_with_exact_pairs():
    rng = np.random.default_rng(12603)
    X = np.column_stack([np.ones(30), rng.normal(size=30)])
    resid = rng.normal(size=30)
    c1 = np.repeat(np.arange(5), 6)
    c2 = np.tile(np.arange(6), 5)
    pair_codes = np.unique(np.column_stack([c1, c2]), axis=0, return_inverse=True)[1]
    expected = (
        clustered_covariance(X, resid, c1)
        + clustered_covariance(X, resid, c2)
        - clustered_covariance(X, resid, pair_codes)
    )
    actual = two_way_clustered_covariance(X, resid, c1, c2)
    assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)


def test_qs_kernel_uses_all_observed_lags_not_bandwidth_cutoff():
    name, weights = _dk_kernel_weights("qs", bandwidth=2, max_lag=8)
    assert name == "qs"
    assert weights.shape == (9,)
    assert weights[0] == 1.0
    assert np.count_nonzero(weights[1:]) == 8
    assert weights[5] != 0.0

    _, bartlett = _dk_kernel_weights("bartlett", bandwidth=2, max_lag=8)
    assert np.count_nonzero(bartlett[1:]) == 2
    assert np.all(bartlett[3:] == 0.0)


def test_driscoll_kraay_bartlett_matches_direct_grouped_score_formula():
    rng = np.random.default_rng(12604)
    entity = np.repeat(np.arange(8), 6)
    time = np.tile(np.arange(6), 8)
    X = np.column_stack([np.ones(entity.size), rng.normal(size=(entity.size, 2))])
    y = X @ np.array([0.2, 0.6, -0.25]) + rng.normal(scale=0.25, size=entity.size)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    actual = driscoll_kraay_covariance(
        X, resid, time, bandwidth=2, kernel="bartlett", extra_df=0
    )
    expected = _manual_dk(X, resid, time, bandwidth=2)
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)


def test_driscoll_kraay_qs_metadata_records_all_lag_support():
    rng = np.random.default_rng(12605)
    time = np.tile(np.arange(9), 5)
    X = np.column_stack([np.ones(time.size), rng.normal(size=time.size)])
    resid = rng.normal(size=time.size)
    metadata = {}
    cov = driscoll_kraay_covariance(
        X,
        resid,
        time,
        bandwidth=2,
        kernel="quadratic-spectral",
        metadata=metadata,
    )
    assert np.all(np.isfinite(cov))
    assert metadata["kernel"] == "qs"
    assert metadata["bandwidth"] == 2
    assert metadata["max_weighted_lag"] == 8
    assert metadata["all_observed_lags_weighted"] is True


def test_legacy_hac_bartlett_formula_is_unchanged():
    rng = np.random.default_rng(12606)
    X = np.column_stack([np.ones(25), rng.normal(size=25)])
    resid = rng.normal(size=25)
    direct = hac_covariance(X, resid, bandwidth=3, kernel="bartlett")
    dispatched = ols_covariance(
        X, resid, cov_type="hac", bandwidth=3, kernel="bartlett"
    )
    assert_allclose(dispatched, direct, rtol=0, atol=0)


def test_pooled_robust_and_hc1_alias_are_identical():
    X, y, entity, time = _panel(seed=12607)
    robust = PooledOLS(cov_type="robust").fit(X, y, entity_ids=entity)
    hc1 = PooledOLS(cov_type="hc1").fit(X, y, entity_ids=entity)
    assert_allclose(hc1.coef_, robust.coef_, rtol=0, atol=0)
    assert_allclose(hc1.bse_, robust.bse_, rtol=0, atol=0)
    assert_allclose(hc1.pvalues_, robust.pvalues_, rtol=0, atol=0)
    assert hc1._cov_type == "robust"


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_pooled_stage_c_hc_changes_inference_not_coefficients(cov_type):
    X, y, entity, time = _panel(seed=12608)
    base = PooledOLS().fit(X, y, entity_ids=entity)
    model = PooledOLS(cov_type=cov_type).fit(X, y, entity_ids=entity)
    assert_allclose(model.coef_, base.coef_, rtol=2e-13, atol=2e-14)
    assert np.all(np.isfinite(model.bse_))
    assert model._covariance_metadata["covariance"] == cov_type


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_panel_stage_c_hc_changes_inference_not_coefficients(cov_type):
    X, y, entity, time = _panel(seed=12609, unbalanced=True)
    base = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)
    model = PanelOLS(entity_effects=True, cov_type=cov_type).fit(
        X, y, entity_ids=entity
    )
    assert_allclose(model.coef_, base.coef_, rtol=2e-12, atol=2e-13)
    assert np.all(np.isfinite(model.bse_))


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_between_stage_c_hc_changes_inference_not_coefficients(cov_type):
    X, y, entity, time = _panel(seed=12610, unbalanced=True)
    base = BetweenOLS().fit(X, y, entity_ids=entity)
    model = BetweenOLS(cov_type=cov_type).fit(X, y, entity_ids=entity)
    assert_allclose(model.coef_, base.coef_, rtol=2e-12, atol=2e-13)
    assert np.all(np.isfinite(model.bse_))


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_first_difference_stage_c_hc_changes_inference_not_coefficients(cov_type):
    X, y, entity, time = _panel(seed=12611)
    base = FirstDifferenceOLS().fit(X, y, entity_ids=entity, time_ids=time)
    model = FirstDifferenceOLS(cov_type=cov_type).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert_allclose(model.coef_, base.coef_, rtol=2e-12, atol=2e-13)
    assert np.all(np.isfinite(model.bse_))


@pytest.mark.parametrize("cov_type", ["robust", "hc0", "hc2", "hc3"])
def test_random_effects_stage_c_hc_does_not_change_swamy_arora_estimate(cov_type):
    X, y, entity, time = _panel(seed=12612, unbalanced=True)
    X = np.column_stack([np.ones(len(y)), X])
    base = RandomEffects().fit(X, y, entity_ids=entity)
    model = RandomEffects(cov_type=cov_type).fit(X, y, entity_ids=entity)
    assert_allclose(model.coef_, base.coef_, rtol=2e-12, atol=2e-13)
    assert_allclose(
        [model.variance_components_["sigma2_e"], model.variance_components_["sigma2_a"]],
        [base.variance_components_["sigma2_e"], base.variance_components_["sigma2_a"]],
        rtol=2e-13,
        atol=2e-14,
    )
    assert np.all(np.isfinite(model.bse_))


def test_pooled_and_random_effects_driscoll_kraay_execute_with_time_metadata():
    X, y, entity, time = _panel(seed=12613, unbalanced=True)
    pooled = PooledOLS(cov_type="dk", bandwidth=2).fit(
        X, y, entity_ids=entity, time_index=time
    )
    re = RandomEffects(cov_type="driscoll-kraay", bandwidth=2).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert pooled._covariance_metadata["covariance"] == "driscoll-kraay"
    assert re._covariance_metadata["covariance"] == "driscoll-kraay"
    assert pooled._covariance_metadata["n_periods"] == len(np.unique(time))
    assert re._covariance_metadata["n_periods"] == len(np.unique(time))
    assert np.all(np.isfinite(pooled.bse_))
    assert np.all(np.isfinite(re.bse_))


def test_panel_driscoll_kraay_records_standard_effect_rank_extra_df():
    X, y, entity, time = _panel(seed=12614, unbalanced=True)
    model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="driscoll-kraay",
        bandwidth=2,
    ).fit(X, y, entity_ids=entity, time_ids=time)
    expected = model.fit_statistics_.metadata["diagnostic_df"]["effect_rank"]
    assert model._covariance_metadata["extra_df"] == expected
    assert np.all(np.isfinite(model.bse_))


def test_cluster_group_debias_is_opt_in_and_coefficients_unchanged():
    X, y, entity, time = _panel(seed=12615)
    clusters = np.column_stack([entity, time])
    base = PooledOLS(cov_type="clustered").fit(X, y, cluster=clusters)
    corrected = PooledOLS(cov_type="clustered", group_debias=True).fit(
        X, y, cluster=clusters
    )
    assert_allclose(corrected.coef_, base.coef_, rtol=0, atol=0)
    assert corrected._covariance_metadata["group_debias"] is True
    assert corrected._covariance_metadata["cluster_dimensions"] == 2
    assert corrected._covariance_metadata["cluster_group_counts"][:2] == [10, 7]


def test_group_debias_rejected_for_noncluster_covariance():
    X, y, entity, time = _panel(seed=12616)
    with pytest.raises(ValueError, match="requires cov_type='clustered'"):
        PooledOLS(cov_type="hc2", group_debias=True).fit(X, y)
    with pytest.raises(ValueError, match="requires cov_type='clustered'"):
        RandomEffects(cov_type="hc2", group_debias=True).fit(
            X, y, entity_ids=entity
        )


def test_driscoll_kraay_requires_time_metadata():
    X, y, entity, time = _panel(seed=12617)
    with pytest.raises(ValueError, match="time_index is required"):
        PooledOLS(cov_type="dk").fit(X, y)
    with pytest.raises(ValueError, match="time_ids is required"):
        PanelOLS(entity_effects=True, cov_type="dk").fit(
            X, y, entity_ids=entity
        )
    with pytest.raises(ValueError, match="time_ids is required"):
        RandomEffects(cov_type="dk").fit(X, y, entity_ids=entity)


def test_between_and_first_difference_explicitly_reject_dk():
    with pytest.raises(ValueError, match="cov_type"):
        BetweenOLS(cov_type="dk")
    with pytest.raises(ValueError, match="cov_type"):
        FirstDifferenceOLS(cov_type="dk")


def test_covariance_symmetrization_preserves_huge_and_subnormal_finite_entries():
    huge = 1.0e308
    tiny = np.nextafter(0.0, 1.0)
    matrix = np.asarray([[huge, tiny], [tiny, -huge]], dtype=np.float64)
    actual = _symmetrize(matrix)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, matrix)


def test_cluster_and_two_way_inclusion_exclusion_preserve_huge_finite_covariance():
    X = np.ones((4, 1), dtype=np.float64)
    amplitude = 1.4e154
    resid = np.asarray([amplitude, amplitude, -amplitude, -amplitude])
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    expected = np.asarray([[0.5 * amplitude * amplitude]], dtype=np.float64)

    one_way = clustered_covariance(X, resid, groups)
    two_way = two_way_clustered_covariance(X, resid, groups, groups)
    assert np.all(np.isfinite(one_way))
    assert np.all(np.isfinite(two_way))
    np.testing.assert_allclose(one_way, expected, rtol=4e-15, atol=0.0)
    np.testing.assert_allclose(two_way, expected, rtol=5e-15, atol=0.0)


def test_grouped_score_reduction_survives_finite_partial_sum_overflow():
    amplitude = 1.6e308
    scores = np.asarray(
        [[amplitude], [amplitude], [-amplitude], [-amplitude], [1.0], [-1.0]],
        dtype=np.float64,
    )
    groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)
    grouped = _grouped_score_sums(scores, groups, n_groups=2, xp=np)
    assert np.all(np.isfinite(grouped))
    np.testing.assert_array_equal(grouped, np.zeros((2, 1)))


def test_hac_and_dk_weighted_lags_do_not_overflow_before_finite_cancellation():
    n = 16
    influence_amplitude = 3.0e153
    X = np.ones((n, 1), dtype=np.float64)
    resid = n * influence_amplitude * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    time = np.arange(n, dtype=np.int64)
    expected_hac = np.asarray([[influence_amplitude ** 2]], dtype=np.float64)
    expected_dk = expected_hac * (n / (n - 1.0))

    hac = hac_covariance(X, resid, bandwidth=1)
    dk = driscoll_kraay_covariance(X, resid, time, bandwidth=1)
    assert np.all(np.isfinite(hac))
    assert np.all(np.isfinite(dk))
    np.testing.assert_allclose(hac, expected_hac, rtol=8e-15, atol=0.0)
    np.testing.assert_allclose(dk, expected_dk, rtol=8e-15, atol=0.0)


def test_hac_and_dk_lag_accumulator_survives_finite_final_after_overflowing_partial_sum():
    n = 7
    bandwidth = 4
    influence_sq = 2.0e307
    influence_amp = float(np.sqrt(influence_sq))
    signs = np.asarray([1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    X = np.ones((n, 1), dtype=np.float64)
    resid = n * influence_amp * signs
    time = np.arange(n, dtype=np.int64)

    # In exact arithmetic the Bartlett meat coefficients are
    # 7 + 3.2 - 3.6 - 1.6 + 0.4 = 5.4. The historical sequential path
    # overflowed after 7 + 3.2 = 10.2 even though the final result is finite.
    expected_hac = np.asarray([[5.4 * influence_sq]], dtype=np.float64)
    expected_dk = expected_hac * (n / (n - 1.0))
    hac = hac_covariance(X, resid, bandwidth=bandwidth)
    dk = driscoll_kraay_covariance(
        X, resid, time, bandwidth=bandwidth, kernel="bartlett"
    )
    assert np.all(np.isfinite(hac))
    assert np.all(np.isfinite(dk))
    np.testing.assert_allclose(hac, expected_hac, rtol=1.2e-14, atol=0.0)
    np.testing.assert_allclose(dk, expected_dk, rtol=1.5e-14, atol=0.0)



def test_hac_and_dk_normalize_scores_before_zero_lag_gram_overflow():
    n = 10
    influence_sq = 1.0e308
    influence_amp = float(np.sqrt(influence_sq))
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    X = np.ones((n, 1), dtype=np.float64)
    resid = n * influence_amp * signs
    time = np.arange(n, dtype=np.int64)

    # The zero-lag meat is n*a^2 and overflows, but Bartlett bandwidth=1
    # subtracts (n-1)*a^2 through the lag pair, leaving exactly a^2.
    expected_hac = np.asarray([[influence_sq]], dtype=np.float64)
    expected_dk = expected_hac * (n / (n - 1.0))
    hac = hac_covariance(X, resid, bandwidth=1)
    dk = driscoll_kraay_covariance(X, resid, time, bandwidth=1)
    assert np.all(np.isfinite(hac))
    assert np.all(np.isfinite(dk))
    np.testing.assert_allclose(hac, expected_hac, rtol=2.0e-14, atol=0.0)
    np.testing.assert_allclose(dk, expected_dk, rtol=2.5e-14, atol=0.0)


def test_two_way_cluster_combines_components_before_overflowing_restore():
    n = 4
    influence_sq = 5.0e307
    influence_amp = float(np.sqrt(influence_sq))
    X = np.ones((n, 1), dtype=np.float64)
    resid = n * influence_amp * np.asarray([1.0, -1.0, 1.0, -1.0])
    unique = np.arange(n, dtype=np.int64)
    pairs = np.asarray([0, 0, 1, 1], dtype=np.int64)

    # The unique-cluster component and its intersection component are each
    # 4*a^2 > DBL_MAX, but they are algebraically identical and cancel.  The
    # final two-way covariance is therefore the finite coarse-cluster component.
    reference = clustered_covariance(X, resid, pairs)
    actual = two_way_clustered_covariance(X, resid, unique, pairs)
    assert np.all(np.isfinite(reference))
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, reference, rtol=3.0e-14, atol=0.0)



def test_clustered_covariance_delays_tiny_design_scale_until_after_group_cancellation():
    tiny = 1.0e-320
    X = np.ones((4, 1), dtype=np.float64) * tiny
    resid = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)

    # The single-column design has condition number one.  Original-scale
    # observation influences exceed DBL_MAX, but each cluster score is exactly
    # zero, so the mathematically defined cluster covariance is zero.
    actual = clustered_covariance(X, resid, groups)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, np.zeros((1, 1)), rtol=0.0, atol=0.0)



def test_clustered_covariance_preserves_small_group_beside_huge_exact_cancellation():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    groups = np.asarray([0, 0, 1], dtype=np.int64)
    actual = clustered_covariance(X, resid, groups)
    expected = np.asarray([[1.0e-200]], dtype=np.float64)
    assert actual[0, 0] > 0.0
    np.testing.assert_allclose(actual, expected, rtol=3.0e-14, atol=0.0)


def test_two_way_nested_cluster_preserves_small_component_after_huge_cancellation():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    unique = np.asarray([0, 1, 2], dtype=np.int64)
    coarse = np.asarray([0, 0, 1], dtype=np.int64)
    reference = clustered_covariance(X, resid, coarse)
    actual = two_way_clustered_covariance(X, resid, unique, coarse)
    np.testing.assert_allclose(actual, reference, rtol=3.0e-14, atol=0.0)


def test_dk_groups_before_gram_scaling_preserves_small_period_score():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    time = np.asarray([0, 0, 1], dtype=np.int64)
    actual = driscoll_kraay_covariance(X, resid, time, bandwidth=0)
    expected = np.asarray([[1.5e-200]], dtype=np.float64)
    assert actual[0, 0] > 0.0
    np.testing.assert_allclose(actual, expected, rtol=4.0e-14, atol=0.0)



def test_two_way_nested_partition_detection_is_invariant_to_code_permutation():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    coarse = np.asarray([1, 1, 0], dtype=np.int64)
    fine = np.asarray([0, 1, 2], dtype=np.int64)

    # The fine partition equals the intersection partition, but paired-code
    # factorization orders (coarse, fine) lexicographically and therefore
    # produces a permutation of the fine integer codes. Statistical cancellation
    # is partition-based and must not depend on that arbitrary code numbering.
    reference = clustered_covariance(X, resid, coarse)
    actual = two_way_clustered_covariance(X, resid, coarse, fine)
    assert reference[0, 0] > 0.0
    np.testing.assert_allclose(reference, np.asarray([[1.0e-200]]), rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(actual, reference, rtol=3e-14, atol=0.0)


def test_two_way_nonnested_structural_cancellation_preserves_low_group_sum():
    n = 4
    amplitude = 1.0e154
    small = 1.0e-154
    X = np.full((n, 1), 0.5, dtype=np.float64)
    target_scores = np.asarray(
        [-amplitude, small, amplitude, -small], dtype=np.float64
    )
    resid = 2.0 * target_scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(X, resid, cluster1, cluster2)
    expected = np.asarray([[-4.0 * amplitude * small]], dtype=np.float64)
    assert_allclose(actual, expected, rtol=2e-12, atol=0.0)


def test_two_way_nonnested_low_order_correction_does_not_require_gram_scaling():
    amplitude = 1.0e150
    small = 1.0e-150
    X = np.full((4, 1), 0.5, dtype=np.float64)
    scores = np.asarray([-amplitude, small, amplitude, -small], dtype=np.float64)
    resid = 2.0 * scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(X, resid, cluster1, cluster2)
    assert_allclose(actual, np.asarray([[-4.0]]), rtol=5e-13, atol=0.0)


def test_two_way_group_debias_preserves_weighted_low_order_cancellation():
    amplitude = 1.0e154
    small = 1.0e-154
    X = np.full((4, 1), 0.5, dtype=np.float64)
    scores = np.asarray([-amplitude, amplitude, amplitude, small], dtype=np.float64)
    resid = 2.0 * scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(
        X, resid, cluster1, cluster2, group_debias=True
    )
    expected = np.asarray([[6.0 * amplitude * small + 2.0 * small * small]])
    assert_allclose(actual, expected, rtol=5e-13, atol=0.0)

def test_grouped_score_multiscale_cancellation_survives_three_levels():
    scores = np.asarray(
        [[1.0e154], [-1.0e154], [1.0e138], [1.0], [-1.0e138], [-1.0]],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 0, 1, 0, 0], dtype=np.int64)
    grouped = _grouped_score_sums(
        scores,
        cluster1,
        n_groups=2,
        xp=np,
    )
    np.testing.assert_array_equal(grouped, np.asarray([[-1.0], [1.0]]))

    # Use a binary-exact constant design: with n=8 and X=0.5, X'X=2,
    # bread=0.5 and every influence multiplier is exactly 0.25.  This keeps
    # the regression focused on grouped-score cancellation rather than SVD
    # representation error from a non-binary 1/6 design.
    deep_scores = np.asarray(
        [1.0e154, -1.0e154, 1.0e138, 1.0, -1.0e138, -1.0, 0.0, 0.0],
        dtype=np.float64,
    )
    deep_cluster1 = np.asarray([0, 0, 0, 1, 0, 0, 2, 3], dtype=np.int64)
    deep_cluster2 = np.asarray([0, 0, 0, 1, 0, 1, 2, 3], dtype=np.int64)
    X = np.full((8, 1), 0.5, dtype=np.float64)
    actual = two_way_clustered_covariance(
        X,
        4.0 * deep_scores,
        deep_cluster1,
        deep_cluster2,
    )
    np.testing.assert_allclose(actual, np.zeros((1, 1)), rtol=0.0, atol=0.0)


def test_one_way_and_dk_preserve_small_score_after_same_sign_swallow():
    scores = np.asarray([1.0e154, -1.0e154, 1.0, 1.0])
    groups = np.asarray([0, 0, 0, 1], dtype=np.int64)
    X = np.full((4, 1), 0.5, dtype=np.float64)
    resid = 2.0 * scores

    grouped = _grouped_score_sums(
        scores[:, None], groups, n_groups=2, xp=np
    )
    np.testing.assert_array_equal(
        grouped, np.asarray([[1.0], [1.0]])
    )
    one_way = clustered_covariance(X, resid, groups)
    dk = driscoll_kraay_covariance(
        X, resid, groups, bandwidth=0
    )
    np.testing.assert_allclose(
        one_way, np.asarray([[2.0]]), rtol=2e-15, atol=0.0
    )
    np.testing.assert_allclose(
        dk, np.asarray([[8.0 / 3.0]]), rtol=3e-15, atol=0.0
    )

def test_two_way_nonnested_unsafe_cross_cancels_before_restore():
    amplitude = 1.0e200
    low1 = 1.0e108
    low2 = low1 * (1.0 + 1.0e-3)
    scores = np.asarray(
        [
            -amplitude, low1, amplitude, -low1,
            -amplitude, -low2, amplitude, low2,
        ],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    X = np.full((8, 1), 0.5, dtype=np.float64)

    actual = two_way_clustered_covariance(
        X, 4.0 * scores, cluster1, cluster2
    )
    expected = np.asarray(
        [[4.0 * amplitude * (low2 - low1)]], dtype=np.float64
    )
    assert np.isfinite(expected[0, 0])
    assert_allclose(actual, expected, rtol=4e-12, atol=0.0)

def test_two_way_nonnested_preserves_third_magnitude_component():
    amplitude = 2.0 ** 660
    middle = 2.0 ** 600
    tiny = 2.0 ** 350
    scores = np.asarray(
        [
            -amplitude, middle, tiny, amplitude, -middle, -tiny,
            -amplitude, -middle, amplitude, middle,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )
    cluster1 = np.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    cluster2 = np.asarray(
        [0, 1, 1, 0, 1, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    X = np.full((16, 1), 0.5, dtype=np.float64)
    actual = two_way_clustered_covariance(
        X, 8.0 * scores, cluster1, cluster2
    )
    expected = np.asarray([[-4.0 * amplitude * tiny]], dtype=np.float64)
    assert np.isfinite(expected[0, 0])
    assert_allclose(actual, expected, rtol=3e-12, atol=0.0)

def test_influence_rows_certified_gram_preserves_constant_design_symmetry():
    amplitude = 2.0 ** 660
    middle = 2.0 ** 600
    tiny = 2.0 ** 350
    scores = np.asarray(
        [
            -amplitude, middle, tiny, amplitude, -middle, -tiny,
            -amplitude, -middle, amplitude, middle,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )
    X = np.full((16, 1), 0.5, dtype=np.float64)
    influence, projection_scale, design_scale, *_ = _influence_rows(
        X, 8.0 * scores, np
    )
    np.testing.assert_array_equal(influence[:, 0], scores)
    np.testing.assert_array_equal(projection_scale, np.ones(1))
    assert float(np.asarray(design_scale)) == 1.0

def test_two_way_ordinary_compensation_stays_vectorized(monkeypatch):
    rng = np.random.default_rng(126)
    scores = rng.normal(size=32)
    cluster1 = np.repeat(np.arange(4), 8)
    cluster2 = np.tile(np.arange(8), 4)
    X = np.ones((32, 1), dtype=np.float64)
    resid = 32.0 * scores

    components = _grouped_score_sums(
        scores[:, None], cluster1, n_groups=4, xp=np,
        return_components=True,
    )
    assert any(np.any(component != 0.0) for component in components[1:])

    observed_term_counts = []
    original = covariance_module._stable_matrix_expansion_sum

    def wrapped(terms, xp):
        observed_term_counts.append(len(terms))
        return original(terms, xp)

    monkeypatch.setattr(
        covariance_module, "_stable_matrix_expansion_sum", wrapped
    )
    covariance_module.two_way_clustered_covariance(
        X, resid, cluster1, cluster2
    )
    assert observed_term_counts
    # Two components in each of three cluster dimensions require only
    # 3 vectorized component-pair terms per dimension, not O(group-count) terms.
    assert max(observed_term_counts) <= 9

def test_two_way_extreme_many_groups_retiers_before_vectorized_gram(monkeypatch):
    n = 256
    large = 2.0 ** 500
    small = 2.0 ** 400
    scores = np.where(np.arange(n) % 2 == 0, large, small).astype(np.float64)
    cluster1 = np.repeat(np.arange(16), 16)
    cluster2 = np.tile(np.arange(16), 16)
    X = np.ones((n, 1), dtype=np.float64)
    resid = float(n) * scores

    observed_term_counts = []
    original = covariance_module._stable_matrix_expansion_sum

    def wrapped(terms, xp):
        observed_term_counts.append(len(terms))
        return original(terms, xp)

    monkeypatch.setattr(
        covariance_module, "_stable_matrix_expansion_sum", wrapped
    )
    actual = covariance_module.two_way_clustered_covariance(
        X, resid, cluster1, cluster2
    )
    assert np.all(np.isfinite(actual))
    assert observed_term_counts
    # The fallback complexity is tied to magnitude tiers, not 256 intersection
    # rows.  This fixture previously produced roughly one thousand row terms.
    assert max(observed_term_counts) <= 30



def test_two_way_clustered_covariance_fails_closed_on_common_scale_product_underflow():
    amplitude = 1.0e308
    tiny = 1.0e-100
    X = np.full((4, 1), 0.25, dtype=np.float64)
    resid = np.asarray([-amplitude, tiny, amplitude, tiny], dtype=np.float64)
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)

    # With influence rows equal to resid, CGM gives exactly 4*tiny**2, which is
    # representable. A single common value scale, however, cannot represent the
    # low-low product beside the 1e308 tier. Never silently publish zero.
    expected_meat = 4.0 * tiny * tiny
    assert expected_meat > 0.0 and np.isfinite(expected_meat)
    with pytest.raises(FloatingPointError, match="common-scale product range"):
        two_way_clustered_covariance(X, resid, cluster1, cluster2)
