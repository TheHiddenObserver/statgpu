"""Focused Stage-C covariance correctness and compatibility contracts."""

from __future__ import annotations

import numpy as np
import pytest
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
