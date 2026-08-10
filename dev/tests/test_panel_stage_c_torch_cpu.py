"""Maintained Torch-CPU parity for Panel Tier-1 Stage C covariance."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import BetweenOLS, FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects
from statgpu.panel._covariance import driscoll_kraay_covariance, ols_covariance, two_way_clustered_covariance


torch = pytest.importorskip("torch")


def _panel(seed=12900, *, unbalanced=True):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 8, 6
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.35, size=n_entities), n_times)
    y = 0.75 * X[:, 0] - 0.3 * X[:, 1] + alpha + rng.normal(scale=0.2, size=entity.size)
    if unbalanced:
        keep = np.ones(entity.size, dtype=bool)
        keep[[2, 11, 25, 40]] = False
        X, y, entity, time = X[keep], y[keep], entity[keep], time[keep]
    return X, y, entity, time


def _torch_arrays(X, y, entity, time):
    return (
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        torch.as_tensor(entity, dtype=torch.int64),
        torch.as_tensor(time, dtype=torch.int64),
    )


def _assert_inference(actual, expected, *, rtol=2e-8, atol=2e-10):
    assert_allclose(actual.coef_, expected.coef_, rtol=rtol, atol=atol)
    assert_allclose(actual.bse_, expected.bse_, rtol=rtol, atol=atol)
    assert_allclose(actual.tvalues_, expected.tvalues_, rtol=rtol, atol=atol)
    assert_allclose(actual.pvalues_, expected.pvalues_, rtol=rtol, atol=atol)
    assert_allclose(actual.conf_int_, expected.conf_int_, rtol=rtol, atol=atol)
    assert_allclose(actual._panel_cov_params_raw, expected._panel_cov_params_raw, rtol=rtol, atol=atol)


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_hc_primitives_torch_cpu_match_numpy(cov_type):
    rng = np.random.default_rng(12901)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, 2))])
    resid = rng.normal(size=40)
    X_t = torch.as_tensor(X, dtype=torch.float64)
    resid_t = torch.as_tensor(resid, dtype=torch.float64)
    expected = ols_covariance(X, resid, cov_type=cov_type, xp=np)
    actual = ols_covariance(X_t, resid_t, cov_type=cov_type)
    assert torch.is_tensor(actual)
    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)


def test_stage_c_two_way_cluster_torch_cpu_matches_numpy_with_group_debias():
    rng = np.random.default_rng(12902)
    X = np.column_stack([np.ones(48), rng.normal(size=(48, 2))])
    resid = rng.normal(size=48)
    c1 = np.repeat(np.arange(8), 6)
    c2 = np.tile(np.arange(6), 8)
    expected = two_way_clustered_covariance(
        X, resid, c1, c2, xp=np, group_debias=True
    )
    actual = two_way_clustered_covariance(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(resid, dtype=torch.float64),
        c1,
        c2,
        group_debias=True,
    )
    assert torch.is_tensor(actual)
    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
def test_stage_c_dk_torch_cpu_matches_numpy(kernel):
    rng = np.random.default_rng(12903)
    time = np.tile(np.arange(8), 7)
    X = np.column_stack([np.ones(time.size), rng.normal(size=(time.size, 2))])
    resid = rng.normal(size=time.size)
    expected = driscoll_kraay_covariance(
        X, resid, time, bandwidth=2, kernel=kernel, extra_df=3, xp=np
    )
    actual = driscoll_kraay_covariance(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(resid, dtype=torch.float64),
        time,
        bandwidth=2,
        kernel=kernel,
        extra_df=3,
    )
    assert torch.is_tensor(actual)
    assert_allclose(actual.detach().cpu().numpy(), expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_pooled_torch_cpu_hc_matches_numpy(cov_type):
    X, y, entity, time = _panel(12904)
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)
    expected = PooledOLS(cov_type=cov_type).fit(X, y, entity_ids=entity)
    actual = PooledOLS(cov_type=cov_type).fit(X_t, y_t, entity_ids=entity_t)
    _assert_inference(actual, expected)


@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_panel_fe_torch_cpu_hc_matches_numpy(cov_type):
    X, y, entity, time = _panel(12905)
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)
    expected = PanelOLS(entity_effects=True, cov_type=cov_type).fit(X, y, entity_ids=entity)
    actual = PanelOLS(entity_effects=True, cov_type=cov_type).fit(X_t, y_t, entity_ids=entity_t)
    _assert_inference(actual, expected)


def test_stage_c_panel_dk_torch_cpu_matches_numpy_and_effect_rank_metadata():
    X, y, entity, time = _panel(12906)
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)
    expected = PanelOLS(entity_effects=True, cov_type="dk", bandwidth=2).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    actual = PanelOLS(entity_effects=True, cov_type="dk", bandwidth=2).fit(
        X_t, y_t, entity_ids=entity_t, time_ids=time_t
    )
    _assert_inference(actual, expected)
    assert actual._covariance_metadata["extra_df"] == expected._covariance_metadata["extra_df"]
    assert actual._covariance_metadata["design_rank"] == expected._covariance_metadata["design_rank"]


@pytest.mark.parametrize("cov_type", ["robust", "hc0", "hc2", "hc3"])
def test_stage_c_random_effects_torch_cpu_covariance_matches_numpy(cov_type):
    X, y, entity, time = _panel(12907)
    X = np.column_stack([np.ones(len(y)), X])
    X_t, y_t, entity_t, _ = _torch_arrays(X, y, entity, time)
    expected = RandomEffects(cov_type=cov_type).fit(X, y, entity_ids=entity)
    actual = RandomEffects(cov_type=cov_type).fit(X_t, y_t, entity_ids=entity_t)
    _assert_inference(actual, expected, rtol=5e-8, atol=5e-10)
    assert_allclose(
        [actual.variance_components_["sigma2_e"], actual.variance_components_["sigma2_a"]],
        [expected.variance_components_["sigma2_e"], expected.variance_components_["sigma2_a"]],
        rtol=2e-10,
        atol=2e-12,
    )


def test_stage_c_random_effects_dk_torch_cpu_matches_numpy():
    X, y, entity, time = _panel(12908)
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)
    expected = RandomEffects(cov_type="dk", bandwidth=2, kernel="qs").fit(
        X, y, entity_ids=entity, time_ids=time
    )
    actual = RandomEffects(cov_type="dk", bandwidth=2, kernel="qs").fit(
        X_t, y_t, entity_ids=entity_t, time_ids=time_t
    )
    _assert_inference(actual, expected, rtol=5e-8, atol=5e-10)
    assert actual._covariance_metadata["all_observed_lags_weighted"] is True


@pytest.mark.parametrize("estimator", [BetweenOLS, FirstDifferenceOLS])
@pytest.mark.parametrize("cov_type", ["hc0", "hc2", "hc3"])
def test_stage_c_between_and_fd_torch_cpu_hc_match_numpy(estimator, cov_type):
    X, y, entity, time = _panel(12909, unbalanced=False)
    X_t, y_t, entity_t, time_t = _torch_arrays(X, y, entity, time)
    if estimator is BetweenOLS:
        expected = estimator(cov_type=cov_type).fit(X, y, entity_ids=entity)
        actual = estimator(cov_type=cov_type).fit(X_t, y_t, entity_ids=entity_t)
    else:
        expected = estimator(cov_type=cov_type).fit(
            X, y, entity_ids=entity, time_ids=time
        )
        actual = estimator(cov_type=cov_type).fit(
            X_t, y_t, entity_ids=entity_t, time_ids=time_t
        )
    _assert_inference(actual, expected, rtol=5e-8, atol=5e-10)
