"""Pinned external-definition checks for Panel Stage C covariance."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

linearmodels = pytest.importorskip("linearmodels")
statsmodels = pytest.importorskip("statsmodels.api")

from linearmodels.iv.covariance import KERNEL_LOOKUP
from linearmodels.panel.covariance import ClusteredCovariance, DriscollKraay

from statgpu.panel._covariance import (
    _dk_kernel_weights,
    clustered_covariance,
    driscoll_kraay_covariance,
    ols_covariance,
    two_way_clustered_covariance,
)


def _regression(seed=12700, *, n_entities=9, n_times=7):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = np.column_stack([np.ones(entity.size), rng.normal(size=(entity.size, 2))])
    beta = np.array([0.25, 0.7, -0.4])
    eps = rng.normal(scale=0.3, size=entity.size)
    y = X @ beta + eps
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    return X, y, params, entity, time


@pytest.mark.parametrize("cov_type", ["HC2", "HC3"])
def test_hc2_hc3_fit_space_covariance_matches_statsmodels(cov_type):
    X, y, params, entity, time = _regression(seed=12701)
    resid = y - X @ params
    actual = ols_covariance(X, resid, cov_type=cov_type.lower())
    expected = (
        statsmodels.OLS(y, X)
        .fit()
        .get_robustcov_results(cov_type=cov_type)
        .cov_params()
    )
    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)


def test_one_way_group_debiased_cluster_matches_linearmodels_primitive():
    X, y, params, entity, time = _regression(seed=12702)
    resid = y - X @ params
    actual = clustered_covariance(
        X, resid, entity, group_debias=True
    )
    expected = ClusteredCovariance(
        y[:, None],
        X,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=False,
        extra_df=0,
        clusters=entity,
        group_debias=True,
    ).cov
    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)


def test_two_way_group_debiased_cluster_matches_linearmodels_primitive():
    X, y, params, entity, time = _regression(seed=12703)
    resid = y - X @ params
    clusters = np.column_stack([entity, time])
    actual = two_way_clustered_covariance(
        X,
        resid,
        entity,
        time,
        group_debias=True,
    )
    expected = ClusteredCovariance(
        y[:, None],
        X,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=False,
        extra_df=0,
        clusters=clusters,
        group_debias=True,
    ).cov
    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)


@pytest.mark.parametrize(
    "kernel,lm_kernel",
    [
        ("bartlett", "bartlett"),
        ("parzen", "parzen"),
        ("qs", "qs"),
    ],
)
def test_dk_kernel_weights_match_linearmodels(kernel, lm_kernel):
    canonical, actual = _dk_kernel_weights(kernel, bandwidth=2, max_lag=8)
    expected = np.asarray(KERNEL_LOOKUP[lm_kernel](2.0, 8), dtype=np.float64)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected, rtol=5e-14, atol=5e-15)
    if canonical == "qs":
        assert np.count_nonzero(actual[1:]) == 8


@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "qs"])
def test_driscoll_kraay_full_rank_covariance_matches_linearmodels(kernel):
    X, y, params, entity, time = _regression(seed=12704)
    resid = y - X @ params
    actual = driscoll_kraay_covariance(
        X,
        resid,
        time,
        bandwidth=2,
        kernel=kernel,
        extra_df=0,
    )
    expected = DriscollKraay(
        y[:, None],
        X,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=True,
        extra_df=0,
        kernel=kernel,
        bandwidth=2.0,
    ).cov
    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)


def test_driscoll_kraay_extra_df_matches_linearmodels_scale():
    X, y, params, entity, time = _regression(seed=12705)
    resid = y - X @ params
    extra_df = 5
    actual = driscoll_kraay_covariance(
        X,
        resid,
        time,
        bandwidth=2,
        kernel="bartlett",
        extra_df=extra_df,
    )
    expected = DriscollKraay(
        y[:, None],
        X,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=True,
        extra_df=extra_df,
        kernel="bartlett",
        bandwidth=2.0,
    ).cov
    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)
