"""Additional pinned external defaults for Panel Stage-C covariance."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("linearmodels")

from linearmodels.panel.covariance import DriscollKraay

from statgpu.panel._covariance import driscoll_kraay_covariance


def test_driscoll_kraay_default_bandwidth_matches_linearmodels_7_0():
    rng = np.random.default_rng(12706)
    n_entities, n_times = 12, 15
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = np.column_stack([np.ones(entity.size), rng.normal(size=(entity.size, 2))])
    beta = np.array([0.2, 0.65, -0.3])
    y = X @ beta + rng.normal(scale=0.25, size=entity.size)
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ params

    actual_meta = {}
    actual = driscoll_kraay_covariance(
        X,
        resid,
        time,
        bandwidth=None,
        kernel="bartlett",
        extra_df=0,
        metadata=actual_meta,
    )
    expected_bandwidth = int(np.floor(4.0 * (n_times / 100.0) ** (2.0 / 9.0)))
    expected = DriscollKraay(
        y[:, None],
        X,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=True,
        extra_df=0,
        kernel="bartlett",
        bandwidth=None,
    ).cov

    assert actual_meta["bandwidth"] == expected_bandwidth
    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen"])
def test_driscoll_kraay_oversized_truncated_kernels_are_documented_extension(kernel):
    rng = np.random.default_rng(12707)
    n_entities, n_times = 10, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = np.column_stack([np.ones(entity.size), rng.normal(size=(entity.size, 2))])
    beta = np.array([0.15, 0.55, -0.2])
    y = X @ beta + rng.normal(scale=0.2, size=entity.size)
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ params
    bandwidth = 9

    meta = {}
    actual = driscoll_kraay_covariance(
        X, resid, time, bandwidth=bandwidth, kernel=kernel, metadata=meta
    )

    # linearmodels 7.0 materializes bw+1 Bartlett/Parzen weights and its
    # cov_kernel rejects that vector when it is longer than the T grouped
    # scores.  Stage C deliberately extends this edge by retaining the requested
    # bandwidth in the weight denominator while accumulating only observed lags.
    with pytest.raises(ValueError, match="Length of w"):
        _ = DriscollKraay(
            y[:, None],
            X,
            params[:, None],
            entity[:, None],
            time[:, None],
            debiased=True,
            extra_df=0,
            kernel=kernel,
            bandwidth=float(bandwidth),
        ).cov

    labels = np.unique(time)
    grouped = np.stack([(X[time == t] * resid[time == t, None]).sum(axis=0) for t in labels])
    meat = grouped.T @ grouped
    for lag in range(1, len(labels)):
        z = lag / float(bandwidth + 1)
        if kernel == "bartlett":
            weight = 1.0 - z
        elif z <= 0.5:
            weight = 1.0 - 6.0 * z**2 + 6.0 * z**3
        else:
            weight = 2.0 * (1.0 - z) ** 3
        gamma = grouped[lag:].T @ grouped[:-lag]
        meat = meat + weight * (gamma + gamma.T)
    bread = np.linalg.inv(X.T @ X)
    scale = len(y) / float(len(y) - X.shape[1])
    expected = scale * (bread @ meat @ bread)
    expected = 0.5 * (expected + expected.T)

    assert meta["bandwidth"] == bandwidth
    assert meta["max_weighted_lag"] == n_times - 1
    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)


def test_driscoll_kraay_oversized_qs_matches_linearmodels_7_0():
    rng = np.random.default_rng(12708)
    n_entities, n_times = 10, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = np.column_stack([np.ones(entity.size), rng.normal(size=(entity.size, 2))])
    beta = np.array([0.1, 0.5, -0.25])
    y = X @ beta + rng.normal(scale=0.2, size=entity.size)
    params = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ params
    bandwidth = 9

    meta = {}
    actual = driscoll_kraay_covariance(
        X, resid, time, bandwidth=bandwidth, kernel="qs", metadata=meta
    )
    expected = DriscollKraay(
        y[:, None],
        X,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=True,
        extra_df=0,
        kernel="qs",
        bandwidth=float(bandwidth),
    ).cov

    assert meta["bandwidth"] == bandwidth
    assert meta["all_observed_lags_weighted"] is True
    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)
