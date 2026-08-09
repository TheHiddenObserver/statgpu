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
