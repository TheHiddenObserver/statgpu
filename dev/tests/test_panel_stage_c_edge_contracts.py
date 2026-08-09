"""Edge contracts for public Panel Stage-C covariance primitives."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import driscoll_kraay_covariance
from statgpu.panel._covariance import clustered_covariance


def _fit_space(seed=12950, n_times=6):
    rng = np.random.default_rng(seed)
    time = np.tile(np.arange(n_times), 5)
    X = np.column_stack([np.ones(time.size), rng.normal(size=time.size)])
    resid = rng.normal(size=time.size)
    return X, resid, time


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0.0, 1.0, np.nan, 2.0]),
        np.array([0.0, 1.0, np.inf, 2.0]),
        np.array(["a", "b", None, "c"], dtype=object),
        np.array([np.datetime64("2026-01-01"), np.datetime64("NaT")]),
    ],
)
def test_group_label_factorization_rejects_missing_or_nonfinite_values(labels):
    n = len(labels)
    X = np.column_stack([np.ones(n), np.arange(float(n))])
    resid = np.linspace(-0.2, 0.3, n)
    with pytest.raises(ValueError, match="must not contain missing or non-finite values"):
        clustered_covariance(X, resid, labels)


def test_public_driscoll_kraay_rejects_missing_time_labels():
    X, resid, time = _fit_space()
    time = time.astype(float)
    time[7] = np.nan
    with pytest.raises(ValueError, match="must not contain missing or non-finite values"):
        driscoll_kraay_covariance(X, resid, time, bandwidth=2)


@pytest.mark.parametrize("kernel", ["bartlett", "parzen"])
def test_truncated_kernels_do_not_silently_cap_oversized_bandwidth(kernel):
    X, resid, time = _fit_space(n_times=5)
    meta = {}
    cov_large = driscoll_kraay_covariance(
        X, resid, time, bandwidth=9, kernel=kernel, metadata=meta
    )
    cov_capped = driscoll_kraay_covariance(
        X, resid, time, bandwidth=4, kernel=kernel
    )
    assert meta["bandwidth"] == 9
    assert meta["max_weighted_lag"] == 4
    # The requested bandwidth remains in the Bartlett/Parzen weight formula;
    # silently replacing 9 by T-1=4 would make these matrices identical.
    assert not np.allclose(cov_large, cov_capped, rtol=1e-12, atol=1e-14)


def test_qs_oversized_bandwidth_remains_a_smoothing_scale():
    X, resid, time = _fit_space(n_times=5)
    meta = {}
    cov = driscoll_kraay_covariance(
        X, resid, time, bandwidth=9, kernel="qs", metadata=meta
    )
    assert np.all(np.isfinite(cov))
    assert meta["bandwidth"] == 9
    assert meta["all_observed_lags_weighted"] is True
    assert meta["max_weighted_lag"] == 4
