"""External Fama-MacBeth definition alignment against linearmodels 7.0.

The comparison is deliberately parameterization-specific.  statgpu's
``nonrobust`` covariance matches the standard covariance of the retained period
coefficients with a T-1 covariance denominator, which is linearmodels
``cov_type='unadjusted', debiased=True``.  The two packages use different
reference degrees of freedom for the resulting p-values in this branch, so the
external contract stops at coefficients/covariance/standard errors/t-statistics.

statgpu's coefficient-series Newey-West covariance matches linearmodels' Bartlett
kernel covariance with the same fixed bandwidth and ``debiased=False``.  Both
then use normal-reference inference, so p-values and confidence intervals are
also compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip(
    "linearmodels",
    reason="linearmodels 7.0 is an external Fama-MacBeth definition gate",
)

from linearmodels.panel import FamaMacBeth as LMFamaMacBeth

from statgpu.panel import FamaMacBeth


def _fixture(seed=20260816):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 14, 9
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))

    # Deliberately vary all period coefficients so both the standard and lagged
    # coefficient-series covariance definitions are exercised with real power.
    intercept_t = 0.25 + 0.06 * np.arange(n_times)
    slope1_t = 0.8 + 0.12 * np.sin(np.arange(n_times))
    slope2_t = -0.45 + 0.08 * np.cos(0.7 * np.arange(n_times))
    y = (
        intercept_t[time]
        + slope1_t[time] * X[:, 0]
        + slope2_t[time] * X[:, 1]
        + rng.normal(scale=0.18, size=entity.size)
    )
    return X, y, entity, time


def _lm_data(X, y, entity, time):
    index = pd.MultiIndex.from_arrays([entity, time], names=["entity", "time"])
    dependent = pd.Series(y, index=index, name="y")
    exog = pd.DataFrame(
        {
            "const": np.ones(len(y), dtype=np.float64),
            "x1": X[:, 0],
            "x2": X[:, 1],
        },
        index=index,
    )
    return dependent, exog


def _fit_pair(*, cov_type, bandwidth=None):
    X, y, entity, time = _fixture()
    dependent, exog = _lm_data(X, y, entity, time)

    if cov_type == "nonrobust":
        lm = LMFamaMacBeth(dependent, exog).fit(
            cov_type="unadjusted",
            debiased=True,
        )
        sg = FamaMacBeth(cov_type="nonrobust", device="cpu").fit(
            X,
            y,
            time_ids=time,
            entity_ids=entity,
        )
    elif cov_type == "newey-west":
        assert bandwidth is not None
        lm = LMFamaMacBeth(dependent, exog).fit(
            cov_type="kernel",
            kernel="bartlett",
            bandwidth=bandwidth,
            debiased=False,
        )
        sg = FamaMacBeth(
            cov_type="newey-west",
            bandwidth=bandwidth,
            device="cpu",
        ).fit(
            X,
            y,
            time_ids=time,
            entity_ids=entity,
        )
    else:  # pragma: no cover - local helper contract
        raise ValueError(cov_type)

    return sg, lm


def _assert_period_and_average_coefficients(sg, lm):
    assert list(lm.params.index) == ["const", "x1", "x2"]
    assert list(lm.all_params.columns) == ["const", "x1", "x2"]
    assert sg.n_periods == lm.all_params.shape[0]
    assert_allclose(
        np.asarray(sg.betas_, dtype=np.float64),
        lm.all_params.to_numpy(dtype=np.float64),
        rtol=2e-11,
        atol=2e-12,
    )
    assert_allclose(
        np.asarray(sg.coef_, dtype=np.float64),
        lm.params.to_numpy(dtype=np.float64),
        rtol=2e-11,
        atol=2e-12,
    )


def test_fama_macbeth_nonrobust_matches_linearmodels_standard_covariance():
    sg, lm = _fit_pair(cov_type="nonrobust")
    _assert_period_and_average_coefficients(sg, lm)

    assert_allclose(sg.cov_params_, lm.cov.to_numpy(), rtol=2e-10, atol=2e-12)
    assert_allclose(sg.bse_, lm.std_errors.to_numpy(), rtol=2e-10, atol=2e-12)
    assert_allclose(sg.tvalues_, lm.tstats.to_numpy(), rtol=2e-10, atol=2e-12)

    # Covariance definitions align, but the inferential reference df do not:
    # statgpu intentionally uses T-1 while linearmodels' debiased post-estimation
    # uses stacked-panel residual df.  Do not assert p-value/CI equality here.
    assert sg._inference_result.statistic_name == "t"
    assert sg._inference_result.distribution == "t"
    assert sg._inference_result.df == float(sg.n_periods - 1)
    assert float(lm.df_resid) != float(sg.n_periods - 1)


def test_fama_macbeth_newey_west_matches_linearmodels_bartlett_kernel():
    bandwidth = 2
    sg, lm = _fit_pair(cov_type="newey-west", bandwidth=bandwidth)
    _assert_period_and_average_coefficients(sg, lm)

    assert_allclose(sg.cov_params_, lm.cov.to_numpy(), rtol=3e-10, atol=3e-12)
    assert_allclose(sg.bse_, lm.std_errors.to_numpy(), rtol=3e-10, atol=3e-12)
    assert_allclose(sg.tvalues_, lm.tstats.to_numpy(), rtol=3e-10, atol=3e-12)
    assert_allclose(sg.pvalues_, lm.pvalues.to_numpy(), rtol=2e-9, atol=2e-12)
    assert_allclose(
        sg.conf_int_,
        lm.conf_int(level=1.0 - sg.alpha).to_numpy(),
        rtol=2e-9,
        atol=2e-12,
    )

    result = sg._inference_result
    assert result.statistic_name == "z"
    assert result.distribution == "normal"
    assert result.df is None
    assert result.metadata["effective_bandwidth"] == bandwidth
