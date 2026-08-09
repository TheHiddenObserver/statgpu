"""Estimator-level Stage-C covariance alignment against linearmodels 7.0.

These tests complement the covariance-primitive checks by proving that public
PooledOLS/PanelOLS integrations pass the intended fit space and fixed-effect
rank into Driscoll-Kraay and clustered covariance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("linearmodels")

from linearmodels.panel import PanelOLS as LMPanelOLS
from linearmodels.panel import PooledOLS as LMPooledOLS

from statgpu.panel import PanelOLS, PooledOLS


def _panel(seed=12720, *, unbalanced=True):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 10, 8
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.45, size=n_entities), n_times)
    tau = np.tile(np.linspace(-0.2, 0.25, n_times), n_entities)
    y = 0.8 * X[:, 0] - 0.35 * X[:, 1] + alpha + 0.3 * tau
    y += rng.normal(scale=0.22, size=entity.size)
    if unbalanced:
        keep = np.ones(entity.size, dtype=bool)
        keep[[1, 10, 19, 37, 58, 71]] = False
        X, y, entity, time = X[keep], y[keep], entity[keep], time[keep]
    return X, y, entity, time


def _lm_data(X, y, entity, time, *, constant=False):
    index = pd.MultiIndex.from_arrays([entity, time], names=["entity", "time"])
    y_series = pd.Series(y, index=index, name="y")
    X_frame = pd.DataFrame(X, index=index, columns=["x1", "x2"])
    if constant:
        X_frame.insert(0, "const", 1.0)
    return y_series, X_frame


def test_pooled_dk_estimator_matches_linearmodels_full_integration():
    X, y, entity, time = _panel(12721)
    y_lm, X_lm = _lm_data(X, y, entity, time, constant=True)
    lm = LMPooledOLS(y_lm, X_lm).fit(
        cov_type="kernel",
        kernel="bartlett",
        bandwidth=2,
        debiased=True,
    )
    sg = PooledOLS(cov_type="dk", kernel="bartlett", bandwidth=2).fit(
        X, y, time_index=time
    )

    assert_allclose(sg.coef_, lm.params.to_numpy(), rtol=2e-10, atol=2e-11)
    assert_allclose(sg._panel_cov_params_raw, lm.cov.to_numpy(), rtol=5e-9, atol=5e-11)
    assert_allclose(sg.bse_, lm.std_errors.to_numpy(), rtol=5e-9, atol=5e-11)
    assert sg._covariance_metadata["extra_df"] == 0


def test_pooled_group_debiased_cluster_estimator_matches_linearmodels():
    X, y, entity, time = _panel(12722)
    y_lm, X_lm = _lm_data(X, y, entity, time, constant=True)
    clusters = pd.Series(entity, index=y_lm.index, name="cluster")
    lm = LMPooledOLS(y_lm, X_lm).fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=False,
        group_debias=True,
    )
    sg = PooledOLS(cov_type="clustered", group_debias=True).fit(
        X, y, cluster=entity
    )

    assert_allclose(sg.coef_, lm.params.to_numpy(), rtol=2e-10, atol=2e-11)
    assert_allclose(sg._panel_cov_params_raw, lm.cov.to_numpy(), rtol=5e-9, atol=5e-11)
    assert sg._covariance_metadata["group_debias"] is True


def test_one_way_panel_dk_matches_linearmodels_effect_df_integration():
    X, y, entity, time = _panel(12723)
    y_lm, X_lm = _lm_data(X, y, entity, time, constant=False)
    lm = LMPanelOLS(y_lm, X_lm, entity_effects=True).fit(
        cov_type="kernel",
        kernel="bartlett",
        bandwidth=2,
        debiased=True,
        auto_df=True,
        count_effects=True,
    )
    sg = PanelOLS(
        entity_effects=True,
        cov_type="dk",
        kernel="bartlett",
        bandwidth=2,
    ).fit(X, y, entity_ids=entity, time_ids=time)

    assert_allclose(sg.coef_, lm.params.to_numpy(), rtol=2e-10, atol=2e-11)
    assert_allclose(sg._panel_cov_params_raw, lm.cov.to_numpy(), rtol=5e-9, atol=5e-11)
    assert_allclose(sg.bse_, lm.std_errors.to_numpy(), rtol=5e-9, atol=5e-11)
    effect_rank = sg.fit_statistics_.metadata["diagnostic_df"]["effect_rank"]
    assert sg._covariance_metadata["extra_df"] == effect_rank


def test_two_way_panel_dk_matches_linearmodels_effect_df_integration():
    X, y, entity, time = _panel(12724)
    y_lm, X_lm = _lm_data(X, y, entity, time, constant=False)
    lm = LMPanelOLS(
        y_lm,
        X_lm,
        entity_effects=True,
        time_effects=True,
    ).fit(
        cov_type="kernel",
        kernel="parzen",
        bandwidth=2,
        debiased=True,
        auto_df=True,
        count_effects=True,
    )
    sg = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="dk",
        kernel="parzen",
        bandwidth=2,
    ).fit(X, y, entity_ids=entity, time_ids=time)

    assert_allclose(sg.coef_, lm.params.to_numpy(), rtol=2e-10, atol=2e-11)
    assert_allclose(sg._panel_cov_params_raw, lm.cov.to_numpy(), rtol=5e-9, atol=5e-11)
    effect_rank = sg.fit_statistics_.metadata["diagnostic_df"]["effect_rank"]
    assert sg._covariance_metadata["extra_df"] == effect_rank
