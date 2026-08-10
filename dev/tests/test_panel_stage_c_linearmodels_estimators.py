"""Estimator-level Stage-C covariance alignment against pinned references.

These tests complement the covariance-primitive checks by proving that public
PooledOLS/PanelOLS integrations pass the intended fit space and fixed-effect
rank into Driscoll-Kraay and clustered covariance. RandomEffects is checked on
statgpu's own Swamy-Arora quasi-demeaned fit space so covariance alignment does
not redefine its coefficient/variance-component contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("linearmodels")
statsmodels = pytest.importorskip("statsmodels.api")

from linearmodels.panel import PanelOLS as LMPanelOLS
from linearmodels.panel import PooledOLS as LMPooledOLS
from linearmodels.panel.covariance import (
    ClusteredCovariance,
    DriscollKraay,
    HeteroskedasticCovariance,
)

from statgpu.panel import PanelOLS, PooledOLS, RandomEffects


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


def _re_fit_space(X, y, entity, model):
    """Reconstruct statgpu's Swamy-Arora quasi-demeaned regression space."""
    _, codes = np.unique(entity, return_inverse=True)
    counts_by_group = np.bincount(codes).astype(np.float64)
    counts = counts_by_group[codes]

    y_sums = np.bincount(codes, weights=y)
    y_bar = (y_sums / counts_by_group)[codes]
    X_bar = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        x_sums = np.bincount(codes, weights=X[:, j])
        X_bar[:, j] = (x_sums / counts_by_group)[codes]

    sigma2_e = float(model.variance_components_["sigma2_e"])
    sigma2_a = float(model.variance_components_["sigma2_a"])
    denom = sigma2_e + counts * sigma2_a
    theta = np.where(
        denom > 0.0,
        1.0 - np.sqrt(sigma2_e / denom),
        0.0,
    )
    y_star = y - theta * y_bar
    X_star = X - theta[:, None] * X_bar
    params = np.asarray(model.coef_, dtype=np.float64).ravel()
    resid = y_star - X_star @ params
    return X_star, y_star, params, resid


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


def test_random_effects_robust_matches_linearmodels_on_statgpu_fit_space():
    X, y, entity, time = _panel(12725)
    X = np.column_stack([np.ones(len(y)), X])
    sg = RandomEffects(cov_type="robust").fit(X, y, entity_ids=entity)
    X_star, y_star, params, _ = _re_fit_space(X, y, entity, sg)

    expected = HeteroskedasticCovariance(
        y_star[:, None],
        X_star,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=True,
        extra_df=0,
    ).cov
    assert_allclose(sg._panel_cov_params_raw, expected, rtol=5e-9, atol=5e-11)
    assert sg._covariance_metadata["hc_equivalent"] == "hc1"


@pytest.mark.parametrize("cov_type", ["hc2", "hc3"])
def test_random_effects_hc2_hc3_matches_statsmodels_on_statgpu_fit_space(cov_type):
    X, y, entity, _time = _panel(12726)
    X = np.column_stack([np.ones(len(y)), X])
    sg = RandomEffects(cov_type=cov_type).fit(X, y, entity_ids=entity)
    X_star, y_star, _params, _ = _re_fit_space(X, y, entity, sg)

    expected = (
        statsmodels.OLS(y_star, X_star)
        .fit()
        .get_robustcov_results(cov_type=cov_type.upper())
        .cov_params()
    )
    assert_allclose(sg._panel_cov_params_raw, expected, rtol=5e-9, atol=5e-11)


def test_random_effects_two_way_cluster_fails_closed_when_reference_variance_is_negative():
    X, y, entity, time = _panel(12727)
    X = np.column_stack([np.ones(len(y)), X])
    clusters = np.column_stack([entity, time])

    # Reconstruct the same Swamy-Arora fit space from a covariance-invariant
    # successful fit, then verify the external two-way cluster definition really
    # has a materially negative variance for this deterministic fixture.
    base = RandomEffects(cov_type="nonrobust").fit(X, y, entity_ids=entity)
    X_star, y_star, params, _ = _re_fit_space(X, y, entity, base)
    expected = ClusteredCovariance(
        y_star[:, None],
        X_star,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=False,
        extra_df=0,
        clusters=clusters,
        group_debias=True,
    ).cov
    assert np.min(np.diag(expected)) < -1.0e-12

    with pytest.raises(
        ValueError,
        match="materially negative diagonal variance",
    ):
        RandomEffects(cov_type="clustered", group_debias=True).fit(
            X, y, entity_ids=entity, cluster=clusters
        )


def test_random_effects_dk_matches_linearmodels_on_statgpu_fit_space():
    X, y, entity, time = _panel(12728)
    X = np.column_stack([np.ones(len(y)), X])
    sg = RandomEffects(cov_type="dk", kernel="parzen", bandwidth=2).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    X_star, y_star, params, _ = _re_fit_space(X, y, entity, sg)

    expected = DriscollKraay(
        y_star[:, None],
        X_star,
        params[:, None],
        entity[:, None],
        time[:, None],
        debiased=True,
        extra_df=0,
        kernel="parzen",
        bandwidth=2.0,
    ).cov
    assert_allclose(sg._panel_cov_params_raw, expected, rtol=5e-9, atol=5e-11)
    assert sg._covariance_metadata["extra_df"] == 0
