"""Pre-refactor golden behavior for Panel P1 Stage A (#93).

This file is intentionally committed before any Stage-A panel source refactor.
It freezes current public numerical and output contracts so the subsequent
shared-base/covariance-registry migration cannot silently change econometric
behavior while removing duplication.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)


RTOL = 2e-7
ATOL = 2e-9


def _balanced_panel():
    rng = np.random.default_rng(20260807)
    n_entities, n_times = 8, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    entity_effect = np.repeat(rng.normal(scale=0.7, size=n_entities), n_times)
    time_trend = np.tile(np.linspace(-0.3, 0.4, n_times), n_entities)
    y = (
        0.7
        + 1.4 * X[:, 0]
        - 0.8 * X[:, 1]
        + entity_effect
        + 0.25 * time_trend
        + rng.normal(scale=0.15, size=entity.size)
    )
    return X, y, entity, time


def _unbalanced_panel():
    X, y, entity, time = _balanced_panel()
    keep = np.ones(y.shape[0], dtype=bool)
    keep[[3, 7, 14, 22, 37]] = False
    return X[keep], y[keep], entity[keep], time[keep]


def _assert_summary_contract(model, expected_model_type):
    summary = model.summary()
    payload = summary.to_dict()
    assert payload["model_type"] == expected_model_type
    assert payload["nobs"] == model.nobs
    assert payload["df_resid"] == model.df_resid
    assert_allclose(payload["coef"], np.asarray(model.coef_), rtol=0, atol=0)
    assert_allclose(payload["bse"], np.asarray(model.bse_), rtol=0, atol=0)
    assert_allclose(payload["pvalues"], np.asarray(model.pvalues_), rtol=0, atol=0)
    assert set(payload) == {
        "model_type",
        "nobs",
        "df_resid",
        "coef",
        "bse",
        "tvalues",
        "pvalues",
        "conf_int",
        "feature_names",
        "rsquared_within",
        "cov_type",
        "entity_effects",
        "time_effects",
        "variance_components",
        "theta",
        "alpha",
    }


def test_pooled_ols_nonrobust_golden_contract():
    X, y, _, _ = _balanced_panel()
    model = PooledOLS(cov_type="nonrobust", device="cpu").fit(X, y)

    assert_allclose(
        model.coef_,
        [0.49305911, 1.53649894, -0.75433471],
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        model.bse_,
        [0.10263772, 0.11417175, 0.12463030],
        rtol=RTOL,
        atol=ATOL,
    )
    assert model.df_resid == 37
    assert_allclose(model.rsquared, 0.8461259680614999, rtol=1e-12)
    assert_allclose(
        model.predict(X[:4]),
        [-0.71288515, -1.19011624, 1.05980653, 2.26266654],
        rtol=RTOL,
        atol=ATOL,
    )
    assert isinstance(model.predict(X[:1]), np.ndarray)
    _assert_summary_contract(model, "PooledOLS")


def test_pooled_ols_hc1_golden_scaling():
    X, y, _, _ = _balanced_panel()
    model = PooledOLS(cov_type="robust", device="cpu").fit(X, y)
    assert_allclose(
        model.bse_,
        [0.10101535, 0.11343041, 0.11585405],
        rtol=RTOL,
        atol=ATOL,
    )


def test_between_ols_golden_contract():
    X, y, entity, _ = _balanced_panel()
    model = BetweenOLS(device="cpu").fit(X, y, entity_ids=entity)
    assert_allclose(
        model.coef_,
        [0.34827978, 1.82753777, 0.20274932],
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        model.bse_,
        [0.33521424, 0.56349468, 1.15218346],
        rtol=RTOL,
        atol=ATOL,
    )
    assert model.nobs == 8
    assert model.df_resid == 5
    assert_allclose(model.rsquared, 0.7106688231281271, rtol=1e-12)
    assert isinstance(model.predict(X[:1]), np.ndarray)
    _assert_summary_contract(model, "BetweenOLS")


def test_first_difference_ols_golden_contract():
    X, y, entity, time = _balanced_panel()
    model = FirstDifferenceOLS(device="cpu").fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert_allclose(model.coef_, [1.38013929, -0.84275104], rtol=RTOL, atol=ATOL)
    assert_allclose(model.bse_, [0.02976731, 0.02997679], rtol=RTOL, atol=ATOL)
    assert model.nobs == 32
    assert model.df_resid == 30
    assert_allclose(model.rsquared, 0.9879000717161126, rtol=1e-12)
    assert_allclose(
        model.predict(X[:4]),
        [-1.24412427, -1.60521768, 0.40716805, 1.79119049],
        rtol=RTOL,
        atol=ATOL,
    )
    assert isinstance(model.predict(X[:1]), np.ndarray)
    _assert_summary_contract(model, "FirstDifferenceOLS")


def test_panel_ols_entity_effect_golden_contract():
    X, y, entity, _ = _balanced_panel()
    model = PanelOLS(entity_effects=True, device="cpu").fit(
        X, y, entity_ids=entity
    )
    assert_allclose(model.coef_, [1.40026731, -0.83017523], rtol=RTOL, atol=ATOL)
    assert_allclose(model.bse_, [0.03123062, 0.03078126], rtol=RTOL, atol=ATOL)
    assert model.df_resid == 31
    assert_allclose(model.rsquared_within, 0.9881458232165806, rtol=1e-12)
    assert_allclose(model._grand_mean, 0.49095910222140393, rtol=1e-12)
    assert set(model._entity_effects_map) == set(np.unique(entity))
    # Full level prediction restores the separately stored grand mean when
    # a fitted entity effect is used.
    assert_allclose(
        model.predict(X[:4], entity_ids=entity[:4]),
        np.array([-1.17565613, -1.55218831, 0.49083732, 1.84934445])
        + model._grand_mean,
        rtol=RTOL,
        atol=ATOL,
    )
    assert isinstance(model.predict(X[:1]), np.ndarray)
    _assert_summary_contract(model, "PanelOLS")


def test_panel_ols_two_way_golden_contract():
    X, y, entity, time = _balanced_panel()
    model = PanelOLS(
        entity_effects=True, time_effects=True, device="cpu"
    ).fit(X, y, entity_ids=entity, time_ids=time)
    assert_allclose(model.coef_, [1.39729530, -0.82027722], rtol=RTOL, atol=ATOL)
    assert_allclose(model.bse_, [0.03275315, 0.03341182], rtol=RTOL, atol=ATOL)
    assert model.df_resid == 27
    assert_allclose(model.rsquared_within, 0.9878363795416668, rtol=1e-12)
    assert set(model._entity_effects_map) == set(np.unique(entity))
    assert set(model._time_effects_map) == set(np.unique(time))


def test_random_effects_balanced_golden_contract():
    X, y, entity, _ = _balanced_panel()
    model = RandomEffects(device="cpu").fit(X, y, entity_ids=entity)
    assert_allclose(model.coef_, [1.40386230, -0.81694311], rtol=RTOL, atol=ATOL)
    assert_allclose(model.bse_, [0.04730892, 0.04688587], rtol=RTOL, atol=ATOL)
    assert model.df_resid == 38
    assert_allclose(model.variance_components_["sigma2_e"], 0.02224409581268407, rtol=1e-11)
    assert_allclose(model.variance_components_["sigma2_a"], 0.07872909189887668, rtol=1e-11)
    assert_allclose(model.theta_, 0.7687304945253098, rtol=1e-11)
    assert_allclose(
        model.predict(X[:4]),
        [-1.22625927, -1.61004314, 0.43902542, 1.77278516],
        rtol=RTOL,
        atol=ATOL,
    )
    assert isinstance(model.predict(X[:1]), np.ndarray)
    _assert_summary_contract(model, "RandomEffects")


def test_random_effects_unbalanced_golden_contract():
    X, y, entity, _ = _unbalanced_panel()
    model = RandomEffects(device="cpu").fit(X, y, entity_ids=entity)
    assert_allclose(model.coef_, [1.40160512, -0.80248338], rtol=RTOL, atol=ATOL)
    assert_allclose(model.bse_, [0.05058670, 0.05546781], rtol=RTOL, atol=ATOL)
    assert model.df_resid == 33
    assert_allclose(model.variance_components_["sigma2_e"], 0.025407186230309235, rtol=1e-11)
    assert_allclose(model.variance_components_["sigma2_a"], 0.12320985173284713, rtol=1e-11)
    assert_allclose(model.theta_, 0.7869825108926163, rtol=1e-11)


def test_fama_macbeth_nonrobust_golden_contract():
    X, y, _, time = _balanced_panel()
    model = FamaMacBeth(cov_type="nonrobust", device="cpu").fit(
        X, y, time_ids=time
    )
    assert_allclose(
        model.coef_,
        [0.44144492, 1.53311895, -0.72086425],
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        model.bse_,
        [0.06680718, 0.13928856, 0.12691874],
        rtol=RTOL,
        atol=ATOL,
    )
    assert model.n_periods == 5
    assert model.df_resid == 4
    prediction = model.predict(X[:3])
    assert isinstance(prediction, np.ndarray)
    _assert_summary_contract(model, "FamaMacBeth")


def test_fama_macbeth_newey_west_golden_scaling():
    X, y, _, time = _balanced_panel()
    model = FamaMacBeth(cov_type="newey-west", device="cpu").fit(
        X, y, time_ids=time
    )
    assert_allclose(
        model.bse_,
        [0.05134669, 0.05855774, 0.09904631],
        rtol=RTOL,
        atol=ATOL,
    )
