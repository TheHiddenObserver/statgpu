from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import BetweenOLS, FirstDifferenceOLS, PanelOLS, PooledOLS, RandomEffects
from statgpu.panel._utils import demean_variables


def _explicit_two_way_residual(values, entity, time):
    entity_levels = np.unique(entity)
    time_levels = np.unique(time)
    cols = [np.ones(len(entity), dtype=np.float64)]
    cols.extend((entity == level).astype(np.float64) for level in entity_levels[1:])
    cols.extend((time == level).astype(np.float64) for level in time_levels[1:])
    design = np.column_stack(cols)
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coef


def test_two_way_demeaning_waits_for_x_even_when_y_is_already_projected():
    entity = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    time = np.array([0, 1, 3, 0, 2, 1, 2, 3, 0, 2, 3], dtype=np.int64)
    rng = np.random.default_rng(12920)
    raw_y = rng.normal(size=len(entity))
    y = _explicit_two_way_residual(raw_y, entity, time)
    X = rng.normal(size=(len(entity), 2))

    y_d, X_d = demean_variables(y, X, entity, time, xp=np, max_iter=200, tol=1e-12)
    expected_y = _explicit_two_way_residual(y, entity, time)
    expected_X = np.column_stack(
        [_explicit_two_way_residual(X[:, j], entity, time) for j in range(X.shape[1])]
    )
    assert_allclose(y_d, expected_y, rtol=0, atol=2e-11)
    assert_allclose(X_d, expected_X, rtol=0, atol=2e-11)

    with pytest.raises(RuntimeError, match="did not converge"):
        demean_variables(y, X, entity, time, xp=np, max_iter=1, tol=1e-12)


def test_rank_deficient_fits_keep_fit_space_but_disable_coordinate_inference():
    rng = np.random.default_rng(12921)
    n_entities, n_times = 12, 4
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    x = rng.normal(size=entity.size)
    X = np.column_stack([x, 2.0 * x])
    y = 0.4 + 0.8 * x + np.repeat(rng.normal(scale=0.3, size=n_entities), n_times)
    y += rng.normal(scale=0.15, size=entity.size)

    models = [
        PooledOLS(cov_type="hc0").fit(X, y, entity_ids=entity),
        PanelOLS(entity_effects=True, cov_type="hc0").fit(X, y, entity_ids=entity),
        BetweenOLS(cov_type="hc0").fit(X, y, entity_ids=entity),
        FirstDifferenceOLS(cov_type="hc0").fit(X, y, entity_ids=entity, time_ids=time),
        RandomEffects(cov_type="hc0").fit(
            np.column_stack([np.ones(len(y)), X]), y, entity_ids=entity
        ),
    ]

    for model in models:
        assert model.coef_ is not None
        assert model._coefficient_inference_available is False
        assert model.bse_ is None
        assert model.tvalues_ is None
        assert model.pvalues_ is None
        assert model.conf_int_ is None
        assert model._inference_result.metadata["applicable"] is False
        assert "rank deficient" in model._inference_result.metadata["reason"]
        with pytest.raises(ValueError, match="rank deficient"):
            model.summary()


def test_full_rank_inference_contract_is_unchanged():
    rng = np.random.default_rng(12922)
    X = rng.normal(size=(60, 2))
    y = 0.4 + X @ np.array([0.7, -0.2]) + rng.normal(scale=0.2, size=60)
    model = PooledOLS(cov_type="hc0").fit(X, y)
    assert model._coefficient_inference_available is True
    assert model.bse_ is not None
    assert model.pvalues_ is not None
    assert model.conf_int_ is not None


def test_first_difference_rejects_duplicate_entity_time_pairs():
    X = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 0.0, 1.5, 3.0])
    entity = np.array([0, 0, 0, 1, 1, 1])
    time = np.array([0, 0, 2, 0, 1, 3])
    with pytest.raises(ValueError, match=r"unique \(entity_id, time_id\)"):
        FirstDifferenceOLS().fit(X, y, entity_ids=entity, time_ids=time)


def test_first_difference_keeps_consecutive_observed_gap_semantics():
    X = np.array([[0.0], [2.0], [5.0], [1.0], [4.0], [8.0]])
    y = np.array([0.0, 2.0, 5.0, 1.0, 4.0, 8.0])
    entity = np.array([0, 0, 0, 1, 1, 1])
    time = np.array([1, 3, 7, 1, 4, 9])
    model = FirstDifferenceOLS().fit(X, y, entity_ids=entity, time_ids=time)
    assert_allclose(model.coef_, np.array([1.0]), rtol=0, atol=1e-12)


def test_panel_predict_full_rank_effect_semantics_remain_numpy_visible():
    rng = np.random.default_rng(12923)
    entity = np.repeat(np.arange(8), 4)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(np.linspace(-0.3, 0.4, 8), 4)
    y = X @ np.array([0.6, -0.25]) + alpha
    model = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)
    pred = model.predict(X[:8], entity_ids=entity[:8])
    assert isinstance(pred, np.ndarray)
    assert model._predict_backend_name == "numpy"
    assert pred.shape == (8,)
