"""Regression coverage for the final PR126 code-review repair cycle."""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from statgpu.panel import (
    BetweenOLS,
    FamaMacBeth,
    FirstDifferenceOLS,
    PooledOLS,
    RandomEffects,
)


def _panel(seed=12690, *, n_entities=10, n_times=4):
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.35, size=n_entities), n_times)
    y = 0.4 + 0.8 * X[:, 0] - 0.3 * X[:, 1] + alpha
    y += rng.normal(scale=0.2, size=entity.size)
    return X, y, entity, time


def _hac_chronology_fixture(seed=12691):
    rng = np.random.default_rng(seed)
    n_entities = 16
    labels = np.tile(np.array(["t1", "t2", "t10"], dtype=object), n_entities)
    numeric = np.tile(np.arange(3), n_entities)
    ordered = pd.Categorical(
        labels,
        categories=["t1", "t2", "t10"],
        ordered=True,
    )
    x = rng.normal(size=labels.size)
    time_shock = np.tile(np.array([0.0, 1.4, -0.9]), n_entities)
    y = 0.5 + 0.7 * x + time_shock + rng.normal(scale=0.12, size=labels.size)
    return x[:, None], y, ordered, numeric


def test_pooled_legacy_hac_preserves_ordered_categorical_chronology():
    X, y, ordered, numeric = _hac_chronology_fixture()

    categorical = PooledOLS(cov_type="hac", bandwidth=4).fit(
        X, y, time_index=ordered
    )
    chronological = PooledOLS(cov_type="hac", bandwidth=4).fit(
        X, y, time_index=numeric
    )
    lexical = PooledOLS(cov_type="hac", bandwidth=4).fit(
        X,
        y,
        time_index=np.asarray(ordered, dtype=object),
    )

    np.testing.assert_allclose(
        categorical.coef_, chronological.coef_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        categorical.bse_, chronological.bse_, rtol=2e-13, atol=2e-15
    )
    np.testing.assert_allclose(
        categorical.pvalues_, chronological.pvalues_, rtol=2e-13, atol=2e-15
    )
    assert not np.allclose(
        categorical.bse_, lexical.bse_, rtol=1e-10, atol=1e-12
    )


def test_pooled_formula_hac_keeps_chronology_after_missing_row_alignment():
    X, y, ordered, numeric = _hac_chronology_fixture(seed=12692)
    x = X[:, 0].copy()
    x[[5, 28]] = np.nan
    data = pd.DataFrame({"y": y, "x": x})

    categorical = PooledOLS(cov_type="hac", bandwidth=3).fit(
        formula="y ~ x",
        data=data,
        time_index=ordered,
    )
    chronological = PooledOLS(cov_type="hac", bandwidth=3).fit(
        formula="y ~ x",
        data=data,
        time_index=numeric,
    )

    np.testing.assert_allclose(
        categorical.coef_, chronological.coef_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        categorical.bse_, chronological.bse_, rtol=2e-13, atol=2e-15
    )


def test_pooled_legacy_hac_rejects_missing_ordered_categorical_time():
    X = np.arange(8.0).reshape(-1, 1)
    y = 0.5 + 0.2 * X[:, 0]
    time_index = pd.Categorical(
        ["t1", "t2", "t10", "t1", None, "t10", "t1", "t2"],
        categories=["t1", "t2", "t10"],
        ordered=True,
    )

    # The repository-wide public finite guard may reject this before the
    # panel-specific chronology factorizer runs. Both paths are deliberately
    # fail-closed, so assert the public contract rather than one internal layer's
    # exact wording.
    with pytest.raises(ValueError, match="finite|missing"):
        PooledOLS(cov_type="hac", bandwidth=1).fit(
            X,
            y,
            time_index=time_index,
        )


def _assert_failed_refit_clears_public_surface(model, X):
    assert model.__sklearn_is_fitted__() is False
    assert model.fit_statistics_ is None
    for name in (
        "coef_",
        "bse_",
        "tvalues_",
        "pvalues_",
        "conf_int_",
        "_inference_result",
    ):
        assert not hasattr(model, name), name
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X[:3])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.summary()


def test_pooled_failed_refit_clears_previous_inference():
    X, y, entity, _time = _panel(seed=12693)
    model = PooledOLS(cov_type="clustered").fit(X, y, cluster=entity)
    assert model._inference_result is not None

    with pytest.raises(ValueError, match="cluster is required"):
        model.fit(X, y)

    _assert_failed_refit_clears_public_surface(model, X)


def test_between_failed_refit_without_entity_ids_clears_previous_inference():
    X, y, entity, _time = _panel(seed=12694)
    model = BetweenOLS(cov_type="hc0").fit(X, y, entity_ids=entity)
    assert model._inference_result is not None

    with pytest.raises(ValueError, match="entity_ids is required"):
        model.fit(X, y)

    _assert_failed_refit_clears_public_surface(model, X)


def test_first_difference_failed_refit_without_entity_ids_clears_previous_inference():
    X, y, entity, time = _panel(seed=12695)
    model = FirstDifferenceOLS(cov_type="hc0").fit(
        X,
        y,
        entity_ids=entity,
        time_ids=time,
    )
    assert model._inference_result is not None

    with pytest.raises(ValueError, match="entity_ids is required"):
        model.fit(X, y, time_ids=time)

    _assert_failed_refit_clears_public_surface(model, X)


def test_random_effects_failed_refit_clears_previous_inference():
    X, y, entity, _time = _panel(seed=12696)
    model = RandomEffects(cov_type="hc0").fit(X, y, entity_ids=entity)
    assert model._inference_result is not None
    model.predict(X[:3])
    assert hasattr(model, "_predict_backend_name")

    with pytest.raises(ValueError, match="entity_ids is required"):
        model.fit(X, y)

    _assert_failed_refit_clears_public_surface(model, X)
    assert not hasattr(model, "theta_")
    assert not hasattr(model, "variance_components_")
    assert not hasattr(model, "_predict_backend_name")


@pytest.mark.parametrize("kind", ["pooled", "between", "first_difference", "random_effects"])
def test_late_refit_failure_clears_partial_new_inference(monkeypatch, kind):
    """Even an exception after inference storage must leave the model unfitted."""
    X, y, entity, time = _panel(seed=12700)

    if kind == "pooled":
        model = PooledOLS(cov_type="hc0")
        fit_kwargs = {}
    elif kind == "between":
        model = BetweenOLS(cov_type="hc0")
        fit_kwargs = {"entity_ids": entity}
    elif kind == "first_difference":
        model = FirstDifferenceOLS(cov_type="hc0")
        fit_kwargs = {"entity_ids": entity, "time_ids": time}
    else:
        model = RandomEffects(cov_type="hc0")
        fit_kwargs = {"entity_ids": entity}

    model.fit(X, y, **fit_kwargs)
    assert model._inference_result is not None

    import statgpu.panel._diagnostic_context as diagnostic_context

    def _late_failure(*args, **kwargs):
        raise RuntimeError("forced late panel fit failure")

    monkeypatch.setattr(
        diagnostic_context,
        "build_model_fit_statistics",
        _late_failure,
    )
    with pytest.raises(RuntimeError, match="forced late panel fit failure"):
        model.fit(X, y, **fit_kwargs)

    _assert_failed_refit_clears_public_surface(model, X)
    if kind == "random_effects":
        assert not hasattr(model, "theta_")
        assert not hasattr(model, "variance_components_")


def test_fama_macbeth_keeps_only_zero_length_prediction_device_anchor():
    X, y, _entity, time = _panel(seed=12697, n_entities=12, n_times=4)
    model = FamaMacBeth(device="cpu", bandwidth=1).fit(X, y, time_ids=time)

    assert tuple(model._fit_ref_.shape) == (0,)
    prediction = np.asarray(model.predict(X[:6]), dtype=np.float64)
    expected = np.column_stack([np.ones(6), X[:6]]) @ np.asarray(model.coef_)
    np.testing.assert_allclose(prediction, expected, rtol=2e-12, atol=2e-13)
