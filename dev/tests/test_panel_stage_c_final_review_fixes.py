"""Regression coverage for the final independent Panel Stage-C review fixes."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import PanelOLS, PooledOLS, RandomEffects
from statgpu.panel._utils import demean_variables


def _unbalanced_two_way(seed=20260814):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 8, 6
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    keep = np.ones(entity.size, dtype=bool)
    keep[[1, 7, 14, 20, 33, 46]] = False
    entity = entity[keep]
    time = time[keep]
    X = rng.normal(size=(entity.size, 2))
    alpha = rng.normal(scale=0.7, size=n_entities)
    tau = rng.normal(scale=0.4, size=n_times)
    y = (
        0.75 * X[:, 0]
        - 0.35 * X[:, 1]
        + alpha[entity]
        + tau[time]
        + rng.normal(scale=0.08, size=entity.size)
    )
    return X, y, entity, time


def _assert_two_way_means_zero(values, entity, time, *, atol=2e-9):
    for level in np.unique(entity):
        assert abs(float(np.mean(values[entity == level]))) <= atol
    for level in np.unique(time):
        assert abs(float(np.mean(values[time == level]))) <= atol


def test_panel_predict_rejects_one_missing_slope_without_fitted_constant():
    rng = np.random.default_rng(20260815)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([0.8, -0.25]) + rng.normal(scale=0.1, size=60)
    model = PanelOLS().fit(X, y)

    with pytest.raises(ValueError, match="incompatible feature count"):
        model.predict(X[:8, :1])


def test_random_effects_predict_rejects_one_missing_slope_without_fitted_constant():
    rng = np.random.default_rng(20260816)
    entity = np.repeat(np.arange(12), 5)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.25, size=12), 5)
    y = X @ np.array([0.6, -0.4]) + alpha + rng.normal(scale=0.1, size=entity.size)
    model = RandomEffects().fit(X, y, entity_ids=entity)

    with pytest.raises(ValueError, match="incompatible feature count"):
        model.predict(X[:8, :1])


@pytest.mark.parametrize("estimator_name", ["panel", "random-effects"])
def test_omitted_nonunit_explicit_constant_restores_exact_training_value(estimator_name):
    rng = np.random.default_rng(20260817)
    entity = np.repeat(np.arange(12), 5)
    x = rng.normal(size=entity.size)
    constant_value = 2.5
    X_full = np.column_stack([np.full(entity.size, constant_value), x])
    alpha = np.repeat(rng.normal(scale=0.2, size=12), 5)
    y = 0.7 + 0.9 * x + alpha + rng.normal(scale=0.1, size=entity.size)

    if estimator_name == "panel":
        model = PanelOLS().fit(X_full, y)
    else:
        model = RandomEffects().fit(X_full, y, entity_ids=entity)

    expected = model.predict(X_full[:10])
    actual = model.predict(x[:10, None])
    assert model._predict_constant_index == 0
    assert model._predict_constant_value == constant_value
    assert_allclose(actual, expected, rtol=0, atol=3e-12)


def test_short_prediction_with_constant_still_present_is_rejected_as_ambiguous():
    rng = np.random.default_rng(202608171)
    n = 70
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X_full = np.column_stack([np.full(n, 3.0), x1, x2])
    y = 0.4 + 0.7 * x1 - 0.2 * x2 + rng.normal(scale=0.08, size=n)
    model = PanelOLS().fit(X_full, y)

    # This matrix did not omit the constant; it omitted x2.  It must not be
    # reinterpreted as an omitted-constant prediction by shape alone.
    with pytest.raises(ValueError, match="ambiguous"):
        model.predict(X_full[:9, :2])


def test_random_effects_formula_prediction_all_one_slope_is_not_intercept_ambiguous():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    rng = np.random.default_rng(2026081701)
    entity = np.repeat(np.arange(10), 5)
    x = rng.normal(size=entity.size)
    alpha = np.repeat(rng.normal(scale=0.2, size=10), 5)
    y = 0.6 + 0.8 * x + alpha + rng.normal(scale=0.08, size=entity.size)
    data = pd.DataFrame({"y": y, "x": x, "entity": entity})
    model = RandomEffects().fit(formula="y ~ x | entity", data=data)

    new_data = pd.DataFrame({"x": np.ones(7, dtype=np.float64)})
    actual = model.predict(new_data)
    expected = np.full(7, model.coef_[0] + model.coef_[1], dtype=np.float64)
    assert_allclose(actual, expected, rtol=0, atol=3e-12)


def test_panel_failed_refit_invalidates_previous_fitted_state():
    rng = np.random.default_rng(2026082301)
    entity = np.repeat(np.arange(8), 5)
    X = rng.normal(size=(entity.size, 2))
    y = X @ np.array([0.6, -0.2]) + rng.normal(scale=0.1, size=entity.size)
    model = PanelOLS(entity_effects=True, cov_type="clustered").fit(
        X, y, entity_ids=entity, cluster=entity
    )
    assert model._fitted is True
    assert model.coef_ is not None
    assert model._inference_result is not None
    assert model._bse is not None
    assert model._tvalues is not None
    assert model._zvalues is not None
    assert model._pvalues is not None
    assert model._conf_int is not None
    assert model._inference_backend_name == model._backend_name

    with pytest.raises(ValueError, match="cluster length"):
        model.fit(
            X, y, entity_ids=entity, cluster=np.arange(entity.size - 1)
        )

    assert model._fitted is False
    assert model.coef_ is None
    assert model.fit_statistics_ is None
    assert model._inference_result is None
    assert model._bse is None
    assert model._tvalues is None
    assert model._zvalues is None
    assert model._pvalues is None
    assert model._conf_int is None
    assert model._panel_cov_params_raw is None
    assert model._covariance_metadata == {}
    assert model._backend_name is None
    assert model._inference_backend_name is None
    assert model._predict_backend_name is None
    assert model.nobs is None
    assert model._panel_index_info is None
    assert model._entity_effects_map == {}
    assert model._time_effects_map == {}
    assert model.entity_effects is True  # constructor contract restored
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X[:3], entity_ids=entity[:3])
    with pytest.raises(RuntimeError, match="not fitted"):
        model.summary()


def test_panel_failed_formula_refit_clears_prior_formula_metadata():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    rng = np.random.default_rng(2026082302)
    n = 30
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "entity": np.repeat(np.arange(6), 5),
            "time": np.tile(np.arange(5), 6),
            "industry": np.repeat(np.arange(3), 10),
        }
    )
    model = PanelOLS().fit(formula="y ~ x | entity", data=data)
    assert model._design_info is not None

    with pytest.raises(ValueError, match="at most two"):
        model.fit(formula="y ~ x | entity + time + industry", data=data)

    assert model._fitted is False
    assert model.coef_ is None
    assert model._design_info is None
    assert model.entity_effects is False
    assert model.time_effects is False


def test_panel_effects_only_formula_has_explicit_failure_mode():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    n = 24
    data = pd.DataFrame(
        {
            "y": np.linspace(-1.0, 1.0, n),
            "entity": np.repeat(np.arange(6), 4),
        }
    )
    model = PanelOLS().fit(np.linspace(-0.5, 0.5, n)[:, None], data["y"].to_numpy())
    with pytest.raises(ValueError, match="effects-only formulas are not supported"):
        model.fit(formula="y ~ 1 | entity", data=data)
    assert model._fitted is False
    assert model.entity_effects is False
    assert model.time_effects is False
    assert model.get_params()["entity_effects"] is False
    assert model.get_params()["time_effects"] is False
    assert model._backend_name is None
    assert model.nobs is None
    assert model._design_info is None


def test_panel_formula_without_fixed_effects_preserves_patsy_intercept():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    rng = np.random.default_rng(2026082201)
    x = rng.normal(size=90)
    y = 1.35 + 0.72 * x + rng.normal(scale=0.12, size=x.size)
    data = pd.DataFrame({"y": y, "x": x})

    panel = PanelOLS(cov_type="nonrobust").fit(formula="y ~ x", data=data)
    pooled = PooledOLS(cov_type="nonrobust").fit(formula="y ~ x", data=data)

    assert panel._feature_names == ["Intercept", "x"]
    assert panel._predict_constant_index == 0
    assert_allclose(panel.coef_, pooled.coef_, rtol=0, atol=3e-12)
    assert_allclose(panel.bse_, pooled.bse_, rtol=0, atol=3e-12)
    assert_allclose(panel.predict(data.iloc[:12]), pooled.predict(data.iloc[:12]), rtol=0, atol=3e-12)
    for field in ("rsquared_overall", "rsquared_adj", "f_statistic", "f_pvalue"):
        assert_allclose(
            getattr(panel.fit_statistics_, field),
            getattr(pooled.fit_statistics_, field),
            rtol=0,
            atol=3e-12,
        )
    assert panel.fit_statistics_.f_df == pooled.fit_statistics_.f_df
    assert panel.df_resid == pooled.df_resid
    assert panel.fit_statistics_.metadata["diagnostic_df"]["df_total"] == x.size - 1


def test_panel_formula_explicit_no_intercept_remains_no_intercept():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    rng = np.random.default_rng(2026082202)
    x = rng.normal(size=70)
    y = 0.8 * x + rng.normal(scale=0.1, size=x.size)
    data = pd.DataFrame({"y": y, "x": x})

    formula_model = PanelOLS().fit(formula="y ~ 0 + x", data=data)
    array_model = PanelOLS().fit(x[:, None], y)
    assert formula_model._feature_names == ["x"]
    assert formula_model._predict_constant_index is None
    assert_allclose(formula_model.coef_, array_model.coef_, rtol=0, atol=3e-12)
    assert_allclose(formula_model.predict(data.iloc[:10]), array_model.predict(x[:10, None]), rtol=0, atol=3e-12)


def test_panel_explicit_level_constant_fit_statistics_match_pooled():
    rng = np.random.default_rng(2026082203)
    x = rng.normal(size=84)
    y = -0.9 + 0.55 * x + rng.normal(scale=0.14, size=x.size)
    X_full = np.column_stack([np.ones(x.size), x])
    panel = PanelOLS(cov_type="nonrobust").fit(X_full, y)
    pooled = PooledOLS(cov_type="nonrobust").fit(x[:, None], y)

    assert_allclose(panel.coef_, pooled.coef_, rtol=0, atol=3e-12)
    assert_allclose(panel.bse_, pooled.bse_, rtol=0, atol=3e-12)
    for field in ("rsquared_overall", "rsquared_adj", "f_statistic", "f_pvalue"):
        assert_allclose(
            getattr(panel.fit_statistics_, field),
            getattr(pooled.fit_statistics_, field),
            rtol=0,
            atol=3e-12,
        )
    assert panel.fit_statistics_.f_df == pooled.fit_statistics_.f_df
    assert panel.fit_statistics_.metadata["diagnostic_df"]["df_total"] == x.size - 1


def test_panel_no_intercept_rsquared_within_uses_uncentered_tss():
    rng = np.random.default_rng(2026082402)
    x = rng.normal(loc=0.8, scale=1.1, size=90)
    y = 2.0 + 0.65 * x + rng.normal(scale=0.2, size=x.size)
    X = x[:, None]
    model = PanelOLS().fit(X, y)
    resid = y - X @ model.coef_
    expected = 1.0 - float(resid @ resid) / float(y @ y)
    assert_allclose(model.rsquared_within, expected, rtol=0, atol=3e-12)
    assert_allclose(
        model.fit_statistics_.rsquared_overall, expected, rtol=0, atol=3e-12
    )


def test_one_way_fixed_effect_prediction_restores_grand_mean_only_for_known_labels():
    rng = np.random.default_rng(2026082101)
    entity = np.repeat(np.arange(8), 5)
    X = rng.normal(size=(entity.size, 2))
    alpha = rng.normal(loc=1.4, scale=0.3, size=8)
    y = 0.7 * X[:, 0] - 0.25 * X[:, 1] + alpha[entity]
    model = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)

    known = model.predict(X, entity_ids=entity)
    dummies = np.column_stack(
        [(entity == level).astype(np.float64) for level in np.unique(entity)]
    )
    design = np.column_stack([X, dummies])
    reference = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    assert_allclose(known, reference, rtol=0, atol=2e-9)

    # Unknown FE labels retain the documented linear-only fallback; the grand
    # mean is restored only for rows that actually use a fitted FE value.
    unknown = model.predict(X[:4], entity_ids=np.full(4, 999))
    assert_allclose(unknown, X[:4] @ model.coef_, rtol=0, atol=3e-12)


def test_formula_enabled_effects_do_not_leak_into_later_refit():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    rng = np.random.default_rng(2026082102)
    entity = np.repeat(np.arange(8), 4)
    time = np.tile(np.arange(4), 8)
    x = rng.normal(size=entity.size)
    y = 0.8 * x + rng.normal(scale=0.1, size=entity.size)
    data = pd.DataFrame({"y": y, "x": x, "entity": entity, "time": time})

    model = PanelOLS()
    model.fit(formula="y ~ x | entity + time", data=data)
    assert model.entity_effects is True
    assert model.time_effects is True

    model.fit(X=x[:, None], y=y)
    assert model.entity_effects is False
    assert model.time_effects is False
    assert model.coef_ is not None


def test_formula_more_than_two_fixed_effects_fails_closed():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("patsy")
    rng = np.random.default_rng(2026082103)
    n = 24
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "entity": np.repeat(np.arange(6), 4),
            "time": np.tile(np.arange(4), 6),
            "industry": np.repeat(np.arange(3), 8),
        }
    )
    with pytest.raises(ValueError, match="at most two"):
        PanelOLS().fit(
            formula="y ~ x | entity + time + industry",
            data=data,
        )


def test_unbalanced_two_way_prediction_uses_joint_fixed_effect_solution():
    X, y, entity, time = _unbalanced_two_way()
    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    prediction = model.predict(X, entity_ids=entity, time_ids=time)

    # Level prediction must equal the joint least-squares projection on X
    # plus the two fixed-effect dummy spaces, including the common grand mean.
    entity_levels = np.unique(entity)
    time_levels = np.unique(time)
    dummies = [X]
    dummies.extend((entity == level).astype(np.float64)[:, None] for level in entity_levels)
    dummies.extend((time == level).astype(np.float64)[:, None] for level in time_levels)
    design = np.column_stack(dummies)
    reference = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    assert_allclose(prediction, reference, rtol=0, atol=2e-9)

    residual = y - prediction
    _assert_two_way_means_zero(residual, entity, time)


def test_connected_two_way_prediction_rejects_one_sided_or_known_unknown_effects():
    rng = np.random.default_rng(2026082401)
    entity = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    time = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)
    X = rng.normal(size=(entity.size, 1))
    alpha = np.array([0.4, -0.3, 0.8])
    tau = np.array([0.25, -0.15, 0.45])
    y = 0.75 * X[:, 0] + alpha[entity] + tau[time]
    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert model.fit_statistics_.metadata["diagnostic_df"]["incidence_components"] == 1
    assert np.all(np.isfinite(model.predict(
        X[:1], entity_ids=np.array([0]), time_ids=np.array([1])
    )))
    for kwargs in (
        {"entity_ids": np.array([0])},
        {"time_ids": np.array([0])},
        {"entity_ids": np.array([0]), "time_ids": np.array([99])},
        {"entity_ids": np.array([99]), "time_ids": np.array([0])},
    ):
        with pytest.raises(ValueError, match="both entity and time labels are known"):
            model.predict(X[:1], **kwargs)
    both_unseen = model.predict(
        X[:1], entity_ids=np.array([98]), time_ids=np.array([99])
    )
    assert_allclose(both_unseen, X[:1] @ model.coef_, rtol=0, atol=3e-12)


def test_disconnected_two_way_prediction_rejects_cross_component_effect_sum():
    rng = np.random.default_rng(202608172)
    # Two disconnected incidence components: entities 0/1 only meet times 0/1,
    # while entities 2/3 only meet times 2/3.
    entity = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    time = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    X = rng.normal(size=(entity.size, 1))
    alpha = np.array([0.5, -0.2, 1.1, -0.7])
    tau = np.array([0.25, -0.15, 0.6, -0.4])
    y = 0.8 * X[:, 0] + alpha[entity] + tau[time]

    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    observed = model.predict(X, entity_ids=entity, time_ids=time)
    assert np.all(np.isfinite(observed))
    assert model.fit_statistics_.metadata["diagnostic_df"]["incidence_components"] == 2

    with pytest.raises(ValueError, match="both entity and time labels are known"):
        model.predict(
            X[:1],
            entity_ids=np.array([0]),
            time_ids=np.array([2]),
        )
    with pytest.raises(ValueError, match="both entity and time labels are known"):
        model.predict(X[:1], entity_ids=np.array([0]))
    with pytest.raises(ValueError, match="both entity and time labels are known"):
        model.predict(X[:1], time_ids=np.array([0]))
    with pytest.raises(ValueError, match="both entity and time labels are known"):
        model.predict(
            X[:1],
            entity_ids=np.array([0]),
            time_ids=np.array([99]),
        )
    with pytest.raises(ValueError, match="both entity and time labels are known"):
        model.predict(
            X[:1],
            entity_ids=np.array([99]),
            time_ids=np.array([0]),
        )

    # If both labels are unseen, no fitted fixed effect is used and the historical
    # zero-effect fallback remains a well-defined linear prediction.
    unknown_pair = model.predict(
        X[:1], entity_ids=np.array([98]), time_ids=np.array([99])
    )
    assert np.all(np.isfinite(unknown_pair))

    # A not-yet-observed entity/time pair inside one connected component remains
    # identified by the additive two-way fit and is therefore allowed.
    within_component = model.predict(
        X[:1],
        entity_ids=np.array([1]),
        time_ids=np.array([1]),
    )
    assert np.all(np.isfinite(within_component))


def test_weakly_connected_two_way_panel_converges_beyond_legacy_100_iterations():
    rng = np.random.default_rng(20260818)
    n_entities = 30
    entity = np.repeat(np.arange(n_entities), 3)
    time = np.column_stack(
        [
            np.arange(n_entities),
            np.arange(1, n_entities + 1),
            np.arange(2, n_entities + 2),
        ]
    ).ravel()
    X = rng.normal(size=(entity.size, 2))
    y = rng.normal(size=entity.size)

    with pytest.raises(RuntimeError, match="did not converge"):
        demean_variables(
            y,
            X,
            entity,
            time,
            xp=np,
            max_iter=100,
            tol=1e-10,
        )

    y_d, X_d = demean_variables(
        y,
        X,
        entity,
        time,
        xp=np,
        max_iter=10_000,
        tol=1e-10,
    )
    _assert_two_way_means_zero(y_d, entity, time, atol=3e-10)
    for column in range(X_d.shape[1]):
        _assert_two_way_means_zero(X_d[:, column], entity, time, atol=3e-10)


def test_panel_exposes_fail_closed_two_way_convergence_controls():
    rng = np.random.default_rng(202608181)
    n_entities = 30
    entity = np.repeat(np.arange(n_entities), 3)
    time = np.column_stack(
        [
            np.arange(n_entities),
            np.arange(1, n_entities + 1),
            np.arange(2, n_entities + 2),
        ]
    ).ravel()
    X = rng.normal(size=(entity.size, 2))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.2, size=entity.size)

    with pytest.raises(RuntimeError, match="did not converge"):
        PanelOLS(
            entity_effects=True,
            time_effects=True,
            demean_max_iter=100,
            demean_tol=1e-10,
        ).fit(X, y, entity_ids=entity, time_ids=time)

    model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        demean_max_iter=10_000,
        demean_tol=1e-10,
    ).fit(X, y, entity_ids=entity, time_ids=time)
    assert model.demean_max_iter == 10_000
    assert model.demean_tol == 1e-10
    assert model.get_params()["demean_max_iter"] == 10_000
    assert model.get_params()["demean_tol"] == 1e-10

    with pytest.raises(ValueError, match="positive integer"):
        PanelOLS(demean_max_iter=0)
    with pytest.raises(ValueError, match="finite and positive"):
        PanelOLS(demean_tol=0.0)
    with pytest.raises(ValueError, match="positive integer"):
        PanelOLS().set_params(demean_max_iter=False)


def test_numerically_absorbed_two_way_direction_terminates_without_relative_zero_trap():
    n_entities = 16
    entity = np.repeat(np.arange(n_entities), 2)
    time = np.column_stack(
        [np.arange(n_entities), np.arange(1, n_entities + 1)]
    ).ravel()
    entity_effect = np.linspace(-1.0, 1.0, n_entities)
    time_effect = np.linspace(0.5, -0.5, n_entities + 1)
    absorbed = entity_effect[entity] + time_effect[time]
    X = absorbed[:, None]

    y_d, X_d = demean_variables(
        absorbed,
        X,
        entity,
        time,
        xp=np,
        max_iter=50_000,
        tol=1e-10,
    )
    assert np.max(np.abs(y_d)) < 5e-14
    assert np.max(np.abs(X_d)) < 5e-14


def test_physical_stage_c_runner_covers_new_prediction_contracts_on_numpy():
    import importlib.util
    from pathlib import Path

    runner_path = Path(__file__).parents[1] / "benchmarks" / "validate_panel_stage_c_gpu.py"
    spec = importlib.util.spec_from_file_location("stage_c_gpu_validation_review", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    X, y, entity, time, clusters = module._dataset()
    models = module._fit_cases(X, y, entity, time, clusters, "numpy")
    assert models["panel_entity_hc0"]._physical_prediction_contract == (
        "entity_effect_prediction"
    )
    assert models["panel_two_way_hc3"]._physical_prediction_contract == (
        "two_way_effect_prediction"
    )
    re_model = models["random_effects_explicit_constant_hc0"]
    assert re_model._physical_prediction_contract == "omitted_explicit_constant"
    assert re_model._predict_constant_index == 0
    assert_allclose(
        re_model._physical_prediction,
        re_model.predict(X[:8]),
        rtol=0,
        atol=3e-12,
    )

    constant_audit = module._level_constant_contract_audit(
        "numpy", rtol=0, atol=3e-12
    )
    assert constant_audit["status"] == "success"
    assert constant_audit["executed_backend"] == "numpy"
    assert constant_audit["prediction_backend"] == "numpy"
    assert constant_audit["constant_index"] == 0
    assert constant_audit["constant_value"] == 1.0

    connected_audit = module._connected_two_way_prediction_audit("numpy")
    assert connected_audit["status"] == "success"
    assert connected_audit["executed_backend"] == "numpy"
    assert connected_audit["prediction_backend"] == "numpy"
    assert all(connected_audit["guards"].values())
    assert np.all(np.isfinite(connected_audit["both_known"]))
    assert np.all(np.isfinite(connected_audit["both_unseen"]))

    audit = module._disconnected_two_way_prediction_audit("numpy")
    assert audit["executed_backend"] == "numpy"
    assert audit["prediction_backend"] == "numpy"
    assert all(audit["guards"].values())
    assert np.all(np.isfinite(audit["observed"]))
    assert np.all(np.isfinite(audit["same_component"]))
    assert np.all(np.isfinite(audit["both_unseen"]))


def test_torch_cpu_two_way_projection_and_prediction_match_numpy():
    torch = pytest.importorskip("torch")
    X, y, entity, time = _unbalanced_two_way(20260819)
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    entity_t = torch.as_tensor(entity, dtype=torch.int64)
    time_t = torch.as_tensor(time, dtype=torch.int64)

    # The maintained CPU gate exercises the backend-neutral projection kernels
    # directly with xp=torch.  Explicit device="torch" is intentionally CUDA-
    # only by project contract and must not be used in a CPU-only environment.
    y_np, X_np = demean_variables(y, X, entity, time, xp=np, tol=1e-11)
    y_t_out, X_t_out = demean_variables(
        y_t, X_t, entity_t, time_t, xp=torch, tol=1e-11
    )
    assert_allclose(y_t_out.detach().cpu().numpy(), y_np, rtol=0, atol=3e-11)
    assert_allclose(X_t_out.detach().cpu().numpy(), X_np, rtol=0, atol=3e-11)

    # Estimator integration follows the existing Torch-CPU suite convention:
    # Torch tensors are accepted without pretending that a CUDA Torch device is
    # available.  The direct kernel checks above carry the Torch parity claim.
    expected = PanelOLS(entity_effects=True, time_effects=True).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    actual = PanelOLS(entity_effects=True, time_effects=True).fit(
        X_t, y_t, entity_ids=entity_t, time_ids=time_t
    )
    expected_prediction = expected.predict(X, entity_ids=entity, time_ids=time)
    actual_prediction = actual.predict(
        X_t, entity_ids=entity_t, time_ids=time_t
    )
    assert_allclose(actual.coef_, expected.coef_, rtol=2e-9, atol=2e-11)
    assert_allclose(actual_prediction, expected_prediction, rtol=2e-9, atol=2e-10)
