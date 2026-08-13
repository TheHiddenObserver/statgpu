"""Regression coverage for the final independent Panel Stage-C review fixes."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import PanelOLS, RandomEffects
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


def test_unbalanced_two_way_prediction_uses_joint_fixed_effect_solution():
    X, y, entity, time = _unbalanced_two_way()
    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X, y, entity_ids=entity, time_ids=time
    )
    prediction = model.predict(X, entity_ids=entity, time_ids=time)

    # PanelOLS historically excludes the separately stored grand mean from
    # predict().  After adding it back, fitted values must equal the joint
    # least-squares projection on X plus the two fixed-effect dummy spaces.
    entity_levels = np.unique(entity)
    time_levels = np.unique(time)
    dummies = [X]
    dummies.extend((entity == level).astype(np.float64)[:, None] for level in entity_levels)
    dummies.extend((time == level).astype(np.float64)[:, None] for level in time_levels)
    design = np.column_stack(dummies)
    reference = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    assert_allclose(
        prediction + model._grand_mean,
        reference,
        rtol=0,
        atol=2e-9,
    )

    residual = y - model._grand_mean - prediction
    _assert_two_way_means_zero(residual, entity, time)


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

    with pytest.raises(ValueError, match="not identified on a disconnected incidence graph"):
        model.predict(
            X[:1],
            entity_ids=np.array([0]),
            time_ids=np.array([2]),
        )
    with pytest.raises(ValueError, match="not identified on a disconnected incidence graph"):
        model.predict(X[:1], entity_ids=np.array([0]))
    with pytest.raises(ValueError, match="not identified on a disconnected incidence graph"):
        model.predict(X[:1], time_ids=np.array([0]))
    with pytest.raises(ValueError, match="not identified on a disconnected incidence graph"):
        model.predict(
            X[:1],
            entity_ids=np.array([0]),
            time_ids=np.array([99]),
        )
    with pytest.raises(ValueError, match="not identified on a disconnected incidence graph"):
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
