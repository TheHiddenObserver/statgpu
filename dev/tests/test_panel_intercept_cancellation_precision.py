"""Regression coverage for cancellation-safe panel response projections."""
from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import BetweenOLS, PooledOLS, RandomEffects
from statgpu.panel._intercept import panel_lstsq_exact_constant
from statgpu.panel._linalg import panel_lstsq


def _fixture():
    amplitude = float(2.0**55)
    X = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    y = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    return X, y, amplitude


def _assert_coefficients(model, amplitude):
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    # The SVD basis itself is rounded, so the restored coefficient need not be
    # bitwise identical to the binary representation of 1/3.  Keep the oracle at
    # a few float64 ulps; the historical cancellation bug returned exactly zero.
    np.testing.assert_allclose(
        coef[0], 1.0 / 3.0, rtol=4.0 * np.finfo(np.float64).eps, atol=0.0
    )
    np.testing.assert_allclose(coef[1], -amplitude, rtol=2.0e-15, atol=0.0)


def _random_effects_component_loss_fixture():
    amplitude = float(2.0**55)
    within = 8.0
    levels = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    y = np.concatenate(
        [np.asarray([level + within, level - within]) for level in levels]
    )
    X = np.ones((y.size, 1), dtype=np.float64)
    entity = np.repeat(np.arange(levels.size, dtype=np.int64), 2)
    return X, y, entity


def _random_effects_common_scale_loss_fixture(*, amplitude, tiny_within):
    y = np.asarray(
        [amplitude, amplitude, tiny_within, -tiny_within, -amplitude, -amplitude],
        dtype=np.float64,
    )
    X = np.ones((y.size, 1), dtype=np.float64)
    entity = np.repeat(np.arange(3, dtype=np.int64), 2)
    return X, y, entity


def _random_effects_theta_rounding_fixture():
    # The within/between residual ratio is about 1e-34.  Its square root is a
    # representable ~1e-17 complement, while 1-complement rounds to theta == 1.
    return _random_effects_common_scale_loss_fixture(
        amplitude=1.0e17,
        tiny_within=1.0,
    )


def test_pooled_ols_preserves_cancellation_tail_in_automatic_intercept_numpy():
    X, y, amplitude = _fixture()
    model = PooledOLS().fit(X, y)
    _assert_coefficients(model, amplitude)


def test_between_ols_preserves_cancellation_tail_in_automatic_intercept_numpy():
    X, y, amplitude = _fixture()
    X_level = np.repeat(X, 2, axis=0)
    y_level = np.repeat(y, 2)
    entity = np.repeat(np.arange(3), 2)
    model = BetweenOLS().fit(X_level, y_level, entity_ids=entity)
    _assert_coefficients(model, amplitude)


def test_exact_constant_helper_preserves_rank_deficient_minimum_norm_contract_numpy():
    y = np.asarray([3.0, 4.0, 5.0], dtype=np.float64)
    X = np.column_stack([np.ones(3), np.ones(3)])
    expected_params, expected_rank = panel_lstsq(X, y, np)
    actual_params, actual_rank = panel_lstsq_exact_constant(
        X, y, np, constant_index=0
    )
    assert actual_rank == expected_rank == 1
    np.testing.assert_array_equal(actual_params, expected_params)


def test_exact_constant_helper_keeps_risky_rank_deficient_minimum_norm_direction_numpy():
    amplitude = float(2.0**55)
    y = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    X = np.column_stack([np.ones(3), np.ones(3)])
    params, rank = panel_lstsq_exact_constant(X, y, np, constant_index=0)
    assert rank == 1
    np.testing.assert_allclose(params, np.full(2, 1.0 / 6.0), rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(params[0], params[1], rtol=0.0, atol=0.0)


def test_random_effects_fails_closed_when_quasi_demeaning_discards_component_numpy():
    X, y, entity = _random_effects_component_loss_fixture()
    with pytest.raises(
        FloatingPointError,
        match="quasi-demeaning exceeds the float64 component range",
    ):
        RandomEffects().fit(X, y, entity_ids=entity)


@pytest.mark.parametrize(
    ("amplitude", "tiny_within"),
    [
        (1.0e308, 1.0e-100),
        (1.0e100, 1.0e-100),
    ],
)
def test_random_effects_fails_closed_when_common_rss_scale_loses_within_variance_numpy(
    amplitude, tiny_within
):
    X, y, entity = _random_effects_common_scale_loss_fixture(
        amplitude=amplitude,
        tiny_within=tiny_within,
    )
    with pytest.raises(
        FloatingPointError,
        match="variance-component scaling exceeds the float64 common-residual range",
    ):
        RandomEffects().fit(X, y, entity_ids=entity)


def test_random_effects_preserves_pre_rounded_theta_complement_for_loss_certificate_numpy():
    X, y, entity = _random_effects_theta_rounding_fixture()
    with pytest.raises(
        FloatingPointError,
        match="quasi-demeaning exceeds the float64 component range",
    ):
        RandomEffects().fit(X, y, entity_ids=entity)


def test_torch_cpu_public_automatic_intercepts_match_numpy_cancellation_tail():
    torch = pytest.importorskip("torch")
    X, y, amplitude = _fixture()

    pooled = PooledOLS().fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
    )
    _assert_coefficients(pooled, amplitude)

    X_level = np.repeat(X, 2, axis=0)
    y_level = np.repeat(y, 2)
    entity = np.repeat(np.arange(3), 2)
    between = BetweenOLS().fit(
        torch.as_tensor(X_level, dtype=torch.float64),
        torch.as_tensor(y_level, dtype=torch.float64),
        entity_ids=torch.as_tensor(entity, dtype=torch.int64),
    )
    _assert_coefficients(between, amplitude)


def test_torch_cpu_random_effects_precision_guards_match_numpy():
    torch = pytest.importorskip("torch")
    fixtures = [
        (
            _random_effects_component_loss_fixture(),
            "quasi-demeaning exceeds the float64 component range",
        ),
        (
            _random_effects_common_scale_loss_fixture(
                amplitude=1.0e308,
                tiny_within=1.0e-100,
            ),
            "variance-component scaling exceeds the float64 common-residual range",
        ),
        (
            _random_effects_common_scale_loss_fixture(
                amplitude=1.0e100,
                tiny_within=1.0e-100,
            ),
            "variance-component scaling exceeds the float64 common-residual range",
        ),
        (
            _random_effects_theta_rounding_fixture(),
            "quasi-demeaning exceeds the float64 component range",
        ),
    ]
    for (X, y, entity), message in fixtures:
        with pytest.raises(FloatingPointError, match=message):
            RandomEffects().fit(
                torch.as_tensor(X, dtype=torch.float64),
                torch.as_tensor(y, dtype=torch.float64),
                entity_ids=torch.as_tensor(entity, dtype=torch.int64),
            )
