"""Regression coverage for cancellation-safe automatic panel intercepts."""
from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import BetweenOLS, PooledOLS
from statgpu.panel._intercept import panel_lstsq_exact_constant
from statgpu.panel._linalg import panel_lstsq


def _fixture():
    amplitude = float(2.0**55)
    X = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    y = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    return X, y, amplitude


def _assert_coefficients(model, amplitude):
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    np.testing.assert_allclose(coef[0], 1.0 / 3.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(coef[1], -amplitude, rtol=2.0e-15, atol=0.0)


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
