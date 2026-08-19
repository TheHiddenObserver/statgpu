"""Regression coverage for shared panel least-squares resolution limits."""
from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import FamaMacBeth, FirstDifferenceOLS, PanelOLS, PooledOLS
from statgpu.panel._linalg import panel_lstsq


def _cancellation_mean_fixture():
    amplitude = float(2.0**55)
    X = np.ones((3, 1), dtype=np.float64)
    y = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    return X, y


def _mixed_coefficient_resolution_fixture():
    """Well-conditioned design with one coefficient below SVD coordinate resolution."""
    amplitude = float(2.0**55)
    X = np.asarray(
        [
            [2.0, 2.0],
            [-1.0, 2.0],
            [2.0, -2.0],
            [-2.0, 1.0],
        ],
        dtype=np.float64,
    )
    beta = np.asarray([amplitude, 16.0], dtype=np.float64)
    y = X @ beta
    assert np.linalg.cond(X) < 2.0
    return X, y


def _fama_macbeth_resolution_fixture(n_periods=3):
    X_period, _ = _mixed_coefficient_resolution_fixture()
    beta = np.asarray([float(2.0**55), 16.0], dtype=np.float64)
    y_period = X_period @ beta
    X = np.tile(X_period, (n_periods, 1))
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), X_period.shape[0])
    design = np.column_stack([np.ones(X_period.shape[0]), X_period])
    assert np.linalg.cond(design) < 3.0
    return X, y, time


def _first_difference_from_transformed(X_diff, y_diff):
    n = int(X_diff.shape[0])
    X = np.zeros((2 * n, int(X_diff.shape[1])), dtype=np.float64)
    y = np.zeros(2 * n, dtype=np.float64)
    X[1::2] = X_diff
    y[1::2] = y_diff
    entity = np.repeat(np.arange(n, dtype=np.int64), 2)
    time = np.tile(np.arange(2, dtype=np.int64), n)
    return X, y, entity, time


def _first_difference_extreme_r2_fixture():
    y_diff = np.concatenate(
        [np.asarray([1.0e308]), np.full(10, -1.0e308, dtype=np.float64)]
    )
    X_diff = (y_diff / 1.0e308).reshape(-1, 1)
    return _first_difference_from_transformed(X_diff, y_diff)


def test_shared_panel_lstsq_preserves_constant_only_cancellation_tail_numpy():
    X, y = _cancellation_mean_fixture()
    params, rank = panel_lstsq(X, y, np)
    assert rank == 1
    np.testing.assert_allclose(
        params[0], 1.0 / 3.0, rtol=4.0 * np.finfo(np.float64).eps, atol=0.0
    )


def test_panelols_preserves_constant_only_cancellation_tail_numpy():
    X, y = _cancellation_mean_fixture()
    model = PanelOLS(cov_type="hc0").fit(X, y)
    np.testing.assert_allclose(
        np.asarray(model.coef_)[0],
        1.0 / 3.0,
        rtol=4.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )


def test_shared_panel_lstsq_fails_closed_on_unresolved_mixed_coefficients_numpy():
    X, y = _mixed_coefficient_resolution_fixture()
    with pytest.raises(
        FloatingPointError,
        match="coefficient resolution exceeds float64 precision",
    ):
        panel_lstsq(X, y, np)


def test_panelols_fails_closed_on_unresolved_mixed_coefficients_numpy():
    X, y = _mixed_coefficient_resolution_fixture()
    with pytest.raises(
        FloatingPointError,
        match="coefficient resolution exceeds float64 precision",
    ):
        PanelOLS(cov_type="hc0").fit(X, y)


def test_pooledols_fails_closed_on_unresolved_nonconstant_slope_numpy():
    X, y = _mixed_coefficient_resolution_fixture()
    with pytest.raises(
        FloatingPointError,
        match="coefficient resolution exceeds float64 precision",
    ):
        PooledOLS(cov_type="hc0").fit(X, y)


def test_first_difference_fails_closed_on_unresolved_mixed_coefficients_numpy():
    X_diff, y_diff = _mixed_coefficient_resolution_fixture()
    X, y, entity, time = _first_difference_from_transformed(X_diff, y_diff)
    with pytest.raises(
        FloatingPointError,
        match="coefficient resolution exceeds float64 precision",
    ):
        FirstDifferenceOLS(cov_type="hc0").fit(
            X, y, entity_ids=entity, time_ids=time
        )


def test_fama_macbeth_reports_resolution_failure_instead_of_rank_deficiency_numpy():
    X, y, time = _fama_macbeth_resolution_fixture()
    with pytest.raises(
        FloatingPointError,
        match="FamaMacBeth period coefficient resolution exceeds float64 precision",
    ):
        FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)


def test_first_difference_legacy_r2_handles_unrepresentable_physical_centering_numpy():
    X, y, entity, time = _first_difference_extreme_r2_fixture()
    model = FirstDifferenceOLS(cov_type="hc0").fit(
        X, y, entity_ids=entity, time_ids=time
    )
    assert model.rsquared == 1.0
    np.testing.assert_allclose(
        np.asarray(model.coef_)[0], 1.0e308, rtol=2.0e-15, atol=0.0
    )


def test_torch_cpu_shared_panel_resolution_contract_matches_numpy():
    torch = pytest.importorskip("torch")

    X_mean, y_mean = _cancellation_mean_fixture()
    params, rank = panel_lstsq(
        torch.as_tensor(X_mean, dtype=torch.float64),
        torch.as_tensor(y_mean, dtype=torch.float64),
        torch,
    )
    assert rank == 1
    np.testing.assert_allclose(
        float(params[0].item()),
        1.0 / 3.0,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )

    X_bad, y_bad = _mixed_coefficient_resolution_fixture()
    with pytest.raises(
        FloatingPointError,
        match="coefficient resolution exceeds float64 precision",
    ):
        panel_lstsq(
            torch.as_tensor(X_bad, dtype=torch.float64),
            torch.as_tensor(y_bad, dtype=torch.float64),
            torch,
        )

    X_fmb, y_fmb, time_fmb = _fama_macbeth_resolution_fixture()
    with pytest.raises(
        FloatingPointError,
        match="FamaMacBeth period coefficient resolution exceeds float64 precision",
    ):
        FamaMacBeth(bandwidth=0).fit(
            torch.as_tensor(X_fmb, dtype=torch.float64),
            torch.as_tensor(y_fmb, dtype=torch.float64),
            time_ids=torch.as_tensor(time_fmb, dtype=torch.int64),
        )

    X_fd, y_fd, entity, time = _first_difference_extreme_r2_fixture()
    model = FirstDifferenceOLS(cov_type="hc0").fit(
        torch.as_tensor(X_fd, dtype=torch.float64),
        torch.as_tensor(y_fd, dtype=torch.float64),
        entity_ids=torch.as_tensor(entity, dtype=torch.int64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    assert model.rsquared == 1.0
    np.testing.assert_allclose(
        float(model.coef_[0].item()), 1.0e308, rtol=3.0e-15, atol=0.0
    )
