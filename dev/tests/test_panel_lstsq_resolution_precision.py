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


def _fama_macbeth_intercept_cancellation_fixture(n_periods=3):
    amplitude = float(2.0**55)
    x_period = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    y_period = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    X = np.tile(x_period, n_periods)[:, None]
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    design = np.column_stack([np.ones(x_period.size), x_period])
    assert np.linalg.cond(design) < 2.0
    return X, y, time, amplitude


def _fama_macbeth_nonconstant_zero_rhs_fixture(n_periods=3):
    """Return a condition-one period design whose slope RHS loses a nonzero tail."""
    amplitude = float(2.0**55)
    x_period = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    y_period = np.asarray([-3.0, amplitude, 2.0, -amplitude], dtype=np.float64)
    X = np.tile(x_period, n_periods)[:, None]
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    design = np.column_stack([np.ones(x_period.size), x_period])
    assert np.linalg.cond(design) == 1.0
    # In exact arithmetic over the supplied float64 values,
    # sum(y) = sum(x*y) = -1, so both coefficients equal -1/4. A BLAS
    # reduction can nevertheless return zero for the nonconstant RHS.
    return X, y, time


def _fama_macbeth_genuine_zero_rhs_fixture(n_periods=3):
    """Return a condition-one period whose nonconstant RHS is genuinely zero."""
    x_period = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float64)
    y_period = np.asarray([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    X = np.tile(x_period, n_periods)[:, None]
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    design = np.column_stack([np.ones(x_period.size), x_period])
    assert np.linalg.cond(design) == 1.0
    assert np.array_equal(design.T @ y_period, np.zeros(2, dtype=np.float64))
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


def test_fama_macbeth_preserves_period_intercept_cancellation_tail_numpy():
    X, y, time, amplitude = _fama_macbeth_intercept_cancellation_fixture()
    model = FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)
    betas = np.asarray(model.betas_, dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64)
    np.testing.assert_allclose(
        betas[:, 0],
        np.full(model.n_periods, 1.0 / 3.0),
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    np.testing.assert_allclose(
        coef[0], 1.0 / 3.0, rtol=8.0 * np.finfo(np.float64).eps, atol=0.0
    )
    np.testing.assert_allclose(betas[:, 1], -amplitude, rtol=2.0e-15, atol=0.0)
    assert model._period_svd_fallbacks == 0
    assert model._period_solver_mode == "gram-certified"


def test_fama_macbeth_fails_closed_when_stable_rhs_recovers_lost_nonconstant_tail_numpy():
    X, y, time = _fama_macbeth_nonconstant_zero_rhs_fixture()
    with pytest.raises(
        FloatingPointError,
        match="FamaMacBeth period coefficient resolution exceeds float64 precision",
    ):
        FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)


def test_fama_macbeth_allows_genuine_zero_nonconstant_rhs_numpy():
    X, y, time = _fama_macbeth_genuine_zero_rhs_fixture()
    model = FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)
    betas = np.asarray(model.betas_, dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64)
    np.testing.assert_allclose(betas, np.zeros_like(betas), rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(coef, np.zeros_like(coef), rtol=0.0, atol=2.0e-15)
    assert model._period_svd_fallbacks == model.n_periods


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

    X_tail, y_tail, time_tail, amplitude = _fama_macbeth_intercept_cancellation_fixture()
    fmb_tail = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X_tail, dtype=torch.float64),
        torch.as_tensor(y_tail, dtype=torch.float64),
        time_ids=torch.as_tensor(time_tail, dtype=torch.int64),
    )
    betas_tail = np.asarray(fmb_tail.betas_.detach().cpu(), dtype=np.float64)
    coef_tail = np.asarray(fmb_tail.coef_.detach().cpu(), dtype=np.float64)
    np.testing.assert_allclose(
        betas_tail[:, 0],
        np.full(fmb_tail.n_periods, 1.0 / 3.0),
        rtol=1.2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(coef_tail[0], 1.0 / 3.0, rtol=1.2e-14, atol=0.0)
    np.testing.assert_allclose(betas_tail[:, 1], -amplitude, rtol=4.0e-15, atol=0.0)
    assert fmb_tail._period_svd_fallbacks == 0
    assert fmb_tail._period_solver_mode == "gram-certified"

    X_zero, y_zero, time_zero = _fama_macbeth_nonconstant_zero_rhs_fixture()
    with pytest.raises(
        FloatingPointError,
        match="FamaMacBeth period coefficient resolution exceeds float64 precision",
    ):
        FamaMacBeth(bandwidth=0).fit(
            torch.as_tensor(X_zero, dtype=torch.float64),
            torch.as_tensor(y_zero, dtype=torch.float64),
            time_ids=torch.as_tensor(time_zero, dtype=torch.int64),
        )

    X_exact, y_exact, time_exact = _fama_macbeth_genuine_zero_rhs_fixture()
    fmb_exact = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X_exact, dtype=torch.float64),
        torch.as_tensor(y_exact, dtype=torch.float64),
        time_ids=torch.as_tensor(time_exact, dtype=torch.int64),
    )
    np.testing.assert_allclose(
        fmb_exact.betas_.detach().cpu().numpy(),
        np.zeros_like(fmb_exact.betas_.detach().cpu().numpy()),
        rtol=0.0,
        atol=4.0e-15,
    )
    np.testing.assert_allclose(
        fmb_exact.coef_.detach().cpu().numpy(),
        np.zeros_like(fmb_exact.coef_.detach().cpu().numpy()),
        rtol=0.0,
        atol=4.0e-15,
    )
    assert fmb_exact._period_svd_fallbacks == fmb_exact.n_periods

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
