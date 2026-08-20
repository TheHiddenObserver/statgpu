"""Regression for full-rank Fama-MacBeth fallbacks with an exact stable zero RHS."""
from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import FamaMacBeth


def _stable_zero_rhs_fixture(n_periods: int = 3):
    amplitude = float(2.0**55)
    x_period = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    y_period = np.asarray([amplitude, -amplitude, -amplitude, amplitude], dtype=np.float64)
    design = np.column_stack([np.ones(x_period.size), x_period])
    assert np.linalg.cond(design) == 1.0
    np.testing.assert_array_equal(
        design.T @ y_period,
        np.zeros(2, dtype=np.float64),
    )
    # A rounded SVD basis can produce a spurious finite slope (about -4 on
    # NumPy) even though the full-rank normal equations have the unique solution
    # beta=(0, 0). The maintained stable RHS must therefore own this boundary.
    X = np.tile(x_period, n_periods)[:, None]
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    return X, y, time


def _assert_exact_zero_result(model):
    betas = np.asarray(model.betas_ if isinstance(model.betas_, np.ndarray) else model.betas_.detach().cpu())
    coef = np.asarray(model.coef_ if isinstance(model.coef_, np.ndarray) else model.coef_.detach().cpu())
    np.testing.assert_array_equal(betas, np.zeros_like(betas))
    np.testing.assert_array_equal(coef, np.zeros_like(coef))
    assert model._period_svd_fallbacks == model.n_periods
    assert model._period_solver_mode == "gram-certified+svd-fallback"


def test_fama_macbeth_canonicalizes_full_rank_stable_zero_rhs_numpy():
    X, y, time = _stable_zero_rhs_fixture()
    model = FamaMacBeth(bandwidth=0, device="cpu").fit(X, y, time_ids=time)
    _assert_exact_zero_result(model)


def test_fama_macbeth_canonicalizes_full_rank_stable_zero_rhs_torch_cpu():
    torch = pytest.importorskip("torch")
    X, y, time = _stable_zero_rhs_fixture()
    model = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    assert model._backend_name == "torch"
    _assert_exact_zero_result(model)
