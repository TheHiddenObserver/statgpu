"""Regression coverage for Fama-MacBeth rank/precision failure precedence."""
from __future__ import annotations

import numpy as np
import pytest

from statgpu.panel import FamaMacBeth
from statgpu.panel._linalg import panel_lstsq_batched, panel_lstsq_deferred_rank


def _rank_deficient_cancellation_fixture(n_periods: int = 3):
    """Return rank-deficient periods whose intercept RHS loses a low-order tail."""
    amplitude = float(2.0**55)
    x_period = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    X_period = np.column_stack([x_period, x_period])
    y_period = np.asarray([amplitude, 1.0, -amplitude], dtype=np.float64)
    design = np.column_stack([np.ones(x_period.size), X_period])

    assert np.linalg.matrix_rank(design) == 2
    assert design.shape[1] == 3
    assert (design.T @ y_period)[0] == 0.0

    X = np.tile(X_period, (n_periods, 1))
    y = np.tile(y_period, n_periods)
    time = np.repeat(np.arange(n_periods, dtype=np.int64), x_period.size)
    return X, y, time, design, y_period


def test_deferred_rank_preserves_svd_rank_before_precision_sentinel_numpy():
    _X, _y, _time, design, y_period = _rank_deficient_cancellation_fixture()
    _params, rank = panel_lstsq_deferred_rank(design, y_period, np)
    assert int(rank) == 2


def test_fama_macbeth_rank_deficiency_precedes_precision_failure_numpy():
    X, y, time, _design, _y_period = _rank_deficient_cancellation_fixture()
    with pytest.raises(
        ValueError,
        match=r"full column rank.*rank deficient.*rank=2, columns=3",
    ):
        FamaMacBeth(device="cpu", bandwidth=0).fit(X, y, time_ids=time)


def test_fama_macbeth_rank_deficiency_precedes_precision_failure_torch_cpu():
    torch = pytest.importorskip("torch")
    X, y, time, design, y_period = _rank_deficient_cancellation_fixture()

    design_batch = torch.as_tensor(
        np.stack([design, design], axis=0), dtype=torch.float64
    )
    y_batch = torch.as_tensor(
        np.stack([y_period, y_period], axis=0), dtype=torch.float64
    )
    _params, ranks = panel_lstsq_batched(design_batch, y_batch, torch)
    assert ranks.tolist() == [2, 2]

    with pytest.raises(
        ValueError,
        match=r"full column rank.*rank deficient.*rank=2, columns=3",
    ):
        FamaMacBeth(bandwidth=0).fit(
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(y, dtype=torch.float64),
            time_ids=torch.as_tensor(time, dtype=torch.int64),
        )
