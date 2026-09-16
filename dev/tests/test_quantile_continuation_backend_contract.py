"""Backend-preservation checks for weighted Quantile continuation starts."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.solvers import _quantile_continuation as continuation


def _fixture(seed=16681):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(16, 3)).astype(np.float64)
    y = (0.2 + X @ np.array([0.6, -0.25, 0.1]) + rng.normal(scale=0.08, size=16)).astype(np.float64)
    w = np.linspace(0.3, 2.2, 16, dtype=np.float64)
    rng.shuffle(w)
    path = continuation.mark_auto_quantile_continuation_path(
        np.geomspace(0.25, 0.03, 3)
    )
    return X, y, w, path


def test_torch_cpu_weighted_start_matches_numpy_and_syncs_only_scalar(monkeypatch):
    torch = pytest.importorskip("torch")
    X, y, w, path = _fixture()
    loss = QuantileLoss(quantile=0.27)

    expected = continuation.resolve_auto_quantile_continuation_path(
        loss,
        X,
        y,
        path,
        sample_weight=w,
        fit_intercept=True,
    )

    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    w_t = torch.as_tensor(w, dtype=torch.float64)
    real_to_numpy = continuation._to_numpy
    synced_shapes = []

    def scalar_only_to_numpy(value):
        shape = tuple(getattr(value, "shape", ()))
        synced_shapes.append(shape)
        assert shape == ()
        return real_to_numpy(value)

    monkeypatch.setattr(continuation, "_to_numpy", scalar_only_to_numpy)
    observed = continuation.resolve_auto_quantile_continuation_path(
        loss,
        X_t,
        y_t,
        path,
        sample_weight=w_t,
        fit_intercept=True,
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)
    assert synced_shapes == [()]


def test_integer_numpy_inputs_compute_float64_weighted_score():
    X = np.array([[-2, 1], [0, 3], [2, -1], [4, 2]], dtype=np.int64)
    y = np.array([-1, 0, 2, 4], dtype=np.int64)
    w = np.array([1, 4, 2, 3], dtype=np.int64)
    path = continuation.mark_auto_quantile_continuation_path(
        np.array([0.4, 0.1, 0.02], dtype=np.float64)
    )

    observed = continuation.resolve_auto_quantile_continuation_path(
        QuantileLoss(quantile=0.3),
        X,
        y,
        path,
        sample_weight=w,
        fit_intercept=True,
    )

    assert observed.dtype == np.float64
    assert np.all(np.isfinite(observed))
    assert observed[-1] == pytest.approx(0.02, rel=0.0, abs=0.0)
