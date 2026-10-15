"""Torch device-binding regressions for the Issue #163 Quantile routes."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.losses import QuantileLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers._proximal_irls_quantile import proximal_irls_quantile_solver


def _torch_data(torch, seed=16321, n=48):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = np.column_stack([x, np.ones(n)]).astype(np.float64)
    y = (0.35 * x + 0.1 + rng.laplace(scale=0.08, size=n)).astype(np.float64)
    return (
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
    )


def test_quantile_irls_torch_warm_start_is_backend_native_and_not_mutated():
    torch = pytest.importorskip("torch")
    X, y = _torch_data(torch)
    init = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    init_before = init.clone()

    coef, n_iter = QuantileLoss(quantile=0.35).irls(
        X,
        y,
        penalty=None,
        fit_intercept=True,
        init_coef=init,
        max_iter=8,
        tol=1e-9,
    )

    assert isinstance(coef, torch.Tensor)
    assert coef.device == X.device
    assert coef.dtype == X.dtype
    assert bool(torch.all(torch.isfinite(coef)).item())
    assert torch.equal(init, init_before)
    assert 1 <= n_iter <= 8


def test_proximal_quantile_torch_does_not_materialize_cpu_scalar_tensors(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    X_aug, y = _torch_data(torch, seed=16322)
    X = X_aug[:, :1]

    real_asarray = torch.asarray

    def guarded_asarray(obj, *args, **kwargs):
        if np.isscalar(obj):
            raise AssertionError(
                "Torch scalar temporaries in proximal Quantile must be created "
                "relative to an existing tensor device, not via torch.asarray()."
            )
        return real_asarray(obj, *args, **kwargs)

    monkeypatch.setattr(torch, "asarray", guarded_asarray)

    coef, intercept, n_iter = proximal_irls_quantile_solver(
        QuantileLoss(quantile=0.35),
        SCADPenalty(alpha=1.0),
        X,
        y,
        alpha_path=np.asarray([1.0], dtype=np.float64),
        max_lla_per_step=1,
        max_iter=3,
        tol=1e-8,
        fit_intercept=True,
    )

    assert coef.shape == (X.shape[1],)
    assert np.all(np.isfinite(coef))
    assert np.isfinite(intercept)
    assert 1 <= n_iter <= 3
