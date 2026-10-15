"""PR151 closure for scale-invariant Newton/L-BFGS weight validation."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core._logistic import LogisticLoss
from statgpu.solvers import lbfgs_solver, newton_solver
from statgpu.solvers._lbfgs import _prepare_lbfgs_sample_weight
from statgpu.solvers._newton import _prepare_newton_sample_weight


def _data(seed=151931, n=96, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.55, size=(n, p)).astype(np.float64)
    beta = np.array([0.38, -0.24, 0.13], dtype=np.float64)[:p]
    prob = 1.0 / (1.0 + np.exp(-(X @ beta)))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    return X, y


def _float32_overflow_weights(n):
    shape = np.linspace(0.25, 1.0, n, dtype=np.float32)
    huge = shape * np.float32(3.0e38)
    assert np.all(np.isfinite(huge))
    with np.errstate(over="ignore"):
        assert np.isinf(np.sum(huge, dtype=np.float32))
    return shape, huge


@pytest.mark.parametrize("solver_fn", [newton_solver, lbfgs_solver])
def test_float32_raw_sum_overflow_does_not_break_global_weight_rescaling(solver_fn):
    X, y = _data()
    shape, huge = _float32_overflow_weights(X.shape[0])
    loss = LogisticLoss()

    base, _ = solver_fn(
        loss, None, X, y, max_iter=700, tol=1.0e-10, sample_weight=shape
    )
    scaled, _ = solver_fn(
        loss, None, X, y, max_iter=700, tol=1.0e-10, sample_weight=huge
    )

    np.testing.assert_allclose(
        np.asarray(scaled), np.asarray(base), rtol=2.0e-7, atol=2.0e-8
    )


def test_float32_raw_sum_overflow_preparation_normalizes_before_execution():
    X = np.ones((4, 2), dtype=np.float32)
    shape, huge = _float32_overflow_weights(4)

    newton_weight = _prepare_newton_sample_weight(huge, 4, "numpy", X)
    lbfgs_weight = _prepare_lbfgs_sample_weight(
        huge, 4, "numpy", X, LogisticLoss()
    )

    for values in (newton_weight, lbfgs_weight):
        assert values is not None
        values = np.asarray(values)
        assert values.dtype == np.float32
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values, shape, rtol=2.0e-7, atol=0.0)


def test_torch_float32_raw_sum_overflow_preparation_when_available():
    torch = pytest.importorskip("torch")
    X = torch.ones((4, 2), dtype=torch.float32)
    shape_np, huge_np = _float32_overflow_weights(4)
    huge = torch.as_tensor(huge_np, dtype=torch.float32)
    assert bool(torch.isinf(torch.sum(huge)).item())

    newton_weight = _prepare_newton_sample_weight(huge, 4, "torch", X)
    lbfgs_weight = _prepare_lbfgs_sample_weight(
        huge, 4, "torch", X, LogisticLoss()
    )

    expected = torch.as_tensor(shape_np, dtype=torch.float32)
    for values in (newton_weight, lbfgs_weight):
        assert values is not None
        assert values.dtype == torch.float32
        assert torch.all(torch.isfinite(values))
        torch.testing.assert_close(values, expected, rtol=2.0e-7, atol=0.0)
