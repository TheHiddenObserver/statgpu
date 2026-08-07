"""Regression coverage for Issue #112 Torch LogisticRegressionCV dtype handling."""

from __future__ import annotations

import numpy as np
import pytest


def _torch_path_problem(torch, dtype):
    X = torch.tensor(
        [
            [
                [-2.0, -1.0],
                [-1.5, 0.5],
                [-1.0, -0.2],
                [-0.5, 1.0],
                [0.5, -1.0],
                [1.0, 0.2],
                [1.5, -0.5],
                [2.0, 1.0],
            ],
            [
                [-2.2, 0.4],
                [-1.4, -0.8],
                [-0.9, 0.7],
                [-0.3, -1.2],
                [0.4, 1.1],
                [0.9, -0.4],
                [1.6, 0.6],
                [2.1, -0.9],
            ],
        ],
        dtype=dtype,
    )
    y = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=dtype,
    )
    sample_weight = torch.tensor(
        [
            [1.0, 1.5, 0.8, 1.2, 1.1, 0.9, 1.4, 1.0],
            [1.2, 0.7, 1.3, 1.0, 0.9, 1.4, 0.8, 1.1],
        ],
        dtype=dtype,
    )
    return X, y, sample_weight


def _solve(torch, dtype, *, sample_weight=True):
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._logistic_cv import (
        _solve_logistic_path_gpu_from_batch,
    )

    backend = TorchBackend(device="cpu")
    X, y, weights = _torch_path_problem(torch, dtype)
    coefs, intercepts = _solve_logistic_path_gpu_from_batch(
        X,
        y,
        np.asarray([8, 8], dtype=np.int32),
        np.asarray([0.5, 2.0], dtype=np.float64),
        backend,
        fit_intercept=True,
        max_iter=60,
        tol=1e-7,
        sw_batch=weights if sample_weight else None,
    )
    return backend, coefs, intercepts


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_logistic_cv_path_preserves_working_dtype(dtype_name):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)

    _, coefs, intercepts = _solve(torch, dtype)

    assert isinstance(coefs, torch.Tensor)
    assert isinstance(intercepts, torch.Tensor)
    assert coefs.device.type == "cpu"
    assert intercepts.device.type == "cpu"
    assert coefs.dtype == dtype
    assert intercepts.dtype == dtype
    assert tuple(coefs.shape) == (2, 2, 2)
    assert tuple(intercepts.shape) == (2, 2)
    assert bool(torch.isfinite(coefs).all().item())
    assert bool(torch.isfinite(intercepts).all().item())


def test_torch_logistic_cv_path_does_not_roundtrip_parameters_to_numpy(monkeypatch):
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv._logistic_cv import (
        _solve_logistic_path_gpu_from_batch,
    )

    backend = TorchBackend(device="cpu")
    X, y, weights = _torch_path_problem(torch, torch.float32)

    def forbid_to_numpy(*args, **kwargs):
        raise AssertionError("path solver must keep parameter batches backend-native")

    monkeypatch.setattr(backend, "to_numpy", forbid_to_numpy)
    coefs, intercepts = _solve_logistic_path_gpu_from_batch(
        X,
        y,
        np.asarray([8, 8], dtype=np.int32),
        np.asarray([0.5, 2.0], dtype=np.float64),
        backend,
        fit_intercept=True,
        max_iter=60,
        tol=1e-7,
        sw_batch=weights,
    )

    assert coefs.dtype == torch.float32
    assert intercepts.dtype == torch.float32


def test_torch_logistic_cv_float32_and_float64_paths_agree():
    torch = pytest.importorskip("torch")

    _, coef32, intercept32 = _solve(torch, torch.float32)
    _, coef64, intercept64 = _solve(torch, torch.float64)

    torch.testing.assert_close(
        coef32.to(torch.float64), coef64, rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(
        intercept32.to(torch.float64), intercept64, rtol=2e-4, atol=2e-5
    )


def test_torch_logistic_cv_unweighted_path_keeps_dtype():
    torch = pytest.importorskip("torch")

    _, coefs, intercepts = _solve(
        torch, torch.float32, sample_weight=False
    )

    assert coefs.dtype == torch.float32
    assert intercepts.dtype == torch.float32
    assert bool(torch.isfinite(coefs).all().item())
    assert bool(torch.isfinite(intercepts).all().item())
