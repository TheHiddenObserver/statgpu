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


def _solve(torch, dtype, *, sample_weight=True, fit_intercept=True):
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
        fit_intercept=fit_intercept,
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


def test_torch_logistic_cv_no_intercept_path_keeps_dtype_and_zero_intercept():
    torch = pytest.importorskip("torch")

    _, coefs, intercepts = _solve(
        torch, torch.float32, fit_intercept=False
    )

    assert coefs.dtype == torch.float32
    assert intercepts.dtype == torch.float32
    assert tuple(coefs.shape) == (2, 2, 2)
    assert tuple(intercepts.shape) == (2, 2)
    assert bool(torch.isfinite(coefs).all().item())
    assert bool(torch.isfinite(intercepts).all().item())
    assert int(torch.count_nonzero(intercepts).item()) == 0


@pytest.mark.parametrize("mixed_precision", [True, False])
def test_torch_logistic_cv_selection_consumes_backend_native_path(
    monkeypatch, mixed_precision
):
    """Exercise the only path-solver caller without requiring physical CUDA."""
    torch = pytest.importorskip("torch")
    from statgpu.backends import TorchBackend
    from statgpu.linear_model.cv import _logistic_cv as logistic_cv

    X = np.asarray(
        [
            [-2.0, -1.0],
            [-1.8, 0.4],
            [-1.4, -0.3],
            [-0.9, 0.8],
            [-0.5, -0.7],
            [-0.2, 1.0],
            [0.2, -1.0],
            [0.5, 0.7],
            [0.9, -0.8],
            [1.4, 0.3],
            [1.8, -0.4],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    weights = np.asarray(
        [1.0, 1.2, 0.9, 1.1, 0.8, 1.3, 1.0, 0.9, 1.2, 1.1, 0.8, 1.4],
        dtype=np.float64,
    )
    splits = [
        (
            np.asarray([1, 3, 5, 7, 9, 11], dtype=np.int64),
            np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int64),
        ),
        (
            np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int64),
            np.asarray([1, 3, 5, 7, 9, 11], dtype=np.int64),
        ),
    ]
    Cs = np.asarray([0.25, 1.0, 4.0], dtype=np.float64)

    cpu = logistic_cv._select_logistic_c_cv(
        X,
        y,
        Cs=Cs,
        cv_folds=2,
        cv_splits=splits,
        sample_weight=weights,
        max_iter=80,
        tol=1e-7,
        device="cpu",
        return_details=True,
    )

    backend = TorchBackend(device="cpu")
    monkeypatch.setattr(
        logistic_cv,
        "resolve_cv_backend",
        lambda device, X: ("torch", "torch", backend, True, False, False),
    )
    torch_details = logistic_cv._select_logistic_c_cv(
        X,
        y,
        Cs=Cs,
        cv_folds=2,
        cv_splits=splits,
        sample_weight=weights,
        max_iter=80,
        tol=1e-7,
        device="torch",
        gpu_cv_mixed_precision=mixed_precision,
        return_details=True,
    )

    assert torch_details["C"] == cpu["C"]
    assert tuple(torch_details["loss_path"].shape) == (3, 2)
    assert np.isfinite(torch_details["loss_path"]).all()
    np.testing.assert_allclose(
        torch_details["mean_loss"],
        cpu["mean_loss"],
        rtol=5e-4 if mixed_precision else 1e-8,
        atol=5e-5 if mixed_precision else 1e-10,
    )
