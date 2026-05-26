"""
Tests for the statgpu.backends module.

These tests run on CPU-only environments (no GPU required) and validate:
* NumpyBackend correctness
* CuPyBackend availability detection
* TorchBackend availability detection
* get_backend() factory behaviour
* BaseEstimator._get_backend() integration
"""

import numpy as np
import pytest

from statgpu.backends import (
    BackendBase,
    NumpyBackend,
    CuPyBackend,
    TorchBackend,
    _resolve_backend,
    get_backend,
)
from statgpu._base import BaseEstimator
from statgpu._config import Device


# ---------------------------------------------------------------------------
# NumpyBackend tests – always run (no GPU required)
# ---------------------------------------------------------------------------

class TestNumpyBackend:
    def setup_method(self):
        self.backend = NumpyBackend()

    def test_name(self):
        assert self.backend.name == "numpy"

    def test_is_available(self):
        assert self.backend.is_available() is True

    def test_xp_is_numpy(self):
        assert self.backend.xp is np

    def test_asarray_from_list(self):
        arr = self.backend.asarray([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_asarray_with_dtype(self):
        arr = self.backend.asarray([1.0, 2.0], dtype=np.float32)
        assert arr.dtype == np.float32

    def test_to_numpy_roundtrip(self):
        arr = np.array([4.0, 5.0])
        result = self.backend.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_solve(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([4.0, 9.0])
        x = self.backend.solve(A, b)
        np.testing.assert_allclose(x, [2.0, 3.0], rtol=1e-6)

    def test_lstsq(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([1.0, 2.0, 3.0])
        x, *_ = self.backend.lstsq(A, b, rcond=None)
        # Least-squares solution should be close to [1, 2]
        np.testing.assert_allclose(x, [1.0, 2.0], atol=1e-6)

    def test_repr(self):
        r = repr(self.backend)
        assert "NumpyBackend" in r
        assert "available" in r


def test_resolve_backend_accepts_legacy_cpu_backend_alias():
    assert _resolve_backend("cpu") == "numpy"


# ---------------------------------------------------------------------------
# CuPyBackend tests – skip gracefully when CuPy / GPU not available
# ---------------------------------------------------------------------------

_cupy_available = CuPyBackend().is_available()


class TestCuPyBackend:
    def setup_method(self):
        self.backend = CuPyBackend()

    def test_name(self):
        assert self.backend.name == "cupy"

    def test_is_available_returns_bool(self):
        result = self.backend.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not _cupy_available, reason="CuPy / CUDA not available")
    def test_asarray_roundtrip(self):
        import cupy as cp
        arr = self.backend.asarray([1.0, 2.0, 3.0])
        assert isinstance(arr, cp.ndarray)
        np.testing.assert_array_equal(self.backend.to_numpy(arr), [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not _cupy_available, reason="CuPy / CUDA not available")
    def test_to_numpy_from_numpy_input(self):
        arr_np = np.array([7.0, 8.0])
        result = self.backend.to_numpy(arr_np)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr_np)

    @pytest.mark.skipif(not _cupy_available, reason="CuPy / CUDA not available")
    def test_cummin_empty_returns_empty(self):
        import cupy as cp
        arr = cp.array([], dtype=cp.float64)
        out = self.backend.cummin(arr)
        assert isinstance(out, cp.ndarray)
        assert out.shape == (0,)

    @pytest.mark.skipif(not _cupy_available, reason="CuPy / CUDA not available")
    def test_cummax_empty_last_axis_returns_empty(self):
        import cupy as cp
        arr = cp.empty((3, 0), dtype=cp.float32)
        out = self.backend.cummax(arr, axis=1)
        assert isinstance(out, cp.ndarray)
        assert out.shape == (3, 0)


# ---------------------------------------------------------------------------
# TorchBackend tests – skip gracefully when PyTorch not installed
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    _torch_importable = True
except ImportError:
    _torch_importable = False

_torch_cuda_available = TorchBackend().is_available()


class TestTorchBackend:
    def setup_method(self):
        self.backend = TorchBackend(device="cpu")

    def test_name(self):
        assert self.backend.name == "torch"

    def test_is_available_returns_bool(self):
        result = TorchBackend(device="cpu").is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not _torch_importable, reason="PyTorch not installed")
    def test_asarray_cpu_roundtrip(self):
        import torch
        backend = TorchBackend(device="cpu")
        arr = backend.asarray([1.0, 2.0, 3.0])
        assert isinstance(arr, torch.Tensor)
        np.testing.assert_array_equal(backend.to_numpy(arr), [1.0, 2.0, 3.0])

    @pytest.mark.skipif(not _torch_importable, reason="PyTorch not installed")
    def test_solve_cpu(self):
        backend = TorchBackend(device="cpu")
        A = backend.asarray([[2.0, 0.0], [0.0, 3.0]])
        b = backend.asarray([4.0, 9.0])
        x = backend.solve(A, b)
        np.testing.assert_allclose(backend.to_numpy(x), [2.0, 3.0], rtol=1e-5)

    @pytest.mark.skipif(not _torch_importable, reason="PyTorch not installed")
    def test_to_numpy_from_numpy_input(self):
        backend = TorchBackend(device="cpu")
        arr_np = np.array([5.0, 6.0])
        result = backend.to_numpy(arr_np)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr_np)


# ---------------------------------------------------------------------------
# get_backend() factory tests
# ---------------------------------------------------------------------------

class TestGetBackend:
    def test_numpy_explicit(self):
        b = get_backend(backend="numpy")
        assert isinstance(b, NumpyBackend)

    def test_cupy_explicit(self):
        b = get_backend(backend="cupy")
        assert isinstance(b, CuPyBackend)

    def test_torch_explicit(self):
        b = get_backend(backend="torch")
        assert isinstance(b, TorchBackend)

    def test_auto_cpu_returns_numpy(self):
        b = get_backend(backend="auto", device="cpu")
        assert isinstance(b, NumpyBackend)

    def test_auto_returns_backend_base(self):
        b = get_backend(backend="auto")
        assert isinstance(b, BackendBase)

    def test_auto_backend_is_available(self):
        b = get_backend(backend="auto")
        assert b.is_available()


# ---------------------------------------------------------------------------
# BaseEstimator._get_backend() integration
# ---------------------------------------------------------------------------

class _DummyEstimator(BaseEstimator):
    """Minimal concrete estimator for testing."""

    def fit(self, X, y=None, **fit_params):
        self._fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        return X


class TestBaseEstimatorBackend:
    def test_cpu_device_returns_numpy_backend(self):
        est = _DummyEstimator(device="cpu")
        backend = est._get_backend()
        assert isinstance(backend, NumpyBackend)

    def test_explicit_numpy_backend(self):
        est = _DummyEstimator(device="cpu")
        backend = est._get_backend(backend="numpy")
        assert isinstance(backend, NumpyBackend)

    def test_explicit_cupy_backend_instance(self):
        est = _DummyEstimator(device="cuda")
        backend = est._get_backend(backend="cupy")
        assert isinstance(backend, CuPyBackend)

    def test_explicit_torch_backend_instance(self):
        est = _DummyEstimator(device="cuda")
        backend = est._get_backend(backend="torch")
        assert isinstance(backend, TorchBackend)

    def test_backend_xp_usable(self):
        """xp should expose an array-creation function."""
        est = _DummyEstimator(device="cpu")
        backend = est._get_backend()
        xp = backend.xp
        arr = xp.zeros(5)
        assert arr.shape == (5,)
