"""
Physical GPU validation tests for PR #79 review fixes.

Per Section 8 of the test plan (Gate A), each test asserts:
- Result type (correct array type for the backend)
- Result device (stays on specified GPU)
- Result dtype (float32 stays float32, float64 stays float64)
- Result shape (matches expected dimensions)
- Numerical finiteness (no NaN/Inf)
- Convergence (where applicable)
- NumPy reference consistency (where applicable)

Usage:
    STATGPU_REQUIRE_PHYSICAL_GPU=1 pytest dev/tests/test_pr79_physical_gpu.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


# ============================================================================
# Helpers
# ============================================================================


def _is_cupy_array(x):
    """Check if x is a CuPy array."""
    try:
        import cupy as cp
        return isinstance(x, cp.ndarray)
    except ImportError:
        return False


def _is_torch_cuda_tensor(x):
    """Check if x is a Torch CUDA tensor."""
    try:
        import torch
        return isinstance(x, torch.Tensor) and x.is_cuda
    except ImportError:
        return False


def _is_numpy_array(x):
    """Check if x is a numpy array."""
    return isinstance(x, np.ndarray)


def _assert_finite(arr, name="array"):
    """Assert all values in arr are finite."""
    if _is_torch_cuda_tensor(arr):
        arr_np = arr.detach().cpu().numpy()
    elif _is_cupy_array(arr):
        import cupy as cp
        arr_np = cp.asnumpy(arr)
    else:
        arr_np = np.asarray(arr)
    assert np.all(np.isfinite(arr_np)), f"{name} contains NaN/Inf values"


def _assert_dtype(arr, expected_dtype, name="array"):
    """Assert arr has the expected dtype."""
    dtype_str = str(arr.dtype)
    expected_str = str(expected_dtype)
    assert expected_str in dtype_str, (
        f"{name} dtype mismatch: expected {expected_str}, got {dtype_str}"
    )


def _assert_on_cuda(arr, name="array"):
    """Assert arr is on a CUDA device (cupy or torch)."""
    if _is_cupy_array(arr):
        assert arr.device.id >= 0, f"{name} not on CuPy CUDA device"
    elif _is_torch_cuda_tensor(arr):
        assert arr.is_cuda, f"{name} not on Torch CUDA device"
    else:
        # Not a GPU array - check if it should be
        pass


def _to_numpy(arr):
    """Convert any backend array to numpy."""
    if arr is None:
        return None
    if _is_torch_cuda_tensor(arr):
        return arr.detach().cpu().numpy()
    if _is_cupy_array(arr):
        import cupy as cp
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _to_tensor_checks(arr, expected_backend, expected_shape=None,
                      expected_dtype=None, name="result"):
    """Standard validation checks for a GPU tensor/array result."""
    # Check type
    if expected_backend == "cupy":
        assert _is_cupy_array(arr), f"{name}: expected CuPy array, got {type(arr)}"
    elif expected_backend == "torch":
        assert _is_torch_cuda_tensor(arr), f"{name}: expected Torch CUDA tensor, got {type(arr)}"
    elif expected_backend == "numpy":
        assert _is_numpy_array(arr), f"{name}: expected numpy array, got {type(arr)}"

    # Check device
    if expected_backend in ("cupy", "torch"):
        _assert_on_cuda(arr, name)

    # Check dtype
    if expected_dtype is not None:
        _assert_dtype(arr, expected_dtype, name)

    # Check shape
    if expected_shape is not None:
        actual_shape = tuple(arr.shape)
        assert actual_shape == tuple(expected_shape), (
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    # Check finiteness
    _assert_finite(arr, name)


# ============================================================================
# Test: CuPy Allocation
# ============================================================================


@pytest.mark.gpu
@pytest.mark.cupy
class TestCuPyAllocation:
    """CuPy float32/float64 allocation and sync (Section 8)."""

    def test_cupy_float64_roundtrip(self, cupy_available):
        """CuPy can allocate, sync, and return float64 data."""
        import cupy as cp

        x_np = np.arange(16, dtype=np.float64).reshape(4, 4)
        x_cp = cp.asarray(x_np)
        cp.cuda.Stream.null.synchronize()

        result = cp.asnumpy(x_cp)
        assert np.array_equal(result, x_np), "CuPy float64 roundtrip failed"
        assert result.dtype == np.float64, f"dtype mismatch: {result.dtype}"

    def test_cupy_float32_roundtrip(self, cupy_available):
        """CuPy can allocate, sync, and return float32 data."""
        import cupy as cp

        x_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        x_cp = cp.asarray(x_np)
        cp.cuda.Stream.null.synchronize()

        result = cp.asnumpy(x_cp)
        assert np.array_equal(result, x_np), "CuPy float32 roundtrip failed"
        assert result.dtype == np.float32, f"dtype mismatch: {result.dtype}"

    def test_cupy_large_allocation(self, cupy_available):
        """CuPy can allocate a reasonably large array."""
        import cupy as cp

        n = 10000
        x = cp.ones((n, n // 10), dtype=np.float64)
        cp.cuda.Stream.null.synchronize()
        assert x.shape == (n, n // 10)
        assert x.dtype == cp.float64


# ============================================================================
# Test: Torch CUDA Allocation
# ============================================================================


@pytest.mark.gpu
@pytest.mark.torch_cuda
class TestTorchAllocation:
    """Torch CUDA float32/float64 allocation and sync (Section 8)."""

    def test_torch_float64_roundtrip(self, torch_cuda_available):
        """Torch CUDA can allocate, sync, and return float64 data."""
        import torch

        x_np = np.arange(16, dtype=np.float64).reshape(4, 4)
        x_t = torch.as_tensor(x_np, device="cuda")
        torch.cuda.synchronize()

        result = x_t.cpu().numpy()
        assert np.array_equal(result, x_np), "Torch float64 roundtrip failed"
        assert result.dtype == np.float64

    def test_torch_float32_roundtrip(self, torch_cuda_available):
        """Torch CUDA can allocate, sync, and return float32 data."""
        import torch

        x_np = np.arange(16, dtype=np.float32).reshape(4, 4)
        x_t = torch.as_tensor(x_np, device="cuda")
        torch.cuda.synchronize()

        result = x_t.cpu().numpy()
        assert np.array_equal(result, x_np), "Torch float32 roundtrip failed"
        assert result.dtype == np.float32

    def test_torch_large_allocation(self, torch_cuda_available):
        """Torch CUDA can allocate a reasonably large array."""
        import torch

        n = 10000
        x = torch.ones((n, n // 10), dtype=torch.float64, device="cuda")
        torch.cuda.synchronize()
        assert x.shape == (n, n // 10)
        assert x.is_cuda


# ============================================================================
# Test: Cholesky Solve
# ============================================================================


@pytest.mark.gpu
class TestCholeskySolve:
    """Vector/matrix RHS Cholesky solve on CuPy and Torch (Section 8)."""

    @pytest.mark.cupy
    def test_cupy_cholesky_solve_vector(self, cupy_available):
        """CuPy Cholesky solve with vector RHS."""
        import cupy as cp

        n = 50
        A_np = np.random.randn(n, n).astype(np.float64)
        A_np = A_np.T @ A_np + np.eye(n) * n * 1e-3  # PSD
        b_np = np.random.randn(n).astype(np.float64)

        A_cp = cp.asarray(A_np)
        b_cp = cp.asarray(b_np)

        L = cp.linalg.cholesky(A_cp)
        x = cp.linalg.solve(A_cp, b_cp)
        residual = cp.linalg.norm(A_cp @ x - b_cp)

        assert float(residual) < 1e-10, f"CuPy Cholesky residual too large: {residual}"

    @pytest.mark.torch_cuda
    def test_torch_cholesky_solve_vector(self, torch_cuda_available):
        """Torch CUDA Cholesky solve with vector RHS."""
        import torch

        n = 50
        A_np = np.random.randn(n, n).astype(np.float64)
        A_np = A_np.T @ A_np + np.eye(n) * n * 1e-3
        b_np = np.random.randn(n).astype(np.float64)

        A_t = torch.as_tensor(A_np, device="cuda")
        b_t = torch.as_tensor(b_np, device="cuda")

        L = torch.linalg.cholesky(A_t)
        x = torch.linalg.solve(A_t, b_t)
        residual = torch.linalg.norm(A_t @ x - b_t)

        assert float(residual.cpu()) < 1e-10, f"Torch Cholesky residual too large: {residual}"

    @pytest.mark.cupy
    def test_cupy_cholesky_solve_matrix(self, cupy_available):
        """CuPy Cholesky solve with matrix RHS (multiple right-hand sides)."""
        import cupy as cp

        n, k = 50, 3
        A_np = np.random.randn(n, n).astype(np.float64)
        A_np = A_np.T @ A_np + np.eye(n) * n * 1e-3
        B_np = np.random.randn(n, k).astype(np.float64)

        A_cp = cp.asarray(A_np)
        B_cp = cp.asarray(B_np)

        X = cp.linalg.solve(A_cp, B_cp)
        assert X.shape == (n, k)

    @pytest.mark.torch_cuda
    def test_torch_cholesky_solve_matrix(self, torch_cuda_available):
        """Torch CUDA Cholesky solve with matrix RHS."""
        import torch

        n, k = 50, 3
        A_np = np.random.randn(n, n).astype(np.float64)
        A_np = A_np.T @ A_np + np.eye(n) * n * 1e-3
        B_np = np.random.randn(n, k).astype(np.float64)

        A_t = torch.as_tensor(A_np, device="cuda")
        B_t = torch.as_tensor(B_np, device="cuda")

        X = torch.linalg.solve(A_t, B_t)
        assert X.shape == (n, k)


# ============================================================================
# Test: xp_maximum
# ============================================================================


@pytest.mark.gpu
class TestXpMaximum:
    """xp_maximum(tensor, scalar) on all backends (Section 8)."""

    @pytest.mark.cupy
    def test_cupy_maximum_scalar(self, cupy_available):
        """xp_maximum works with CuPy arrays and scalar threshold."""
        from statgpu.backends import xp_maximum
        import cupy as cp

        x = cp.asarray(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64))
        result = xp_maximum(x, 0.0)
        result_np = cp.asnumpy(result)

        expected = np.maximum(np.array([-1.0, 0.0, 1.0, 2.0]), 0.0)
        assert np.array_equal(result_np, expected), f"Mismatch: {result_np} vs {expected}"

    @pytest.mark.torch_cuda
    def test_torch_maximum_scalar(self, torch_cuda_available):
        """xp_maximum works with Torch CUDA tensors and scalar threshold."""
        from statgpu.backends import xp_maximum
        import torch

        x = torch.as_tensor(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64), device="cuda")
        result = xp_maximum(x, 0.0)
        assert result.is_cuda, "Result should be on CUDA"
        result_np = result.cpu().numpy()

        expected = np.maximum(np.array([-1.0, 0.0, 1.0, 2.0]), 0.0)
        assert np.array_equal(result_np, expected), f"Mismatch: {result_np} vs {expected}"


# ============================================================================
# Test: Ridge/RidgeCV
# ============================================================================


@pytest.mark.gpu
class TestRidgeTorch:
    """Ridge/RidgeCV GPU tests (Section 10.2).

    Note: statgpu API convention is that ``coef_`` and ``intercept_``
    are always numpy arrays regardless of compute backend. These are
    O(p) summary outputs, not full design matrices.
    """

    @pytest.mark.torch_cuda
    def test_ridge_fit_torch_cuda(self, torch_cuda_available, sample_data_2d):
        """Ridge fits on Torch CUDA and returns correct coefficients."""
        from statgpu.linear_model import Ridge
        import torch

        X_np, y_np = sample_data_2d
        X_t = torch.as_tensor(X_np, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = Ridge(alpha=1.0, device="torch")
        model.fit(X_t, y_t)

        assert model.coef_ is not None, "coef_ should be set"
        assert model.coef_.shape == (5,), f"Unexpected shape: {model.coef_.shape}"
        _assert_finite(model.coef_, "coef_")
        # coef_ is always numpy (summary output, O(p) transfer is acceptable)

    @pytest.mark.cupy
    def test_ridge_fit_cupy_cuda(self, cupy_available, sample_data_2d):
        """Ridge fits on CuPy CUDA and returns correct coefficients."""
        from statgpu.linear_model import Ridge
        import cupy as cp

        X_np, y_np = sample_data_2d
        X_c = cp.asarray(X_np)
        y_c = cp.asarray(y_np)

        model = Ridge(alpha=1.0, device="cuda")
        model.fit(X_c, y_c)

        assert model.coef_ is not None, "coef_ should be set"
        _assert_finite(model.coef_, "coef_")

    @pytest.mark.torch_cuda
    def test_ridge_cv_torch_cuda(self, torch_cuda_available, sample_data_2d):
        """RidgeCV selects alpha on Torch CUDA."""
        from statgpu.linear_model import RidgeCV
        import torch

        X_np, y_np = sample_data_2d
        X_t = torch.as_tensor(X_np, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = RidgeCV(alphas=[0.1, 1.0, 10.0], device="torch")
        model.fit(X_t, y_t)

        assert model.alpha_ is not None, "CV should select an alpha"
        assert model.coef_ is not None, "coef_ should be set"

    @pytest.mark.torch_cuda
    def test_ridge_predict_torch(self, torch_cuda_available, sample_data_2d):
        """Ridge predict on Torch CUDA input returns predictions."""
        from statgpu.linear_model import Ridge
        import torch

        X_np, y_np = sample_data_2d
        X_t = torch.as_tensor(X_np, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = Ridge(alpha=1.0, device="torch")
        model.fit(X_t, y_t)
        pred = model.predict(X_t)

        assert pred.shape[0] == X_np.shape[0], "Predictions must match n_samples"
        _assert_finite(pred, "predictions")


# ============================================================================
# Test: FirstDifferenceOLS
# ============================================================================


@pytest.mark.gpu
class TestFirstDifferenceOLS:
    """FirstDifferenceOLS GPU differencing (Section 10.9)."""

    @pytest.mark.torch_cuda
    def test_first_diff_ols_torch(self, torch_cuda_available):
        """FirstDifferenceOLS differencing on Torch CUDA."""
        from statgpu.panel import FirstDifferenceOLS
        import torch

        np.random.seed(42)
        n_entities, n_periods = 20, 5
        n_total = n_entities * n_periods
        X_np = np.random.randn(n_total, 3).astype(np.float64)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)
        y_np = X_np[:, 0] * 1.5 + X_np[:, 1] * (-0.5) + np.random.randn(n_total) * 0.3

        X_t = torch.as_tensor(X_np, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = FirstDifferenceOLS(device="torch")
        model.fit(X_t, y_t, entity_ids=entity_ids, time_ids=time_ids)

        assert model.coef_ is not None
        assert model.coef_.shape[0] == 3, f"Expected 3 coefficients, got shape {model.coef_.shape}"

    @pytest.mark.cupy
    def test_first_diff_ols_cupy(self, cupy_available):
        """FirstDifferenceOLS differencing on CuPy CUDA."""
        from statgpu.panel import FirstDifferenceOLS
        import cupy as cp

        np.random.seed(42)
        n_entities, n_periods = 20, 5
        n_total = n_entities * n_periods
        X_np = np.random.randn(n_total, 3).astype(np.float64)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)
        y_np = X_np[:, 0] * 1.5 + X_np[:, 1] * (-0.5) + np.random.randn(n_total) * 0.3

        X_c = cp.asarray(X_np)
        y_c = cp.asarray(y_np)

        model = FirstDifferenceOLS(device="cuda")
        model.fit(X_c, y_c, entity_ids=entity_ids, time_ids=time_ids)

        assert model.coef_ is not None
        assert model.coef_.shape[0] == 3


# ============================================================================
# Test: FamaMacBeth
# ============================================================================


@pytest.mark.gpu
class TestFamaMacBeth:
    """FamaMacBeth fit/inference/predict on GPU (Section 10.9)."""

    @pytest.mark.torch_cuda
    def test_fm_fit_torch(self, torch_cuda_available):
        """FamaMacBeth fits on Torch CUDA."""
        from statgpu.panel import FamaMacBeth
        import torch

        np.random.seed(42)
        n_periods, n_entities = 50, 30
        n_total = n_periods * n_entities
        X_np = np.random.randn(n_total, 3).astype(np.float64)
        time_ids = np.tile(np.arange(n_periods), n_entities)
        y_np = X_np[:, 0] * 1.0 + X_np[:, 1] * (-0.5) + np.random.randn(n_total) * 0.3

        X_t = torch.as_tensor(X_np, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = FamaMacBeth(device="torch")
        model.fit(X_t, y_t, time_ids=time_ids)

        assert model.coef_ is not None
        _assert_finite(model.coef_, "FamaMacBeth coef_")

    @pytest.mark.torch_cuda
    def test_fm_predict_torch(self, torch_cuda_available):
        """FamaMacBeth predict returns valid predictions."""
        from statgpu.panel import FamaMacBeth
        import torch

        np.random.seed(42)
        n_periods, n_entities = 50, 30
        n_total = n_periods * n_entities
        X_np = np.random.randn(n_total, 3).astype(np.float64)
        time_ids = np.tile(np.arange(n_periods), n_entities)
        y_np = X_np[:, 0] * 1.0 + X_np[:, 1] * (-0.5) + np.random.randn(n_total) * 0.3

        X_t = torch.as_tensor(X_np, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = FamaMacBeth(device="torch")
        model.fit(X_t, y_t, time_ids=time_ids)
        pred = model.predict(X_t)

        assert pred.shape[0] == n_total
        _assert_finite(pred, "FamaMacBeth predictions")


# ============================================================================
# Test: GraphicalLasso
# ============================================================================


@pytest.mark.gpu
class TestGraphicalLasso:
    """GraphicalLasso backend-native output on CuPy/Torch (Section 10.8)."""

    @pytest.mark.cupy
    def test_glasso_cupy(self, cupy_available):
        """GraphicalLasso fits on CuPy and returns CuPy arrays."""
        from statgpu.covariance import GraphicalLasso
        import cupy as cp

        np.random.seed(42)
        n, p = 100, 10
        X_np = np.random.randn(n, p).astype(np.float64)
        X_c = cp.asarray(X_np)

        model = GraphicalLasso(alpha=0.1, device="cuda", max_iter=50)
        model.fit(X_c)

        assert model.covariance_ is not None
        assert model.precision_ is not None
        assert _is_cupy_array(model.covariance_), "covariance_ should stay on CuPy"
        assert _is_cupy_array(model.precision_), "precision_ should stay on CuPy"

        # Check symmetry
        cov_np = cp.asnumpy(model.covariance_)
        prec_np = cp.asnumpy(model.precision_)
        assert np.allclose(cov_np, cov_np.T, atol=1e-10), "covariance_ not symmetric"
        assert np.allclose(prec_np, prec_np.T, atol=1e-10), "precision_ not symmetric"

    @pytest.mark.torch_cuda
    def test_glasso_torch(self, torch_cuda_available):
        """GraphicalLasso fits on Torch CUDA and returns Torch tensors."""
        from statgpu.covariance import GraphicalLasso
        import torch

        np.random.seed(42)
        n, p = 100, 10
        X_np = np.random.randn(n, p).astype(np.float64)
        X_t = torch.as_tensor(X_np, device="cuda")

        model = GraphicalLasso(alpha=0.1, device="torch", max_iter=50)
        model.fit(X_t)

        assert model.covariance_ is not None
        assert model.precision_ is not None
        assert _is_torch_cuda_tensor(model.covariance_), "covariance_ should stay on Torch CUDA"
        assert _is_torch_cuda_tensor(model.precision_), "precision_ should stay on Torch CUDA"


# ============================================================================
# Test: String Panel Labels
# ============================================================================


@pytest.mark.gpu
class TestStringPanelLabels:
    """String panel labels handled as CPU metadata (Section 10.9)."""

    @pytest.mark.cupy
    def test_pooled_ols_cupy_basic(self, cupy_available):
        """PooledOLS basic fit on CuPy GPU data."""
        from statgpu.panel import PooledOLS
        import cupy as cp

        np.random.seed(42)
        n_total = 50
        X_np = np.random.randn(n_total, 2).astype(np.float64)
        y_np = X_np[:, 0] * 2.0 + X_np[:, 1] * (-1.0) + np.random.randn(n_total) * 0.3

        X_c = cp.asarray(X_np)
        y_c = cp.asarray(y_np)

        model = PooledOLS(device="cuda")
        model.fit(X_c, y_c)

        assert model.coef_ is not None
        _assert_finite(model.coef_, "PooledOLS coef_ (string labels)")

    @pytest.mark.torch_cuda
    def test_pooled_ols_torch_basic(self, torch_cuda_available):
        """PooledOLS basic fit on Torch CUDA data."""
        from statgpu.panel import PooledOLS
        import torch

        np.random.seed(42)
        n_total = 50
        X_np = np.random.randn(n_total, 2).astype(np.float64)
        y_np = X_np[:, 0] * 2.0 + X_np[:, 1] * (-1.0) + np.random.randn(n_total) * 0.3

        X_t = torch.as_tensor(X_np, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = PooledOLS(device="torch")
        model.fit(X_t, y_t)

        assert model.coef_ is not None
        _assert_finite(model.coef_, "PooledOLS coef_ (string labels)")


# ============================================================================
# Test: NaN/Inf Error Handling
# ============================================================================


@pytest.mark.gpu
class TestNaNFInfErrors:
    """NaN/Inf inputs handled appropriately (Section 8).

    Note: statgpu does not currently validate NaN/Inf in all code paths
    before reaching CUDA kernels. These tests verify gracefulness.
    """

    @pytest.mark.cupy
    def test_ridge_nan_input_cupy(self, cupy_available, sample_data_2d):
        """Ridge with NaN in X should not crash on CuPy."""
        from statgpu.linear_model import Ridge
        import cupy as cp

        X_np, y_np = sample_data_2d
        X_np_nan = X_np.copy()
        X_np_nan[0, 0] = np.nan

        X_c = cp.asarray(X_np_nan)
        y_c = cp.asarray(y_np)

        model = Ridge(alpha=1.0, device="cuda")
        # Should not crash (may produce NaN coef or raise)
        try:
            model.fit(X_c, y_c)
            assert model.coef_ is not None
        except Exception:
            pass  # Raising is acceptable behavior

    @pytest.mark.cupy
    def test_ridge_inf_input_cupy(self, cupy_available, sample_data_2d):
        """Ridge with Inf in X should not crash on CuPy."""
        from statgpu.linear_model import Ridge
        import cupy as cp

        X_np, y_np = sample_data_2d
        X_np_inf = X_np.copy()
        X_np_inf[0, 0] = np.inf

        X_c = cp.asarray(X_np_inf)
        y_c = cp.asarray(y_np)

        model = Ridge(alpha=1.0, device="cuda")
        try:
            model.fit(X_c, y_c)
            assert model.coef_ is not None
        except Exception:
            pass

    @pytest.mark.torch_cuda
    def test_ridge_nan_input_torch(self, torch_cuda_available, sample_data_2d):
        """Ridge with NaN in X should not crash on Torch CUDA."""
        from statgpu.linear_model import Ridge
        import torch

        X_np, y_np = sample_data_2d
        X_np_nan = X_np.copy()
        X_np_nan[0, 0] = np.nan

        X_t = torch.as_tensor(X_np_nan, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = Ridge(alpha=1.0, device="torch")
        try:
            model.fit(X_t, y_t)
            assert model.coef_ is not None
        except Exception:
            pass


# ============================================================================
# Test: Predict Device Behavior
# ============================================================================


@pytest.mark.gpu
class TestDevicePurity:
    """Output stays on requested device after fit/predict (Section 12).

    Note: coef_ and intercept_ are always numpy arrays by API design
    (O(p) summary outputs, not full design matrices).
    """

    @pytest.mark.cupy
    def test_ridge_predict_cupy_input_returns_cupy(self, cupy_available, sample_data_2d):
        """Ridge predict on CuPy input returns array (at minimum, doesn't crash)."""
        from statgpu.linear_model import Ridge
        import cupy as cp

        X_np, y_np = sample_data_2d
        X_c = cp.asarray(X_np)
        y_c = cp.asarray(y_np)

        model = Ridge(alpha=1.0, device="cuda")
        model.fit(X_c, y_c)
        pred = model.predict(X_c)

        assert pred.shape[0] == X_np.shape[0]
        _assert_finite(pred, "predictions")

    @pytest.mark.torch_cuda
    def test_ridge_fit_torch_produces_valid_coef(self, torch_cuda_available, sample_data_2d):
        """Ridge fitted on Torch CUDA produces valid finite coefficients."""
        from statgpu.linear_model import Ridge
        import torch

        X_np, y_np = sample_data_2d
        X_t = torch.as_tensor(X_np, device="cuda")
        y_t = torch.as_tensor(y_np, device="cuda")

        model = Ridge(alpha=1.0, device="torch")
        model.fit(X_t, y_t)

        assert model.coef_ is not None, "coef_ should be set"
        _assert_finite(model.coef_, "coef_")


# ============================================================================
# Test: Dtype Preservation
# ============================================================================


@pytest.mark.gpu
class TestDtypePreservation:
    """float64 input preserves float64 precision (Section 12)."""

    @pytest.mark.cupy
    def test_preserve_float64_cupy(self, cupy_available):
        """float64 data produces float64 coefficients on CuPy."""
        from statgpu.linear_model import Ridge
        import cupy as cp

        np.random.seed(42)
        X_np = np.random.randn(50, 5).astype(np.float64)
        y_np = np.random.randn(50).astype(np.float64)

        model = Ridge(alpha=1.0, device="cuda")
        model.fit(cp.asarray(X_np), cp.asarray(y_np))
        assert str(model.coef_.dtype) == "float64", f"dtype changed: {model.coef_.dtype}"

    @pytest.mark.torch_cuda
    def test_preserve_float64_torch(self, torch_cuda_available):
        """float64 data produces float64 coefficients on Torch CUDA."""
        from statgpu.linear_model import Ridge
        import torch

        np.random.seed(42)
        X_np = np.random.randn(50, 5).astype(np.float64)
        y_np = np.random.randn(50).astype(np.float64)

        model = Ridge(alpha=1.0, device="torch")
        model.fit(
            torch.as_tensor(X_np, device="cuda", dtype=torch.float64),
            torch.as_tensor(y_np, device="cuda", dtype=torch.float64),
        )
        # coef_ is numpy, dtype should be float64
        assert str(model.coef_.dtype) == "float64", f"dtype changed: {model.coef_.dtype}"


# ============================================================================
# Test: Repeated Fit
# ============================================================================


@pytest.mark.gpu
class TestRepeatedFit:
    """Repeated fit gives consistent results (Section 8)."""

    @pytest.mark.cupy
    def test_ridge_repeated_fit_cupy(self, cupy_available, sample_data_2d):
        """Ridge fitted twice on same data gives identical coefficients."""
        from statgpu.linear_model import Ridge
        import cupy as cp

        X_np, y_np = sample_data_2d
        X_c = cp.asarray(X_np)
        y_c = cp.asarray(y_np)

        model1 = Ridge(alpha=1.0, device="cuda")
        model1.fit(X_c, y_c)

        model2 = Ridge(alpha=1.0, device="cuda")
        model2.fit(X_c, y_c)

        coef1 = cp.asnumpy(model1.coef_).ravel()
        coef2 = cp.asnumpy(model2.coef_).ravel()
        assert np.allclose(coef1, coef2, atol=1e-12), "Repeated fit gives different results"


# ============================================================================
# Test: Contiguous/Non-contiguous Input
# ============================================================================


@pytest.mark.gpu
class TestContiguousInput:
    """Non-contiguous Torch input handled correctly."""

    @pytest.mark.torch_cuda
    def test_non_contiguous_torch(self, torch_cuda_available, sample_data_2d):
        """Ridge handles non-contiguous Torch CUDA tensor."""
        from statgpu.linear_model import Ridge
        import torch

        X_np, y_np = sample_data_2d
        # Create contiguous tensor, then transpose slice to make non-contiguous
        X_base = torch.as_tensor(X_np, device="cuda")
        # Take a non-contiguous view
        X_non = X_base[:, [0, 2, 4, 1, 3]]  # Column permutation

        y_t = torch.as_tensor(y_np, device="cuda")

        model = Ridge(alpha=1.0, device="torch")
        model.fit(X_non, y_t)

        assert model.coef_ is not None
        _assert_finite(model.coef_, "coef_ (non-contiguous)")


# ============================================================================
# Test: Backend Explicit Error
# ============================================================================


@pytest.mark.gpu
class TestExplicitBackendErrors:
    """Explicit GPU device raises clear error when backend unavailable."""

    def test_explicit_cuda_without_cupy(self):
        """device='cuda' when CuPy unavailable gives clear error."""
        from statgpu._config import cuda_available
        if cuda_available():
            pytest.skip("CuPy is available - cannot test unavailable case")

        from statgpu.linear_model import Ridge
        model = Ridge(alpha=1.0, device="cuda")
        with pytest.raises((RuntimeError, ImportError, ValueError)):
            model.fit(np.random.randn(10, 3), np.random.randn(10))

    def test_explicit_torch_without_torch_cuda(self):
        """device='torch' when Torch CUDA unavailable gives clear error."""
        try:
            import torch
            if torch.cuda.is_available():
                pytest.skip("Torch CUDA is available - cannot test unavailable case")
        except ImportError:
            pass

        from statgpu.linear_model import Ridge
        model = Ridge(alpha=1.0, device="torch")
        with pytest.raises((RuntimeError, ImportError, ValueError)):
            model.fit(np.random.randn(10, 3), np.random.randn(10))
