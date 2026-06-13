"""
Tests for all code-review fixes in PR #47.

Covers:
  1. _norm() CuPy branch
  2. max_iter=0 guard (UnboundLocalError)
  3. _to_numpy .detach() for grad-tracked tensors
  4. CUDA cummin/cummax float32 support
  5. CUDA cummin/cummax None kernel guard
  6. Newton solver loss.preprocess() call
  7. Link-based intercept augmentation (gamma/tweedie)
  8. _get_torch_device_str exception handling
  9. _smooth_penalty_hessian returns scalar 0.0
 10. penalty.value() direct call (no double GPU transfer)
"""

import numpy as np
import pytest


# ── 1. _norm() CuPy branch ──────────────────────────────────────────────────

class TestNormCuPy:
    """Verify _norm works on CuPy arrays (was using np.linalg.norm)."""

    def test_norm_cupy(self):
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not installed")
        from statgpu.glm_core._irls import _norm
        x = cp.array([3.0, 4.0])
        result = _norm(x, "cupy")
        assert abs(result - 5.0) < 1e-10

    def test_norm_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("Torch not installed")
        from statgpu.glm_core._irls import _norm
        x = torch.tensor([3.0, 4.0])
        result = _norm(x, "torch")
        assert abs(result - 5.0) < 1e-10

    def test_norm_numpy(self):
        from statgpu.glm_core._irls import _norm
        x = np.array([3.0, 4.0])
        result = _norm(x, "numpy")
        assert abs(result - 5.0) < 1e-10


# ── 2. max_iter=0 guard ─────────────────────────────────────────────────────

class TestMaxIterZero:
    """Verify irls_solver doesn't crash when max_iter=0."""

    def test_max_iter_zero_numpy(self):
        from statgpu.glm_core._irls import irls_solver
        from statgpu.glm_core._family import Gaussian

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(20) * 0.1
        family = Gaussian()

        # Should not raise UnboundLocalError
        params, n_iter = irls_solver(family, X, y, max_iter=0)
        assert n_iter == 0


# ── 3. _to_numpy .detach() ──────────────────────────────────────────────────

class TestToNumpyDetach:
    """Verify _to_numpy handles grad-tracked torch tensors."""

    def test_detach_grad_tensor(self):
        try:
            import torch
        except ImportError:
            pytest.skip("Torch not installed")
        from statgpu.backends._utils import _to_numpy

        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = _to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_detach_no_grad(self):
        try:
            import torch
        except ImportError:
            pytest.skip("Torch not installed")
        from statgpu.backends._utils import _to_numpy

        x = torch.tensor([4.0, 5.0])
        result = _to_numpy(x)
        np.testing.assert_array_equal(result, [4.0, 5.0])


# ── 4. CUDA cummin/cummax float32 support ───────────────────────────────────

class TestCumopFloat32:
    """Verify cummin/cummax produce correct results for float32 arrays."""

    def test_cummin_float32(self):
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not installed")
        from statgpu.backends._cupy import CuPyBackend
        backend = CuPyBackend()

        x = cp.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=cp.float32)
        result = backend.cummin(x)
        expected = np.array([3.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(cp.asnumpy(result), expected)
        assert result.dtype == cp.float32

    def test_cummax_float32(self):
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not installed")
        from statgpu.backends._cupy import CuPyBackend
        backend = CuPyBackend()

        x = cp.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=cp.float32)
        result = backend.cummax(x)
        expected = np.array([3.0, 3.0, 4.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_array_equal(cp.asnumpy(result), expected)
        assert result.dtype == cp.float32

    def test_cummin_2d_float32(self):
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not installed")
        from statgpu.backends._cupy import CuPyBackend
        backend = CuPyBackend()

        x = cp.array([[3.0, 1.0, 4.0], [2.0, 5.0, 0.0]], dtype=cp.float32)
        result = backend.cummin(x, axis=1)
        expected = np.array([[3.0, 1.0, 1.0], [2.0, 2.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(cp.asnumpy(result), expected)
        assert result.dtype == cp.float32

    def test_cummin_float64_unchanged(self):
        """Ensure float64 still works (regression)."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not installed")
        from statgpu.backends._cupy import CuPyBackend
        backend = CuPyBackend()

        x = cp.array([3.0, 1.0, 4.0], dtype=cp.float64)
        result = backend.cummin(x)
        assert result.dtype == cp.float64
        np.testing.assert_array_equal(cp.asnumpy(result), [3.0, 1.0, 1.0])


# ── 5. CUDA cummin/cummax None kernel guard ─────────────────────────────────

class TestCumopNoneGuard:
    """Verify _launch_cumop raises RuntimeError when arr/result is None."""

    def test_launch_1d_none_guard(self):
        from statgpu.backends._cupy import _launch_cumop_1d
        with pytest.raises(RuntimeError, match="failed to compile|unavailable"):
            _launch_cumop_1d(None, None, 1, True)

    def test_launch_2d_none_guard(self):
        from statgpu.backends._cupy import _launch_cumop_2d
        with pytest.raises(RuntimeError, match="failed to compile|unavailable"):
            _launch_cumop_2d(None, None, 1, 1, True)


# ── 6. Newton solver loss.preprocess() ──────────────────────────────────────

class TestNewtonPreprocess:
    """Verify newton_solver calls loss.preprocess()."""

    def test_newton_with_preprocessing(self):
        from statgpu.glm_core._solver import newton_solver
        from statgpu.glm_core._squared import SquaredErrorLoss
        from statgpu.penalties._l2 import L2Penalty

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(50) * 0.1

        loss = SquaredErrorLoss()
        penalty = L2Penalty(alpha=0.0)

        # Should work without error (preprocess was missing before)
        params, n_iter = newton_solver(loss, penalty, X, y, max_iter=50)
        assert params.shape == (3,)
        assert n_iter > 0


# ── 7. Link-based intercept augmentation ────────────────────────────────────

class TestLinkBasedIntercept:
    """Verify FISTA uses link-based check for intercept augmentation."""

    def test_gamma_fista_fit(self):
        """Gamma uses log link — should augment, not center."""
        from statgpu.linear_model._gamma_glm import GammaRegression

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        # Gamma: positive y
        eta = X @ np.array([0.5, -0.3, 0.2]) + 1.0
        y = np.exp(eta) + rng.standard_normal(100) * 0.01
        y = np.clip(y, 0.01, None)

        model = GammaRegression(fit_intercept=True, solver="fista", max_iter=200)
        model.fit(X, y)

        # Should have fitted without error and have an intercept
        assert hasattr(model, 'intercept_')
        assert hasattr(model, 'coef_')
        assert model.coef_.shape == (3,)


# ── 8. _get_torch_device_str exception handling ─────────────────────────────

class TestTorchDeviceStr:
    """Verify _get_torch_device_str handles exceptions properly."""

    def test_returns_string(self):
        from statgpu.backends._utils import _get_torch_device_str
        result = _get_torch_device_str()
        assert result in ("cpu", "cuda")


# ── 9. _smooth_penalty_hessian returns scalar ───────────────────────────────

class TestSmoothPenaltyHessian:
    """Verify _smooth_penalty_hessian returns scalar 0.0 for no penalty."""

    def test_returns_scalar(self):
        from statgpu.glm_core._solver import _smooth_penalty_hessian
        coef = np.array([1.0, 2.0, 3.0])
        result = _smooth_penalty_hessian(None, coef)
        assert result == 0.0
        assert not isinstance(result, np.ndarray)

    def test_returns_scalar_null_penalty(self):
        from statgpu.glm_core._solver import _smooth_penalty_hessian
        # Pass a mock penalty with name="null"
        class _NullPenalty:
            name = "null"
        coef = np.array([1.0, 2.0, 3.0])
        result = _smooth_penalty_hessian(_NullPenalty(), coef)
        assert result == 0.0


# ── 10. penalty.value() direct call ─────────────────────────────────────────

class TestPenaltyValueDirect:
    """Verify penalty.value() works without _to_numpy wrapping."""

    def test_l2_penalty_value(self):
        from statgpu.penalties._l2 import L2Penalty
        penalty = L2Penalty(alpha=1.0)
        coef = np.array([1.0, 2.0, 3.0])
        val = penalty.value(coef)
        assert isinstance(val, (float, np.floating))
        assert val >= 0

    def test_l1_penalty_value(self):
        from statgpu.penalties._l1 import L1Penalty
        penalty = L1Penalty(alpha=1.0)
        coef = np.array([1.0, -2.0, 3.0])
        val = penalty.value(coef)
        assert abs(float(val) - 6.0) < 1e-10


# ── 11. FISTA convergence check deduplication ───────────────────────────────

class TestFistaConvergence:
    """Verify FISTA solver convergence check works after deduplication."""

    def test_fista_squared_error(self):
        from statgpu.glm_core._solver import fista_solver
        from statgpu.glm_core._squared import SquaredErrorLoss
        from statgpu.penalties._l2 import L2Penalty

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.array([1.0, -1.0, 0.5, 0.0, 0.0]) + rng.standard_normal(100) * 0.1

        loss = SquaredErrorLoss()
        penalty = L2Penalty(alpha=0.0)
        params, n_iter = fista_solver(loss, penalty, X, y, max_iter=500)

        assert params.shape == (5,)
        assert n_iter > 0
        # Should converge to something reasonable
        assert np.linalg.norm(params[:3] - np.array([1.0, -1.0, 0.5])) < 0.5


# ── 12. IRLS end-to-end with fit_intercept guard ────────────────────────────

class TestIRLSInterceptGuard:
    """Verify IRLS log-link warm-start only when intercept column present."""

    def test_poisson_with_intercept(self):
        """fit_intercept=True → ones column → warm-start should fire."""
        from statgpu.linear_model._poisson_glm import PoissonRegression

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        eta = X @ np.array([0.5, -0.3, 0.2]) + 1.0
        y = rng.poisson(np.exp(eta))

        model = PoissonRegression(fit_intercept=True, max_iter=100)
        model.fit(X, y)
        assert hasattr(model, 'intercept_')
        assert model.coef_.shape == (3,)

    def test_poisson_without_intercept(self):
        """fit_intercept=False → no ones column → no warm-start."""
        from statgpu.linear_model._poisson_glm import PoissonRegression

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        eta = X @ np.array([0.5, -0.3, 0.2])
        y = rng.poisson(np.exp(eta))

        model = PoissonRegression(fit_intercept=False, max_iter=100)
        model.fit(X, y)
        assert model.intercept_ == 0.0
        assert model.coef_.shape == (3,)


# ── 13. Exception narrowing (smoke tests) ───────────────────────────────────

class TestExceptionNarrowing:
    """Verify that narrow exceptions don't break normal operation."""

    def test_irls_squared_error(self):
        """IRLS for squared error should work normally."""
        from statgpu.linear_model._linear import LinearRegression

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(100) * 0.1

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        assert model.coef_.shape == (3,)

    def test_irls_logistic(self):
        """IRLS for logistic should work normally."""
        from statgpu.linear_model._logistic import LogisticRegression

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        eta = X @ np.array([0.5, -0.3, 0.2])
        prob = 1 / (1 + np.exp(-eta))
        y = rng.binomial(1, prob)

        model = LogisticRegression(fit_intercept=True, max_iter=100)
        model.fit(X, y)
        assert model.coef_.shape == (3,)

    def test_irls_poisson(self):
        """IRLS for Poisson should work normally."""
        from statgpu.linear_model._poisson_glm import PoissonRegression

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        eta = X @ np.array([0.5, -0.3, 0.2]) + 1.0
        y = rng.poisson(np.exp(eta))

        model = PoissonRegression(fit_intercept=True, max_iter=100)
        model.fit(X, y)
        assert model.coef_.shape == (3,)

    def test_irls_gamma(self):
        """IRLS for Gamma should work normally."""
        from statgpu.linear_model._gamma_glm import GammaRegression

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        eta = X @ np.array([0.5, -0.3, 0.2]) + 1.0
        y = np.exp(eta) + rng.standard_normal(200) * 0.01
        y = np.clip(y, 0.01, None)

        model = GammaRegression(fit_intercept=True, max_iter=100)
        model.fit(X, y)
        assert model.coef_.shape == (3,)
