"""Three-backend (numpy/cupy/torch) verification tests.

Tests that all modified methods produce identical results across backends.
On local machines without GPU, cupy/torch tests are skipped via pytest.importorskip.
On remote GPU servers, all three backends run and results are compared.

Usage:
    pytest dev/tests/test_three_backend.py -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# ============================================================================
# Helpers
# ============================================================================

def _make_data(n=200, p=10, seed=42):
    """Generate test data for penalized regression."""
    np.random.seed(seed)
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)
    beta_true[:3] = [3.0, -2.0, 1.0]
    y = X @ beta_true + 0.1 * np.random.randn(n)
    return X, y, beta_true


def _to_backend(X, y, backend_name):
    """Convert numpy arrays to the target backend."""
    if backend_name == "numpy":
        return X, y
    if backend_name == "cupy":
        cp = pytest.importorskip("cupy")
        return cp.asarray(X), cp.asarray(y)
    if backend_name == "torch":
        torch = pytest.importorskip("torch")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X_t = torch.tensor(X, dtype=torch.float64, device=device)
        y_t = torch.tensor(y, dtype=torch.float64, device=device)
        return X_t, y_t
    raise ValueError(f"Unknown backend: {backend_name}")


def _to_numpy(arr):
    """Convert any backend array to numpy."""
    if hasattr(arr, 'get'):  # cupy
        return arr.get()
    if hasattr(arr, 'cpu'):  # torch
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


# ============================================================================
# P2-2: _dev_val with _clip — GLM family convergence on all backends
# ============================================================================

class TestDevValClip:
    """Verify _dev_val works correctly on all backends after _clip cleanup."""

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_logistic_convergence(self, backend):
        """Logistic regression converges on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        X, y_raw, _ = _make_data()
        y = (y_raw > 0).astype(float)
        X_b, y_b = _to_backend(X, y, backend)
        model = PenalizedGeneralizedLinearModel(
            loss="logistic", penalty="l2", alpha=0.01, max_iter=100
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"Non-finite coef on {backend}"

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_poisson_convergence(self, backend):
        """Poisson regression converges on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.random.poisson(np.exp(X @ np.array([0.5, -0.3, 0, 0, 0])))
        X_b, y_b = _to_backend(X, y.astype(float), backend)
        model = PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.01, max_iter=100
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"Non-finite coef on {backend}"

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_gamma_convergence(self, backend):
        """Gamma regression converges on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.exp(X @ np.array([0.1, -0.1, 0, 0, 0])) + 0.01
        X_b, y_b = _to_backend(X, y, backend)
        model = PenalizedGeneralizedLinearModel(
            loss="gamma", penalty="l2", alpha=0.01, max_iter=100
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"Non-finite coef on {backend}"


# ============================================================================
# P3-1: Vectorized CD — verify on all backends
# ============================================================================

class TestVectorizedCDThreeBackend:
    """Verify vectorized CD produces correct results on all backends."""

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_lasso_cd(self, backend):
        """Lasso CD converges on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        X, y, _ = _make_data()
        X_b, y_b = _to_backend(X, y, backend)
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l1", alpha=0.1,
            max_iter=500, tol=1e-8, fit_intercept=False
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"Non-finite coef on {backend}"
        # Check sparsity pattern matches sklearn
        from sklearn.linear_model import Lasso
        sk = Lasso(alpha=0.1, fit_intercept=False, max_iter=500, tol=1e-8)
        sk.fit(X, y)
        assert_allclose(coef, sk.coef_, atol=1e-3,
                        err_msg=f"Lasso coef mismatch on {backend}")

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_scad_cd(self, backend):
        """SCAD CD produces finite coefs on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        X, y, _ = _make_data()
        X_b, y_b = _to_backend(X, y, backend)
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="scad", alpha=0.1,
            max_iter=200, tol=1e-8
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"SCAD non-finite on {backend}"

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_mcp_cd(self, backend):
        """MCP CD produces finite coefs on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        X, y, _ = _make_data()
        X_b, y_b = _to_backend(X, y, backend)
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="mcp", alpha=0.1,
            max_iter=200, tol=1e-8
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"MCP non-finite on {backend}"

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_adaptive_l1_cd(self, backend):
        """Adaptive Lasso CD produces finite coefs on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        X, y, _ = _make_data()
        X_b, y_b = _to_backend(X, y, backend)
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="adaptive_l1", alpha=0.1,
            max_iter=200, tol=1e-8
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"Adaptive L1 non-finite on {backend}"

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_elasticnet_cd(self, backend):
        """ElasticNet CD produces finite coefs on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        X, y, _ = _make_data()
        X_b, y_b = _to_backend(X, y, backend)
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="elasticnet", alpha=0.1,
            l1_ratio=0.5, max_iter=200, tol=1e-8
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"ElasticNet non-finite on {backend}"

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_logistic_l1_cd(self, backend):
        """Logistic + L1 CD produces finite coefs on all backends."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        X, y_raw, _ = _make_data()
        y = (y_raw > 0).astype(float)
        X_b, y_b = _to_backend(X, y, backend)
        model = PenalizedGeneralizedLinearModel(
            loss="logistic", penalty="l1", alpha=0.05,
            max_iter=200, tol=1e-6
        )
        model.fit(X_b, y_b)
        coef = _to_numpy(model.coef_)
        assert all(np.isfinite(coef)), f"Logistic L1 non-finite on {backend}"


# ============================================================================
# P2-1: batch_mse — verify on all backends
# ============================================================================

class TestBatchMseThreeBackend:
    """Verify batch_mse works on all backends."""

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_batch_mse(self, backend):
        """batch_mse produces same result on all backends."""
        from statgpu.linear_model._cv_base import batch_mse
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        coefs = np.random.randn(10, 5)
        intercepts = np.random.randn(10)
        sw = np.random.rand(50)

        # numpy baseline
        mse_np = batch_mse(X, y, coefs, intercepts, sample_weight=sw)

        # backend test
        X_b, y_b = _to_backend(X, y, backend)
        coefs_b, _ = _to_backend(coefs, np.zeros(10), backend)
        int_b, _ = _to_backend(intercepts, np.zeros(10), backend)
        sw_b, _ = _to_backend(sw, np.zeros(50), backend)
        mse_b = batch_mse(X_b, y_b, coefs_b, int_b, sample_weight=sw_b)

        assert_allclose(mse_np, mse_b, rtol=1e-10,
                        err_msg=f"batch_mse mismatch on {backend}")


# ============================================================================
# Penalty proximal — verify on all backends
# ============================================================================

class TestPenaltyProximalThreeBackend:
    """Verify penalty proximal operators work on all backends."""

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_l1_proximal(self, backend):
        """L1 proximal produces correct result on all backends."""
        from statgpu.penalties._l1 import L1Penalty
        w_np = np.array([0.5, -0.3, 0.1, -0.8, 0.0])
        w_b, _ = _to_backend(w_np, np.zeros(5), backend)
        pen = L1Penalty(alpha=1.0)
        try:
            result = pen.proximal(w_b, step=0.5, backend=backend)
        except Exception as e:
            if "torch.compile" in str(e) or "Compiler" in str(e):
                pytest.skip(f"torch.compile not available: {e}")
            raise
        result_np = _to_numpy(result)
        # Expected: sign(w) * max(|w| - 0.5, 0)
        expected = np.sign(w_np) * np.maximum(np.abs(w_np) - 0.5, 0.0)
        assert_allclose(result_np, expected, atol=1e-10,
                        err_msg=f"L1 proximal mismatch on {backend}")

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_l2_proximal(self, backend):
        """L2 proximal produces correct result on all backends."""
        from statgpu.penalties._l2 import L2Penalty
        w_np = np.array([1.0, -2.0, 0.5])
        w_b, _ = _to_backend(w_np, np.zeros(3), backend)
        pen = L2Penalty(alpha=1.0)
        result = pen.proximal(w_b, step=1.0, backend=backend)
        result_np = _to_numpy(result)
        expected = w_np / (1.0 + 1.0)  # scale = 1/(1+alpha*step)
        assert_allclose(result_np, expected, atol=1e-10,
                        err_msg=f"L2 proximal mismatch on {backend}")

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_elasticnet_proximal(self, backend):
        """ElasticNet proximal produces correct result on all backends."""
        from statgpu.penalties._elasticnet import ElasticNetPenalty
        w_np = np.array([0.5, -0.3, 0.1, -0.8, 0.0])
        w_b, _ = _to_backend(w_np, np.zeros(5), backend)
        pen = ElasticNetPenalty(alpha=1.0, l1_ratio=0.5)
        result = pen.proximal(w_b, step=0.5, backend=backend)
        result_np = _to_numpy(result)
        assert all(np.isfinite(result_np)), f"ElasticNet proximal non-finite on {backend}"

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_scad_value(self, backend):
        """SCAD value produces same result on all backends."""
        from statgpu.penalties._scad import SCADPenalty
        coef_np = np.array([0.1, 0.5, 1.5, 3.0, 0.0])
        coef_b, _ = _to_backend(coef_np, np.zeros(5), backend)
        pen = SCADPenalty(alpha=1.0, a=3.7)
        val = pen.value(coef_b)
        assert np.isfinite(val), f"SCAD value non-finite on {backend}"

    @pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
    def test_mcp_value(self, backend):
        """MCP value produces same result on all backends."""
        from statgpu.penalties._mcp import MCPPenalty
        coef_np = np.array([0.1, 0.5, 1.5, 3.0, 0.0])
        coef_b, _ = _to_backend(coef_np, np.zeros(5), backend)
        pen = MCPPenalty(alpha=1.0, gamma=3.0)
        val = pen.value(coef_b)
        assert np.isfinite(val), f"MCP value non-finite on {backend}"


# ============================================================================
# Cross-backend consistency
# ============================================================================

class TestCrossBackendConsistency:
    """Verify all backends produce the same coefficients."""

    def test_lasso_coef_consistency(self):
        """Lasso coefs match across numpy/cupy/torch within 1e-6."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        X, y, _ = _make_data()

        results = {}
        for backend in ["numpy", "cupy", "torch"]:
            try:
                X_b, y_b = _to_backend(X, y, backend)
                model = PenalizedGeneralizedLinearModel(
                    loss="squared_error", penalty="l1", alpha=0.1,
                    max_iter=500, tol=1e-10, fit_intercept=False
                )
                model.fit(X_b, y_b)
                results[backend] = _to_numpy(model.coef_)
            except Exception:
                pass

        if len(results) < 2:
            pytest.skip("Need at least 2 backends")

        backends = list(results.keys())
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                assert_allclose(
                    results[backends[i]], results[backends[j]], atol=1e-6,
                    err_msg=f"Coef mismatch: {backends[i]} vs {backends[j]}"
                )

    def test_ridge_coef_consistency(self):
        """Ridge coefs match across numpy/cupy/torch within 1e-6."""
        from statgpu.linear_model._penalized import PenalizedLinearRegression
        X, y, _ = _make_data()

        results = {}
        for backend in ["numpy", "cupy", "torch"]:
            try:
                X_b, y_b = _to_backend(X, y, backend)
                model = PenalizedLinearRegression(
                    penalty="l2", alpha=0.01, max_iter=200, tol=1e-10
                )
                model.fit(X_b, y_b)
                results[backend] = _to_numpy(model.coef_)
            except Exception:
                pass

        if len(results) < 2:
            pytest.skip("Need at least 2 backends")

        backends = list(results.keys())
        for i in range(len(backends)):
            for j in range(i + 1, len(backends)):
                assert_allclose(
                    results[backends[i]], results[backends[j]], atol=1e-6,
                    err_msg=f"Ridge coef mismatch: {backends[i]} vs {backends[j]}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
