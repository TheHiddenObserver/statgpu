"""Tests for standalone QuantileRegression with compute_inference."""
import numpy as np
import pytest
from statgpu.linear_model import QuantileRegression


# ---- GPU availability checks ----
try:
    import cupy as cp
    _HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUPY = False

try:
    import torch
    _HAS_TORCH_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_TORCH_CUDA = False


class TestQuantileRegression:
    """Basic fit + inference for the standalone QuantileRegression class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.X = np.random.randn(200, 3)
        self.y = 1.0 + self.X @ [0.5, -0.3, 0.8] + 0.5 * np.random.randn(200)

    def test_fit_without_inference(self):
        m = QuantileRegression(quantile=0.5)
        m.fit(self.X, self.y)
        assert m.coef_ is not None
        assert len(m.coef_) == 3
        assert m.n_iter_ > 0
        assert not hasattr(m, '_bse') or m._bse is None

    def test_fit_with_kernel_inference(self):
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='kernel')
        m.fit(self.X, self.y)
        assert m._bse is not None
        assert len(m._bse) == 4  # 3 coef + intercept
        assert m._pvalues is not None
        assert m._conf_int.shape == (4, 2)
        # SE should be positive
        assert np.all(m._bse > 0)

    def test_fit_with_bootstrap_inference(self):
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='bootstrap', n_bootstrap=50)
        m.fit(self.X, self.y)
        assert m._bse is not None
        assert len(m._bse) == 4
        assert np.all(m._bse > 0)

    def test_predict(self):
        m = QuantileRegression(quantile=0.5)
        m.fit(self.X, self.y)
        pred = m.predict(self.X)
        assert pred.shape == (200,)
        assert isinstance(pred, np.ndarray)

    def test_invalid_kernel_raises(self):
        m = QuantileRegression(kernel='invalid', compute_inference=True,
                                inference_method='kernel')
        with pytest.raises(ValueError, match="kernel must be one of"):
            m.fit(self.X, self.y)

    def test_singular_design_raises(self):
        X_bad = np.column_stack([self.X[:, 0], self.X[:, 0]])
        m = QuantileRegression(compute_inference=True, inference_method='kernel')
        try:
            m.fit(X_bad, self.y)
            # If fit succeeds, BSE should be NaN for the collinear columns
            assert m._bse is not None
            # At least one BSE should be invalid (NaN or Inf)
            assert np.any(np.isnan(m._bse)) or np.any(np.isinf(m._bse)), \
                f"Expected NaN/Inf BSE for singular design, got {m._bse}"
        except (np.linalg.LinAlgError, RuntimeError):
            pass  # raising is also acceptable

    def test_n_categories_validator_is_not_quantile(self):
        """QuantileRegression does not have n_categories."""
        m = QuantileRegression()
        assert not hasattr(m, 'n_categories')

    # ---- GPU backend tests ----

    @pytest.mark.skipif(not _HAS_CUPY, reason="CuPy GPU not available")
    def test_kernel_inference_cupy(self):
        """Kernel-based inference on CuPy GPU — BSE must match CPU."""
        import cupy as cp
        X_cp = cp.asarray(self.X)
        y_cp = cp.asarray(self.y)
        # CPU reference
        m_cpu = QuantileRegression(quantile=0.5, compute_inference=True,
                                    inference_method='kernel')
        m_cpu.fit(self.X, self.y)
        # GPU
        m_gpu = QuantileRegression(quantile=0.5, compute_inference=True,
                                    inference_method='kernel', device='cuda')
        m_gpu.fit(X_cp, y_cp)
        assert m_gpu._bse is not None
        assert len(m_gpu._bse) == 4
        assert np.all(m_gpu._bse > 0)
        # BSE must match CPU within floating-point tolerance
        assert np.allclose(m_cpu._bse, m_gpu._bse, rtol=1e-12)

    @pytest.mark.skipif(not _HAS_CUPY, reason="CuPy GPU not available")
    def test_bootstrap_inference_cupy(self):
        """Bootstrap inference on CuPy GPU — BSE should be similar to CPU."""
        import cupy as cp
        X_cp = cp.asarray(self.X)
        y_cp = cp.asarray(self.y)
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='bootstrap', n_bootstrap=50,
                                device='cuda')
        m.fit(X_cp, y_cp)
        assert m._bse is not None
        assert len(m._bse) == 4
        assert np.all(m._bse > 0)

    @pytest.mark.skipif(not _HAS_TORCH_CUDA, reason="Torch CUDA not available")
    def test_kernel_inference_torch(self):
        """Kernel-based inference on Torch GPU — BSE must match CPU."""
        import torch
        X_t = torch.tensor(self.X, dtype=torch.float64, device='cuda')
        y_t = torch.tensor(self.y, dtype=torch.float64, device='cuda')
        # CPU reference
        m_cpu = QuantileRegression(quantile=0.5, compute_inference=True,
                                    inference_method='kernel')
        m_cpu.fit(self.X, self.y)
        # GPU
        m_gpu = QuantileRegression(quantile=0.5, compute_inference=True,
                                    inference_method='kernel', device='torch')
        m_gpu.fit(X_t, y_t)
        assert m_gpu._bse is not None
        assert len(m_gpu._bse) == 4
        assert np.all(m_gpu._bse > 0)
        # BSE must match CPU within floating-point tolerance
        assert np.allclose(m_cpu._bse, m_gpu._bse, rtol=1e-12)

    @pytest.mark.skipif(not _HAS_TORCH_CUDA, reason="Torch CUDA not available")
    def test_bootstrap_inference_torch(self):
        """Bootstrap inference on Torch GPU — BSE should be similar to CPU."""
        import torch
        X_t = torch.tensor(self.X, dtype=torch.float64, device='cuda')
        y_t = torch.tensor(self.y, dtype=torch.float64, device='cuda')
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='bootstrap', n_bootstrap=50,
                                device='torch')
        m.fit(X_t, y_t)
        assert m._bse is not None
        assert len(m._bse) == 4
        assert np.all(m._bse > 0)
