"""Tests for linear models."""

import numpy as np
import pytest

from statgpu.linear_model import LinearRegression
from statgpu._config import set_device, Device


class TestLinearRegression:
    """Test LinearRegression class."""
    
    def test_basic_fit_cpu(self):
        """Test basic fitting on CPU."""
        set_device('cpu')
        
        # Generate simple linear data
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # y = 2*x
        
        model = LinearRegression(device='cpu')
        model.fit(X, y)
        
        # Check coefficients are close to expected
        assert abs(model.coef_[0] - 2.0) < 0.01
        assert abs(model.intercept_ - 0.0) < 0.01
    
    def test_fit_with_intercept(self):
        """Test fitting with intercept."""
        set_device('cpu')
        
        X = np.random.randn(100, 3)
        true_coef = np.array([1.5, -2.0, 3.0])
        true_intercept = 5.0
        y = X @ true_coef + true_intercept
        
        model = LinearRegression(fit_intercept=True, device='cpu')
        model.fit(X, y)
        
        # Check coefficients
        assert np.allclose(model.coef_, true_coef, atol=0.01)
        assert abs(model.intercept_ - true_intercept) < 0.01
    
    def test_fit_without_intercept(self):
        """Test fitting without intercept."""
        set_device('cpu')
        
        X = np.random.randn(100, 3)
        true_coef = np.array([1.5, -2.0, 3.0])
        y = X @ true_coef  # No intercept
        
        model = LinearRegression(fit_intercept=False, device='cpu')
        model.fit(X, y)
        
        assert np.allclose(model.coef_, true_coef, atol=0.01)
        assert model.intercept_ == 0.0
    
    def test_predict(self):
        """Test prediction."""
        set_device('cpu')
        
        X = np.random.randn(50, 3)
        true_coef = np.array([1.0, 2.0, 3.0])
        y = X @ true_coef + 10
        
        model = LinearRegression(device='cpu')
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == (50,)
        assert np.allclose(y_pred, y, atol=0.01)
    
    def test_score(self):
        """Test R^2 score."""
        set_device('cpu')
        
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + 5
        
        model = LinearRegression(device='cpu')
        model.fit(X, y)
        
        score = model.score(X, y)
        assert score > 0.99  # Should be very close to 1
    
    def test_multitarget(self):
        """Test multi-target regression."""
        set_device('cpu')
        
        X = np.random.randn(50, 3)
        y = np.column_stack([
            X @ np.array([1, 2, 3]),
            X @ np.array([4, 5, 6])
        ])
        
        model = LinearRegression(device='cpu')
        model.fit(X, y)
        
        assert model.coef_.shape == (2, 3)
        assert model.intercept_.shape == (2,)

        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

        score = model.score(X, y)
        assert score > 0.99

    def test_multitarget_inference_shapes(self):
        """Test multi-target inference statistics are computed with target axis."""
        set_device("cpu")

        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 4))
        y = np.column_stack(
            [
                X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(scale=0.1, size=200),
                X @ np.array([0.3, 1.2, -1.0, 2.4]) + rng.normal(scale=0.2, size=200),
            ]
        )

        model = LinearRegression(device="cpu", compute_inference=True, cov_type="hc1")
        model.fit(X, y)

        assert model._bse.shape == (5, 2)
        assert model._tvalues.shape == (5, 2)
        assert model._pvalues.shape == (5, 2)
        assert model._conf_int.shape == (5, 2, 2)

    def test_multitarget_summary_not_supported(self):
        """summary() should be unavailable for multi-target outputs."""
        set_device("cpu")
        X = np.random.randn(50, 3)
        y = np.column_stack([X @ np.array([1.0, 2.0, 3.0]), X @ np.array([2.0, -1.0, 0.5])])

        model = LinearRegression(device="cpu", compute_inference=True)
        model.fit(X, y)

        with pytest.raises(RuntimeError):
            model.summary()
    
    def test_not_fitted_error(self):
        """Test error when predicting before fitting."""
        model = LinearRegression(device='cpu')
        
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1, 2]]))

    def test_invalid_hac_maxlags_raises(self):
        with pytest.raises(ValueError):
            LinearRegression(device="cpu", cov_type="hac", hac_maxlags=-1)

    @pytest.mark.parametrize("cov_type", ["hc2", "hc3", "hac"])
    def test_extended_cov_types_cpu(self, cov_type):
        """Extended robust covariance types should run and produce finite inference."""
        set_device("cpu")
        rng = np.random.default_rng(123)
        X = rng.normal(size=(600, 8))
        beta = rng.normal(size=8)
        noise_scale = 0.2 + 0.8 * np.abs(X[:, 0])
        y = X @ beta + 2.0 + rng.normal(scale=noise_scale, size=600)

        kwargs = {"hac_maxlags": 4} if cov_type == "hac" else {}
        model = LinearRegression(device="cpu", cov_type=cov_type, compute_inference=True, **kwargs)
        model.fit(X, y)

        assert model._bse is not None
        assert np.all(np.isfinite(model._bse))
        assert np.all(model._bse > 0)
        assert np.all(np.isfinite(model._pvalues))
        assert np.all((model._pvalues >= 0) & (model._pvalues <= 1))


class TestGPU:
    """GPU-specific tests (only run if CUDA available)."""
    
    @pytest.mark.skipif(
        not LinearRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_fit(self):
        """Test fitting on GPU."""
        set_device('cuda')
        
        X = np.random.randn(100, 5).astype(np.float32)
        y = X @ np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        model = LinearRegression(device='cuda')
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert len(model.coef_) == 5
    
    @pytest.mark.skipif(
        not LinearRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_matches_cpu(self):
        """Test GPU and CPU produce same results."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float64)
        y = X @ np.array([1, 2, 3, 4, 5], dtype=np.float64)
        
        # CPU model
        model_cpu = LinearRegression(device='cpu')
        model_cpu.fit(X, y)
        
        # GPU model
        model_gpu = LinearRegression(device='cuda')
        model_gpu.fit(X, y)
        
        # Compare coefficients
        assert np.allclose(model_cpu.coef_, model_gpu.coef_, rtol=1e-4)
        assert np.allclose(model_cpu.intercept_, model_gpu.intercept_, rtol=1e-4)
