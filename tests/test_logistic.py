"""Tests for logistic regression."""

import numpy as np
import pytest

from statgpu.linear_model import LogisticRegression
from statgpu._config import set_device, Device


class TestLogisticRegression:
    """Test LogisticRegression class."""
    
    def test_basic_fit_cpu(self):
        """Test basic fitting on CPU."""
        set_device('cpu')
        
        # Generate simple binary classification data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        # Check that coefficients are reasonable
        assert model.coef_ is not None
        assert len(model.coef_) == 2
        assert model.intercept_ is not None
        assert model.n_iter_ <= 100
    
    def test_fit_with_intercept(self):
        """Test fitting with intercept."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(200, 3)
        # True model with intercept
        true_coef = np.array([1.5, -2.0, 3.0])
        true_intercept = 0.5
        z = X @ true_coef + true_intercept
        y = (z > 0).astype(int)
        
        model = LogisticRegression(fit_intercept=True, device='cpu', max_iter=100)
        model.fit(X, y)
        
        # Check coefficients have correct signs
        assert np.sign(model.coef_[0]) == np.sign(true_coef[0])
        assert np.sign(model.coef_[1]) == np.sign(true_coef[1])
        assert np.sign(model.coef_[2]) == np.sign(true_coef[2])
    
    def test_fit_without_intercept(self):
        """Test fitting without intercept."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(fit_intercept=False, device='cpu', max_iter=100)
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert model.intercept_ == 0.0
    
    def test_predict_proba(self):
        """Test probability predictions."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)
    
    def test_predict(self):
        """Test class predictions.""" 
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == (50,)
        assert np.all(np.isin(y_pred, [0, 1]))
    
    def test_score(self):
        """Test accuracy score."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        score = model.score(X, y)
        assert 0 <= score <= 1
        assert score > 0.7  # Should be reasonably accurate
    
    def test_regularization(self):
        """Test L2 regularization."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Strong regularization
        model_strong = LogisticRegression(C=0.01, device='cpu', max_iter=100)
        model_strong.fit(X, y)
        
        # Weak regularization
        model_weak = LogisticRegression(C=1000, device='cpu', max_iter=100)
        model_weak.fit(X, y)
        
        # Strong regularization should produce smaller coefficients
        assert np.linalg.norm(model_strong.coef_) < np.linalg.norm(model_weak.coef_)
    
    def test_stats(self):
        """Test statistical outputs."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        # Check that stats are computed
        assert model.loglikelihood is not None
        assert model.loglikelihood_null is not None
        assert model.aic is not None
        assert model.bic is not None
        assert model.pseudo_rsquared is not None
        assert model.accuracy is not None
        assert model.precision is not None
        assert model.recall is not None
        assert model.f1 is not None
        
        # Check ranges
        assert model.pseudo_rsquared >= 0 and model.pseudo_rsquared <= 1
        assert model.aic < model.bic  # BIC penalizes more
    
    def test_summary(self):
        """Test summary output."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        # Just make sure it doesn't raise
        model.summary()
    
    def test_not_fitted_error(self):
        """Test error when predicting before fitting."""
        model = LogisticRegression(device='cpu')
        
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1, 2]]))
        
        with pytest.raises(RuntimeError):
            model.predict_proba(np.array([[1, 2]]))


class TestGPU:
    """GPU-specific tests (only run if CUDA available)."""
    
    @pytest.mark.skipif(
        not LogisticRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_fit(self):
        """Test fitting on GPU."""
        set_device('cuda')
        
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cuda', max_iter=100)
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert len(model.coef_) == 5
    
    @pytest.mark.skipif(
        not LogisticRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_matches_cpu(self):
        """Test GPU and CPU produce same results."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float64)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # CPU model
        model_cpu = LogisticRegression(device='cpu', max_iter=100)
        model_cpu.fit(X, y)
        
        # GPU model
        model_gpu = LogisticRegression(device='cuda', max_iter=100)
        model_gpu.fit(X, y)
        
        # Compare coefficients
        assert np.allclose(model_cpu.coef_, model_gpu.coef_, rtol=1e-3)
        assert np.allclose(model_cpu.intercept_, model_gpu.intercept_, rtol=1e-3)
