"""
Ridge regression with FULL GPU computation (no intermediate CPU transfers).
Core computation stays on GPU. Diagnostics computed on GPU or skipped.
"""

from typing import Optional, Union
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device


class RidgeFullGPU(BaseEstimator):
    """
    Ridge regression with FULL GPU computation.
    
    All core computation (fit, predict, score) stays on GPU.
    Only final results transferred to CPU.
    
    ⚠️  Warning: Statistical inference (std err, p-values) may be 
        computed on GPU with reduced precision or skipped.
        Use Ridge class for full statistical output.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    device : str or Device, default='auto'
        Computation device.
    
    Attributes
    ----------
    coef_ : ndarray
        Estimated coefficients (on CPU).
    intercept_ : float
        Independent term.
    rsquared_ : float
        R-squared (computed on GPU).
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.rsquared_ = None
        
        # GPU storage (kept on GPU)
        self._X_design_gpu = None
        self._y_gpu = None
        self._resid_gpu = None
        self._params_gpu = None
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit Ridge regression model.
        
        Parameters
        ----------
        X : array-like or cupy.ndarray
            Training data. If cupy array, stays on GPU.
        y : array-like or cupy.ndarray
            Target values. If cupy array, stays on GPU.
        sample_weight : array-like, optional
            Sample weights.
        
        Returns
        -------
        self : object
        """
        import cupy as cp
        
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            # Ensure data is on GPU
            if hasattr(X, 'get'):  # Already CuPy
                X_gpu = X
            else:
                X_gpu = cp.asarray(X)
            
            if hasattr(y, 'get'):  # Already CuPy
                y_gpu = y
            else:
                y_gpu = cp.asarray(y)
            
            self._fit_gpu_full(X_gpu, y_gpu, sample_weight)
        else:
            # CPU fallback
            X_arr = np.asarray(X)
            y_arr = np.asarray(y)
            self._fit_cpu(X_arr, y_arr, sample_weight)
        
        self._fitted = True
        return self
    
    def _fit_gpu_full(self, X, y, sample_weight=None):
        """Full GPU fit with all computation on GPU."""
        import cupy as cp
        
        n_samples, n_features = X.shape
        
        # Handle sample weights on GPU
        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw
        
        # Center data on GPU
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0, dtype=X.dtype)
        
        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)
        
        # Ridge closed-form: (X'X + alpha*I)^-1 X'y
        XtX = cp.matmul(X_centered.T, X_centered)
        Xty = cp.matmul(X_centered.T, y_centered)
        
        # Add regularization
        I = cp.eye(n_features, dtype=XtX.dtype)
        XtX_reg = XtX + self.alpha * I
        
        # Solve using Cholesky
        try:
            L = cp.linalg.cholesky(XtX_reg)
            tmp = cp.linalg.solve_triangular(L, Xty, lower=True)
            coef = cp.linalg.solve_triangular(L.T, tmp, lower=False)
        except Exception:
            coef = cp.linalg.solve(XtX_reg, Xty)
        
        # Build full coefficients [intercept, coef]
        if self.fit_intercept:
            intercept_gpu = y_mean - cp.dot(X_mean, coef)
            coef_full = cp.concatenate([intercept_gpu.reshape(1), coef.flatten()])
            ones_col = cp.ones((n_samples, 1), dtype=X.dtype)
            X_design = cp.concatenate([ones_col, X], axis=1)
        else:
            coef_full = coef.flatten()
            X_design = X
        
        # Compute predictions and residuals on GPU
        y_pred = cp.dot(X_design, coef_full)
        resid = y - y_pred
        
        # Compute R² on GPU
        y_mean_all = y.mean()
        ss_res = cp.sum(resid ** 2)
        ss_tot = cp.sum((y - y_mean_all) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # Store GPU arrays for later use
        self._X_design_gpu = X_design
        self._y_gpu = y
        self._resid_gpu = resid
        self._params_gpu = coef_full
        
        # Transfer only final results to CPU
        self.coef_ = cp.asnumpy(coef).flatten()
        if self.fit_intercept:
            self.intercept_ = float(cp.asnumpy(intercept_gpu))
        else:
            self.intercept_ = 0.0
        self.rsquared_ = float(cp.asnumpy(r2))
    
    def _fit_cpu(self, X, y, sample_weight=None):
        """CPU fallback."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            sqrt_sw = np.sqrt(sample_weight)
            X = X * sqrt_sw[:, np.newaxis]
            y = y * sqrt_sw
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = 0.0
        
        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)
        
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered
        
        I = np.eye(n_features)
        XtX_reg = XtX + self.alpha * I
        
        try:
            coef = np.linalg.solve(XtX_reg, Xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]
        
        coef = coef.flatten()
        
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ coef)
            self.coef_ = coef
            X_design = np.column_stack([np.ones(n_samples), X])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
            X_design = X
        
        y_pred = X_design @ self._params if hasattr(self, '_params') else X_design @ np.concatenate([[self.intercept_], self.coef_])
        resid = y - y_pred
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.rsquared_ = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    def predict(self, X):
        """
        Predict using the model.
        
        If X is CuPy array, prediction is done on GPU.
        If X is NumPy array, prediction is done on CPU.
        """
        self._check_is_fitted()
        
        if hasattr(X, 'get'):  # CuPy array
            # GPU prediction
            import cupy as cp
            coef_gpu = cp.asarray(self.coef_)
            intercept_gpu = cp