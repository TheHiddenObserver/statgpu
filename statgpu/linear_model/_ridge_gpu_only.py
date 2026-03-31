"""
Ridge regression with GPU-only computation (no intermediate transfers).
"""

from typing import Optional, Union
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device


class RidgeGPUOnly(BaseEstimator):
    """
    Ridge regression with GPU-only computation.
    All calculations stay on GPU until final result.
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
        self._X_design = None
        self._y = None
        self._resid = None
        self._scale = None
        self._nobs = None
        self._df_resid = None
        self._params = None
    
    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model."""
        self._y = np.asarray(y)
        X_arr = self._to_array(X)
        y_arr = self._to_array(y)
        
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            self._fit_gpu_only(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)
        
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
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
            self._params = np.concatenate([[self.intercept_], self.coef_])
            self._X_design = np.column_stack([np.ones(n_samples, dtype=X.dtype), X])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
            self._params = self.coef_.copy()
            self._X_design = X.copy()
        
        y_pred = self._X_design @ self._params
        self._resid = self._y - y_pred
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        
        if self._df_resid > 0:
            self._scale = np.sum(self._resid ** 2) / self._df_resid
        else:
            self._scale = np.nan
    
    def _fit_gpu_only(self, X, y, sample_weight=None):
        """Fit using GPU with ZERO intermediate transfers to CPU."""
        import cupy as cp
        import cupyx
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Ensure CuPy arrays (single transfer from CPU if needed)
        X = cp.asarray(X)
        y = cp.asarray(y)
        
        # Handle sample weights on GPU
        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cupyx.rsqrt(sample_weight)  # Use rsqrt for speed
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw
        
        # Pre-allocate arrays to avoid memory allocation overhead
        if self.fit_intercept:
            # Use cupy's mean which is optimized
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            
            # Center data in-place to save memory
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0, dtype=X.dtype)
        
        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)
        
        # Compute X'X and X'y on GPU
        # Use cupy's optimized matmul
        XtX = cp.matmul(X_centered.T, X_centered)
        Xty = cp.matmul(X_centered.T, y_centered)
        
        # Add regularization using in-place operation
        # Create diagonal matrix efficiently
        reg_diag = cp.full(n_features, self.alpha, dtype=XtX.dtype)
        XtX_reg = XtX + cp.diag(reg_diag)
        
        # Solve using Cholesky (most efficient for SPD matrices)
        try:
            # Cholesky decomposition: XtX_reg = L @ L.T
            L = cp.linalg.cholesky(XtX_reg)
            # Solve L @ tmp = Xty (forward substitution)
            tmp = cp.linalg.solve_triangular(L, Xty, lower=True)
            # Solve L.T @ coef = tmp (backward substitution)
            coef = cp.linalg.solve_triangular(L.T, tmp, lower=False)
        except Exception:
            # Fallback to standard solve
            coef = cp.linalg.solve(XtX_reg, Xty)
        
        # Compute intercept and full coefficients ON GPU
        if self.fit_intercept:
            # intercept = y_mean - X_mean @ coef
            intercept_gpu = y_mean - cp.dot(X_mean, coef)
            # Full coefficients [intercept, coef]
            coef_full = cp.concatenate([intercept_gpu.reshape(1), coef.flatten()])
            # Design matrix with intercept column
            ones_col = cp.ones((n_samples, 1), dtype=X.dtype)
            X_design = cp.concatenate([ones_col, X], axis=1)
        else:
            coef_full = coef.flatten()
            X_design = X
        
        # Compute predictions and residuals ON GPU
        y_pred = cp.dot(X_design, coef_full)
        resid = y - y_pred
        
        # Compute scale ON GPU
        df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        if df_resid > 0:
            # Use cupy's sum
            scale = cp.dot(resid, resid) / df_resid
        else:
            scale = cp.nan
        
        # SINGLE TRANSFER TO CPU - only at the very end
        coef_full_np = cp.asnumpy(coef_full)  # More efficient than .get()
        
        # Store results
        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np
        
        # These are computed on CPU for diagnostics
        self._X_design = cp.asnumpy(X_design)
        self._resid = cp.asnumpy(resid)
        self._df_resid = df_resid
        self._scale = float(cp.asnumpy(scale)) if not cp.isnan(scale) else np.nan
    
    def predict(self, X):
        """Predict."""
        self._check_is_fitted()
        X = self._to_array(X, Device.CPU)
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """R² score."""
        y_pred = self.predict(X)
        y = np.asarray(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    @property
    def rsquared(self):
        """R-squared.