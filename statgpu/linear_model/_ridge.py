"""
Optimized Ridge regression with GPU support.
"""

from typing import Optional, Union
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device


class Ridge(BaseEstimator):
    """
    Ridge regression (L2 regularization) with optimized GPU acceleration.
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
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)
        
        self._compute_inference()
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
    
    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU (optimized)."""
        import cupy as cp
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Ensure CuPy arrays
        X = cp.asarray(X)
        y = cp.asarray(y)
        
        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, np.newaxis]
            y = y * sqrt_sw
        
        if self.fit_intercept:
            X_mean = cp.mean(X, axis=0)
            y_mean = cp.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0)
        
        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)
        
        # Ridge closed-form
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered
        
        I = cp.eye(n_features)
        XtX_reg = XtX + self.alpha * I
        
        try:
            # Cholesky for better performance
            L = cp.linalg.cholesky(XtX_reg)
            tmp = cp.linalg.solve_triangular(L, Xty, lower=True)
            coef = cp.linalg.solve_triangular(L.T, tmp, lower=False)
        except Exception:
            coef = cp.linalg.solve(XtX_reg, Xty)
        
        # Keep on GPU for residuals
        if self.fit_intercept:
            X_design = cp.column_stack([cp.ones(n_samples, dtype=X.dtype), X])
            coef_full = cp.concatenate([y_mean - X_mean @ coef, coef.flatten()])
        else:
            X_design = X
            coef_full = coef.flatten()
        
        y_pred = X_design @ coef_full
        resid = y - y_pred
        
        df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        if df_resid > 0:
            scale = cp.sum(resid ** 2) / df_resid
        else:
            scale = cp.nan
        
        # Single transfer
        coef_full_np = coef_full.get()
        resid_np = resid.get()
        scale_float = float(scale.get()) if not cp.isnan(scale) else np.nan
        X_design_np = X_design.get()
        
        # Store
        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np
        
        self._X_design = X_design_np
        self._resid = resid_np
        self._df_resid = df_resid
        self._scale = scale_float
    
    def _compute_inference(self):
        """Compute standard errors."""
        if self._X_design is None or self._scale is None or np.isnan(self._scale):
            return
        
        X = self._X_design
        try:
            XtX = X.T @ X
            I = np.eye(XtX.shape[0])
            if self.fit_intercept:
                I[0, 0] = 0
            XtX_inv = np.linalg.inv(XtX + self.alpha * I)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)
        
        self._bse = np.sqrt(self._scale * np.diag(XtX_inv))
        self._tvalues = self._params / self._bse
        self._pvalues = 2 * (1 - stats.t.cdf(np.abs(self._tvalues), self._df_resid))
        
        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, self._df_resid)
        self._conf_int = np.column_stack([
            self._params - t_crit * self._bse,
            self._params + t_crit * self._bse
        ])
    
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
        """R-squared."""
        if self._y is None or self._resid is None:
            return None
        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
