"""
Linear regression with full statistical inference and GPU support.
"""

from typing import Optional, Union
import numpy as np
from scipy import stats

from .._base import BaseEstimator
from .._config import Device


class LinearRegression(BaseEstimator):
    """
    Ordinary least squares linear regression with GPU acceleration
    and full statistical inference.
    
    Similar to R's lm() and statsmodels OLS.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    n_jobs : int, optional
        Number of parallel jobs for CPU computation. -1 means all CPUs.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term.
    
    Examples
    --------
    >>> from statgpu.linear_model import LinearRegression
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X @ np.array([1, 2, 3, 4, 5]) + 10
    >>> model = LinearRegression(device='cuda')
    >>> model.fit(X, y)
    >>> print(model.summary())
    """
    
    def __init__(
        self,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
        # Internal storage for inference
        self._X_design = None
        self._y = None
        self._resid = None
        self._scale = None
        self._nobs = None
        self._df_resid = None
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights.
        
        Returns
        -------
        self : object
        """
        # Store original y for inference
        self._y = np.asarray(y)
        
        X_arr = self._to_array(X)
        y_arr = self._to_array(y)
        
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)
        
        # Compute inference statistics
        self._compute_inference()
        
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Handle sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            sqrt_sw = np.sqrt(sample_weight)
            X = X * sqrt_sw[:, np.newaxis]
            y = y * sqrt_sw
        
        # Store design matrix
        if self.fit_intercept:
            self._X_design = np.column_stack([np.ones(n_samples, dtype=X.dtype), X])
        else:
            self._X_design = X.copy()
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Solve using least squares
        coef, _, _, _ = np.linalg.lstsq(self._X_design, y, rcond=None)
        
        # Store parameters
        if self.fit_intercept:
            self.intercept_ = float(coef[0])
            self.coef_ = coef[1:].flatten()
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef.flatten()
            self._params = self.coef_.copy()
        
        # Compute residuals
        y_pred = self._X_design @ self._params
        self._resid = self._y - y_pred
        
        # Degrees of freedom
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        
        # Estimate of error variance (sigma^2)
        if self._df_resid > 0:
            self._scale = np.sum(self._resid ** 2) / self._df_resid
        else:
            self._scale = np.nan
    
    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU."""
        import cupy as cp
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Handle sample weights
        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw
        
        # Build design matrix on GPU
        if self.fit_intercept:
            X_design = cp.column_stack([cp.ones(n_samples, dtype=X.dtype), X])
        else:
            X_design = X
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Solve
        coef, _, _, _ = cp.linalg.lstsq(X_design, y, rcond=None)
        
        # Transfer back to CPU
        coef_np = coef.get().flatten()
        X_design_np = X_design.get()
        
        # Store results
        if self.fit_intercept:
            self.intercept_ = float(coef_np[0])
            self.coef_ = coef_np[1:]
            self._params = coef_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_np
            self._params = coef_np
        
        self._X_design = X_design_np
        
        # Compute residuals on CPU
        y_pred = self._X_design @ self._params
        self._resid = self._y - y_pred
        
        # Degrees of freedom
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        
        # Scale
        if self._df_resid > 0:
            self._scale = np.sum(self._resid ** 2) / self._df_resid
        else:
            self._scale = np.nan
    
    def _compute_inference(self):
        """Compute standard errors, t-stats, p-values."""
        if self._X_design is None or self._scale is None:
            return
        
        # (X'X)^-1
        try:
            XtX_inv = np.linalg.inv(self._X_design.T @ self._X_design)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(self._X_design.T @ self._X_design)
        
        # Standard errors
        self._bse = np.sqrt(self._scale * np.diag(XtX_inv))
        
        # t-statistics
        self._tvalues = self._params / self._bse
        
        # p-values (two-tailed)
        self._pvalues = 2 * (1 - stats.t.cdf(np.abs(self._tvalues), self._df_resid))
        
        # Confidence intervals (95%)
        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, self._df_resid)
        self._conf_int = np.column_stack([
            self._params - t_crit * self._bse,
            self._params + t_crit * self._bse
        ])
    
    @property
    def rsquared(self):
        """R-squared."""
        if self._y is None or self._resid is None:
            return None
        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    @property
    def rsquared_adj(self):
        """Adjusted R-squared."""
        if self._nobs is None:
            return None
        r2 = self.rsquared
        k = len(self.coef_)
        return 1 - (1 - r2) * (self._nobs - 1) / self._df_resid
    
    @property
    def fvalue(self):
        """F-statistic."""
        if self._y is None