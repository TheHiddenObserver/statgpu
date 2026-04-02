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
    and full statistical inference (R/statsmodels style).
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term.
    """
    
    def __init__(
        self,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.fit_intercept = fit_intercept
        self.compute_inference = compute_inference
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.coef_ = None
        self.intercept_ = None
        
        # Internal storage for inference
        self._X_design = None
        self._y = None
        self._resid = None
        self._scale = None
        self._nobs = None
        self._df_resid = None
        self._params = None
        self._bse = None
        self._tvalues = None
        self._pvalues = None
        self._conf_int = None

    def _cleanup_cuda_memory(self):
        """Best-effort CuPy memory pool cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    
    def fit(self, X, y, sample_weight=None):
        """Fit linear model."""
        # Store y (may be CuPy array, convert later for CPU)
        self._y = y
        
        X_arr = self._to_array(X)
        y_arr = self._to_array(y)
        
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)
        
        # Convert y to numpy for diagnostics if needed
        if hasattr(self._y, 'get'):
            self._y = self._y.get()
        else:
            self._y = np.asarray(self._y)

        # GPU path already computes inference on-device in _fit_gpu().
        # If compute_inference=False, skip all inference work.
        if self.compute_inference and device != Device.CUDA:
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
            self._X_design = np.column_stack([np.ones(n_samples, dtype=X.dtype), X])
        else:
            self._X_design = X.copy()
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        coef, _, _, _ = np.linalg.lstsq(self._X_design, y, rcond=None)
        coef = coef.flatten()  # Ensure 1D
        
        if self.fit_intercept:
            self.intercept_ = float(coef[0])
            self.coef_ = coef[1:]
            self._params = coef.copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = coef.copy()
            self._params = coef.copy()
        
        y_pred = self._X_design @ self._params
        self._resid = self._y - y_pred
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        
        if self._df_resid > 0:
            self._scale = np.sum(self._resid ** 2) / self._df_resid
        else:
            self._scale = np.nan
    
    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU with FULL GPU computation (including inference)."""
        import cupy as cp
        from .._gpu_utils import compute_inference_gpu, compute_r2_gpu, compute_aic_bic_gpu, compute_f_stat_gpu
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Ensure CuPy arrays
        X = cp.asarray(X)
        y = cp.asarray(y)
        
        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw
        
        if self.fit_intercept:
            X_design = cp.column_stack([cp.ones(n_samples, dtype=X.dtype), X])
        else:
            X_design = X
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Use normal equations: (X'X)^-1 X'y
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        
        try:
            # Cholesky decomposition
            L = cp.linalg.cholesky(XtX)
            tmp = cp.linalg.solve_triangular(L, Xty, lower=True)
            coef = cp.linalg.solve_triangular(L.T, tmp, lower=False)
        except Exception:
            coef = cp.linalg.solve(XtX, Xty)
        
        # Compute predictions and residuals on GPU
        y_pred = X_design @ coef
        resid = y - y_pred
        
        # Compute scale on GPU
        df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        if df_resid > 0:
            scale = cp.sum(resid ** 2) / df_resid
        else:
            scale = cp.nan
        
        coef_flat = coef.flatten()

        # Compute inference-related statistics only when requested.
        if self.compute_inference:
            self._bse_gpu, self._tvalues_gpu, self._pvalues_gpu, self._conf_int_gpu = \
                compute_inference_gpu(X_design, resid, scale, df_resid, coef_flat)

            # R-squared on GPU
            self._rsquared_gpu = compute_r2_gpu(y, resid)

            # AIC/BIC on GPU
            k = n_features + (1 if self.fit_intercept else 0)
            scale_mle = cp.sum(resid ** 2) / n_samples
            self._aic_gpu, self._bic_gpu = compute_aic_bic_gpu(n_samples, k, scale_mle)

            # F-statistic on GPU
            self._fvalue_gpu, self._f_pvalue = compute_f_stat_gpu(y, resid, X_design, df_resid)

        # Single transfer to CPU at the end
        coef_np = coef.get().flatten()
        resid_np = resid.get().flatten()
        scale_float = float(scale.get()) if not cp.isnan(scale) else np.nan
        X_design_np = X_design.get()
        
        if self.compute_inference:
            # Transfer inference results
            self._bse = self._bse_gpu.get()
            self._tvalues = self._tvalues_gpu.get()
            self._pvalues = self._pvalues_gpu.get()
            self._conf_int = self._conf_int_gpu.get()
        
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
        self._resid = resid_np
        self._df_resid = df_resid
        self._scale = scale_float

        # Release large temporary GPU tensors early.
        try:
            del X_design
        except Exception:
            pass
        try:
            del resid
        except Exception:
            pass
        try:
            del XtX
        except Exception:
            pass
        try:
            del Xty
        except Exception:
            pass
        try:
            del coef
        except Exception:
            pass
        self._cleanup_cuda_memory()
    
    def _compute_inference(self):
        """Compute standard errors, t-stats, p-values."""
        if self._X_design is None or self._scale is None or np.isnan(self._scale):
            return
        
        try:
            XtX_inv = np.linalg.inv(self._X_design.T @ self._X_design)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(self._X_design.T @ self._X_design)
        
        self._bse = np.sqrt(self._scale * np.diag(XtX_inv))
        self._tvalues = self._params / self._bse
        self._pvalues = 2 * (1 - stats.t.cdf(np.abs(self._tvalues), self._df_resid))
        
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
        if self._y is None or self._resid is None:
            return None
        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        ss_reg = ss_tot - ss_res
        k = len(self.coef_)
        if k == 0 or ss_res <= 0:
            return np.inf
        return (ss_reg / k) / (ss_res / self._df_resid)
    
    @property
    def f_pvalue(self):
        """p-value for F-statistic."""
        fv = self.fvalue
        if fv is None or fv == np.inf:
            return 1.0
        k = len(self.coef_)
        return 1 - stats.f.cdf(fv, k, self._df_resid)
    
    @property
    def aic(self):
        """Akaike Information Criterion."""
        if self._nobs is None or np.isnan(self._scale):
            return None
        # AIC = -2 * log-likelihood + 2 * k
        return -2 * self.llf + 2 * len(self._params)
    
    @property
    def bic(self):
        """Bayesian Information Criterion."""
        if self._nobs is None or np.isnan(self._scale):
            return None
        n = self._nobs
        k = len(self._params)
        # BIC = -2 * log-likelihood + k * log(n)
        return -2 * self.llf + k * np.log(n)
    
    @property
    def llf(self):
        """Log-likelihood (matches statsmodels/R)."""
        if self._nobs is None or self._resid is None:
            return None
        n = self._nobs
        # Use MLE estimate of sigma^2 = RSS/n (not RSS/df_resid)
        sigma2_mle = np.sum(self._resid ** 2) / n
        # LL = -n/2 * log(2*pi*sigma2_mle) - n/2
        return -n/2 * np.log(2 * np.pi * sigma2_mle) - n/2
    
    def summary(self):
        """Print summary table similar to R's summary(lm())."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if not self.compute_inference:
            raise RuntimeError(
                "compute_inference=False: summary/inference statistics are not available. "
                "Re-fit with compute_inference=True (default)."
            )
        
        # Build feature names
        if self.fit_intercept:
            feature_names = ['(Intercept)'] + [f'x{i+1}' for i in range(len(self.coef_))]
        else:
            feature_names = [f'x{i+1}' for i in range(len(self.coef_))]
        
        print("=" * 80)
        print("                            Linear Regression Results")
        print("=" * 80)
        print(f"No. Observations:           {self._nobs:>15}")
        print(f"Degrees of Freedom:         {self._df_resid:>15}")
        print(f"R-squared:                  {self.rsquared:>15.4f}")
        print(f"Adj. R-squared:             {self.rsquared_adj:>15.4f}")
        print(f"F-statistic:                {self.fvalue:>15.4f}")
        print(f"Prob (F-statistic):         {self.f_pvalue:>15.4e}")
        print(f"Log-Likelihood:             {self.llf:>15.4f}")
        print(f"AIC:                        {self.aic:>15.4f}")
        print(f"BIC:                        {self.bic:>15.4f}")
        print("-" * 80)
        print(f"{'':<15} {'coef':>12} {'std err':>12} {'t':>10} {'P>|t|':>10} {'[0.025':>12} {'0.975]':>12}")
        print("-" * 80)
        
        for i, name in enumerate(feature_names):
            print(f"{name:<15} {self._params[i]:>12.4f} {self._bse[i]:>12.4f} "
                  f"{self._tvalues[i]:>10.3f} {self._pvalues[i]:>10.4f} "
                  f"{self._conf_int[i, 0]:>12.4f} {self._conf_int[i, 1]:>12.4f}")
        
        print("=" * 80)
    
    def predict(self, X):
        """Predict using the linear model."""
        self._check_is_fitted()
        X = self._to_array(X, Device.CPU)
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """Return R^2 score."""
        y_pred = self.predict(X)
        y = np.asarray(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
