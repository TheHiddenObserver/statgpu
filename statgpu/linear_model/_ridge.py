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
    Ridge regression (L2 regularization) with optimized GPU acceleration
    and statistical inference (R/statsmodels style).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    n_jobs : int or None, default=None
        Number of parallel jobs.
    gpu_memory_cleanup : bool, default=False
        Whether to free CuPy memory pool after fitting.
    compute_inference : bool, default=True
        Whether to compute standard errors, t-stats, p-values and CI.
    cov_type : str, default='nonrobust'
        Covariance estimator for inference. One of:
        ``'nonrobust'`` (classical), ``'hc0'`` (White HC0), ``'hc1'`` (HC1),
        ``'hc2'`` (leverage-adjusted HC2), ``'hc3'`` (jackknife-style HC3),
        or ``'hac'`` (Newey-West HAC with Bartlett kernel).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        gpu_memory_cleanup: bool = False,
        compute_inference: bool = True,
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.compute_inference = compute_inference
        self.cov_type = cov_type.lower()
        if self.cov_type not in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"):
            raise ValueError(
                "cov_type must be one of: 'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac'"
            )
        if hac_maxlags is not None and int(hac_maxlags) < 0:
            raise ValueError("hac_maxlags must be a non-negative integer or None")
        self.hac_maxlags = None if hac_maxlags is None else int(hac_maxlags)
        self.coef_ = None
        self.intercept_ = None
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

    def _resolve_hac_maxlags(self, n_obs: int) -> int:
        """Resolve HAC lag count with a Newey-West style default rule."""
        if n_obs <= 1:
            return 0
        if self.hac_maxlags is None:
            maxlags = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
        else:
            maxlags = int(self.hac_maxlags)
        return max(0, min(maxlags, n_obs - 1))

    def _hac_meat_numpy(self, scores: np.ndarray) -> np.ndarray:
        """Bartlett-kernel HAC meat from per-observation score matrix."""
        n_obs = int(scores.shape[0])
        meat = scores.T @ scores
        maxlags = self._resolve_hac_maxlags(n_obs)
        if maxlags == 0:
            return meat
        for lag in range(1, maxlags + 1):
            weight = 1.0 - (lag / (maxlags + 1.0))
            gamma = scores[lag:].T @ scores[:-lag]
            meat = meat + weight * (gamma + gamma.T)
        return meat

    def _hac_meat_cupy(self, scores):
        """CuPy Bartlett-kernel HAC meat from per-observation score matrix."""
        import cupy as cp

        n_obs = int(scores.shape[0])
        meat = scores.T @ scores
        maxlags = self._resolve_hac_maxlags(n_obs)
        if maxlags == 0:
            return meat
        for lag in range(1, maxlags + 1):
            weight = 1.0 - (lag / (maxlags + 1.0))
            gamma = scores[lag:].T @ scores[:-lag]
            meat = meat + weight * (gamma + gamma.T)
        return meat

    def _robust_covariance_numpy(self, X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
        """Compute robust/HAC covariance matrix for Ridge score equations."""
        n, k = X.shape
        e = np.asarray(resid, dtype=float).reshape(-1)

        if self.cov_type == "hac":
            scores = X * e[:, np.newaxis]
            meat = self._hac_meat_numpy(scores)
            return XtX_inv @ meat @ XtX_inv

        if self.cov_type in ("hc2", "hc3"):
            leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
            leverage = np.clip(leverage, 0.0, 1.0 - 1e-12)
            if self.cov_type == "hc2":
                e2 = (e ** 2) / (1.0 - leverage)
            else:
                e2 = (e ** 2) / ((1.0 - leverage) ** 2)
        else:
            e2 = e ** 2

        Xw = X * e2[:, np.newaxis]
        meat = X.T @ Xw
        cov_params = XtX_inv @ meat @ XtX_inv
        if self.cov_type == "hc1" and n > k:
            cov_params *= n / (n - k)
        return cov_params

    def _robust_covariance_cupy(self, X, resid, XtX_inv):
        """Compute robust/HAC covariance matrix for Ridge score equations on GPU."""
        import cupy as cp

        n, k = X.shape
        e = resid.reshape(-1)

        if self.cov_type == "hac":
            scores = X * e[:, cp.newaxis]
            meat = self._hac_meat_cupy(scores)
            return XtX_inv @ meat @ XtX_inv

        if self.cov_type in ("hc2", "hc3"):
            leverage = cp.einsum("ij,jk,ik->i", X, XtX_inv, X)
            leverage = cp.clip(leverage, 0.0, 1.0 - 1e-12)
            if self.cov_type == "hc2":
                e2 = cp.square(e) / (1.0 - leverage)
            else:
                e2 = cp.square(e) / cp.square(1.0 - leverage)
        else:
            e2 = cp.square(e)

        Xw = X * e2[:, cp.newaxis]
        meat = X.T @ Xw
        cov_params = XtX_inv @ meat @ XtX_inv
        if self.cov_type == "hc1" and n > k:
            cov_params = cov_params * (n / (n - k))
        return cov_params
    
    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model."""
        # Store y (may be CuPy array, convert later)
        self._y = y
        X_arr = self._to_array(X)
        y_arr = self._to_array(y)
        
        device = self._get_compute_device()
        
        if device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)
        
        # Now convert y to numpy for diagnostics
        if hasattr(self._y, 'get'):
            self._y = self._y.get()
        else:
            self._y = np.asarray(self._y)

        # GPU path already computes inference on-device in _fit_gpu().
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
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
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
        
        # Compute ALL statistics on GPU
        from .._gpu_utils import compute_inference_gpu, compute_r2_gpu, compute_aic_bic_gpu, compute_f_stat_gpu
        from .._gpu_utils import norm_two_tail_pvalues_gpu, norm_crit_gpu_two_tail

        if self.compute_inference:
            if self.cov_type == "nonrobust":
                self._bse_gpu, self._tvalues_gpu, self._pvalues_gpu, self._conf_int_gpu = \
                    compute_inference_gpu(X_design, resid, scale, df_resid, coef_full)
            else:
                XtX_cov = X_design.T @ X_design
                # Apply ridge penalty excluding the intercept column
                k_design = X_design.shape[1]
                penalty_diag = cp.ones(k_design, dtype=cp.float64) * self.alpha
                if self.fit_intercept:
                    penalty_diag[0] = 0.0  # no penalty on the intercept term
                XtX_pen = XtX_cov + cp.diag(penalty_diag)
                try:
                    XtX_inv = cp.linalg.inv(XtX_pen)
                except Exception:
                    XtX_inv = cp.linalg.pinv(XtX_pen)
                cov_params = self._robust_covariance_cupy(X_design, resid, XtX_inv)
                self._bse_gpu = cp.sqrt(cp.maximum(cp.diag(cov_params), 0.0))
                self._tvalues_gpu = coef_full / (self._bse_gpu + 1e-30)
                self._pvalues_gpu = norm_two_tail_pvalues_gpu(cp.abs(self._tvalues_gpu))
                z_crit = norm_crit_gpu_two_tail(0.05)
                self._conf_int_gpu = cp.stack([
                    coef_full - z_crit * self._bse_gpu,
                    coef_full + z_crit * self._bse_gpu,
                ], axis=1)

            self._rsquared_gpu = compute_r2_gpu(y, resid)

            k = n_features + (1 if self.fit_intercept else 0)
            scale_mle = cp.sum(resid ** 2) / n_samples
            self._aic_gpu, self._bic_gpu = compute_aic_bic_gpu(n_samples, k, scale_mle)

            self._fvalue_gpu, self._f_pvalue = compute_f_stat_gpu(y, resid, X_design, df_resid)
        
        # Single transfer to CPU at the end
        coef_full_np = coef_full.get()
        resid_np = resid.get()
        scale_float = float(scale.get()) if not cp.isnan(scale) else np.nan
        X_design_np = X_design.get()

        # Transfer inference results
        if self.compute_inference:
            self._bse = self._bse_gpu.get()
            self._tvalues = self._tvalues_gpu.get()
            self._pvalues = self._pvalues_gpu.get()
            self._conf_int = self._conf_int_gpu.get()
        
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
            del XtX_reg
        except Exception:
            pass
        self._cleanup_cuda_memory()
    
    def _compute_inference(self):
        """Compute standard errors, t-stats, p-values, and CIs."""
        if self._X_design is None or self._scale is None or np.isnan(self._scale):
            return

        X = self._X_design
        n = X.shape[0]
        k = X.shape[1]

        # Build the penalized bread (X'X + alpha·P)^{-1} where the penalty
        # matrix P excludes the intercept column (if fit_intercept is True).
        # This ensures SE/t/p are consistent with the ridge fit rather than OLS.
        XtX = X.T @ X
        penalty_diag = np.ones(k) * self.alpha
        if self.fit_intercept:
            penalty_diag[0] = 0.0  # no penalty on the intercept term
        XtX_pen = XtX + np.diag(penalty_diag)
        try:
            XtX_inv = np.linalg.inv(XtX_pen)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX_pen)

        alpha = 0.05

        if self.cov_type == "nonrobust":
            cov_params = self._scale * XtX_inv
            self._bse = np.sqrt(np.diag(cov_params))
            self._tvalues = self._params / (self._bse + 1e-30)
            self._pvalues = 2 * (1 - stats.t.cdf(np.abs(self._tvalues), self._df_resid))
            t_crit = stats.t.ppf(1 - alpha / 2, self._df_resid)
            self._conf_int = np.column_stack([
                self._params - t_crit * self._bse,
                self._params + t_crit * self._bse,
            ])
        else:
            cov_params = self._robust_covariance_numpy(X, self._resid, XtX_inv)
            self._bse = np.sqrt(np.maximum(np.diag(cov_params), 0.0))
            self._tvalues = self._params / (self._bse + 1e-30)
            # Robust path uses large-sample normal approximation.
            self._pvalues = 2 * (1 - stats.norm.cdf(np.abs(self._tvalues)))
            z_crit = stats.norm.ppf(1 - alpha / 2)
            self._conf_int = np.column_stack([
                self._params - z_crit * self._bse,
                self._params + z_crit * self._bse,
            ])

    def predict(self, X):
        """Predict."""
        self._check_is_fitted()
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            X_gpu = cp.asarray(self._to_array(X, Device.CUDA))
            coef_gpu = cp.asarray(self.coef_)
            intercept_gpu = cp.asarray(self.intercept_, dtype=coef_gpu.dtype)
            return X_gpu @ coef_gpu + intercept_gpu
        X = self._to_array(X, Device.CPU)
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R² score."""
        y_pred = self._to_numpy(self.predict(X))
        y = self._to_numpy(y)
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

    @property
    def rsquared_adj(self):
        """Adjusted R-squared."""
        if self._nobs is None or self._X_design is None:
            return None
        r2 = self.rsquared
        if r2 is None:
            return None
        k = int(self._X_design.shape[1] - (1 if self.fit_intercept else 0))
        return 1 - (1 - r2) * (self._nobs - 1) / self._df_resid

    @property
    def fvalue(self):
        """F-statistic."""
        if self._y is None or self._resid is None or self._X_design is None:
            return None
        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        ss_reg = ss_tot - ss_res
        k = int(self._X_design.shape[1] - (1 if self.fit_intercept else 0))
        if k == 0 or ss_res <= 0:
            return np.inf
        return (ss_reg / k) / (ss_res / self._df_resid)

    @property
    def f_pvalue(self):
        """p-value for F-statistic."""
        fv = self.fvalue
        if fv is None or fv == np.inf:
            return 1.0
        k = int(self._X_design.shape[1] - (1 if self.fit_intercept else 0))
        return 1 - stats.f.cdf(fv, k, self._df_resid)

    @property
    def llf(self):
        """Log-likelihood (Gaussian MLE)."""
        if self._nobs is None or self._resid is None:
            return None
        n = self._nobs
        sigma2_mle = np.sum(self._resid ** 2) / n
        return -n / 2 * np.log(2 * np.pi * sigma2_mle) - n / 2

    @property
    def aic(self):
        """Akaike Information Criterion."""
        if self._nobs is None or self._scale is None or np.isnan(self._scale):
            return None
        return -2 * self.llf + 2 * len(self._params)

    @property
    def bic(self):
        """Bayesian Information Criterion."""
        if self._nobs is None or self._scale is None or np.isnan(self._scale):
            return None
        n = self._nobs
        k = len(self._params)
        return -2 * self.llf + k * np.log(n)

    def summary(self):
        """Print summary table similar to R's summary(lm())."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        if not self.compute_inference:
            raise RuntimeError(
                "compute_inference=False: summary/inference statistics are not available. "
                "Re-fit with compute_inference=True (default)."
            )
        if self._bse is None:
            raise RuntimeError("Inference statistics are not available.")

        if self.fit_intercept:
            feature_names = ['(Intercept)'] + [f'x{i+1}' for i in range(len(self.coef_))]
        else:
            feature_names = [f'x{i+1}' for i in range(len(self.coef_))]

        print("=" * 80)
        print("                              Ridge Regression Results")
        print("=" * 80)
        print(f"Alpha (L2 penalty):         {self.alpha:>15.4f}")
        print(f"Covariance Type:            {self.cov_type:>15}")
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
