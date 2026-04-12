"""
Linear regression with full statistical inference and GPU support.
"""

from typing import Optional, Union
import numpy as np
from scipy import stats
from time import perf_counter

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
        cov_type: str = "nonrobust",
        hac_maxlags: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.fit_intercept = fit_intercept
        self.compute_inference = compute_inference
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
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
        self._is_multi_output = False
        self._hac_mixed_precision_preference = {}

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

    def _benchmark_hac_numpy_kernel(
        self,
        scores: np.ndarray,
        maxlags: int,
        use_mixed_precision: bool,
    ) -> float:
        """Benchmark a tiny HAC kernel to choose the faster precision path."""
        probe_maxlags = min(maxlags, 2)
        if use_mixed_precision:
            scores32 = scores.astype(np.float32, copy=False)
            t0 = perf_counter()
            meat = (scores32.T @ scores32).astype(np.float64)
            for lag in range(1, probe_maxlags + 1):
                weight = 1.0 - (lag / (maxlags + 1.0))
                gamma = scores32[lag:].T @ scores32[:-lag]
                meat = meat + float(weight) * (gamma + gamma.T).astype(np.float64)
            _ = float(meat[0, 0])
            return perf_counter() - t0

        t0 = perf_counter()
        meat = scores.T @ scores
        for lag in range(1, probe_maxlags + 1):
            weight = 1.0 - (lag / (maxlags + 1.0))
            gamma = scores[lag:].T @ scores[:-lag]
            meat = meat + weight * (gamma + gamma.T)
        _ = float(meat[0, 0])
        return perf_counter() - t0

    def _should_use_mixed_precision_hac_numpy(self, scores: np.ndarray, maxlags: int) -> bool:
        """Choose HAC precision path adaptively and cache by problem shape."""
        n_obs = int(scores.shape[0])
        n_features = int(scores.shape[1])
        if not (scores.dtype == np.float64 and n_obs >= 4096 and n_features <= 64):
            return False

        if n_obs < 32768:
            n_bucket = "small"
        elif n_obs < 65536:
            n_bucket = "medium"
        else:
            n_bucket = "large"

        key = (n_features, int(min(maxlags, 8)), n_bucket)
        cached = self._hac_mixed_precision_preference.get(key)
        if cached is not None:
            return bool(cached)

        probe_cap = 12288 if n_bucket != "large" else 24576
        probe_n = min(n_obs, probe_cap)
        if probe_n <= maxlags + 16:
            self._hac_mixed_precision_preference[key] = True
            return True

        probe_scores = np.asarray(scores[:probe_n], dtype=np.float64, order="C")
        try:
            # Warmup to reduce one-time BLAS startup noise.
            self._benchmark_hac_numpy_kernel(probe_scores, maxlags, use_mixed_precision=True)
            self._benchmark_hac_numpy_kernel(probe_scores, maxlags, use_mixed_precision=False)
            mixed_time = self._benchmark_hac_numpy_kernel(
                probe_scores, maxlags, use_mixed_precision=True
            )
            float64_time = self._benchmark_hac_numpy_kernel(
                probe_scores, maxlags, use_mixed_precision=False
            )
            # Keep mixed path only if it clears a small speed margin.
            use_mixed = mixed_time <= 0.95 * float64_time
        except Exception:
            use_mixed = True

        self._hac_mixed_precision_preference[key] = use_mixed
        return use_mixed

    def _hac_meat_numpy(self, scores: np.ndarray) -> np.ndarray:
        """Bartlett-kernel HAC meat from per-observation score matrix."""
        n_obs = int(scores.shape[0])
        maxlags = self._resolve_hac_maxlags(n_obs)
        weights = 1.0 - (np.arange(1, maxlags + 1, dtype=float) / (maxlags + 1.0))

        # Adaptive mixed precision: select per-shape path by quick local probe,
        # then cache the decision to avoid recurring benchmark overhead.
        use_mixed_precision = self._should_use_mixed_precision_hac_numpy(scores, maxlags)

        if use_mixed_precision:
            scores32 = scores.astype(np.float32, copy=False)
            meat = (scores32.T @ scores32).astype(np.float64)
            if maxlags == 0:
                return meat
            for lag, weight in enumerate(weights, start=1):
                gamma = scores32[lag:].T @ scores32[:-lag]
                meat = meat + float(weight) * (gamma + gamma.T).astype(np.float64)
            return meat

        meat = scores.T @ scores
        if maxlags == 0:
            return meat
        for lag, weight in enumerate(weights, start=1):
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
        """Compute robust/HAC covariance matrix for OLS-like score equations."""
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
            cov_params *= (n / (n - k))
        return cov_params

    def _robust_covariance_cupy(self, X, resid, XtX_inv):
        """Compute robust/HAC covariance matrix for OLS-like score equations on GPU."""
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
        """Fit linear model."""
        # Store y (may be CuPy array, convert later for CPU)
        self._y = y
        
        X_arr = self._to_array(X)
        y_arr = self._to_array(y)
        self._is_multi_output = y_arr.ndim > 1 and y_arr.shape[1] > 1
        
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

        # GPU single-output inference is computed in _fit_gpu().
        if self.compute_inference and (self._is_multi_output or device != Device.CUDA):
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

        if self.fit_intercept:
            if coef.shape[1] > 1:
                self.intercept_ = coef[0, :].copy()
                self.coef_ = coef[1:, :].T
                self._params = coef.copy()
            else:
                coef_1d = coef[:, 0]
                self.intercept_ = float(coef_1d[0])
                self.coef_ = coef_1d[1:]
                self._params = coef_1d.copy()
        else:
            if coef.shape[1] > 1:
                self.intercept_ = np.zeros(coef.shape[1], dtype=coef.dtype)
                self.coef_ = coef.T
                self._params = coef.copy()
            else:
                self.intercept_ = 0.0
                self.coef_ = coef[:, 0].copy()
                self._params = self.coef_.copy()

        y_pred = self._X_design @ coef
        self._resid = y - y_pred
        if self._resid.shape[1] == 1:
            self._resid = self._resid[:, 0]
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        
        if self._df_resid > 0:
            if np.asarray(self._resid).ndim == 1:
                self._scale = np.sum(self._resid ** 2) / self._df_resid
            else:
                self._scale = np.sum(self._resid ** 2, axis=0) / self._df_resid
        else:
            self._scale = np.nan
    
    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU with FULL GPU computation (including inference)."""
        import cupy as cp
        from .._gpu_utils import (
            compute_inference_gpu,
            compute_r2_gpu,
            compute_aic_bic_gpu,
            compute_f_stat_gpu,
            norm_two_tail_pvalues_gpu,
            norm_crit_gpu_two_tail,
        )
        
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
            if y.shape[1] > 1:
                scale = cp.sum(resid ** 2, axis=0) / df_resid
            else:
                scale = cp.sum(resid ** 2) / df_resid
        else:
            if y.shape[1] > 1:
                scale = cp.full((y.shape[1],), cp.nan, dtype=y.dtype)
            else:
                scale = cp.nan
        
        # Compute inference-related statistics only when requested.
        if self.compute_inference and not self._is_multi_output:
            coef_flat = coef.flatten()
            if self.cov_type == "nonrobust":
                self._bse_gpu, self._tvalues_gpu, self._pvalues_gpu, self._conf_int_gpu = \
                    compute_inference_gpu(X_design, resid, scale, df_resid, coef_flat)
            else:
                XtX_cov = X_design.T @ X_design
                try:
                    XtX_inv = cp.linalg.inv(XtX_cov)
                except Exception:
                    XtX_inv = cp.linalg.pinv(XtX_cov)
                cov_params = self._robust_covariance_cupy(X_design, resid, XtX_inv)
                self._bse_gpu = cp.sqrt(cp.maximum(cp.diag(cov_params), 0.0))
                self._tvalues_gpu = coef_flat / (self._bse_gpu + 1e-30)
                self._pvalues_gpu = norm_two_tail_pvalues_gpu(cp.abs(self._tvalues_gpu))
                z_crit = norm_crit_gpu_two_tail(0.05)
                self._conf_int_gpu = cp.stack([
                    coef_flat - z_crit * self._bse_gpu,
                    coef_flat + z_crit * self._bse_gpu,
                ], axis=1)

            # R-squared on GPU
            self._rsquared_gpu = compute_r2_gpu(y, resid)

            # AIC/BIC on GPU
            k = n_features + (1 if self.fit_intercept else 0)
            scale_mle = cp.sum(resid ** 2) / n_samples
            self._aic_gpu, self._bic_gpu = compute_aic_bic_gpu(n_samples, k, scale_mle)

            # F-statistic on GPU
            self._fvalue_gpu, self._f_pvalue = compute_f_stat_gpu(y, resid, X_design, df_resid)

        # Single transfer to CPU at the end
        coef_np = coef.get()
        resid_np = resid.get()
        if y.shape[1] > 1:
            scale_np = scale.get()
        else:
            scale_np = float(scale.get()) if not cp.isnan(scale) else np.nan
        X_design_np = X_design.get()
        
        if self.compute_inference and not self._is_multi_output:
            # Transfer inference results
            self._bse = self._bse_gpu.get()
            self._tvalues = self._tvalues_gpu.get()
            self._pvalues = self._pvalues_gpu.get()
            self._conf_int = self._conf_int_gpu.get()
        
        # Store results
        if self.fit_intercept:
            if coef_np.shape[1] > 1:
                self.intercept_ = coef_np[0, :].copy()
                self.coef_ = coef_np[1:, :].T
                self._params = coef_np.copy()
            else:
                self.intercept_ = float(coef_np[0, 0])
                self.coef_ = coef_np[1:, 0]
                self._params = coef_np[:, 0]
        else:
            if coef_np.shape[1] > 1:
                self.intercept_ = np.zeros(coef_np.shape[1], dtype=coef_np.dtype)
                self.coef_ = coef_np.T
                self._params = coef_np.copy()
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np[:, 0]
                self._params = coef_np[:, 0]
        
        self._X_design = X_design_np
        if resid_np.shape[1] == 1:
            self._resid = resid_np[:, 0]
        else:
            self._resid = resid_np
        self._df_resid = df_resid
        self._scale = scale_np

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
        if self._X_design is None or self._scale is None:
            return
        if np.any(np.isnan(np.asarray(self._scale, dtype=float))):
            return

        X = self._X_design
        n = X.shape[0]
        k = X.shape[1]
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        if np.asarray(self._params).ndim == 2:
            params = np.asarray(self._params, dtype=float)
            resid = np.asarray(self._resid, dtype=float)
            scale = np.asarray(self._scale, dtype=float).reshape(-1)
            n_targets = params.shape[1]
            self._bse = np.empty_like(params)
            self._tvalues = np.empty_like(params)
            self._pvalues = np.empty_like(params)
            self._conf_int = np.empty((params.shape[0], n_targets, 2), dtype=float)
            alpha = 0.05

            for j in range(n_targets):
                if self.cov_type == "nonrobust":
                    cov_params = scale[j] * XtX_inv
                    bse = np.sqrt(np.diag(cov_params))
                    tvalues = params[:, j] / (bse + 1e-30)
                    pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), self._df_resid))
                    t_crit = stats.t.ppf(1 - alpha / 2, self._df_resid)
                    conf_int = np.column_stack([
                        params[:, j] - t_crit * bse,
                        params[:, j] + t_crit * bse,
                    ])
                else:
                    cov_params = self._robust_covariance_numpy(X, resid[:, j], XtX_inv)
                    bse = np.sqrt(np.maximum(np.diag(cov_params), 0.0))
                    tvalues = params[:, j] / (bse + 1e-30)
                    pvalues = 2 * (1 - stats.norm.cdf(np.abs(tvalues)))
                    z_crit = stats.norm.ppf(1 - alpha / 2)
                    conf_int = np.column_stack([
                        params[:, j] - z_crit * bse,
                        params[:, j] + z_crit * bse,
                    ])

                self._bse[:, j] = bse
                self._tvalues[:, j] = tvalues
                self._pvalues[:, j] = pvalues
                self._conf_int[:, j, :] = conf_int
            return

        alpha = 0.05
        if self.cov_type == "nonrobust":
            cov_params = self._scale * XtX_inv
            self._bse = np.sqrt(np.diag(cov_params))
            self._tvalues = self._params / self._bse
            self._pvalues = 2 * (1 - stats.t.cdf(np.abs(self._tvalues), self._df_resid))
            t_crit = stats.t.ppf(1 - alpha / 2, self._df_resid)
            self._conf_int = np.column_stack([
                self._params - t_crit * self._bse,
                self._params + t_crit * self._bse,
            ])
            return

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
        k = int(self._X_design.shape[1] - (1 if self.fit_intercept else 0))
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
    def aic(self):
        """Akaike Information Criterion."""
        if self._is_multi_output:
            return None
        if self._nobs is None or self._scale is None:
            return None
        if np.any(np.isnan(self._scale)):
            return None
        # AIC = -2 * log-likelihood + 2 * k
        return -2 * self.llf + 2 * len(self._params)
    
    @property
    def bic(self):
        """Bayesian Information Criterion."""
        if self._is_multi_output:
            return None
        if self._nobs is None or self._scale is None:
            return None
        if np.any(np.isnan(self._scale)):
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
        if self._is_multi_output:
            raise RuntimeError("summary() is only available for single-output linear regression.")
        
        # Build feature names
        if self.fit_intercept:
            feature_names = ['(Intercept)'] + [f'x{i+1}' for i in range(len(self.coef_))]
        else:
            feature_names = [f'x{i+1}' for i in range(len(self.coef_))]
        
        print("=" * 80)
        print("                            Linear Regression Results")
        print("=" * 80)
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
    
    def predict(self, X):
        """Predict using the linear model."""
        self._check_is_fitted()
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            X_gpu = cp.asarray(self._to_array(X, Device.CUDA))
            coef_gpu = cp.asarray(self.coef_)
            intercept_gpu = cp.asarray(self.intercept_, dtype=coef_gpu.dtype)
            if coef_gpu.ndim == 2:
                return X_gpu @ coef_gpu.T + intercept_gpu
            return X_gpu @ coef_gpu + intercept_gpu
        X = self._to_array(X, Device.CPU)
        X = np.asarray(X)
        if np.asarray(self.coef_).ndim == 2:
            return X @ self.coef_.T + self.intercept_
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """Return R^2 score."""
        y_pred = self.predict(X)
        y = np.asarray(y)
        if y_pred.ndim == 1:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ss_res = np.sum((y - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
        return float(np.mean(r2))
