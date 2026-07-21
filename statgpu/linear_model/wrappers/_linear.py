"""
Linear regression with full statistical inference and GPU support.
"""

__all__ = ["LinearRegression"]

from typing import Optional, Union
import numpy as np
from scipy import stats
from time import perf_counter

from statgpu._base import BaseEstimator
from statgpu._config import Device
from statgpu.backends import _get_torch_device_str
from statgpu.inference._results import GaussianInferenceResult
from statgpu.linear_model._gaussian_inference import (
    compute_gaussian_inference,
    validate_cov_type,
    validate_hac_maxlags,
)


def _parse_formula_if_provided(formula, data, X, y):
    """Parse formula data and return retained source-row positions."""
    if formula is not None:
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser(formula)
        y_arr, X_arr, info = parser.eval(data)
        return y_arr, X_arr, info, parser.row_positions
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return y, np.asarray(X), None, None


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
        self.cov_type = validate_cov_type(cov_type)
        self.hac_maxlags = validate_hac_maxlags(hac_maxlags)
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
        self._inference_result = None
        self._is_multi_output = False
        self._hac_mixed_precision_preference = {}
        self._feature_names = None
        self._design_info = None
        self._formula_has_intercept = None
        self._effective_fit_intercept = bool(fit_intercept)
        self._sample_weight_fit = None
        self._raw_resid = None

    def _clear_inference_result(self):
        self._bse = None
        self._tvalues = None
        self._pvalues = None
        self._conf_int = None
        self._inference_result = None

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
    
    def fit(self, X=None, y=None, sample_weight=None, formula=None, data=None):
        """Fit linear model.

        Parameters
        ----------
        X : array-like or None
            Predictor matrix. Required if ``formula`` is None.
        y : array-like or None
            Response vector. Required if ``formula`` is None.
        sample_weight : array-like or None
            Sample weights.
        formula : str or None
            R-style formula string (e.g. ``"y ~ x1 + x2"``). Mutually
            exclusive with ``X``/``y``.
        data : pd.DataFrame or None
            DataFrame used with ``formula`` for column lookup.
        """
        self._clear_inference_result()
        self._sample_weight_fit = None
        self._raw_resid = None

        # Formula syntax controls the fitted design without mutating the
        # public constructor parameter required by sklearn-style cloning.
        effective_fit_intercept = bool(self.fit_intercept)
        if formula is not None:
            if data is None:
                raise ValueError(
                    "formula was provided but data is None. "
                    "Pass data=your_dataframe when using formula."
                )
            y_arr, X_arr, design_info, retained_rows = _parse_formula_if_provided(
                formula, data, None, None
            )
            self._design_info = design_info
            formula_column_names = list(design_info.column_names)
            self._formula_has_intercept = "Intercept" in formula_column_names
            self._feature_names = [name for name in formula_column_names if name != "Intercept"]

            if sample_weight is not None:
                from statgpu.backends import _to_numpy

                weights = np.asarray(_to_numpy(sample_weight), dtype=float)
                if weights.ndim != 1:
                    raise ValueError("sample_weight must be one-dimensional")
                retained_rows = np.asarray(retained_rows, dtype=np.int64)
                if weights.shape[0] == len(data):
                    sample_weight = weights[retained_rows]
                elif weights.shape[0] == len(y_arr):
                    # Already aligned weights are accepted for programmatic use.
                    sample_weight = weights
                else:
                    raise ValueError(
                        "sample_weight must match the original data length or "
                        "the number of formula rows retained after missing-value filtering"
                    )

            if self._formula_has_intercept:
                intercept_idx = formula_column_names.index("Intercept")
                # Drop the intercept column — let the fitting methods handle it
                X_arr = np.delete(X_arr, intercept_idx, axis=1)
                effective_fit_intercept = True
            else:
                # Formula syntax owns intercept semantics, matching statsmodels/R.
                effective_fit_intercept = False
        else:
            if X is None or y is None:
                raise ValueError(
                    "Either formula+data or X+y must be provided."
            )
            self._feature_names = None
            self._design_info = None
            self._formula_has_intercept = None
            # Preserve backend-native inputs. Conversion is performed only
            # after the estimator backend has been resolved below.
            X_arr = X
            y_arr = y

        self._effective_fit_intercept = effective_fit_intercept

        # Resolve the backend before converting raw arrays so CuPy/Torch inputs
        # never make a GPU -> CPU -> GPU round trip.
        backend = self._get_backend(backend="auto")
        backend_name = backend.name

        X_arr = self._to_array(X_arr, backend=backend_name)
        y_arr = self._to_array(y_arr, backend=backend_name)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.reshape(-1)
        self._y = y_arr
        self._is_multi_output = y_arr.ndim > 1 and y_arr.shape[1] > 1

        device = self._get_compute_device()

        # Route to appropriate backend
        if backend_name == "torch":
            self._fit_torch(X_arr, y_arr, sample_weight)
        elif backend_name == "cupy":
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)

        # Convert y to numpy for diagnostics if needed
        if hasattr(self._y, 'get'):  # CuPy
            self._y = self._y.get()
        elif hasattr(self._y, 'cpu'):  # Torch
            self._y = self._y.cpu().numpy()
        else:
            self._y = np.asarray(self._y)

        # GPU single-output inference is computed in _fit_gpu/_fit_torch().
        # Multi-output GPU inference is not implemented yet; do not fall back to
        # the NumPy inference path when the user selected a GPU backend.
        if self.compute_inference and self._is_multi_output and device in (Device.CUDA, Device.TORCH):
            raise NotImplementedError(
                "Multi-output LinearRegression inference is not implemented for "
                f"device='{device.value}'. Set compute_inference=False or use device='cpu'."
            )
        if self.compute_inference and device == Device.CPU:
            self._compute_inference()
        self._fitted = True
        return self
    
    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU."""
        X_raw = np.asarray(X)
        y_raw = np.asarray(y)

        n_samples, n_features = X_raw.shape
        self._nobs = n_samples
        y_2d = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sw.shape[0] != n_samples:
                raise ValueError("sample_weight must have length n_samples")
            if not np.all(np.isfinite(sw)) or np.any(sw < 0) or float(sw.sum()) <= 0:
                raise ValueError("sample_weight must be finite, non-negative, and have positive sum")
            sqrt_sw = np.sqrt(sw)
            X_fit = X_raw * sqrt_sw[:, None]
            y_fit = y_2d * sqrt_sw[:, None]
            intercept_column = sqrt_sw[:, None]
            self._sample_weight_fit = sw.copy()
        else:
            X_fit = X_raw
            y_fit = y_2d
            intercept_column = np.ones((n_samples, 1), dtype=X_raw.dtype)

        if self._effective_fit_intercept:
            self._X_design = np.column_stack([intercept_column, X_fit])
        else:
            self._X_design = X_fit.copy()

        coef, _, _, _ = np.linalg.lstsq(self._X_design, y_fit, rcond=None)

        if self._effective_fit_intercept:
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
        self._resid = y_fit - y_pred
        raw_pred = (
            coef[0] + X_raw @ coef[1:]
            if self._effective_fit_intercept
            else X_raw @ coef
        )
        raw_resid = y_2d - raw_pred
        self._raw_resid = raw_resid[:, 0] if raw_resid.shape[1] == 1 else raw_resid
        if self._resid.shape[1] == 1:
            self._resid = self._resid[:, 0]
        self._df_resid = n_samples - (n_features + (1 if self._effective_fit_intercept else 0))
        
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
        from statgpu.backends._gpu_inference_cupy import (
            compute_inference_gpu,
            compute_r2_gpu,
            compute_aic_bic_gpu,
            compute_f_stat_gpu,
        )
        from statgpu.inference._distributions_backend import norm
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        # Ensure CuPy arrays and retain raw arrays for weighted diagnostics.
        X_raw = cp.asarray(X)
        y_raw = cp.asarray(y)
        y_2d = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

        sw = None
        if sample_weight is not None:
            sw = cp.asarray(sample_weight, dtype=cp.float64).reshape(-1)
            if sw.shape[0] != n_samples:
                raise ValueError("sample_weight must have length n_samples")
            valid = cp.all(cp.isfinite(sw)) & cp.all(sw >= 0) & (cp.sum(sw) > 0)
            if not bool(valid.item()):
                raise ValueError("sample_weight must be finite, non-negative, and have positive sum")
            sqrt_sw = cp.sqrt(sw)
            X_fit = X_raw * sqrt_sw[:, cp.newaxis]
            y_fit = y_2d * sqrt_sw[:, cp.newaxis]
            intercept_column = sqrt_sw[:, cp.newaxis]
        else:
            X_fit = X_raw
            y_fit = y_2d
            intercept_column = cp.ones((n_samples, 1), dtype=X_raw.dtype)

        if self._effective_fit_intercept:
            X_design = cp.column_stack([intercept_column, X_fit])
        else:
            X_design = X_fit
        y = y_fit
        
        # Use normal equations: (X'X)^-1 X'y
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        
        try:
            # Cholesky decomposition
            L = cp.linalg.cholesky(XtX)
            tmp = cp.linalg.solve_triangular(L, Xty, lower=True)
            coef = cp.linalg.solve_triangular(L.T, tmp, lower=False)
        except Exception:
            coef = cp.linalg.lstsq(X_design, y, rcond=None)[0]

        # Compute weighted inference residuals and raw diagnostic residuals.
        y_pred = X_design @ coef
        resid = y - y_pred
        raw_pred = (
            coef[0] + X_raw @ coef[1:]
            if self._effective_fit_intercept
            else X_raw @ coef
        )
        raw_resid = y_2d - raw_pred
        
        # Compute scale on GPU
        df_resid = n_samples - (n_features + (1 if self._effective_fit_intercept else 0))
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
                self._pvalues_gpu = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(self._tvalues_gpu)))
                z_crit = norm.ppf(0.975)
                self._conf_int_gpu = cp.stack([
                    coef_flat - z_crit * self._bse_gpu,
                    coef_flat + z_crit * self._bse_gpu,
                ], axis=1)

            # R-squared on GPU
            self._rsquared_gpu = compute_r2_gpu(y, resid)

            # AIC/BIC on GPU
            k = n_features + (1 if self._effective_fit_intercept else 0)
            scale_mle = cp.sum(resid ** 2) / n_samples
            self._aic_gpu, self._bic_gpu = compute_aic_bic_gpu(n_samples, k, scale_mle)

            # F-statistic on GPU
            self._fvalue_gpu, self._f_pvalue = compute_f_stat_gpu(y, resid, X_design, df_resid)

        # Single transfer to CPU at the end
        coef_np = coef.get()
        resid_np = resid.get()
        raw_resid_np = raw_resid.get()
        self._sample_weight_fit = None if sw is None else sw.get()
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
        if self._effective_fit_intercept:
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
        self._raw_resid = (
            raw_resid_np[:, 0] if raw_resid_np.shape[1] == 1 else raw_resid_np
        )
        self._df_resid = df_resid
        self._scale = scale_np
        if self.compute_inference and not self._is_multi_output:
            self._wrap_gaussian_inference_result()

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

    def _cleanup_torch_memory(self):
        """Best-effort Torch memory cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _hac_meat_torch(self, scores):
        """Torch Bartlett-kernel HAC meat from per-observation score matrix."""
        import torch

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

    def _robust_covariance_torch(self, X, resid, XtX_inv, device=None):
        """Compute robust/HAC covariance matrix for OLS-like score equations on Torch GPU."""
        import torch

        n, k = X.shape
        e = resid.reshape(-1)

        if device is None:
            device = 'cuda' if X.is_cuda else 'cpu'

        if self.cov_type == "hac":
            # HAC requires temporal ordering - compute score matrix and apply Bartlett kernel
            scores = X * e[:, None]
            meat = self._hac_meat_torch(scores)
            return XtX_inv @ meat @ XtX_inv

        if self.cov_type in ("hc2", "hc3"):
            leverage = torch.einsum("ij,jk,ik->i", X, XtX_inv, X)
            leverage = torch.clamp(leverage, 0.0, 1.0 - 1e-12)
            if self.cov_type == "hc2":
                e2 = torch.square(e) / (1.0 - leverage)
            else:
                e2 = torch.square(e) / torch.square(1.0 - leverage)
        else:
            e2 = torch.square(e)

        Xw = X * e2[:, None]
        meat = X.T @ Xw
        cov_params = XtX_inv @ meat @ XtX_inv
        if self.cov_type == "hc1" and n > k:
            cov_params = cov_params * (n / (n - k))
        return cov_params

    def _fit_torch(self, X, y, sample_weight=None):
        """Fit using Torch GPU with FULL GPU computation (including inference)."""
        import torch
        from statgpu.backends._gpu_inference_torch import (
            compute_inference_torch,
            compute_r2_torch,
            compute_aic_bic_torch,
            compute_f_stat_torch,
        )
        from statgpu.inference._distributions_backend import norm

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # Ensure Torch tensors on correct device
        # Note: Device.TORCH.value is 'torch', but Torch expects 'cuda' or 'cpu'
        torch_device = _get_torch_device_str()
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(np.asarray(X)).to(torch_device)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(np.asarray(y)).to(torch_device)

        if X.dtype != torch.float64:
            X = X.to(torch.float64)
        if y.dtype != torch.float64:
            y = y.to(torch.float64)

        X_raw = X
        y_raw = y
        y_2d = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

        sw = None
        if sample_weight is not None:
            sw = torch.as_tensor(sample_weight, dtype=torch.float64, device=torch_device).reshape(-1)
            if sw.shape[0] != n_samples:
                raise ValueError("sample_weight must have length n_samples")
            valid = torch.all(torch.isfinite(sw)) & torch.all(sw >= 0) & (torch.sum(sw) > 0)
            if not bool(valid.item()):
                raise ValueError("sample_weight must be finite, non-negative, and have positive sum")
            sqrt_sw = torch.sqrt(sw)
            X_fit = X_raw * sqrt_sw[:, None]
            y_fit = y_2d * sqrt_sw[:, None]
            intercept_column = sqrt_sw[:, None]
        else:
            X_fit = X_raw
            y_fit = y_2d
            intercept_column = torch.ones(
                n_samples, 1, dtype=X_raw.dtype, device=X_raw.device
            )

        if self._effective_fit_intercept:
            X_design = torch.cat([intercept_column, X_fit], dim=1)
        else:
            X_design = X_fit.clone()
        y = y_fit

        # Use normal equations: (X'X)^-1 X'y
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y

        try:
            # Cholesky decomposition
            L = torch.linalg.cholesky(XtX)
            # Solve L @ tmp = Xty (L is lower triangular)
            tmp = torch.linalg.solve_triangular(L, Xty, upper=False)
            # Solve L.T @ coef = tmp (L.T is upper triangular)
            coef = torch.linalg.solve_triangular(L.T, tmp, upper=True)
        except Exception:
            coef = torch.linalg.lstsq(X_design, y).solution

        # Compute weighted inference residuals and raw diagnostic residuals.
        y_pred = X_design @ coef
        resid = y - y_pred
        raw_pred = (
            coef[0] + X_raw @ coef[1:]
            if self._effective_fit_intercept
            else X_raw @ coef
        )
        raw_resid = y_2d - raw_pred

        # Compute scale on Torch
        df_resid = n_samples - (n_features + (1 if self._effective_fit_intercept else 0))
        if df_resid > 0:
            if y.shape[1] > 1:
                scale = torch.sum(resid ** 2, dim=0) / df_resid
            else:
                scale = torch.sum(resid ** 2) / df_resid
        else:
            if y.shape[1] > 1:
                scale = torch.full((y.shape[1],), float('nan'), dtype=y.dtype, device=torch_device)
            else:
                scale = torch.tensor(float('nan'), dtype=y.dtype, device=torch_device)

        # Compute inference-related statistics only when requested.
        if self.compute_inference and not self._is_multi_output:
            coef_flat = coef.flatten()
            if self.cov_type == "nonrobust":
                self._bse_gpu, self._tvalues_gpu, self._pvalues_gpu, self._conf_int_gpu = \
                    compute_inference_torch(X_design, resid, scale, df_resid, coef_flat, cov_type="nonrobust", device=torch_device)
            else:
                XtX_cov = X_design.T @ X_design
                try:
                    XtX_inv = torch.linalg.inv(XtX_cov)
                except Exception:
                    XtX_inv = torch.linalg.pinv(XtX_cov)
                cov_params = self._robust_covariance_torch(X_design, resid, XtX_inv, device=torch_device)
                self._bse_gpu = torch.sqrt(torch.clamp(torch.diag(cov_params), 0.0))
                self._tvalues_gpu = coef_flat / (self._bse_gpu + 1e-30)
                self._pvalues_gpu = torch.clamp(2.0 * norm.sf(torch.abs(self._tvalues_gpu), device=torch_device), 0.0, 1.0)
                z_crit = norm.ppf(0.975, device=torch_device)
                self._conf_int_gpu = torch.stack([
                    coef_flat - z_crit * self._bse_gpu,
                    coef_flat + z_crit * self._bse_gpu,
                ], dim=1)

            # R-squared on Torch
            self._rsquared_gpu = compute_r2_torch(y, resid)

            # AIC/BIC on Torch
            k = n_features + (1 if self._effective_fit_intercept else 0)
            scale_mle = torch.sum(resid ** 2) / n_samples
            self._aic_gpu, self._bic_gpu = compute_aic_bic_torch(n_samples, k, scale_mle, device=torch_device)

            # F-statistic on Torch
            self._fvalue_gpu, self._f_pvalue = compute_f_stat_torch(y, resid, X_design, df_resid, device=torch_device)

        # Single transfer to CPU at the end
        coef_np = coef.detach().cpu().numpy()
        resid_np = resid.detach().cpu().numpy()
        raw_resid_np = raw_resid.detach().cpu().numpy()
        self._sample_weight_fit = (
            None if sw is None else sw.detach().cpu().numpy()
        )
        if y.shape[1] > 1:
            scale_np = scale.detach().cpu().numpy()
        else:
            scale_val = scale.detach().cpu().item()
            scale_np = float(scale_val) if not np.isnan(scale_val) else np.nan
        X_design_np = X_design.detach().cpu().numpy()

        if self.compute_inference and not self._is_multi_output:
            # Transfer inference results
            self._bse = self._bse_gpu.detach().cpu().numpy()
            self._tvalues = self._tvalues_gpu.detach().cpu().numpy()
            self._pvalues = self._pvalues_gpu.detach().cpu().numpy()
            self._conf_int = self._conf_int_gpu.detach().cpu().numpy()

        # Store results
        if self._effective_fit_intercept:
            if coef_np.shape[1] > 1:
                self.intercept_ = coef_np[0, :].copy()
                self.coef_ = coef_np[1:, :].T
                self._params = coef_np.copy()
            else:
                self.intercept_ = float(coef_np[0, 0])
                self.coef_ = coef_np[1:, 0].copy()  # Ensure 1D array
                self._params = coef_np[:, 0].copy()
        else:
            if coef_np.shape[1] > 1:
                self.intercept_ = np.zeros(coef_np.shape[1], dtype=coef_np.dtype)
                self.coef_ = coef_np.T
                self._params = coef_np.copy()
            else:
                self.intercept_ = 0.0
                self.coef_ = coef_np[:, 0].copy()  # Ensure 1D array
                self._params = coef_np[:, 0].copy()

        self._X_design = X_design_np
        if resid_np.shape[1] == 1:
            self._resid = resid_np[:, 0]
        else:
            self._resid = resid_np
        self._raw_resid = (
            raw_resid_np[:, 0] if raw_resid_np.shape[1] == 1 else raw_resid_np
        )
        self._df_resid = df_resid
        self._scale = scale_np
        if self.compute_inference and not self._is_multi_output:
            self._wrap_gaussian_inference_result()

        # Release large temporary Torch tensors early.
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
        self._cleanup_torch_memory()
    
    def _compute_inference(self):
        """Compute standard errors, t-stats, p-values."""
        result = compute_gaussian_inference(
            self._X_design,
            self._params,
            self._resid,
            self._scale,
            self._df_resid,
            self.cov_type,
            hac_maxlags=self.hac_maxlags,
        )
        if result is None:
            self._clear_inference_result()
            return
        result.feature_names = self._inference_feature_names()
        result.apply_to(self)

    def _inference_feature_names(self):
        if self._feature_names is not None:
            names = list(self._feature_names)
            if self._effective_fit_intercept:
                names.insert(0, "(Intercept)")
            return names
        if self.coef_ is None:
            return None
        n_features = int(np.asarray(self.coef_).shape[-1])
        if self._effective_fit_intercept:
            return ["(Intercept)"] + [f"x{i+1}" for i in range(n_features)]
        return [f"x{i+1}" for i in range(n_features)]

    def _wrap_gaussian_inference_result(self):
        method = "classical" if self.cov_type == "nonrobust" else "sandwich"
        distribution = "t" if self.cov_type == "nonrobust" else "normal"
        result = GaussianInferenceResult(
            params=self._params,
            bse=self._bse,
            statistic=self._tvalues,
            pvalues=self._pvalues,
            conf_int=self._conf_int,
            cov_type=self.cov_type,
            distribution=distribution,
            df=self._df_resid,
            method=method,
            feature_names=self._inference_feature_names(),
            metadata={"alpha": 0.05},
        )
        result.apply_to(self)

    @property
    def rsquared(self):
        """R-squared."""
        if self._y is None or self._resid is None:
            return None
        y = np.asarray(self._y, dtype=float)
        resid = np.asarray(
            self._raw_resid if self._raw_resid is not None else self._resid,
            dtype=float,
        )
        weights = self._sample_weight_fit
        if weights is None:
            y_mean = np.mean(y, axis=0) if y.ndim > 1 else np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum(resid ** 2)
        else:
            weights = np.asarray(weights, dtype=float)
            y_mean = np.average(y, axis=0, weights=weights)
            weight_shape = (weights.shape[0],) + (1,) * (y.ndim - 1)
            w = weights.reshape(weight_shape)
            ss_tot = np.sum(w * (y - y_mean) ** 2)
            ss_res = np.sum(w * resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    @property
    def rsquared_adj(self):
        """Adjusted R-squared, or NaN when residual degrees of freedom are invalid."""
        if self._nobs is None:
            return None
        if self._df_resid is None or self._df_resid <= 0:
            return np.nan
        r2 = self.rsquared
        if r2 is None:
            return None
        return 1 - (1 - r2) * (self._nobs - 1) / self._df_resid
    
    @property
    def fvalue(self):
        """Overall regression F-statistic.

        The statistic is undefined for an intercept-only model, a constant target,
        or non-positive residual degrees of freedom.  A perfect non-constant fit
        has an infinite F-statistic.
        """
        if self._y is None or self._resid is None:
            return None
        k = int(self._X_design.shape[1] - (1 if self._effective_fit_intercept else 0))
        if k <= 0 or self._df_resid is None or self._df_resid <= 0:
            return np.nan
        y = np.asarray(self._y, dtype=float)
        resid = np.asarray(
            self._raw_resid if self._raw_resid is not None else self._resid,
            dtype=float,
        )
        weights = self._sample_weight_fit
        if weights is None:
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            ss_res = float(np.sum(resid ** 2))
        else:
            weights = np.asarray(weights, dtype=float)
            y_mean = np.average(y, weights=weights)
            ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
            ss_res = float(np.sum(weights * resid ** 2))
        if not np.isfinite(ss_tot) or not np.isfinite(ss_res) or ss_tot <= 0:
            return np.nan
        ss_reg = max(0.0, ss_tot - ss_res)
        tol = np.finfo(float).eps * max(1.0, ss_tot)
        if ss_res <= tol:
            return np.inf if ss_reg > tol else np.nan
        return (ss_reg / k) / (ss_res / self._df_resid)
    
    @property
    def f_pvalue(self):
        """Upper-tail p-value for the overall F-test."""
        fv = self.fvalue
        if fv is None:
            return None
        if np.isnan(fv):
            return np.nan
        if np.isposinf(fv):
            return 0.0
        k = int(self._X_design.shape[1] - (1 if self._effective_fit_intercept else 0))
        return float(stats.f.sf(fv, k, self._df_resid))
    
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
        """Gaussian log-likelihood evaluated at the MLE residual variance."""
        if self._nobs is None or self._resid is None:
            return None
        n = int(self._nobs)
        if n <= 0:
            return np.nan
        sigma2_mle = float(np.sum(np.asarray(self._resid, dtype=float) ** 2) / n)
        if not np.isfinite(sigma2_mle) or sigma2_mle < 0:
            return np.nan
        if sigma2_mle == 0:
            return np.inf
        return -n / 2 * (np.log(2 * np.pi * sigma2_mle) + 1.0)
    
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
        if self._bse is None or self._pvalues is None or self._conf_int is None:
            raise RuntimeError(
                "Inference statistics are not available for the current fit. "
                "This can happen when residual degrees of freedom are non-positive."
            )
        
        # Build feature names
        if self._feature_names is not None:
            feature_names = list(self._feature_names)
            if self._effective_fit_intercept:
                feature_names.insert(0, '(Intercept)')
        elif self._effective_fit_intercept:
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
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            If a DataFrame is passed and the model was trained with a formula,
            the design matrix is automatically built using the stored
            ``design_info``.

        Returns
        -------
        predictions : ndarray
        """
        self._check_is_fitted()

        # If model was trained with formula and X is a DataFrame,
        # rebuild the design matrix using the stored design_info.
        if self._design_info is not None:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                from statgpu.core.formula import FormulaParser
                # Reconstruct parser from design_info
                parser = FormulaParser.__new__(FormulaParser)
                parser._design_info = self._design_info
                parser.formula = None
                X = parser.transform(X)
                # Drop intercept column to match the fitting path
                col_names = list(self._design_info.column_names)
                if self._formula_has_intercept and "Intercept" in col_names:
                    intercept_idx = col_names.index("Intercept")
                    X = np.delete(X, intercept_idx, axis=1)
            else:
                # Preserve backend-native arrays; conversion happens below.
                pass
        else:
            # Preserve backend-native arrays; conversion happens below.
            pass

        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            X_gpu = cp.asarray(self._to_array(X, Device.CUDA))
            coef_gpu = cp.asarray(self.coef_)
            intercept_gpu = cp.asarray(self.intercept_, dtype=coef_gpu.dtype)
            if coef_gpu.ndim == 2:
                return X_gpu @ coef_gpu.T + intercept_gpu
            return X_gpu @ coef_gpu + intercept_gpu
        if device == Device.TORCH:
            import torch

            X_torch = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef_torch = torch.as_tensor(self.coef_, dtype=X_torch.dtype, device=X_torch.device)
            intercept_torch = torch.as_tensor(
                self.intercept_, dtype=X_torch.dtype, device=X_torch.device
            )
            if coef_torch.ndim == 2:
                return X_torch @ coef_torch.T + intercept_torch
            return X_torch @ coef_torch + intercept_torch
        X = self._to_array(X, Device.CPU)
        X = np.asarray(X)
        if np.asarray(self.coef_).ndim == 2:
            return X @ self.coef_.T + self.intercept_
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """Return R^2 score."""
        y_pred = self.predict(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            yb = cp.asarray(self._to_array(y, Device.CUDA))
            if y_pred.ndim == 1:
                ss_res = cp.sum((yb - y_pred) ** 2)
                ss_tot = cp.sum((yb - cp.mean(yb)) ** 2)
                return float((1 - ss_res / ss_tot).item()) if float(ss_tot.item()) > 0 else 0.0
            ss_res = cp.sum((yb - y_pred) ** 2, axis=0)
            ss_tot = cp.sum((yb - cp.mean(yb, axis=0)) ** 2, axis=0)
            r2 = cp.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
            return float(cp.mean(r2).item())
        if device == Device.TORCH:
            import torch

            yb = self._to_array(y, Device.TORCH, backend="torch").to(y_pred.dtype)
            if y_pred.ndim == 1:
                ss_res = torch.sum((yb - y_pred) ** 2)
                ss_tot = torch.sum((yb - torch.mean(yb)) ** 2)
                return float((1 - ss_res / ss_tot).item()) if float(ss_tot.item()) > 0 else 0.0
            ss_res = torch.sum((yb - y_pred) ** 2, dim=0)
            ss_tot = torch.sum((yb - torch.mean(yb, dim=0)) ** 2, dim=0)
            r2 = torch.where(ss_tot > 0, 1 - ss_res / ss_tot, torch.zeros_like(ss_tot))
            return float(torch.mean(r2).item())
        y_pred = np.asarray(y_pred)
        y = self._to_numpy(y)
        if y_pred.ndim == 1:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ss_res = np.sum((y - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
        return float(np.mean(r2))
