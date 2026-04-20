"""
Lasso regression with full statistical inference and GPU support.
"""

from collections import OrderedDict
import hashlib
from typing import Any, Dict, Optional, Tuple, Union
import os
import warnings
import numpy as np
from scipy import stats
from scipy.stats import norm as _norm_dist

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False

from .._base import BaseEstimator
from .._config import Device
from .._cv_base import CVEstimatorBase
from ..backends import get_backend
from ..inference._distributions_gpu import (
    norm,
    t,
)


_NUMBA_CD_DISABLED = str(os.getenv("STATGPU_DISABLE_NUMBA_CD", "0")).strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

_LASSO_CV_ALPHA_CACHE_MAXSIZE = int(os.getenv("STATGPU_LASSO_CV_CACHE_SIZE", "64"))
_LASSO_CV_ALPHA_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
_LASSO_DEBIASED_M_CACHE_MAXSIZE = int(os.getenv("STATGPU_LASSO_DEBIASED_M_CACHE_SIZE", "16"))
_LASSO_DEBIASED_M_CACHE: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
_LASSO_DEBIASED_M_GPU_HASH_ROW_CHUNK = 1024


# ============================================================================
# CuPy Fused Kernels for Lasso - Now implemented as Lasso class methods
# See Lasso._get_cupy_fused_kernels() for details.
# ============================================================================


def _debiased_m_cache_get(key):
    val = _LASSO_DEBIASED_M_CACHE.get(key)
    if val is not None:
        _LASSO_DEBIASED_M_CACHE.move_to_end(key)
    return val


def _debiased_m_cache_put(key, value):
    _LASSO_DEBIASED_M_CACHE[key] = value
    _LASSO_DEBIASED_M_CACHE.move_to_end(key)
    while len(_LASSO_DEBIASED_M_CACHE) > _LASSO_DEBIASED_M_CACHE_MAXSIZE:
        _LASSO_DEBIASED_M_CACHE.popitem(last=False)


def _debiased_m_key_from_numpy_design(
    X: np.ndarray,
    *,
    n: int,
    p: int,
    lam_nw: float,
    tol: float,
):
    X_cache = np.asarray(X)
    if not X_cache.flags["C_CONTIGUOUS"]:
        X_cache = np.ascontiguousarray(X_cache)
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray([int(n), int(p)], dtype=np.int64).tobytes())
    h.update(str(X_cache.dtype).encode("utf-8"))
    h.update(np.asarray([float(lam_nw), float(tol)], dtype=np.float64).tobytes())
    h.update(X_cache.view(np.uint8).tobytes())
    return h.hexdigest()


def _debiased_m_key_from_sample(
    *,
    n: int,
    p: int,
    dtype_name: str,
    sample_block: np.ndarray,
    lam_nw: float,
    tol: float,
):
    """Generate cache key for debiased M matrix from a sample block of X.

    This is used for Torch backend where we don't want to hash the entire matrix.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray([int(n), int(p)], dtype=np.int64).tobytes())
    h.update(dtype_name.encode("utf-8"))
    h.update(np.asarray([float(lam_nw), float(tol)], dtype=np.float64).tobytes())
    if not sample_block.flags["C_CONTIGUOUS"]:
        sample_block = np.ascontiguousarray(sample_block)
    h.update(sample_block.view(np.uint8).tobytes())
    return h.hexdigest()


class Lasso(BaseEstimator):
    """
    Lasso regression (L1 regularization) with GPU acceleration
    and full statistical inference.

    CPU solver supports multiple algorithms (coordinate descent by default, and FISTA when cpu_solver='fista').
    GPU solver supports multiple algorithms via `solver` (e.g. FISTA / ADMM).

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger values specify stronger regularization.
        Must be non-negative.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations for coordinate descent.
    tol : float, default=1e-4
        Tolerance for convergence.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    cpu_solver : str, default='coordinate_descent'
        CPU optimization algorithm: 'coordinate_descent' or 'fista'.
        GPU uses the `solver` parameter instead.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term.
    n_iter_ : int
        Number of iterations run.
    """

    # Internal cache for CuPy fused kernels (populated on first GPU use)
    _cupy_fused_kernels = None

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        inference_method: str = "cpu_ols_inference",
        n_bootstrap: int = 200,
        bootstrap_random_state: Optional[int] = None,
        enable_simultaneous_inference: bool = False,
        simultaneous_method: str = "maxz_bootstrap",
        simultaneous_alpha: float = 0.05,
        simultaneous_n_bootstrap: int = 1000,
        simultaneous_random_state: Optional[int] = None,
        simultaneous_include_intercept: bool = False,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        solver: str = "fista",
        cpu_solver: str = "coordinate_descent",
        lipschitz_L: Optional[float] = None,
        admm_rho: float = 1.0,
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.stopping = stopping.lower()
        self.inference_method = inference_method.lower()
        # Semantic rename with backwards-compatible aliases.
        # - "naive_ols" previously meant CPU-sided t-distribution inference.
        # - "gpu_naive_ols" previously meant GPU-sided t-distribution inference
        #   with minimal residual/design transfers.
        alias_map = {
            "naive_ols": "cpu_ols_inference",
            "gpu_naive_ols": "gpu_ols_inference",
        }
        self.inference_method = alias_map.get(self.inference_method, self.inference_method)
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_random_state = bootstrap_random_state
        self.enable_simultaneous_inference = bool(enable_simultaneous_inference)
        self.simultaneous_method = str(simultaneous_method).lower()
        self.simultaneous_alpha = float(simultaneous_alpha)
        self.simultaneous_n_bootstrap = int(simultaneous_n_bootstrap)
        self.simultaneous_random_state = simultaneous_random_state
        self.simultaneous_include_intercept = bool(simultaneous_include_intercept)
        self.compute_inference = compute_inference
        self.solver = solver.lower()
        self.cpu_solver = cpu_solver.lower()
        self.lipschitz_L = lipschitz_L
        self.admm_rho = admm_rho
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

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
        self._conf_int_simultaneous = None
        self._simultaneous_enabled = False
        self._simultaneous_method = None
        self._simultaneous_alpha = None
        self._simultaneous_n_bootstrap = None
        self._simultaneous_critical_value = None
        self._simultaneous_target_mask = None
        self._debiased_M_cpu = None
        self._inference_cautions = []

    def fit(self, X, y, sample_weight=None):
        """Fit Lasso regression model using coordinate descent."""
        self._validate_simultaneous_config()
        self._reset_simultaneous_outputs()
        device = self._get_compute_device()

        # Get backend - support explicit torch backend selection
        backend = self._get_backend(backend="auto")
        backend_name = backend.name

        # If the user requested GPU-only inference but we ended up on CPU,
        # gracefully fall back to CPU inference instead of leaving
        # inference fields unset.
        if device != Device.CUDA and self.inference_method == "gpu_ols_inference":
            self.inference_method = "cpu_ols_inference"
        if device != Device.CUDA:
            self._y = np.asarray(y)
        else:
            # GPU path: avoid host copies unless CPU-side inference needs y.
            if (not self.compute_inference) or self.inference_method in (
                "gpu_ols_inference",
                "debiased",
            ):
                self._y = None
            else:
                # y may already be a CuPy array; use safe conversion.
                self._y = self._to_numpy(y)

        X_arr = self._to_array(X, backend=backend_name)
        y_arr = self._to_array(y, backend=backend_name)

        # Route to appropriate backend
        if backend_name == "torch":
            self._fit_torch(X_arr, y_arr, sample_weight)
        elif device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)

        _skip_post_fit = {"gpu_ols_inference"}
        if device == Device.CUDA and self.inference_method == "debiased":
            _skip_post_fit.add("debiased")
        if backend_name == "torch" and self.inference_method == "debiased":
            _skip_post_fit.add("debiased")
        if self.compute_inference and self.inference_method not in _skip_post_fit:
            self._compute_inference()
        if self.enable_simultaneous_inference:
            self._compute_simultaneous_inference()
        self._inference_cautions = self._build_inference_cautions()
        for msg in self._inference_cautions:
            warnings.warn(msg, UserWarning, stacklevel=2)
        self._fitted = True
        return self

    def _validate_simultaneous_config(self):
        if not self.enable_simultaneous_inference:
            return
        if not self.compute_inference:
            raise ValueError(
                "enable_simultaneous_inference=True requires compute_inference=True."
            )
        if self.inference_method != "debiased":
            raise ValueError(
                "enable_simultaneous_inference=True currently requires "
                "inference_method='debiased'."
            )
        if self.simultaneous_method != "maxz_bootstrap":
            raise ValueError(
                "simultaneous_method must be 'maxz_bootstrap'."
            )
        if not (0.0 < self.simultaneous_alpha < 1.0):
            raise ValueError("simultaneous_alpha must be in (0, 1).")
        if self.simultaneous_n_bootstrap <= 0:
            raise ValueError("simultaneous_n_bootstrap must be a positive integer.")

    def _reset_simultaneous_outputs(self):
        self._conf_int_simultaneous = None
        self._simultaneous_enabled = False
        self._simultaneous_method = None
        self._simultaneous_alpha = None
        self._simultaneous_n_bootstrap = None
        self._simultaneous_critical_value = None
        self._simultaneous_target_mask = None
        self._debiased_M_cpu = None

    def _build_inference_cautions(self):
        cautions = []
        if not self.compute_inference:
            return cautions

        if self.inference_method in ("cpu_ols_inference", "gpu_ols_inference"):
            cautions.append(
                "Lasso OLS-style post-selection intervals are heuristic and do not "
                "provide valid selective-inference confidence coverage."
            )

        if self.inference_method == "debiased":
            cautions.append(
                "Debiased Lasso currently reports per-coefficient (marginal) confidence "
                "intervals only; joint/multiple-testing coverage is not guaranteed."
            )
        if self._simultaneous_enabled:
            target_txt = (
                "including intercept"
                if (self.fit_intercept and self.simultaneous_include_intercept)
                else "excluding intercept"
            )
            cautions.append(
                "Simultaneous inference enabled via maxz_bootstrap with joint coverage "
                f"target set {target_txt}."
            )
            if self.fit_intercept and self.simultaneous_include_intercept:
                cautions.append(
                    "Intercept is included using the same max-|Z| critical value "
                    "calibrated on feature coefficients."
                )

        return cautions

    @staticmethod
    def _get_cupy_fused_kernels():
        """
        Get cached CuPy fused kernels for Lasso FISTA solver.

        Fused kernels combine multiple elementwise operations into a single
        kernel launch, reducing GPU kernel launch overhead. This is especially
        beneficial for small-to-medium data sizes (n < 2000, p < 100).

        Returns
        -------
        dict or None
            Dictionary of fused kernels, or None if CuPy is not available.
        """
        # Check cache first (class-level cache shared across all instances)
        if Lasso._cupy_fused_kernels is not None:
            return Lasso._cupy_fused_kernels

        try:
            import cupy as cp
        except ImportError:
            return None

        # Fused soft thresholding: sign(x) * max(|x| - gamma, 0)
        @cp.fuse()
        def _soft_threshold_fused(x, gamma):
            """Fused soft thresholding operator."""
            abs_x = abs(x)
            return (x > 0) * (abs_x > gamma) * (abs_x - gamma) - (x < 0) * (abs_x > gamma) * (abs_x - gamma)

        # Fused FISTA momentum update: coef + beta * (coef - coef_old)
        @cp.fuse()
        def _fista_momentum_fused(coef, coef_old, beta):
            """Fused FISTA momentum update."""
            return coef + beta * (coef - coef_old)

        # Fused KKT violation check: max(|grad| - alpha, 0)
        @cp.fuse()
        def _kkt_violation_fused(grad, alpha):
            """Fused KKT violation computation."""
            abs_grad = abs(grad)
            diff = abs_grad - alpha
            return (diff > 0) * diff

        # Custom ElementwiseKernel for soft thresholding
        SOFT_THRESHOLD_KERNEL = cp.ElementwiseKernel(
            'float64 x, float64 gamma',
            'float64 y',
            '''
            double abs_x = abs(x);
            if (abs_x > gamma) {
                y = (x > 0 ? 1.0 : -1.0) * (abs_x - gamma);
            } else {
                y = 0.0;
            }
            ''',
            'lasso_soft_threshold'
        )

        # Custom ElementwiseKernel for absolute delta (convergence check)
        ABS_DELTA_KERNEL = cp.ElementwiseKernel(
            'float64 a, float64 b',
            'float64 y',
            '''
            double diff = a - b;
            y = (diff > 0 ? diff : -diff);
            ''',
            'lasso_abs_delta'
        )

        # Cache and return
        Lasso._cupy_fused_kernels = {
            'soft_threshold': _soft_threshold_fused,
            'fista_momentum': _fista_momentum_fused,
            'kkt_violation': _kkt_violation_fused,
            'elementwise_kernel': SOFT_THRESHOLD_KERNEL,
            'abs_delta_kernel': ABS_DELTA_KERNEL,
        }

        return Lasso._cupy_fused_kernels

    def _soft_threshold(self, x, gamma):
        """Soft thresholding operator: S(x, gamma) = sign(x) * max(|x| - gamma, 0)."""
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    def _fit_cpu(self, X, y, sample_weight=None):
        """Fit using CPU (coordinate descent or FISTA)."""
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
            y_centered = y

        if y.ndim == 1:
            y_centered = y_centered.reshape(-1, 1)

        Xty = X_centered.T @ y_centered.flatten()
        XtX = X_centered.T @ X_centered

        coef = np.zeros(n_features)

        if self.cpu_solver in ("fista",):
            # Proximal gradient / FISTA for L1-regularized least squares:
            #   minimize (1/(2n)) * ||y - Xw||^2 + alpha * ||w||_1
            # Uses the same stopping criterion as coordinate descent in this codebase:
            #   sum(abs(coef - coef_old)) < tol

            if self.lipschitz_L is not None:
                L = float(self.lipschitz_L)
            else:
                L_frob = float(np.sum(X_centered**2) / n_samples)
                try:
                    eigvals = np.linalg.eigvalsh(XtX)
                    L = float(eigvals[-1] / n_samples)
                except Exception:
                    L = L_frob

            if L <= 0:
                coef = np.zeros(n_features)
                self.n_iter_ = 0
            else:
                step = 1.0 / L
                thresh = self.alpha * step

                # FISTA variables
                y_k = coef.copy()
                t_k = 1.0

                for iteration in range(self.max_iter):
                    coef_old = coef.copy()

                    # grad = (XtX @ y_k - Xty) / n
                    grad = (XtX @ y_k - Xty) / n_samples

                    coef = self._soft_threshold(y_k - step * grad, thresh)

                    # Momentum update
                    t_new = (1.0 + np.sqrt(1.0 + 4.0 * (t_k**2))) / 2.0
                    beta = (t_k - 1.0) / t_new
                    y_k = coef + beta * (coef - coef_old)
                    t_k = t_new

                    if self.stopping == "kkt":
                        # KKT violation for Lasso:
                        # grad_sse = (XtX @ w - Xty) / n
                        # optimality: |grad_sse_j| <= alpha when w_j == 0
                        # violation measure: max_j max(|grad_sse_j| - alpha, 0)
                        grad_sse = (XtX @ coef - Xty) / n_samples
                        violation = np.max(np.maximum(np.abs(grad_sse) - self.alpha, 0.0))
                        if violation < self.tol:
                            self.n_iter_ = iteration + 1
                            break
                    else:
                        # Legacy stopping: coefficient delta
                        if np.sum(np.abs(coef - coef_old)) < self.tol:
                            self.n_iter_ = iteration + 1
                            break
                else:
                    self.n_iter_ = self.max_iter

        else:
            # Coordinate descent (legacy CPU path)
            # Precompute squared norms for each feature
            X_sq_norms = np.diag(XtX)

            for iteration in range(self.max_iter):
                coef_old = coef.copy()

                for j in range(n_features):
                    # Compute partial residual
                    rho_j = Xty[j] - np.dot(XtX[j, :], coef) + XtX[j, j] * coef[j]

                    # Update coefficient with soft thresholding
                    if X_sq_norms[j] > 1e-10:
                        coef[j] = self._soft_threshold(rho_j, self.alpha * n_samples) / X_sq_norms[j]
                    else:
                        coef[j] = 0.0

                # Check convergence
                if self.stopping == "kkt":
                    grad_sse = (XtX @ coef - Xty) / n_samples
                    violation = np.max(np.maximum(np.abs(grad_sse) - self.alpha, 0.0))
                    if violation < self.tol:
                        self.n_iter_ = iteration + 1
                        break
                else:
                    if np.sum(np.abs(coef - coef_old)) < self.tol:
                        self.n_iter_ = iteration + 1
                        break
            else:
                self.n_iter_ = self.max_iter

        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ coef)
            self.coef_ = coef
            self._params = np.concatenate([[self.intercept_], self.coef_])
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
            self._params = coef.copy()
        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        if self.compute_inference:
            if self.fit_intercept:
                self._X_design = np.column_stack(
                    [np.ones(n_samples, dtype=X.dtype), X]
                )
            else:
                self._X_design = X.copy()

            y_pred = self._X_design @ self._params
            self._resid = self._y - y_pred

            if self._df_resid > 0:
                self._scale = np.sum(self._resid ** 2) / self._df_resid
            else:
                self._scale = np.nan
        else:
            self._X_design = None
            self._resid = None
            self._scale = np.nan

    def _soft_threshold_cupy(self, x, gamma):
        """Soft thresholding operator for CuPy arrays.

        Uses fused kernel when available for improved performance on
        small-to-medium data sizes.
        """
        import cupy as cp

        # Try to use fused kernel for better performance
        fused = self._get_cupy_fused_kernels()
        if fused is not None:
            # Use ElementwiseKernel for best performance
            return fused['elementwise_kernel'](x, gamma)

        # Fallback to standard implementation
        return cp.sign(x) * cp.maximum(cp.abs(x) - gamma, 0)

    def _cleanup_cuda_memory(self):
        """
        Best-effort CUDA memory pool cleanup.

        CuPy caches freed blocks in its memory pool for speed. Enable
        `gpu_memory_cleanup=True` to return cached blocks after fit when
        VRAM pressure is more important than repeated-fit throughput.
        """
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _fit_gpu(self, X, y, sample_weight=None):
        """Fit using GPU solver."""
        import cupy as cp
        from .._gpu_utils import compute_r2_gpu

        if self.solver not in ("fista", "admm"):
            raise ValueError("solver must be one of: 'fista', 'admm'")

        if self.solver == "admm":
            return self._fit_gpu_admm(X, y, sample_weight=sample_weight)

        # Default: FISTA
        
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

        # Ensure vector y on GPU
        y = y.reshape(-1)

        # Center X/y when fitting intercept to match sklearn Lasso convention.
        if self.fit_intercept:
            X_mean = cp.mean(X, axis=0)
            y_mean = cp.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0, dtype=X.dtype)
            y_centered = y

        # Precompute XtX / Xty for FISTA gradient: grad(w) = (XtX @ w - Xty) / n
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Lipschitz constant L for grad(w): L = lambda_max(XtX) / n
        # If user provides lipschitz_L, trust it (should be safe for convergence).
        if self.lipschitz_L is not None:
            L = cp.array(float(self.lipschitz_L), dtype=X.dtype)
        else:
            L_frob = cp.sum(X_centered ** 2) / n_samples
            try:
                eigvals = cp.linalg.eigvalsh(XtX)
                L = eigvals[-1] / n_samples
            except Exception:
                L = L_frob

        if L <= 0:
            # Degenerate case: return all-zero coefficients
            coef = cp.zeros(n_features, dtype=X.dtype)
            self.n_iter_ = 0
        else:
            step = 1.0 / L
            thresh = self.alpha * step

            # FISTA variables
            coef = cp.zeros(n_features, dtype=X.dtype)  # w_k
            y_k = coef.copy()  # y_k
            t_k = cp.array(1.0, dtype=X.dtype)

            # Get fused kernels for optimized FISTA iterations
            fused = self._get_cupy_fused_kernels()

            for iteration in range(self.max_iter):
                coef_old = coef

                # Gradient at y_k: (1/n) XtX @ y_k - (1/n) Xty
                grad = (XtX @ y_k - Xty) / n_samples

                # Prox step for L1
                coef = self._soft_threshold_cupy(y_k - step * grad, thresh)

                # Momentum update (use fused kernel when available)
                t_new = (1 + cp.sqrt(1 + 4 * (t_k ** 2))) / 2
                beta = (t_k - 1) / t_new
                if fused is not None:
                    y_k = fused['fista_momentum'](coef, coef_old, beta)
                else:
                    y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                # Convergence test
                if self.stopping == "kkt":
                    grad_sse = (XtX @ coef - Xty) / n_samples
                    # Use fused KKT violation check when available
                    if fused is not None:
                        violation = cp.max(fused['kkt_violation'](grad_sse, self.alpha))
                    else:
                        violation = cp.max(cp.maximum(cp.abs(grad_sse) - self.alpha, 0.0))
                    if violation < self.tol:
                        self.n_iter_ = iteration + 1
                        break
                else:
                    # Legacy stopping: coefficient delta (fast but not guaranteed objective optimality)
                    # Use fused delta kernel when available
                    if fused is not None and 'abs_delta_kernel' in fused:
                        delta = cp.sum(fused['abs_delta_kernel'](coef, coef_old))
                    else:
                        delta = cp.sum(cp.abs(coef - coef_old))
                    if delta < self.tol:
                        self.n_iter_ = iteration + 1
                        break
            else:
                self.n_iter_ = self.max_iter

        # Build full coefficients and (optionally) residuals for inference/R^2
        if self.fit_intercept:
            intercept_gpu = y_mean - X_mean @ coef
            coef_full = cp.concatenate([intercept_gpu.reshape(1), coef])
        else:
            coef_full = coef

        # Always transfer coefficients; remaining transfers depend on compute_inference.
        coef_full_np = coef_full.get()

        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np

        df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        self._df_resid = df_resid

        # Inference/diagnostics require residuals and design matrix.
        if self.compute_inference:
            # Only build the design matrix when we need residuals/inference.
            if self.fit_intercept:
                X_design = cp.concatenate(
                    [cp.ones((n_samples, 1), dtype=X.dtype), X], axis=1
                )
            else:
                X_design = X

            y_pred = X_design @ coef_full
            resid = y - y_pred

            if df_resid > 0:
                scale = cp.sum(resid ** 2) / df_resid
                self._scale = float(scale.get()) if not cp.isnan(scale) else np.nan
            else:
                self._scale = np.nan
                scale = cp.nan

            if self.inference_method == "gpu_ols_inference":
                # Compute inference fully on GPU, then transfer only small vectors.
                XtX = X_design.T @ X_design
                try:
                    XtX_inv = cp.linalg.inv(XtX)
                except Exception:
                    XtX_inv = cp.linalg.pinv(XtX)

                bse_gpu = cp.sqrt(scale * cp.diag(XtX_inv))

                # Inference vectors on GPU to avoid scipy/cpu cdf/ppf.
                params_gpu = coef_full  # includes intercept when fit_intercept=True
                tvalues_gpu = params_gpu / (bse_gpu + 1e-30)
                # Two-tailed p-values from the Student-t survival function should
                # already lie in [0, 1]. We still clamp at 1.0 as a defensive
                # safeguard against tiny floating-point overshoots on GPU/backends.
                pvalues_gpu = cp.minimum(1.0, 2.0 * t.sf(cp.abs(tvalues_gpu), df=df_resid))

                alpha = 0.05  # two-tailed for 95% CI
                t_crit_gpu = t.ppf(1.0 - alpha / 2.0, df=df_resid)
                margin_gpu = t_crit_gpu * bse_gpu
                conf_int_gpu = cp.stack([params_gpu - margin_gpu, params_gpu + margin_gpu], axis=1)

                # Transfer only the small inference vectors back to CPU.
                self._bse = cp.asnumpy(bse_gpu)
                self._tvalues = cp.asnumpy(tvalues_gpu)
                self._pvalues = cp.asnumpy(pvalues_gpu)
                self._conf_int = cp.asnumpy(conf_int_gpu)

                # R^2 / keep diagnostics consistent without transferring residuals.
                y_mean_gpu = cp.mean(y)
                ss_tot = cp.sum((y - y_mean_gpu) ** 2)
                ss_res = cp.sum(resid ** 2)
                self._rsquared_gpu = float(cp.asnumpy(1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0

                self._resid = None
                self._X_design = None
            elif self.inference_method == "debiased":
                self._compute_inference_debiased_gpu(X, y, coef)

                y_mean_gpu = cp.mean(y)
                ss_tot = cp.sum((y - y_mean_gpu) ** 2)
                ss_res = cp.sum(resid ** 2)
                self._rsquared_gpu = float(cp.asnumpy(1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0

                self._resid = None
                self._X_design = None
            else:
                # Default: transfer residuals and design to CPU.
                self._resid = resid.get()
                self._X_design = X_design.get()

        else:
            # Strict GPU mode: avoid large residual/host design transfers.
            self._scale = np.nan
            self._resid = None
            self._X_design = None
            # R^2 is optional; keep behavior as None when no residuals are available.
            self._rsquared_gpu = None

        # Drop large temporaries early (before optional pool cleanup).
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
            del X_centered
        except Exception:
            pass
        try:
            del y_centered
        except Exception:
            pass
        try:
            del y_pred
        except Exception:
            pass
        try:
            del coef_full
        except Exception:
            pass
        self._cleanup_cuda_memory()

    def _cleanup_torch_memory(self):
        """Best-effort Torch CUDA memory cleanup."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def _matrix_fingerprint_torch(self, X: "torch.Tensor") -> str:
        """Generate a fingerprint key for caching debiased M matrix (Torch version)."""
        import torch
        n, p = X.shape
        r = min(24, n)
        c = min(24, p)
        sample = X[:r, :c].contiguous()
        h = hashlib.sha1()
        h.update(str((n, p, str(X.dtype))).encode("utf-8"))
        h.update(sample.cpu().numpy().tobytes())
        return h.hexdigest()

    def _solve_lasso_path_torch_fista_multi_fold_from_gram(
        self,
        XtX_batch,
        Xty_batch,
        *,
        n_samples_vec,
        alphas_desc,
        max_iter,
        tol,
        stopping,
        lipschitz_L=None,
        check_every=8,
    ):
        """Solve descending-alpha Lasso paths for all folds together on Torch GPU."""
        import torch

        n_folds = int(XtX_batch.shape[0])
        n_features = int(XtX_batch.shape[1])
        n_alphas = int(alphas_desc.shape[0])
        dtype = XtX_batch.dtype
        device = XtX_batch.device

        coefs = torch.zeros((n_folds, n_features, n_alphas), dtype=dtype, device=device)
        yk = coefs.clone()
        tk = torch.ones((n_folds, n_alphas), dtype=dtype, device=device)
        n_iters = torch.zeros((n_folds, n_alphas), dtype=torch.int32, device=device)

        n_vec = torch.as_tensor(n_samples_vec, dtype=dtype, device=device).reshape(-1)
        if n_vec.size != n_folds:
            raise ValueError("n_samples_vec must have one entry per fold")

        if lipschitz_L is not None:
            L = torch.full((n_folds,), float(lipschitz_L), dtype=dtype, device=device)
        else:
            try:
                eigvals = torch.linalg.eigvalsh(XtX_batch)
                L = eigvals[:, -1] / n_vec
            except Exception:
                row_sum_bound = torch.max(torch.sum(torch.abs(XtX_batch), dim=2), dim=1)[0] / n_vec
                L = torch.maximum(row_sum_bound, torch.tensor(1e-12, dtype=dtype, device=device))

        step = 1.0 / L.reshape(n_folds, 1, 1)
        alpha_gpu = torch.as_tensor(np.asarray(alphas_desc, dtype=np.float64), dtype=dtype, device=device).reshape(1, 1, n_alphas)
        thresholds = alpha_gpu * step

        Xty_expanded = Xty_batch.reshape(n_folds, n_features, 1)
        n_vec_expanded = n_vec.reshape(n_folds, 1, 1)
        stopping_name = str(stopping).lower()
        check_every = max(1, int(check_every))

        active_gpu = torch.ones((n_folds, n_alphas), dtype=torch.bool, device=device)
        active_count = int(n_folds * n_alphas)

        for iteration in range(int(max_iter)):
            if active_count == 0:
                break

            active_expanded = active_gpu[:, None, :]

            coef_old = coefs.clone()
            grad = (torch.matmul(XtX_batch, yk) - Xty_expanded) / n_vec_expanded
            coef_candidate = torch.sign(yk - step * grad) * torch.maximum(torch.abs(yk - step * grad) - thresholds, torch.tensor(0.0, dtype=dtype, device=device))
            coefs = torch.where(active_expanded, coef_candidate, coefs)

            t_old = tk
            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
            beta = (t_old - 1.0) / t_new
            y_candidate = coefs + beta[:, None, :] * (coefs - coef_old)
            yk = torch.where(active_expanded, y_candidate, yk)
            tk = torch.where(active_gpu, t_new, tk)

            active_ratio = float(active_count) / float(max(1, n_folds * n_alphas))
            check_every_eff = max(check_every, 1)
            should_check = ((iteration + 1) % check_every_eff == 0) or (iteration + 1 == int(max_iter))
            if not should_check:
                continue

            if stopping_name == "kkt":
                grad_sse = (torch.matmul(XtX_batch, coefs) - Xty_expanded) / n_vec_expanded
                violation = torch.max(torch.maximum(torch.abs(grad_sse) - alpha_gpu, torch.tensor(0.0, dtype=dtype, device=device)), dim=1)[0]
                converged_local_gpu = violation < float(tol)
            else:
                delta = torch.sum(torch.abs(coefs - coef_old), dim=1)
                converged_local_gpu = delta < float(tol)

            newly_done_gpu = active_gpu & converged_local_gpu
            done_count = int(torch.count_nonzero(newly_done_gpu).item())
            if done_count == 0:
                continue

            n_iters[newly_done_gpu] = int(iteration) + 1
            yk = torch.where(newly_done_gpu[:, None, :], coefs, yk)
            active_gpu = active_gpu & (~converged_local_gpu)
            active_count -= done_count

        return coefs.transpose(1, 2), n_iters.cpu().numpy()

    def _compute_inference_debiased_torch(self, X_torch, y_torch, coef_torch):
        """Torch GPU path for debiased Lasso inference.

        Parameters
        ----------
        X_torch : torch.Tensor, shape (n, p)
            Raw feature matrix on Torch GPU (no intercept column).
        y_torch : torch.Tensor, shape (n,)
            Response on Torch GPU.
        coef_torch : torch.Tensor, shape (p,)
            Lasso coefficients on Torch GPU (no intercept).
        """
        import torch
        from ..inference._distributions_torch import norm

        n, p = X_torch.shape
        dtype = torch.float64
        device = X_torch.device

        # Ensure correct dtype
        if X_torch.dtype != dtype:
            X_torch = X_torch.to(dtype)
        if y_torch.dtype != dtype:
            y_torch = y_torch.to(dtype)
        if coef_torch.dtype != dtype:
            coef_torch = coef_torch.to(dtype)

        # Compute Sigma_hat = X'X / n
        Sigma_hat = X_torch.T @ X_torch / n

        # Compute Lasso residuals
        resid_lasso = y_torch - X_torch @ coef_torch
        if self.fit_intercept:
            resid_lasso = resid_lasso - torch.mean(y_torch) + torch.mean(X_torch, dim=0) @ coef_torch

        # Estimate noise variance sigma^2
        s_hat = torch.sum(torch.abs(coef_torch) > 0).to(dtype)
        denom = torch.maximum(torch.tensor(1.0, dtype=dtype, device=device), torch.tensor(float(n), dtype=dtype, device=device) - s_hat)
        sigma2 = torch.sum(resid_lasso ** 2) / denom

        # Node-wise Lasso for M matrix estimation
        lam_nw = float(np.sqrt(2.0 * np.log(max(p, 2)) / n))
        alpha_nw = np.asarray([lam_nw], dtype=np.float64)
        tiny = 1e-30
        zero = 0.0
        one = 1.0

        # Caching for M matrix
        X_sample = X_torch[: min(24, n), : min(24, p)].cpu().numpy()
        m_cache_key = _debiased_m_key_from_sample(
            n=n,
            p=p,
            dtype_name=str(dtype),
            sample_block=X_sample,
            lam_nw=lam_nw,
            tol=float(self.tol),
        )
        M_cached = _debiased_m_cache_get(m_cache_key)

        if M_cached is not None:
            M = torch.from_numpy(M_cached).to(dtype).to(device)
        else:
            M = torch.zeros((p, p), dtype=dtype, device=device)
            XtX_full = X_torch.T @ X_torch
            Sigma_diag = torch.diag(Sigma_hat)

            # Batch node-wise problems for efficiency
            try:
                # Estimate available GPU memory for batching
                if torch.cuda.is_available():
                    free_mem = torch.cuda.mem_get_info(device)[0]
                    bytes_per_fold = max(8, (p - 1) * (p - 1) * 8 * 2)
                    chunk_size = int(max(4, min(64, free_mem // max(bytes_per_fold, 1))))
                else:
                    chunk_size = 16
            except Exception:
                chunk_size = 16
            chunk_size = max(4, min(int(p), chunk_size))

            for j0 in range(0, p, chunk_size):
                j1 = min(p, j0 + chunk_size)
                bsz = j1 - j0
                j_batch = torch.arange(j0, j1, dtype=torch.int32, device=device)

                # Build "all except j" column index matrix
                base = torch.arange(p - 1, dtype=torch.int32, device=device).reshape(1, -1)
                cols_batch = base + (base >= j_batch.reshape(-1, 1))

                # Gather batched Gram/Xty blocks
                XtX_batch = XtX_full[
                    cols_batch[:, :, None],
                    cols_batch[:, None, :],
                ]
                Xty_batch = XtX_full[cols_batch, j_batch.reshape(-1, 1)].reshape(bsz, p - 1)

                # Solve node-wise Lasso problems
                coefs_batch_desc, _ = self._solve_lasso_path_torch_fista_multi_fold_from_gram(
                    XtX_batch,
                    Xty_batch,
                    n_samples_vec=np.full((bsz,), float(n), dtype=np.float64),
                    alphas_desc=alpha_nw,
                    max_iter=500,
                    tol=1e-5,
                    stopping="coef_delta",
                    lipschitz_L=None,
                    check_every=8,
                )
                gamma_batch = torch.from_numpy(np.asarray(coefs_batch_desc[:, 0, :], dtype=np.float64)).to(dtype).to(device)

                # C_j = Sigma_jj - Sigma_{j,-j} gamma_j
                sigma_j_cols = Sigma_hat[j_batch[:, None], cols_batch]
                C_batch = Sigma_diag[j_batch] - torch.sum(sigma_j_cols * gamma_batch, dim=1)

                small_c = torch.abs(C_batch) < tiny
                inv_c = torch.where(small_c, torch.tensor(zero, dtype=dtype, device=device), torch.tensor(one, dtype=dtype, device=device) / C_batch)
                M[j_batch, j_batch] = torch.where(small_c, torch.tensor(one, dtype=dtype, device=device), inv_c)
                M[j_batch[:, None], cols_batch] = -gamma_batch * inv_c.reshape(-1, 1)

                # Cleanup
                del XtX_batch
                del Xty_batch
                del coefs_batch_desc
                del gamma_batch
                del sigma_j_cols

            _debiased_m_cache_put(m_cache_key, M.cpu().numpy())

        # Compute full residual
        if self.fit_intercept:
            y_pred = X_torch @ coef_torch + torch.tensor(self.intercept_, dtype=dtype, device=device)
        else:
            y_pred = X_torch @ coef_torch
        resid_full = y_torch - y_pred

        # Debiased estimate: theta_db = coef + M @ X' @ resid / n
        theta_db = coef_torch + (M @ X_torch.T @ resid_full) / n

        # Variance estimation: V = M @ Sigma_hat @ M'
        V = M @ Sigma_hat @ M.T
        se = torch.sqrt(sigma2 * torch.diag(V) / n)

        # z-statistics and p-values
        z_stats = theta_db / (se + 1e-30)
        pvalues = torch.minimum(torch.tensor(1.0, dtype=dtype, device=device), 2.0 * norm.sf(torch.abs(z_stats)))

        # Confidence intervals
        alpha_ci = 0.05
        z_crit = norm.ppf(1.0 - alpha_ci / 2.0)
        ci = torch.stack([theta_db - z_crit * se, theta_db + z_crit * se], dim=1)

        # Handle intercept
        if self.fit_intercept:
            X_full = torch.cat([torch.ones((n, 1), dtype=dtype, device=device), X_torch], dim=1)
            XtX_full = X_full.T @ X_full
            try:
                XtX_inv = torch.linalg.inv(XtX_full)
            except Exception:
                XtX_inv = torch.linalg.pinv(XtX_full)
            se_intercept = torch.sqrt(sigma2 * XtX_inv[0, 0])
            intercept_torch = torch.tensor(self.intercept_, dtype=dtype, device=device)
            z_intercept = intercept_torch / (se_intercept + 1e-30)
            p_intercept = torch.minimum(torch.tensor(1.0, dtype=dtype, device=device), 2.0 * norm.sf(torch.abs(z_intercept).reshape(1)))
            ci_intercept = torch.stack([
                intercept_torch - z_crit * se_intercept,
                intercept_torch + z_crit * se_intercept,
            ]).reshape(1, 2)

            bse_torch = torch.cat([se_intercept.reshape(1), se])
            tvalues_torch = torch.cat([z_intercept.reshape(1), z_stats])
            pvalues_torch = torch.cat([p_intercept.reshape(1), pvalues])
            conf_int_torch = torch.cat([ci_intercept, ci], dim=0)
            params_torch = torch.cat([intercept_torch.reshape(1), theta_db])
        else:
            bse_torch = se
            tvalues_torch = z_stats
            pvalues_torch = pvalues
            conf_int_torch = ci
            params_torch = theta_db

        # Transfer to CPU
        self._bse = bse_torch.cpu().numpy()
        self._tvalues = tvalues_torch.cpu().numpy()
        self._pvalues = pvalues_torch.cpu().numpy()
        self._conf_int = conf_int_torch.cpu().numpy()
        self._params = params_torch.cpu().numpy()

        # Store M matrix for simultaneous inference
        self._debiased_M_cpu = M.cpu().numpy()

        # Simultaneous inference (max-|Z| bootstrap)
        if self.enable_simultaneous_inference:
            self._compute_simultaneous_inference_torch(
                params_torch, bse_torch, se, M, X_torch, resid_full, n
            )

    def _compute_simultaneous_inference_torch(
        self, params_torch, bse_torch, se_feat_torch, M_torch, X_torch, resid_full_torch, n
    ):
        """Torch GPU implementation of simultaneous inference via max-|Z| bootstrap."""
        import torch

        # Get target indices
        param_target_idx_np = self._get_simultaneous_target_indices(int(params_torch.shape[0]))
        param_target_idx_torch = torch.as_tensor(param_target_idx_np, dtype=torch.int32, device=params_torch.device)

        if param_target_idx_torch.size == 0:
            raise RuntimeError("No coefficients selected for simultaneous inference target set.")

        feature_offset = 1 if self.fit_intercept else 0
        feature_target_torch = param_target_idx_torch - feature_offset
        feature_target_torch = feature_target_torch[feature_target_torch >= 0]

        if feature_target_torch.size == 0:
            raise RuntimeError("No feature coefficients selected for simultaneous inference target set.")

        se_target_torch = torch.index_select(se_feat_torch, 0, feature_target_torch)
        M_target = torch.index_select(M_torch, 0, feature_target_torch)

        B = int(self.simultaneous_n_bootstrap)
        if self.simultaneous_random_state is not None:
            torch.manual_seed(self.simultaneous_random_state)

        # Bootstrap in chunks to manage memory
        try:
            # Try one-shot computation
            xi = torch.randn((B, n), dtype=torch.float64, device=X_torch.device)
            weighted = xi * resid_full_torch.reshape(1, -1)
            score_target = (weighted @ X_torch) @ M_target.T / float(max(n, 1))
            z_star_target = score_target / (se_target_torch.reshape(1, -1) + 1e-30)
            max_stats_torch = torch.max(torch.abs(z_star_target), dim=1)[0]
        except Exception:
            # Fallback to chunked computation
            max_stats_torch = torch.empty((B,), dtype=torch.float64, device=X_torch.device)
            chunk = min(B, 64)
            filled = 0
            while filled < B:
                bsz = min(chunk, B - filled)
                xi = torch.randn((bsz, n), dtype=torch.float64, device=X_torch.device)
                weighted = xi * resid_full_torch.reshape(1, -1)
                score_target = (weighted @ X_torch) @ M_target.T / float(max(n, 1))
                z_star_target = score_target / (se_target_torch.reshape(1, -1) + 1e-30)
                max_stats_torch[filled : filled + bsz] = torch.max(torch.abs(z_star_target), dim=1)[0]
                filled += bsz

        # Compute critical value
        critical_torch = torch.quantile(max_stats_torch, 1.0 - float(self.simultaneous_alpha))

        # Build simultaneous confidence intervals
        conf_sim_torch = conf_int_torch.clone()
        lower_torch = torch.index_select(params_torch, 0, param_target_idx_torch) - critical_torch * torch.index_select(bse_torch, 0, param_target_idx_torch)
        upper_torch = torch.index_select(params_torch, 0, param_target_idx_torch) + critical_torch * torch.index_select(bse_torch, 0, param_target_idx_torch)
        conf_sim_torch[param_target_idx_torch, 0] = lower_torch
        conf_sim_torch[param_target_idx_torch, 1] = upper_torch

        # Store results
        target_mask = np.zeros(int(params_torch.shape[0]), dtype=bool)
        target_mask[param_target_idx_np] = True
        self._conf_int_simultaneous = conf_sim_torch.cpu().numpy()
        self._simultaneous_enabled = True
        self._simultaneous_method = self.simultaneous_method
        self._simultaneous_alpha = float(self.simultaneous_alpha)
        self._simultaneous_n_bootstrap = B
        self._simultaneous_critical_value = float(critical_torch.cpu().numpy())
        self._simultaneous_target_mask = target_mask

    def _soft_threshold_torch(self, x, gamma):
        """Soft thresholding operator for Torch tensors."""
        import torch
        return torch.sign(x) * torch.maximum(torch.abs(x) - gamma, torch.tensor(0.0, dtype=x.dtype, device=x.device))

    def _fit_torch(self, X, y, sample_weight=None):
        """Fit using Torch GPU with FISTA solver."""
        import torch
        from .._gpu_utils_torch import compute_r2_torch

        if self.solver not in ("fista", "admm"):
            raise ValueError("Torch backend currently only supports 'fista' solver")

        # For now, only FISTA is implemented for Torch backend
        if self.solver == "admm":
            raise NotImplementedError("ADMM solver not yet implemented for Torch backend")

        n_samples, n_features = X.shape
        self._nobs = n_samples

        # Ensure Torch tensors on GPU
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to('cuda')
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to('cuda')
        if y.dtype != torch.float64:
            y = y.to(torch.float64)
        if X.dtype != torch.float64:
            X = X.to(torch.float64)

        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                sample_weight = torch.from_numpy(sample_weight).to('cuda')
            sqrt_sw = torch.sqrt(sample_weight)
            X = X * sqrt_sw[:, None]
            y = y * sqrt_sw

        # Ensure vector y on GPU
        y = y.reshape(-1)

        # Center for intercept
        if self.fit_intercept:
            X_mean = torch.mean(X, dim=0)
            y_mean = torch.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = torch.tensor(0.0, dtype=X.dtype, device=X.device)
            y_centered = y

        # Precompute XtX / Xty for FISTA gradient
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Lipschitz constant L
        if self.lipschitz_L is not None:
            L = torch.tensor(float(self.lipschitz_L), dtype=X.dtype, device=X.device)
        else:
            L_frob = torch.sum(X_centered ** 2) / n_samples
            try:
                eigvals = torch.linalg.eigvalsh(XtX)
                L = eigvals[-1] / n_samples
            except Exception:
                L = L_frob

        if L <= 0:
            coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            self.n_iter_ = 0
        else:
            step = 1.0 / L
            thresh = self.alpha * step

            # FISTA variables
            coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
            y_k = coef.clone()
            t_k = torch.tensor(1.0, dtype=X.dtype, device=X.device)

            for iteration in range(self.max_iter):
                coef_old = coef.clone()

                # Gradient at y_k
                grad = (XtX @ y_k - Xty) / n_samples

                # Prox step for L1
                coef = self._soft_threshold_torch(y_k - step * grad, thresh)

                # Momentum update
                t_new = (1.0 + torch.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
                beta = (t_k - 1.0) / t_new
                y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                # Convergence test
                if self.stopping == "kkt":
                    grad_sse = (XtX @ coef - Xty) / n_samples
                    violation = torch.max(torch.maximum(torch.abs(grad_sse) - self.alpha, torch.tensor(0.0, dtype=X.dtype, device=X.device)))
                    if violation < self.tol:
                        self.n_iter_ = iteration + 1
                        break
                else:
                    if torch.sum(torch.abs(coef - coef_old)) < self.tol:
                        self.n_iter_ = iteration + 1
                        break
            else:
                self.n_iter_ = self.max_iter

        # Build full coefficients
        if self.fit_intercept:
            intercept_torch = y_mean - X_mean @ coef
            coef_full = torch.cat([intercept_torch.reshape(1), coef])
        else:
            coef_full = coef

        # Transfer coefficients to CPU
        coef_full_np = coef_full.cpu().numpy()

        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np

        df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        self._df_resid = df_resid

        # Inference/diagnostics
        if self.compute_inference:
            if self.fit_intercept:
                X_design = torch.cat([torch.ones((n_samples, 1), dtype=X.dtype, device=X.device), X], dim=1)
            else:
                X_design = X

            y_pred = X_design @ coef_full
            resid = y - y_pred

            if df_resid > 0:
                scale = torch.sum(resid ** 2) / df_resid
                self._scale = float(scale.cpu().numpy()) if not torch.isnan(scale) else np.nan
            else:
                self._scale = np.nan
                scale = torch.tensor(np.nan, dtype=X.dtype, device=X.device)

            if self.inference_method == "gpu_ols_inference":
                # Compute inference fully on GPU
                XtX_inf = X_design.T @ X_design
                try:
                    XtX_inv = torch.linalg.inv(XtX_inf)
                except Exception:
                    XtX_inv = torch.linalg.pinv(XtX_inf)

                bse_gpu = torch.sqrt(scale * torch.diag(XtX_inv))
                params_gpu = coef_full
                tvalues_gpu = params_gpu / (bse_gpu + 1e-30)

                from ..inference._distributions_torch import t as t_dist
                pvalues_gpu = torch.minimum(torch.tensor(1.0, device=X.device), 2.0 * t_dist.sf(torch.abs(tvalues_gpu), df=df_resid, device=X.device))

                alpha = 0.05
                t_crit_gpu = t_dist.ppf(1.0 - alpha / 2.0, df=df_resid, device=X.device)
                margin_gpu = t_crit_gpu * bse_gpu
                conf_int_gpu = torch.stack([params_gpu - margin_gpu, params_gpu + margin_gpu], dim=1)

                # Transfer to CPU
                self._bse = bse_gpu.cpu().numpy()
                self._tvalues = tvalues_gpu.cpu().numpy()
                self._pvalues = pvalues_gpu.cpu().numpy()
                self._conf_int = conf_int_gpu.cpu().numpy()

                # R^2
                y_mean_gpu = torch.mean(y)
                ss_tot = torch.sum((y - y_mean_gpu) ** 2)
                ss_res = torch.sum(resid ** 2)
                self._rsquared_gpu = float((1 - ss_res / ss_tot).cpu().numpy()) if ss_tot > 0 else 0.0

                self._resid = None
                self._X_design = None
            elif self.inference_method == "debiased":
                # Debiased Lasso inference on Torch GPU
                self._compute_inference_debiased_torch(X, y, coef)

                # R^2 computation
                y_mean_gpu = torch.mean(y)
                ss_tot = torch.sum((y - y_mean_gpu) ** 2)
                ss_res = torch.sum(resid ** 2)
                self._rsquared_gpu = float((1 - ss_res / ss_tot).cpu().numpy()) if ss_tot > 0 else 0.0

                self._resid = None
                self._X_design = None
            else:
                # Transfer residuals and design to CPU
                self._resid = resid.cpu().numpy()
                self._X_design = X_design.cpu().numpy()
        else:
            self._scale = np.nan
            self._resid = None
            self._X_design = None
            self._rsquared_gpu = None

        # Cleanup
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
            del X_centered
        except Exception:
            pass
        try:
            del y_centered
        except Exception:
            pass
        try:
            del y_pred
        except Exception:
            pass
        try:
            del coef_full
        except Exception:
            pass
        self._cleanup_torch_memory()

    def _fit_gpu_admm(self, X, y, sample_weight=None):
        """Fit using GPU with ADMM solver.

        Objective matches sklearn:
          (1/(2n)) * ||y - Xw||^2 + alpha * ||w||_1
        """
        import cupy as cp
        import cupyx.scipy.linalg as cpx_linalg

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

        # Ensure vector y on GPU
        y = y.reshape(-1)

        # Center for intercept
        if self.fit_intercept:
            X_mean = cp.mean(X, axis=0)
            y_mean = cp.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_mean = cp.array(0.0, dtype=X.dtype)
            y_centered = y

        # ADMM variables for constraint w=z
        coef = cp.zeros(n_features, dtype=X.dtype)  # w
        z = cp.zeros(n_features, dtype=X.dtype)  # z
        u = cp.zeros(n_features, dtype=X.dtype)  # scaled dual

        # Precompute XtX and Xty
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # w-update solves:
        # (XtX + rho*n*I) w = Xty + rho*n * (z - u)
        rho = float(self.admm_rho)
        if rho <= 0:
            raise ValueError("admm_rho must be > 0")

        lhs = XtX + (rho * n_samples) * cp.eye(n_features, dtype=X.dtype)

        # Pre-factorize once
        Lmat = cp.linalg.cholesky(lhs)

        def solve_w(rhs):
            # Solve Lmat @ (Lmat.T @ w) = rhs
            tmp = cpx_linalg.solve_triangular(Lmat, rhs, lower=True)
            return cpx_linalg.solve_triangular(Lmat.T, tmp, lower=False)

        thresh = self.alpha / rho

        for iteration in range(self.max_iter):
            coef_old = coef

            rhs = Xty + (rho * n_samples) * (z - u)
            coef = solve_w(rhs)

            # z-update (prox of l1)
            z_old = z
            z = self._soft_threshold_cupy(coef + u, thresh)

            # dual update
            u = u + (coef - z)

            # Convergence test
            if self.stopping == "kkt":
                grad_sse = (XtX @ coef - Xty) / n_samples
                violation = cp.max(cp.maximum(cp.abs(grad_sse) - self.alpha, 0.0))
                if violation < self.tol:
                    self.n_iter_ = iteration + 1
                    break
            else:
                # Legacy stopping: coefficient delta
                if cp.sum(cp.abs(coef - coef_old)) < self.tol:
                    self.n_iter_ = iteration + 1
                    break
            z = z  # keep for clarity
        else:
            self.n_iter_ = self.max_iter

        # Build full coefficients and (optionally) residuals for inference/R^2
        if self.fit_intercept:
            intercept_gpu = y_mean - X_mean @ coef
            coef_full = cp.concatenate([intercept_gpu.reshape(1), coef])
            X_design = cp.concatenate([cp.ones((n_samples, 1), dtype=X.dtype), X], axis=1)
        else:
            coef_full = coef
            X_design = X

        coef_full_np = coef_full.get()
        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np

        df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))
        self._df_resid = df_resid

        if self.compute_inference:
            y_pred = X_design @ coef_full
            resid = y - y_pred
            if df_resid > 0:
                scale = cp.sum(resid ** 2) / df_resid
                self._scale = float(scale.get()) if not cp.isnan(scale) else np.nan
            else:
                self._scale = np.nan
                scale = cp.nan

            if self.inference_method == "gpu_ols_inference":
                # Keep the inference path on GPU and transfer only small vectors.
                XtX_inf = X_design.T @ X_design
                try:
                    XtX_inv = cp.linalg.inv(XtX_inf)
                except Exception:
                    XtX_inv = cp.linalg.pinv(XtX_inf)

                bse_gpu = cp.sqrt(scale * cp.diag(XtX_inv))
                params_gpu = coef_full
                tvalues_gpu = params_gpu / (bse_gpu + 1e-30)
                pvalues_gpu = cp.minimum(1.0, 2.0 * t.sf(cp.abs(tvalues_gpu), df=df_resid))

                alpha = 0.05
                t_crit_gpu = t.ppf(1.0 - alpha / 2.0, df=df_resid)
                margin_gpu = t_crit_gpu * bse_gpu
                conf_int_gpu = cp.stack([params_gpu - margin_gpu, params_gpu + margin_gpu], axis=1)

                self._bse = cp.asnumpy(bse_gpu)
                self._tvalues = cp.asnumpy(tvalues_gpu)
                self._pvalues = cp.asnumpy(pvalues_gpu)
                self._conf_int = cp.asnumpy(conf_int_gpu)

                y_mean_gpu = cp.mean(y)
                ss_tot = cp.sum((y - y_mean_gpu) ** 2)
                ss_res = cp.sum(resid ** 2)
                self._rsquared_gpu = float(cp.asnumpy(1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0

                self._resid = None
                self._X_design = None
            elif self.inference_method == "debiased":
                self._compute_inference_debiased_gpu(X, y, coef)

                y_mean_gpu = cp.mean(y)
                ss_tot = cp.sum((y - y_mean_gpu) ** 2)
                ss_res = cp.sum(resid ** 2)
                self._rsquared_gpu = float(cp.asnumpy(1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0

                self._resid = None
                self._X_design = None
            else:
                # CPU-side inference path.
                self._resid = resid.get()
                self._X_design = X_design.get()
        else:
            self._scale = np.nan
            self._resid = None
            self._X_design = None
            self._rsquared_gpu = None

        # Drop large temporaries early (before optional pool cleanup).
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
            del X_centered
        except Exception:
            pass
        try:
            del y_centered
        except Exception:
            pass
        try:
            del y_pred
        except Exception:
            pass
        try:
            del coef_full
        except Exception:
            pass
        try:
            del lhs
        except Exception:
            pass
        try:
            del Lmat
        except Exception:
            pass
        self._cleanup_cuda_memory()

    def _compute_inference(self):
        """Compute standard errors, t-stats, p-values."""
        if self.inference_method == "bootstrap":
            return self._compute_inference_bootstrap()
        if self.inference_method == "debiased":
            return self._compute_inference_debiased()
        if self.inference_method == "gpu_ols_inference":
            # Inference already computed on GPU in _fit_gpu().
            return
        if self._X_design is None or self._scale is None or np.isnan(self._scale):
            return

        X = self._X_design

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
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

    def _compute_inference_bootstrap(self) -> None:
        """
        Bootstrap inference for Lasso via residual resampling.

        Notes
        -----
        This is more robust than the naive OLS-based inference, but it is still
        not full "post-selection inference" for Lasso.
        """
        if self._X_design is None or self._resid is None or self._y is None:
            return

        if self.n_bootstrap <= 0:
            return

        rng = np.random.default_rng(self.bootstrap_random_state)
        X = self._X_design
        y = self._y
        y_pred = y - self._resid
        resid = self._resid

        params_dim = self._params.shape[0]
        boot_params = np.zeros((self.n_bootstrap, params_dim), dtype=float)

        # Precompute Lipschitz constant if needed for CPU FISTA.
        lipschitz_L = self.lipschitz_L
        if self.cpu_solver == "fista" and lipschitz_L is None:
            # L = lambda_max(Xc^T Xc) / n for centered design
            X_nopen = X[:, 1:] if self.fit_intercept else X
            X_centered = X_nopen - X_nopen.mean(axis=0, keepdims=True)
            XtX = X_centered.T @ X_centered
            eigvals = np.linalg.eigvalsh(XtX)
            lipschitz_L = float(eigvals[-1] / X_nopen.shape[0])

        for b in range(self.n_bootstrap):
            eps_star = rng.choice(resid, size=resid.shape[0], replace=True)
            y_star = y_pred + eps_star

            refit = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                stopping=self.stopping,
                inference_method="cpu_ols_inference",
                n_bootstrap=0,
                bootstrap_random_state=None,
                device="cpu",
                compute_inference=False,
                solver=self.solver,
                cpu_solver=self.cpu_solver,
                lipschitz_L=lipschitz_L,
                admm_rho=self.admm_rho,
            )

            # Refit expects raw X (without intercept column).
            if self.fit_intercept:
                X_refit = X[:, 1:]
            else:
                X_refit = X

            refit.fit(X_refit, y_star)
            boot_params[b, :] = refit._params

        # Standard errors and bootstrap-based p-values/CI.
        self._bse = np.std(boot_params, axis=0, ddof=1)
        self._params = np.asarray(self._params, dtype=float)

        # Two-sided p-values using sign-change probability.
        pvalues = np.zeros(params_dim, dtype=float)
        for i in range(params_dim):
            coef_b = boot_params[:, i]
            p_lower = np.mean(coef_b <= 0.0)
            p_upper = np.mean(coef_b >= 0.0)
            p = 2.0 * min(p_lower, p_upper)
            pvalues[i] = min(p, 1.0)
        self._pvalues = pvalues

        # Percentile confidence intervals.
        lower_q = (0.05 / 2.0) * 1.0
        upper_q = 1.0 - (0.05 / 2.0) * 1.0
        self._conf_int = np.column_stack([
            np.quantile(boot_params, lower_q, axis=0),
            np.quantile(boot_params, upper_q, axis=0),
        ])

        # t-stats (approx) from bootstrap SE.
        self._tvalues = self._params / (self._bse + 1e-30)

    def _compute_inference_debiased(self) -> None:
        """Debiased Lasso inference (Javanmard-Montanari / Zhang-Zhang).

        Constructs the decorrelation matrix M via node-wise Lasso,
        then computes the debiased estimator, standard errors,
        z-statistics, p-values, and per-coefficient (marginal)
        confidence intervals.
        """
        if self._X_design is None or self._resid is None:
            return

        if self.fit_intercept:
            X = self._X_design[:, 1:]
        else:
            X = self._X_design

        n, p = X.shape
        coef = self.coef_.copy()

        Sigma_hat = X.T @ X / n
        resid_lasso = self._resid

        # --- noise variance: sigma^2 = RSS / (n - s_hat) ---
        s_hat = int(np.sum(np.abs(coef) > 0))
        denom = max(n - s_hat, 1)
        sigma2 = np.sum(resid_lasso ** 2) / denom

        # --- node-wise Lasso to build M (p x p), with cross-fit cache ---
        lam_nw = np.sqrt(2.0 * np.log(max(p, 2)) / n)
        m_cache_key = _debiased_m_key_from_numpy_design(
            X,
            n=n,
            p=p,
            lam_nw=lam_nw,
            tol=float(self.tol),
        )
        M_cached = _debiased_m_cache_get(m_cache_key)
        if M_cached is not None:
            M = np.asarray(M_cached, dtype=X.dtype)
        else:
            M = np.zeros((p, p), dtype=X.dtype)
            for j in range(p):
                cols = np.concatenate([np.arange(0, j), np.arange(j + 1, p)])
                X_minus_j = X[:, cols]
                x_j = X[:, j]

                nw = Lasso(
                    alpha=lam_nw,
                    fit_intercept=False,
                    max_iter=500,
                    tol=1e-5,
                    device="cpu",
                    cpu_solver="fista",
                    compute_inference=False,
                )
                nw.fit(X_minus_j, x_j)
                gamma_j = nw.coef_

                z_j = x_j - X_minus_j @ gamma_j
                C_j = z_j @ x_j / n

                if abs(C_j) < 1e-30:
                    M[j, j] = 1.0
                    continue

                M[j, j] = 1.0 / C_j
                M[j, cols] = -gamma_j / C_j
            _debiased_m_cache_put(m_cache_key, np.asarray(M, dtype=np.float64))

        # --- debiased estimates ---
        theta_db = coef + (M @ X.T @ resid_lasso) / n
        self._debiased_M_cpu = M

        # --- covariance and standard errors ---
        V = M @ Sigma_hat @ M.T
        se = np.sqrt(sigma2 * np.diag(V) / n)

        z_stats = theta_db / (se + 1e-30)
        pvalues = 2.0 * (1.0 - _norm_dist.cdf(np.abs(z_stats)))

        alpha_ci = 0.05
        z_crit = _norm_dist.ppf(1.0 - alpha_ci / 2.0)
        ci = np.column_stack([theta_db - z_crit * se, theta_db + z_crit * se])

        if self.fit_intercept:
            # Intercept SE via OLS formula: sigma * sqrt([1/n + xbar' (X'X)^-1 xbar])
            X_full = self._X_design
            try:
                XtX_inv = np.linalg.inv(X_full.T @ X_full)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(X_full.T @ X_full)
            se_intercept = np.sqrt(sigma2 * XtX_inv[0, 0])
            z_intercept = self.intercept_ / (se_intercept + 1e-30)
            p_intercept = 2.0 * (1.0 - _norm_dist.cdf(np.abs(z_intercept)))
            ci_intercept = np.array([
                self.intercept_ - z_crit * se_intercept,
                self.intercept_ + z_crit * se_intercept,
            ])

            self._bse = np.concatenate([[se_intercept], se])
            self._tvalues = np.concatenate([[z_intercept], z_stats])
            self._pvalues = np.concatenate([[p_intercept], pvalues])
            self._conf_int = np.vstack([ci_intercept[np.newaxis, :], ci])
            self._params = np.concatenate([[self.intercept_], theta_db])
        else:
            self._bse = se
            self._tvalues = z_stats
            self._pvalues = pvalues
            self._conf_int = ci
            self._params = theta_db

    def _compute_inference_debiased_gpu(self, X_gpu, y_gpu, coef_gpu):
        """GPU path for debiased Lasso inference.

        Parameters
        ----------
        X_gpu : cupy.ndarray, shape (n, p)
            Raw feature matrix on GPU (no intercept column).
        y_gpu : cupy.ndarray, shape (n,)
            Response on GPU.
        coef_gpu : cupy.ndarray, shape (p,)
            Lasso coefficients on GPU (no intercept).
        """
        import cupy as cp

        n, p = X_gpu.shape
        Sigma_hat = X_gpu.T @ X_gpu / n

        resid_lasso = y_gpu - X_gpu @ coef_gpu
        if self.fit_intercept:
            resid_lasso = resid_lasso - cp.mean(y_gpu) + cp.mean(X_gpu, axis=0) @ coef_gpu

        s_hat_gpu = cp.sum(cp.abs(coef_gpu) > 0).astype(cp.float64)
        denom_gpu = cp.maximum(1.0, float(n) - s_hat_gpu)
        sigma2_gpu = cp.asarray(cp.sum(resid_lasso ** 2) / denom_gpu, dtype=cp.float64)

        lam_nw = float(np.sqrt(2.0 * np.log(max(p, 2)) / n))
        alpha_nw = np.asarray([lam_nw], dtype=np.float64)
        tiny = X_gpu.dtype.type(1e-30)
        zero = X_gpu.dtype.type(0.0)
        one = X_gpu.dtype.type(1.0)

        # Keep node-wise Lasso solves on GPU to avoid per-feature host round-trips.
        x_hasher = hashlib.blake2b(digest_size=32)
        x_hasher.update(np.asarray([int(n), int(p)], dtype=np.int64).tobytes())
        x_hasher.update(str(X_gpu.dtype).encode("utf-8"))
        x_hasher.update(np.asarray([float(lam_nw), float(self.tol)], dtype=np.float64).tobytes())
        row_chunk = max(1, min(int(n), _LASSO_DEBIASED_M_GPU_HASH_ROW_CHUNK))
        for start in range(0, int(n), row_chunk):
            stop = min(int(n), start + row_chunk)
            x_chunk = cp.asnumpy(X_gpu[start:stop])
            x_hasher.update(x_chunk.tobytes())
        m_cache_key = x_hasher.hexdigest()
        M_cached = _debiased_m_cache_get(m_cache_key)
        if M_cached is not None:
            M = cp.asarray(M_cached, dtype=X_gpu.dtype)
        else:
            M = cp.zeros((p, p), dtype=X_gpu.dtype)
            # Reuse full Gram to avoid repeated X_minus_j.T @ X_minus_j products.
            XtX_full = X_gpu.T @ X_gpu
            Sigma_diag = cp.diag(Sigma_hat)
            n_samp_vec_dtype = np.float64

            # Batch node-wise problems so GPU can process many j's together.
            try:
                free_mem, _ = cp.cuda.Device().mem_info
                bytes_per_fold = int(max(8, (p - 1) * (p - 1) * 8 * 2))
                chunk_size = int(max(4, min(64, free_mem // max(bytes_per_fold, 1))))
            except Exception:
                chunk_size = 16
            chunk_size = max(4, min(int(p), chunk_size))

            for j0 in range(0, p, chunk_size):
                j1 = min(p, j0 + chunk_size)
                bsz = j1 - j0
                j_batch = cp.arange(j0, j1, dtype=cp.int32)
                if int(j_batch.size) == 0:
                    continue

                # Build per-j "all except j" column index matrix of shape (bsz, p-1).
                base = cp.arange(p - 1, dtype=cp.int32).reshape(1, -1)
                cols_batch = base + (base >= j_batch.reshape(-1, 1))

                # Gather batched Gram/Xty blocks.
                XtX_batch = XtX_full[
                    cols_batch[:, :, cp.newaxis],
                    cols_batch[:, cp.newaxis, :],
                ]
                Xty_batch = XtX_full[cols_batch, j_batch.reshape(-1, 1)].reshape(bsz, p - 1)

                coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram(
                    XtX_batch,
                    Xty_batch,
                    n_samples_vec=np.full((bsz,), float(n), dtype=n_samp_vec_dtype),
                    alphas_desc=alpha_nw,
                    max_iter=500,
                    tol=1e-5,
                    stopping="coef_delta",
                    lipschitz_L=None,
                    check_every=8,
                )
                gamma_batch = cp.asarray(coefs_batch_desc[:, 0, :], dtype=X_gpu.dtype)

                # C_j = Sigma_jj - Sigma_{j,-j} gamma_j
                sigma_j_cols = Sigma_hat[j_batch[:, cp.newaxis], cols_batch]
                C_batch = Sigma_diag[j_batch] - cp.sum(sigma_j_cols * gamma_batch, axis=1)

                small_c = cp.abs(C_batch) < tiny
                inv_c = cp.where(small_c, zero, one / C_batch)
                M[j_batch, j_batch] = cp.where(small_c, one, inv_c)
                M[j_batch[:, cp.newaxis], cols_batch] = -gamma_batch * inv_c.reshape(-1, 1)

                del XtX_batch
                del Xty_batch
                del coefs_batch_desc
                del gamma_batch
                del sigma_j_cols
            _debiased_m_cache_put(m_cache_key, cp.asnumpy(M))

        # Recompute full residual from the original fit
        if self.fit_intercept:
            y_pred = X_gpu @ coef_gpu + cp.asarray(self.intercept_, dtype=X_gpu.dtype)
        else:
            y_pred = X_gpu @ coef_gpu
        resid_full = y_gpu - y_pred

        theta_db = coef_gpu + (M @ X_gpu.T @ resid_full) / n

        V = M @ Sigma_hat @ M.T
        se = cp.sqrt(sigma2_gpu * cp.diag(V) / n)

        z_stats = theta_db / (se + 1e-30)
        pvalues = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(z_stats)))

        alpha_ci = 0.05
        z_crit = norm.ppf(1.0 - alpha_ci / 2.0)
        ci = cp.stack([theta_db - z_crit * se, theta_db + z_crit * se], axis=1)

        if self.fit_intercept:
            X_full = cp.concatenate(
                [cp.ones((n, 1), dtype=X_gpu.dtype), X_gpu], axis=1
            )
            XtX_full = X_full.T @ X_full
            try:
                XtX_inv = cp.linalg.inv(XtX_full)
            except Exception:
                XtX_inv = cp.linalg.pinv(XtX_full)
            se_intercept = cp.sqrt(sigma2_gpu * XtX_inv[0, 0])
            intercept_gpu = cp.asarray(self.intercept_, dtype=cp.float64)
            z_intercept = intercept_gpu / (se_intercept + 1e-30)
            p_intercept = cp.minimum(1.0, 2.0 * norm.sf(cp.abs(z_intercept).reshape(1)))
            ci_intercept = cp.stack([
                intercept_gpu - z_crit * se_intercept,
                intercept_gpu + z_crit * se_intercept,
            ]).reshape(1, 2)

            bse_gpu = cp.concatenate([se_intercept.reshape(1), se])
            tvalues_gpu = cp.concatenate([z_intercept.reshape(1), z_stats])
            pvalues_gpu = cp.concatenate([p_intercept.reshape(1), pvalues])
            conf_int_gpu = cp.concatenate([ci_intercept, ci], axis=0)
            params_gpu = cp.concatenate([intercept_gpu.reshape(1), theta_db])
        else:
            bse_gpu = se
            tvalues_gpu = z_stats
            pvalues_gpu = pvalues
            conf_int_gpu = ci
            params_gpu = theta_db

        if self.enable_simultaneous_inference:
            # GPU-native simultaneous CI via max-|Z| multiplier bootstrap.
            param_target_idx_np = self._get_simultaneous_target_indices(
                int(params_gpu.shape[0])
            )
            param_target_idx_gpu = cp.asarray(param_target_idx_np, dtype=cp.int32)
            if param_target_idx_gpu.size == 0:
                raise RuntimeError(
                    "No coefficients selected for simultaneous inference target set."
                )

            feature_offset = 1 if self.fit_intercept else 0
            feature_target_gpu = param_target_idx_gpu - feature_offset
            feature_target_gpu = feature_target_gpu[feature_target_gpu >= 0]
            if feature_target_gpu.size == 0:
                raise RuntimeError(
                    "No feature coefficients selected for simultaneous inference target set."
                )

            se_feat_gpu = se
            B = int(self.simultaneous_n_bootstrap)
            rng = cp.random.RandomState(self.simultaneous_random_state)
            se_target_gpu = cp.take(se_feat_gpu, feature_target_gpu)
            M_target = cp.take(M, feature_target_gpu, axis=0)
            # Run bootstrap in one shot when memory allows to reduce kernel-launch overhead.
            try:
                xi = rng.standard_normal(size=(B, n)).astype(cp.float64, copy=False)
                weighted = xi * resid_full.reshape(1, -1)
                score_target = (weighted @ X_gpu) @ M_target.T / float(max(n, 1))
                z_star_target = score_target / (se_target_gpu.reshape(1, -1) + 1e-30)
                max_stats_gpu = cp.max(cp.abs(z_star_target), axis=1)
            except Exception:
                free_mem, _ = cp.cuda.Device().mem_info
                bytes_per_row = max(8 * (3 * n + 2 * p + 64), 8)
                est_chunk = int(max(64, min(4096, free_mem // bytes_per_row)))
                chunk = min(B, max(64, est_chunk))
                max_stats_gpu = cp.empty((B,), dtype=cp.float64)
                filled = 0
                while filled < B:
                    bsz = min(chunk, B - filled)
                    xi = rng.standard_normal(size=(bsz, n)).astype(cp.float64, copy=False)
                    weighted = xi * resid_full.reshape(1, -1)
                    score_target = (weighted @ X_gpu) @ M_target.T / float(max(n, 1))
                    z_star_target = score_target / (se_target_gpu.reshape(1, -1) + 1e-30)
                    max_stats_gpu[filled : filled + bsz] = cp.max(
                        cp.abs(z_star_target), axis=1
                    )
                    filled += bsz

            critical_gpu = cp.quantile(
                max_stats_gpu, 1.0 - float(self.simultaneous_alpha)
            )
            conf_sim_gpu = cp.array(conf_int_gpu, copy=True)
            lower_gpu = cp.take(params_gpu, param_target_idx_gpu) - critical_gpu * cp.take(
                bse_gpu, param_target_idx_gpu
            )
            upper_gpu = cp.take(params_gpu, param_target_idx_gpu) + critical_gpu * cp.take(
                bse_gpu, param_target_idx_gpu
            )
            conf_sim_gpu[param_target_idx_gpu, 0] = lower_gpu
            conf_sim_gpu[param_target_idx_gpu, 1] = upper_gpu

            target_mask = np.zeros(int(params_gpu.shape[0]), dtype=bool)
            target_mask[param_target_idx_np] = True
            self._conf_int_simultaneous = cp.asnumpy(conf_sim_gpu)
            self._simultaneous_enabled = True
            self._simultaneous_method = self.simultaneous_method
            self._simultaneous_alpha = float(self.simultaneous_alpha)
            self._simultaneous_n_bootstrap = B
            self._simultaneous_critical_value = float(cp.asnumpy(critical_gpu))
            self._simultaneous_target_mask = target_mask

        self._bse = cp.asnumpy(bse_gpu)
        self._tvalues = cp.asnumpy(tvalues_gpu)
        self._pvalues = cp.asnumpy(pvalues_gpu)
        self._conf_int = cp.asnumpy(conf_int_gpu)
        self._params = cp.asnumpy(params_gpu)

    def _get_simultaneous_target_indices(self, n_params: int):
        if self.fit_intercept and (not self.simultaneous_include_intercept):
            return np.arange(1, n_params, dtype=int)
        return np.arange(n_params, dtype=int)

    def _compute_simultaneous_inference(self):
        if not self.enable_simultaneous_inference:
            return
        if self._simultaneous_enabled and self._conf_int_simultaneous is not None:
            return
        if self.inference_method != "debiased":
            return
        if self._params is None or self._bse is None or self._conf_int is None:
            return
        if self._X_design is None or self._resid is None:
            raise RuntimeError(
                "Simultaneous debiased inference requires accessible design/residual "
                "state; re-fit with compute_inference=True."
            )
        self._compute_simultaneous_ci_maxz_bootstrap()

    def compute_debiased_inference(self):
        """Explicitly recompute debiased inference for a fitted model."""
        self._check_is_fitted()
        if self.inference_method != "debiased":
            raise ValueError("compute_debiased_inference requires inference_method='debiased'.")
        self._compute_inference()
        return self

    def compute_debiased_inference_(self):
        """Deprecated alias for :meth:`compute_debiased_inference`."""
        warnings.warn(
            "compute_debiased_inference_ is deprecated and will be removed in a future "
            "release; use compute_debiased_inference instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_debiased_inference()

    def compute_simultaneous_inference(self):
        """Explicitly (re)compute simultaneous inference for a fitted model."""
        self._check_is_fitted()
        if not self.enable_simultaneous_inference:
            raise ValueError(
                "compute_simultaneous_inference requires enable_simultaneous_inference=True."
            )
        self._compute_simultaneous_inference()
        return self

    def compute_simultaneous_inference_(self):
        """Deprecated alias for :meth:`compute_simultaneous_inference`."""
        warnings.warn(
            "compute_simultaneous_inference_ is deprecated and will be removed in a "
            "future release; use compute_simultaneous_inference instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_simultaneous_inference()

    def _compute_simultaneous_ci_maxz_bootstrap(self):
        """Compute simultaneous CIs using max-|Z| multiplier bootstrap."""
        # Feature-only design used by debiased estimator.
        if self.fit_intercept:
            X = np.asarray(self._X_design[:, 1:], dtype=float)
        else:
            X = np.asarray(self._X_design, dtype=float)
        resid = np.asarray(self._resid, dtype=float).reshape(-1)
        n, p = X.shape
        if p == 0:
            raise RuntimeError("Simultaneous inference requires at least one feature.")

        # Reuse M from debiased inference when available to avoid duplicate node-wise solves.
        M = self._debiased_M_cpu
        if M is None or M.shape != (p, p):
            lam_nw = np.sqrt(2.0 * np.log(max(p, 2)) / max(n, 1))
            M = np.zeros((p, p), dtype=float)
            for j in range(p):
                cols = np.concatenate([np.arange(0, j), np.arange(j + 1, p)])
                X_minus_j = X[:, cols]
                x_j = X[:, j]
                nw = Lasso(
                    alpha=lam_nw,
                    fit_intercept=False,
                    max_iter=500,
                    tol=1e-5,
                    device="cpu",
                    cpu_solver="fista",
                    compute_inference=False,
                )
                nw.fit(X_minus_j, x_j)
                gamma_j = nw.coef_
                z_j = x_j - X_minus_j @ gamma_j
                c_j = float(z_j @ x_j / max(n, 1))
                if abs(c_j) < 1e-30:
                    M[j, j] = 1.0
                    continue
                M[j, j] = 1.0 / c_j
                M[j, cols] = -gamma_j / c_j
            self._debiased_M_cpu = M

        # Bootstrap the studentized process max_j |Z*_j|.
        param_target_idx = self._get_simultaneous_target_indices(len(self._params))
        feature_target_idx = param_target_idx - (1 if self.fit_intercept else 0)
        feature_target_idx = feature_target_idx[feature_target_idx >= 0]
        if feature_target_idx.size == 0:
            raise RuntimeError(
                "No feature coefficients selected for simultaneous inference target set."
            )

        se_feat = np.asarray(self._bse[(1 if self.fit_intercept else 0):], dtype=float)
        eps = resid
        rng = np.random.default_rng(self.simultaneous_random_state)
        B = int(self.simultaneous_n_bootstrap)
        chunk = min(256, B)
        max_stats = np.empty(B, dtype=float)
        filled = 0
        while filled < B:
            bsz = min(chunk, B - filled)
            xi = rng.standard_normal(size=(bsz, n))
            weighted = xi * eps.reshape(1, -1)
            score = (weighted @ X) @ M.T / float(max(n, 1))
            z_star = score / (se_feat.reshape(1, -1) + 1e-30)
            max_stats[filled:filled + bsz] = np.max(
                np.abs(z_star[:, feature_target_idx]), axis=1
            )
            filled += bsz

        critical = float(np.quantile(max_stats, 1.0 - self.simultaneous_alpha))
        params = np.asarray(self._params, dtype=float)
        bse = np.asarray(self._bse, dtype=float)
        conf_sim = np.array(self._conf_int, copy=True, dtype=float)
        conf_sim[param_target_idx, 0] = params[param_target_idx] - critical * bse[param_target_idx]
        conf_sim[param_target_idx, 1] = params[param_target_idx] + critical * bse[param_target_idx]

        mask = np.zeros(len(params), dtype=bool)
        mask[param_target_idx] = True
        self._conf_int_simultaneous = conf_sim
        self._simultaneous_enabled = True
        self._simultaneous_method = self.simultaneous_method
        self._simultaneous_alpha = float(self.simultaneous_alpha)
        self._simultaneous_n_bootstrap = B
        self._simultaneous_critical_value = critical
        self._simultaneous_target_mask = mask

    @property
    def rsquared(self):
        """R-squared."""
        if self._resid is None:
            # In compute_inference=False GPU mode we may avoid transferring residuals.
            if hasattr(self, "_rsquared_gpu") and self._rsquared_gpu is not None:
                return self._rsquared_gpu
            return None
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
        if r2 is None:
            return None
        k = len(self.coef_)
        return 1 - (1 - r2) * (self._nobs - 1) / self._df_resid

    @property
    def fvalue(self):
        """F-statistic."""
        if self._y is not None and self._resid is not None:
            y_mean = np.mean(self._y)
            ss_tot = np.sum((self._y - y_mean) ** 2)
            ss_res = np.sum(self._resid ** 2)
            ss_reg = ss_tot - ss_res
            k = len(self.coef_)
            if k == 0 or ss_res <= 0:
                return np.inf
            return (ss_reg / k) / (ss_res / self._df_resid)

        # GPU inference mode may skip transferring residual vectors to host.
        r2 = self.rsquared
        if r2 is None:
            return None
        k = len(self.coef_)
        if k <= 0 or self._df_resid is None or self._df_resid <= 0:
            return None
        if r2 >= 1.0:
            return np.inf
        return (r2 / k) / ((1.0 - r2) / self._df_resid)

    @property
    def f_pvalue(self):
        """p-value for F-statistic."""
        k = len(self.coef_)
        if k <= 0 or self._df_resid is None or self._df_resid <= 0:
            return None
        fv = self.fvalue
        if fv is None:
            return None
        if fv == np.inf:
            # An infinite F-statistic corresponds to a perfect-fit / zero-residual
            # case, so the upper-tail probability tends to 0.
            return 0.0
        if fv == np.inf:
            return 0.0
        pval = 1.0 - stats.f.cdf(fv, k, self._df_resid)
        if not np.isfinite(pval):
            return None
        return float(np.clip(pval, 0.0, 1.0))

    @property
    def aic(self):
        """Akaike Information Criterion."""
        if self._nobs is None or np.isnan(self._scale):
            return None
        return -2 * self.llf + 2 * len(self._params)

    @property
    def bic(self):
        """Bayesian Information Criterion."""
        if self._nobs is None or np.isnan(self._scale):
            return None
        n = self._nobs
        k = len(self._params)
        return -2 * self.llf + k * np.log(n)

    @property
    def llf(self):
        """Log-likelihood."""
        if self._nobs is None:
            return None
        n = self._nobs
        if self._resid is not None:
            sigma2_mle = np.sum(self._resid ** 2) / n
        else:
            if self._scale is None or np.isnan(self._scale):
                return None
            if self._df_resid is None or self._df_resid <= 0:
                return None
            sigma2_mle = (self._scale * self._df_resid) / n
        if sigma2_mle <= 0:
            return None
        return -n/2 * np.log(2 * np.pi * sigma2_mle) - n/2

    def summary(self):
        """Print summary table."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if self._bse is None or self._pvalues is None or self._conf_int is None:
            raise RuntimeError(
                "compute_inference=False: inference statistics are not available. "
                "Re-fit with compute_inference=True (default) to use summary()."
            )

        if self.fit_intercept:
            feature_names = ['(Intercept)'] + [f'x{i+1}' for i in range(len(self.coef_))]
        else:
            feature_names = [f'x{i+1}' for i in range(len(self.coef_))]

        is_debiased = self.inference_method == "debiased"
        title = "Debiased Lasso Results" if is_debiased else "Lasso Regression Results"
        stat_label = "z" if is_debiased else "t"
        pval_label = "P>|z|" if is_debiased else "P>|t|"

        def _fmt_stat(value, fmt_spec: str) -> str:
            if value is None:
                return f"{'nan':>15}"
            try:
                value_f = float(value)
            except Exception:
                return f"{'nan':>15}"
            if np.isnan(value_f):
                return f"{'nan':>15}"
            if np.isposinf(value_f):
                return f"{'inf':>15}"
            if np.isneginf(value_f):
                return f"{'-inf':>15}"
            return format(value_f, fmt_spec)

        print("=" * 80)
        if self._inference_cautions:
            print("Notes:")
            for note in self._inference_cautions:
                print(f"- {note}")
            print("=" * 80)
        print(f"                            {title}")
        print(f"                            (alpha = {self.alpha:.4f})")
        print("=" * 80)
        print(f"No. Observations:           {self._nobs:>15}")
        print(f"Degrees of Freedom:         {self._df_resid:>15}")
        print(f"Iterations:                 {self.n_iter_:>15}")
        print(f"R-squared:                  {_fmt_stat(self.rsquared, '>15.4f')}")
        print(f"Adj. R-squared:             {_fmt_stat(self.rsquared_adj, '>15.4f')}")
        print(f"F-statistic:                {_fmt_stat(self.fvalue, '>15.4f')}")
        print(f"Prob (F-statistic):         {_fmt_stat(self.f_pvalue, '>15.4e')}")
        print(f"Log-Likelihood:             {_fmt_stat(self.llf, '>15.4f')}")
        print(f"AIC:                        {_fmt_stat(self.aic, '>15.4f')}")
        print(f"BIC:                        {_fmt_stat(self.bic, '>15.4f')}")
        print("-" * 80)
        print(f"{'':<15} {'coef':>12} {'std err':>12} {stat_label:>10} {pval_label:>10} {'[0.025':>12} {'0.975]':>12}")
        print("-" * 80)

        for i, name in enumerate(feature_names):
            print(f"{name:<15} {self._params[i]:>12.4f} {self._bse[i]:>12.4f} "
                  f"{self._tvalues[i]:>10.3f} {self._pvalues[i]:>10.4f} "
                  f"{self._conf_int[i, 0]:>12.4f} {self._conf_int[i, 1]:>12.4f}")

        if self._simultaneous_enabled and self._conf_int_simultaneous is not None:
            target_txt = (
                "include_intercept=True"
                if (self.fit_intercept and self.simultaneous_include_intercept)
                else "include_intercept=False"
            )
            print("-" * 80)
            print("Simultaneous inference")
            print(f"method:                     {self._simultaneous_method}")
            print(f"alpha:                      {self._simultaneous_alpha:.6f}")
            print(f"n_bootstrap:                {self._simultaneous_n_bootstrap}")
            print(f"critical value (max|Z|):    {self._simultaneous_critical_value:.6f}")
            print(f"target set:                 {target_txt}")

        print("=" * 80)

    def predict(self, X):
        """Predict using the Lasso model."""
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
        """Return R^2 score."""
        y_pred = self._to_numpy(self.predict(X))
        y = self._to_numpy(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _lasso_alpha_heuristic(y_centered: np.ndarray, n_features: int) -> float:
    n_samples = int(y_centered.shape[0])
    if n_samples > 1:
        sigma_hat = float(np.std(y_centered, ddof=1))
    else:
        sigma_hat = float(np.std(y_centered))
    sigma_hat = max(sigma_hat, 1e-8)
    penalty_scale = np.sqrt(2.0 * np.log(max(2, int(n_features))) / max(1, n_samples))
    return float(sigma_hat * penalty_scale)


def _default_lasso_alpha_grid(
    X: np.ndarray,
    y: np.ndarray,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
) -> np.ndarray:
    n_samples = int(X.shape[0])
    corr = np.abs(X.T @ y) / float(max(1, n_samples))
    alpha_max = float(np.max(corr)) if corr.size else 1.0
    alpha_max = max(alpha_max, _lasso_alpha_heuristic(y, n_features=int(X.shape[1])))
    alpha_max = max(alpha_max, 1e-6)

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


def _default_lasso_alpha_grid_backend(
    X,
    y,
    backend,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
) -> np.ndarray:
    """Generate default alpha grid for Lasso using backend abstraction."""
    X_arr = backend.asarray(X, dtype=backend.float64)
    y_arr = backend.asarray(y, dtype=backend.float64).reshape(-1)

    n_samples = int(X_arr.shape[0])
    corr = backend.abs(X_arr.T @ y_arr) / float(max(1, n_samples))
    # Use shape to check size - works for both numpy and torch
    corr_size = int(corr.shape[0]) if hasattr(corr, 'shape') else len(corr)
    alpha_max = float(backend.to_numpy(backend.max(corr))) if corr_size > 0 else 1.0

    if n_samples > 1:
        y_std = backend.sqrt(backend.mean((y_arr - backend.mean(y_arr)) ** 2))
        sigma_hat = float(backend.to_numpy(y_std))
    else:
        sigma_hat = 0.0

    sigma_hat = max(sigma_hat, 1e-8)
    penalty_scale = np.sqrt(2.0 * np.log(max(2, int(X_arr.shape[1]))) / max(1, n_samples))
    alpha_max = max(alpha_max, float(sigma_hat * penalty_scale), 1e-6)

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


def _default_lasso_alpha_grid_cupy(
    X,
    y,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
) -> np.ndarray:
    import cupy as cp

    X_cp = cp.asarray(X, dtype=cp.float64)
    y_cp = cp.asarray(y, dtype=cp.float64).reshape(-1)

    n_samples = int(X_cp.shape[0])
    corr = cp.abs(X_cp.T @ y_cp) / float(max(1, n_samples))
    alpha_max = float(cp.max(corr).item()) if int(corr.size) > 0 else 1.0

    if n_samples > 1:
        sigma_hat = float(cp.std(y_cp, ddof=1).item())
    else:
        sigma_hat = float(cp.std(y_cp).item())

    sigma_hat = max(sigma_hat, 1e-8)
    penalty_scale = np.sqrt(2.0 * np.log(max(2, int(X_cp.shape[1]))) / max(1, n_samples))
    alpha_max = max(alpha_max, float(sigma_hat * penalty_scale), 1e-6)

    if int(n_alphas) <= 1:
        return np.asarray([alpha_max], dtype=np.float64)

    alpha_min = max(float(alpha_min_ratio) * alpha_max, 1e-6)
    return np.geomspace(alpha_max, alpha_min, num=int(n_alphas)).astype(np.float64)


def _kfold_indices(n_samples: int, n_splits: int, random_state: Optional[int]):
    n = int(n_samples)
    k = max(2, min(int(n_splits), n))

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)

    fold_sizes = np.full(k, n // k, dtype=np.int64)
    fold_sizes[: n % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        current = stop
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        folds.append((train_idx, val_idx))

    if len(folds) == 0:
        all_idx = np.arange(n, dtype=np.int64)
        return [(all_idx, all_idx)]

    return folds


def _normalize_cv_splits(cv_splits, n_samples: int):
    if cv_splits is None:
        return None

    n = int(n_samples)
    folds = []

    for split in cv_splits:
        if not isinstance(split, (tuple, list)) or len(split) != 2:
            raise ValueError("Each cv_splits entry must be a (train_idx, val_idx) pair")

        train_idx = np.asarray(split[0], dtype=np.int64).reshape(-1)
        val_idx = np.asarray(split[1], dtype=np.int64).reshape(-1)

        if train_idx.size == 0 or val_idx.size == 0:
            continue

        if (
            bool(np.any(train_idx < 0))
            or bool(np.any(train_idx >= n))
            or bool(np.any(val_idx < 0))
            or bool(np.any(val_idx >= n))
        ):
            raise ValueError("cv_splits indices are out of range")

        folds.append((train_idx, val_idx))

    if len(folds) == 0:
        raise ValueError("cv_splits must contain at least one non-empty split")

    return folds


def _folds_are_complements(folds, n_samples: int) -> bool:
    """Return True when each fold uses train as the exact complement of validation."""
    n = int(n_samples)
    for train_idx, val_idx in folds:
        train_arr = np.asarray(train_idx, dtype=np.int64).reshape(-1)
        val_arr = np.asarray(val_idx, dtype=np.int64).reshape(-1)

        if int(train_arr.size + val_arr.size) != n:
            return False

        mask = np.zeros((n,), dtype=np.int8)
        mask[train_arr] = 1
        if bool(np.any(mask[val_arr] != 0)):
            return False
        mask[val_arr] = 1
        if bool(np.any(mask == 0)):
            return False

    return True


def _array_identity_token(x: Any) -> Tuple[Any, ...]:
    if x is None:
        return ("none",)

    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return ("cupy", int(x.data.ptr), tuple(int(v) for v in x.shape), str(x.dtype))
    except Exception:
        pass

    # Check for Torch tensors
    try:
        import torch

        if isinstance(x, torch.Tensor):
            # For GPU tensors, use the data pointer; for CPU, use storage pointer
            if x.is_cuda:
                ptr = int(x.data_ptr())
            else:
                # CPU tensor - use underlying storage pointer
                ptr = int(x.untyped_storage().data_ptr()) if hasattr(x, 'untyped_storage') else id(x)
            return ("torch", ptr, tuple(int(v) for v in x.shape), str(x.dtype))
    except Exception:
        pass

    arr = np.asarray(x)
    ptr = int(arr.__array_interface__["data"][0]) if int(arr.size) > 0 else 0
    return ("numpy", ptr, tuple(int(v) for v in arr.shape), str(arr.dtype))


def _alphas_signature(alphas: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(alphas, dtype=np.float64).reshape(-1))
    return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()


def _folds_signature(folds) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for train_idx, val_idx in folds:
        train_arr = np.ascontiguousarray(np.asarray(train_idx, dtype=np.int64).reshape(-1))
        val_arr = np.ascontiguousarray(np.asarray(val_idx, dtype=np.int64).reshape(-1))
        hasher.update(train_arr.tobytes())
        hasher.update(b"|")
        hasher.update(val_arr.tobytes())
        hasher.update(b";")
    return hasher.hexdigest()


def _make_lasso_cv_auto_cache_key(
    *,
    X,
    y,
    sample_weight,
    alpha_grid: np.ndarray,
    folds,
    fit_intercept: bool,
    use_gpu: bool,
    max_iter: int,
    tol: float,
    cpu_solver: str,
    cv_method: str,
    cd_kkt_check_every: Optional[int],
    gpu_cv_mixed_precision: bool,
) -> Tuple[Any, ...]:
    return (
        "lasso_cv_auto_v1",
        _array_identity_token(X),
        _array_identity_token(y),
        _array_identity_token(sample_weight),
        _alphas_signature(alpha_grid),
        _folds_signature(folds),
        bool(fit_intercept),
        bool(use_gpu),
        int(max_iter),
        float(tol),
        str(cpu_solver).lower(),
        str(cv_method).lower(),
        None if cd_kkt_check_every is None else int(cd_kkt_check_every),
        bool(gpu_cv_mixed_precision),
    )


def _clone_lasso_cv_cache_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "alpha": float(payload["alpha"]),
        "alphas": np.asarray(payload["alphas"], dtype=np.float64).copy(),
        "mse_path": np.asarray(payload["mse_path"], dtype=np.float64).copy(),
        "mean_mse": np.asarray(payload["mean_mse"], dtype=np.float64).copy(),
    }


def _lasso_cv_cache_get(cache_key: Optional[Tuple[Any, ...]]) -> Optional[Dict[str, Any]]:
    if cache_key is None or _LASSO_CV_ALPHA_CACHE_MAXSIZE <= 0:
        return None

    cached = _LASSO_CV_ALPHA_CACHE.get(cache_key)
    if cached is None:
        return None

    _LASSO_CV_ALPHA_CACHE.move_to_end(cache_key)
    return _clone_lasso_cv_cache_payload(cached)


def _lasso_cv_cache_put(cache_key: Optional[Tuple[Any, ...]], payload: Dict[str, Any]) -> None:
    if cache_key is None or _LASSO_CV_ALPHA_CACHE_MAXSIZE <= 0:
        return

    _LASSO_CV_ALPHA_CACHE[cache_key] = _clone_lasso_cv_cache_payload(payload)
    _LASSO_CV_ALPHA_CACHE.move_to_end(cache_key)

    while len(_LASSO_CV_ALPHA_CACHE) > int(_LASSO_CV_ALPHA_CACHE_MAXSIZE):
        _LASSO_CV_ALPHA_CACHE.popitem(last=False)


def _adaptive_gpu_check_every(
    *,
    base_check_every: int,
    iteration: int,
    max_iter: int,
    active_ratio: float,
) -> int:
    """Adaptive cadence for expensive global convergence checks on GPU."""
    base = max(1, int(base_check_every))
    ratio = float(max(0.0, min(1.0, active_ratio)))

    if ratio >= 0.75:
        interval = max(base, 16)
    elif ratio >= 0.40:
        interval = max(base, 12)
    elif ratio >= 0.15:
        interval = max(4, base)
    else:
        interval = max(2, base // 2)

    progress = float(iteration + 1) / float(max(1, int(max_iter)))
    if progress >= 0.90:
        interval = min(interval, 2)
    elif progress >= 0.75:
        interval = min(interval, 4)

    return max(1, int(interval))


def _soft_threshold_numpy(x: np.ndarray, gamma: float) -> np.ndarray:
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    return np.sign(x) * np.maximum(np.abs(x) - gamma_arr, 0.0)


def _soft_threshold_scalar(x: float, gamma: float) -> float:
    ax = abs(float(x))
    g = float(gamma)
    if ax <= g:
        return 0.0
    return float(np.sign(x) * (ax - g))


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _soft_threshold_scalar_numba(x: float, gamma: float) -> float:
        ax = abs(x)
        if ax <= gamma:
            return 0.0
        if x >= 0.0:
            return ax - gamma
        return -(ax - gamma)


    @njit(cache=True)
    def _solve_lasso_path_cpu_cd_numba_impl(
        XtX: np.ndarray,
        Xty: np.ndarray,
        n_samples: int,
        alphas_desc: np.ndarray,
        max_iter: int,
        tol: float,
        stopping_is_kkt: bool,
        cd_kkt_check_every: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_features = XtX.shape[0]
        n_alphas = alphas_desc.shape[0]

        coefs_path = np.zeros((n_alphas, n_features), dtype=np.float64)
        n_iters = np.zeros((n_alphas,), dtype=np.int32)

        coef = np.zeros((n_features,), dtype=np.float64)
        grad = -Xty.copy()

        X_sq_norms = np.empty((n_features,), dtype=np.float64)
        for j in range(n_features):
            X_sq_norms[j] = XtX[j, j]

        n_samp = float(max(1, n_samples))
        alpha_scaled_desc = np.empty((n_alphas,), dtype=np.float64)
        for idx in range(n_alphas):
            alpha_scaled_desc[idx] = alphas_desc[idx] * n_samp

        active_mask = np.zeros((n_features,), dtype=np.bool_)
        check_every = max(1, int(cd_kkt_check_every))

        for alpha_idx in range(n_alphas):
            alpha = float(alphas_desc[alpha_idx])
            alpha_scaled = float(alpha_scaled_desc[alpha_idx])
            if alpha_idx > 0:
                prev_alpha_scaled = float(alpha_scaled_desc[alpha_idx - 1])
            else:
                prev_alpha_scaled = alpha_scaled

            strong_thresh = 2.0 * alpha_scaled - prev_alpha_scaled
            if strong_thresh < 0.0:
                strong_thresh = 0.0

            any_active = False
            max_abs_xty = -1.0
            max_abs_xty_idx = 0
            for j in range(n_features):
                abs_xty = abs(Xty[j])
                if abs_xty >= strong_thresh:
                    active_mask[j] = True
                    any_active = True
                if abs_xty > max_abs_xty:
                    max_abs_xty = abs_xty
                    max_abs_xty_idx = j

            if not any_active:
                active_mask[max_abs_xty_idx] = True

            converged = False

            for iteration in range(int(max_iter)):
                coef_delta_l1 = 0.0

                for j in range(n_features):
                    if not active_mask[j]:
                        continue

                    denom = float(X_sq_norms[j])
                    old_val = float(coef[j])

                    if denom > 1e-10:
                        rho_j = -float(grad[j]) + denom * old_val
                        new_val = _soft_threshold_scalar_numba(rho_j, alpha_scaled) / denom
                    else:
                        new_val = 0.0

                    delta = new_val - old_val
                    if delta != 0.0:
                        coef[j] = new_val
                        coef_delta_l1 += abs(delta)
                        for row_idx in range(n_features):
                            grad[row_idx] += XtX[row_idx, j] * delta

                should_kkt_scan = (
                    ((iteration + 1) % check_every == 0)
                    or (coef_delta_l1 < float(tol))
                    or (iteration + 1 == int(max_iter))
                )

                violation = 0.0
                has_inactive_violation = False

                if should_kkt_scan:
                    for j in range(n_features):
                        v = abs(grad[j] / n_samp) - alpha
                        if v < 0.0:
                            v = 0.0
                        if v > violation:
                            violation = v
                        if v > float(tol) and (not active_mask[j]):
                            active_mask[j] = True
                            has_inactive_violation = True

                if stopping_is_kkt:
                    if should_kkt_scan and violation < float(tol):
                        n_iters[alpha_idx] = int(iteration) + 1
                        converged = True
                        break
                else:
                    if coef_delta_l1 < float(tol) and (not has_inactive_violation):
                        n_iters[alpha_idx] = int(iteration) + 1
                        converged = True
                        break

            if not converged:
                n_iters[alpha_idx] = int(max_iter)

            for j in range(n_features):
                coefs_path[alpha_idx, j] = coef[j]
                if abs(coef[j]) > 0.0:
                    active_mask[j] = True

        return coefs_path, n_iters


def _solve_lasso_path_cpu_cd_numba(
    XtX: np.ndarray,
    Xty: np.ndarray,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    cd_kkt_check_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    XtX_c = np.ascontiguousarray(XtX, dtype=np.float64)
    Xty_c = np.ascontiguousarray(Xty, dtype=np.float64)
    alphas_c = np.ascontiguousarray(np.asarray(alphas_desc, dtype=np.float64))
    stopping_is_kkt = str(stopping).lower() == "kkt"
    return _solve_lasso_path_cpu_cd_numba_impl(
        XtX_c,
        Xty_c,
        int(n_samples),
        alphas_c,
        int(max_iter),
        float(tol),
        bool(stopping_is_kkt),
        int(cd_kkt_check_every),
    )


def _normalize_lassocv_method(method: str) -> str:
    """Normalize CV optimization profile name."""
    key = str(method).strip().lower()
    alias_map = {
        "default": "standard",
        "classic": "standard",
        "glmnet_cv": "glmnet",
        "glmnet.cv": "glmnet",
    }
    key = alias_map.get(key, key)
    if key not in ("standard", "glmnet"):
        raise ValueError("method must be one of: 'standard', 'glmnet'")
    return key


def _normalize_cd_kkt_check_every(cd_kkt_check_every: Optional[int]) -> Optional[int]:
    """Validate optional coordinate-descent global KKT scan cadence."""
    if cd_kkt_check_every is None:
        return None
    value = int(cd_kkt_check_every)
    if value <= 0:
        raise ValueError("cd_kkt_check_every must be a positive integer or None")
    return value


def _solve_lasso_path_cpu_fista_batched_from_gram(
    XtX: np.ndarray,
    Xty: np.ndarray,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve descending-alpha Lasso path with a batched CPU FISTA update."""
    n_features = int(XtX.shape[0])
    n_alphas = int(alphas_desc.shape[0])

    coefs = np.zeros((n_features, n_alphas), dtype=np.float64)
    yk = coefs.copy()
    tk = np.ones((n_alphas,), dtype=np.float64)
    n_iters = np.zeros((n_alphas,), dtype=np.int32)

    if lipschitz_L is not None:
        L = float(lipschitz_L)
    else:
        try:
            eigvals = np.linalg.eigvalsh(XtX)
            L = float(eigvals[-1] / float(max(1, n_samples)))
        except Exception:
            row_sum_bound = float(np.max(np.sum(np.abs(XtX), axis=1)) / float(max(1, n_samples)))
            L = max(row_sum_bound, 1e-12)

    if L <= 0.0:
        return coefs.T, n_iters

    n_samp = float(max(1, n_samples))
    step = 1.0 / L
    alphas_desc = np.asarray(alphas_desc, dtype=np.float64)
    thresholds = alphas_desc * step
    stopping_name = str(stopping).lower()
    check_every = max(1, int(check_every))

    active = np.arange(n_alphas, dtype=np.int64)

    for iteration in range(int(max_iter)):
        if active.size == 0:
            break

        y_active = yk[:, active]
        coef_old = coefs[:, active]

        grad = (XtX @ y_active - Xty.reshape(-1, 1)) / n_samp
        thresh = thresholds[active].reshape(1, -1)
        coef_new = _soft_threshold_numpy(y_active - step * grad, thresh)

        t_old = tk[active]
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
        beta = (t_old - 1.0) / t_new
        y_new = coef_new + beta.reshape(1, -1) * (coef_new - coef_old)

        coefs[:, active] = coef_new
        yk[:, active] = y_new
        tk[active] = t_new

        should_check = ((iteration + 1) % check_every == 0) or (iteration + 1 == int(max_iter))
        if not should_check:
            continue

        if stopping_name == "kkt":
            grad_sse = (XtX @ coef_new - Xty.reshape(-1, 1)) / n_samp
            viol = np.max(
                np.maximum(
                    np.abs(grad_sse) - alphas_desc[active].reshape(1, -1),
                    0.0,
                ),
                axis=0,
            )
            converged_local = viol < float(tol)
        else:
            delta = np.sum(np.abs(coef_new - coef_old), axis=0)
            converged_local = delta < float(tol)

        if not np.any(converged_local):
            continue

        done = active[converged_local]
        n_iters[done] = int(iteration) + 1
        yk[:, done] = coefs[:, done]
        active = active[~converged_local]

    if active.size > 0:
        n_iters[active] = int(max_iter)

    return coefs.T, n_iters


def _solve_lasso_path_gpu_fista_batched_from_gram(
    XtX,
    Xty,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 8,
):
    """Solve descending-alpha Lasso path with a batched GPU FISTA update."""
    import cupy as cp

    n_features = int(XtX.shape[0])
    n_alphas = int(alphas_desc.shape[0])

    coefs = cp.zeros((n_features, n_alphas), dtype=XtX.dtype)
    yk = coefs.copy()
    tk = cp.ones((n_alphas,), dtype=XtX.dtype)
    n_iters_gpu = cp.zeros((n_alphas,), dtype=cp.int32)

    if lipschitz_L is not None:
        L = cp.array(float(lipschitz_L), dtype=XtX.dtype)
    else:
        try:
            eigvals = cp.linalg.eigvalsh(XtX)
            L = eigvals[-1] / float(max(1, n_samples))
        except Exception:
            row_sum_bound = cp.max(cp.sum(cp.abs(XtX), axis=1)) / float(max(1, n_samples))
            L = cp.maximum(row_sum_bound, cp.asarray(1e-12, dtype=XtX.dtype))

    L_scalar = float(cp.asnumpy(L))
    if L_scalar <= 0.0:
        return coefs.T, np.zeros((n_alphas,), dtype=np.int32)

    n_samp = float(max(1, n_samples))
    step = 1.0 / L
    alphas_desc = np.asarray(alphas_desc, dtype=np.float64)
    alpha_gpu = cp.asarray(alphas_desc, dtype=XtX.dtype)
    thresholds = alpha_gpu * step
    stopping_name = str(stopping).lower()
    check_every = max(1, int(check_every))

    active_gpu = cp.arange(n_alphas, dtype=cp.int32)

    for iteration in range(int(max_iter)):
        if int(active_gpu.size) == 0:
            break

        y_active = yk[:, active_gpu]
        coef_old = coefs[:, active_gpu]

        grad = (XtX @ y_active - Xty.reshape(-1, 1)) / n_samp
        thresh = thresholds[active_gpu].reshape(1, -1)
        coef_new = cp.sign(y_active - step * grad) * cp.maximum(cp.abs(y_active - step * grad) - thresh, 0.0)

        t_old = tk[active_gpu]
        t_new = (1.0 + cp.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
        beta = (t_old - 1.0) / t_new
        y_new = coef_new + beta.reshape(1, -1) * (coef_new - coef_old)

        coefs[:, active_gpu] = coef_new
        yk[:, active_gpu] = y_new
        tk[active_gpu] = t_new

        active_ratio = float(int(active_gpu.size)) / float(max(1, n_alphas))
        check_every_eff = _adaptive_gpu_check_every(
            base_check_every=check_every,
            iteration=iteration,
            max_iter=int(max_iter),
            active_ratio=active_ratio,
        )
        should_check = ((iteration + 1) % check_every_eff == 0) or (iteration + 1 == int(max_iter))
        if not should_check:
            continue

        if stopping_name == "kkt":
            grad_sse = (XtX @ coef_new - Xty.reshape(-1, 1)) / n_samp
            viol = cp.max(
                cp.maximum(
                    cp.abs(grad_sse) - alpha_gpu[active_gpu].reshape(1, -1),
                    0.0,
                ),
                axis=0,
            )
            converged_local_gpu = viol < float(tol)
        else:
            delta = cp.sum(cp.abs(coef_new - coef_old), axis=0)
            converged_local_gpu = delta < float(tol)

        done_gpu = active_gpu[converged_local_gpu]
        if int(done_gpu.size) == 0:
            continue

        n_iters_gpu[done_gpu] = int(iteration) + 1
        yk[:, done_gpu] = coefs[:, done_gpu]
        active_gpu = active_gpu[~converged_local_gpu]

    if int(active_gpu.size) > 0:
        n_iters_gpu[active_gpu] = int(max_iter)

    return coefs.T, cp.asnumpy(n_iters_gpu)


def _solve_lasso_path_gpu_fista_multi_fold_from_gram(
    XtX_batch,
    Xty_batch,
    *,
    n_samples_vec,
    alphas_desc,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 8,
):
    """Solve descending-alpha Lasso paths for all folds together on GPU.

    Note: Fused kernel optimization is disabled for multi-fold solver due to
    dtype complexity. The single-fold Lasso solver uses fused kernels.
    """
    import cupy as cp

    n_folds = int(XtX_batch.shape[0])
    n_features = int(XtX_batch.shape[1])
    n_alphas = int(alphas_desc.shape[0])

    coefs = cp.zeros((n_folds, n_features, n_alphas), dtype=XtX_batch.dtype)
    yk = coefs.copy()
    tk = cp.ones((n_folds, n_alphas), dtype=XtX_batch.dtype)
    n_iters_gpu = cp.zeros((n_folds, n_alphas), dtype=cp.int32)

    # Convert n_samples_vec to numpy using .get() if it's a CuPy array
    if hasattr(n_samples_vec, 'get'):
        n_vec_cpu = n_samples_vec.get().astype(np.float64).reshape(-1)
    else:
        n_vec_cpu = np.asarray(n_samples_vec, dtype=np.float64).reshape(-1)
    if n_vec_cpu.size != n_folds:
        raise ValueError("n_samples_vec must have one entry per fold")
    n_vec = cp.asarray(n_vec_cpu, dtype=XtX_batch.dtype)

    if lipschitz_L is not None:
        L = cp.full((n_folds,), float(lipschitz_L), dtype=XtX_batch.dtype)
    else:
        try:
            eigvals = cp.linalg.eigvalsh(XtX_batch)
            L = eigvals[:, -1] / n_vec
        except Exception:
            row_sum_bound = cp.max(cp.sum(cp.abs(XtX_batch), axis=2), axis=1) / n_vec
            L = cp.maximum(row_sum_bound, cp.asarray(1e-12, dtype=XtX_batch.dtype))

    step = 1.0 / L.reshape(n_folds, 1, 1)
    # Convert alphas_desc to numpy using .get() if it's a CuPy array
    if hasattr(alphas_desc, 'get'):
        alphas_cpu = alphas_desc.get().astype(np.float64)
    else:
        alphas_cpu = np.asarray(alphas_desc, dtype=np.float64)
    alpha_gpu = cp.asarray(alphas_cpu, dtype=XtX_batch.dtype).reshape(1, 1, n_alphas)
    thresholds = alpha_gpu * step

    Xty_expanded = Xty_batch.reshape(n_folds, n_features, 1)
    n_vec_expanded = n_vec.reshape(n_folds, 1, 1)
    stopping_name = str(stopping).lower()
    check_every = max(1, int(check_every))

    active_gpu = cp.ones((n_folds, n_alphas), dtype=cp.bool_)
    active_count = int(n_folds * n_alphas)

    # Note: Fused kernels disabled for multi-fold solver due to dtype complexity
    # The single-fold Lasso._fit_gpu uses fused kernels
    use_fused = False
    fused = None

    for iteration in range(int(max_iter)):
        if active_count == 0:
            break

        active_expanded = active_gpu[:, cp.newaxis, :]

        coef_old = coefs.copy()
        grad = (cp.matmul(XtX_batch, yk) - Xty_expanded) / n_vec_expanded

        # Proximal step: soft thresholding
        yk_step = yk - step * grad
        coef_candidate = cp.sign(yk_step) * cp.maximum(cp.abs(yk_step) - thresholds, 0.0)
        coefs = cp.where(active_expanded, coef_candidate, coefs)

        t_old = tk
        t_new = (1.0 + cp.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
        beta = (t_old - 1.0) / t_new
        y_candidate = coefs + beta[:, cp.newaxis, :] * (coefs - coef_old)
        yk = cp.where(active_expanded, y_candidate, yk)
        tk = cp.where(active_gpu, t_new, tk)

        active_ratio = float(active_count) / float(max(1, n_folds * n_alphas))
        check_every_eff = _adaptive_gpu_check_every(
            base_check_every=check_every,
            iteration=iteration,
            max_iter=int(max_iter),
            active_ratio=active_ratio,
        )
        should_check = ((iteration + 1) % check_every_eff == 0) or (iteration + 1 == int(max_iter))
        if not should_check:
            continue

        if stopping_name == "kkt":
            grad_sse = (cp.matmul(XtX_batch, coefs) - Xty_expanded) / n_vec_expanded
            violation = cp.max(cp.maximum(cp.abs(grad_sse) - alpha_gpu, 0.0), axis=1)
            converged_local_gpu = violation < float(tol)
        else:
            delta = cp.sum(cp.abs(coefs - coef_old), axis=1)
            converged_local_gpu = delta < float(tol)

        newly_done_gpu = active_gpu & converged_local_gpu
        done_count = int(cp.count_nonzero(newly_done_gpu).item())
        if done_count == 0:
            continue

        n_iters_gpu[newly_done_gpu] = int(iteration) + 1
        yk = cp.where(newly_done_gpu[:, cp.newaxis, :], coefs, yk)
        active_gpu = active_gpu & (~converged_local_gpu)
        active_count -= done_count

    n_iters_gpu[active_gpu] = int(max_iter)

    return cp.transpose(coefs, (0, 2, 1)), cp.asnumpy(n_iters_gpu)


def _solve_lasso_path_cpu_from_gram(
    XtX: np.ndarray,
    Xty: np.ndarray,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    cpu_solver: str,
    lipschitz_L: Optional[float] = None,
    cd_kkt_check_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a descending-alpha Lasso path on CPU using one precomputed Gram matrix."""
    n_features = int(XtX.shape[0])
    n_alphas = int(alphas_desc.shape[0])

    coefs_path = np.zeros((n_alphas, n_features), dtype=np.float64)
    n_iters = np.zeros(n_alphas, dtype=np.int32)

    coef = np.zeros(n_features, dtype=np.float64)
    stopping_name = str(stopping).lower()
    solver_name = str(cpu_solver).lower()

    if solver_name == "fista":
        return _solve_lasso_path_cpu_fista_batched_from_gram(
            XtX,
            Xty,
            n_samples=n_samples,
            alphas_desc=alphas_desc,
            max_iter=max_iter,
            tol=tol,
            stopping=stopping,
            lipschitz_L=lipschitz_L,
            check_every=2,
        )

    global _NUMBA_CD_DISABLED
    use_numba_cd = (
        _NUMBA_AVAILABLE
        and (not _NUMBA_CD_DISABLED)
        and solver_name == "coordinate_descent"
    )

    if use_numba_cd:
        try:
            return _solve_lasso_path_cpu_cd_numba(
                XtX,
                Xty,
                n_samples=n_samples,
                alphas_desc=alphas_desc,
                max_iter=max_iter,
                tol=tol,
                stopping=stopping,
                cd_kkt_check_every=cd_kkt_check_every,
            )
        except Exception:
            _NUMBA_CD_DISABLED = True

    # Coordinate descent with incremental gradient updates.
    X_sq_norms = np.diag(XtX).astype(np.float64, copy=False)
    grad = XtX @ coef - Xty
    alpha_scaled_desc = np.asarray(alphas_desc, dtype=np.float64) * float(max(1, n_samples))
    active_mask = np.zeros((n_features,), dtype=bool)
    cd_kkt_check_every = max(1, int(cd_kkt_check_every))

    for alpha_idx, alpha in enumerate(alphas_desc):
        alpha_scaled = float(alpha_scaled_desc[alpha_idx])
        prev_alpha_scaled = float(alpha_scaled_desc[alpha_idx - 1]) if alpha_idx > 0 else alpha_scaled

        # Strong rule screening: expand active set before cyclic updates.
        strong_thresh = max(0.0, 2.0 * alpha_scaled - prev_alpha_scaled)
        active_mask |= np.abs(Xty) >= strong_thresh
        if not bool(np.any(active_mask)):
            active_mask[int(np.argmax(np.abs(Xty)))] = True

        converged = False

        for iteration in range(int(max_iter)):
            coef_delta_l1 = 0.0

            active_idx = np.flatnonzero(active_mask)
            for j in active_idx:
                denom = float(X_sq_norms[j])
                old_val = float(coef[j])

                if denom > 1e-10:
                    rho_j = -float(grad[j]) + denom * old_val
                    new_val = _soft_threshold_scalar(rho_j, alpha_scaled) / denom
                else:
                    new_val = 0.0

                delta = new_val - old_val
                if abs(delta) > 0.0:
                    coef[j] = new_val
                    grad += XtX[:, j] * delta
                    coef_delta_l1 += abs(delta)

            # glmnet-style optimization can skip full inactive KKT scans on every pass,
            # then force a check when updates become small.
            should_kkt_scan = (
                ((iteration + 1) % cd_kkt_check_every == 0)
                or (coef_delta_l1 < float(tol))
                or (iteration + 1 == int(max_iter))
            )
            violation = float("inf")
            inactive_violation_idx = np.empty((0,), dtype=np.int64)

            if should_kkt_scan:
                violation_vec = np.maximum(
                    np.abs(grad / float(max(1, n_samples))) - float(alpha),
                    0.0,
                )
                inactive_violation_idx = np.where((violation_vec > float(tol)) & (~active_mask))[0]
                if inactive_violation_idx.size > 0:
                    active_mask[inactive_violation_idx] = True
                violation = float(np.max(violation_vec))

            if stopping_name == "kkt":
                if should_kkt_scan and violation < float(tol):
                    n_iters[alpha_idx] = iteration + 1
                    converged = True
                    break
            else:
                if coef_delta_l1 < float(tol) and inactive_violation_idx.size == 0:
                    n_iters[alpha_idx] = iteration + 1
                    converged = True
                    break

        if not converged:
            n_iters[alpha_idx] = int(max_iter)

        coefs_path[alpha_idx, :] = coef
        active_mask |= np.abs(coef) > 0.0

    return coefs_path, n_iters


def _solve_lasso_path_gpu_from_gram(
    XtX,
    Xty,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 8,
):
    """Solve a descending-alpha Lasso path on GPU using one precomputed Gram matrix."""
    return _solve_lasso_path_gpu_fista_batched_from_gram(
        XtX,
        Xty,
        n_samples=n_samples,
        alphas_desc=alphas_desc,
        max_iter=max_iter,
        tol=tol,
        stopping=stopping,
        lipschitz_L=lipschitz_L,
        check_every=check_every,
    )


def _batch_mse_numpy(
    X_val: np.ndarray,
    y_val: np.ndarray,
    coefs_path: np.ndarray,
    intercepts_path: np.ndarray,
    sample_weight_val: Optional[np.ndarray],
) -> np.ndarray:
    preds = X_val @ coefs_path.T + intercepts_path.reshape(1, -1)
    sq_err = (y_val.reshape(-1, 1) - preds) ** 2

    if sample_weight_val is None:
        return np.mean(sq_err, axis=0)

    denom = float(np.sum(sample_weight_val))
    if denom <= 0.0:
        return np.mean(sq_err, axis=0)

    return np.sum(sample_weight_val.reshape(-1, 1) * sq_err, axis=0) / denom


def _batch_mse(
    X_val,
    y_val,
    coefs_path,
    intercepts_path,
    backend,
    sample_weight_val,
) -> np.ndarray:
    """
    Compute MSE for multiple coefficient vectors.

    Parameters
    ----------
    X_val : array-like
        Validation design matrix.
    y_val : array-like
        Validation response.
    coefs_path : array-like
        Coefficient matrix (n_alphas, n_features).
    intercepts_path : array-like
        Intercept vector (n_alphas,).
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    sample_weight_val : array-like or None
        Sample weights.

    Returns
    -------
    mse : ndarray
        MSE for each alpha.
    """
    preds = X_val @ coefs_path.T + intercepts_path.reshape(1, -1)
    sq_err = (y_val.reshape(-1, 1) - preds) ** 2

    if sample_weight_val is None:
        mse = backend.mean(sq_err, axis=0)
    else:
        denom = backend.sum(sample_weight_val)
        if float(backend.to_numpy(denom)) <= 0.0:
            mse = backend.mean(sq_err, axis=0)
        else:
            mse = backend.sum(sample_weight_val.reshape(-1, 1) * sq_err, axis=0) / denom

    return backend.to_numpy(mse)


def _soft_threshold_torch(x, gamma):
    """Soft thresholding operator for Torch tensors."""
    import torch
    return torch.sign(x) * torch.maximum(torch.abs(x) - gamma, torch.tensor(0.0, dtype=x.dtype, device=x.device))


def _fit_lasso_single_alpha_fast(
    X,
    y,
    *,
    alpha: float,
    fit_intercept: bool,
    max_iter: int,
    tol: float,
    stopping: str,
    device: str,
    cpu_solver: str,
    cd_kkt_check_every: int = 1,
    sample_weight=None,
) -> Dict[str, object]:
    """Fast single-alpha Lasso fit using optimized Gram-based path solvers."""
    device_name = str(device).lower()
    alpha_vec = np.asarray([float(alpha)], dtype=np.float64)

    # Check if inputs are torch tensors on GPU
    is_torch_gpu = False
    try:
        import torch
        is_torch_gpu = device_name == Device.CUDA.value and isinstance(X, torch.Tensor)
    except Exception:
        pass

    if device_name == Device.CUDA.value and not is_torch_gpu:
        # CuPy GPU path
        import cupy as cp

        X_arr = cp.asarray(X)
        y_arr = cp.asarray(y).reshape(-1)

        if sample_weight is not None:
            sw = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sw)
            X_arr = X_arr * sqrt_sw[:, cp.newaxis]
            y_arr = y_arr * sqrt_sw

        if bool(fit_intercept):
            X_mean = cp.mean(X_arr, axis=0)
            y_mean = cp.mean(y_arr)
            X_centered = X_arr - X_mean
            y_centered = y_arr - y_mean
        else:
            X_mean = cp.zeros((X_arr.shape[1],), dtype=X_arr.dtype)
            y_mean = cp.array(0.0, dtype=X_arr.dtype)
            X_centered = X_arr
            y_centered = y_arr

        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        coefs_desc, n_iters = _solve_lasso_path_gpu_from_gram(
            XtX,
            Xty,
            n_samples=int(X_arr.shape[0]),
            alphas_desc=alpha_vec,
            max_iter=int(max_iter),
            tol=float(tol),
            stopping=str(stopping),
            lipschitz_L=None,
            check_every=8,
        )

        coef_gpu = coefs_desc[0]
        if bool(fit_intercept):
            intercept_gpu = y_mean - X_mean @ coef_gpu
            intercept = float(cp.asnumpy(intercept_gpu))
        else:
            intercept = 0.0

        coef = np.asarray(cp.asnumpy(coef_gpu), dtype=np.float64)
        return {
            "coef": coef,
            "intercept": float(intercept),
            "n_iter": int(n_iters[0]),
            "n_samples": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
        }

    elif is_torch_gpu:
        # Torch GPU path - use FISTA solver directly on GPU tensors
        import torch

        n_samples = int(X_arr.shape[0])
        n_features = int(X_arr.shape[1])

        # Precompute Gram matrix and X'y for FISTA gradient
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Compute Lipschitz constant L = max eigenvalue of XtX / n
        try:
            eigvals = torch.linalg.eigvalsh(XtX)
            L = eigvals[-1] / n_samples
        except Exception:
            L = torch.sum(X_centered ** 2) / n_samples
        L = max(L, 1e-10)

        step = 1.0 / L
        thresh = float(alpha) * step

        # FISTA initialization
        coef = torch.zeros(n_features, dtype=X_arr.dtype, device=X_arr.device)
        z = coef.clone()
        t = torch.tensor(1.0, dtype=X_arr.dtype, device=X_arr.device)

        # FISTA iterations
        for iteration in range(int(max_iter)):
            coef_old = coef.clone()

            # Gradient step at z
            grad = (XtX @ z - Xty) / n_samples
            coef = _soft_threshold_torch(z - step * grad, thresh)

            # Momentum update
            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            z = coef + ((t - 1.0) / t_new) * (coef - coef_old)
            t = t_new

            # Convergence check
            if str(stopping).lower() == "kkt":
                grad_sse = (XtX @ coef - Xty) / n_samples
                violation = torch.max(torch.maximum(torch.abs(grad_sse) - float(alpha), torch.tensor(0.0, dtype=X_arr.dtype, device=X_arr.device)))
                if violation < float(tol):
                    break
            else:
                if torch.sum(torch.abs(coef - coef_old)) < float(tol):
                    break

        # Build coefficients
        if bool(fit_intercept):
            intercept_torch = y_mean - X_mean @ coef
            intercept = float(intercept_torch.cpu().numpy())
        else:
            intercept = 0.0

        coef_np = np.asarray(coef.cpu().numpy(), dtype=np.float64)
        return {
            "coef": coef_np,
            "intercept": float(intercept),
            "n_iter": int(iteration + 1),
            "n_samples": n_samples,
            "n_features": n_features,
        }

    X_arr = np.asarray(X)
    y_arr = np.asarray(y).reshape(-1)

    if sample_weight is not None:
        sw = np.asarray(sample_weight)
        sqrt_sw = np.sqrt(sw)
        X_arr = X_arr * sqrt_sw[:, np.newaxis]
        y_arr = y_arr * sqrt_sw

    if bool(fit_intercept):
        X_mean = np.mean(X_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        X_centered = X_arr - X_mean
        y_centered = y_arr - y_mean
    else:
        X_mean = np.zeros((X_arr.shape[1],), dtype=np.float64)
        y_mean = 0.0
        X_centered = X_arr
        y_centered = y_arr

    XtX = X_centered.T @ X_centered
    Xty = X_centered.T @ y_centered

    coefs_desc, n_iters = _solve_lasso_path_cpu_from_gram(
        XtX,
        Xty,
        n_samples=int(X_arr.shape[0]),
        alphas_desc=alpha_vec,
        max_iter=int(max_iter),
        tol=float(tol),
        stopping=str(stopping),
        cpu_solver=str(cpu_solver),
        lipschitz_L=None,
        cd_kkt_check_every=int(cd_kkt_check_every),
    )

    coef = np.asarray(coefs_desc[0], dtype=np.float64)
    if bool(fit_intercept):
        intercept = float(y_mean - X_mean @ coef)
    else:
        intercept = 0.0

    return {
        "coef": coef,
        "intercept": float(intercept),
        "n_iter": int(n_iters[0]),
        "n_samples": int(X_arr.shape[0]),
        "n_features": int(X_arr.shape[1]),
    }


def _select_lasso_alpha_cv(
    X,
    y,
    *,
    alphas=None,
    n_alphas: int = 12,
    alpha_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state: Optional[int] = None,
    sample_weight=None,
    fit_intercept: bool = False,
    device: Union[str, Device] = Device.CPU,
    max_iter: int = 3000,
    tol: float = 1e-4,
    cpu_solver: str = "coordinate_descent",
    method: str = "standard",
    cd_kkt_check_every: Optional[int] = None,
    gpu_cv_mixed_precision: bool = True,
    return_details: bool = False,
    cache_key: Optional[Tuple[Any, ...]] = None,
):
    """
    Select alpha via K-fold CV using statgpu's own Lasso implementation.

    Notes
    -----
    - Does not depend on sklearn.
    - Supports GPU path by setting ``device='cuda'``.
    """
    device_name = str(device).lower()
    use_gpu = device_name == Device.CUDA.value
    gpu_requested = use_gpu

    gpu_input_cupy = False
    gpu_input_torch = False
    if use_gpu:
        # Check if inputs are already on GPU (CuPy or Torch)
        try:
            import cupy as cp
            gpu_input_cupy = isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)
            if sample_weight is not None and not isinstance(sample_weight, cp.ndarray):
                gpu_input_cupy = False
        except Exception:
            pass

        # Also check for torch tensors
        if not gpu_input_cupy:
            try:
                import torch
                gpu_input_torch = isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)
                if sample_weight is not None and not isinstance(sample_weight, torch.Tensor):
                    gpu_input_torch = False
            except Exception:
                pass

    X_np = None
    y_np = None
    sample_weight_np = None

    if gpu_input_cupy or gpu_input_torch:
        # GPU inputs - get backend for validation
        backend = get_backend(backend='auto', device='cuda')
        if len(tuple(X.shape)) != 2:
            raise ValueError("X must be a 2D array")
        n_samples = int(X.shape[0])
    else:
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1)
        if sample_weight is not None:
            sample_weight_np = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_np.shape[0] != X_np.shape[0]:
            raise ValueError("y must have the same number of rows as X")
        if sample_weight_np is not None and sample_weight_np.shape[0] != X_np.shape[0]:
            raise ValueError("sample_weight must have the same number of rows as X")
        n_samples = int(X_np.shape[0])

    cv_method = _normalize_lassocv_method(method)
    requested_cd_kkt_check_every = _normalize_cd_kkt_check_every(cd_kkt_check_every)

    if alphas is None:
        if gpu_input_cupy or gpu_input_torch:
            # Get backend based on input type
            if gpu_input_torch:
                backend = get_backend(backend='torch', device='cuda')
            else:
                backend = get_backend(backend='cupy', device='cuda')
            alpha_grid = _default_lasso_alpha_grid_backend(
                X,
                y,
                backend,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio,
            )
        else:
            alpha_grid = _default_lasso_alpha_grid(
                X_np,
                y_np,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio,
            )
    else:
        alpha_grid = np.asarray(alphas, dtype=np.float64).reshape(-1)
        alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
        alpha_grid = alpha_grid[alpha_grid > 0.0]
        if alpha_grid.size == 0:
            if gpu_input_cupy or gpu_input_torch:
                # Get backend based on input type
                if gpu_input_torch:
                    backend = get_backend(backend='torch', device='cuda')
                else:
                    backend = get_backend(backend='cupy', device='cuda')
                alpha_grid = _default_lasso_alpha_grid_backend(
                    X,
                    y,
                    backend,
                    n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio,
                )
            else:
                alpha_grid = _default_lasso_alpha_grid(
                    X_np,
                    y_np,
                    n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio,
                )

    user_folds = _normalize_cv_splits(cv_splits, n_samples=n_samples)
    effective_n_folds = int(len(user_folds)) if user_folds is not None else int(cv_folds)

    if int(n_samples) < 4 or int(alpha_grid.size) == 1 or int(effective_n_folds) < 2:
        alpha0 = float(alpha_grid[0])
        if not return_details:
            return alpha0
        return {
            "alpha": alpha0,
            "alphas": alpha_grid.astype(np.float64, copy=False),
            "mse_path": np.full((int(alpha_grid.size), 1), np.nan, dtype=np.float64),
            "mean_mse": np.full(int(alpha_grid.size), np.nan, dtype=np.float64),
        }

    if user_folds is not None:
        folds = user_folds
    else:
        folds = _kfold_indices(
            n_samples=int(n_samples),
            n_splits=int(cv_folds),
            random_state=random_state,
        )

    folds_are_complements = _folds_are_complements(folds, n_samples=int(n_samples))

    alpha_grid = alpha_grid.astype(np.float64, copy=False)
    n_alpha = int(alpha_grid.size)
    n_folds = int(len(folds))

    cache_key_eff = cache_key
    if cache_key_eff is None and _LASSO_CV_ALPHA_CACHE_MAXSIZE > 0:
        cache_key_eff = _make_lasso_cv_auto_cache_key(
            X=X,
            y=y,
            sample_weight=sample_weight,
            alpha_grid=alpha_grid,
            folds=folds,
            fit_intercept=bool(fit_intercept),
            use_gpu=bool(use_gpu),
            max_iter=int(max_iter),
            tol=float(tol),
            cpu_solver=str(cpu_solver),
            cv_method=str(cv_method),
            cd_kkt_check_every=requested_cd_kkt_check_every,
            gpu_cv_mixed_precision=bool(gpu_cv_mixed_precision),
        )

    cached_details = _lasso_cv_cache_get(cache_key_eff)
    if cached_details is not None:
        if return_details:
            return cached_details
        return float(cached_details["alpha"])

    # Evaluate alpha path in descending order for warm-start efficiency.
    alpha_order_desc = np.argsort(-alpha_grid)
    alpha_desc = alpha_grid[alpha_order_desc]

    mse_path = np.full((n_alpha, n_folds), np.nan, dtype=np.float64)

    best_alpha = float(alpha_grid[0])
    best_mse = float("inf")

    if use_gpu:
        try:
            # Get backend based on input type - prefer Torch backend for Torch tensors
            if gpu_input_torch:
                backend = get_backend(backend='torch', device='cuda')
            elif gpu_input_cupy:
                backend = get_backend(backend='cupy', device='cuda')
            else:
                backend = get_backend(backend='auto', device='cuda')
            xp = backend.xp

            cv_dtype = backend.float32 if bool(gpu_cv_mixed_precision) else backend.float64

            # Convert inputs to backend arrays
            if gpu_input_cupy or gpu_input_torch:
                # Already on GPU (CuPy or Torch)
                X_full = backend.asarray(X, dtype=cv_dtype)
                y_full = backend.asarray(y, dtype=cv_dtype).reshape(-1)
                if sample_weight is not None:
                    sw_full = backend.asarray(sample_weight, dtype=cv_dtype).reshape(-1)
                else:
                    sw_full = None
            else:
                # Convert from numpy
                X_full = backend.asarray(X_np, dtype=cv_dtype)
                y_full = backend.asarray(y_np, dtype=cv_dtype)
                if sample_weight_np is not None:
                    sw_full = backend.asarray(sample_weight_np, dtype=cv_dtype)
                else:
                    sw_full = None

            XtX_folds = []
            Xty_folds = []
            n_train_folds = []
            X_mean_folds = []
            y_mean_folds = []
            fold_eval_payload = []

            fast_fold_stats = (sw_full is None) and bool(folds_are_complements)
            if fast_fold_stats:
                n_total = int(X_full.shape[0])
                XtX_full = X_full.T @ X_full
                Xty_full = X_full.T @ y_full
                if bool(fit_intercept):
                    X_sum_full = backend.sum(X_full, axis=0)
                    y_sum_full = backend.sum(y_full)
                else:
                    X_sum_full = None
                    y_sum_full = None

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                train_idx_gpu = backend.asarray(train_idx)
                val_idx_gpu = backend.asarray(val_idx)

                X_val = X_full[val_idx_gpu]
                y_val = y_full[val_idx_gpu]
                sw_val = None if sw_full is None else sw_full[val_idx_gpu]

                if fast_fold_stats:
                    n_val = int(val_idx_gpu.shape[0])
                    n_train = int(n_total - n_val)

                    XtX_val = X_val.T @ X_val
                    Xty_val = X_val.T @ y_val
                    XtX_raw = XtX_full - XtX_val
                    Xty_raw = Xty_full - Xty_val

                    if bool(fit_intercept):
                        X_sum_val = backend.sum(X_val, axis=0)
                        y_sum_val = backend.sum(y_val)
                        X_sum_train = X_sum_full - X_sum_val
                        y_sum_train = y_sum_full - y_sum_val

                        inv_n = backend.asarray(1.0 / float(max(1, n_train)), dtype=X_full.dtype)
                        X_mean = X_sum_train * inv_n
                        y_mean = y_sum_train * inv_n
                        XtX = XtX_raw - backend.outer(X_sum_train, X_sum_train) * inv_n
                        Xty = Xty_raw - X_sum_train * y_mean
                    else:
                        X_mean = backend.zeros((X_full.shape[1],), dtype=X_full.dtype)
                        y_mean = backend.array(0.0, dtype=X_full.dtype)
                        XtX = XtX_raw
                        Xty = Xty_raw
                else:
                    X_train = X_full[train_idx_gpu]
                    y_train = y_full[train_idx_gpu]
                    sw_train = None if sw_full is None else sw_full[train_idx_gpu]

                    if sw_train is not None:
                        sqrt_sw = backend.sqrt(sw_train)
                        X_train = X_train * sqrt_sw[:, backend.newaxis]
                        y_train = y_train * sqrt_sw

                    if bool(fit_intercept):
                        X_mean = backend.mean(X_train, axis=0)
                        y_mean = backend.mean(y_train)
                        X_centered = X_train - X_mean
                        y_centered = y_train - y_mean
                    else:
                        X_mean = backend.zeros((X_train.shape[1],), dtype=X_train.dtype)
                        y_mean = backend.array(0.0, dtype=X_train.dtype)
                        X_centered = X_train
                        y_centered = y_train

                    XtX = X_centered.T @ X_centered
                    Xty = X_centered.T @ y_centered
                    n_train = int(X_train.shape[0])

                XtX_folds.append(XtX)
                Xty_folds.append(Xty)
                n_train_folds.append(int(n_train))
                X_mean_folds.append(X_mean)
                y_mean_folds.append(y_mean)
                fold_eval_payload.append((X_val, y_val, sw_val))

            XtX_batch = backend.stack(XtX_folds, axis=0)
            Xty_batch = backend.stack(Xty_folds, axis=0)

            # Use native Torch FISTA solver for Torch backend
            if hasattr(xp, '__name__') and 'torch' in xp.__name__.lower():
                import torch
                n_samples_vec_torch = torch.tensor(np.asarray(n_train_folds, dtype=np.int32), device=XtX_batch.device, dtype=XtX_batch.dtype)

                coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch(
                    XtX_batch,
                    Xty_batch,
                    n_samples_vec=n_samples_vec_torch,
                    alphas_desc=alpha_desc,
                    max_iter=int(max_iter),
                    tol=float(tol),
                    stopping="coef_delta",
                    lipschitz_L=None,
                    check_every=8,
                )

                # Convert results back to numpy for evaluation
                for fold_idx in range(int(len(folds))):
                    coefs_desc_np = coefs_batch_desc[fold_idx]  # already numpy from the solver

                    if bool(fit_intercept):
                        y_mean_val = float(y_mean_folds[fold_idx])
                        X_mean_val = X_mean_folds[fold_idx]
                        intercepts_desc = y_mean_val - X_mean_val @ coefs_desc_np.T
                        intercepts_desc_gpu = backend.asarray(intercepts_desc)
                        coefs_desc_gpu = backend.asarray(coefs_desc_np)
                    else:
                        intercepts_desc_gpu = backend.zeros((coefs_desc_np.shape[0],), dtype=coefs_desc_np.dtype)
                        coefs_desc_gpu = backend.asarray(coefs_desc_np)

                    X_val, y_val, sw_val = fold_eval_payload[fold_idx]
                    mse_desc = _batch_mse(X_val, y_val, coefs_desc_gpu, intercepts_desc_gpu, backend, sw_val)

                    mse_path[alpha_order_desc, fold_idx] = mse_desc
            else:
                # CuPy backend - use existing solver directly
                import cupy as cp
                n_samples_vec_cp = cp.asarray(np.asarray(n_train_folds, dtype=np.int32))

                coefs_batch_desc, _ = _solve_lasso_path_gpu_fista_multi_fold_from_gram(
                    XtX_batch,
                    Xty_batch,
                    n_samples_vec=n_samples_vec_cp,
                    alphas_desc=alpha_desc,
                    max_iter=int(max_iter),
                    tol=float(tol),
                    stopping="coef_delta",
                    lipschitz_L=None,
                    check_every=8,
                )

                for fold_idx in range(int(len(folds))):
                    coefs_desc = coefs_batch_desc[fold_idx]

                    if bool(fit_intercept):
                        intercepts_desc = y_mean_folds[fold_idx] - X_mean_folds[fold_idx] @ coefs_desc.T
                    else:
                        intercepts_desc = backend.zeros((coefs_desc.shape[0],), dtype=coefs_desc.dtype)

                    X_val, y_val, sw_val = fold_eval_payload[fold_idx]
                    mse_desc = _batch_mse(X_val, y_val, coefs_desc, intercepts_desc, backend, sw_val)

                    mse_path[alpha_order_desc, fold_idx] = mse_desc

        except Exception as exc:
            raise RuntimeError(
                "GPU path failed in _select_lasso_alpha_cv with device='cuda'; "
                "CPU fallback is disabled for strict CUDA execution."
            ) from exc

    if not use_gpu:
        if gpu_requested:
            raise RuntimeError(
                "device='cuda' requested but GPU path was not executed; "
                "CPU fallback is disabled for strict CUDA execution."
            )
        cpu_solver_name = str(cpu_solver).lower()

        if cv_method == "glmnet":
            # glmnet-like CV profile: coordinate-descent path with periodic full KKT scans.
            cpu_solver_name = "coordinate_descent"

        if requested_cd_kkt_check_every is None:
            cd_kkt_check_every_effective = 4 if cv_method == "glmnet" else 1
        else:
            cd_kkt_check_every_effective = int(requested_cd_kkt_check_every)

        fast_fold_stats = (sample_weight_np is None) and bool(folds_are_complements)
        if fast_fold_stats:
            n_total = int(X_np.shape[0])
            XtX_full = X_np.T @ X_np
            Xty_full = X_np.T @ y_np
            if bool(fit_intercept):
                X_sum_full = np.sum(X_np, axis=0)
                y_sum_full = float(np.sum(y_np))
            else:
                X_sum_full = None
                y_sum_full = None

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_val = X_np[val_idx]
            y_val = y_np[val_idx]
            sw_val = None if sample_weight_np is None else sample_weight_np[val_idx]

            if fast_fold_stats:
                n_val = int(np.asarray(val_idx, dtype=np.int64).reshape(-1).size)
                n_train = int(n_total - n_val)

                XtX_val = X_val.T @ X_val
                Xty_val = X_val.T @ y_val
                XtX_raw = XtX_full - XtX_val
                Xty_raw = Xty_full - Xty_val

                if bool(fit_intercept):
                    X_sum_val = np.sum(X_val, axis=0)
                    y_sum_val = float(np.sum(y_val))
                    X_sum_train = X_sum_full - X_sum_val
                    y_sum_train = y_sum_full - y_sum_val

                    inv_n = 1.0 / float(max(1, n_train))
                    X_mean = X_sum_train * inv_n
                    y_mean = y_sum_train * inv_n
                    XtX = XtX_raw - np.outer(X_sum_train, X_sum_train) * inv_n
                    Xty = Xty_raw - X_sum_train * y_mean
                else:
                    X_mean = np.zeros((X_np.shape[1],), dtype=np.float64)
                    y_mean = 0.0
                    XtX = XtX_raw
                    Xty = Xty_raw
            else:
                X_train = X_np[train_idx]
                y_train = y_np[train_idx]
                sw_train = None if sample_weight_np is None else sample_weight_np[train_idx]

                if sw_train is not None:
                    sqrt_sw = np.sqrt(sw_train)
                    X_train = X_train * sqrt_sw[:, np.newaxis]
                    y_train = y_train * sqrt_sw

                if bool(fit_intercept):
                    X_mean = np.mean(X_train, axis=0)
                    y_mean = float(np.mean(y_train))
                    X_centered = X_train - X_mean
                    y_centered = y_train - y_mean
                else:
                    X_mean = np.zeros((X_train.shape[1],), dtype=np.float64)
                    y_mean = 0.0
                    X_centered = X_train
                    y_centered = y_train

                XtX = X_centered.T @ X_centered
                Xty = X_centered.T @ y_centered
                n_train = int(X_train.shape[0])

            coefs_desc, _ = _solve_lasso_path_cpu_from_gram(
                XtX,
                Xty,
                n_samples=int(n_train),
                alphas_desc=alpha_desc,
                max_iter=int(max_iter),
                tol=float(tol),
                stopping="coef_delta",
                cpu_solver=cpu_solver_name,
                lipschitz_L=None,
                cd_kkt_check_every=cd_kkt_check_every_effective,
            )

            if bool(fit_intercept):
                intercepts_desc = y_mean - X_mean @ coefs_desc.T
            else:
                intercepts_desc = np.zeros((coefs_desc.shape[0],), dtype=np.float64)

            mse_desc = _batch_mse_numpy(
                X_val,
                y_val,
                coefs_desc,
                intercepts_desc,
                sw_val,
            )

            mse_path[alpha_order_desc, fold_idx] = np.asarray(mse_desc, dtype=np.float64)

    for alpha_idx, alpha in enumerate(alpha_grid):
        alpha_f = float(alpha)
        valid = np.isfinite(mse_path[alpha_idx])
        if not bool(np.any(valid)):
            continue

        mean_mse = float(np.mean(mse_path[alpha_idx, valid]))
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = alpha_f

    mean_mse_vec = np.full(int(alpha_grid.size), np.nan, dtype=np.float64)
    for alpha_idx in range(int(alpha_grid.size)):
        valid = np.isfinite(mse_path[alpha_idx])
        if bool(np.any(valid)):
            mean_mse_vec[alpha_idx] = float(np.mean(mse_path[alpha_idx, valid]))

    details = {
        "alpha": float(best_alpha),
        "alphas": alpha_grid.astype(np.float64, copy=False),
        "mse_path": mse_path,
        "mean_mse": mean_mse_vec,
    }

    _lasso_cv_cache_put(cache_key_eff, details)

    if return_details:
        return details

    return float(details["alpha"])


class LassoCV(CVEstimatorBase):
    """
    Cross-validated Lasso built on top of statgpu's own ``Lasso`` implementation.

    This class mirrors the common sklearn-style usage pattern while keeping
    backend/device behavior consistent with statgpu models.
    """

    def __init__(
        self,
        alphas=None,
        n_alphas: int = 12,
        alpha_min_ratio: float = 1e-3,
        cv: int = 5,
        cv_splits=None,
        fit_intercept: bool = True,
        max_iter: int = 3000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        inference_method: str = "cpu_ols_inference",
        n_bootstrap: int = 200,
        bootstrap_random_state: Optional[int] = None,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        solver: str = "fista",
        cpu_solver: str = "coordinate_descent",
        method: str = "standard",
        cd_kkt_check_every: Optional[int] = None,
        lipschitz_L: Optional[float] = None,
        admm_rho: float = 1.0,
        gpu_memory_cleanup: bool = False,
        gpu_cv_mixed_precision: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            cv=cv,
            random_state=random_state,
            device=device,
            n_jobs=n_jobs,
        )
        self.alphas = alphas
        self.n_alphas = int(n_alphas)
        self.alpha_min_ratio = float(alpha_min_ratio)
        self.cv = int(cv)
        self.cv_splits = cv_splits
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.stopping = str(stopping)
        self.inference_method = str(inference_method)
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap_random_state = bootstrap_random_state
        self.compute_inference = bool(compute_inference)
        self.solver = str(solver)
        self.cpu_solver = str(cpu_solver)
        self.method = _normalize_lassocv_method(method)
        self.cd_kkt_check_every = _normalize_cd_kkt_check_every(cd_kkt_check_every)
        self.lipschitz_L = lipschitz_L
        self.admm_rho = float(admm_rho)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.gpu_cv_mixed_precision = bool(gpu_cv_mixed_precision)
        self.random_state = random_state

        self.alpha_ = None
        self.alphas_ = None
        self.mse_path_ = None
        self.mean_mse_ = None
        self.best_score_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.estimator_ = None

    def fit(self, X, y, sample_weight=None):
        device_name = self._get_compute_device().value
        effective_cpu_solver = (
            "coordinate_descent" if str(self.method).lower() == "glmnet" else str(self.cpu_solver)
        )

        details = _select_lasso_alpha_cv(
            X,
            y,
            alphas=self.alphas,
            n_alphas=self.n_alphas,
            alpha_min_ratio=self.alpha_min_ratio,
            cv_folds=self.cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            sample_weight=sample_weight,
            fit_intercept=self.fit_intercept,
            device=device_name,
            max_iter=self.max_iter,
            tol=self.tol,
            cpu_solver=effective_cpu_solver,
            method=self.method,
            cd_kkt_check_every=self.cd_kkt_check_every,
            gpu_cv_mixed_precision=self.gpu_cv_mixed_precision,
            return_details=True,
        )

        effective_cd_kkt_check_every = self.cd_kkt_check_every
        if effective_cd_kkt_check_every is None:
            effective_cd_kkt_check_every = 4 if str(self.method).lower() == "glmnet" else 1

        self.alpha_ = float(details["alpha"])
        self.alphas_ = np.asarray(details["alphas"], dtype=np.float64)
        self.mse_path_ = np.asarray(details["mse_path"], dtype=np.float64)
        self.mean_mse_ = np.asarray(details["mean_mse"], dtype=np.float64)

        if np.any(np.isfinite(self.mean_mse_)):
            self.best_score_ = float(np.nanmin(self.mean_mse_))
        else:
            self.best_score_ = np.nan

        estimator = Lasso(
            alpha=self.alpha_,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            stopping=self.stopping,
            inference_method=self.inference_method,
            n_bootstrap=self.n_bootstrap,
            bootstrap_random_state=self.bootstrap_random_state,
            device=self.device,
            n_jobs=self.n_jobs,
            compute_inference=self.compute_inference,
            solver=self.solver,
            cpu_solver=effective_cpu_solver,
            lipschitz_L=self.lipschitz_L,
            admm_rho=self.admm_rho,
            gpu_memory_cleanup=self.gpu_memory_cleanup,
        )

        fast_refit_enabled = (
            (not bool(self.compute_inference))
            and str(self.solver).lower() == "fista"
            and str(self.stopping).lower() in ("coef_delta", "kkt")
        )

        if fast_refit_enabled:
            fast = _fit_lasso_single_alpha_fast(
                X,
                y,
                alpha=float(self.alpha_),
                fit_intercept=bool(self.fit_intercept),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                stopping=str(self.stopping),
                device=str(device_name),
                cpu_solver=str(effective_cpu_solver),
                cd_kkt_check_every=int(effective_cd_kkt_check_every),
                sample_weight=sample_weight,
            )

            estimator.coef_ = np.asarray(fast["coef"], dtype=np.float64)
            estimator.intercept_ = float(fast["intercept"])
            estimator.n_iter_ = int(fast["n_iter"])
            estimator._nobs = int(fast["n_samples"])
            estimator._df_resid = int(fast["n_samples"]) - (
                int(fast["n_features"]) + (1 if bool(self.fit_intercept) else 0)
            )

            if bool(self.fit_intercept):
                estimator._params = np.concatenate(
                    [[estimator.intercept_], estimator.coef_]
                )
            else:
                estimator._params = estimator.coef_.copy()

            estimator._scale = np.nan
            estimator._resid = None
            estimator._X_design = None
            estimator._fitted = True
        else:
            estimator.fit(X, y, sample_weight=sample_weight)

        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = int(estimator.n_iter_)

        self._fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        return self.estimator_.predict(X)

    def score(self, X, y):
        self._check_is_fitted()
        return self.estimator_.score(X, y)


# =============================================================================
# Torch FISTA Solvers
# =============================================================================

def _solve_lasso_path_gpu_fista_batched_from_gram_torch(
    XtX,
    Xty,
    *,
    n_samples: int,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 8,
):
    """Solve descending-alpha Lasso path with a batched Torch FISTA update."""
    import torch

    n_features = int(XtX.shape[0])
    n_alphas = int(alphas_desc.shape[0])

    coefs = torch.zeros((n_features, n_alphas), dtype=XtX.dtype, device=XtX.device)
    yk = coefs.clone()
    tk = torch.ones((n_alphas,), dtype=XtX.dtype, device=XtX.device)
    n_iters_gpu = torch.zeros((n_alphas,), dtype=torch.int32, device=XtX.device)

    if lipschitz_L is not None:
        L = torch.tensor(float(lipschitz_L), dtype=XtX.dtype, device=XtX.device)
    else:
        try:
            eigvals = torch.linalg.eigvalsh(XtX)
            L = eigvals[-1] / float(max(1, n_samples))
        except Exception:
            row_sum_bound = torch.max(torch.sum(torch.abs(XtX), dim=1)) / float(max(1, n_samples))
            L = torch.maximum(row_sum_bound, torch.tensor(1e-12, dtype=XtX.dtype, device=XtX.device))

    L_scalar = float(L.item())
    if L_scalar <= 0.0:
        return coefs.T, torch.zeros((n_alphas,), dtype=torch.int32, device=XtX.device).cpu().numpy()

    n_samp = float(max(1, n_samples))
    step = 1.0 / L
    alphas_desc = np.asarray(alphas_desc, dtype=np.float64)
    alpha_gpu = torch.from_numpy(alphas_desc).to(XtX.device).to(XtX.dtype)
    thresholds = alpha_gpu * step
    stopping_name = str(stopping).lower()
    check_every = max(1, int(check_every))

    active_gpu = torch.arange(n_alphas, dtype=torch.int32, device=XtX.device)

    for iteration in range(int(max_iter)):
        if int(active_gpu.numel()) == 0:
            break

        y_active = yk[:, active_gpu]
        coef_old = coefs[:, active_gpu]

        grad = (XtX @ y_active - Xty.reshape(-1, 1)) / n_samp
        thresh = thresholds[active_gpu].reshape(1, -1)
        coef_new = torch.sign(y_active - step * grad) * torch.maximum(torch.abs(y_active - step * grad) - thresh, torch.tensor(0.0, dtype=XtX.dtype, device=XtX.device))

        t_old = tk[active_gpu]
        t_new = (1.0 + torch.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
        beta = (t_old - 1.0) / t_new
        y_new = coef_new + beta.reshape(1, -1) * (coef_new - coef_old)

        coefs[:, active_gpu] = coef_new
        yk[:, active_gpu] = y_new
        tk[active_gpu] = t_new

        active_ratio = float(int(active_gpu.numel())) / float(max(1, n_alphas))
        check_every_eff = _adaptive_gpu_check_every(
            base_check_every=check_every,
            iteration=iteration,
            max_iter=int(max_iter),
            active_ratio=active_ratio,
        )
        should_check = ((iteration + 1) % check_every_eff == 0) or (iteration + 1 == int(max_iter))
        if not should_check:
            continue

        if stopping_name == "kkt":
            grad_sse = (XtX @ coef_new - Xty.reshape(-1, 1)) / n_samp
            viol = torch.max(
                torch.maximum(
                    torch.abs(grad_sse) - alpha_gpu[active_gpu].reshape(1, -1),
                    torch.tensor(0.0, dtype=XtX.dtype, device=XtX.device),
                ),
                dim=0,
            ).values
            converged_local_gpu = viol < float(tol)
        else:
            delta = torch.sum(torch.abs(coef_new - coef_old), dim=0)
            converged_local_gpu = delta < float(tol)

        done_gpu = active_gpu[converged_local_gpu]
        if int(done_gpu.numel()) == 0:
            continue

        n_iters_gpu[done_gpu] = int(iteration) + 1
        yk[:, done_gpu] = coefs[:, done_gpu]
        active_gpu = active_gpu[~converged_local_gpu]

    if int(active_gpu.numel()) > 0:
        n_iters_gpu[active_gpu] = int(max_iter)

    return coefs.T, n_iters_gpu.cpu().numpy()


def _solve_lasso_path_gpu_fista_multi_fold_from_gram_torch(
    XtX_batch,
    Xty_batch,
    *,
    n_samples_vec: np.ndarray,
    alphas_desc: np.ndarray,
    max_iter: int,
    tol: float,
    stopping: str,
    lipschitz_L: Optional[float] = None,
    check_every: int = 8,
):
    """Solve descending-alpha Lasso paths for all folds together on Torch GPU."""
    import torch

    n_folds = int(XtX_batch.shape[0])
    n_features = int(XtX_batch.shape[1])
    n_alphas = int(alphas_desc.shape[0])

    coefs = torch.zeros((n_folds, n_features, n_alphas), dtype=XtX_batch.dtype, device=XtX_batch.device)
    yk = coefs.clone()
    tk = torch.ones((n_folds, n_alphas), dtype=XtX_batch.dtype, device=XtX_batch.device)
    n_iters_gpu = torch.zeros((n_folds, n_alphas), dtype=torch.int32, device=XtX_batch.device)

    n_vec_cpu = n_samples_vec.cpu().numpy().astype(np.float64).reshape(-1)
    if n_vec_cpu.size != n_folds:
        raise ValueError("n_samples_vec must have one entry per fold")
    n_vec = torch.from_numpy(n_vec_cpu).to(XtX_batch.device).to(XtX_batch.dtype)

    if lipschitz_L is not None:
        L = torch.full((n_folds,), float(lipschitz_L), dtype=XtX_batch.dtype, device=XtX_batch.device)
    else:
        try:
            eigvals = torch.linalg.eigvalsh(XtX_batch)
            L = eigvals[:, -1] / n_vec
        except Exception:
            row_sum_bound = torch.max(torch.sum(torch.abs(XtX_batch), dim=2), dim=1).values / n_vec
            L = torch.maximum(row_sum_bound, torch.tensor(1e-12, dtype=XtX_batch.dtype, device=XtX_batch.device))

    step = 1.0 / L.reshape(n_folds, 1, 1)
    alpha_gpu = torch.from_numpy(np.asarray(alphas_desc, dtype=np.float64)).to(XtX_batch.device).to(XtX_batch.dtype).reshape(1, 1, n_alphas)
    thresholds = alpha_gpu * step

    Xty_expanded = Xty_batch.reshape(n_folds, n_features, 1)
    n_vec_expanded = n_vec.reshape(n_folds, 1, 1)
    stopping_name = str(stopping).lower()
    check_every = max(1, int(check_every))

    active_gpu = torch.ones((n_folds, n_alphas), dtype=torch.bool, device=XtX_batch.device)
    active_count = int(n_folds * n_alphas)

    for iteration in range(int(max_iter)):
        if active_count == 0:
            break

        active_expanded = active_gpu.unsqueeze(1)

        coef_old = coefs.clone()
        grad = (torch.matmul(XtX_batch, yk) - Xty_expanded) / n_vec_expanded
        coef_candidate = torch.sign(yk - step * grad) * torch.maximum(torch.abs(yk - step * grad) - thresholds, torch.tensor(0.0, dtype=XtX_batch.dtype, device=XtX_batch.device))
        coefs = torch.where(active_expanded, coef_candidate, coefs)

        t_old = tk
        t_new = (1.0 + torch.sqrt(1.0 + 4.0 * (t_old ** 2))) / 2.0
        beta = (t_old - 1.0) / t_new
        y_candidate = coefs + beta.unsqueeze(1) * (coefs - coef_old)
        yk = torch.where(active_expanded, y_candidate, yk)
        tk = torch.where(active_gpu, t_new, tk)

        active_ratio = float(active_count) / float(max(1, n_folds * n_alphas))
        check_every_eff = _adaptive_gpu_check_every(
            base_check_every=check_every,
            iteration=iteration,
            max_iter=int(max_iter),
            active_ratio=active_ratio,
        )
        should_check = ((iteration + 1) % check_every_eff == 0) or (iteration + 1 == int(max_iter))
        if not should_check:
            continue

        if stopping_name == "kkt":
            grad_sse = (torch.matmul(XtX_batch, coefs) - Xty_expanded) / n_vec_expanded
            violation = torch.max(torch.maximum(torch.abs(grad_sse) - alpha_gpu, torch.tensor(0.0, dtype=XtX_batch.dtype, device=XtX_batch.device)), dim=1).values
            converged_local_gpu = violation < float(tol)
        else:
            delta = torch.sum(torch.abs(coefs - coef_old), dim=1)
            converged_local_gpu = delta < float(tol)

        newly_done_gpu = active_gpu & converged_local_gpu
        done_count = int(torch.count_nonzero(newly_done_gpu).item())
        if done_count == 0:
            continue

        n_iters_gpu[newly_done_gpu] = int(iteration) + 1
        yk = torch.where(newly_done_gpu.unsqueeze(1), coefs, yk)
        active_gpu = active_gpu & (~converged_local_gpu)
        active_count -= done_count

    n_iters_gpu[active_gpu] = int(max_iter)

    return coefs.permute(0, 2, 1), n_iters_gpu.cpu().numpy()

    def summary(self):
        self._check_is_fitted()
        return self.estimator_.summary()
