"""
Elastic Net regression with GPU acceleration and full statistical inference.

Elastic Net combines L1 and L2 regularization:
    minimize (1/(2n)) * ||y - Xw||²₂ + α * l1_ratio * ||w||₁ + 0.5 * α * (1 - l1_ratio) * ||w||²₂

where:
- α (alpha) controls the overall regularization strength
- l1_ratio controls the mix: 1.0 = Lasso, 0.0 = Ridge, 0.5 = balanced Elastic Net

Optimized implementations:
- CPU: FISTA with pre-computed Gram matrix
- GPU (CuPy): Fused kernel operations with @cp.fuse()
- GPU (Torch): torch.compile() with warm-up strategy
"""

from typing import Optional, Union
import warnings
import numpy as np

from .._base import BaseEstimator
from .._config import Device


# Lazy import optimized implementations
def _get_cupy_optimized():
    """Lazy import of CuPy optimized implementation."""
    try:
        from ._elasticnet_cupy_optimized import fit_elasticnet_optimized
        return fit_elasticnet_optimized
    except ImportError:
        return None


def _get_torch_optimized():
    """Lazy import of Torch optimized implementation."""
    try:
        from ._elasticnet_torch_optimized import fit_elasticnet_torch_optimized
        return fit_elasticnet_torch_optimized
    except ImportError:
        return None


class ElasticNet(BaseEstimator):
    """
    Elastic Net regression with GPU acceleration.

    Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization, controlled by
    the `l1_ratio` parameter. This provides:
    - Feature selection from L1 (sparse solutions)
    - Grouping effect from L2 (handles correlated features)

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger values specify stronger regularization.
        Must be non-negative.
    l1_ratio : float, default=0.5
        Elastic Net mixing parameter, between 0 and 1 inclusive.
        - l1_ratio = 1: L1 penalty only (Lasso)
        - l1_ratio = 0: L2 penalty only (Ridge)
        - 0 < l1_ratio < 1: Combination of L1 and L2 penalties
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    tol : float, default=1e-4
        Tolerance for convergence.
    stopping : str, default='coef_delta'
        Stopping criterion: 'coef_delta' or 'kkt'.
    device : str or Device, default='auto'
        Computation device: 'cpu', 'cuda', or 'auto'.
    solver : str, default='fista'
        GPU optimization algorithm: 'fista' or 'admm'.
        Note: ADMM not yet implemented for Elastic Net.
    cpu_solver : str, default='fista'
        CPU optimization algorithm: 'fista' or 'coordinate_descent'.
        Note: coordinate_descent not yet implemented for Elastic Net.
    lipschitz_L : float, optional
        Pre-computed Lipschitz constant. If not provided, will be estimated.
    gpu_memory_cleanup : bool, default=False
        If True, free GPU memory pool after fitting.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term.
    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    Lasso : Lasso regression with L1 regularization.
    Ridge : Ridge regression with L2 regularization.

    Notes
    -----
    The objective function is:

        (1 / (2 * n_samples)) * ||y - Xw||²₂ + α * l1_ratio * ||w||₁ + 0.5 * α * (1 - l1_ratio) * ||w||²₂

    References
    ----------
    .. [1] Zou, H., & Hastie, T. (2005). Regularization and variable selection
           via the elastic net. Journal of the Royal Statistical Society:
           Series B, 67(2), 301-320.
    .. [2] Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
           algorithm for linear inverse problems. SIAM Journal on Imaging Sciences,
           2(1), 183-202.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.linear_model import ElasticNet
    >>> X = np.random.randn(100, 10)
    >>> y = X @ np.random.randn(10) + np.random.randn(100)
    >>> model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    >>> model.fit(X, y)
    >>> print(model.coef_)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        stopping: str = "coef_delta",
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        solver: str = "fista",
        cpu_solver: str = "fista",
        lipschitz_L: Optional[float] = None,
        gpu_memory_cleanup: bool = False,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.stopping = stopping.lower()
        self.solver = solver.lower()
        self.cpu_solver = cpu_solver.lower()
        self.lipschitz_L = lipschitz_L
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)

        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0

        # Internal storage
        self._params = None
        self._scale = None
        self._df_resid = None
        self._nobs = None
        self._X_design = None
        self._resid = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit Elastic Net model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self : ElasticNet
            Fitted estimator.
        """
        device = self._get_compute_device()
        backend = self._get_backend(backend="auto")
        backend_name = backend.name

        X_arr = self._to_array(X, backend=backend_name)
        y_arr = self._to_array(y, backend=backend_name)

        # Route to appropriate backend
        if backend_name == "torch":
            self._fit_torch(X_arr, y_arr, sample_weight)
        elif device == Device.CUDA:
            self._fit_gpu(X_arr, y_arr, sample_weight)
        else:
            self._fit_cpu(X_arr, y_arr, sample_weight)

        self._fitted = True
        return self

    def predict(self, X):
        """
        Predict using Elastic Net model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = np.asarray(X)
        y_pred = X @ self.coef_
        if self.fit_intercept:
            y_pred += self.intercept_
        return y_pred

    def score(self, X, y):
        """
        Return the coefficient of determination R².

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        r2 : float
            R² score.
        """
        y_pred = self._to_numpy(self.predict(X))
        y = self._to_numpy(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _soft_threshold(self, x, gamma):
        """Standard soft thresholding operator for Lasso."""
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    def _soft_threshold_elastic(self, x, gamma, l2_scale):
        """
        Elastic Net soft thresholding operator.

        Applies soft thresholding then divides by L2 scaling factor.
        This is the proximal operator for L1 + L2 regularization.

        Parameters
        ----------
        x : ndarray
            Input array
        gamma : float
            Threshold parameter (alpha * l1_ratio * step)
        l2_scale : float
            L2 scaling factor (1 + alpha * (1 - l1_ratio) * step)

        Returns
        -------
        ndarray
            Soft thresholded and scaled result
        """
        return self._soft_threshold(x, gamma) / l2_scale

    def _fit_cpu(self, X, y, sample_weight=None):
        """
        Fit using CPU FISTA solver.

        Elastic Net proximal gradient update:
          grad = (XtX @ w - Xty) / n        # gradient of RSS only
          w = soft_threshold(w - step*grad, α*l1_ratio*step) / (1 + α*(1-l1_ratio)*step)

        Note: L2 regularization is handled in the proximal step, NOT in the gradient.
        """
        # Use optimized CPU implementation
        self._fit_cpu_optimized(X, y, sample_weight)

    def _fit_cpu_optimized(self, X, y, sample_weight=None):
        """
        Optimized CPU FISTA solver with pre-computed Gram matrix.
        """
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

        # Precompute XtX and Xty for FISTA gradient
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered.flatten()

        # Elastic Net parameters
        alpha = float(self.alpha)
        l1_ratio = float(self.l1_ratio)
        l2_ratio = 1.0 - l1_ratio

        # Lipschitz constant: L = λ_max(XtX)/n (RSS only, L2 is handled in proximal step)
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            try:
                eig_max = np.linalg.eigvalsh(XtX)[-1]
                L = float(eig_max / n_samples)
            except Exception:
                L_frob = float(np.sum(X_centered ** 2) / n_samples)
                L = L_frob

        if L <= 0:
            # Degenerate case: apply pure proximal operator
            thresh = alpha * l1_ratio
            l2_scale = 1.0 + alpha * l2_ratio
            coef = self._soft_threshold_elastic(np.zeros(n_features), thresh, l2_scale)
            self.n_iter_ = 0
        else:
            step = 1.0 / L

            # Elastic Net proximal parameters
            thresh = alpha * l1_ratio * step
            l2_scale = 1.0 + alpha * l2_ratio * step

            # FISTA variables
            coef = np.zeros(n_features)
            y_k = coef.copy()
            t_k = 1.0

            for iteration in range(self.max_iter):
                coef_old = coef.copy()

                # Gradient of RSS ONLY (L2 is handled in proximal step)
                grad = (XtX @ y_k - Xty) / n_samples

                # Proximal gradient step with Elastic Net soft thresholding
                w_tilde = y_k - step * grad
                coef = self._soft_threshold_elastic(w_tilde, thresh, l2_scale)

                # Momentum update (FISTA)
                t_new = (1.0 + np.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
                beta = (t_k - 1.0) / t_new
                y_k = coef + beta * (coef - coef_old)
                t_k = t_new

                # Convergence test
                if self.stopping == "kkt":
                    # KKT violation for Elastic Net
                    # Optimality condition: 0 ∈ (XtX @ coef - Xty)/n + α*l1_ratio*∂||coef||_1 + α*l2_ratio*coef
                    grad_rss = (XtX @ coef - Xty) / n_samples
                    # L2 gradient
                    grad_l2 = alpha * l2_ratio * coef
                    # L1 subgradient
                    sign_coef = np.sign(coef)
                    sign_coef[coef == 0] = 0
                    kkt_violation = np.zeros(n_features)
                    for j in range(n_features):
                        if coef[j] != 0:
                            # For non-zero coefficients: gradient + L1 subgradient should be zero
                            kkt_violation[j] = np.abs(grad_rss[j] + grad_l2[j] + alpha * l1_ratio * sign_coef[j])
                        else:
                            # For zero coefficients: |gradient| should be <= alpha * l1_ratio
                            kkt_violation[j] = max(0, np.abs(grad_rss[j] + grad_l2[j]) - alpha * l1_ratio)
                    violation = np.max(kkt_violation)
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

    def _soft_threshold_cupy(self, x, gamma, l2_scale=None):
        """Soft thresholding operator for CuPy arrays."""
        import cupy as cp
        if l2_scale is not None:
            return cp.sign(x) * cp.maximum(cp.abs(x) - gamma, 0) / l2_scale
        return cp.sign(x) * cp.maximum(cp.abs(x) - gamma, 0)

    def _cleanup_cuda_memory(self):
        """Free CuPy memory pool."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _fit_gpu(self, X, y, sample_weight=None):
        """
        Fit using GPU (CuPy) with FISTA solver.

        Uses optimized implementation with fused kernels if available.
        """
        import cupy as cp

        if self.solver not in ("fista",):
            raise ValueError("Elastic Net currently only supports 'fista' solver")

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

        # Try to use optimized implementation
        fit_optimized = _get_cupy_optimized()
        if fit_optimized is not None:
            # Use optimized implementation with fused kernels
            coef, self.n_iter_ = fit_optimized(
                X=X_centered,
                y=y_centered,
                alpha=float(self.alpha),
                l1_ratio=float(self.l1_ratio),
                n_samples=n_samples,
                n_features=n_features,
                max_iter=self.max_iter,
                tol=self.tol,
                lipschitz_L=self.lipschitz_L,
                stopping=self.stopping,
                warmup=True  # Enable warm-up to avoid JIT overhead
            )
        else:
            # Fallback to base implementation
            coef, self.n_iter_ = self._fit_gpu_base(
                X_centered, y_centered, n_samples, n_features
            )

        # Build full coefficients
        if self.fit_intercept:
            intercept_gpu = y_mean - X_mean @ coef
            coef_full = cp.concatenate([intercept_gpu.reshape(1), coef])
        else:
            coef_full = coef

        # Transfer to CPU
        coef_full_np = coef_full.get()

        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

        # Cleanup
        self._cleanup_cuda_memory()

    def _fit_gpu_base(self, X_centered, y_centered, n_samples, n_features):
        """
        Base GPU implementation (fallback).
        """
        import cupy as cp

        # Precompute XtX / Xty
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Elastic Net parameters
        alpha = float(self.alpha)
        l1_ratio = float(self.l1_ratio)
        l2_ratio = 1.0 - l1_ratio

        # Lipschitz constant: L = λ_max(XtX)/n (RSS only)
        if self.lipschitz_L is not None:
            L = cp.array(float(self.lipschitz_L), dtype=X_centered.dtype)
        else:
            try:
                eigvals = cp.linalg.eigvalsh(XtX)
                L = eigvals[-1] / n_samples
            except Exception:
                L_frob = cp.sum(X_centered ** 2) / n_samples
                L = L_frob

        if L <= 0:
            return cp.zeros(n_features, dtype=X_centered.dtype), 0

        step = 1.0 / L
        thresh = alpha * l1_ratio * step
        l2_scale = 1.0 + alpha * l2_ratio * step

        # FISTA variables
        coef = cp.zeros(n_features, dtype=X_centered.dtype)
        y_k = coef.copy()
        t_k = cp.array(1.0, dtype=X_centered.dtype)

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            # Gradient of RSS ONLY (L2 is handled in proximal step)
            grad = (XtX @ y_k - Xty) / n_samples

            # Proximal step
            w_tilde = y_k - step * grad
            coef = self._soft_threshold_elastic_cupy(w_tilde, thresh, l2_scale)

            # Momentum update
            t_new = (1 + cp.sqrt(1 + 4 * (t_k ** 2))) / 2
            beta = (t_k - 1) / t_new
            y_k = coef + beta * (coef - coef_old)
            t_k = t_new

            # Convergence test
            if self.stopping == "kkt":
                grad_rss = (XtX @ coef - Xty) / n_samples
                grad_l2 = alpha * l2_ratio * coef
                sign_coef = cp.sign(coef)
                sign_coef[coef == 0] = 0
                kkt_nonzero = cp.abs(grad_rss + grad_l2 + alpha * l1_ratio * sign_coef)
                kkt_zero = cp.maximum(cp.abs(grad_rss + grad_l2) - alpha * l1_ratio, 0)
                kkt_violation = cp.where(coef != 0, kkt_nonzero, kkt_zero)
                violation = cp.max(kkt_violation)
                if violation < self.tol:
                    return coef, iteration + 1
            else:
                if cp.sum(cp.abs(coef - coef_old)) < self.tol:
                    return coef, iteration + 1

        return coef, self.max_iter

    def _soft_threshold_elastic_cupy(self, x, gamma, l2_scale):
        """Elastic Net soft thresholding for CuPy."""
        import cupy as cp
        return cp.sign(x) * cp.maximum(cp.abs(x) - gamma, 0) / l2_scale

    def _soft_threshold_torch(self, x, gamma, l2_scale=None):
        """Soft thresholding operator for Torch tensors."""
        import torch
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if l2_scale is not None:
            return torch.sign(x) * torch.maximum(torch.abs(x) - gamma, zero) / l2_scale
        return torch.sign(x) * torch.maximum(torch.abs(x) - gamma, zero)

    def _soft_threshold_elastic_torch(self, x, gamma, l2_scale):
        """Elastic Net soft thresholding for Torch."""
        import torch
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.sign(x) * torch.maximum(torch.abs(x) - gamma, zero) / l2_scale

    def _cleanup_torch_memory(self):
        """Free Torch memory pool."""
        if not self.gpu_memory_cleanup:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _fit_torch(self, X, y, sample_weight=None):
        """
        Fit using Torch GPU with FISTA solver.

        Uses optimized implementation with torch.compile() if available.
        """
        import torch

        if self.solver not in ("fista",):
            raise ValueError("Torch backend currently only supports 'fista' solver")

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

        # Try to use optimized implementation
        fit_optimized = _get_torch_optimized()
        if fit_optimized is not None:
            # Use optimized implementation with torch.compile()
            coef, self.n_iter_ = fit_optimized(
                X=X_centered,
                y=y_centered,
                alpha=float(self.alpha),
                l1_ratio=float(self.l1_ratio),
                n_samples=n_samples,
                n_features=n_features,
                max_iter=self.max_iter,
                tol=self.tol,
                lipschitz_L=self.lipschitz_L,
                stopping=self.stopping,
                warmup=True  # Enable warm-up to avoid JIT overhead
            )
        else:
            # Fallback to base implementation
            coef, self.n_iter_ = self._fit_torch_base(
                X_centered, y_centered, n_samples, n_features
            )

        # Build full coefficients
        if self.fit_intercept:
            intercept_torch = y_mean - X_mean @ coef
            coef_full = torch.cat([intercept_torch.reshape(1), coef])
        else:
            coef_full = coef

        # Transfer to CPU
        coef_full_np = coef_full.cpu().numpy()

        if self.fit_intercept:
            self.intercept_ = float(coef_full_np[0])
            self.coef_ = coef_full_np[1:]
            self._params = coef_full_np
        else:
            self.intercept_ = 0.0
            self.coef_ = coef_full_np
            self._params = coef_full_np

        self._df_resid = n_samples - (n_features + (1 if self.fit_intercept else 0))

        # Cleanup
        self._cleanup_torch_memory()

    def _fit_torch_base(self, X_centered, y_centered, n_samples, n_features):
        """
        Base Torch implementation (fallback).
        """
        import torch

        # Precompute XtX / Xty
        XtX = X_centered.T @ X_centered
        Xty = X_centered.T @ y_centered

        # Elastic Net parameters
        alpha = float(self.alpha)
        l1_ratio = float(self.l1_ratio)
        l2_ratio = 1.0 - l1_ratio

        # Lipschitz constant: L = λ_max(XtX)/n (RSS only)
        if self.lipschitz_L is not None:
            L = torch.tensor(float(self.lipschitz_L), dtype=X_centered.dtype, device=X_centered.device)
        else:
            try:
                eigvals = torch.linalg.eigvalsh(XtX)
                L = eigvals[-1] / n_samples
            except Exception:
                L_frob = torch.sum(X_centered ** 2) / n_samples
                L = L_frob

        if L <= 0:
            return torch.zeros(n_features, dtype=X_centered.dtype, device=X_centered.device), 0

        step = 1.0 / L
        thresh = alpha * l1_ratio * step
        l2_scale = 1.0 + alpha * l2_ratio * step

        # FISTA variables
        coef = torch.zeros(n_features, dtype=X_centered.dtype, device=X_centered.device)
        y_k = coef.clone()
        t_k = torch.tensor(1.0, dtype=X_centered.dtype, device=X_centered.device)

        for iteration in range(self.max_iter):
            coef_old = coef.clone()

            # Gradient of RSS ONLY (L2 is handled in proximal step)
            grad = (XtX @ y_k - Xty) / n_samples

            # Proximal step
            w_tilde = y_k - step * grad
            coef = self._soft_threshold_elastic_torch(w_tilde, thresh, l2_scale)

            # Momentum update
            t_new = (1.0 + torch.sqrt(1.0 + 4.0 * (t_k ** 2))) / 2.0
            beta = (t_k - 1.0) / t_new
            y_k = coef + beta * (coef - coef_old)
            t_k = t_new

            # Convergence test
            if self.stopping == "kkt":
                grad_rss = (XtX @ coef - Xty) / n_samples
                grad_l2 = alpha * l2_ratio * coef
                sign_coef = torch.sign(coef)
                sign_coef[coef == 0] = 0
                kkt_nonzero = torch.abs(grad_rss + grad_l2 + alpha * l1_ratio * sign_coef)
                kkt_zero = torch.maximum(torch.abs(grad_rss + grad_l2) - alpha * l1_ratio, torch.tensor(0.0, dtype=X_centered.dtype, device=X_centered.device))
                violation = torch.max(torch.where(coef != 0, kkt_nonzero, kkt_zero))
                if violation < self.tol:
                    return coef, iteration + 1
            else:
                if torch.sum(torch.abs(coef - coef_old)) < self.tol:
                    return coef, iteration + 1

        return coef, self.max_iter
