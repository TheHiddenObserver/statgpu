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

from statgpu._base import BaseEstimator
from statgpu._config import Device


# ============================================================================
# CuPy Fused Kernels for Elastic Net
# ============================================================================

def _get_cupy_fused_kernels():
    """Lazy load CuPy fused kernels."""
    try:
        import cupy as cp
    except ImportError:
        return None, None, None, None

    @cp.fuse()
    def _elastic_net_proximal(x, thresh, l2_scale):
        """Fused soft thresholding with L2 scaling."""
        return cp.sign(x) * cp.maximum(cp.abs(x) - thresh, 0) / l2_scale

    @cp.fuse()
    def _fista_momentum_update(coef, coef_old, t_old, t_new):
        """Fused FISTA momentum update."""
        beta = (t_old - 1) / t_new
        return coef + beta * (coef - coef_old)

    @cp.fuse()
    def _compute_coef_delta(coef, coef_old):
        """Compute absolute coefficient change."""
        return cp.abs(coef - coef_old)

    ELASTIC_NET_PROXIMAL_KERNEL = cp.ElementwiseKernel(
        'float64 w_tilde, float64 thresh, float64 l2_scale',
        'float64 coef',
        '''
        double abs_w = abs(w_tilde);
        if (abs_w > thresh) {
            coef = (w_tilde > 0 ? 1.0 : -1.0) * (abs_w - thresh) / l2_scale;
        } else {
            coef = 0.0;
        }
        ''',
        'elastic_net_proximal'
    )

    return _elastic_net_proximal, _fista_momentum_update, _compute_coef_delta, ELASTIC_NET_PROXIMAL_KERNEL


def _fit_elasticnet_cupy_optimized(X, y, alpha, l1_ratio, n_samples, n_features,
                                    max_iter=1000, tol=1e-4, lipschitz_L=None,
                                    stopping='coef_delta', warmup=True):
    """
    Fit Elastic Net using optimized CuPy operations with fused kernels.
    """
    import cupy as cp

    # Get fused kernels
    _elastic_net_proximal, _fista_momentum_update, _compute_coef_delta, _ = _get_cupy_fused_kernels()
    if _elastic_net_proximal is None:
        raise ImportError("CuPy not available")

    # Precompute Gram matrix and cross product
    XtX = X.T @ X
    Xty = X.T @ y

    # Parameters
    l2_ratio = 1.0 - l1_ratio

    # Lipschitz constant: L = lambda_max(XtX) / n
    if lipschitz_L is not None:
        L = float(lipschitz_L)
    else:
        eigvals = cp.linalg.eigvalsh(XtX)
        L = float(eigvals[-1]) / n_samples

    if L <= 0:
        return cp.zeros(n_features), 0

    step = 1.0 / L
    thresh = alpha * l1_ratio * step
    l2_scale = 1.0 + alpha * l2_ratio * step

    # Pre-compute inverse for multiplication (faster than division)
    inv_n_samples = 1.0 / n_samples
    inv_l2_scale = 1.0 / l2_scale

    # Allocate buffers (reuse to minimize allocation overhead)
    coef = cp.zeros(n_features, dtype=X.dtype)
    y_k = cp.zeros(n_features, dtype=X.dtype)
    coef_old = cp.zeros(n_features, dtype=X.dtype)
    grad = cp.empty(n_features, dtype=X.dtype)
    w_tilde = cp.empty(n_features, dtype=X.dtype)

    # FISTA state
    t_k = 1.0
    n_iter = 0

    # Warm-up: Call fused kernel once to trigger JIT compilation
    if warmup:
        _ = _elastic_net_proximal(w_tilde, thresh, l2_scale)
        _ = (1.0 + cp.sqrt(1.0 + 4.0 * t_k * t_k)) * 0.5

    for iteration in range(max_iter):
        # Store old coefficients for convergence check
        coef_old[:] = coef

        # Gradient step: grad = (XtX @ y_k - Xty) / n
        grad = XtX @ y_k
        grad -= Xty
        grad *= inv_n_samples

        # Proximal step: w_tilde = y_k - step * grad
        w_tilde = y_k - step * grad

        # Soft thresholding with L2 scaling (using fused kernel)
        coef = _elastic_net_proximal(w_tilde, thresh, l2_scale)

        # FISTA momentum update
        t_new = (1.0 + cp.sqrt(1.0 + 4.0 * t_k * t_k)) * 0.5
        beta = (t_k - 1.0) / t_new
        y_k = coef + beta * (coef - coef_old)
        t_k = t_new

        n_iter = iteration + 1

        # Convergence check
        if stopping == 'kkt':
            kkt_grad = XtX @ coef
            kkt_grad -= Xty
            kkt_grad *= inv_n_samples

            grad_l2 = alpha * l2_ratio * coef
            sign_coef = cp.sign(coef)
            sign_coef[coef == 0] = 0

            kkt_violation = cp.maximum(
                cp.abs(kkt_grad + grad_l2 + alpha * l1_ratio * sign_coef),
                cp.maximum(cp.abs(kkt_grad + grad_l2) - alpha * l1_ratio, 0)
            )
            violation = float(cp.max(kkt_violation))
        else:
            delta = cp.abs(coef - coef_old)
            violation = float(cp.max(delta))

        if violation < tol:
            break

    return coef, n_iter


# ============================================================================
# Torch Compiled Kernels for Elastic Net
# ============================================================================

def _get_torch_compiled_proximal():
    """Lazy load torch.compile proximal operator."""
    try:
        import torch
    except ImportError:
        return None

    def _elastic_net_proximal_torch(w_tilde, thresh, l2_scale):
        """Soft thresholding with L2 scaling for Elastic Net."""
        return torch.sign(w_tilde) * torch.maximum(
            torch.abs(w_tilde) - thresh,
            torch.tensor(0.0, device=w_tilde.device, dtype=w_tilde.dtype)
        ) / l2_scale

    # Compile the proximal operator
    try:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.guard_immutable_object = False
        _elastic_net_proximal_compiled = torch.compile(
            _elastic_net_proximal_torch, mode='reduce-overhead'
        )
    except (AttributeError, RuntimeError):
        _elastic_net_proximal_compiled = _elastic_net_proximal_torch

    return _elastic_net_proximal_compiled


def _fit_elasticnet_torch_optimized(X, y, alpha, l1_ratio, n_samples, n_features,
                                     max_iter=1000, tol=1e-4, lipschitz_L=None,
                                     stopping='coef_delta', warmup=True):
    """
    Fit Elastic Net using optimized PyTorch operations with torch.compile().
    """
    import torch

    # Get compiled proximal operator
    _elastic_net_proximal_compiled = _get_torch_compiled_proximal()
    if _elastic_net_proximal_compiled is None:
        raise ImportError("Torch not available")

    # Precompute Gram matrix and cross product
    XtX = X.T @ X
    Xty = X.T @ y

    # Parameters
    l2_ratio = 1.0 - l1_ratio

    # Lipschitz constant: L = lambda_max(XtX) / n
    if lipschitz_L is not None:
        L = float(lipschitz_L)
    else:
        eigvals = torch.linalg.eigvalsh(XtX)
        L = float(eigvals[-1]) / n_samples

    if L <= 0:
        return torch.zeros(n_features, device=X.device, dtype=X.dtype), 0

    step = 1.0 / L
    thresh = alpha * l1_ratio * step
    l2_scale = 1.0 + alpha * l2_ratio * step

    # Pre-compute inverse for multiplication (faster than division)
    inv_n_samples = 1.0 / n_samples
    inv_l2_scale = 1.0 / l2_scale

    # Allocate buffers (reuse to minimize allocation overhead)
    coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
    y_k = torch.zeros(n_features, dtype=X.dtype, device=X.device)
    coef_old = torch.zeros(n_features, dtype=X.dtype, device=X.device)
    grad = torch.empty(n_features, dtype=X.dtype, device=X.device)
    w_tilde = torch.empty(n_features, dtype=X.dtype, device=X.device)

    # FISTA state
    t_k = 1.0
    n_iter = 0

    # Warm-up: Call compiled kernel once to trigger JIT compilation
    if warmup:
        _ = _elastic_net_proximal_compiled(w_tilde, thresh, l2_scale)
        _ = (1.0 + torch.sqrt(1.0 + 4.0 * t_k * t_k)) * 0.5

    for iteration in range(max_iter):
        # Store old coefficients for convergence check
        coef_old.copy_(coef)

        # Gradient step: grad = (XtX @ y_k - Xty) / n
        torch.matmul(XtX, y_k, out=grad)
        grad -= Xty
        grad *= inv_n_samples

        # Proximal step: w_tilde = y_k - step * grad
        torch.subtract(y_k, grad, alpha=step, out=w_tilde)

        # Soft thresholding with L2 scaling (using compiled fused kernel)
        coef = _elastic_net_proximal_compiled(w_tilde, thresh, l2_scale)

        # FISTA momentum update
        t_new = (1.0 + torch.sqrt(1.0 + 4.0 * t_k * t_k)) * 0.5
        beta = (t_k - 1.0) / t_new
        y_k = coef + beta * (coef - coef_old)
        t_k = t_new

        n_iter = iteration + 1

        # Convergence check
        if stopping == 'kkt':
            kkt_grad = torch.matmul(XtX, coef, out=grad)
            kkt_grad -= Xty
            kkt_grad *= inv_n_samples

            grad_l2 = alpha * l2_ratio * coef
            sign_coef = torch.sign(coef)
            sign_coef[coef == 0] = 0

            kkt_violation = torch.maximum(
                torch.abs(kkt_grad + grad_l2 + alpha * l1_ratio * sign_coef),
                torch.maximum(
                    torch.abs(kkt_grad + grad_l2) - alpha * l1_ratio,
                    torch.tensor(0.0, device=X.device)
                )
            )
            violation = float(torch.max(kkt_violation).item())
        else:
            delta = torch.abs(coef - coef_old)
            violation = float(torch.max(delta).item())

        if violation < tol:
            break

    return coef, n_iter


# ============================================================================
# Elastic Net Estimator Class
# ============================================================================

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

    def fit(self, X, y, sample_weight=None, initial_coef=None):
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
        initial_coef : array-like of shape (n_features,), optional
            Initial coefficient vector for warm-start. When fitting along a
            regularization path (alphas from large to small), passing the
            previous solution can significantly reduce iterations.

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
            self._fit_cpu(X_arr, y_arr, sample_weight, initial_coef=initial_coef)

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

        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp
            X_gpu = cp.asarray(self._to_array(X, Device.CUDA))
            coef_gpu = cp.asarray(self.coef_)
            y_pred = X_gpu @ coef_gpu
            if self.fit_intercept:
                y_pred += cp.asarray(self.intercept_, dtype=coef_gpu.dtype)
            return y_pred
        if device == Device.TORCH:
            import torch
            X_torch = self._to_array(X, Device.TORCH, backend="torch").to(torch.float64)
            coef_torch = torch.as_tensor(self.coef_, dtype=X_torch.dtype, device=X_torch.device)
            y_pred = X_torch @ coef_torch
            if self.fit_intercept:
                y_pred = y_pred + torch.as_tensor(
                    self.intercept_, dtype=y_pred.dtype, device=y_pred.device
                )
            return y_pred

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
        y_pred = self.predict(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            yb = cp.asarray(self._to_array(y, Device.CUDA))
            ss_res = cp.sum((yb - y_pred) ** 2)
            ss_tot = cp.sum((yb - cp.mean(yb)) ** 2)
            return float((1 - ss_res / ss_tot).item()) if float(ss_tot.item()) > 0 else 0.0
        if device == Device.TORCH:
            import torch

            yb = self._to_array(y, Device.TORCH, backend="torch").to(y_pred.dtype)
            ss_res = torch.sum((yb - y_pred) ** 2)
            ss_tot = torch.sum((yb - torch.mean(yb)) ** 2)
            return float((1 - ss_res / ss_tot).item()) if float(ss_tot.item()) > 0 else 0.0
        y_pred = np.asarray(y_pred)
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

    def _fit_cpu(self, X, y, sample_weight=None, initial_coef=None):
        """
        Fit using CPU FISTA solver with optimized implementation.

        Elastic Net proximal gradient update:
          grad = (XtX @ w - Xty) / n        # gradient of RSS only
          w = soft_threshold(w - step*grad, alpha*l1_ratio*step) / (1 + alpha*(1-l1_ratio)*step)

        Note: L2 regularization is handled in the proximal step, NOT in the gradient.

        Parameters
        ----------
        X : ndarray
            Training data (n_samples, n_features).
        y : ndarray
            Target values (n_samples,).
        sample_weight : ndarray, optional
            Sample weights.
        initial_coef : ndarray, optional
            Initial coefficient vector for warm-start. If provided, avoids starting from zero
            and can significantly speed up convergence along a regularization path.
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
            # Memory-efficient centering: avoid creating full X_centered (n×p) matrix
            XtX = X.T @ X - n_samples * np.outer(X_mean, X_mean)
            Xty = X.T @ y - n_samples * X_mean * y_mean
        else:
            y_mean = 0.0
            XtX = X.T @ X
            Xty = X.T @ y

        if Xty.ndim == 0:
            Xty = Xty.reshape(1)
        if Xty.ndim == 1:
            Xty = Xty.reshape(-1, 1)
        Xty_flat = Xty.flatten()

        # Elastic Net parameters
        alpha = float(self.alpha)
        l1_ratio = float(self.l1_ratio)
        l2_ratio = 1.0 - l1_ratio

        # Lipschitz constant: L = lambda_max(XtX)/n (RSS only, L2 is handled in proximal step)
        if self.lipschitz_L is not None:
            L = float(self.lipschitz_L)
        else:
            try:
                eig_max = np.linalg.eigvalsh(XtX)[-1]
                L = float(eig_max / n_samples)
            except Exception:
                # Frobenius norm squared / n = trace(XtX) / n = sum(X_centered^2) / n
                L = float(np.trace(XtX) / n_samples)

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
            inv_l2_scale = 1.0 / l2_scale
            inv_n_samples = 1.0 / n_samples

            # FISTA variables - use warm-start if available
            if initial_coef is not None and len(initial_coef) == n_features:
                coef = np.asarray(initial_coef, dtype=np.float64).copy()
            else:
                coef = np.zeros(n_features)
            y_k = coef.copy()
            t_k = 1.0

            # Pre-allocate buffers to reduce allocation overhead
            coef_old = np.empty_like(coef)
            grad = np.empty_like(coef)
            w_tilde = np.empty_like(coef)
            delta = np.empty_like(coef)

            for iteration in range(self.max_iter):
                # Store old coefficients (in-place copy)
                coef_old[:] = coef

                # Gradient of RSS ONLY (L2 is handled in proximal step)
                # grad = (XtX @ y_k - Xty) / n_samples
                np.matmul(XtX, y_k, out=grad)
                grad -= Xty_flat
                grad *= inv_n_samples

                # Proximal gradient step with Elastic Net soft thresholding
                # w_tilde = y_k - step * grad
                np.subtract(y_k, step * grad, out=w_tilde)

                # coef = soft_threshold(w_tilde, thresh) / l2_scale
                # Using vectorized operations with pre-computed inv_l2_scale
                np.abs(w_tilde, out=delta)
                np.maximum(delta - thresh, 0, out=delta)
                coef[:] = np.sign(w_tilde) * delta * inv_l2_scale

                # Momentum update (FISTA)
                sqrt_arg = 1.0 + 4.0 * t_k * t_k
                t_new = (1.0 + np.sqrt(sqrt_arg)) * 0.5
                beta = (t_k - 1.0) / t_new
                # y_k = coef + beta * (coef - coef_old)
                np.subtract(coef, coef_old, out=y_k)
                y_k *= beta
                y_k += coef
                t_k = t_new

                # Convergence test - use L-infinity norm of coefficient change
                np.abs(coef - coef_old, out=delta)
                violation = float(np.max(delta))

                if violation < self.tol:
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
        Fit using GPU (CuPy) with optimized FISTA solver and fused kernels.
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

        # Use optimized implementation with fused kernels
        coef, self.n_iter_ = _fit_elasticnet_cupy_optimized(
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
        Fit using Torch GPU with optimized FISTA solver and torch.compile().
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

        # Use optimized implementation with torch.compile()
        coef, self.n_iter_ = _fit_elasticnet_torch_optimized(
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


# =============================================================================
# V9 thin wrapper
# =============================================================================

from ._penalized import PenalizedLinearRegression as _PenalizedLinearRegression


class ElasticNet(_PenalizedLinearRegression):
    """Thin sklearn-style wrapper over ``PenalizedLinearRegression`` with Elastic Net penalty."""

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
        **kwargs,
    ):
        self.stopping = str(stopping).lower()
        self._ignored_kwargs = dict(kwargs)
        super().__init__(
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            device=device,
            n_jobs=n_jobs,
            solver=solver,
            cpu_solver=cpu_solver,
            lipschitz_L=lipschitz_L,
            gpu_memory_cleanup=gpu_memory_cleanup,
            stopping=stopping,
        )

    def fit(self, X=None, y=None, sample_weight=None, initial_coef=None, **kwargs):
        """Fit Elastic Net model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        initial_coef : array-like of shape (n_features,), optional
            Warm-start coefficients. Passed to the underlying solver.
        """
        if initial_coef is not None:
            self.init_coef = initial_coef
        return super().fit(X=X, y=y, sample_weight=sample_weight, **kwargs)
