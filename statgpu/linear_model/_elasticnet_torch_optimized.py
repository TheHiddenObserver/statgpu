"""
PyTorch Kernel Fusion Optimization for Elastic Net FISTA

This module provides optimized PyTorch operations using:
1. torch.compile() for kernel fusion (PyTorch 2.0+)
2. In-place operations to reduce memory allocation
3. Warm-up to trigger JIT compilation

Key optimizations:
- Fused element-wise operations via torch.compile
- Pre-allocated buffers to minimize allocation overhead
- Warm-up runs to eliminate JIT compilation overhead

Note: torch.compile will automatically fall back to eager mode on unsupported devices
(e.g., Tesla P100 with CUDA capability 6.0)
"""

import torch
import torch.nn.functional as F


# ============================================================================
# Configure torch.compile to suppress errors and fall back to eager
# ============================================================================

try:
    # Suppress torch.compile errors and fall back to eager mode
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

try:
    # Don't guard on unbacked symbolic shapes (helps with compatibility)
    torch._dynamo.config.guard_immutable_object = False
except Exception:
    pass


# ============================================================================
# Fused operations using torch.compile (PyTorch 2.0+)
# ============================================================================

def _elastic_net_proximal_torch(w_tilde, thresh, l2_scale):
    """
    Soft thresholding with L2 scaling for Elastic Net.

    coef = sign(w_tilde) * max(|w_tilde| - thresh, 0) / l2_scale
    """
    return torch.sign(w_tilde) * torch.maximum(torch.abs(w_tilde) - thresh, torch.tensor(0.0, device=w_tilde.device, dtype=w_tilde.dtype)) / l2_scale


# Compile the proximal operator for faster execution
# Will automatically fall back to eager mode on unsupported devices
try:
    _elastic_net_proximal_compiled = torch.compile(_elastic_net_proximal_torch, mode='max-autotune')
except (AttributeError, RuntimeError):
    # torch.compile not available or not supported on this device
    _elastic_net_proximal_compiled = _elastic_net_proximal_torch


# ============================================================================
# Optimized FISTA implementation
# ============================================================================

def fit_elasticnet_torch_optimized(X, y, alpha, l1_ratio, n_samples, n_features,
                                    max_iter=1000, tol=1e-4, lipschitz_L=None,
                                    stopping='coef_delta', warmup=True):
    """
    Fit Elastic Net using optimized PyTorch operations.

    Key optimizations:
    1. Pre-compute Gram matrix (XtX) and Xty once
    2. Use torch.compile for fused element-wise operations
    3. Minimize temporary tensor allocations
    4. Use in-place operations where possible
    5. Warm-up to trigger JIT compilation

    Parameters
    ----------
    X : torch.Tensor of shape (n, p)
        Centered design matrix
    y : torch.Tensor of shape (n,)
        Centered target
    alpha : float
        Regularization strength
    l1_ratio : float
        L1 mixing parameter
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    lipschitz_L : float, optional
        Pre-computed Lipschitz constant
    stopping : str
        'coef_delta' or 'kkt'
    warmup : bool
        If True, run warm-up to JIT-compile fused kernels

    Returns
    -------
    coef : torch.Tensor
        Fitted coefficients
    n_iter : int
        Number of iterations
    """
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
        # Small dummy momentum update
        _ = (1.0 + torch.sqrt(1.0 + 4.0 * t_k * t_k)) * 0.5

    for iteration in range(max_iter):
        # Store old coefficients for convergence check
        coef_old.copy_(coef)

        # Gradient step: grad = (XtX @ y_k - Xty) / n
        # Using in-place operations
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
            # KKT condition check
            kkt_grad = torch.matmul(XtX, coef, out=grad)
            kkt_grad -= Xty
            kkt_grad *= inv_n_samples

            grad_l2 = alpha * l2_ratio * coef
            sign_coef = torch.sign(coef)
            sign_coef[coef == 0] = 0

            kkt_violation = torch.maximum(
                torch.abs(kkt_grad + grad_l2 + alpha * l1_ratio * sign_coef),
                torch.maximum(torch.abs(kkt_grad + grad_l2) - alpha * l1_ratio, torch.tensor(0.0, device=X.device))
            )
            violation = float(torch.max(kkt_violation).item())
        else:
            # Coefficient delta check
            delta = torch.abs(coef - coef_old)
            violation = float(torch.max(delta).item())

        if violation < tol:
            break

    return coef, n_iter


# ============================================================================
# Alternative implementation without torch.compile (for older PyTorch)
# ============================================================================

def fit_elasticnet_torch_fused(X, y, alpha, l1_ratio, n_samples, n_features,
                                max_iter=1000, tol=1e-4, lipschitz_L=None,
                                stopping='coef_delta', warmup=True):
    """
    Fallback implementation for PyTorch < 2.0 without torch.compile.

    Uses standard PyTorch operations with in-place optimizations.
    """
    # Precompute Gram matrix and cross product
    XtX = X.T @ X
    Xty = X.T @ y

    # Parameters
    l2_ratio = 1.0 - l1_ratio

    # Lipschitz constant
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

    # Pre-compute inverse
    inv_n_samples = 1.0 / n_samples
    inv_l2_scale = 1.0 / l2_scale

    # Allocate buffers
    coef = torch.zeros(n_features, dtype=X.dtype, device=X.device)
    y_k = torch.zeros(n_features, dtype=X.dtype, device=X.device)
    coef_old = torch.zeros(n_features, dtype=X.dtype, device=X.device)
    grad = torch.empty(n_features, dtype=X.dtype, device=X.device)

    # FISTA state
    t_k = 1.0
    n_iter = 0

    for iteration in range(max_iter):
        coef_old.copy_(coef)

        # Gradient: grad = (XtX @ y_k - Xty) / n
        grad = (XtX @ y_k - Xty) * inv_n_samples

        # Proximal step
        w_tilde = y_k - step * grad
        coef = torch.sign(w_tilde) * torch.maximum(torch.abs(w_tilde) - thresh, torch.tensor(0.0, device=X.device)) * inv_l2_scale

        # Momentum update
        t_new = (1.0 + torch.sqrt(torch.tensor(1.0 + 4.0 * t_k * t_k, device=X.device, dtype=X.dtype))) * 0.5
        beta = (t_k - 1.0) / t_new
        y_k = coef + beta * (coef - coef_old)
        t_k = t_new

        n_iter = iteration + 1

        # Convergence check
        delta = torch.abs(coef - coef_old).max().item()
        if delta < tol:
            break

    return coef, n_iter
