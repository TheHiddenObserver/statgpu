"""
CuPy Kernel Fusion Optimization for Elastic Net FISTA

This module provides optimized CUDA kernels that fuse multiple operations
into single kernel launches to reduce memory bandwidth pressure and kernel
launch overhead.

Key optimizations:
1. Use @cp.fuse to combine element-wise operations
2. Minimize temporary array allocations
3. Use in-place operations where possible
4. Reduce kernel launch overhead
"""

import cupy as cp
from cupy import cuda


# ============================================================================
# Fused Element-wise Kernels using CuPy's fuse decorator
# ============================================================================

@cp.fuse()
def _elastic_net_proximal(x, thresh, l2_scale):
    """
    Fused soft thresholding with L2 scaling.

    Combines: sign(x) * maximum(|x| - thresh, 0) / l2_scale
    """
    return cp.sign(x) * cp.maximum(cp.abs(x) - thresh, 0) / l2_scale


@cp.fuse()
def _fista_momentum_update(coef, coef_old, t_old, t_new):
    """
    Fused FISTA momentum update.

    y_new = coef + beta * (coef - coef_old) where beta = (t_old - 1) / t_new
    """
    beta = (t_old - 1) / t_new
    return coef + beta * (coef - coef_old)


@cp.fuse()
def _compute_coef_delta(coef, coef_old):
    """Compute absolute coefficient change."""
    return cp.abs(coef - coef_old)


# ============================================================================
# CuPy ElementwiseKernel for proximal step
# ============================================================================

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


# ============================================================================
# Optimized FISTA implementation
# ============================================================================

def fit_elasticnet_optimized(X, y, alpha, l1_ratio, n_samples, n_features,
                              max_iter=1000, tol=1e-4, lipschitz_L=None,
                              stopping='coef_delta',
                              warmup=True):
    """
    Fit Elastic Net using optimized CuPy operations.

    Key optimizations:
    1. Pre-compute Gram matrix (XtX) and Xty once
    2. Use fused kernels for element-wise operations
    3. Minimize temporary array allocations
    4. Use in-place operations where possible
    5. Warm-up fused kernels to avoid JIT compilation overhead

    Parameters
    ----------
    X : CuPy ndarray of shape (n, p)
        Centered design matrix
    y : CuPy ndarray of shape (n,)
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
    coef : CuPy ndarray
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

    # Pre-bind functions for speed
    matvec = XtX.__matmul__

    # Warm-up: Call fused kernel once to trigger JIT compilation
    if warmup:
        _ = _elastic_net_proximal(w_tilde, thresh, l2_scale)
        # Small dummy momentum update
        _ = (1.0 + cp.sqrt(1.0 + 4.0 * t_k * t_k)) * 0.5

    for iteration in range(max_iter):
        # Store old coefficients for convergence check
        coef_old[:] = coef

        # Gradient step: grad = (XtX @ y_k - Xty) / n
        # Using in-place operations
        grad = matvec(y_k)
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
            # KKT condition check
            kkt_grad = matvec(coef)
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
            # Coefficient delta check
            delta = cp.abs(coef - coef_old)
            violation = float(cp.max(delta))

        if violation < tol:
            break

    return coef, n_iter


# ============================================================================
# Alternative implementation using RawKernel for maximum fusion
# ============================================================================

ELASTIC_NET_FISTA_KERNEL = r'''
extern "C" {
    __global__ void elastic_net_fista_step(
        const double* XtX,
        const double* Xty,
        double* y_k,
        double* coef,
        double* coef_old,
        int n_features,
        int n_samples,
        double step,
        double thresh,
        double inv_l2_scale,
        double* delta_max
    ) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= n_features) return;

        // Compute gradient: grad[j] = (XtX[j,:] @ y_k - Xty[j]) / n
        double grad_j = 0.0;
        for (int k = 0; k < n_features; k++) {
            grad_j += XtX[j * n_features + k] * y_k[k];
        }
        grad_j = (grad_j - Xty[j]) / n_samples;

        // Store old coefficient
        double c_old = coef[j];
        coef_old[j] = c_old;

        // Proximal gradient step
        double w_tilde = y_k[j] - step * grad_j;
        double abs_w = abs(w_tilde);

        double c_new;
        if (abs_w > thresh) {
            c_new = (w_tilde > 0 ? 1.0 : -1.0) * (abs_w - thresh) * inv_l2_scale;
        } else {
            c_new = 0.0;
        }
        coef[j] = c_new;

        // Track maximum delta
        double delta = abs(c_new - c_old);
        atomicExch((unsigned long long*)delta_max, __double_as_longlong(delta));
    }
}
'''

# Note: RawKernel approach has limitations with atomicExch for doubles
# The ElementwiseKernel approach above is preferred for compatibility
