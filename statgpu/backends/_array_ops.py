"""
Backend utilities for GLM loss functions.

Provides wrapper functions that dispatch to numpy/cupy/torch
based on the input array type, so GLM loss functions can use
a single code path for all backends.
"""


def _xp(arr):
    """Get the array module (numpy/cupy/torch) from array type."""
    mod = type(arr).__module__
    if mod.startswith("cupy"):
        import cupy
        return cupy
    if mod.startswith("torch"):
        import torch
        return torch
    import numpy
    return numpy


def _clip(arr, lo, hi):
    """Clip array values."""
    xp = _xp(arr)
    if xp.__name__ == "torch":
        result = arr.clone()
        if lo is not None:
            result = xp.clamp(result, min=lo)
        if hi is not None:
            result = xp.clamp(result, max=hi)
        return result
    return xp.clip(arr, lo, hi)


def _exp(arr):
    """Element-wise exponential."""
    xp = _xp(arr)
    return xp.exp(arr)


def _log(arr):
    """Element-wise natural log."""
    xp = _xp(arr)
    return xp.log(arr)


def _log1p(arr):
    """Element-wise log(1+x)."""
    xp = _xp(arr)
    return xp.log1p(arr)


def _sigmoid(arr):
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    xp = _xp(arr)
    z = _clip(arr, -500, 500)
    return 1.0 / (1.0 + xp.exp(-z))


def _sum(arr):
    """Sum of all elements."""
    xp = _xp(arr)
    return xp.sum(arr)


def _eigvalsh(arr):
    """Eigenvalues of a symmetric matrix (sorted ascending)."""
    xp = _xp(arr)
    return xp.linalg.eigvalsh(arr)


def _max_eigval_power(mat, n_iter=20, tol=1e-8):
    """Largest eigenvalue of a symmetric matrix via power iteration.

    Much faster than full eigendecomposition, especially on GPU
    where cuSOLVER eigvalsh has high kernel compilation overhead.
    O(p^2) vs O(p^3). Accuracy within 1% for 20 iterations.

    Parameters
    ----------
    mat : 2-d array (p, p), symmetric positive semi-definite.
    n_iter : int
        Max power iterations.
    tol : float
        Early stopping tolerance on eigenvalue change.

    Returns
    -------
    float : max eigenvalue estimate.
    """
    xp = _xp(mat)
    p = mat.shape[0]
    dtype = getattr(mat, 'dtype', None)
    # Use deterministic initial vector (ones) instead of random to ensure
    # identical eigenvalue estimates across numpy/cupy/torch backends.
    if xp.__name__ == "torch":
        v = xp.ones(p, dtype=dtype, device=mat.device)
    else:
        v = xp.ones(p)
        if dtype is not None and hasattr(v, 'astype'):
            v = v.astype(dtype)

    v_norm = float(xp.sqrt(xp.dot(v, v)))
    if v_norm < 1e-15:
        return 1.0
    v = v / v_norm

    lambda_old = 0.0
    for _ in range(n_iter):
        v_new = mat @ v
        v_norm = float(xp.sqrt(xp.dot(v_new, v_new)))
        if v_norm < 1e-15:
            return 1.0
        v = v_new / v_norm
        lambda_new = float(v @ (mat @ v))
        if abs(lambda_new - lambda_old) < tol * abs(lambda_new):
            break
        lambda_old = lambda_new
    return lambda_new
