"""
Pairwise kernel functions with backend-agnostic (xp) interface.

All functions accept an ``xp`` argument that should be a NumPy-compatible
array module (numpy, cupy, or torch).  When *xp* is ``None`` the functions
fall back to ``numpy``.
"""

from __future__ import annotations

from typing import Optional

import inspect
import numpy as np

from statgpu.backends import _to_float_scalar, xp_maximum


# ---------------------------------------------------------------------------
# Individual kernel functions
# ---------------------------------------------------------------------------

def rbf_kernel(X, Y=None, gamma=None, xp=None):
    r"""Radial basis function (Gaussian) kernel.

    .. math::
        K(x, y) = \exp(-\gamma \|x - y\|^2)

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional
    gamma : float, optional
        Kernel coefficient.  Defaults to ``1 / n_features``.
    xp : module, optional
        Array module (numpy / cupy / torch).

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
    """
    if xp is None:
        xp = np
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    n, m = X.shape[0], Y.shape[0]

    # NumPy uses chunking for large matrices but never silently changes an
    # explicitly floating input dtype. Integer inputs are promoted because the
    # exponential kernel is intrinsically floating point.
    if xp is np:
        X_np = np.asarray(X)
        Y_np = np.asarray(Y)
        if X_np.ndim != 2 or Y_np.ndim != 2:
            raise ValueError("X and Y must be two-dimensional arrays")
        if X_np.shape[1] != Y_np.shape[1]:
            raise ValueError("X and Y must have the same number of features")
        dt = np.result_type(X_np.dtype, Y_np.dtype, np.float32)
        if np.issubdtype(dt, np.complexfloating):
            raise ValueError("rbf_kernel does not support complex-valued inputs")
        X_np = np.asarray(X_np, dtype=dt)
        Y_np = np.asarray(Y_np, dtype=dt)
        XX = np.sum(X_np ** 2, axis=1)  # (n,)
        YY = np.sum(Y_np ** 2, axis=1)  # (m,)
        chunk = max(1, min(n, int(5e8 / (m * np.dtype(dt).itemsize))))
        if chunk >= n:
            X_a = X_np
            Y_a = Y_np
            K = X_a @ Y_a.T
            K *= -2.0
            K += XX[:, None]
            K += YY[None, :]
            np.maximum(K, 0.0, out=K)
            np.exp(-gamma * K, out=K)
            return K
        else:
            K = np.empty((n, m), dtype=dt)
            Y_a = Y_np
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                Kc = X_np[s:e] @ Y_a.T
                Kc *= -2.0
                Kc += XX[s:e, None]
                Kc += YY[None, :]
                np.maximum(Kc, 0.0, out=Kc)
                np.exp(-gamma * Kc, out=Kc)
                K[s:e] = Kc
            return K

    if getattr(X, "ndim", None) != 2 or getattr(Y, "ndim", None) != 2:
        raise ValueError("X and Y must be two-dimensional arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    x_complex_flag = getattr(X, 'is_complex', None)
    y_complex_flag = getattr(Y, 'is_complex', None)
    x_is_complex = bool(x_complex_flag()) if callable(x_complex_flag) else False
    y_is_complex = bool(y_complex_flag()) if callable(y_complex_flag) else False
    x_kind = getattr(getattr(X, 'dtype', None), 'kind', None)
    y_kind = getattr(getattr(Y, 'dtype', None), 'kind', None)
    if x_is_complex or y_is_complex or x_kind == 'c' or y_kind == 'c':
        raise ValueError('rbf_kernel does not support complex-valued inputs')

    # Torch integer tensors cannot be updated in-place with floating kernel
    # coefficients. Promote integer inputs while preserving floating dtypes.
    if getattr(xp, "__name__", "") == "torch":
        if not X.is_floating_point() or not Y.is_floating_point():
            X = X.to(dtype=xp.float64)
            Y = Y.to(dtype=xp.float64)
    else:
        x_kind = getattr(getattr(X, "dtype", None), "kind", None)
        y_kind = getattr(getattr(Y, "dtype", None), "kind", None)
        if x_kind not in ("f",) or y_kind not in ("f",):
            X = X.astype(xp.float64, copy=False)
            Y = Y.astype(xp.float64, copy=False)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
    # Single n×m buffer, all in-place.
    K = X @ Y.T                     # (n, m) — BLAS gemm
    K *= -2.0                       # in-place
    # norms — avoid n×n temporary via row-wise sum
    K += xp.sum(X * X, axis=1)[:, None]
    K += xp.sum(Y * Y, axis=1)[None, :]
    K = xp_maximum(K, 0.0, xp)      # clamp negatives, including Torch scalar handling
    K *= -gamma
    if hasattr(K, 'exp_'):
        K.exp_()                    # torch in-place
    else:
        xp.exp(K, out=K)            # numpy/cupy in-place
    return K


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1, xp=None):
    r"""Polynomial kernel.

    .. math::
        K(x, y) = (\gamma \, x^\top y + c_0)^d

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional
    degree : int, default=3
    gamma : float, optional
        Defaults to ``1 / n_features``.
    coef0 : float, default=1
    xp : module, optional

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
    """
    if xp is None:
        xp = np
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    return (gamma * (X @ Y.T) + coef0) ** degree


def linear_kernel(X, Y=None, xp=None):
    r"""Linear kernel.

    .. math::
        K(x, y) = x^\top y

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional
    xp : module, optional

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
    """
    if xp is None:
        xp = np
    if Y is None:
        Y = X
    return X @ Y.T


def laplacian_kernel(X, Y=None, gamma=None, xp=None):
    r"""Laplacian kernel.

    .. math::
        K(x, y) = \exp(-\gamma \|x - y\|_1)

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional
    gamma : float, optional
        Defaults to ``1 / n_features``.
    xp : module, optional

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
    """
    if xp is None:
        xp = np
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # L1 distance — use scipy cdist for numpy (avoids huge temporary)
    if xp is np:
        from scipy.spatial.distance import cdist
        dist = cdist(np.asarray(X), np.asarray(Y), metric='cityblock')
        return xp.exp(-gamma * dist, out=dist)
    else:
        dist = xp.sum(xp.abs(X[:, None, :] - Y[None, :, :]), axis=2)
        return xp.exp(-gamma * dist)


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1, xp=None):
    r"""Sigmoid (hyperbolic tangent) kernel.

    .. math::
        K(x, y) = \tanh(\gamma \, x^\top y + c_0)

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional
    gamma : float, optional
        Defaults to ``1 / n_features``.
    coef0 : float, default=1
    xp : module, optional

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
    """
    if xp is None:
        xp = np
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    return xp.tanh(gamma * (X @ Y.T) + coef0)


def cosine_kernel(X, Y=None, xp=None):
    r"""Cosine similarity kernel.

    .. math::
        K(x, y) = \frac{x^\top y}{\|x\| \, \|y\|}

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional
    xp : module, optional

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
    """
    if xp is None:
        xp = np
    if Y is None:
        Y = X

    if xp is np:
        X_norm = np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
        Y_norm = np.sqrt(np.einsum('ij,ij->i', Y, Y))[None, :]
    else:
        X_norm = xp.sqrt(xp.sum(X * X, axis=1))[:, None]
        Y_norm = xp.sqrt(xp.sum(Y * Y, axis=1))[None, :]
    return (X @ Y.T) / (X_norm * Y_norm + 1e-10)


def _chi2_kernel_numpy_fallback(X, Y, gamma=1.0, max_elements=2_000_000):
    """Chunked NumPy chi-squared kernel used when sklearn is unavailable."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    n, p = X.shape
    m = Y.shape[0]
    chunk = min(p, max(1, int(max_elements) // max(n * m, 1)))
    chi2_dist = np.zeros((n, m), dtype=np.result_type(X.dtype, Y.dtype, np.float64))
    for start in range(0, p, chunk):
        end = min(start + chunk, p)
        Xc = X[:, None, start:end]
        Yc = Y[None, :, start:end]
        numerator = (Xc - Yc) ** 2
        denominator = Xc + Yc
        contribution = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=chi2_dist.dtype),
            where=denominator > 0,
        )
        chi2_dist += np.sum(contribution, axis=2)
    return np.exp(-float(gamma) * chi2_dist)


def chi2_kernel(X, Y=None, gamma=1.0, xp=None):
    r"""Chi-squared kernel.

    .. math::
        K(x, y) = \exp\left(-\gamma \sum_i \frac{(x_i - y_i)^2}{x_i + y_i}\right)

    This is an RBF-like kernel that works well with histograms.
    Requires non-negative input features.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Must be non-negative.
    Y : array-like of shape (n_samples_Y, n_features), optional
        Must be non-negative.
    gamma : float, default=1.0
        Kernel coefficient.
    xp : module, optional
        Array module (numpy / cupy / torch).

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)

    Notes
    -----
    This is the exponentiated chi-squared kernel, which is related to
    the additive chi-squared kernel by exponentiation.  It is commonly
    used for histogram-based features in computer vision.
    """
    if xp is None:
        xp = np
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")

    if xp is np:
        X = np.asarray(X)
        Y = X if Y is None else np.asarray(Y)
    elif Y is None:
        Y = X

    if getattr(X, "ndim", None) != 2 or getattr(Y, "ndim", None) != 2:
        raise ValueError("X and Y must be two-dimensional arrays")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    if _to_float_scalar(xp.min(X)) < 0 or _to_float_scalar(xp.min(Y)) < 0:
        raise ValueError("chi2_kernel requires non-negative input features")

    if xp is np:
        try:
            from sklearn.metrics.pairwise import chi2_kernel as _sk_chi2
            return _sk_chi2(X, Y, gamma=gamma)
        except ImportError:
            return _chi2_kernel_numpy_fallback(X, Y, gamma=gamma)

    X_exp = X[:, None, :]
    Y_exp = Y[None, :, :]
    numerator = (X_exp - Y_exp) ** 2
    denominator = X_exp + Y_exp
    denom_safe = xp_maximum(denominator, 1e-10, xp)
    chi2_dist = xp.sum(numerator / denom_safe, axis=2)
    return xp.exp(-gamma * chi2_dist)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

KERNEL_REGISTRY = {
    "rbf": rbf_kernel,
    "gaussian": rbf_kernel,
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "poly": polynomial_kernel,
    "laplacian": laplacian_kernel,
    "sigmoid": sigmoid_kernel,
    "cosine": cosine_kernel,
    "chi2": chi2_kernel,
    "chi-squared": chi2_kernel,
}


def pairwise_kernels(X, Y=None, metric="rbf", xp=None, **params):
    """Compute the kernel between arrays X and Y using the given metric.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional
    metric : str or callable, default='rbf'
        Kernel metric name or a callable.
    xp : module, optional
        Array module.
    **params
        Additional keyword arguments forwarded to the kernel function.

    Returns
    -------
    K : array of shape (n_samples_X, n_samples_Y)
    """
    if callable(metric):
        # Decide whether the callable accepts ``xp`` before invoking it. Catching
        # TypeError around the call itself masks genuine errors raised inside a
        # user kernel and can execute a stateful callable twice.
        try:
            signature = inspect.signature(metric)
        except (TypeError, ValueError):
            signature = None
        accepts_xp = signature is None or "xp" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_xp:
            return metric(X, Y, xp=xp, **params)
        return metric(X, Y, **params)

    key = str(metric).strip().lower()
    func = KERNEL_REGISTRY.get(key)
    if func is None:
        raise ValueError(
            f"Unknown kernel metric '{metric}'. "
            f"Available: {sorted(KERNEL_REGISTRY.keys())}"
        )
    return func(X, Y, xp=xp, **params)
