"""
Pairwise kernel functions with backend-agnostic (xp) interface.

All functions accept an ``xp`` argument that should be a NumPy-compatible
array module (numpy, cupy, or torch).  When *xp* is ``None`` the functions
fall back to ``numpy``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from statgpu.backends import xp_maximum


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

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
    XX = xp.sum(X ** 2, axis=1)[:, None]
    YY = xp.sum(Y ** 2, axis=1)[None, :]
    dist = XX + YY - 2.0 * (X @ Y.T)
    # Clamp to avoid negative values from numerical noise
    dist = xp_maximum(dist, 0.0, xp)
    return xp.exp(-gamma * dist)


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

    # L1 distance using broadcasting
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

    X_norm = xp.sqrt(xp.sum(X ** 2, axis=1))[:, None]
    Y_norm = xp.sqrt(xp.sum(Y ** 2, axis=1))[None, :]
    return (X @ Y.T) / (X_norm * Y_norm + 1e-10)


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
        # Try calling with xp parameter first; fall back without it
        # for user-defined callables that don't accept xp.
        # Also ensure Y is not None (use X for self-kernel).
        Y_arg = Y if Y is not None else X
        try:
            return metric(X, Y_arg, xp=xp, **params)
        except TypeError:
            return metric(X, Y_arg, **params)

    key = str(metric).strip().lower()
    func = KERNEL_REGISTRY.get(key)
    if func is None:
        raise ValueError(
            f"Unknown kernel metric '{metric}'. "
            f"Available: {sorted(KERNEL_REGISTRY.keys())}"
        )
    return func(X, Y, xp=xp, **params)
