"""Shared numerical linear algebra policy for panel fit spaces."""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar


def _rank_mask_backend(X, singular_values, xp):
    """Return the shared SVD retention mask and backend-native rank scalar."""
    if int(singular_values.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")
    cutoff_scale = (
        max(int(X.shape[-2]), int(X.shape[-1]))
        * np.finfo(np.float64).eps
    )
    cutoff = xp.max(singular_values) * float(cutoff_scale)
    retained = singular_values > cutoff
    return retained, xp.sum(retained)


def _rank_mask(X, singular_values, xp):
    retained, rank_backend = _rank_mask_backend(X, singular_values, xp)
    # Keep the singular-value cutoff on the active backend.  Only the final
    # integer rank must cross the device boundary for fail-closed Python control
    # flow; extracting s_max separately would add an avoidable GPU sync.
    rank = int(_to_float_scalar(rank_backend))
    return retained, rank


def _inverse_values(singular_values, retained, xp):
    safe_values = xp.where(retained, singular_values, xp.ones_like(singular_values))
    return xp.where(retained, 1.0 / safe_values, xp.zeros_like(singular_values))


def _svd_inverse_factors(X, xp):
    """Return SVD factors, inverse singular values, and shared numerical rank."""
    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    retained, rank = _rank_mask(X, singular_values, xp)
    return U, Vh, _inverse_values(singular_values, retained, xp), rank


def panel_svd_pseudoinverse(X, xp):
    """Return X+, (X'X)+, and rank from one explicit float64 SVD mask."""
    U, Vh, inverse_values, rank = _svd_inverse_factors(X, xp)
    X_pinv = (Vh.T * inverse_values) @ U.T
    bread = X_pinv @ X_pinv.T
    return X_pinv, bread, rank


def panel_lstsq(X, y, xp):
    """Return the minimum-norm least-squares solution under the panel SVD policy."""
    U, Vh, inverse_values, rank = _svd_inverse_factors(X, xp)
    projected = U.T @ y
    if getattr(projected, "ndim", 1) == 1:
        scaled = inverse_values * projected
    else:
        scaled = inverse_values.reshape(-1, 1) * projected
    return Vh.T @ scaled, rank


def panel_lstsq_deferred_rank(X, y, xp):
    """Return least-squares parameters plus a backend-native rank scalar.

    This is the same two-dimensional SVD policy as :func:`panel_lstsq`, but it
    deliberately does not convert the final rank to Python.  GPU callers that
    must solve several independent matrices with a backend lacking a supported
    stacked-SVD API can therefore launch all solves first and transfer one packed
    rank vector after the loop rather than synchronizing once per matrix.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    retained, rank_backend = _rank_mask_backend(X, singular_values, xp)
    inverse_values = _inverse_values(singular_values, retained, xp)
    projected = U.T @ y
    if getattr(projected, "ndim", 1) == 1:
        scaled = inverse_values * projected
    else:
        scaled = inverse_values.reshape(-1, 1) * projected
    return Vh.T @ scaled, rank_backend


def panel_lstsq_batched(X, y, xp):
    """Solve a batch of equal-shaped panel least-squares problems with one SVD call.

    Parameters
    ----------
    X : array-like, shape (batch, n_obs, n_features)
        Equal-shaped design matrices resident on a namespace with documented
        stacked-SVD support.
    y : array-like, shape (batch, n_obs) or (batch, n_obs, n_targets)
        Matching responses.
    xp : module
        NumPy or Torch numerical namespace.  CuPy's maintained public SVD API
        documents two-dimensional input, so CuPy callers use
        :func:`panel_lstsq_deferred_rank` instead of relying on undocumented
        stacked-SVD behavior.

    Returns
    -------
    params : array-like
        Batched minimum-norm solutions.
    ranks : array-like, shape (batch,)
        Backend-native numerical ranks.  Callers can transfer the complete rank
        vector once per batch instead of synchronizing one scalar per period.

    Notes
    -----
    The singular-value cutoff is exactly the same policy as :func:`panel_lstsq`,
    applied independently to each matrix in the batch.
    """
    namespace = getattr(xp, "__name__", "")
    if namespace not in {"numpy", "torch"}:
        raise NotImplementedError(
            "panel_lstsq_batched requires a namespace with documented stacked-SVD "
            "support; use panel_lstsq_deferred_rank for two-dimensional CuPy solves"
        )
    if getattr(X, "ndim", None) != 3:
        raise ValueError("batched panel design must have shape (batch, n_obs, n_features)")
    if int(X.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")
    if getattr(y, "ndim", None) not in (2, 3):
        raise ValueError("batched panel response must have shape (batch, n_obs[, n_targets])")
    if int(y.shape[0]) != int(X.shape[0]) or int(y.shape[1]) != int(X.shape[1]):
        raise ValueError("batched panel design and response shapes are incompatible")

    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    cutoff_scale = (
        max(int(X.shape[-2]), int(X.shape[-1]))
        * np.finfo(np.float64).eps
    )
    if namespace == "torch":
        largest = xp.max(singular_values, dim=-1, keepdim=True).values
        ranks = xp.sum(singular_values > largest * float(cutoff_scale), dim=-1)
    else:
        largest = xp.max(singular_values, axis=-1, keepdims=True)
        ranks = xp.sum(
            singular_values > largest * float(cutoff_scale),
            axis=-1,
        )
    retained = singular_values > largest * float(cutoff_scale)
    inverse_values = _inverse_values(singular_values, retained, xp)

    rhs = y[..., None] if getattr(y, "ndim", None) == 2 else y
    projected = xp.matmul(xp.swapaxes(U, -2, -1), rhs)
    scaled = inverse_values[..., None] * projected
    params = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)
    if getattr(y, "ndim", None) == 2:
        params = params[..., 0]
    return params, ranks


def panel_matrix_rank(X, xp):
    """Return numerical rank using exactly the panel pseudoinverse cutoff."""
    if getattr(xp, "__name__", "") == "torch":
        singular_values = xp.linalg.svdvals(X)
    else:
        singular_values = xp.linalg.svd(X, compute_uv=False)
    _retained, rank = _rank_mask(X, singular_values, xp)
    return rank
