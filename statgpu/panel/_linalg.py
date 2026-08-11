"""Shared numerical linear algebra policy for panel fit spaces."""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar


def _rank_mask(X, singular_values, xp):
    if int(singular_values.shape[0]) == 0:
        raise ValueError("panel design must contain at least one column")
    s_max = _to_float_scalar(xp.max(singular_values))
    cutoff = (
        max(int(X.shape[0]), int(X.shape[1]))
        * np.finfo(np.float64).eps
        * float(s_max)
    )
    retained = singular_values > float(cutoff)
    rank = int(_to_float_scalar(xp.sum(retained)))
    return retained, rank


def panel_svd_pseudoinverse(X, xp):
    """Return X+, (X'X)+, and rank from one explicit float64 SVD mask."""
    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    retained, rank = _rank_mask(X, singular_values, xp)
    safe_values = xp.where(retained, singular_values, xp.ones_like(singular_values))
    inverse_values = xp.where(
        retained, 1.0 / safe_values, xp.zeros_like(singular_values)
    )
    X_pinv = (Vh.T * inverse_values) @ U.T
    bread = X_pinv @ X_pinv.T
    return X_pinv, bread, rank


def panel_lstsq(X, y, xp):
    """Return the minimum-norm least-squares solution under the panel SVD policy."""
    X_pinv, _bread, rank = panel_svd_pseudoinverse(X, xp)
    return X_pinv @ y, rank


def panel_matrix_rank(X, xp):
    """Return numerical rank using exactly the panel pseudoinverse cutoff."""
    if getattr(xp, "__name__", "") == "torch":
        singular_values = xp.linalg.svdvals(X)
    else:
        singular_values = xp.linalg.svd(X, compute_uv=False)
    _retained, rank = _rank_mask(X, singular_values, xp)
    return rank
