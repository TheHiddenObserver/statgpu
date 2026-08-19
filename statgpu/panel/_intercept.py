"""Stable helpers for panel regressions with an exact constant column."""
from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar
from statgpu.panel._linalg import panel_lstsq
from statgpu.panel._reductions import stable_mean, stable_reduction_flags


def panel_lstsq_exact_constant(X, y, xp, *, constant_index: int = 0):
    """Solve full-rank OLS while protecting the exact constant direction.

    The ordinary path is exactly :func:`panel_lstsq`.  Only responses whose
    dynamic range is already classified as cancellation/range sensitive use a
    stable mean recentering.  Subtracting a constant from the response changes
    only the coefficient of an exact nonzero constant column, so after solving
    the centered problem the removed mean can be restored without changing any
    slope coefficient.

    Rank-deficient designs retain the historical shared-SVD minimum-norm
    solution.  Likewise, if physical response centering could itself overflow,
    this helper leaves the existing solve untouched rather than changing the
    estimator parameterization merely to handle a diagnostic-scale edge case.
    """
    constant_index = int(constant_index)
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    if not 0 <= constant_index < int(X.shape[1]):
        raise ValueError("constant_index is out of range")

    constant = X[0, constant_index]
    exact_constant = xp.all(X[:, constant_index] == constant) & (constant != 0.0)
    if not bool(_to_float_scalar(exact_constant)):
        return panel_lstsq(X, y, xp)

    if not bool(stable_reduction_flags(y, xp)[0]):
        return panel_lstsq(X, y, xp)

    response_mean = stable_mean(y, xp)
    maximum = xp.max(xp.abs(y))
    safe_limit = float(np.finfo(np.float64).max) - xp.abs(response_mean)
    if not bool(_to_float_scalar(maximum <= safe_limit)):
        return panel_lstsq(X, y, xp)

    centered = y - response_mean
    params, rank = panel_lstsq(X, centered, xp)
    if int(rank) < int(X.shape[1]):
        return panel_lstsq(X, y, xp)

    params = params.clone() if getattr(xp, "__name__", "") == "torch" else params.copy()
    params[constant_index] = params[constant_index] + response_mean / constant
    return params, rank
