"""Stable helpers for panel regressions with an exact constant column."""
from __future__ import annotations

import numpy as np

from statgpu.panel._linalg import (
    _lstsq_working_design,
    _svd_inverse_factors,
    panel_lstsq,
)
from statgpu.panel._reductions import grouped_score_sums, stable_reduction_flags


def panel_lstsq_exact_constant(X, y, xp, *, constant_index: int = 0):
    """Use the shared SVD policy with a cancellation-safe response projection.

    ``PooledOLS`` and ``BetweenOLS`` construct an exact nonzero constant column,
    so a cancellation tail in the response can directly become the intercept.
    The historical SVD path computes ``weighted_u_t @ y`` inside BLAS; reduction
    order can erase a representable low-order term before the coefficient is
    formed.  For ordinary responses this helper is exactly :func:`panel_lstsq`.
    Only responses already classified as cancellation/range sensitive replace
    that one matrix-vector reduction with the shared magnitude-tiered grouped
    sum.  SVD factors, rank cutoff, design rescaling, and minimum-norm
    parameterization remain unchanged.

    The implementation deliberately does *not* center observations first.  At
    extreme magnitudes ``y - mean(y)`` can round the removed constant away on
    large rows, making a later intercept restoration backend-order dependent.
    """
    constant_index = int(constant_index)
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    if not 0 <= constant_index < int(X.shape[1]):
        raise ValueError("constant_index is out of range")

    if not bool(stable_reduction_flags(y, xp)[0]):
        return panel_lstsq(X, y, xp)

    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    products = weighted_u_t.T * y.reshape(-1, 1)
    codes_np = np.zeros(int(products.shape[0]), dtype=np.int64)
    projected = grouped_score_sums(
        products,
        codes_np,
        n_groups=1,
        xp=xp,
    )[0]
    return (Vh.T @ projected) * design_scale, rank
