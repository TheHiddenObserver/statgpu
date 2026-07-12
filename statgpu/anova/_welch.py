"""GPU-accelerated Welch ANOVA.

Provides :func:`f_welch`, a backend-agnostic replacement for
``scipy.stats.alexandergovern`` (or R's ``oneway.test``) that handles
unequal variances across groups.
"""

from __future__ import annotations

__all__ = ["f_welch"]

from typing import Any

import numpy as np

from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar, _to_numpy
from statgpu.anova._oneway import AnovaResult


def f_welch(
    *groups: Any,
    backend: str = "auto",
    dtype: Any = None,
) -> AnovaResult:
    """Perform Welch's one-way ANOVA (unequal variances).

    Parameters
    ----------
    *groups : array-like
        Two or more sample arrays, one per group.  Each must be 1-D.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend.  **Note:** computation currently runs on CPU
        regardless of backend selection.
    dtype : dtype or None, default=None
        Float dtype for computation.  ``None`` uses ``float64``.

    Returns
    -------
    AnovaResult
        Dataclass with ``statistic``, ``pvalue``, ``df_between``,
        ``df_within``, and ``eta_squared`` (set to NaN -- not meaningful
        for Welch's test).

    Raises
    ------
    ValueError
        If fewer than 2 groups or any group has < 2 observations.

    Notes
    -----
    Welch's ANOVA (Welch 1951) does not assume equal variances.  The
    test statistic is:

        W = (sum_k w_k * (xbar_k - xbar_w)**2 / (K-1)) /
            (1 + 2*(K-2)/(K^2-1) * sum_k (1-w_k/W)^2 / (n_k-1))

    where w_k = n_k / s_k^2, W = sum w_k, and xbar_w = sum(w_k*xbar_k)/W.

    The p-value uses an F distribution with df1 = K-1 and df2 from the
    Welch-Satterthwaite equation.

    References
    ----------
    Welch, B. L. (1951). On the comparison of several mean values: an
    alternative approach. *Biometrika*, 38(3/4), 330-336.
    """
    if len(groups) < 2:
        raise ValueError("f_welch requires at least 2 groups")

    resolved = _resolve_backend(backend, *groups)
    xp = _get_xp(resolved)
    float_dtype = dtype if dtype is not None else xp.float64

    # Convert groups to flat numpy arrays for statistics
    flat_groups = []
    for g in groups:
        arr = np.asarray(_to_numpy(g), dtype=np.float64).ravel()
        if arr.size < 2:
            raise ValueError("Welch ANOVA requires at least 2 observations per group")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Welch ANOVA groups must contain only finite values")
        flat_groups.append(arr)

    k = len(flat_groups)

    # Group statistics
    n_k = np.array([g.size for g in flat_groups], dtype=np.float64)
    xbar_k = np.array([g.mean() for g in flat_groups], dtype=np.float64)
    s2_k = np.array([g.var(ddof=1) for g in flat_groups], dtype=np.float64)

    # Guard against zero variance groups
    if np.any(s2_k == 0):
        # If all groups have zero variance and same mean, F=NaN
        # If means differ, F=inf (perfect separation)
        if np.all(s2_k == 0):
            if np.allclose(xbar_k, xbar_k[0]):
                return AnovaResult(float("nan"), float("nan"), k - 1, int(sum(n_k)) - k, float("nan"))
            else:
                return AnovaResult(float("inf"), 0.0, k - 1, int(sum(n_k)) - k, float("nan"))
        # Dropping only the zero-variance groups changes the null hypothesis.
        # Require the caller to handle this degenerate mixed case explicitly.
        raise ValueError(
            "Welch ANOVA is undefined when only some groups have zero variance"
        )

    # Weights (inverse variance)
    w_k = n_k / s2_k
    W = w_k.sum()

    # Weighted grand mean
    xbar_w = np.dot(w_k, xbar_k) / W

    # Numerator
    numer = np.dot(w_k, (xbar_k - xbar_w) ** 2) / (k - 1)

    # Denominator (Welch-Satterthwaite adjustment)
    lam_k = (1 - w_k / W) ** 2 / (n_k - 1)
    denom = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam_k.sum()

    f_stat = numer / denom

    # Welch-Satterthwaite degrees of freedom
    df1 = k - 1
    df2_num = (k ** 2 - 1) / 3.0
    df2_den = lam_k.sum()
    df2 = df2_num / df2_den if df2_den > 0 else float("inf")

    # P-value from F distribution
    from statgpu.inference._distributions_backend import get_distribution

    f_dist = get_distribution("f", backend=resolved)
    pvalue = _to_float_scalar(f_dist.sf(f_stat, df1, df2))

    # Welch's ANOVA does not assume equal variances, so a pooled
    # eta-squared is not well-defined.  Return NaN.
    return AnovaResult(
        statistic=float(f_stat),
        pvalue=float(pvalue),
        df_between=int(df1),
        df_within=float(df2),
        eta_squared=float("nan"),
    )
