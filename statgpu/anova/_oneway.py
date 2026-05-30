"""GPU-accelerated one-way ANOVA.

Provides :func:`f_oneway`, a backend-agnostic replacement for
``scipy.stats.f_oneway`` that can run on NumPy, CuPy, or PyTorch arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union

import numpy as np

from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar, _to_numpy


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnovaResult:
    """Result of a one-way ANOVA.

    Attributes
    ----------
    statistic : float
        The F-statistic.
    pvalue : float
        P-value from the F-distribution survival function.
    df_between : float
        Degrees of freedom between groups (k - 1).
    df_within : float
        Degrees of freedom within groups (N - k).
    eta_squared : float
        Effect size: SSB / (SSB + SSW).
    """

    statistic: float
    pvalue: float
    df_between: float
    df_within: float
    eta_squared: float


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def f_oneway(
    *groups: Any,
    backend: str = "auto",
) -> AnovaResult:
    """Perform a one-way ANOVA.

    Parameters
    ----------
    *groups : array-like
        Two or more sample arrays, one per group.  Each must be 1-D (or
        flattenable to 1-D).
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend.  ``'auto'`` inspects the input arrays and picks the
        best match.

    Returns
    -------
    AnovaResult
        Dataclass with ``statistic``, ``pvalue``, ``df_between``,
        ``df_within``, and ``eta_squared``.

    Raises
    ------
    ValueError
        If fewer than 2 groups are supplied or any group has fewer than 1
        observation.

    Examples
    --------
    >>> import numpy as np
    >>> from statgpu.anova import f_oneway
    >>> g1 = np.array([5.1, 4.9, 5.0])
    >>> g2 = np.array([6.2, 6.0, 6.3])
    >>> g3 = np.array([7.1, 7.3, 7.0])
    >>> result = f_oneway(g1, g2, g3)
    >>> result.statistic  # doctest: +SKIP
    114.54545454545453
    """
    if len(groups) < 2:
        raise ValueError("f_oneway requires at least 2 groups")

    # Resolve backend from input arrays
    resolved = _resolve_backend(backend, *groups)
    xp = _get_xp(resolved)

    # Convert groups to flat arrays in the target backend
    flat_groups = []
    for g in groups:
        arr = xp.asarray(g, dtype=xp.float64).ravel()
        n_i = int(arr.shape[0])
        if n_i < 1:
            raise ValueError("each group must contain at least 1 observation")
        flat_groups.append(arr)

    k = len(flat_groups)
    # Use first group as device reference for torch
    ref = flat_groups[0]
    group_sizes = xp.asarray([int(g.shape[0]) for g in flat_groups], dtype=xp.float64)
    # Ensure group_sizes is on same device as groups (torch CUDA)
    if hasattr(group_sizes, 'to') and hasattr(ref, 'device'):
        group_sizes = group_sizes.to(device=ref.device)
    N = _to_float_scalar(xp.sum(group_sizes))

    if N <= k:
        raise ValueError(
            f"total observations ({int(N)}) must exceed number of groups ({k})"
        )

    # Group means — computed per-group, kept on device
    group_means = xp.asarray(
        [float(_to_float_scalar(xp.sum(g) / g.shape[0])) for g in flat_groups],
        dtype=xp.float64,
    )
    # Ensure group_means is on same device as groups (torch CUDA)
    if hasattr(group_means, 'to') and hasattr(ref, 'device'):
        group_means = group_means.to(device=ref.device)

    # Grand mean (weighted by group sizes)
    grand_mean = _to_float_scalar(xp.sum(group_means * group_sizes) / N)

    # SSB (between-group sum of squares)
    ssb = _to_float_scalar(
        xp.sum(group_sizes * (group_means - grand_mean) ** 2)
    )

    # SSW (within-group sum of squares)
    ssw = 0.0
    for i, g in enumerate(flat_groups):
        diff = g - group_means[i]
        ssw += _to_float_scalar(xp.sum(diff * diff))

    df_between = float(k - 1)
    df_within = float(N - k)

    # Edge case: no within-group variance
    if ssw == 0.0:
        if ssb == 0.0:
            # All observations identical
            return AnovaResult(
                statistic=float("nan"),
                pvalue=float("nan"),
                df_between=df_between,
                df_within=df_within,
                eta_squared=float("nan"),
            )
        # Perfect separation
        return AnovaResult(
            statistic=float("inf"),
            pvalue=0.0,
            df_between=df_between,
            df_within=df_within,
            eta_squared=1.0,
        )

    ms_between = ssb / df_between
    ms_within = ssw / df_within
    f_stat = ms_between / ms_within

    eta_squared = ssb / (ssb + ssw)

    # P-value from F survival function via statgpu.inference
    from statgpu.inference._distributions_backend import get_distribution

    f_dist = get_distribution("f", backend=resolved)
    pvalue_arr = f_dist.sf(f_stat, df_between, df_within)
    pvalue = _to_float_scalar(pvalue_arr)

    return AnovaResult(
        statistic=f_stat,
        pvalue=pvalue,
        df_between=df_between,
        df_within=df_within,
        eta_squared=eta_squared,
    )
