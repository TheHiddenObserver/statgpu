"""Backend-native Welch one-way ANOVA."""

from __future__ import annotations

__all__ = ["f_welch"]

from typing import Any

import numpy as np

from statgpu.anova._oneway import AnovaResult
from statgpu.backends import (
    _get_xp,
    _resolve_backend,
    _to_float_scalar,
    xp_asarray,
)


def _array_size(arr, xp) -> int:
    return int(arr.numel()) if xp.__name__ == "torch" else int(arr.size)


def f_welch(
    *groups: Any,
    backend: str = "auto",
    dtype: Any = None,
) -> AnovaResult:
    """Perform Welch's one-way ANOVA for unequal group variances.

    Group validation, means, variances, weights, and the Welch statistic remain
    on the selected NumPy, CuPy, or Torch backend. Only the final scalar
    statistic/degrees of freedom are synchronized for the result container and
    distribution evaluation.

    Parameters
    ----------
    *groups : array-like
        Two or more one-dimensional samples. Each group must contain at least
        two finite observations.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend.
    dtype : dtype or None, default=None
        Floating-point dtype. ``None`` uses backend float64.

    Returns
    -------
    AnovaResult
        Welch F statistic, p-value, numerator df, fractional denominator df,
        and ``eta_squared=NaN`` because a pooled eta-squared is not defined for
        the heteroskedastic Welch model.
    """
    if len(groups) < 2:
        raise ValueError("f_welch requires at least 2 groups")

    resolved = _resolve_backend(backend, *groups)
    xp = _get_xp(resolved)
    float_dtype = xp.float64 if dtype is None else dtype
    ref = next(
        (
            group
            for group in groups
            if type(group).__module__.startswith(("torch", "cupy"))
        ),
        None,
    )

    arrays = []
    sizes = []
    for index, group in enumerate(groups):
        arr = xp_asarray(group, dtype=float_dtype, xp=xp, ref_arr=ref).ravel()
        size = _array_size(arr, xp)
        if size < 2:
            raise ValueError(
                f"Group {index} must contain at least 2 observations for Welch ANOVA"
            )
        if not bool(_to_float_scalar(xp.all(xp.isfinite(arr)))):
            raise ValueError(f"Group {index} contains NaN or infinite values")
        arrays.append(arr)
        sizes.append(size)

    k = len(arrays)
    ref_arr = arrays[0]
    n_k = xp_asarray(sizes, dtype=float_dtype, xp=xp, ref_arr=ref_arr)
    means = xp.stack([xp.mean(group) for group in arrays])
    variances = xp.stack(
        [
            xp.sum((group - mean) ** 2) / float(size - 1)
            for group, mean, size in zip(arrays, means, sizes)
        ]
    )

    zero_variance = variances == 0
    n_zero = int(round(_to_float_scalar(xp.sum(zero_variance))))
    if n_zero:
        if n_zero == k:
            spread = _to_float_scalar(xp.max(xp.abs(means - means[0])))
            mean_scale = max(1.0, abs(_to_float_scalar(means[0])))
            df_within = int(sum(sizes) - k)
            if spread <= 1e-12 * mean_scale:
                return AnovaResult(
                    float("nan"),
                    float("nan"),
                    k - 1,
                    df_within,
                    float("nan"),
                )
            return AnovaResult(
                float("inf"), 0.0, k - 1, df_within, float("nan")
            )
        raise ValueError(
            "Welch ANOVA is undefined when only some groups have zero variance"
        )

    weights = n_k / variances
    weight_sum = xp.sum(weights)
    weighted_mean = xp.sum(weights * means) / weight_sum
    numerator = xp.sum(weights * (means - weighted_mean) ** 2) / float(k - 1)

    adjustment_terms = (1.0 - weights / weight_sum) ** 2 / (n_k - 1.0)
    adjustment_sum = xp.sum(adjustment_terms)
    denominator = 1.0 + (2.0 * (k - 2) / float(k**2 - 1)) * adjustment_sum
    statistic_backend = numerator / denominator

    df1 = k - 1
    adjustment_scalar = _to_float_scalar(adjustment_sum)
    df2 = (
        float("inf")
        if adjustment_scalar <= 0.0
        else ((k**2 - 1) / 3.0) / adjustment_scalar
    )
    statistic = _to_float_scalar(statistic_backend)

    from statgpu.inference._distributions_backend import get_distribution

    device = str(ref_arr.device) if resolved == "torch" else None
    f_dist = get_distribution("f", backend=resolved, device=device)
    pvalue = _to_float_scalar(f_dist.sf(statistic, df1, df2))

    return AnovaResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        df_between=int(df1),
        df_within=float(df2),
        eta_squared=float("nan"),
    )
