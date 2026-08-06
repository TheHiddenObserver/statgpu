"""Backend-aware post-hoc tests for ANOVA."""

from __future__ import annotations

__all__ = ["tukey_hsd", "bonferroni", "TukeyResult", "PosthocResult"]

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar, xp_asarray


@dataclass
class PairwiseComparison:
    group_i: int
    group_j: int
    mean_diff: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    reject: bool


@dataclass
class TukeyResult:
    comparisons: List[PairwiseComparison]
    alpha: float
    n_groups: int
    df_within: int
    mse: float


@dataclass
class PosthocResult:
    comparisons: List[PairwiseComparison]
    alpha: float
    n_comparisons: int


def _array_size(arr, xp):
    return int(arr.numel()) if xp.__name__ == "torch" else int(arr.size)


def _prepare_groups(groups, backend, dtype, min_size, label):
    resolved = _resolve_backend(backend, *groups)
    xp = _get_xp(resolved)
    float_dtype = xp.float64 if dtype is None else dtype
    ref = None
    for group in groups:
        if type(group).__module__.startswith(("torch", "cupy")):
            ref = group
            break

    arrays = []
    for index, group in enumerate(groups):
        arr = xp_asarray(group, dtype=float_dtype, xp=xp, ref_arr=ref).ravel()
        if _array_size(arr, xp) < min_size:
            raise ValueError(
                f"Group {index} must have at least {min_size} observations for {label}"
            )
        if not bool(_to_float_scalar(xp.all(xp.isfinite(arr)))):
            raise ValueError(f"Group {index} contains NaN or infinite values")
        arrays.append(arr)
    return resolved, xp, arrays


def tukey_hsd(
    *groups: Any,
    alpha: float = 0.05,
    backend: str = "auto",
    dtype: Any = None,
) -> TukeyResult:
    """Perform Tukey's honestly significant difference test.

    Group reductions remain on the selected backend. The studentized-range CDF
    and quantile are scalar SciPy operations because neither CuPy nor Torch
    provides that distribution.
    """
    if len(groups) < 2:
        raise ValueError("tukey_hsd requires at least 2 groups")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")

    _, xp, arrays = _prepare_groups(groups, backend, dtype, 2, "Tukey HSD")
    k = len(arrays)
    sizes = [_array_size(group, xp) for group in arrays]
    means = [_to_float_scalar(xp.mean(group)) for group in arrays]
    N = int(sum(sizes))
    df_within = N - k
    if df_within <= 0:
        raise ValueError("Tukey HSD requires positive within-group degrees of freedom")
    ss_within = sum(
        _to_float_scalar(xp.sum((group - mean) ** 2))
        for group, mean in zip(arrays, means)
    )
    mse = ss_within / float(df_within)

    try:
        from scipy.stats import studentized_range
    except ImportError as exc:
        raise ImportError("Tukey HSD requires scipy.stats.studentized_range") from exc

    q_crit = float(studentized_range.ppf(1.0 - alpha, k, df_within))
    comparisons = []
    for i in range(k):
        for j in range(i + 1, k):
            mean_diff = means[i] - means[j]
            harmonic = 2.0 / (1.0 / sizes[i] + 1.0 / sizes[j])
            se = float(np.sqrt(mse / harmonic))
            if se == 0.0:
                q_stat = 0.0 if mean_diff == 0.0 else float("inf")
            else:
                q_stat = abs(mean_diff) / se
            pvalue = float(studentized_range.sf(q_stat, k, df_within))
            if not np.isfinite(pvalue):
                pvalue = 1.0 if q_stat == 0.0 else 0.0
            margin = q_crit * se
            comparisons.append(
                PairwiseComparison(
                    group_i=i,
                    group_j=j,
                    mean_diff=float(mean_diff),
                    pvalue=float(min(max(pvalue, 0.0), 1.0)),
                    ci_lower=float(mean_diff - margin),
                    ci_upper=float(mean_diff + margin),
                    reject=bool(pvalue < alpha),
                )
            )

    return TukeyResult(
        comparisons=comparisons,
        alpha=float(alpha),
        n_groups=k,
        df_within=df_within,
        mse=float(mse),
    )


def bonferroni(
    *groups: Any,
    alpha: float = 0.05,
    backend: str = "auto",
    dtype: Any = None,
) -> PosthocResult:
    """Perform Bonferroni-corrected pairwise Welch tests.

    Means and variances are computed on the selected backend. Only the scalar
    Welch statistics are passed to the CPU t distribution.
    """
    if len(groups) < 2:
        raise ValueError("bonferroni requires at least 2 groups")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")

    _, xp, arrays = _prepare_groups(groups, backend, dtype, 2, "t-test")
    k = len(arrays)
    m = k * (k - 1) // 2
    alpha_bonf = alpha / m

    from statgpu.inference._distributions_backend import get_distribution

    t_dist = get_distribution("t", backend="numpy")
    sizes = [_array_size(group, xp) for group in arrays]
    means = [_to_float_scalar(xp.mean(group)) for group in arrays]
    variances = [
        _to_float_scalar(xp.sum((group - mean) ** 2)) / float(size - 1)
        for group, mean, size in zip(arrays, means, sizes)
    ]

    comparisons = []
    for i in range(k):
        for j in range(i + 1, k):
            ni, nj = sizes[i], sizes[j]
            mean_diff = means[i] - means[j]
            var_i, var_j = variances[i], variances[j]
            se2 = var_i / ni + var_j / nj
            se = float(np.sqrt(max(se2, 0.0)))

            if se == 0.0:
                pvalue = 1.0 if mean_diff == 0.0 else 0.0
                margin = 0.0
            else:
                t_stat = mean_diff / se
                numerator = se2**2
                denominator = (
                    (var_i / ni) ** 2 / (ni - 1)
                    + (var_j / nj) ** 2 / (nj - 1)
                )
                df = numerator / denominator if denominator > 0.0 else float("inf")
                pvalue = min(
                    2.0 * _to_float_scalar(t_dist.sf(abs(t_stat), df)), 1.0
                )
                critical = _to_float_scalar(t_dist.isf(alpha_bonf / 2.0, df))
                margin = critical * se

            comparisons.append(
                PairwiseComparison(
                    group_i=i,
                    group_j=j,
                    mean_diff=float(mean_diff),
                    pvalue=float(pvalue),
                    ci_lower=float(mean_diff - margin),
                    ci_upper=float(mean_diff + margin),
                    reject=bool(pvalue < alpha_bonf),
                )
            )

    return PosthocResult(
        comparisons=comparisons,
        alpha=float(alpha),
        n_comparisons=m,
    )
