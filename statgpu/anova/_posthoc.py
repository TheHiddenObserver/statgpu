"""GPU-accelerated post-hoc tests for ANOVA.

Provides :func:`tukey_hsd` and :func:`bonferroni` for pairwise comparisons
after a significant ANOVA result.
"""

from __future__ import annotations

__all__ = ["tukey_hsd", "bonferroni", "TukeyResult", "PosthocResult"]

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np

from statgpu.backends import _get_xp, _resolve_backend, _to_float_scalar, _to_numpy


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PairwiseComparison:
    """A single pairwise comparison.

    Attributes
    ----------
    group_i : int
        Index of the first group.
    group_j : int
        Index of the second group.
    mean_diff : float
        Difference in group means (mean_i - mean_j).
    pvalue : float
        Two-sided p-value for the comparison.
    ci_lower : float
        Lower bound of the confidence interval for mean_diff.
    ci_upper : float
        Upper bound of the confidence interval for mean_diff.
    reject : bool
        True if the null hypothesis (equal means) is rejected at the
        given significance level.
    """
    group_i: int
    group_j: int
    mean_diff: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    reject: bool


@dataclass
class TukeyResult:
    """Result of Tukey HSD post-hoc test.

    Attributes
    ----------
    comparisons : list of PairwiseComparison
        All pairwise comparisons.
    alpha : float
        Significance level used.
    n_groups : int
        Number of groups.
    df_within : int
        Within-group degrees of freedom.
    mse : float
        Mean square error (within-group variance).
    """
    comparisons: List[PairwiseComparison]
    alpha: float
    n_groups: int
    df_within: int
    mse: float


@dataclass
class PosthocResult:
    """Result of Bonferroni post-hoc test.

    Attributes
    ----------
    comparisons : list of PairwiseComparison
        All pairwise comparisons.
    alpha : float
        Significance level used.
    n_comparisons : int
        Number of pairwise comparisons.
    """
    comparisons: List[PairwiseComparison]
    alpha: float
    n_comparisons: int


# ---------------------------------------------------------------------------
# Tukey HSD
# ---------------------------------------------------------------------------

def tukey_hsd(
    *groups: Any,
    alpha: float = 0.05,
    backend: str = "auto",
    dtype: Any = None,
) -> TukeyResult:
    """Perform Tukey's Honestly Significant Difference test.

    Parameters
    ----------
    *groups : array-like
        Two or more sample arrays, one per group.
    alpha : float, default=0.05
        Family-wise significance level.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend.
    dtype : dtype or None, default=None
        Float dtype for computation.

    Returns
    -------
    TukeyResult
        Dataclass with all pairwise comparisons.

    Notes
    -----
    Uses the studentized range distribution for p-value computation.
    When the studentized range distribution is not natively available,
    falls back to scipy.stats.studentized_range.
    """
    if len(groups) < 2:
        raise ValueError("tukey_hsd requires at least 2 groups")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")

    resolved = _resolve_backend(backend, *groups)

    # The studentized-range calculation is CPU based.  Convert through the
    # backend boundary so CuPy arrays and CUDA tensors are supported.
    flat_groups = [np.asarray(_to_numpy(g), dtype=np.float64).ravel() for g in groups]
    for i, g in enumerate(flat_groups):
        if g.size < 2:
            raise ValueError(f"Group {i} must have at least 2 observations for Tukey HSD")
        if not np.all(np.isfinite(g)):
            raise ValueError(f"Group {i} contains NaN or infinite values")

    k = len(flat_groups)
    n_k = np.array([g.size for g in flat_groups], dtype=np.float64)
    means = np.array([g.mean() for g in flat_groups], dtype=np.float64)

    N = n_k.sum()
    df_within = int(N - k)

    # MSE (pooled within-group variance)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in flat_groups)
    mse = ss_within / df_within if df_within > 0 else float("inf")

    # Studentized range distribution
    try:
        from scipy.stats import studentized_range as _srange
        _has_scipy_srange = True
    except ImportError:
        _has_scipy_srange = False

    # Pre-compute F distribution fallback (outside loop)
    if not _has_scipy_srange:
        from statgpu.inference._distributions_backend import get_distribution
        _f_dist = get_distribution("f", backend=resolved)

    comparisons = []
    for i in range(k):
        for j in range(i + 1, k):
            mean_diff = means[i] - means[j]

            # Standard error for the difference (harmonic mean for unequal sizes)
            n_harmonic = 2.0 / (1.0 / n_k[i] + 1.0 / n_k[j])
            se = np.sqrt(mse / n_harmonic) if mse < float("inf") else float("inf")

            # Studentized range statistic
            if se > 0:
                q_stat = abs(mean_diff) / se
            else:
                q_stat = 0.0 if mean_diff == 0.0 else float("inf")

            # P-value from studentized range distribution
            if _has_scipy_srange:
                pvalue = float(_srange.sf(q_stat, k, df_within))
            else:
                pvalue = _to_float_scalar(_f_dist.sf(q_stat ** 2 / 2, k - 1, df_within))

            # Critical value for CI
            if _has_scipy_srange:
                q_crit = float(_srange.ppf(1 - alpha, k, df_within))
            else:
                q_crit = np.sqrt(_to_float_scalar(_f_dist.isf(alpha, k - 1, df_within)) * 2)

            margin = q_crit * se
            ci_lower = mean_diff - margin
            ci_upper = mean_diff + margin

            comparisons.append(PairwiseComparison(
                group_i=i,
                group_j=j,
                mean_diff=mean_diff,
                pvalue=pvalue,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                reject=pvalue < alpha,
            ))

    return TukeyResult(
        comparisons=comparisons,
        alpha=alpha,
        n_groups=k,
        df_within=df_within,
        mse=mse,
    )


# ---------------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------------

def bonferroni(
    *groups: Any,
    alpha: float = 0.05,
    backend: str = "auto",
    dtype: Any = None,
) -> PosthocResult:
    """Perform Bonferroni-corrected pairwise t-tests.

    Parameters
    ----------
    *groups : array-like
        Two or more sample arrays, one per group.
    alpha : float, default=0.05
        Family-wise significance level.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend.
    dtype : dtype or None, default=None
        Float dtype for computation.

    Returns
    -------
    PosthocResult
        Dataclass with all pairwise comparisons.

    Notes
    -----
    Uses Welch's t-test for each pair (does not assume equal variances),
    with Bonferroni correction: the per-comparison alpha is ``alpha / m``
    where ``m`` is the number of comparisons.
    """
    if len(groups) < 2:
        raise ValueError("bonferroni requires at least 2 groups")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")

    resolved = _resolve_backend(backend, *groups)

    # Pairwise Welch tests are CPU based; use the explicit backend boundary.
    flat_groups = [np.asarray(_to_numpy(g), dtype=np.float64).ravel() for g in groups]
    for i, g in enumerate(flat_groups):
        if g.size < 2:
            raise ValueError(f"Group {i} must have at least 2 observations for t-test")
        if not np.all(np.isfinite(g)):
            raise ValueError(f"Group {i} contains NaN or infinite values")

    k = len(flat_groups)
    m = k * (k - 1) // 2  # number of pairwise comparisons
    alpha_bonf = alpha / m if m > 0 else alpha

    # Use t distribution from statgpu.inference
    from statgpu.inference._distributions_backend import get_distribution
    t_dist = get_distribution("t", backend=resolved)

    comparisons = []
    for i in range(k):
        for j in range(i + 1, k):
            ni = flat_groups[i].size
            nj = flat_groups[j].size
            mean_i = flat_groups[i].mean()
            mean_j = flat_groups[j].mean()
            var_i = flat_groups[i].var(ddof=1)
            var_j = flat_groups[j].var(ddof=1)

            mean_diff = mean_i - mean_j

            # Welch's t-test
            se = np.sqrt(var_i / ni + var_j / nj)
            if se == 0.0:
                df = float("inf")
                if mean_diff == 0.0:
                    t_stat = 0.0
                    pvalue = 1.0
                else:
                    t_stat = np.copysign(float("inf"), mean_diff)
                    pvalue = 0.0
                margin = 0.0
            else:
                t_stat = mean_diff / se

                # Welch-Satterthwaite df
                num = (var_i / ni + var_j / nj) ** 2
                den = (var_i / ni) ** 2 / (ni - 1) + (var_j / nj) ** 2 / (nj - 1)
                df = num / den if den > 0 else float("inf")

                # Two-sided p-value
                pvalue_raw = _to_float_scalar(t_dist.sf(abs(t_stat), df)) * 2
                pvalue = min(pvalue_raw, 1.0)

                # Bonferroni-corrected CI
                t_crit = _to_float_scalar(t_dist.isf(alpha_bonf / 2, df))
                margin = t_crit * se
            ci_lower = mean_diff - margin
            ci_upper = mean_diff + margin

            comparisons.append(PairwiseComparison(
                group_i=i,
                group_j=j,
                mean_diff=mean_diff,
                pvalue=pvalue,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                reject=pvalue < alpha_bonf,
            ))

    return PosthocResult(
        comparisons=comparisons,
        alpha=alpha,
        n_comparisons=m,
    )
