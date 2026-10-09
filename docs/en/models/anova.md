# ANOVA

> Language: English  
> Last updated: 2026-10-09\
> Switch: [Chinese](../../cn/models/anova.md)

## Overview

One-way ANOVA asks whether independent groups have the same population mean.
It compares variation between group means with variation within groups. The
classical F test assumes independent observations, normal errors within groups,
and equal population variances; use Welch ANOVA when the equal-variance
assumption is unsuitable. Neither test identifies which means differ by itself.

The ANOVA module provides one-way ANOVA, balanced two-way ANOVA, Welch ANOVA,
Tukey HSD, Bonferroni-adjusted pairwise Welch tests, and effect-size helpers.
Group reductions support NumPy, CuPy, and Torch backends.

## Paths

- `statgpu.anova.f_oneway`, `statgpu.anova.AnovaResult`
- `statgpu.anova.f_twoway`, `statgpu.anova.TwoWayAnovaResult`
- `statgpu.anova.f_welch`
- `statgpu.anova.tukey_hsd`, `statgpu.anova.TukeyResult`
- `statgpu.anova.bonferroni`, `statgpu.anova.PosthocResult`
- `statgpu.anova.cohens_f`
- `statgpu.anova.partial_eta_squared`

## One-Way ANOVA

For groups with sizes $n_i$, means $\bar y_i$, and total size
$N=\sum_i n_i$, the grand mean is

$$
\bar y = \frac{\sum_i n_i\bar y_i}{N}.
$$

The between- and within-group sums of squares are

$$
SSB = \sum_i n_i(\bar y_i-\bar y)^2,
\qquad
SSW = \sum_i\sum_j(y_{ij}-\bar y_i)^2.
$$

The test statistic is

$$
F = \frac{SSB/(k-1)}{SSW/(N-k)}.
$$

`f_oneway` computes these quantities directly with backend-native reductions; no
iterative solver is used. The p-value is obtained from the F-distribution survival
function. Eta-squared is

$$
\eta^2 = \frac{SSB}{SSB+SSW}.
$$

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `*groups` | required | Two or more one-dimensional samples |
| `backend` | `"auto"` | `"auto"`, `"numpy"`, `"cupy"`, or `"torch"` |
| `dtype` | `None` | Computation dtype where exposed by the function |

### Output

`AnovaResult` exposes:

| Field | Description |
|---|---|
| `statistic` | F statistic |
| `pvalue` | F-distribution tail probability |
| `df_between` | Numerator degrees of freedom |
| `df_within` | Denominator degrees of freedom |
| `eta_squared` | One-way effect size |

<a id="anova-basic"></a>

## Compare two groups on CPU

Run the following blocks in order. Each input is a one-dimensional array of
measurements from one group; the arrays do not need to have the same length.
First import the function and NumPy.

<!-- example: anova-basic -->
```python
import numpy as np
from statgpu.anova import f_oneway
```

Generate two independent samples with equal population variances and different
population means. Both arrays have shape `(100,)`.

```python
rng = np.random.default_rng(7)
g1 = rng.normal(0.0, 1.0, 100)
g2 = rng.normal(0.5, 1.0, 100)
```

Compute the F test, then inspect its statistic, p-value and effect size.

```python
result = f_oneway(g1, g2, backend="numpy")
print(round(result.statistic, 3), result.pvalue)
print(round(result.eta_squared, 3))
```
<!-- example-end: anova-basic -->

The F statistic is about `22.079`, the p-value is about `4.89e-6`, and
eta-squared is about `0.100`. At a prespecified 5% level, reject equal means for
these groups. Eta-squared describes the share of sample variation associated
with the group split; it is not the probability that the null is false or a
causal effect. For more than two groups, a significant F test only says that
not all means are equal; use planned or multiplicity-adjusted comparisons to
locate differences.

## Two-Way ANOVA

`f_twoway` analyzes a balanced two-factor design. It tests factor A, factor B,
and, when requested, the interaction. Unbalanced cell sizes are rejected until
the public API exposes an explicit Type I, II, or III sums-of-squares convention.
When `interaction=False`, the additive model uses the remaining interaction
variation in the residual term.

### Parameters

| Parameter | Default | Description |
|---|---:|---|
| `data` | required | Nested `(a, b)` cells containing observations |
| `interaction` | `True` | Fit and test the interaction term |
| `backend` | `"auto"` | Numerical backend |
| `dtype` | `None` | Computation dtype |

### Output

`TwoWayAnovaResult` reports factor-A, factor-B, and optional interaction
statistics, p-values, degrees of freedom, eta-squared values, residual degrees of
freedom, and residual sum of squares.

## Welch ANOVA

`f_welch` is the unequal-variance alternative to one-way ANOVA. It uses the
Welch-Satterthwaite denominator degrees of freedom, which are generally
fractional. Its returned `AnovaResult.df_within` is therefore a floating-point
value. `eta_squared` is reported as `NaN` because the ordinary pooled-variance
one-way effect size is not the corresponding Welch estimand.

Reuse `rng`, `g1` and `g2` from [the CPU example](#anova-basic).
Add a third group with a larger population variance, then use Welch's test.

<!-- example-requires: anova-basic -->
<!-- example: anova-welch -->
```python
from statgpu.anova import f_welch

g3 = rng.normal(-0.2, 2.0, 80)
welch = f_welch(g1, g2, g3, backend="numpy")
print(welch.statistic, welch.pvalue, welch.df_within)
```
<!-- example-end: anova-welch -->

The p-value tests equality of all three means without pooling their variances.
It still relies on independent observations and does not locate pairwise differences.

## Post-Hoc Comparisons

### Tukey HSD

`tukey_hsd` performs all pairwise mean comparisons using the studentized-range
distribution. It controls family-wise error and reports simultaneous confidence
intervals. `TukeyResult` contains the comparison list, significance level,
number of groups, residual degrees of freedom, and pooled mean square error.
Each comparison reports group indices, mean difference, adjusted p-value,
confidence interval, and rejection decision.

For the equal-variance `g1` and `g2` from [the CPU example](#anova-basic),
inspect the adjusted comparison and its simultaneous confidence interval.
Do not include the unequal-variance `g3` in this pooled-variance example.

<!-- example-requires: anova-basic -->
<!-- example: anova-tukey -->
```python
from statgpu.anova import tukey_hsd

posthoc = tukey_hsd(g1, g2, alpha=0.05, backend="numpy")
comparison = posthoc.comparisons[0]
print(comparison.mean_diff, comparison.ci_lower, comparison.ci_upper)
print(comparison.pvalue, comparison.reject)
```
<!-- example-end: anova-tukey -->

The difference is group 0 minus group 1, about `-0.582`; the interval is
approximately `[-0.826, -0.338]`. Its exclusion of zero agrees with `reject=True`.
With more groups, Tukey adjusts across all pairwise comparisons.

### Bonferroni Pairwise Welch Tests

`bonferroni` applies Welch's pairwise t-test and Bonferroni correction. It does
not assume equal variances. `PosthocResult` reports all pairwise comparisons,
the family-wise significance level, and the number of comparisons.

## Effect Sizes

- `partial_eta_squared(ss_effect, ss_error)` computes
  $ss_{effect}/(ss_{effect}+ss_{error})$ and validates finite, non-negative sums
  of squares.
- `cohens_f(*groups)` derives Cohen's $f$ from eta-squared:

$$
f = \sqrt{\frac{\eta^2}{1-\eta^2}}.
$$

<a id="cpu-and-gpu-examples"></a>

## Optional GPU execution

After [the CPU example](#anova-basic), reuse its `g1`, `g2` and `f_oneway`.
Choose one of the following blocks when its CUDA runtime is available; neither
block regenerates data or changes the statistical question.

### CuPy

```python
import cupy as cp

g1_gpu, g2_gpu = cp.asarray(g1), cp.asarray(g2)
result_gpu = f_oneway(g1_gpu, g2_gpu, backend="cupy")
```

### Torch CUDA

```python
import torch

g1_torch = torch.as_tensor(g1, device="cuda", dtype=torch.float64)
g2_torch = torch.as_tensor(g2, device="cuda", dtype=torch.float64)
result_torch = f_oneway(g1_torch, g2_torch, backend="torch")
```

## Backend and Execution Boundaries

Means, variances, sums of squares, and group reductions remain on the selected
backend. Scalar F, t, normal, or studentized-range distribution evaluations may
cross to CPU where the selected GPU backend does not provide the required
function. Complete group vectors are not transferred solely to compute a
p-value.

`backend="cupy"` selects CuPy and `backend="torch"` selects Torch. Explicit
backend requests do not silently select another backend.

## Strict and Approximate Modes

ANOVA functions do not expose separate strict and approximate statistical
modes. All backends use the same test definitions. A scalar distribution call
on CPU is an execution boundary, not an alternative ANOVA formula.

## Limitations and Failure Modes

- One-way and Welch tests require at least two non-empty groups.
- Two-way ANOVA currently requires balanced cell sizes.
- Non-finite observations are rejected.
- Tukey HSD relies on the studentized-range distribution and may use a CPU scalar
  distribution implementation.
- Effect-size helpers reject invalid sums of squares rather than returning a
  misleading finite value.

## External Validation

Maintained tests compare Welch ANOVA with `statsmodels.stats.oneway.anova_oneway`
and exercise NumPy/Torch parity, degrees-of-freedom semantics, balanced-design
restrictions, effect-size validation, and backend execution boundaries.
Validation claims remain scoped to the exact function, backend, environment, and
commit tested.

## FAQ

### Does Torch input require `backend="torch"`?

Use `backend="torch"` for an explicit Torch execution request. `"auto"` may infer
the backend from input type, but explicit selection is preferable in tests and
benchmarks.

### Why can the returned p-value be a Python scalar?

ANOVA result objects expose statistical summaries as scalars. The sufficient
statistics used to obtain them remain on the selected backend until the final
scalar distribution boundary.

### Why is an unbalanced two-way design rejected?

Different sums-of-squares conventions answer different hypotheses in an
unbalanced design. The implementation fails explicitly rather than silently
choosing a convention.

## References

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). On the comparison of several mean values.
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
