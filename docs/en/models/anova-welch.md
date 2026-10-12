# Welch one-way ANOVA

> Language: English\
> Last updated: 2026-10-12\
> Switch: [Chinese](../../cn/models/anova-welch.md)

<a id="the-question"></a>

## Compare means without pooling group variances

Suppose independent treatment groups have different levels of variability,
perhaps with different sample sizes as well. `f_welch` tests the global null

$$H_0:\mu_1=\mu_2=\cdots=\mu_k$$

while estimating uncertainty separately for each group. Its model allows

$$y_{ij}=\mu_i+\varepsilon_{ij},\qquad
\varepsilon_{ij}\sim\mathcal N(0,\sigma_i^2),$$

with independent errors and a potentially different variance $\sigma_i^2$ in
each group. Welch adjusts the F statistic and its reference degrees of freedom
to account for this uncertainty. For shared concepts and method selection, see
the [ANOVA overview](anova.md).

<a id="assumptions"></a>

## When Welch is appropriate

- Use independent observations and a numerical response when equal population
  variances are not a reasonable assumption. Group sizes may differ.
- The F reference distribution is approximate, relying on a normal-error model
  within groups. Unequal-variance adjustment does not protect against severe
  skewness, influential outliers or dependent observations.
- Repeated, paired and clustered observations require a model for dependence.
  `f_welch` does not implement repeated-measures ANOVA, ANCOVA or MANOVA.
- Each group needs at least two observations to estimate its variance.
  The weighted formula below requires all group variances to be positive;
  [constant-group behavior](#parameters-and-pitfalls) is handled separately.

<a id="welch-intuition"></a>

## Welch: weight each mean by its precision

A group with a noisy mean should contribute less evidence than an equally large,
precisely measured group. For $k$ groups, write $n_i$ for group size and
$\bar y_i$ for its mean, with total sample size $N=\sum_i n_i$.
The sample variance $s_i^2=\sum_j(y_{ij}-\bar y_i)^2/(n_i-1)$ estimates that
group's variance. Because the estimated variance of its mean is $s_i^2/n_i$,
Welch uses inverse-variance weights $w_i=n_i/s_i^2$.
For positive group variances, define

$$W=\sum_i w_i,\qquad \bar y_w=\frac{\sum_i w_i\bar y_i}{W},\qquad
U=\sum_i\frac{(1-w_i/W)^2}{n_i-1}.$$

Then the statistic and approximate reference degrees of freedom are

$$F_W=\frac{\sum_i w_i(\bar y_i-\bar y_w)^2/(k-1)}
{1+\frac{2(k-2)}{k^2-1}U},\qquad
\nu_1=k-1,\qquad \nu_2=\frac{k^2-1}{3U}.$$

The denominator correction accounts for estimating the unequal variances.
$\nu_2$ represents the effective denominator degrees of freedom, so a fractional
value is expected.
It is not the classical $N-k$. This is the approach of
[Welch (1951)](https://doi.org/10.2307/2332579).

<a id="anova-welch"></a>

## A complete unequal-variance example

Run the three blocks below in order in a fresh Python session. They compare
three groups with different sizes and population standard deviations, using
NumPy on CPU; no GPU installation or variables from another page are needed.

<!-- example: anova-welch -->
```python
import numpy as np
from statgpu.anova import f_welch
```

Generate 20, 28 and 36 observations. The second argument to `rng.normal` is a
standard deviation, not a variance.

```python
rng = np.random.default_rng(2026)
unequal_groups = [
    rng.normal(10.0, 0.7, 20),
    rng.normal(10.8, 1.5, 28),
    rng.normal(13.0, 3.0, 36),
]
```

Pass one array per group using `*unequal_groups`, and inspect the fractional
denominator degrees of freedom explicitly.

```python
welch = f_welch(*unequal_groups, backend="numpy")
print(f"F={welch.statistic:.3f}, p={welch.pvalue:.3g}")
print(f"df=({welch.df_between}, {welch.df_within:.3f})")
print(f"eta_squared={welch.eta_squared}")
```
<!-- example-end: anova-welch -->

The output is `F=24.825`, `p=3.82e-08`, degrees of freedom `(2, 48.370)`,
and `eta_squared=nan`. There is evidence against equality of all three means;
the test does not by itself locate a difference. For unequal-variance pairwise
questions, see the [Bonferroni pairwise Welch example](anova-posthoc.md#anova-bonferroni).
Do not automatically follow Welch with pooled-variance Tukey comparisons.

<a id="effect-sizes"></a>

## Why the effect-size field is NaN

`f_welch` always returns `eta_squared=NaN`: this API does not provide a
Welch-specific effect-size estimate. In the example, this does not indicate
that the F test failed or that the effect is zero.
Applying `cohens_f` to these observations would run the classical calculation;
it would not convert the Welch result into an unequal-variance effect size.
Report group means and differences in the response's units, with uncertainty
appropriate to the comparison. See the [effect-size overview](anova.md#effect-sizes)
for the distinction between significance and effect magnitude.

<a id="parameters-and-pitfalls"></a>

## Parameters and input checks that matter

Pass one array per group to `f_welch(*groups, backend="auto", dtype=None)`.
The default dtype is float64. There is no trimming option or switch to remove
the Welch correction. Full signatures, result fields, flattening rules and
shared input requirements are in the [API reference](../reference/anova-api.md#f-welch).

- At least two groups, each containing at least two finite real observations,
  are required. NaN and infinite observations raise `ValueError`; missing values
  are not silently omitted.
- If only some groups have zero sample variance, `ValueError` is raised because
  inverse-variance weighting is undefined.
- If every group is constant, F and p-value are `NaN` when the means are treated
  as equal; otherwise F is infinite and p-value is zero. This special case
  reports `df_within=N-k`; eta-squared remains `NaN`. See the
  [degenerate-input rules](../reference/anova-api.md#finite-and-degenerate-inputs)
  for the mean-equality tolerance.
- For ordinary positive-variance samples, retain the fractional `df_within`.
  Rounding it or replacing it with $N-k$ changes the reference distribution.
- A significant global test does not identify a particular pair; a
  non-significant result does not prove equivalence.

<a id="backend-execution"></a>

## Backend selection

`f_welch` supports NumPy, CuPy and Torch arrays. The example explicitly uses
NumPy; for GPU reductions, supply CuPy arrays or Torch CUDA tensors on a common
device and select the matching backend. `backend="torch"` alone does not move
data to CUDA. Results are Python scalars, and distribution evaluation is not
guaranteed to run entirely on GPU. Automatic selection, precision choices and
tail-accuracy limits are described in the
[API execution notes](../reference/anova-api.md#backend-dtype-device).

<a id="comparing-implementations"></a>

## Comparing implementations

Use the same finite one-dimensional observations and group boundaries, float64
precision and positive sample variances for a direct comparison.

- [statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html):
  set `use_var="unequal"`, `welch_correction=True` and `trim_frac=0`.
  Compare F, p-value and both numerator and denominator degrees of freedom.
- [SciPy `f_oneway`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html):
  versions supporting `equal_var=False` provide Welch's test; explicitly set
  that option and compare F and p-value. The default equal-variance calculation
  is a different test, and the SciPy result does not expose degrees of freedom.

A trimmed-mean test changes the target of the comparison, and disabling the
Welch correction changes the statistic. These are not aligned alternatives.

<a id="references"></a>

## References and next steps

- [Welch (1951): On the Comparison of Several Mean Values: An Alternative Approach](https://doi.org/10.2307/2332579).
- [statsmodels: one-way ANOVA methods](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html).
- [SciPy: `f_oneway`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html).
- Method guides: [classical one-way](anova-oneway.md), [Welch one-way](anova-welch.md),
  and [balanced two-way](anova-twoway.md).
- [ANOVA overview](anova.md), [post-hoc comparisons](anova-posthoc.md),
  and the [complete API reference](../reference/anova-api.md).
