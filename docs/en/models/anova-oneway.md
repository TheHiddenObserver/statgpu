# One-way ANOVA: classical and Welch tests

> Language: English\
> Last updated: 2026-10-12\
> Switch: [Chinese](../../cn/models/anova-oneway.md)

<a id="the-question"></a>

## Start with the scientific question

Suppose independent experimental units receive one of several treatments. You
measure one numerical response per unit and want to know whether the population
means differ. One-way ANOVA tests the single global null

$$H_0:\mu_1=\mu_2=\cdots=\mu_k.$$

A small p-value is evidence that **at least one mean differs**. It does not say
which pair differs, that every pair differs, or that the difference is useful.
Report group means, uncertainty and an effect size alongside the test. See
[post-hoc comparisons](anova-posthoc.md) when the question concerns individual
pairs, and the [ANOVA overview](anova.md) for other supported designs.

<a id="choose-the-test"></a>

## Choose the variance model before interpreting a p-value

- Use `f_oneway` for independent groups when a common population variance is
  reasonable. Group sample sizes may differ.
- Use `f_welch` when population variances need not be equal, especially when
  sample sizes also differ. It estimates uncertainty separately for each group.
- Both methods require independent observations and use normal-error models
  for their reference distributions; Welch uses an approximate F distribution.
  Neither solves dependence, severe outliers, or a badly misspecified response
  model. Inspect distributions and the study design rather than choosing the
  method that yields the smaller p-value.
- Repeated measurements on the same people, matched samples and clustered units
  need an analysis that represents that dependence. These functions do not
  implement repeated-measures ANOVA, ANCOVA or MANOVA.

The normality assumption concerns errors within groups, not the pooled outcome
histogram. Independence comes from how observations were collected; a large
sample cannot repair pseudoreplication. The [SciPy ANOVA notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
summarize the classical assumptions and distinguish the Welch alternative.

<a id="classical-anova"></a>

## Why a variance ratio tests means

Let group $i$ have $n_i$ observations, mean $\bar y_i$, and let $N=\sum_i n_i$.
The grand mean weights each group by its size:

$$\bar y=\frac{\sum_i n_i\bar y_i}{N}.$$

Variation between group means and variation left within groups are

$$SS_B=\sum_i n_i(\bar y_i-\bar y)^2,\qquad
SS_W=\sum_i\sum_j(y_{ij}-\bar y_i)^2.$$

Dividing by degrees of freedom gives two variance estimates. Under equal
population means and a common variance, they estimate the same noise level.
A large between-group estimate relative to within-group noise gives evidence
against the null:

$$F=\frac{SS_B/(k-1)}{SS_W/(N-k)},\qquad
(df_1,df_2)=(k-1,N-k).$$

`f_oneway` returns the F statistic and its upper-tail p-value. It also returns
$\eta^2=SS_B/(SS_B+SS_W)$; see [effect sizes](#effect-sizes).
The calculation uses sums of squares, with no iterative solver or tuning grid.

<a id="anova-basic"></a>

## A complete classical CPU example

Compare two independent groups with a common population standard deviation.
Run these three blocks in order in a fresh Python session. The first block
imports everything used in this example.

<!-- example: anova-basic -->
```python
import numpy as np
from statgpu.anova import f_oneway, cohens_f
```

The fixed seed makes the illustration reproducible. Each array contains 100
observations; the order of the arguments identifies the groups.

```python
rng = np.random.default_rng(7)
g1 = rng.normal(0.0, 1.0, 100)
g2 = rng.normal(0.5, 1.0, 100)
```

Compute the omnibus test and summarize the observed means and effect size.
The computation explicitly selects NumPy, so no GPU installation is needed.

```python
result = f_oneway(g1, g2, backend="numpy")
f_effect = cohens_f(g1, g2, backend="numpy")
print(np.round([g1.mean(), g2.mean()], 3))
print(f"F={result.statistic:.3f}, p={result.pvalue:.3g}")
print(f"df=({result.df_between}, {result.df_within})")
print(f"eta_squared={result.eta_squared:.3f}, Cohen f={f_effect:.3f}")
```
<!-- example-end: anova-basic -->

The group means are `[-0.173, 0.409]`. The test gives `F=22.079`,
`p=4.89e-06`, and degrees of freedom `(1, 198)`. At a prespecified 5% level,
reject equal means. With two groups, the classical F statistic equals the
square of the equal-variance two-sample t statistic. With three or more groups,
the same global test does not identify which pair differs.

`eta_squared=0.100` means that about 10.0% of the total **sample** variation is
associated with this grouping. It is not the probability that the null is false,
a causal claim, or a population variance fraction known without uncertainty.
The estimated mean difference is about `0.582` in the response's original units;
its practical importance depends on the application.

<a id="welch-intuition"></a>

## Welch: weight each mean by its precision

A group with a noisy mean should contribute less evidence than an equally large,
precisely measured group. Welch uses sample variances $s_i^2$ and weights
$w_i=n_i/s_i^2$. For positive group variances, define

$$W=\sum_i w_i,\qquad \bar y_w=\frac{\sum_i w_i\bar y_i}{W},\qquad
U=\sum_i\frac{(1-w_i/W)^2}{n_i-1}.$$

Then the statistic and approximate reference degrees of freedom are

$$F_W=\frac{\sum_i w_i(\bar y_i-\bar y_w)^2/(k-1)}
{1+\frac{2(k-2)}{k^2-1}U},\qquad
\nu_1=k-1,\qquad \nu_2=\frac{k^2-1}{3U}.$$

The denominator correction accounts for estimating the unequal variances.
$\nu_2$ is an effective degrees of freedom, so a fractional value is expected.
It is not the classical $N-k$. This is the approach of
[Welch (1951)](https://doi.org/10.2307/2332579).

<a id="anova-welch"></a>

## A complete unequal-variance example

This example starts afresh and needs no variables from the classical example.
It compares three groups with different sizes and population standard deviations.

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

## Distinguish significance from effect size

- Classical one-way `eta_squared` is $SS_B/SS_T$, where $SS_T=SS_B+SS_W$.
  It is a descriptive sample fraction; a small p-value can accompany a small
  effect in a large sample.
- `cohens_f(*groups)` computes $f=\sqrt{\eta^2/(1-\eta^2)}$ by running the
  classical calculation. The example gives `0.334`. Common labels such as
  small or large are conventions, not a substitute for domain knowledge.
- `partial_eta_squared(ss_effect, ss_error)` computes
  $\eta_p^2=SS_{effect}/(SS_{effect}+SS_{error})$. In classical one-way ANOVA
  it equals eta-squared because there is only one effect. In a multifactor
  design, the denominator excludes the other effects, so the two measures
  differ; see the [two-way example](anova-twoway.md#effect-sizes).
- Welch's `eta_squared` is deliberately `NaN`: this API does not return a
  Welch-specific effect-size estimate. Applying `cohens_f` to the same data
  still computes the **classical** descriptive quantity; it does not convert
  Welch's result into a variance-adjusted effect size.

Neither helper supplies an effect-size confidence interval. Report the scale,
comparison and uncertainty appropriate to the scientific question.

<a id="parameters-and-pitfalls"></a>

## Parameters and input checks that matter

`f_oneway`, `f_welch` and `cohens_f` accept `*groups`, `backend="auto"` and
`dtype=None`. `None` uses the selected backend's float64. Prefer this default
for inference; lower precision can matter when differences are small relative
to the outcome scale. Full signatures and result fields are in the
[ANOVA API reference](../reference/anova-api.md).

- Pass separate one-dimensional arrays. Inputs are flattened; a matrix is not
  interpreted as many separate tests. There is no `axis` or `nan_policy` option.
- Classical ANOVA requires at least two non-empty groups and $N>k$. Welch
  requires at least two observations in **each** group.
- NaN and infinite observations raise `ValueError`; missing values are not
  silently omitted. Decide how missingness should be handled before testing.
- If every observation is identical, classical ANOVA returns undefined
  (`NaN`) statistic, p-value and eta-squared. If all groups are constant but
  their means differ, it returns infinite F, p-value zero and eta-squared one.
- Welch raises `ValueError` when only some groups have zero variance. If all
  groups are constant, equal means give undefined F/p and different means give
  infinite F/p zero; its eta-squared remains `NaN`.
- A non-significant result does not establish equivalence. ANOVA also does not
  diagnose outliers, normality or independence for you.

<a id="gpu-execution"></a>

## Optional GPU execution

Complete [the classical CPU example](#anova-basic) first, then choose one block
for an installed, working CUDA runtime. Reuse its data so the statistical
question stays the same. Small examples often do not benefit from GPU transfer
and launch costs.

### CuPy

<!-- example-requires: anova-basic -->
<!-- example: anova-cupy -->
```python
import cupy as cp

g1_gpu, g2_gpu = cp.asarray(g1), cp.asarray(g2)
result_gpu = f_oneway(g1_gpu, g2_gpu, backend="cupy")
```
<!-- example-end: anova-cupy -->

### Torch CUDA

<!-- example-requires: anova-basic -->
<!-- example: anova-torch -->
```python
import torch

g1_torch = torch.as_tensor(g1, device="cuda", dtype=torch.float64)
g2_torch = torch.as_tensor(g2, device="cuda", dtype=torch.float64)
result_torch = f_oneway(g1_torch, g2_torch, backend="torch")
```
<!-- example-end: anova-torch -->

Here `backend` chooses the array library; the Torch tensors explicitly select
CUDA. Explicit backends do not silently select a different array library.
`"auto"` infers the library from group arrays and uses NumPy for ordinary lists
or NumPy inputs. Reductions use the selected backend, while result fields are
Python scalars. Distribution evaluation may involve CPU scalar calculations;
this does not change the statistical test. These functions do not expose
separate strict/approximate mode switches. See the
[API execution notes](../reference/anova-api.md) for the full backend behavior.

<a id="comparing-implementations"></a>

## Comparing implementations

Use the same finite observations, group boundaries and float64 precision.
Classical `f_oneway` can be compared with SciPy's equal-variance `f_oneway`.
For Welch, [statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html)
uses the matching settings `use_var="unequal"`, `welch_correction=True` and
`trim_frac=0`. Compare F, p-value and both degrees of freedom. An equal-variance
reference or a trimmed-mean test answers a different question. Similar function
names do not imply identical options or result containers.

<a id="references"></a>

## References and next steps

- [NIST: one-way ANOVA](https://itl.nist.gov/div898/handbook/ppc/section2/ppc231.htm).
- [Welch (1951): On the Comparison of Several Mean Values: An Alternative Approach](https://doi.org/10.2307/2332579).
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.
- Continue with [balanced two-way ANOVA](anova-twoway.md),
  [post-hoc comparisons](anova-posthoc.md), or the [complete API](../reference/anova-api.md).
