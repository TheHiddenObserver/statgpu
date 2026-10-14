# Classical one-way ANOVA

<a id="one-way-anova-classical-and-welch-tests"></a>

> Language: English\
> Last updated: 2026-10-12\
> Switch: [Chinese](../../../cn/models/testing/anova-oneway.md)

<a id="the-question"></a>

## Start with the scientific question

Use classical one-way ANOVA when independent groups share a common population
variance and the question is whether their population means are all equal:

$$H_0:\mu_1=\mu_2=\cdots=\mu_k.$$

For example, compare a numerical response across treatments applied to different
experimental units. `f_oneway` tests this global null; it does not identify
individual pairs. For the shared concepts and method selection, start with the
[ANOVA overview](anova.md).

<a id="choose-the-test"></a>
<a id="choose-the-variance-model-before-interpreting-a-p-value"></a>

## When the classical model is appropriate

- Observations must be independent, with normally distributed errors within
  each group and a common population variance. Equal group sizes are not required.
- Check the study design and within-group distributions before testing.
  The function does not diagnose dependence, outliers or its model assumptions.
- If a common variance is not reasonable, use the separate
  [Welch one-way guide](anova-welch.md). Repeated, paired or clustered observations
  need a model for their dependence; `f_oneway` does not provide that analysis.

<a id="classical-anova"></a>

## Why a variance ratio tests means

For observation $j$ in group $i$, the classical model is

$$y_{ij}=\mu_i+\varepsilon_{ij},\qquad
\varepsilon_{ij}\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2).$$

Each group has its own mean $\mu_i$; all errors are independent and identically
distributed normal variables with one common variance $\sigma^2>0$.
For $k$ groups, let group $i$ have $n_i$ observations and mean $\bar y_i$,
with total sample size $N=\sum_i n_i$.
The grand mean weights each group by its size:

$$\bar y=\frac{\sum_i n_i\bar y_i}{N}.$$

Variation between group means and variation left within groups are

$$SS_B=\sum_i n_i(\bar y_i-\bar y)^2,\qquad
SS_W=\sum_i\sum_j(y_{ij}-\bar y_i)^2.$$

The $k$ group means supply $k-1$ independent between-group contrasts.
Estimating one mean in each group leaves $\sum_i(n_i-1)=N-k$ within-group
degrees of freedom. Dividing the sums of squares by these degrees of freedom
gives two variance estimates. Under equal population means and a common
variance, they estimate the same noise level.
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
associated with this grouping. For interpretation and uncertainty, see the
[effect-size overview](anova.md#effect-sizes). The estimated mean difference
is about `0.582` in the response's original units; its practical importance
depends on the application.

<a id="effect-sizes"></a>
<a id="distinguish-significance-from-effect-size"></a>

## Effect sizes for the classical calculation

`f_oneway` returns $\eta^2=SS_B/(SS_B+SS_W)$.
`cohens_f(*groups)` runs the same classical calculation and transforms it as
$f=\sqrt{\eta^2/(1-\eta^2)}$; the example gives `0.334`.
In this single-effect design, partial eta-squared equals eta-squared.
For what these measures mean, how they differ in other designs, and the limits
of the available helpers, see the [effect-size overview](anova.md#effect-sizes).

<a id="parameters-and-pitfalls"></a>

## Parameters and input checks that matter

Pass one array per group to `f_oneway(*groups, backend="auto", dtype=None)`.
`cohens_f` accepts the same inputs. `dtype=None` uses float64, the recommended
starting point for inference. Full signatures, result fields, flattening rules
and backend behavior are in the [ANOVA API reference](../../reference/anova-api.md#f-oneway).

- Supply at least two non-empty groups, with total sample size $N>k$.
  A singleton group is allowed if this total-size condition holds.
- Use finite real observations. NaN and infinite values raise `ValueError`;
  decide how to handle missing data before testing.
- If every observation is identical, statistic, p-value and eta-squared are
  undefined (`NaN`). If all groups are constant but their means differ, F is
  infinite, p-value is zero and eta-squared is one.
- A non-significant result does not establish equivalence, and a small p-value
  does not establish a useful difference. See [interpreting ANOVA](anova.md).

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
CUDA. Explicit backends do not silently select another array library.
Results contain Python scalars. See the
[API execution notes](../../reference/anova-api.md#backend-dtype-device) for automatic
selection, device placement, scalar distribution calculations and numerical limits.

<a id="comparing-implementations"></a>

## Comparing implementations

Use the same finite one-dimensional observations, group boundaries and float64
precision. Compare F and p-value with
[SciPy `f_oneway`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html),
using its equal-variance calculation (`equal_var=True` when that option is
available; this is also the default in older versions). SciPy's array-axis,
missing-value and invalid-input behavior differs from statgpu's; compare the
same valid samples rather than assuming the interfaces are interchangeable.

<a id="welch-intuition"></a>
<a id="welch-weight-each-mean-by-its-precision"></a>
<a id="anova-welch"></a>
<a id="a-complete-unequal-variance-example"></a>

## Unequal-variance alternative

See the standalone Welch guide for
[precision weighting and the corrected F test](anova-welch.md#welch-intuition)
and the [complete unequal-variance example](anova-welch.md#anova-welch).

<a id="references"></a>

## References and next steps

- [NIST: one-way ANOVA](https://itl.nist.gov/div898/handbook/ppc/section2/ppc231.htm).
- [SciPy: classical one-way ANOVA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html).
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed.
- Method guides: [classical one-way](anova-oneway.md), [Welch one-way](anova-welch.md),
  and [balanced two-way](anova-twoway.md).
- [ANOVA overview](anova.md), [post-hoc comparisons](anova-posthoc.md),
  and the [complete API reference](../../reference/anova-api.md).
