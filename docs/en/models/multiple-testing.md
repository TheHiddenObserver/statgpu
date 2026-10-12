# Multiple testing and p-value combination

> Language: English  
> Last updated: 2026-10-12\
> Switch: [Chinese](../../cn/models/multiple-testing.md)

This page explains why multiple testing changes the question, how to choose an
error criterion, and how the main procedures work. For complete signatures,
return shapes, aliases and backend options, use the [API reference](../guides/multiple-testing-combine-pvalues.md).

## Why repeated 5% tests accumulate risk

Suppose all null hypotheses are true and each test has an exact 5% chance of a
false rejection. For **independent** tests, the chance of avoiding every false
rejection is the product of their individual probabilities, so

$$P(\text{at least one false rejection})=1-(1-0.05)^m.$$

With five tests this is about 22.6%; with twenty it is about 64.2%. The 5%
threshold applies to each test separately, not to the whole collection. Under
dependence the product formula need not hold, but testing each hypothesis at
5% still does not generally provide 5% family-wise protection.

An adjustment accounts for the opportunities to make a false discovery. It
cannot fix a poorly calibrated original test, an invalid model, or excluding
performed tests from the adjustment family after inspecting their p-values.

## Two different questions

Use `adjust_pvalues` when you need decisions for a family of individual hypotheses. Use `combine_pvalues` when several tests contribute to one global null hypothesis. A significant combined p-value does not identify which component hypotheses are false and does not provide individual FDR control.

For example, “which of these five outcomes changed?” calls for individual
adjusted decisions; “is there evidence that at least one outcome changed?” can
call for a global test. Combining several studies of one scientific null is
another global-testing use. Neither operation estimates effect sizes or tells
you whether a detected effect matters in practice.

## Choose the error criterion first

For individual discoveries, let $V$ be the number of false rejections and $R$
the total number of rejections:

$$\mathrm{FWER}=P(V\ge1),\qquad
\mathrm{FDR}=E\left[\frac{V}{\max(R,1)}\right].$$

- **FWER at 5%** limits the probability of making even one false rejection in
  the prespecified family. This is useful when a single false claim is costly.
- **FDR at 5%** limits the expected fraction of false discoveries among the
  rejections, with the fraction defined as zero when nothing is rejected. The
  expectation is over repeated data sets: it does **not** guarantee that every
  realized discovery list contains at most 5% errors, or assign a 5% probability
  of being false to each reported discovery. It is often useful for screening
  when some false leads are tolerable and will be investigated further.

Both require valid individual p-values: under a true null,
$P(p_i\le t)\le t$ for $0\le t\le1$. A number merely lying in `[0,1]` is not
enough. The procedure's dependence assumptions must also hold.

- **Holm or Bonferroni:** FWER control under arbitrary dependence. Bonferroni uses the same cutoff `alpha / m` for every test; Holm relaxes its cutoff as the smallest p-values pass successive comparisons. Holm is never less powerful than Bonferroni for the same family.
- **BH:** FDR control under independence or suitable positive dependence, such as PRDS on true nulls. Its rank-dependent cutoffs allow a larger set of small p-values to be selected together. It is not guaranteed for every dependence structure.
- **BY:** FDR control under arbitrary dependence, with an extra harmonic-factor correction. This broader protection generally comes with fewer discoveries than BH.
- **Hochberg:** FWER control under independence or dependence conditions that justify the Simes inequality. Pairwise nonnegative correlations alone are not a universal guarantee.

Choose the family and error criterion before looking at results; do not keep only the smallest observed p-values and then adjust that subset. With a matrix, `axis=1` treats each row as a separate family; it does not control errors across all rows jointly. The [R p.adjust reference](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html) discusses these procedures and their dependence assumptions.

## A complete CPU example

The values below represent five illustrative tests in one prespecified family.
Run the blocks in order; first import the adjustment and combination functions.

<!-- api-example: multiple-testing-learner -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues
```

The one-dimensional `p` array has shape `(5,)`, with one valid p-value per test.
Its order identifies the hypotheses and is preserved in the results.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

### Make individual decisions with Holm

Control family-wise error at a prespecified 5% level. Holm orders the p-values
from smallest to largest and compares rank $i$ with $\alpha/(m-i+1)$:

| Rank | Sorted p-value | Holm cutoff at 5% | Decision |
|---|---|---|---|
| 1 | 0.001 | 0.05 / 5 = 0.0100 | Reject; continue |
| 2 | 0.010 | 0.05 / 4 = 0.0125 | Reject; continue |
| 3 | 0.030 | 0.05 / 3 ≈ 0.0167 | Do not reject; stop |
| 4 | 0.050 | 0.05 / 2 = 0.0250 | Not rejected after the stop |
| 5 | 0.500 | 0.05 / 1 = 0.0500 | Not rejected after the stop |

This is a **step-down** rule: once a comparison fails, all later hypotheses
remain unrejected. The API expresses the same rule through adjusted p-values.

```python
reject, adjusted = adjust_pvalues(p, method="holm", alpha=0.05, backend="numpy")
print(reject.tolist())
print(np.round(adjusted, 3))
```

The mask is `[True, True, False, False, False]`; adjusted values are
`[0.005, 0.04, 0.09, 0.1, 0.5]`. Reject the first two hypotheses at the stated
family/error criterion. Adjusted p-values are not an importance ranking, and
non-rejection does not establish that a null hypothesis is true.

### Compare with BH for FDR control

If FDR is the intended criterion and the dependence assumptions are appropriate,
BH compares rank $i$ with $i\alpha/m$. Find the **largest** rank $k$ satisfying
$p_{(k)}\le k\alpha/m$, then reject all ranks from 1 through $k$. If no rank
passes, reject none.

| Rank | Sorted p-value | BH cutoff at 5% | Comparison |
|---|---|---|---|
| 1 | 0.001 | 0.010 | Pass |
| 2 | 0.010 | 0.020 | Pass |
| 3 | 0.030 | 0.030 | Pass: largest passing rank |
| 4 | 0.050 | 0.040 | Fail |
| 5 | 0.500 | 0.050 | Fail |

```python
reject_bh, adjusted_bh = adjust_pvalues(p, method="bh", alpha=0.05, backend="numpy")
print(reject_bh.tolist())
print(np.round(adjusted_bh, 4))
```

BH returns `[True, True, True, False, False]` and
`[0.005, 0.025, 0.05, 0.0625, 0.5]`. The third test is selected because the
criterion is now FDR, not the stronger FWER requirement used by Holm. This is
not evidence that BH is always the right choice.

BH is a **step-up** procedure: it must not stop at the first failure when
scanning from the smallest p-value. For example, with `[0.02, 0.03, 0.04]`,
`m=3` and `alpha=0.05`, the first comparison fails (`0.02 > 0.05 / 3`), but
the largest rank passes (`0.04 <= 0.05`), so BH rejects all three hypotheses.

```python
p_step_up = np.array([0.02, 0.03, 0.04])
reject_step_up, adjusted_step_up = adjust_pvalues(
    p_step_up, method="bh", alpha=0.05, backend="numpy",
)
print(reject_step_up.tolist())  # [True, True, True]
```

### Ask a global question with Fisher

Now reuse the same `p` values for a different question: is the intersection null
that all five hypotheses hold inconsistent with the combined evidence? This
interpretation requires independent, continuous uniform null p-values.

```python
statistic, global_p = combine_pvalues(p, method="fisher", backend="numpy")
print(round(float(statistic), 4), round(float(global_p), 6))
```
<!-- example-end: multiple-testing-learner -->

The statistic is about `37.4167` and the global p-value is about `0.000048`.
This provides evidence against the intersection null under the stated assumptions;
it does not identify which component hypotheses are false.

## Adjustment formulas

For ordered p-values $p_{(1)}\le\cdots\le p_{(m)}$, let $c_m=\sum_{j=1}^m1/j$. Restore the original input order after applying:

$$
\begin{aligned}
\tilde p^{\rm Bonf}_{(i)} &= \min(1,mp_{(i)}),\\
\tilde p^{\rm Holm}_{(i)} &= \min\{1,\max_{j\le i}(m-j+1)p_{(j)}\},\\
\tilde p^{\rm BH}_{(i)} &= \min\{1,\min_{j\ge i}(m/j)p_{(j)}\},\\
\tilde p^{\rm BY}_{(i)} &= \min\{1,c_m\min_{j\ge i}(m/j)p_{(j)}\},\\
\tilde p^{\rm Hochberg}_{(i)} &= \min\{1,\min_{j\ge i}(m-j+1)p_{(j)}\}.
\end{aligned}
$$

The returned decision is `adjusted <= alpha`. The cumulative maximum in Holm
preserves its stopping rule; the reverse cumulative minima in BH, BY and
Hochberg preserve their step-up decisions. These adjusted p-values let you
apply a chosen level without manually sorting and restoring the order.
Failing to reject is not evidence that a null hypothesis is true.

## Combination formulas and assumptions

### Fisher: add log evidence

A small p-value contributes a large positive $-2\log p_i$. Adding these terms
is equivalent to using the product of the p-values as evidence against the
global null; it does not average the p-values.

$$T_F=-2\sum_{i=1}^m\log p_i,\qquad p_{\rm global}=P(\chi^2_{2m}\ge T_F).$$

For a continuous uniform null p-value, $-2\log p_i$ has a chi-square
distribution with two degrees of freedom. Independence lets the degrees of
freedom add, giving the $2m$ reference above. This exact calibration does not
automatically apply to discrete or dependent tests. No dependence adjustment is estimated here. `weights` must be `None`; supplied weights raise `ValueError`.

### Stouffer: add standardized evidence

Transform each p-value to $z_i=\Phi^{-1}(1-p_i)$, where $\Phi$ is the standard
normal CDF. Small p-values become large positive scores; values above 0.5
contribute negative scores. For nonnegative weights with positive total,

$$T_Z=\frac{\sum_i w_i\Phi^{-1}(1-p_i)}{\sqrt{\sum_iw_i^2}},\qquad p_{\rm global}=1-\Phi(T_Z).$$

With independent continuous uniform null p-values, the scores are independent
standard normals, and the weighted sum has variance $\sum_i w_i^2$. Dividing
by its square root restores a standard-normal reference. Correlated scores
also contribute covariance terms; this API does not estimate them or provide
a covariance-aware denominator. Equal weights reduce the statistic to
$\sum_i z_i/\sqrt m$. For directional evidence, use consistently oriented one-sided p-values; passing two-sided p-values does not recover effect signs. See the [SciPy combination reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html) for independent-test calibration.

### Cauchy/ACAT: emphasize small p-values through a heavy tail

The tangent transform makes a very small p-value a large positive contribution:
$\tan\{\pi(1/2-p_i)\}\approx1/(\pi p_i)$ near zero. A few small p-values
can therefore carry much of the evidence, though values near one contribute
large negative terms. The implementation normalizes weights,
$a_i=w_i/\sum_jw_j$ (equal weights when omitted), then computes

$$T_C=\sum_i a_i\tan\{\pi(1/2-p_i)\},\qquad p_{\rm global}=1/2-\arctan(T_C)/\pi.$$

With independent continuous uniform null p-values and fixed normalized
nonnegative weights, the statistic is standard Cauchy in exact arithmetic.
Under dependence the Cauchy tail is an approximation under regularity
conditions. For example, the fixed-dimension result in the
[Liu–Xie paper](https://arxiv.org/abs/1808.09011) uses two-sided p-values from
pairwise jointly normal null statistics and concerns the limiting tail. It is
not exact calibration at every significance level for every joint distribution.
It combines evidence; it does not replace multiple-testing adjustment for
individual discoveries.

### Choose a combination method and its weights

After deciding that the target is a global null, check the calibration before
choosing a statistic. Fisher offers an unweighted log-evidence calculation for
independent tests. Stouffer allows relative emphasis through weighted normal
scores under independence. Cauchy/ACAT can be useful for dependent tests when
its tail approximation is justified; it is not a general correction for an
unknown dependence structure.

Stouffer and Cauchy use relative weights: multiplying all weights by the same
positive constant leaves the mathematical result unchanged. The API normalizes
both; this cancels between Stouffer's numerator and denominator, whereas
Cauchy's weights sum to one. A zero weight removes that term from the statistic,
but its p-value must still be a valid input. The functions do not learn weights
from the data. Prespecify them, or justify their independence from the null test
statistics, rather than upweighting small observed p-values. Fisher does not
support weights here.

Combining two-sided p-values cannot recover effect direction; a small global
p-value also does not estimate a common effect. If the scientific question asks
for an effect size and its uncertainty, p-value combination alone is insufficient.

## Practical limits

Supply finite probabilities in `[0,1]`, and a finite `alpha` strictly between 0 and 1. Currently `alpha=NaN` is not rejected and yields an all-false mask; validate a computed level before calling. Invalid p-values are rejected.

Combination formulas use endpoint clipping, and distribution-tail cancellation can limit tiny returned probabilities. Do not interpret a reported zero as an exact probability of zero. See the [complete API guide](../guides/multiple-testing-combine-pvalues.md) for signatures, aliases, weights, axes, backend behavior and numerical limits. Estimator helpers return dictionaries instead of these module-function tuples; see [shared estimator methods](../reference/estimator-api.md).

For Cauchy/Stouffer, also [validate and rescale large weights](../guides/multiple-testing-combine-pvalues.md#validate-and-rescale-combination-weights) before calling: their current normalization can overflow even when each weight is finite, producing a wrong finite result or NaN.

For a large correlated analysis, define the global versus individual hypothesis question and the dependence assumptions first; a domain name such as GWAS does not determine the correct procedure.
