# Multiple testing and p-value combination

> Language: English  
> Last updated: 2026-10-05  
> Switch: [Chinese](../../cn/models/multiple-testing.md)

## Two different questions

Use `adjust_pvalues` when you need decisions for a family of individual hypotheses. Use `combine_pvalues` when several tests contribute to one global null hypothesis. A significant combined p-value does not identify which component hypotheses are false and does not provide individual FDR control.

Let V be the number of false rejections and R the total number of rejections. FWER is `P(V >= 1)`; FDR is `E[V / max(R, 1)]`. These guarantees require valid individual p-values and the chosen method's dependence assumptions. Increasing the number of tests does not repair invalid inputs.

## Choose the error criterion first

- **Holm or Bonferroni:** FWER control under arbitrary dependence. Holm is never less powerful than Bonferroni for the same family.
- **BH:** FDR control under independence or suitable positive dependence, such as PRDS on true nulls. It is not guaranteed for every dependence structure.
- **BY:** FDR control under arbitrary dependence, with an extra harmonic-factor correction.
- **Hochberg:** FWER control under independence or dependence conditions that justify the Simes inequality. Pairwise nonnegative correlations alone are not a universal guarantee.

Choose the family before looking at results. With a matrix, `axis=1` treats each row as a separate family; it does not control errors across all rows jointly. The [R p.adjust reference](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html) discusses these procedures and their dependence assumptions.

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

The returned decision is `adjusted <= alpha`. Holm stops rejecting at the first failed step-down comparison. Failing to reject is not evidence that a null hypothesis is true.

## Combination formulas and assumptions

**Fisher:**

$$T_F=-2\sum_{i=1}^m\log p_i,\qquad p_{\rm global}=P(\chi^2_{2m}\ge T_F).$$

The usual chi-square reference assumes independent continuous uniform null p-values. No dependence adjustment is estimated here. `weights` must be `None`; supplied weights raise `ValueError`.

**Stouffer:** for nonnegative weights with positive total,

$$T_Z=\frac{\sum_i w_i\Phi^{-1}(1-p_i)}{\sqrt{\sum_iw_i^2}},\qquad p_{\rm global}=1-\Phi(T_Z).$$

This denominator assumes independent standard-normal null scores. Correlated scores need a covariance-aware calibration that this API does not implement. For directional evidence, use consistently oriented one-sided p-values; passing two-sided p-values does not recover effect signs. See the [SciPy combination reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html) for independent-test calibration.

**Cauchy/ACAT:** the implementation normalizes weights, $a_i=w_i/\sum_jw_j$ (equal weights when omitted), then computes

$$T_C=\sum_i a_i\tan\{\pi(1/2-p_i)\},\qquad p_{\rm global}=1/2-\arctan(T_C)/\pi.$$

Under dependence this is a tail approximation under regularity conditions, not an exact Cauchy distribution or a finite-level guarantee for every possible joint distribution. The [Liu–Xie paper](https://arxiv.org/abs/1808.09011) explains its theoretical scope. It combines evidence; it does not replace multiple-testing adjustment for individual discoveries.

## A complete CPU example

The values below are illustrative p-values; Fisher's interpretation assumes they came from suitably independent tests.

<!-- api-example: multiple-testing-learner -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
reject, adjusted = adjust_pvalues(p, method="holm", alpha=0.05, backend="numpy")
print(reject.tolist())
print(np.round(adjusted, 3))
statistic, global_p = combine_pvalues(p, method="fisher", backend="numpy")
print(round(float(statistic), 4), round(float(global_p), 6))
```

The first output is `[True, True, False, False, False]`; adjusted values are `[0.005, 0.04, 0.09, 0.1, 0.5]`. Interpret the rejection mask at the stated family/error criterion, rather than ranking “importance” by adjusted p-values.

## Practical limits

Supply finite probabilities in `[0,1]`, and a finite `alpha` strictly between 0 and 1. Currently `alpha=NaN` is not rejected and yields an all-false mask; validate a computed level before calling. Invalid p-values are rejected.

Combination formulas use endpoint clipping, and distribution-tail cancellation can limit tiny returned probabilities. Do not interpret a reported zero as an exact probability of zero. See the [complete API guide](../guides/multiple-testing-combine-pvalues.md) for signatures, aliases, weights, axes, backend behavior and numerical limits. Estimator helpers return dictionaries instead of these module-function tuples; see [shared estimator methods](../reference/estimator-api.md).

For a large correlated analysis, define the global versus individual hypothesis question and the dependence assumptions first; a domain name such as GWAS does not determine the correct procedure.
