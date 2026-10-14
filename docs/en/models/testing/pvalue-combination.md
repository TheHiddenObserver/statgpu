# Combine p-values: Fisher, Stouffer and Cauchy

> Language: English  
> Last updated: 2026-10-14  
> Switch: [Chinese](../../../cn/models/testing/pvalue-combination.md)

Several tests can contribute evidence to one scientific question: is their
intersection null, under which every component null is true, compatible with
the data? P-value combination produces one statistic and one global p-value
for that question. Combining several studies of one scientific null is another
use, provided the method's calibration assumptions hold.

A small global p-value does not identify which component hypotheses are false.
It does not provide individual FDR control, estimate effect sizes, or establish
practical importance. Use the [multiple-testing overview](multiple-testing.md)
to distinguish this goal from individual decisions with
[FWER](multiple-testing-fwer.md) or [FDR](multiple-testing-fdr.md) adjustment.

## Understand the reference distribution before combining

The transformation is only part of a combination method. A global p-value also
needs a null reference distribution justified for the tests being combined.
Merely having numbers in `[0,1]` is not enough. The exact Fisher and
Stouffer references below assume independent, continuous uniform null p-values;
discrete tests or dependent evidence require additional justification.
Choose the tests and any weights before inspecting which p-values are small.

## Combination formulas and assumptions

### Fisher: add log evidence

A small p-value contributes a large positive $-2\log p_i$. Adding these terms
is equivalent to using the product of the p-values as evidence against the
global null; it does not average the p-values.

$$T_F=-2\sum_{i=1}^m\log p_i,\qquad p_{\rm global}=P(\chi^2_{2m}\ge T_F).$$

For a continuous uniform null p-value, $-2\log p_i$ has a chi-square
distribution with two degrees of freedom. Independence lets the degrees of
freedom add, giving the $2m$ reference above. This exact calibration does not
automatically apply to discrete or dependent tests. No dependence adjustment is estimated here. Fisher is unweighted here.

### Ask a global question with Fisher

The following standalone CPU example asks: is the intersection null
that all five hypotheses hold inconsistent with the combined evidence? This
interpretation requires independent, continuous uniform null p-values.

Fisher adds the contributions $-2\log p_i$: a smaller p-value gives a larger
positive contribution, so a large total is evidence against the global null.
This combines evidence through log p-values rather than averaging them.

<!-- api-example: pvalue-combination-fisher -->
```python
import numpy as np
from statgpu.inference import combine_pvalues
```

Create one p-value per hypothesis. The order identifies the five hypotheses.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

Combine the evidence using Fisher's independent-null reference.

```python
statistic, global_p = combine_pvalues(p, method="fisher", backend="numpy")
print(round(float(statistic), 4), round(float(global_p), 6))
```
<!-- example-end: pvalue-combination-fisher -->

The statistic is about `37.4167` and the global p-value is about `0.000048`.
This provides evidence against the intersection null under the stated assumptions;
it does not identify which component hypotheses are false.

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

Use Stouffer when normal-score evidence and prespecified relative weights fit
the scientific question. In this standalone example, the first test receives
twice the weight of each other test. That choice is illustrative and must be
justified before observing the results.

<!-- api-example: pvalue-combination-stouffer -->
```python
import numpy as np
from statgpu.inference import combine_pvalues
```

Create one p-value per hypothesis. The order identifies the five hypotheses.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

Set the prespecified relative weights and combine the normal scores.

```python
weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
statistic_stouffer, global_p_stouffer = combine_pvalues(
    p, method="stouffer", weights=weights, backend="numpy",
)
print(round(float(statistic_stouffer), 4), round(float(global_p_stouffer), 6))
```
<!-- example-end: pvalue-combination-stouffer -->

The standardized statistic is about `4.2541` and the global p-value is
about `0.000010`. Under the stated independence and calibration
assumptions, this provides evidence against the intersection null. The larger
weight gives the first test more influence; it does not make that component
an individually adjusted discovery or estimate a common effect.

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

Cauchy can be useful when small component p-values are expected to carry the
evidence and its reference or tail approximation is justified. Run this
standalone example with the same prespecified relative weights.

<!-- api-example: pvalue-combination-cauchy -->
```python
import numpy as np
from statgpu.inference import combine_pvalues
```

Create one p-value per hypothesis. The order identifies the five hypotheses.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

Set the prespecified relative weights and apply the tangent transformation.

```python
weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
statistic_cauchy, global_p_cauchy = combine_pvalues(
    p, method="cauchy", weights=weights, backend="numpy",
)
print(round(float(statistic_cauchy), 4), round(float(global_p_cauchy), 6))
```
<!-- example-end: pvalue-combination-cauchy -->

The statistic is about `114.2218` and the global p-value is about
`0.002787`. These are results for the global null under the stated
calibration assumptions. A small value in this example does not establish
exact validity for dependent data at every significance level.

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

## Keep the testing question and numerical limits in view

For a large correlated analysis, define the global versus individual hypothesis
question and the dependence assumptions first; a domain name such as GWAS does
not determine the correct procedure. If many families each produce a global
p-value and you report discoveries among those families, apply an appropriate
multiple-testing procedure to the global p-values as well. See the
[family-selection example](multiple-testing.md#choose-the-hypothesis-family-with-axis).

The [API guide](../../guides/multiple-testing-combine-pvalues.md) is the complete
reference for signatures, aliases, return shapes, axes, backend behavior,
missing-value handling, endpoint clipping and numerical
limits. In particular, review its
[weight validation and rescaling advice](../../guides/multiple-testing-combine-pvalues.md#validate-and-rescale-combination-weights)
before using large weights. Those numerical details matter even when the
statistical assumptions are appropriate.

