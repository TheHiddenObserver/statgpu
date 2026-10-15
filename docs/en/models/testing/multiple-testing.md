# Multiple testing and p-value combination

> Language: English  
> Last updated: 2026-10-14\
> Switch: [Chinese](../../../cn/models/testing/multiple-testing.md)

This page explains why multiple testing changes the question, how to choose an
error criterion, and where to learn each procedure. For complete signatures,
return shapes, aliases and backend options, use the [API reference](../../guides/multiple-testing-combine-pvalues.md).

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

## Choose a tutorial

Choose the family and error criterion before looking at results; do not keep only
the smallest observed p-values and then adjust that subset.

| Your question | Error target / assumption | Tutorial |
|---|---|---|
| Which individual hypotheses can I reject while limiting any false claim? | FWER; Holm/Bonferroni allow arbitrary dependence; Hochberg needs independence or Simes conditions | [FWER: Bonferroni, Holm, Hochberg](multiple-testing-fwer.md) |
| Which candidates should enter a discovery list? | FDR; BH needs independence or suitable positive dependence (PRDS); BY allows arbitrary dependence | [FDR: BH and BY](multiple-testing-fdr.md) |
| Does the evidence contradict the intersection null that every component null holds? | One global test; calibration depends on the method and joint null distribution | [P-value combination: Fisher, Stouffer, Cauchy/ACAT](pvalue-combination.md) |

These are different scientific targets. Choose the error criterion and check its
assumptions, rather than choosing whichever method returns more rejections.

## A complete CPU example

These five illustrative p-values form one prespecified family. Holm controls
FWER under arbitrary dependence when the individual p-values are valid. At the
chosen 5% level, compare its adjusted p-values with `alpha` (including equality).
The [FWER tutorial](multiple-testing-fwer.md) explains the stopping rule.

<!-- api-example: multiple-testing-learner -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

Prepare the prespecified family.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

Calculate and inspect the decisions.

```python
reject, adjusted = adjust_pvalues(p, method="holm", alpha=0.05, backend="numpy")
print(reject.tolist())
print(np.round(adjusted, 3))
```
<!-- example-end: multiple-testing-learner -->

The rejection mask is `[True, True, False, False, False]`, and adjusted p-values
are `[0.005, 0.04, 0.09, 0.1, 0.5]`, in the original hypothesis order. Reject the
first two hypotheses under this family/error criterion. Adjusted p-values are
not an importance ranking; non-rejection does not establish that a null hypothesis is true.

## Choose the hypothesis family with `axis`

The following standalone example contrasts one family of four tests with two
families of two tests. `axis=None` pools all entries; `axis=1` processes each row
separately, and `axis=0` processes columns. Negative axes follow array conventions.
Choose the family from the scientific question before inspecting the results.
For the Fisher calculations below, p-values within each row must be independent
and continuous uniform under their null hypotheses.

<!-- api-example: multiple-testing-families -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

p_matrix = np.array([[0.01, 0.04], [0.20, 0.80]])
reject_all, adjusted_all = adjust_pvalues(p_matrix, method="holm", axis=None, backend="numpy")
reject_rows, adjusted_rows = adjust_pvalues(p_matrix, method="holm", axis=1, backend="numpy")
print(reject_all.tolist())
print(np.round(adjusted_all, 4).tolist())
print(reject_rows.tolist())
print(np.round(adjusted_rows, 4).tolist())
row_stat, row_p_global = combine_pvalues(p_matrix, method="fisher", axis=1, backend="numpy")
print(np.round(row_p_global, 6).tolist())
```
<!-- example-end: multiple-testing-families -->

```text
[[True, False], [False, False]]
[[0.04, 0.12], [0.4, 0.8]]
[[True, True], [False, False]]
[[0.02, 0.04], [0.4, 0.8]]
[0.00353, 0.453213]
```

The rowwise adjustments reject both hypotheses in the first row; pooling all
four tests rejects only the first. Separate rowwise FWER/FDR control does **not**
automatically control the corresponding error rate across all entries jointly.
If the discovery claim covers all four tests, adjust the whole family.

Combination removes the selected axis, so `row_p_global.shape == (2,)`.
At 5%, the first row provides evidence against its intersection null, and the
second does not; neither result identifies a false component hypothesis.
With `axis=None`, combination returns one statistic and one global p-value.
If discoveries are made across many row-level global tests, their global
p-values also need an appropriate multiple-testing procedure.

## Continue learning or look up an interface

- [FWER tutorial](multiple-testing-fwer.md): Bonferroni, Holm and Hochberg, their formulas and decisions
- [FDR tutorial](multiple-testing-fdr.md): BH and BY, step-up decisions and discovery-list interpretation
- [P-value combination tutorial](pvalue-combination.md): global nulls, calibration, weights and direction
- [Complete API reference](../../guides/multiple-testing-combine-pvalues.md): parameters, aliases, shapes, backends, missing values, endpoint handling and numerical limits
- [Estimator helpers](../../reference/estimator-api.md): the estimator-level dictionary interface

For a large correlated analysis, define the global versus individual hypothesis
question and the dependence assumptions first; an application name does not
determine the correct procedure.
