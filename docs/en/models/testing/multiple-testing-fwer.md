# Family-wise error control: Bonferroni, Holm and Hochberg

> Language: English  
> Last updated: 2026-10-15  
> Switch: [Chinese](../../../cn/models/testing/multiple-testing-fwer.md)

When even one false claim in a family would be costly, control the chance of
making any false rejection. The family-wise error rate is

$$\mathrm{FWER}=P(V\ge1),$$

where $V$ is the number of false rejections in the prespecified family. FWER
control at 5% limits that probability over repeated data sets; it does not say
that a particular rejection has a 5% chance of being an error.
Start with the [multiple-testing overview](multiple-testing.md) to distinguish
individual discoveries from a global test and to define the family.

## Choose a procedure before looking at the results

All three procedures require valid individual p-values: under each true null,
$P(p_i\le t)\le t$ for $0\le t\le1$. Merely falling in `[0,1]` is
not sufficient. Adjustment cannot repair invalid original tests, and the
family must not exclude performed tests just because their p-values are large.

- Bonferroni uses the same cutoff $\alpha/m$ for every test and controls FWER
  under arbitrary dependence.
- Holm relaxes the cutoff as successive smallest p-values pass. It also
  controls FWER under arbitrary dependence and is never less powerful than
  Bonferroni for the same family.
- Hochberg uses a step-up rule. It controls FWER under independence or
  dependence conditions that justify the Simes inequality. Pairwise
  nonnegative correlations alone are not a universal guarantee.

If some false leads are acceptable in a screening stage, consider
[FDR control with BH or BY](multiple-testing-fdr.md). Choosing a method because
it happens to reject more hypotheses is not a substitute for these assumptions.
The [R p.adjust reference](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
discusses the procedures and their dependence conditions.

## Adjustment formulas

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

This is a **step-down** rule: starting at rank 1, reject and continue while
$p_{(i)} \leq \alpha / (m - i + 1)$. At the first failed comparison, stop;
do not reject that hypothesis or any remaining hypothesis. Equality passes,
and if every comparison passes, reject all hypotheses. The API expresses the
same rule through adjusted p-values.

For ordered p-values $p_{(1)}\le\cdots\le p_{(m)}$, the adjusted values are

$$\tilde p^{\rm Holm}_{(i)}=\min\{1,\max_{j\le i}(m-j+1)p_{(j)}\}.$$

The cumulative maximum preserves the stopping rule. Results are restored to
the original hypothesis order, and the decision is `adjusted <= alpha`.

The following standalone CPU example represents five tests in one prespecified
family. Run the three blocks in order.

**Calculation steps**

1. Define the family with `axis`: `None` pools all entries; an integer
   adjusts each slice along that axis separately. Let $m$ be its size. Supply
   finite p-values in `[0,1]` from valid tests and a finite $0<\alpha<1$.
2. Sort the family's p-values in ascending order, retaining their original
   indices. Assign ranks $i=1,\ldots,m$.
3. Multiply each sorted value by its remaining-test factor:
   $r_i=(m-i+1)p_{(i)}$.
4. From rank 1 toward rank $m$, take the cumulative maximum
   $q_i=\max_{j\le i}r_j$, then cap each value at 1.
5. Restore the original indices and shape. Return the adjusted values and
   the rejection mask `adjusted <= alpha` (including equality).

<!-- api-example: multiple-testing-fwer -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

Create one p-value per hypothesis. The order identifies the five hypotheses.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

Apply Holm at the prespecified 5% family-wise level.

```python
reject, adjusted = adjust_pvalues(p, method="holm", alpha=0.05, backend="numpy")
print(reject.tolist())
print(np.round(adjusted, 3))
```
<!-- example-end: multiple-testing-fwer -->

The mask is `[True, True, False, False, False]`; adjusted values are
`[0.005, 0.04, 0.09, 0.1, 0.5]`. Reject the first two hypotheses at the stated
family/error criterion. Adjusted p-values are not an importance ranking, and
non-rejection does not establish that a null hypothesis is true.

### Bonferroni: use one cutoff for the whole family

Bonferroni is a simple baseline when you want protection against any false
rejection without assumptions about dependence. Giving each of the $m$ tests
a cutoff of $\alpha/m$ makes the sum of their false-rejection bounds at most
$\alpha$. The union bound then controls the probability of at least one error.

For each original p-value, the adjusted value is

$$\tilde p^{\rm Bonf}_i=\min(1,mp_i).$$

Unlike Holm, this cutoff does not become less strict as other tests pass.
Run this standalone example on the same illustrative family.

**Calculation steps**

1. Choose the family with `axis` (`None` pools all entries; an integer
   defines separate slices), and count its $m$ tests. Require finite p-values
   in `[0,1]` from valid tests and a finite $0<\alpha<1$.
2. Multiply each original p-value by $m$: $r_i=mp_i$. No sorting, ranks or
   cumulative operation is needed for this mathematical procedure.
3. Cap each result at 1, keeping the original hypothesis order and shape.
4. Return the adjusted values and `adjusted <= alpha`, including equality.

<!-- api-example: multiple-testing-bonferroni -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

Create one p-value per hypothesis. The order identifies the five hypotheses.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

Apply Bonferroni at the same 5% family-wise level.

```python
reject_bonferroni, adjusted_bonferroni = adjust_pvalues(
    p, method="bonferroni", alpha=0.05, backend="numpy",
)
print(reject_bonferroni.tolist())
print(np.round(adjusted_bonferroni, 3))
```
<!-- example-end: multiple-testing-bonferroni -->

The mask is `[True, True, False, False, False]`; adjusted values are
`[0.005, 0.05, 0.15, 0.25, 1.0]`. Equality counts as a rejection,
so the second test passes at 5%. Holm rejects the same two tests here, but its
adjusted values are smaller for several hypotheses. Matching decisions in
this example does not mean the two procedures are equivalent.

### Hochberg's step-up decision

Under the independence or Simes conditions above, start from the largest
p-value and find the largest rank $k$ satisfying
$p_{(k)} \leq \alpha / (m - k + 1)$. Reject every hypothesis from rank 1
through $k$, including equality. If no rank passes, reject none. For example,
with sorted p-values `[0.0625, 0.0625, 0.5]` and `alpha=0.125`, rank 3 fails
but rank 2 passes at equality, so reject both of the first two hypotheses.

The adjusted-value form is

$$\tilde p^{\rm Hochberg}_{(i)}=\min\{1,\min_{j\ge i}(m-j+1)p_{(j)}\}.$$

The reverse cumulative minimum preserves the step-up decision. Although its
rankwise cutoffs match Holm's, the scan and stopping rule differ. In the
example below, Holm would stop at rank 1 because $0.0625>0.125/3$;
Hochberg can reject the first two hypotheses because rank 2 passes.

**Calculation steps**

1. Choose the family with `axis` (`None` pools all entries; an integer
   defines separate slices), with $m$ finite p-values in `[0,1]` from valid
   tests and a finite $0<\alpha<1$. Check the dependence assumptions above.
2. Sort ascending and retain the original indices; assign ranks
   $i=1,\ldots,m$.
3. Form $r_i=(m-i+1)p_{(i)}$, the same rankwise factors as Holm.
4. From rank $m$ back toward rank 1, take the cumulative minimum
   $q_i=\min_{j\ge i}r_j$, then cap each value at 1.
5. Restore the original indices and shape, and return the adjusted values
   with `adjusted <= alpha`. This reverse minimum, rather than Holm's forward
   maximum, implements the step-up rule.

<!-- api-example: multiple-testing-hochberg -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

Create the three p-values used in the boundary example.

```python
p_hochberg = np.array([0.0625, 0.0625, 0.5])
```

Apply Hochberg at the 12.5% level chosen for this equality example.

```python
reject_hochberg, adjusted_hochberg = adjust_pvalues(
    p_hochberg, method="hochberg", alpha=0.125, backend="numpy",
)
print(reject_hochberg.tolist())
print(adjusted_hochberg.tolist())
```
<!-- example-end: multiple-testing-hochberg -->

The mask is `[True, True, False]`; adjusted values are
`[0.125, 0.125, 0.5]`. Both leading hypotheses pass at equality.
This example illustrates a step-up decision; it does not remove Hochberg's
dependence requirements.

## Interpret the decisions within their family

Each adjusted value can be compared with the chosen level without manually
sorting and restoring the hypothesis order. A rejection is evidence against
that null under the stated assumptions, not an estimate of effect size or
practical importance. Non-rejection does not establish that the null is true.

Separate rowwise adjustment does not jointly control all rows. Use the
[family-selection example](multiple-testing.md#choose-the-hypothesis-family-with-axis)
to decide which tests belong together before calling the function.

For complete signatures, aliases, return shapes, axes, backend choices,
missing-value handling and numerical limitations, see
the [API guide](../../guides/multiple-testing-combine-pvalues.md).

