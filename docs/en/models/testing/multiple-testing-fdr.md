# False discovery rate control: BH and BY

> Language: English  
> Last updated: 2026-10-14  
> Switch: [Chinese](../../../cn/models/testing/multiple-testing-fdr.md)

Screening many hypotheses often means accepting some false leads so that
promising signals can be investigated further. False discovery rate (FDR)
control targets the expected fraction of false discoveries among the rejected
hypotheses. It can be more useful for this goal than controlling the chance
of any error with [FWER methods](multiple-testing-fwer.md).

Start with the [multiple-testing overview](multiple-testing.md) to choose
between individual discoveries and one global test.

## What FDR control does and does not guarantee

Let $V$ be the number of false rejections and $R$ the total number of rejections:

$$\mathrm{FDR}=E\left[\frac{V}{\max(R,1)}\right].$$

The fraction is zero when nothing is rejected. At a 5% FDR level, its expectation
over repeated data sets is at most 5% under the procedure's assumptions. This
does **not** guarantee that every realized discovery list contains at most
5% errors, or assign a 5% probability of being false to each discovery.
A single list can have a larger false-discovery fraction.

Both BH and BY require valid individual p-values: under a true null,
$P(p_i\le t)\le t$ for $0\le t\le1$. A number merely lying in `[0,1]`
is not enough. Choose the family and error criterion before looking at results;
do not retain only the smallest observed p-values and then adjust that subset.

- BH controls FDR under independence or suitable positive dependence, such as
  PRDS on true nulls. Its rank-dependent cutoffs allow a larger set of small
  p-values to be selected together. It is not guaranteed for every dependence
  structure.
- BY controls FDR under arbitrary dependence using an extra harmonic factor.
  The broader protection generally means fewer discoveries than BH.

The [R p.adjust reference](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
describes these procedures and their dependence assumptions.

## Work through the FDR methods

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

For ordered p-values $p_{(1)}\le\cdots\le p_{(m)}$, the adjusted values are

$$\tilde p^{\rm BH}_{(i)}=\min\{1,\min_{j\ge i}(m/j)p_{(j)}\}.$$

The reverse cumulative minimum preserves the step-up decision. Results return
to the original hypothesis order, with `adjusted <= alpha` as the
decision rule. Run the following blocks in order as one standalone CPU example.

<!-- api-example: multiple-testing-fdr -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

Create one p-value per hypothesis. The order identifies the five hypotheses.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

Apply BH at the prespecified 5% FDR level.

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
<!-- example-end: multiple-testing-fdr -->

The second example returns adjusted values `[0.04, 0.04, 0.04]`.
All three hypotheses are selected together even though the first rankwise
comparison fails. The larger acceptable cutoff at a later rank is central to
BH's step-up rule.

### BY: allow arbitrary dependence with a harmonic correction

When BH's dependence assumptions are not justified, BY adds a conservative
correction rather than treating dependence as harmless. Define

$$c_m=\sum_{j=1}^m\frac1j.$$

BY runs the BH step-up comparison at level $\alpha/c_m$: find the largest
rank $k$ such that $p_{(k)}\le k\alpha/(mc_m)$, then reject ranks 1 through $k$.
If no rank passes, reject none. Its adjusted values are

$$\tilde p^{\rm BY}_{(i)}=\min\{1,c_m\min_{j\ge i}(m/j)p_{(j)}\}.$$

As in BH, the reverse cumulative minimum expresses the step-up decision.
Here $m=5$ and $c_m\approx2.2833$, so the cutoffs are smaller than BH's.
This standalone example uses the same five illustrative tests.

<!-- api-example: multiple-testing-by -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

Create one p-value per hypothesis. The order identifies the five hypotheses.

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

Apply BY at the same 5% FDR level.

```python
reject_by, adjusted_by = adjust_pvalues(p, method="by", alpha=0.05, backend="numpy")
print(reject_by.tolist())
print(np.round(adjusted_by, 6))
```
<!-- example-end: multiple-testing-by -->

The mask is `[True, False, False, False, False]`; adjusted values are
`[0.011417, 0.057083, 0.114167, 0.142708, 1.0]`.
Only the first hypothesis is rejected, compared with three under BH. This is
the cost of broader dependence protection for this family, not evidence that
BY provides a more accurate effect-size estimate. Neither procedure estimates
effect sizes.

## Interpret a discovery list cautiously

Adjusted p-values are not importance rankings or probabilities that hypotheses
are true. Non-rejection does not establish that a null is true, and a discovery
may still have little practical importance.

Separate rowwise FDR control does **not** automatically control FDR over all
rows jointly. See the [family-selection example](multiple-testing.md#choose-the-hypothesis-family-with-axis).
If each row instead produces a global combined p-value and you report discoveries
across rows, those global p-values also need an appropriate multiple-testing
procedure.

For complete signatures, aliases, return shapes, axes, backend choices,
missing-value handling and numerical limitations, use
the [API guide](../../guides/multiple-testing-combine-pvalues.md).

