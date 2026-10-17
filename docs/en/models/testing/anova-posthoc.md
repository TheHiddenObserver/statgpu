# Post-hoc comparisons: Tukey and Bonferroni Welch tests

> Language: English\
> Last updated: 2026-10-12\
> Switch: [Chinese](../../../cn/models/testing/anova-posthoc.md)

<a id="the-question"></a>

## From “some means differ” to “which means differ?”

An [omnibus ANOVA](anova.md) tests whether all group means are equal.
It does not identify particular differences. Pairwise comparisons ask a more
specific question, $H_{0,ij}:\mu_i-\mu_j=0$, and estimate each difference in
the response's original units.

Testing every pair creates a family of $m=k(k-1)/2$ comparisons for $k$ groups.
Both functions on this page account for that all-pairs family. A significant
omnibus test does not make unadjusted pairwise tests safe. Conversely, these
functions do not require an ANOVA result as input: a prespecified simultaneous
pairwise analysis can be run directly.

Choose the groups, comparison family and significance level before examining
results. See [multiple testing](multiple-testing.md) for the broader distinction
between family-wise error and false discovery rate, and for other correction
methods. Here the focus is how to obtain and interpret mean comparisons.

<a id="choose-the-comparison"></a>

## Match the comparison to the variance assumptions

| Question or assumption | Use | What the result means |
|---|---|---|
| All pairs, common population variance is reasonable | `tukey_hsd` | Pooled-variance Tukey HSD; Tukey–Kramer scaling for unequal sample sizes |
| All pairs, population variances may differ | `bonferroni` | Separate Welch t tests with a Bonferroni family-wise threshold |
| A selected set of contrasts, dependence, or repeated measurements | A method designed for that question | Neither function accepts a contrast matrix, paired samples or repeated-measures structure |

Both procedures assume independent observations across and within groups and
use normal-error models. Welch accommodates unequal variances, but does not
make outliers or dependence harmless. Its t reference is approximate, so
Bonferroni's error control is only as well calibrated as the underlying tests.
Tukey's all-pairs intervals have the nominal joint coverage in the balanced
normal equal-variance model; the unequal-size extension is conservative under
those assumptions. See [NIST's Tukey explanation](https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm).

In particular, following unequal-variance Welch ANOVA with pooled-variance
Tukey comparisons changes the variance assumption. Use a compatible pairwise
method. `tukey_hsd` here has no `equal_var=False` switch and does not implement
Games–Howell comparisons.

<a id="read-comparisons"></a>

## Read direction, uncertainty and decisions together

Each returned `PairwiseComparison` contains zero-based `group_i` and `group_j`
with `i < j`, following input order. The difference is always

$$\texttt{mean\_diff}=\bar y_i-\bar y_j.$$

A negative value means the earlier input group has the smaller observed mean.
`ci_lower` and `ci_upper` use this same direction. Reversing the comparison
reverses the estimate's sign and changes an interval $[L,U]$ into $[-U,-L]$.

- An interval excluding zero supports a difference at the corresponding
  simultaneous comparison level, subject to numerical boundary rounding.
- An interval containing zero indicates insufficient evidence for that pair;
  it does not establish equality or practical equivalence.
- A simultaneous 95% interval set describes repeated-sampling coverage of
  all differences in the family together. It does not assign a 95% probability
  to an already fixed parameter after observing these data.

**The `pvalue` field has different semantics across these two functions:**

| Function | Returned `comparison.pvalue` | Returned `comparison.reject` |
|---|---|---|
| `tukey_hsd` | Already adjusted for the all-pairs family | `pvalue < alpha` |
| `bonferroni` | Raw, two-sided Welch t-test p-value | `pvalue < alpha / m` |

This distinction matters when exporting a report. Do not label Bonferroni's
returned p-values as adjusted, compare them directly to `alpha`, or adjust
Tukey's p-values a second time for the same family.

<a id="tukey-method"></a>

## Tukey: a common noise estimate and the studentized range

Let $MSE=SS_W/(N-k)$ be the pooled within-group mean square. For a pair with
sample sizes $n_i,n_j$, the studentized-range scale is

$$s_{q,ij}=\sqrt{\frac{MSE}{2}\left(\frac1{n_i}+\frac1{n_j}\right)},\qquad
q_{ij}=\frac{|\bar y_i-\bar y_j|}{s_{q,ij}}.$$

The adjusted p-value is the upper tail of the studentized-range distribution
with $k$ groups and $N-k$ degrees of freedom. The simultaneous interval is

$$(\bar y_i-\bar y_j)\;\pm\;q_{1-\alpha;k,N-k}\,s_{q,ij}.$$

For equal sizes this is Tukey HSD; the pair-specific scale handles unequal
sizes using the Tukey–Kramer extension. It still uses a common variance
estimate. Notice that the range distribution uses the number of groups $k$,
not the number of pairs $m$.

<a id="anova-tukey"></a>

## A complete Tukey CPU example

Start a fresh session and run these three blocks in order. The artificial
samples share a population standard deviation of one.

<!-- example: anova-tukey -->
```python
import numpy as np
from statgpu.anova import tukey_hsd
```

Generate three independent groups of 30 observations. Group indices in the
output refer to this list order; keep your real study's labels alongside it.

```python
rng = np.random.default_rng(42)
groups = [rng.normal(mean, 1.0, 30) for mean in (0.0, 0.2, 0.9)]
```

Set a 5% family-wise level and print all three comparisons, including the
non-rejected pair. The reported p-values are already adjusted.

```python
posthoc = tukey_hsd(*groups, alpha=0.05, backend="numpy")
for c in posthoc.comparisons:
    print(f"{c.group_i}-{c.group_j}: diff={c.mean_diff:.3f}, "
          f"adjusted_p={c.pvalue:.3g}, "
          f"CI=[{c.ci_lower:.3f}, {c.ci_upper:.3f}], reject={c.reject}")
```
<!-- example-end: anova-tukey -->

The output is approximately:

| Pair | Mean difference | Adjusted p | Simultaneous interval | Reject |
|---|---:|---:|---|---|
| 0 − 1 | -0.297 | 0.288 | [-0.764, 0.170] | False |
| 0 − 2 | -0.782 | 0.000394 | [-1.248, -0.315] | True |
| 1 − 2 | -0.485 | 0.0399 | [-0.951, -0.018] | True |

Group 2 has a higher observed mean than groups 0 and 1, with simultaneous
intervals excluding zero. The interval for 0 − 1 includes zero, so that pair
is not rejected at the stated family level. The estimates are in response
units; the p-values do not measure practical importance. `posthoc.df_within`
is `87` and `posthoc.mse` is approximately `0.574286`.

<a id="bonferroni-method"></a>

## Bonferroni: separate Welch uncertainty for each pair

For unequal variances, this function uses

$$SE_{ij}=\sqrt{s_i^2/n_i+s_j^2/n_j},\qquad
t_{ij}=\frac{\bar y_i-\bar y_j}{SE_{ij}},$$

$$\nu_{ij}=\frac{(s_i^2/n_i+s_j^2/n_j)^2}
{(s_i^2/n_i)^2/(n_i-1)+(s_j^2/n_j)^2/(n_j-1)}.$$

It calculates a raw two-sided t p-value for each pair, then rejects only if
$p_{ij}<\alpha/m$. Its intervals use the stricter two-sided critical value

$$(\bar y_i-\bar y_j)\;\pm\;t_{1-\alpha/(2m),\nu_{ij}}\,SE_{ij}.$$

Thus the correction appears in the **decision threshold and confidence
interval**, even though the stored p-value remains raw. If adjusted values are
needed for display, calculate $\min(1,m p_{ij})$ for the same complete family.
[NIST's Bonferroni explanation](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)
describes the simultaneous-coverage principle; this API specifically uses
pairwise Welch uncertainty rather than a pooled error variance.

<a id="anova-bonferroni"></a>

## A complete unequal-variance pairwise example

This illustration is independent of the Tukey example and of the
[Welch ANOVA example](anova-welch.md#anova-welch). It deliberately includes a
raw p-value below 0.05 that does not pass the corrected threshold.

<!-- example: anova-bonferroni -->
```python
import numpy as np
from statgpu.anova import bonferroni
```

Create three groups with different sample sizes and standard deviations.
The middle group's population mean is `10.25` in this illustration.

```python
rng = np.random.default_rng(2026)
unequal_groups = [
    rng.normal(10.0, 0.7, 20),
    rng.normal(10.25, 1.5, 28),
    rng.normal(13.0, 3.0, 36),
]
```

The family contains three pairs, so the raw-p cutoff is `0.05 / 3`, about
`0.01667`. Print the function's actual rejection decision.

```python
bonf = bonferroni(*unequal_groups, alpha=0.05, backend="numpy")
print(f"comparisons={bonf.n_comparisons}, "
      f"cutoff={bonf.alpha / bonf.n_comparisons:.5f}")
for c in bonf.comparisons:
    print(f"{c.group_i}-{c.group_j}: diff={c.mean_diff:.3f}, "
          f"raw_p={c.pvalue:.3g}, "
          f"CI=[{c.ci_lower:.3f}, {c.ci_upper:.3f}], reject={c.reject}")
```

The comparisons are approximately:

| Pair | Mean difference | Raw p | Bonferroni interval | Reject |
|---|---:|---:|---|---|
| 0 − 1 | -0.697 | 0.0220 | [-1.428, 0.034] | False |
| 0 − 2 | -3.551 | 6.38e-07 | [-5.041, -2.061] | True |
| 1 − 2 | -2.854 | 5.04e-05 | [-4.445, -1.263] | True |

The first raw p-value is below 0.05, but above 0.01667, so `reject=False`.
Its simultaneous interval includes zero. For a report that expects adjusted
p-values, derive separate display values without changing the result object.

```python
adjusted_pvalues = np.minimum(
    1.0, bonf.n_comparisons * np.array([c.pvalue for c in bonf.comparisons])
)
print([f"{p:.3g}" for p in adjusted_pvalues])
```
<!-- example-end: anova-bonferroni -->

The adjusted display values are `['0.066', '1.91e-06', '0.000151']`.
Comparing these with 0.05 gives the same `[False, True, True]` decisions.
Never shrink $m$ after dropping inconvenient or non-significant comparisons.

<a id="parameters-and-pitfalls"></a>

## Parameters and common mistakes

- Both functions take separate `*groups`, `alpha=0.05`, `backend="auto"` and
  `dtype=None`. `alpha` must be finite and strictly between zero and one.
  It is the level for the full pairwise family, not a per-pair level.
- Each function requires at least two groups and at least two finite
  observations per group. Unequal group sizes are allowed. NaN and infinite
  observations raise `ValueError`; there is no automatic missing-value removal.
- Inputs are flattened. Supply one vector per group, rather than expecting a
  matrix to describe many tests or paired observations.
- Both functions compare **all** input-group pairs. There is no custom-pair or
  control-only selector. Many outcomes or separately analyzed subgroups may
  create a larger family than one call controls.
- Equal population variances remain required for Tukey, even with unequal
  sample sizes. Bonferroni correction alone cannot repair an inappropriate
  original test or a data-selected family.
- Zero standard errors yield degenerate intervals, with raw Bonferroni p-values
  of one for an equal-mean pair and zero for a different-mean pair. For Tukey,
  zero pooled MSE produces analogous equal/different-mean handling. Do not
  interpret constant observations as proving population effects without error.
- The group reductions support NumPy, CuPy and Torch. Tukey's studentized-range
  CDF and quantile use SciPy on CPU; Bonferroni's scalar t-distribution
  calculations also use CPU. GPU group reductions do not imply that every
  step of either procedure runs on GPU.

For the complete `TukeyResult`, `PosthocResult` and `PairwiseComparison` fields,
validation rules and execution options, see the [ANOVA API reference](../../reference/anova-api.md).
The [multiple-testing tutorial](multiple-testing.md) explains general error
criteria and correction choices without tying them to these two ANOVA helpers.

<a id="references"></a>

## References and next steps

- [NIST: Tukey's method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm).
- [NIST: Bonferroni's method](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm).
- [SciPy Tukey HSD reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html),
  including the original Tukey and Kramer publications. Match variance assumptions
  when comparing implementations; SciPy's API options differ from statgpu's.
- [Classical one-way ANOVA](anova-oneway.md), [Welch ANOVA](anova-welch.md),
  [balanced two-way ANOVA](anova-twoway.md) and the [ANOVA overview](anova.md).
