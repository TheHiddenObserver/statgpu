# P-value adjustment and combination API

> Language: English  
> Last updated: 2026-10-05  
> Switch: [Chinese](../../cn/guides/multiple-testing-combine-pvalues.md)

Import these functions from `statgpu.inference` or `statgpu`. For choosing the hypothesis family, FWER versus FDR, formulas and dependence assumptions, start with [multiple testing](../models/multiple-testing.md).

## Complete signatures

```text
adjust_pvalues(pvalues, method="bh", alpha=0.05, axis=None, backend="auto")
multipletests(pvalues, alpha=0.05, method="bh", axis=None, backend="auto")
combine_pvalues(pvalues, method="fisher", weights=None, axis=None, backend="auto")
```

`multipletests` delegates to `adjust_pvalues`, but their second and third positional arguments are in different orders. Pass `method=` and `alpha=` by keyword. Unlike statsmodels, this alias returns only two values and has no `is_sorted` or `returnsorted` argument.

## Inputs and returns

| Argument | Meaning and restrictions |
|---|---|
| `pvalues` | Numeric finite probabilities in `[0,1]`; preserve the hypothesis order. |
| `method` | Adjustment or combination method listed below. Names ignore surrounding whitespace and case. |
| `alpha=0.05` | Finite level in `(0,1)` for the rejection mask. Currently NaN is not rejected and returns all-false decisions; validate computed levels first. It does not affect adjusted values. |
| `axis=None` | Treat all entries as one family/global test. An integer operates separately along that axis, including negative axis indices. |
| `weights=None` | Combination only. Cauchy/Stouffer use equal weights by default. Supply a finite nonnegative vector matching the combination-axis length, with positive sum. One vector is reused across batches; weights are normalized internally. Fisher rejects supplied weights. |
| `backend="auto"` | Infer `numpy`, `cupy`, or `torch` from p-value arrays; ordinary lists use NumPy. This is array-library selection, not estimator `device` selection. |

`adjust_pvalues` and `multipletests` return `(reject, adjusted)`, both with the input shape, including for `axis=None`. `reject` is boolean and `adjusted` is float64. A scalar accepts only `axis=None`. A flattened empty adjustment returns empty arrays; for batch processing use a nonempty family axis.

`combine_pvalues` returns `(statistic, pvalue)`: scalar/zero-dimensional outputs for `axis=None`, otherwise arrays with the chosen axis removed. Empty combination families raise. Outputs are float64 arrays/scalars of the selected backend. Explicit Torch library use may operate on Torch CPU or CUDA tensors; it is not the estimator-level strict-CUDA `device="torch"` request.

Estimator methods with these names return dictionaries and use a different default adjustment axis; see [shared estimator helpers](../reference/estimator-api.md).

## Methods and aliases

| Canonical method | Accepted aliases |
|---|---|
| `bh` | `fdr_bh`, `benjamini-hochberg`, `benjamini_hochberg` |
| `by` | `fdr_by`, `benjamini-yekutieli`, `benjamini_yekutieli` |
| `holm` | `holm-bonferroni`, `holm_bonferroni` |
| `bonferroni` | `bonf` |
| `hochberg` | `fdr_hochberg`, `step_up`, `stepup` |
| `fisher` | `fisher-combination`, `fisher_combination` |
| `cauchy` | `cauchy-combination`, `cauchy_combination`, `acat` |
| `stouffer` | `z-test`, `ztest`, `weighted_z` |

Despite its legacy name, `fdr_hochberg` invokes **FWER Hochberg**, not FDR BH. There is no Tippett method in this implementation. Bonferroni/Holm allow arbitrary dependence, BH requires independence or suitable positive dependence, BY allows arbitrary dependence, and Hochberg requires independence or an applicable Simes dependence condition. All require valid marginal p-values.

Fisher uses the independent-uniform chi-square reference. Stouffer uses the independent-normal-score variance; it does not estimate covariance between tests. Cauchy/ACAT uses a normalized-weight tangent statistic and Cauchy tail approximation under regularity conditions. It does not give a universal finite-level guarantee for arbitrary p-value dependence. See the [formulas and references](../models/multiple-testing.md#combination-formulas-and-assumptions).

## A complete axis and weight example

The Stouffer calculation assumes independent, consistently oriented null scores. Each row here is a separate global test; this does not additionally adjust the two resulting global p-values.

<!-- api-example: multiple-testing-axis -->
```python
import numpy as np
from statgpu import adjust_pvalues, combine_pvalues, multipletests

p = np.array([[0.01, 0.04, 0.6], [0.02, 0.2, 0.7]])
alpha = 0.05
if not np.isfinite(alpha) or not 0 < alpha < 1:
    raise ValueError("alpha must be finite and in (0, 1)")
reject, adjusted = adjust_pvalues(p, method="holm", alpha=alpha, axis=1, backend="numpy")
reject_alias, adjusted_alias = multipletests(p, method="holm", alpha=alpha, axis=1, backend="numpy")
np.testing.assert_array_equal(reject, reject_alias)
np.testing.assert_allclose(adjusted, adjusted_alias)
statistic, global_p = combine_pvalues(p, method="stouffer", weights=[1, 2, 1], axis=1, backend="numpy")
assert adjusted.shape == p.shape
assert statistic.shape == global_p.shape == (2,)
```

## Numerical limits

- Fisher clips probabilities below the smallest positive normal float64 value before taking logs. Exact zero therefore yields a finite statistic rather than infinity.
- Cauchy and Stouffer clip probabilities to `[eps, 1-eps]`, where `eps` is float64 machine precision. Exact endpoints and more extreme probabilities are altered.
- Fisher and Stouffer also use the [distribution functions](distribution-api.md), whose survival/quantile tails can suffer cancellation or saturation. Cauchy's direct tail subtraction can lose very small probabilities too.
- A returned zero or one can be a numerical limit. Use a validated tail-stable method when extreme-tail resolution matters; converting the same values to GPU does not remove these limitations.

Do not infer method calibration from finite outputs or agreement on a few examples. For individual tests use the adjustment's error criterion; for combination interpret one global null.

## Validate and rescale combination weights

Cauchy and Stouffer depend on relative weights, so multiplying every weight by
the same positive constant should leave the result unchanged. Currently, the
sum of individually finite weights can overflow: with `p=[0.01, 0.1]` and
`weights=[1e308, 1e308]`, Cauchy returns `0.5` instead of about `0.018222`,
and Stouffer returns NaN. Validate weights and divide them by their largest
value before calling. This preserves their ratios while avoiding that overflow;
it does not fix extreme probability tails or change dependence assumptions.

<!-- safety-example: scaled-combination-weights -->
```python
import numpy as np
from statgpu.inference import combine_pvalues


def relative_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty vector")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    largest = weights.max()
    if largest <= 0:
        raise ValueError("at least one weight must be positive")
    return weights / largest


p = np.array([0.01, 0.1])
weights = relative_weights([1e308, 1e308])
for method in ("cauchy", "stouffer"):
    statistic, combined = combine_pvalues(
        p, method=method, weights=weights, backend="numpy",
    )
    print(method, round(float(statistic), 6), round(float(combined), 6))
```

This prints `cauchy 17.4491 0.018222` and `stouffer 2.55117 0.005368`,
the same results as equal unit weights. Keep the weight vector aligned with
the p-values; the API checks its length against the combination axis.

## Historical timing records

Earlier timing tables are retained in the [developer historical record](../../../dev/references/multiple-testing-historical-benchmarks.md). They do not establish a current speedup or a universal CPU/GPU crossover; measure the actual workload when performance matters.
