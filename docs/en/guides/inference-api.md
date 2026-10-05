# Inference API Reference

> Language: English  
> Last updated: 2026-10-05  
> Module: `statgpu.inference`  
> Switch: [Chinese](../../cn/guides/inference-api.md)

`statgpu.inference` collects reusable statistical utilities for probability distributions, multiple testing, p-value combination, permutation tests, and bootstrap calculations.

This page is the **module entry point**. Detailed distribution behavior is documented once in [Distribution API](distribution-api.md), and model coefficient-inference choices are documented in [Inference Modes](inference-modes.md).

## Quick reference

```python
from statgpu.inference import (
    norm,
    t,
    adjust_pvalues,
    combine_pvalues,
    permutation_test,
    bootstrap_statistic,
)
```

| API | Purpose | Detailed documentation |
|---|---|---|
| distribution objects such as `norm`, `t`, `chi2`, `poisson` | CDF/SF/PPF/PDF/PMF/random sampling | [Distribution API](distribution-api.md) |
| `get_distribution(...)` | choose a distribution/backend dynamically | [Distribution API](distribution-api.md) |
| `adjust_pvalues(...)` | multiple-testing adjustment | [Multiple Testing](multiple-testing-combine-pvalues.md) |
| `combine_pvalues(...)` | combine evidence across p-values | [Multiple Testing](multiple-testing-combine-pvalues.md) |
| `permutation_test(...)` | permutation-based hypothesis test | this page |
| `bootstrap_statistic(...)` | generic bootstrap for a user-supplied statistic | this page |

## Distribution functions

Distribution objects expose scipy-style methods such as `cdf`, `sf`, `ppf`, `isf`, `pdf`/`pmf`, and `rvs`.

```python
from statgpu.inference import norm, t

p = norm.cdf(1.96)
q = t.ppf(0.975, df=10)
```

Backend selection, supported distributions, inverse-function precision, R-style compatibility aliases, and legacy names are all documented in [Distribution API](distribution-api.md). They are intentionally not duplicated here.

## Multiple testing and p-value combination

```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.5])

reject, pvals_bh = adjust_pvalues(pvals, method="bh")
stat, p_global = combine_pvalues(pvals, method="fisher")
```

Available adjustment/combination methods and their interpretation are described in [Multiple Testing](multiple-testing-combine-pvalues.md).

## Permutation testing

`permutation_test` repeatedly permutes the supplied data according to its API and recomputes a user-provided statistic to form a permutation reference distribution.

<!-- api-example: module-permutation -->
```python
import numpy as np
from statgpu.inference import permutation_test

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = X @ np.ones(5) + rng.standard_normal(100)

result = permutation_test(
    lambda X_, y_: np.corrcoef(X_[:, 0], y_)[0, 1],
    X,
    y,
    n_resamples=999,
    random_state=42,
)

print(result.pvalue)
```

Choose the statistic and permutation scheme to match the null hypothesis of the application; a generic permutation engine cannot determine exchangeability assumptions on the user's behalf.

## Generic bootstrap

`bootstrap_statistic` bootstraps a statistic supplied by the caller.

<!-- api-example: module-bootstrap -->
```python
import numpy as np
from statgpu.inference import bootstrap_statistic

rng = np.random.default_rng(42)
data = rng.standard_normal(1000)

result = bootstrap_statistic(
    np.mean,
    data,
    n_resamples=9999,
    random_state=42,
)

print(result.observed)
print(result.confidence_interval)
```

This generic utility is different from estimator-specific inference modes such as the penalized Gaussian residual bootstrap. For coefficient inference after a fitted model, start with [Inference Modes](inference-modes.md) instead of assuming the generic bootstrap reproduces a model-specific inferential procedure.

## API map

Use the documentation according to the question you are trying to answer:

- **How do I evaluate a probability distribution on NumPy/CuPy/Torch?** → [Distribution API](distribution-api.md)
- **How do I adjust or combine many p-values?** → [Multiple Testing](multiple-testing-combine-pvalues.md)
- **How do I run a generic permutation or bootstrap calculation?** → this page
- **Which inference method should a regression estimator use?** → [Inference Modes](inference-modes.md)
- **What does penalized-GLM coefficient inference target?** → [Penalized GLM inference](penalized-glm-inference.md)


## Estimator wrappers

The [shared estimator API](../reference/estimator-api.md) documents every argument and return field for the inherited helper methods. It also explains differences from these free functions, including dictionary versus tuple p-value results and supplementary vectorization controls.

## Resampling function signatures

The free functions take explicit data and infer `backend="auto"` from those arrays. They return `BootstrapResult` or `PermutationTestResult`, not an estimator and not a tuple. The shared reference defines the [resampling arguments and result fields](../reference/estimator-api.md#bootstrap_statistic); the two extra controls below are available only on free functions.

```python
bootstrap_statistic(
    statistic, *arrays, n_resamples=200, strategy="iid", strata=None,
    clusters=None, block_size=None, confidence_level=0.95,
    random_state=None, statistic_name="statistic", backend="auto",
    force_vectorized=False, statistic_hint=None,
)
permutation_test(
    statistic, X, y, n_resamples=1000, strategy="iid", strata=None,
    groups=None, alternative="two-sided", random_state=None,
    statistic_name="statistic", backend="auto", force_vectorized=False,
    statistic_hint=None,
)
```

- On batched paths, `force_vectorized=True` rejects an incompatible initial callback probe; later incompatible batches can still fall back to scalar calls. Handle every batch size, including a final one-row batch, and do not treat the flag as a no-fallback guarantee. Callbacks should have no side effects because they may be probed before scalar execution.
- `statistic_hint="mean"` accelerates the bootstrap mean of one one-dimensional array. Omit the hint for matrix data. `"pearson_corr"` accelerates permutation correlation for X shaped `(n,)` or `(n, 1)`. The observed callback must compute the same statistic; hints do not verify equivalence.
- Cluster bootstrap currently preserves whole groups only for equal-size groups. Unequal-size groups can be truncated, so do not use those intervals as whole-cluster inference. Grouped permutation instead permutes responses within groups and does not have that truncation behavior.
- `observed` and every resampled statistic must be finite. Percentile intervals describe the supplied statistic under the chosen resampling scheme; increasing `n_resamples` does not validate that scheme or remove estimator bias.

For example, keep matrix rows together while bootstrapping their overall mean by leaving the fast-path hint unset:

<!-- api-example: module-matrix-mean -->
```python
import numpy as np
from statgpu.inference import bootstrap_statistic

rows = np.arange(12.0).reshape(6, 2)
result = bootstrap_statistic(
    np.mean, rows, n_resamples=99, random_state=7, backend="numpy",
)
print(result.observed, result.samples.shape)
```

This prints `5.5 (99,)`. Each resample selects entire rows with replacement, preserving the two measurements in a row as a pair.

## Validate resampling labels before calling

Supply one nonmissing label for every row whenever using `strata`, `clusters`,
or `groups`. Currently, NaN labels are not rejected consistently: equality-based
grouping omits those rows and may leave uninitialized batch entries. A call can
therefore return invalid finite statistics or fail with an indexing error.
A finite p-value or interval is not proof that the labels were valid.

For a numeric-code workflow, validate labels explicitly before resampling:

<!-- safety-example: validated-resampling-labels -->
```python
import numpy as np
from statgpu.inference import permutation_test


def validated_labels(labels, n):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("Provide one group label per observation")
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all():
        raise ValueError("Group labels must be finite numeric codes")
    return labels


X = np.arange(6.0)
y = np.array([0.2, 1.3, 0.7, 2.5, 3.0, 2.8])
groups = validated_labels([0, 0, 0, 1, 1, 1], len(y))
result = permutation_test(
    lambda X_, y_: np.corrcoef(X_, y_)[0, 1], X, y,
    strategy="grouped", groups=groups, n_resamples=99,
    random_state=7, backend="numpy",
)
print(result.observed, result.pvalue)
```

The correlation is about `0.903`, with a permutation p-value between 0 and 1.
The test assumes responses are exchangeable **within each specified group**
under the null. The helper rejects NaN/infinite codes and incorrect label
shapes; it does not decide whether the groups make statistical sense. Resolve
missing group identities from the data source or choose an explicit missing-data
policy. Do not assign all unknown identities to one artificial group merely to
make the computation run. The same validation applies to bootstrap strata and
clusters; unequal-size cluster limitations still apply.

For `alternative="two-sided"`, the engine counts `abs(resampled) >= abs(observed)`
with the plus-one correction. Use a statistic whose null reference point is zero
and whose absolute value represents extremeness, such as correlation under a
zero-correlation null. The engine does not automatically center a statistic or
construct an equal-tail test for an asymmetric null distribution. For a
prespecified directional alternative, use `"greater"` or `"less"` as appropriate.
