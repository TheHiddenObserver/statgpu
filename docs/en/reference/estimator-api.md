# Shared estimator API

> Language: English  
> Last updated: 2026-10-05  
> Switch: [Chinese](../../cn/reference/estimator-api.md)

This reference describes inherited `BaseEstimator` methods. A method appearing on an estimator does not imply that the model supplies coefficient p-values or cached training arrays. See the model's reference for overrides and available fitted results. These helpers do not refit a model, choose a scientifically valid resampling scheme, or correct variable-selection uncertainty automatically.

## Parameter management

| Method | Arguments and defaults | Return and behavior |
|---|---|---|
| `get_params(deep=True)` | `deep`: include nested estimator settings as `name__parameter` when true | Dictionary of constructor parameters, including runtime-added public parameters; excludes fitted coefficients and arbitrary attributes. |
| `set_params(**params)` | Constructor names from `get_params`; nested names use `__` | Returns `self`. Base implementation validates updates and reconstructs unfitted state; unknown names raise `ValueError`. An empty call changes nothing. Refit before prediction after an update. Model-specific overrides can differ; in particular, consult the GAM reference before reusing fitted state. |

Changing an attribute directly is not equivalent to a guaranteed `set_params`/refit lifecycle. Parameters accepted only by `fit`, such as `sample_weight`, are not constructor settings. `get_params` returns configuration, not a serializable copy of all fitted state.

## Inference helpers

The following are **estimator methods**, called as `model.method(...)`. Their signatures and return contracts differ from same-named functions in `statgpu.inference`.

### adjust_pvalues

```python
model.adjust_pvalues(pvalues=None, method="bh", alpha=0.05, axis=0, backend="auto")
```

- `pvalues`: finite probabilities in `[0, 1]`; `None` uses `model._pvalues`. If unavailable, pass values explicitly or fit with supported inference enabled; otherwise `RuntimeError`.
- `method`: `"bh"`, `"by"`, `"holm"`, `"bonferroni"`, or `"hochberg"`; supported aliases are described in [multiple testing](../guides/multiple-testing-combine-pvalues.md).
- `alpha`: finite significance level in `(0, 1)`. Validate it before calling: the current implementation can accept NaN and return all-false rejection decisions. See the [finite-alpha example](../guides/multiple-testing-combine-pvalues.md#a-complete-axis-and-weight-example).
- `axis=0`: adjust along the first axis; `None` treats every entry as one family. Choose the family deliberately, especially for multi-output coefficient arrays.
- `backend`: `"auto"`, `"numpy"`, `"cupy"`, or `"torch"`; `auto` explicitly selects CuPy/Torch when the estimator resolves to that GPU device. When it resolves to CPU, the helper currently leaves backend selection to the supplied arrays; pass `backend="numpy"` to require NumPy output. Explicit `backend="torch"` on an estimator helper requires Torch CUDA, even for a CPU estimator.

Returns a dictionary with `method`, `alpha`, `axis`, `backend`, `pvalues`, `pvalues_adjusted`, and boolean `reject`. Adjusted values and decisions have the input shape. The `backend` field records the helper's selection argument and can remain `"auto"`; it is not always the final array-library name. `pvalues` can retain the supplied list/array type when no explicit conversion occurs. This is not the `(reject, adjusted)` tuple returned by the module function. Valid marginal p-values and the selected procedure's dependence assumptions are still required.

### combine_pvalues

```python
model.combine_pvalues(pvalues=None, method="fisher", weights=None, axis=None, backend="auto")
```

`pvalues` and `backend` have the meaning above. `method` is `"fisher"`, `"cauchy"`, or `"stouffer"` (aliases in the multiple-testing guide). `weights=None` uses the method's equal-weight convention; supplied finite nonnegative weights must align with the reduction axis and have positive total weight for weighted methods. Fisher requires `weights=None`; supplying weights raises `ValueError`. Very large finite weights can overflow the raw normalization sum used by both Cauchy and Stouffer; rescale them by their positive maximum before calling, as shown in [safe weight scaling](../guides/multiple-testing-combine-pvalues.md#validate-and-rescale-combination-weights). `axis=None` flattens; an integer reduces that axis.

Returns a dictionary with `method`, `axis`, `backend`, `pvalues`, `weights`, `statistic`, and `pvalue`; the last two are scalars when flattening and otherwise have the reduced shape. The module function instead returns `(statistic, pvalue)`. Fisher/Stouffer calibration requires appropriate dependence assumptions; combining p-values does not repair invalid underlying tests.

### bootstrap_statistic

```python
model.bootstrap_statistic(
    statistic, *arrays, n_resamples=200, strategy="iid", strata=None,
    clusters=None, block_size=None, confidence_level=0.95,
    random_state=None, statistic_name="statistic", backend="auto",
)
```

| Argument | Meaning and restrictions |
|---|---|
| `statistic` | Callable receiving the aligned arrays and returning a finite scalar; it must support the selected array backend. |
| `*arrays` | One or more nonempty arrays with the same first-axis length. Pass `data`, not `(data,)`, for one array. If omitted, the method tries cached `_X_design` and `_y`; unavailable caches raise. A cached design can contain an intercept, formula columns, or square-root-weighted rows. Its response cache need not have the same weight transformation. Pass explicit arrays and weights when defining a weighted statistic; caches are not a guarantee of valid raw training pairs. |
| `n_resamples=200` | Positive integer number of resamples. More draws reduce Monte Carlo variability, not model bias. |
| `strategy="iid"` | `iid`: sample rows with replacement; `stratified`: resample within each stratum; `cluster`: resample whole clusters when all clusters have equal size (see limitation below); `block`: sample contiguous blocks. Aligned arrays use the same resampled row indices. |
| `strata=None`, `clusters=None` | One nonmissing label per row, required by `stratified` or `cluster`, respectively. Validate labels before calling; NaN labels currently can leave incomplete resamples rather than raising clearly. |
| `block_size=None` | Positive integer required by `block`; values above n are capped at n. Preserve meaningful row order. |
| `confidence_level=0.95` | Level in `(0, 1)` for percentile intervals. |
| `random_state=None` | Integer seed or no fixed seed. Reproducibility is scoped to the backend and procedure, not identical draws across all libraries. |
| `statistic_name="statistic"` | Reporting label; does not select the callable or change the statistic. |
| `backend="auto"` | Same GPU/CPU-auto distinction as above; pass `numpy` to require NumPy. Explicit CuPy/Torch requests require the respective GPU backend. |

Returns `BootstrapResult`: `observed` (the original scalar), `samples` (length `n_resamples`, backend array), `confidence_interval` (lower/upper pair), `confidence_level`, `n_resamples`, `random_state`, `statistic_name`, `strategy`, and `metadata`. `to_dict()` converts samples to a list; `to_dataframe()` requires pandas and returns `sample_index`/`statistic` columns. The original statistic is named `observed`, not `statistic`.

**Unequal-size cluster limitation.** The current `cluster` implementation stops after collecting at least n rows and truncates the final sampled cluster to n. It can split a cluster and does not implement a valid whole-cluster bootstrap for unequal group sizes. Do not use its intervals for that setting. Use a separately validated whole-cluster resampler that preserves complete groups, or use this helper only when the groups genuinely have equal size; discarding or padding observations just to equalize groups changes the problem.

For a runnable numeric-label validator and the missing-label limitation, see [label validation](../guides/inference-api.md#validate-resampling-labels-before-calling). Do not put unrelated missing group identities into an artificial common group.

The bootstrap interval is the empirical quantile pair at `(1-confidence_level)/2` and `(1+confidence_level)/2`. Exchangeability/resampling-unit assumptions remain the caller's responsibility. Callbacks may also be probed with a leading batch dimension before scalar fallback; avoid side effects and handle the intended axes explicitly if returning batched values. This generic helper is not the residual coefficient-bootstrap inference mode of ElasticNet.

### permutation_test

```python
model.permutation_test(
    statistic, X, y, n_resamples=1000, strategy="iid", strata=None,
    groups=None, alternative="two-sided", random_state=None,
    statistic_name="statistic", backend="auto",
)
```

`statistic(X, y)` must return a finite scalar. `X` and one-dimensional `y` must have the same nonzero row count. `X` stays fixed and response labels are permuted: `iid` globally, `stratified` within length-n `strata`, or `grouped` within length-n `groups`. Grouped permutation is **within groups**, not exchange of entire groups. `n_resamples` is a positive integer; `alternative` is `"two-sided"`, `"greater"`, or `"less"`. Seed, reporting label, backend, and nonmissing-label validation follow the bootstrap conventions.

Returns `PermutationTestResult`: `observed`, backend `samples` of length `n_resamples`, `pvalue`, `n_resamples`, `random_state`, `statistic_name`, `strategy`, `alternative`, and `metadata`. `to_dict()` and `to_dataframe()` follow the bootstrap conventions. The Monte Carlo tail probability uses a plus-one correction; two-sided comparison uses absolute statistic values. Choose a statistic centered at the null reference point zero, with absolute value representing extremeness; this is not an automatically centered or equal-tail test. Use a null hypothesis under which the requested response permutations are exchangeable.

## Runnable helper example

<!-- api-example: estimator-helpers -->
```python
import numpy as np
from statgpu import LinearRegression

model = LinearRegression(device="cpu", compute_inference=False)
adjusted = model.adjust_pvalues([0.01, 0.04, 0.5], method="holm")
print(adjusted["reject"].tolist())
boot = model.bootstrap_statistic(
    np.mean, np.arange(1.0, 11.0), n_resamples=99, random_state=7,
)
print(boot.observed, len(boot.samples))
```

This prints `[True, False, False]` and `5.5 99`. Explicit p-values and data make a fitted model unnecessary for these calls. There is no coefficient inference claim in this example.

## Module-function differences

The free `bootstrap_statistic` and `permutation_test` functions additionally accept `force_vectorized=False` and `statistic_hint=None`; these are not accepted by the estimator wrappers. Batched computation is available for IID and stratified resampling, block bootstrap, equal-size cluster bootstrap, and grouped permutation. On these paths, `force_vectorized=True` rejects an incompatible initial batch probe, but a later incompatible batch can still fall back to scalar calls. It does not guarantee that all resamples use vectorized execution; a vectorized callback should return one result per row for every batch size, including a final one-row batch. Unequal-size cluster bootstrap currently uses scalar calls even with that flag; do not treat it as a universal vectorization guarantee. Use `statistic_hint="mean"` only for a matching bootstrap mean of a one-dimensional array. For a matrix, omit the hint and provide a scalar statistic; the current mean fast path reduces only the last axis and can fail with a broadcast error. Use `"pearson_corr"` for a matching permutation correlation with X shaped `(n,)` or `(n, 1)`. Hints choose built-in resample calculations and do not verify that the supplied observed-statistic callable is equivalent. Free functions infer `backend="auto"` from their arrays and require explicit data. See [the module guide](../guides/inference-api.md) for imports and examples.
