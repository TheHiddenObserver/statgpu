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
- `alpha`: significance level in `(0, 1)`.
- `axis=0`: adjust along the first axis; `None` treats every entry as one family. Choose the family deliberately, especially for multi-output coefficient arrays.
- `backend`: `"auto"`, `"numpy"`, `"cupy"`, or `"torch"`; `auto` follows the estimator's resolved device, rather than simply inferring from the supplied p-value array.

Returns a dictionary with `method`, `alpha`, `axis`, `backend`, `pvalues`, `pvalues_adjusted`, and boolean `reject`. Adjusted values and decisions have the input shape. This is not the `(reject, adjusted)` tuple returned by the module function. Valid marginal p-values and the selected procedure's dependence assumptions are still required.

### combine_pvalues

```python
model.combine_pvalues(pvalues=None, method="fisher", weights=None, axis=None, backend="auto")
```

`pvalues` and `backend` have the meaning above. `method` is `"fisher"`, `"cauchy"`, or `"stouffer"` (aliases in the multiple-testing guide). `weights=None` uses the method's equal-weight convention; supplied finite nonnegative weights must align with the reduction axis and have positive total weight for weighted methods. Fisher requires `weights=None`; supplying weights raises `ValueError`. `axis=None` flattens; an integer reduces that axis.

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
| `*arrays` | One or more nonempty arrays with the same first-axis length. Pass `data`, not `(data,)`, for one array. If omitted, the method tries cached `_X_design` and `_y`; unavailable caches raise. A cached design can contain an intercept or transformed columns, so explicit arrays are usually clearer. |
| `n_resamples=200` | Positive integer number of resamples. More draws reduce Monte Carlo variability, not model bias. |
| `strategy="iid"` | `iid`: sample rows with replacement; `stratified`: resample within each stratum; `cluster`: resample clusters together; `block`: sample contiguous blocks. Aligned arrays use the same resampled row indices. |
| `strata=None`, `clusters=None` | Length-n labels, required by `stratified` or `cluster`, respectively. |
| `block_size=None` | Positive integer required by `block`; values above n are capped at n. Preserve meaningful row order. |
| `confidence_level=0.95` | Level in `(0, 1)` for percentile intervals. |
| `random_state=None` | Integer seed or no fixed seed. Reproducibility is scoped to the backend and procedure, not identical draws across all libraries. |
| `statistic_name="statistic"` | Reporting label; does not select the callable or change the statistic. |
| `backend="auto"` | Same estimator-device selection as above; explicit NumPy/CuPy/Torch is also accepted. |

Returns `BootstrapResult`: `observed` (the original scalar), `samples` (length `n_resamples`, backend array), `confidence_interval` (lower/upper pair), `confidence_level`, `n_resamples`, `random_state`, `statistic_name`, `strategy`, and `metadata`. `to_dict()` converts samples to a list; `to_dataframe()` requires pandas and returns `sample_index`/`statistic` columns. The original statistic is named `observed`, not `statistic`.

The bootstrap interval is the empirical quantile pair at `(1-confidence_level)/2` and `(1+confidence_level)/2`. Exchangeability/resampling-unit assumptions remain the caller's responsibility. This generic helper is not the residual coefficient-bootstrap inference mode of ElasticNet.

### permutation_test

```python
model.permutation_test(
    statistic, X, y, n_resamples=1000, strategy="iid", strata=None,
    groups=None, alternative="two-sided", random_state=None,
    statistic_name="statistic", backend="auto",
)
```

`statistic(X, y)` must return a finite scalar. `X` and one-dimensional `y` must have the same nonzero row count. `X` stays fixed and response labels are permuted: `iid` globally, `stratified` within length-n `strata`, or `grouped` within length-n `groups`. Grouped permutation is **within groups**, not exchange of entire groups. `n_resamples` is a positive integer; `alternative` is `"two-sided"`, `"greater"`, or `"less"`. Seed, reporting label, and backend follow the bootstrap conventions.

Returns `PermutationTestResult`: `observed`, backend `samples` of length `n_resamples`, `pvalue`, `n_resamples`, `random_state`, `statistic_name`, `strategy`, `alternative`, and `metadata`. `to_dict()` and `to_dataframe()` follow the bootstrap conventions. The Monte Carlo tail probability uses a plus-one correction; two-sided comparison uses absolute statistic values. Use a null hypothesis under which the requested response permutations are exchangeable.

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

The free `bootstrap_statistic` and `permutation_test` functions additionally accept `force_vectorized=False` and `statistic_hint=None`; these are not accepted by the estimator wrappers. `force_vectorized=True` requires a compatible IID batched computation. Supported hints are `"mean"` and `"pearson_corr"`, with their matching statistic/procedure; do not use a hint for an unrelated callable. Free functions infer `backend="auto"` from their arrays and require explicit data. See [the module guide](../guides/inference-api.md) for imports and examples.
