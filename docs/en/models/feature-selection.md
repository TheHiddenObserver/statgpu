# Feature Selection

> Language: English  
> Last updated: 2026-10-03  
> Switch: [Chinese](../../cn/models/feature-selection.md)

## Which predictors should stay?

When several measured variables might explain an outcome, a smaller model can
be easier to inspect and use. `StepwiseSelector` repeatedly adds or removes
one predictor and keeps a change when an information criterion improves. It
searches for a useful subset, not for proof that the retained variables cause
the outcome.

Use this approach for a moderate number of candidate predictors and a model
with comparable likelihood-based scores. For predictive shrinkage with many
correlated predictors, consider [Ridge](ridge.md) or [Lasso](lasso.md). If the
primary goal is false discovery rate (FDR) control, see
[knockoff methods](knockoff.md), including their design/distribution assumptions.
AIC/BIC stepwise selection itself does **not** control FDR.

## What the search optimizes

For likelihood-based models, the criteria balance fit and model size:

$$
\mathrm{AIC}=-2\ell+2k,\qquad
\mathrm{BIC}=-2\ell+k\log n.
$$

Here $\ell$ is the fitted log-likelihood, $n$ the observation count, and $k$ the
model's parameter count (OLS uses fitted design rank, including the intercept).
Lower is better **among comparable models fitted to the same outcomes and
observations**. The values are not accuracy percentages. BIC usually penalizes
additional variables more strongly than AIC at ordinary sample sizes.

Forward search starts with no predictors and tries additions. Backward search
starts with all predictors and tries removals. `direction="both"` starts empty
and considers additions and removals at each step. This is a greedy local
search; it does not enumerate all subsets or guarantee the global minimum.

## A complete CPU example

This example creates six candidate predictors, only two of which generate the
outcome. Selection sees the training rows only; the test rows stay untouched
until evaluation.

```python
import numpy as np
from statgpu import LinearRegression, StepwiseSelector

rng = np.random.default_rng(42)
X = rng.normal(size=(240, 6))
y = 1.5 + 3.0 * X[:, 0] - 2.0 * X[:, 2] + rng.normal(scale=0.5, size=240)
X_train, X_test = X[:180], X[180:]
y_train, y_test = y[:180], y[180:]

selector = StepwiseSelector(
    LinearRegression,
    criterion="bic",
    direction="both",
    max_features=3,
    device="cpu",
    compute_inference=False,
).fit(X_train, y_train)

X_selected = selector.transform(X_test)
prediction = selector.predict(X_test)  # pass the original six columns
print("Selected columns:", selector.selected_features_)
print("Selected test shape:", X_selected.shape)
print("Slopes:", np.round(selector.best_model_.coef_, 3))
print("Test R2:", round(selector.score(X_test, y_test), 3))
print("BIC path:", np.round(selector.bic_history_, 3))
```

Expected rounded output:

```text
Selected columns: [0, 2]
Selected test shape: (60, 2)
Slopes: [ 2.982 -2.049]
Test R2: 0.981
BIC path: [993.049 768.053 286.428]
```

The selected indices are zero-based positions in the original matrix. The two
slopes correspond to columns 0 and 2 in that order and are close to the
simulation's true slopes, 3 and -2. The history includes the initial
intercept-only model and each accepted change, not every attempted subset.
The falling BIC shows why this search accepted the additions. This particular
simulation recovers the true subset; other samples, correlated predictors, or
weaker signals need not do so. Test $R^2$ evaluates predictions; it does not
validate each selected variable as a discovery.

## Choosing parameters

| Parameter | Default | Guidance |
|---|---|---|
| `model_class` | Required | Pass an estimator class/callable, such as `LinearRegression`, not a fitted instance. Each candidate is fitted afresh. |
| `criterion` | `"aic"` | `"aic"` or `"bic"`; BIC favors smaller models more strongly for typical sample sizes. |
| `direction` | `"both"` | `"forward"`, `"backward"`, or `"both"`; starting points and paths can lead to different subsets. |
| `max_features` | `None` | Upper bound, not a requested exact count. An integer from 0 through the input width; `None` uses the full width as the cap. |
| `n_jobs` | `None` | Candidate-scoring threads; `None`/`1` is sequential and `-1` follows joblib's all-worker convention. Start sequentially for a small example or constrained GPU memory. |
| `verbose` | `False` | Print accepted search steps. |
| `**model_kwargs` | — | Forwarded to the wrapped model constructor, e.g. `device="cpu"`, `compute_inference=False`. Backend and inference support follow that model. |

For backward selection, the cap is enforced by removals before requiring
criterion improvement; the criterion can temporarily worsen while meeting
that cap. An intercept-only/null model may win in any direction. Setting
`max_features=0` is permitted.

## Inputs, methods, and results

`fit(X, y)` expects finite numeric `X` of shape `(n_samples, n_features)` and
single-output `y` of shape `(n_samples,)` (a one-column array is flattened).
Use an array/design matrix rather than passing a raw DataFrame with names or
categorical values. The selector's `fit` has no `sample_weight`, `formula`, or
`data` arguments.

| API | Result |
|---|---|
| `fit(X, y)` | Returns `self`; repeated fitting clears histories and caches. |
| `transform(X)` | Returns retained columns, in sorted original-column order; can have shape `(n, 0)`. Use `fit(...).transform(...)` to fit and transform. |
| `predict(X)` | Selects columns internally and delegates to the final model. Supply the original feature layout. |
| `score(X, y)` | Delegates to the final model; for `LinearRegression`, this is $R^2$. |
| `summary()` | Prints criterion, direction, selected indices, and final AIC/BIC. |
| `get_params(deep=True)`, `set_params(**params)` | Read/update selector and forwarded model settings; successful nonempty `set_params` clears fitted selection state. |
| `selected_features_` | Sorted list of retained zero-based column indices. |
| `best_model_` | Final estimator refitted on those columns; its coefficients follow the same order. |
| `aic_history_`, `bic_history_` | Initial state and accepted-state criterion values. |
| `selection_history_` | Dictionaries with `action`, `feature`, `features`, `aic`, and `bic`; the initial entry has `action="initial"` and `feature=None`. |

Import `StepwiseSelector` and `stepwise_selection` from `statgpu` or
`statgpu.feature_selection`. The convenience function
`stepwise_selection(X, y, model_class=LinearRegression, criterion="aic", direction="both", **model_kwargs)`
returns a **fitted selector**, not a feature list; additional selector controls
such as `max_features` may be passed as keywords. The complete constructor and
method docstrings are in [`_stepwise.py`](../../../statgpu/feature_selection/_stepwise.py).

## Pitfalls and statistical limits

- Run selection inside each training fold when cross-validating. Selecting once
  on all rows before splitting leaks outcome information into evaluation.
- Do not read ordinary p-values from `best_model_` as selection-adjusted
  inference. The same data chose the variables; simply refitting OLS or applying
  a p-value adjustment afterwards does not correct that search uncertainty.
- Highly correlated variables may substitute for one another. A small change
  in data can change the chosen subset without greatly changing predictions.
- Candidate search can require many fits even though it is not exhaustive.
  Restrict the candidate set using domain knowledge and avoid excessive thread
  counts, especially when each fit consumes substantial GPU memory.
- Results are deterministic when the wrapped estimator is deterministic. Seed
  a stochastic wrapped estimator through its own supported parameters.
- The wrapped model should provide finite `aic` and `bic`. If it does not, but
  supplies finite `rsquared`, the selector uses a Gaussian-style ranking proxy:
  `n * log(max(1 - R2, tiny)) + 2*k` (AIC), or the corresponding `k*log(n)`
  penalty (BIC), with `k = selected columns + fitted-intercept flag`. This
  fallback is not a general likelihood criterion for arbitrary model families.
  Unsupported or nonfinite candidate scores cannot establish a reliable best
  model; inspect the model and histories rather than treating a returned subset
  as evidence of a successful statistical search.

## Related FDR APIs and references

The [feature-selection API reference](../reference/feature-selection-api.md) documents every namespace export, full knockoff signatures/defaults, selector methods, `KnockoffResult` fields and a runnable example.

The feature-selection namespace also exports `KnockoffResult`,
`knockoff_filter`, `fixed_x_knockoff_filter`, `model_x_knockoff_filter`,
`KnockoffSelector`, and `FixedXKnockoffSelector`. Their parameter and output contracts are in the API reference; fixed-X requirements and Gaussian second-order model-X assumptions are explained on the [knockoff page](knockoff.md). They solve a different
selection problem from greedy AIC/BIC search.

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723. [DOI](https://doi.org/10.1109/TAC.1974.1100705)
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*, 6(2), 461–464. [DOI](https://doi.org/10.1214/aos/1176344136)
