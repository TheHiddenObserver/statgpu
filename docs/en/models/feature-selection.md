# Feature Selection

> Language: English  
> Last updated: 2026-07-12  
> Switch: [Chinese](../../cn/models/feature-selection.md)

## Overview and Paths

`StepwiseSelector` and `stepwise_selection` perform AIC/BIC subset search in
`forward`, `backward`, or `both` directions. `knockoff_filter` and its selector
wrappers provide fixed-X and Gaussian second-order model-X FDR control. See the
[detailed knockoff page](knockoff.md).

```python
from statgpu import LinearRegression, StepwiseSelector

selector = StepwiseSelector(
    LinearRegression,
    criterion="bic",
    direction="both",
    max_features=10,
    compute_inference=False,
).fit(X, y)
X_selected = selector.transform(X)
```

## Stepwise Contract

- Candidate subsets are fitted in sorted feature order, matching `predict` and
  `transform`.
- Backward selection starts from the full model and enforces `max_features` as a
  hard cap before requiring criterion improvement.
- An intercept-only/null model may win; no feature is forced into the model.
- Repeated `fit()` resets histories and caches.
- `n_jobs` uses threads so device arrays are not serialized into worker processes.
- The computation backend and inference capability follow `model_class` and its
  keyword arguments.

## Outputs

Fitted selectors expose `selected_features_`, `best_model_`, `aic_history_`,
`bic_history_`, and `selection_history_`, plus `predict`, `transform`, and
`fit_transform`.

## Validation and Limits

Selection is deterministic for deterministic wrapped estimators. Information-criterion
search is combinatorial and is intended for moderate feature counts; knockoff methods
are preferable when FDR control or high-dimensional screening is the primary goal.
