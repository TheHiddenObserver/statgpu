# Feature-selection API reference

> Language: English  
> Last updated: 2026-10-05  
> Switch: [Chinese](../../cn/reference/feature-selection-api.md)

This page covers all eight exports of `statgpu.feature_selection`: `StepwiseSelector`, `stepwise_selection`, `KnockoffResult`, `knockoff_filter`, `fixed_x_knockoff_filter`, `model_x_knockoff_filter`, `KnockoffSelector`, and `FixedXKnockoffSelector`. All except `KnockoffResult` also have top-level `statgpu` aliases. These classes do not inherit BaseEstimator's p-value/bootstrap helpers.

## StepwiseSelector

```python
StepwiseSelector(model_class, criterion='aic', direction='both', max_features=None, n_jobs=None, verbose=False, **model_kwargs)
stepwise_selection(X, y, model_class=LinearRegression, criterion='aic', direction='both', **model_kwargs)
```

Pass a callable/class as `model_class`; candidate fitting needs compatible `fit`, AIC/BIC diagnostics, and the downstream methods you intend to use. Constructor options, all seven public methods, selected/history fields, greedy-search behavior, scoring fallback, and restrictions are fully listed on the [stepwise learner page](../models/feature-selection.md#choosing-parameters).

`stepwise_selection` returns a fitted selector. Its keyword arguments can include selector controls such as `max_features`, `n_jobs`, `verbose` as well as wrapped-model settings; selector controls are consumed by StepwiseSelector rather than forwarded. `fit`/`score` accept a flat response or a single column, which is flattened. No `fit_transform`, `get_support`, sample weights, formula input, or coefficient-inference API is supplied by StepwiseSelector. Use `fit(...).transform(...)` and `selected_features_`.

`get_params(deep=True)` remains flat. `set_params` treats names outside the
selector controls as wrapped-model constructor keywords, so unknown model
parameters can fail only at `fit`. A returned selector does not establish that
all criterion scores were valid: an infinite initial score can block otherwise
finite improvements, and a failed backward search can exceed `max_features`.
Check finite criterion histories and the cap before using the selected model;
see [nonfinite-score limitations](../models/feature-selection.md#nonfinite-score-limitations).

## Knockoff functions and constructors

```python
fixed_x_knockoff_filter(X, y, q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', Xk=None, compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off')

model_x_knockoff_filter(X, y, q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', Xk=None, compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off', modelx_covariance_shrinkage=0.2, modelx_s_scale=0.999, modelx_draws=None, modelx_shrinkage='ledoitwolf', modelx_smatrix_method='mvr', knockpy_sampler=None, knockpy_sampler_method=None)

knockoff_filter(X, y, knockoff_type='fixed_x', q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', Xk=None, compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off', modelx_covariance_shrinkage=0.2, modelx_s_scale=0.999, modelx_draws=None, modelx_shrinkage='ledoitwolf', modelx_smatrix_method='mvr', knockpy_sampler=None, knockpy_sampler_method=None)

KnockoffSelector(knockoff_type='fixed_x', q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off', modelx_covariance_shrinkage=0.2, modelx_s_scale=0.999, modelx_draws=None, modelx_shrinkage='ledoitwolf', modelx_smatrix_method='mvr', knockpy_sampler=None, knockpy_sampler_method=None)

FixedXKnockoffSelector(q=0.1, method='corr_diff', fdr_control='knockoff_plus', random_state=None, backend='auto', compat_mode='statgpu', lasso_cv_impl='auto', lasso_fast_profile='off')
```

The fixed-X function/class accepts only the shared subset shown in its signature. `modelx_*` and sampler parameters apply to the model-X branch; do not expect them to change fixed-X results. Functions return `KnockoffResult`; selector constructors return unfitted selector objects.

| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `X, y` | `required` | Finite numeric X `(n,p)` and y `(n,)`, same n. |
| `knockoff_type` | `"fixed_x"` | Unified function/selector only: `fixed_x` or `model_x`. |
| `q` | `0.1` | Target rate in `(0,1)`; not a confidence level for coefficients. |
| `method` | `"corr_diff"` | `corr_diff`, `ols_coef_diff`, `lasso_coef_diff`; original-minus-knockoff importance statistic. |
| `fdr_control` | `"knockoff_plus"` | `knockoff_plus` uses offset 1, `knockoff` offset 0. The latter has a different modified-FDR guarantee under the applicable theory. |
| `random_state` | `None` | Integer seed for construction/statistic fitting where stochastic. |
| `backend` | `"auto"` | `numpy`, `cupy`, `torch`, or auto inferred from arrays. Not the estimator `device` keyword. |
| `Xk` | `None` | Optional external knockoff matrix `(n,p)`; supplied to functions or selector.fit, never selector constructor. Validity is the caller’s responsibility; shape alone does not establish exchangeability. |
| `compat_mode` | `"statgpu"` | `statgpu` or `knockpy`; compatibility controls change construction/statistic conventions and can require optional packages/CPU work. |
| `lasso_cv_impl` | `"auto"` | `statgpu` or `sklearn`; auto chooses sklearn for knockpy compatibility and statgpu otherwise. Applies to the Lasso statistic. |
| `lasso_fast_profile` | `"off"` | `off`, `auto`, `moderate`, `aggressive`; Lasso computational profile. Start with off for comparisons. |
| `modelx_covariance_shrinkage` | `0.20` | Native model-X covariance shrinkage; choose within `[0,1]`. |
| `modelx_s_scale` | `0.999` | Native model-X S-matrix scale, normally `(0,1]`. |
| `modelx_draws` | `None` | Strict positive integer or None: defaults to 5 for OLS/Lasso differences and 3 for correlation differences. Supplied Xk gives one matrix rather than fresh draws. |
| `modelx_shrinkage` | `"ledoitwolf"` | Compatibility covariance strategy: `ledoitwolf`, `none`/`mle`, or `graphicallasso`/`glasso`. See resolution/fallback behavior below. |
| `modelx_smatrix_method` | `"mvr"` | Requested compatibility S-matrix method, forwarded to knockpy when available; the request can fall back to equicorrelated construction. See below. |
| `knockpy_sampler` | `None` | Optional dispatch name such as gaussian/fx/metro/artk. Current dispatched implementations are placeholders and can raise NotImplementedError; leave None for supported built-in construction. |
| `knockpy_sampler_method` | `None` | Gaussian dispatch submethod, e.g. mvr/sdp/maxent/equi/ci; does not implement an unavailable sampler. |

### Compatibility resolution and fallbacks

With `compat_mode="knockpy"` and built-in model-X construction, covariance estimation runs locally on CPU. `none`/`mle` uses sample covariance but switches to Ledoit–Wolf when its minimum eigenvalue is below the internal threshold. Ledoit–Wolf and graphical lasso use sklearn; if that import fails, sample covariance is used and `metadata["modelx_covariance_estimator"]` reports `"mle_fallback_no_sklearn"`.

S-matrix construction attempts knockpy's requested method. A missing package **or any exception from that call** triggers an equicorrelated fallback; even an invalid method name can therefore produce a result rather than an error. `metadata["modelx_smatrix_method"]` is only the requested name. Inspect `modelx_covariance_estimator` and `modelx_smatrix_source` (`"knockpy"` or `"equicorrelated_fallback"`) before interpreting the result as execution of a particular covariance/S-matrix method or as knockpy parity. These fallback labels do not establish knockoff validity or FDR control.

## Selector methods

Both `KnockoffSelector` and `FixedXKnockoffSelector` expose:

| Method | Input/return |
|---|---|
| `fit(X,y,Xk=None)` | Fits selection and returns **self**, not the result object. Sets `result_` and zero-based NumPy `selected_features_`. |
| `get_support()` | Boolean NumPy mask `(p,)`; no `indices` argument. |
| `transform(X)` | Original feature layout `(m,p)` required; returns selected columns `(m,s)`, preserving NumPy/CuPy/Torch input backend and dtype. Width mismatch raises; s can be zero. |
| `fit_transform(X,y,Xk=None)` | Fits on X/y and returns transformed X; not a prediction or held-out evaluation. |
| `get_params(deep=True)` | Constructor configuration dictionary; these wrappers have no nested estimator expansion. |
| `set_params(**params)` | Returns self; valid nonempty updates clear fitted selection state, unknown names raise. |

A failed `fit` currently leaves a previous successful selection accessible. After any refit error, do not treat `result_`, `get_support()`, or `transform()` as a result for the new data; create a fresh selector and complete a successful fit.

These selectors do not fit a response-prediction model and supply no `predict`, `score`, or `summary`. To evaluate prediction, fit a separate estimator on selected training columns and apply the same selection to held-out columns. Re-select within every training fold.

## KnockoffResult

`KnockoffResult` is a dataclass for reporting, not a fitting estimator. Its constructor requires `knockoff_type`, `selected_features`, `W`, `threshold`, `q`, `estimated_fdr`, `q_trajectory`, `method`, `fdr_control`, `random_state`, and `backend`; `metadata` defaults to a new empty dictionary. Ordinarily obtain it from a filter or `selector.result_` rather than constructing it yourself.

| Field | Meaning |
|---|---|
| `selected_features` | NumPy `int64` indices `(s,)`, zero-based; empty selection is valid. |
| `W` | NumPy `float64` feature statistics `(p,)`, even when numerical work used GPU. Larger positive values favor the original feature. |
| `threshold` | Selection threshold; infinity can represent no eligible threshold. |
| `q`, `estimated_fdr` | Requested target and the threshold-rule estimate, not the unknowable realized false-discovery proportion. |
| `q_trajectory` | Per stable-sorted rank dictionaries with `rank`, `threshold`, `fdr_hat`, `n_selected`. `fdr_hat` is clipped at 1; `n_selected` is the positive-prefix count floored at 1. These rank diagnostics do not certify the full selected set when absolute statistics tie. |
| `knockoff_type`, `method`, `fdr_control`, `random_state`, `backend` | Method/configuration labels. |
| `metadata` | Construction/statistic details such as data dimensions, knockoff source and draw count; inspect actual execution information for compatibility paths. |
| `to_dict()` | Dictionary of all fields, converting W/indices to lists; does not refit. |

## Statistical and backend boundaries

The theoretical knockoff+ threshold must count **all** features at each distinct absolute-statistic threshold. Current tied-|W| handling can select a partial-prefix threshold and understate its estimated FDR before selecting the entire tie. For W=[8,8,-8], q=0.5, the implementation can select [0,1] with estimated_fdr=0.5 while the full theoretical ratio equals 1. Do not claim nominal FDR control for these tied-threshold outputs; see the [learner warning](../models/knockoff.md#tied-statistic-limitation).



Generated fixed-X knockoffs require the rank/sample conditions of their construction (typically n≥2p and full column rank after standardization). A user-supplied Xk bypasses construction; it must actually be a valid matched knockoff design. Model-X uses an estimated Gaussian second-order feature model: matching estimated moments is not a distribution-free guarantee for arbitrary features, and averaging multiple draws is not automatically an independent FDR theorem. Interpret results under the construction/statistic assumptions in the [knockoff guide](../models/knockoff.md).

Knockoff construction/statistic inputs are converted to float64; selector `transform` only selects columns and preserves the original input dtype. Native NumPy/CuPy/Torch numerical paths exist, but choosing `lasso_cv_impl="sklearn"` or certain `compat_mode="knockpy"` construction routes entails host conversion/CPU or optional-library work. `backend` is not a guarantee that every compatibility step stays on GPU. The native alternative with supplied Xk and explicit `lasso_cv_impl="statgpu"` avoids the general knockpy CPU construction route. Unsupported sampler dispatch raises; it does not automatically implement another sampler.

## Runnable fixed-X example

<!-- api-example: knockoff-selector -->
```python
import numpy as np
from statgpu.feature_selection import FixedXKnockoffSelector

rng = np.random.default_rng(12)
X = rng.normal(size=(120, 5))
y = 3 * X[:, 0] + rng.normal(size=120)
selector = FixedXKnockoffSelector(backend="numpy", random_state=7)
assert selector.fit(X, y) is selector
selected = selector.transform(X[:10])
assert selected.shape == (10, int(selector.get_support().sum()))
assert selector.result_.W.shape == (5,)
print(selector.selected_features_.tolist())
```

A small problem can return no features, particularly with knockoff+ at a stringent q. That is an admissible selection outcome; do not loosen the threshold after inspecting results merely to force discoveries.
