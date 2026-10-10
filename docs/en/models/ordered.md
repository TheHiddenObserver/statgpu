# Ordered Generalized Linear Models (Logit/Probit)

> Language: English  
> Last updated: 2026-10-09<br>
> Switch: [Chinese](../../cn/models/ordered.md)

## When to use an ordered model

`OrderedLogitRegression` and `OrderedProbitRegression` model categories with an order, such as low, medium, and high. They use the ordering without assuming equal distances between adjacent categories. Use an unordered classifier for categories without a natural order, and retain continuous information when the response is continuous.

One set of slopes and several thresholds describe the cumulative probabilities. Logit uses the proportional-odds assumption; Probit uses the standard-normal cumulative distribution. Both share slopes across thresholds, which may be too restrictive for some applications.

## Model form

$$
P(y \le j \mid X)=F(\theta_j-X\beta),\qquad j=0,\ldots,K-2.
$$

Labels are coded `0, ..., K-1`, and `theta_j` are strictly increasing interior thresholds. A positive slope shifts probability toward higher categories. In Logit, `exp(beta)` is the higher-versus-lower cumulative odds ratio at each threshold; Probit coefficients do not have that odds-ratio interpretation. Thresholds supply the location parameters, so do not add a constant column to `X`.

<a id="cpu-example"></a>

## A complete CPU example

Run these steps in order to fit and predict three ordered categories with Logit.

### 1. Import

<!-- learner-example: ordered-basic -->
```python
import numpy as np
from statgpu.linear_model import OrderedLogitRegression
```

### 2. Prepare an ordinal response

`X` is a `(400, 2)` numeric matrix, one observation per row; `y` is a length-400 integer label vector. Two cut points turn a latent continuous response with logistic noise into categories 0, 1, and 2. Train on 300 rows and evaluate on 100. For real data, establish the category ordering, handle missing values, and keep prediction columns in the same order.

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(400, 2))
latent = X @ np.array([0.8, -0.5]) + rng.logistic(size=400)
y = np.digitize(latent, [-0.7, 0.8])
```

### 3. Fit and read the coefficients

`n_categories=3` must match the encoding. Leave inference off initially and inspect slopes and interior thresholds on the original feature scale.

```python
model = OrderedLogitRegression(
    n_categories=3, device="cpu", max_iter=200, tol=1e-8,
).fit(X[:300], y[:300])
print("Slopes:", np.round(model.coef_, 3))
print("Thresholds:", np.round(model._thresh_est, 3))
```

The slopes round to `[0.773, -0.458]`, with directions matching the simulation. They are not slopes from a linear regression of the integer labels, and thresholds are not feature coefficients.

### 4. Predict category probabilities

```python
probability = model.predict_proba(X[300:])
prediction = model.predict(X[300:])
print("First probabilities:", np.round(probability[:3], 3))
print("Held-out accuracy:", round(float(model.score(X[300:], y[300:])), 3))
```

The probability array has shape `(100, 3)`, with columns for 0, 1, and 2 and rows summing to one. `predict` selects the most probable category and returns `(100,)` labels. Accuracy is about `0.54` for this seed; accuracy counts all mistakes equally and does not distinguish errors of one category from errors of two.
<!-- example-end: ordered-basic -->

## Choosing settings and checking results

- Encode a meaningful order rather than assuming alphabetical order is ordinal. Sparse or empty categories can destabilize thresholds.
- Check the shared-slope assumption, category probabilities, and held-out performance, rather than training accuracy alone.
- `n_iter_` is an iteration count, not a convergence certificate. If numerical warnings occur, inspect collinearity, separation, and category counts before increasing `max_iter`.
- Slopes describe conditional associations, not automatically causal effects.

## Objective Function

Negative log-likelihood (average-scale):

```
NLL = -(1/n) * Σ_i log P(y_i | X_i)
```

where category probabilities are:

```
P(y=k | X) = F(θ_k - Xβ) - F(θ_{k-1} - Xβ)
```

with boundary conventions `θ_{-1} = -∞`, `θ_{K-1} = ∞`.

## Optimization

Newton-Raphson with trust-region regularization (all 3 backends):

| Backend | Algorithm | Device setting |
|---------|-----------|-------|
| numpy (CPU) | Newton-Raphson + vectorized analytical Hessian | `device="cpu"` |
| cupy (GPU) | Newton-Raphson + vectorized analytical Hessian | `device="cuda"` |
| torch (GPU) | Newton-Raphson + vectorized analytical Hessian | `device="torch"` |

The iteration budget and tolerance control optimization. Numerical regularization stabilizes Newton steps; it is not the statistical slope penalty controlled by C in ordinary GLMs.

**Standardization**: X is internally standardized to mean=0, std=1.
Coefficients and thresholds are converted back to raw (unstandardized) scale
after convergence: `β_raw = β_fit / X_std`, `θ_raw = θ_fit + X_mean @ β_raw`.

## Inference

### Hessian

The analytical observed Hessian is computed from the total negative log-likelihood, even though optimization uses the average loss. This distinction supplies the correct sample-size scaling for covariance.

The Hessian has a block structure:

```
H = [ H_{ββ}   H_{βθ} ]
    [ H_{θβ}   H_{θθ} ]
```

- `H_{ββ}` (p × p): second derivatives w.r.t. coefficients
- `H_{βθ}` (p × K-1): cross-derivatives between coefficients and thresholds
- `H_{θθ}` (K-1 × K-1): second derivatives w.r.t. thresholds

### Covariance

Covariance matrix = `H^{-1}` (inverse observed Hessian at MLE).

Standard errors: `bse = sqrt(diag(H^{-1}))`.

Wald z-statistics: `z = θ / bse`, two-sided p-values via standard normal.

### Attributes (after fit with `compute_inference=True`)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `coef_` | (p,) | Raw-scale coefficient estimates |
| `_thresh_est` | (K-1,) | Raw-scale threshold estimates (internal, no -inf/+inf endpoints) |
| `thresholds_` | (K+1,) | Full threshold vector `[-inf, θ_1, ..., θ_{K-1}, +inf]` |
| `_bse` | (d,) | Standard errors: `[bse_coef, bse_thresh]` where `d = p + K - 1`. Use `_bse[:p]` for coefficients, `_bse[p:]` for thresholds |
| `_zvalues` | (d,) | Wald z-statistics. Use `_zvalues[:p]` / `_zvalues[p:]` to split |
| `_pvalues` | (d,) | Two-sided p-values |
| `_conf_int` | (d, 2) | 95% confidence intervals |
| `loglikelihood` | float | Log-likelihood at MLE |
| `aic` | float | AIC: `-2*loglik + 2*d` |
| `bic` | float | BIC: `-2*loglik + d*log(n)` |
| `n_iter_` | int | Newton-Raphson iterations |

### Current Limitations

- **Nonrobust only**: `cov_type='nonrobust'` is the only supported covariance type.
  HC0/HC1 sandwich, bootstrap, and penalized inference are not yet available.
- **No sample_weight**: `sample_weight` is not supported for ordered models.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_categories` | int | 3 | Number of ordinal categories (>= 2) |
| `fit_intercept` | bool | True | Inherited control; ordered location is represented by thresholds, not a separate intercept |
| `max_iter` | int | 100 | Max Newton-Raphson iterations |
| `tol` | float | 1e-4 | Convergence tolerance (NLL absolute change) |
| `C` | float | 1.0 | Inverse regularization strength (not used; inherited from GLM base) |
| `device` | str | 'auto' | 'auto' \| 'cpu' \| 'cuda' \| 'torch' |
| `compute_inference` | bool | False | Compute SE, z-values, p-values, CI after fit |
| `cov_type` | str | 'nonrobust' | Covariance estimator type (only nonrobust currently) |
| `n_jobs` | int or None | None | Shared configuration; not an ordered solver selector |
| `gpu_memory_cleanup` | bool | False | Clean GPU memory after fit |

## Inference, Probit, and GPU

### Optional: Logit coefficient inference

Reuse the imports and training data from the completed [CPU example](#cpu-example). Refit with `compute_inference` enabled for model-based standard errors and marginal intervals. Only `cov_type="nonrobust"` is supported; these results rely on the model assumptions and large-sample approximation.

<!-- example-requires: ordered-basic -->
<!-- learner-example: ordered-inference -->
```python
inference_model = OrderedLogitRegression(
    n_categories=3, device="cpu", max_iter=200, tol=1e-8,
    compute_inference=True, cov_type="nonrobust",
).fit(X[:300], y[:300])
print("Slope SE:", inference_model._bse[:2])
print("Threshold SE:", inference_model._bse[2:])
print(inference_model.summary())
```
<!-- example-end: ordered-inference -->

The first two entries are slopes and the last two are thresholds; `_pvalues`, `_zvalues`, and `_conf_int` use the same ordering, rather than the ordinary-GLM intercept-first convention. `aic` and `bic` compare compatible likelihood models on the same response and observations.

### Change to a Probit link

Keep `X`, `y` from the completed [CPU example](#cpu-example) and change the link. This fit leaves inference off; enable it as in the preceding subsection when needed, using the same result ordering.

<!-- example-requires: ordered-basic -->
<!-- learner-example: ordered-probit -->
```python
from statgpu.linear_model import OrderedProbitRegression

probit_model = OrderedProbitRegression(
    n_categories=3, device="cpu", max_iter=200, tol=1e-8,
).fit(X[:300], y[:300])
probit_probability = probit_model.predict_proba(X[300:])
```
<!-- example-end: ordered-probit -->

### Request a GPU

Reuse the imports, `X`, and `y` from the completed [CPU example](#cpu-example). The following requests CuPy CUDA; use `device="torch"` for Torch CUDA. The corresponding backend and a usable CUDA device are required; unavailable explicit devices raise.

```python
gpu_model = OrderedLogitRegression(
    n_categories=3, device="cuda", max_iter=200, tol=1e-8,
).fit(X[:300], y[:300])
gpu_probability = gpu_model.predict_proba(X[300:])
```

Coefficients, inference reporting arrays, and `predict` / `predict_proba` outputs are NumPy arrays. See [device and memory](../guides/device-and-memory.md) for setup and device selection.

## Numerical differences and external comparison

This API has no separate strict/approximate mode switch. CPU, CuPy, and Torch use the same model and analytical Hessian, but floating-point arithmetic, tolerance, and conditioning can cause numerical differences. Inference computation uses the selected backend and converts reporting arrays to NumPy.

For comparison with R `MASS::polr`, `ordinal::clm`, or statsmodels `OrderedModel`, align category order, link, design, threshold convention, and convergence settings. statgpu optimizes average negative log-likelihood, while `loglikelihood` reports total log-likelihood. At the same parameters, multiply average negative log-likelihood by the observation count before comparing it with total negative log-likelihood. For standard errors, use the observed Hessian of the total negative log-likelihood and account for parameter order and raw-scale conversion.

The table above covers constructor controls. `fit(X, y)` returns the fitted estimator, `predict_proba` returns category probabilities, `predict` returns labels, and `score` returns accuracy. Use `summary()`, `aic`, `bic`, and inference attributes as explained in the preceding sections. Further signatures are available in the [public implementation](../../../statgpu/linear_model/_glm_base.py) and installed `help(OrderedLogitRegression)` / `help(OrderedProbitRegression)`.

## References

- McCullagh, P. (1980). Regression models for ordinal data. *JRSS B*, 42(2), 109–142.
- Agresti, A. (2010). *Analysis of Ordinal Categorical Data* (2nd ed.). Wiley.
- Christensen, R. H. B. (2019). ordinal—Regression Models for Ordinal Data. R package.
