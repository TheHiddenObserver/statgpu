# Ordered Generalized Linear Models (Logit/Probit)

> Language: English  
> Last updated: 2026-07-07  
> Switch: [Chinese](../../models/ordered.md)

Ordered response models for ordinal categorical outcomes (e.g., "low/medium/high").

## Model Form

P(y <= j | X) = F(theta_j - X * beta)

Where:
- `j = 1, ..., K-1` are the category thresholds
- `F` is the cumulative distribution function (Logit or Probit)
- `theta_j` are threshold parameters (strictly increasing)
- `beta` is the coefficient vector (proportional odds assumption: all categories share the same coefficients)

## Implemented Estimators

### OrderedLogitRegression

Proportional odds model with Logit link.

```python
from statgpu.linear_model import OrderedLogitRegression

model = OrderedLogitRegression(
    n_categories=3,        # Number of categories
    max_iter=100,          # Max Newton-Raphson iterations
    tol=1e-4,              # Convergence tolerance (NLL change)
    device='auto',         # 'auto' | 'cpu' | 'cuda' | 'torch'
    compute_inference=True, # Compute SE, z-values, p-values, CI
    cov_type='nonrobust',  # Covariance type (only nonrobust currently)
)
model.fit(X, y)
print(model.coef_)          # Raw-scale coefficients (p,)
print(model._thresh_est)    # Raw-scale thresholds (K-1,)
print(model._bse)           # Standard errors [coef SEs, threshold SEs]
print(model._pvalues)       # P-values
print(model.summary())      # Full inference summary table
print(model.aic, model.bic) # Information criteria
```

### OrderedProbitRegression

Ordered model with Probit link.

```python
from statgpu.linear_model import OrderedProbitRegression

model = OrderedProbitRegression(n_categories=3, compute_inference=True, device='cpu')
model.fit(X, y)
```

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

| Backend | Algorithm | Notes |
|---------|-----------|-------|
| numpy (CPU) | Newton-Raphson + vectorized analytical Hessian | NumPy `linalg.solve` |
| cupy (GPU) | Newton-Raphson + vectorized analytical Hessian | CuPy native, zero CPU round-trips (logit) |
| torch (GPU) | Newton-Raphson + vectorized analytical Hessian | Torch native, uses `torch.linalg.solve` |

**Convergence**: 5–23 iterations for typical problems. Trust-region inner loop
(up to 20 attempts per iteration) increases ridge penalty until NLL decreases.

**Standardization**: X is internally standardized to mean=0, std=1.
Coefficients and thresholds are converted back to raw (unstandardized) scale
after convergence: `β_raw = β_fit / X_std`, `θ_raw = θ_fit + X_mean @ β_raw`.

## Inference

### Hessian

Analytical observed Hessian (vectorized, backend-agnostic). Matches R `MASS::polr`
and `ordinal::clm` Hessian structure exactly.

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
| `fit_intercept` | bool | True | Whether to fit intercept term |
| `max_iter` | int | 100 | Max Newton-Raphson iterations |
| `tol` | float | 1e-4 | Convergence tolerance (NLL absolute change) |
| `C` | float | 1.0 | Inverse regularization strength (not used; inherited from GLM base) |
| `device` | str | 'auto' | 'auto' \| 'cpu' \| 'cuda' \| 'torch' |
| `compute_inference` | bool | False | Compute SE, z-values, p-values, CI after fit |
| `cov_type` | str | 'nonrobust' | Covariance estimator type (only nonrobust currently) |
| `gpu_memory_cleanup` | bool | False | Clean GPU memory after fit |

## CPU+GPU Examples

### CPU with Inference

```python
import numpy as np
from statgpu.linear_model import OrderedLogitRegression

np.random.seed(42)
X = np.random.randn(5000, 10)
beta = [0.5, -0.3, 0, 0.8, 0, -0.2, 0, 0.4, 0, 0]
y = np.digitize(0.5 + X @ beta + 0.5 * np.random.randn(5000), [-0.5, 0.5])

model = OrderedLogitRegression(n_categories=3, compute_inference=True, max_iter=50)
model.fit(X, y)
print(model.summary())
# OrderedLogitRegression Summary
# =====================
# n_obs=5000  n_params=12  loglik=-2657.066  aic=5338.133  bic=5416.456
# 
#   Param     Coef    StdErr        z   P>|z|  [0.025   0.975]
#   coef_0   1.705    0.294   5.806   0.000   1.130   2.281
#   ...
#   thresh_0 -3.300   0.751  -4.396   0.000  -4.771  -1.829
#   thresh_1 -0.095   0.155  -0.612   0.541  -0.399   0.209
```

### GPU Fit (no inference)

```python
model = OrderedLogitRegression(n_categories=3, device='cuda', max_iter=50)
model.fit(X, y)  # fits on GPU, results transferred to CPU
print(model.coef_)
```

### Probit with Inference

```python
from statgpu.linear_model import OrderedProbitRegression

model = OrderedProbitRegression(n_categories=3, compute_inference=True)
model.fit(X, y)
print(model._bse)
print(model._pvalues)
```

## strict vs approximate

- **strict**: The analytical Hessian at MLE is the exact observed Fisher information.
  Standard errors match R `MASS::polr` and `ordinal::clm` within float64 tolerance
  when using the same optimization objective (unpenalized NLL).
- **approximate**: GPU paths (CuPy/Torch) produce slightly different coefficient
  estimates due to different math libraries (`libm` vs NVIDIA `libdevice`).
  After ~20 Newton iterations, accumulated BSE differences ~4.5e-04.
  Inference uses the same backend as fitting (NumPy, CuPy, or Torch) via backend-agnostic computation.

## External Validation

| Reference | Method | Agreement |
|-----------|--------|-----------|
| R `ordinal::clm` | Newton-Raphson + analytical Hessian | NLL match (statgpu: -0.532, R: -0.497 on benchmark data) |
| R `MASS::polr` | Fisher scoring | Same Hessian structure, coefficients match |
| `statsmodels` `OrderedModel` | L-BFGS + numerical Hessian | NLL comparable (statgpu achieves lower NLL) |

## References

- McCullagh, P. (1980). Regression models for ordinal data. *JRSS B*, 42(2), 109–142.
- Agresti, A. (2010). *Analysis of Ordinal Categorical Data* (2nd ed.). Wiley.
- Christensen, R. H. B. (2019). ordinal—Regression Models for Ordinal Data. R package.
