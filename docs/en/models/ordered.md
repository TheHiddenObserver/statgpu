# Ordered Generalized Linear Models (Logit/Probit)

> Language: English  
> Last updated: 2026-04-26  
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
    fit_intercept=True,    # Whether to fit intercept
    max_iter=100,          # Max iterations
    tol=1e-4,              # Convergence tolerance
    C=1.0,                 # Inverse regularization strength
    device='auto',         # 'auto' | 'cpu' | 'cuda' | 'torch'
)
model.fit(X, y)
print(model.coef_)         # Coefficients
print(model.thresholds_)   # Thresholds [-inf, theta_1, ..., theta_{K-1}, +inf]
```

### OrderedProbitRegression

Ordered model with Probit link.

```python
from statgpu.linear_model import OrderedProbitRegression

model = OrderedProbitRegression(n_categories=3, device='cuda')
model.fit(X, y)
```

## Backend Support

| Backend | Optimizer | Notes |
|---------|-----------|-------|
| numpy (CPU) | scipy L-BFGS-B | Reference implementation |
| cupy (GPU) | Hand-written L-BFGS + Armijo line search | Requires CuPy |
| torch (GPU) | torch.optim.LBFGS + strong_wolfe | Requires PyTorch >= 1.13 |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_categories | int | 3 | Number of ordinal categories (>= 2) |
| fit_intercept | bool | True | Whether to fit intercept term |
| max_iter | int | 100 | Max L-BFGS iterations |
| tol | float | 1e-4 | Convergence tolerance |
| C | float | 1.0 | Inverse regularization strength |
| device | str | 'auto' | Compute device |
| gpu_memory_cleanup | bool | False | Clean GPU memory after fit |

## Attributes (after fit)

| Attribute | Description |
|-----------|-------------|
| coef_ | Coefficient vector (p,) |
| thresholds_ | Threshold vector [-inf, theta_1, ..., +inf] (K+1,) |
| n_iter_ | Actual iteration count |
| _bse | Standard errors |
| _pvalues | P-values |

## Cross-Backend Precision

After 2026-04-26 fixes, max coef difference across backends < 1e-2:

| Comparison | Max |Delta Coef| |
|------------|---------------------------|
| CPU vs CuPy | < 1e-2 |
| CPU vs Torch | < 1e-2 |
| CuPy vs Torch | < 1e-2 |

## References

- McCullagh, P. (1980). Regression models for ordinal data.
- Agresti, A. (2010). Analysis of Ordinal Categorical Data.
