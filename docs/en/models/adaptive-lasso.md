# Adaptive Lasso

> Language: English  
> Last updated: 2026-06-14  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/adaptive-lasso.md)

Language switch: [Chinese](../../cn/models/adaptive-lasso.md)

## Overview

`AdaptiveLasso` provides adaptive L1-penalized linear regression (Zou, 2006). Unlike standard Lasso which uses a uniform penalty, Adaptive Lasso assigns data-driven per-coordinate weights $w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$, achieving the **oracle property** — asymptotically performing as well as if the true model were known.

## Path

`statgpu.linear_model.AdaptiveLasso`

## Objective Function

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \alpha \sum_{j=1}^p w_j |\beta_j|
$$

where $w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$ are adaptive weights computed from an initial estimate (ridge regression by default).

## Algorithm

1. **Initialization**: Compute initial coefficient estimates via ridge-penalized coordinate descent (matching R glmnet's ridge solver).
2. **Weight computation**: $w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$ with $\nu = 1$ (default).
3. **Weighted L1 solve**: Solve the weighted Lasso problem using FISTA with the computed weights.

## Oracle Property

Under regularity conditions (Zou 2006, Theorem 1):
- **Selection consistency**: $\Pr(\hat{S} = S_0) \to 1$ as $n \to \infty$
- **Asymptotic normality**: $\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

where $S_0$ is the true support set and $\Sigma_0$ is the oracle information matrix.

## Covariance/Inference

- `compute_inference=False` by default (adaptive_l1 does not support debiased inference)
- For inference on selected variables, use the oracle approach: refit OLS on the selected support set

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | Regularization strength |
| `nu` | `1.0` | Exponent for weight computation (1 or 2, per Zou 2006) |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `1000` | Maximum iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` |
| `solver` | `"auto"` | Solver selection |
| `gpu_memory_cleanup` | `False` | CuPy pool cleanup after fit |

## CPU+GPU Examples

```python
from statgpu.linear_model import AdaptiveLasso

# Basic usage
model = AdaptiveLasso(alpha=0.1, nu=1.0)
model.fit(X, y)
print(model.coef_)        # sparse coefficients
print(model.score(X, y))  # R-squared

# GPU acceleration
model_gpu = AdaptiveLasso(alpha=0.1, device="cuda")
model_gpu.fit(X, y)
```

## Outputs

- Coefficients: `intercept_`, `coef_`
- Methods: `fit`, `predict`, `score`
- Note: `compute_inference=True` is not supported for adaptive_l1

## External Validation

- `dev/tests/test_refactor_safety_net.py` (solver convergence tests)

## References

- Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476), 1418-1429. [https://doi.org/10.1198/016214506000000735](https://doi.org/10.1198/016214506000000735)
- Wang, H., Li, B., & Leng, C. (2009). Shrinkage tuning parameter selection with a diverging number of parameters. *Journal of the Royal Statistical Society: Series B*, 71(3), 671-683.
