# SCAD

> Language: English  
> Last updated: 2026-06-14  
> This page: Model documentation  
> Switch: [Chinese](../../models/scad.md)

Language switch: [Chinese](../../models/scad.md)

## Overview

`SCADRegression` provides SCAD-penalized (Smoothly Clipped Absolute Deviation) linear regression (Fan & Li, 2001). SCAD is a non-convex penalty that achieves the **oracle property** while producing nearly unbiased estimates for large coefficients — addressing the estimation bias of Lasso.

## Path

`statgpu.linear_model.SCADRegression`

## Objective Function

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \sum_{j=1}^p p_{\lambda,a}(|\beta_j|)
$$

where the SCAD penalty is defined as:

$$
p_{\lambda,a}(\theta) = \begin{cases}
\lambda \theta & \text{if } \theta \le \lambda \\
\frac{2a\lambda\theta - \theta^2 - \lambda^2}{2(a-1)} & \text{if } \lambda < \theta \le a\lambda \\
\frac{(a+1)\lambda^2}{2} & \text{if } \theta > a\lambda
\end{cases}
$$

with concavity parameter $a = 3.7$ (recommended by Fan & Li).

## Algorithm

SCAD uses **LLA (Local Linear Approximation)** + FISTA:

1. **Continuation path**: Start from $\lambda_{max}$ and decrease along a geometric grid.
2. **LLA inner loop** (1-6 iterations per $\lambda$):
   - Compute LLA weights: $w_j = p'_{\lambda,a}(|\beta_j|)$ (the subgradient of SCAD at current estimate)
   - Solve weighted L1 problem: $\min \frac{1}{2n}\|y - X\beta\|_2^2 + \sum w_j |\beta_j|$
   - The weighted L1 is solved by FISTA (proximal gradient with momentum)
3. **Warm-start**: Use previous $\lambda$'s solution as initial point for next $\lambda$.

## Oracle Property

Under regularity conditions (Fan & Li 2001, Theorem 2):
- **Selection consistency**: $\Pr(\hat{S} = S_0) \to 1$
- **Asymptotic normality**: $\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

SCAD produces **unbiased** estimates for large coefficients (unlike Lasso which shrinks all coefficients toward zero).

## Covariance/Inference

- `compute_inference=False` by default (SCAD does not support debiased inference)
- For inference on selected variables, use the oracle approach: refit OLS on the selected support set
- Future: oracle inference and BIC-based hyperparameter selection (see TO_DO.md)

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | Regularization strength ($\lambda$) |
| `a` | `3.7` | Concavity parameter (Fan & Li recommend 3.7) |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `1000` | Maximum FISTA iterations per LLA step |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` |
| `solver` | `"auto"` | Solver selection |
| `gpu_memory_cleanup` | `False` | CuPy pool cleanup after fit |

## CPU+GPU Examples

```python
from statgpu.linear_model import SCADRegression

# Basic usage
model = SCADRegression(alpha=0.1, a=3.7)
model.fit(X, y)
print(model.coef_)        # sparse coefficients (more unbiased than Lasso)
print(model.score(X, y))  # R-squared

# GPU acceleration
model_gpu = SCADRegression(alpha=0.1, device="cuda")
model_gpu.fit(X, y)

# Tuning 'a' (concavity)
model_concave = SCADRegression(alpha=0.1, a=2.5)  # more concave
```

## SCAD vs Lasso

| Property | Lasso | SCAD |
|---|---|---|
| Convexity | Convex | Non-convex |
| Oracle property | No | Yes |
| Bias for large $\beta_j$ | Shrinks toward zero | Nearly unbiased |
| Global optimum | Guaranteed | Multiple local minima possible |
| Sparsity | Yes | Yes (often sparser) |

## Outputs

- Coefficients: `intercept_`, `coef_`
- Methods: `fit`, `predict`, `score`
- Note: `compute_inference=True` is not supported for SCAD

## References

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360. [https://doi.org/10.1198/016214501753382273](https://doi.org/10.1198/016214501753382273)
- Wang, H., Li, R., & Tsai, C.-L. (2007). Tuning parameter selectors for the smoothly clipped absolute deviation method. *Biometrika*, 94(3), 553-568.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509-1533.
