# SCAD

> Language: English  
> Last updated: 2026-10-06  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/scad.md)

Language switch: [Chinese](../../cn/models/scad.md)

## Overview

`SCADRegression` provides SCAD-penalized (Smoothly Clipped Absolute Deviation) linear regression (Fan & Li, 2001). SCAD is a non-convex penalty that reduces shrinkage of large coefficients. Its **oracle property** is an asymptotic result under regularity and tuning conditions, not a finite-sample guarantee for every fitted model.

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
2. **LLA inner loop**:
   - Compute LLA weights: $w_j = p'_{\lambda,a}(|\beta_j|)$ (the subgradient of SCAD at current estimate)
   - Solve weighted L1 problem: $\min \frac{1}{2n}\|y - X\beta\|_2^2 + \sum w_j |\beta_j|$
   - The weighted L1 is solved by FISTA (proximal gradient with momentum)
3. **Warm-start**: Use previous $\lambda$'s solution as initial point for next $\lambda$.

## Oracle Property

Under regularity conditions (Fan & Li 2001, Theorem 2):
- **Selection consistency**: $\Pr(\hat{S} = S_0) \to 1$
- **Asymptotic normality**: $\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

The SCAD penalty derivative is zero for coefficient magnitudes above $a\lambda$, so the penalty no longer directly shrinks those coordinates. This does not guarantee unbiased finite-sample estimates or valid intervals after data-dependent selection.

## Covariance/Inference

`SCADRegression` defaults to `compute_inference=False`. Its constructor does not
accept `inference_method`; fitting with inference enabled raises because the
inherited automatic request does not choose a selection-conditional method.
For an explicit Gaussian `oracle` or `bootstrap` request, use
`PenalizedLinearRegression(penalty="scad", penalty_kwargs={"a": 3.7}, ...)`
with the desired `inference_method` and `compute_inference=True`.

`oracle` reports an unpenalized refit on the selected active set, while prediction
remains penalized. Those ordinary intervals do not correct for choosing variables
on the same responses. The oracle interface rejects GPU parent fits, but its child
currently defaults to `device="auto"`; a CPU parent alone does not guarantee a CPU
child. Residual `bootstrap` is restricted to unweighted Gaussian fits with
`cov_type="nonrobust"`, holds tuning fixed, and supports the fitted numerical
backend. It does not automatically correct tuning or selection uncertainty.
See [inference modes](../guides/inference-modes.md#scadmcp-active-set-inference)
for a complete generic-estimator example and the method-specific limitations.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | Regularization strength ($\lambda$) |
| `a` | `3.7` | Concavity parameter (Fan & Li recommend 3.7) |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `1000` | Maximum FISTA iterations per LLA step |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto`; explicit GPU requests require a usable CUDA backend |
| `compute_inference` | `False` | Keep disabled on this wrapper; use the generic estimator above for an explicit inference method |
| `solver` | `"auto"` | Solver selection |
| `gpu_memory_cleanup` | `False` | CuPy pool cleanup after fit |

## CPU+GPU Examples

```python
from statgpu.linear_model import SCADRegression

# Basic usage
model = SCADRegression(alpha=0.1, a=3.7)
model.fit(X, y)
print(model.coef_)        # sparse coefficients
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
| Oracle property | Not in general | Under regularity and tuning conditions |
| Bias for large $\beta_j$ | Shrinks toward zero | Nearly unbiased |
| Optimization | Convex objective; check numerical convergence | Multiple local minima possible |
| Sparsity | Yes | Yes (often sparser) |

## Outputs

- Coefficients: `intercept_`, `coef_`
- Methods: `fit`, `predict`, `score`
- Coefficient inference: use the explicit generic-estimator interface described above; `SCADRegression` does not expose `inference_method`.

## References

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360. [https://doi.org/10.1198/016214501753382273](https://doi.org/10.1198/016214501753382273)
- Wang, H., Li, R., & Tsai, C.-L. (2007). Tuning parameter selectors for the smoothly clipped absolute deviation method. *Biometrika*, 94(3), 553-568.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509-1533.
