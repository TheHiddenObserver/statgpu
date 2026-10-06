# MCP

> Language: English  
> Last updated: 2026-10-06  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/mcp.md)

Language switch: [Chinese](../../cn/models/mcp.md)

## Overview

`MCPRegression` provides MCP-penalized (Minimax Concave Penalty) linear regression (Zhang, 2010). MCP is a continuous non-convex penalty that reduces shrinkage of large coefficients. Its **oracle property** requires asymptotic regularity and tuning conditions; it is not a guarantee for every finite-sample fit.

## Path

`statgpu.linear_model.MCPRegression`

## Objective Function

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \sum_{j=1}^p p_{\lambda,\gamma}(|\beta_j|)
$$

where the MCP penalty is defined as:

$$
p_{\lambda,\gamma}(\theta) = \begin{cases}
\lambda\theta - \frac{\theta^2}{2\gamma} & \text{if } \theta \le \gamma\lambda \\
\frac{\gamma\lambda^2}{2} & \text{if } \theta > \gamma\lambda
\end{cases}
$$

with concavity parameter $\gamma > 1$ (default 3.0, per Zhang's recommendation).

## Algorithm

MCP uses the same **LLA + FISTA** algorithm as SCAD:

1. **Continuation path**: Decrease $\lambda$ from $\lambda_{max}$ along a geometric grid.
2. **LLA inner loop**:
   - Compute LLA weights: $w_j = p'_{\lambda,\gamma}(|\beta_j|) = \max(\lambda - |\beta_j|/\gamma, 0)$
   - Solve weighted L1 problem via FISTA
3. **Warm-start**: Previous $\lambda$'s solution as initial point.

## Oracle Property

Under regularity conditions (Zhang 2010, Theorem 1):
- **Selection consistency**: $\Pr(\hat{S} = S_0) \to 1$
- **Asymptotic normality**: $\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

For coefficient magnitudes above $\gamma\lambda$, the MCP penalty derivative is zero. Increasing $\gamma$ at fixed $\lambda$ makes the penalty less concave and more Lasso-like; it does not generally reduce shrinkage bias. Zero penalty derivative does not guarantee unbiased finite-sample estimates or selection-adjusted intervals.

## Covariance/Inference

`MCPRegression` defaults to `compute_inference=False`. Its constructor does not
accept `inference_method`; fitting with inference enabled raises because the
inherited automatic request does not choose a selection-conditional method.
For an explicit Gaussian `oracle` or `bootstrap` request, use
`PenalizedLinearRegression(penalty="mcp", penalty_kwargs={"gamma": 3.0}, ...)`
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
| `gamma` | `3.0` | Concavity parameter ($\gamma > 1$, Zhang recommends 3.0) |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `1000` | Maximum FISTA iterations per LLA step |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto`; explicit GPU requests require a usable CUDA backend |
| `compute_inference` | `False` | Keep disabled on this wrapper; use the generic estimator above for an explicit inference method |
| `solver` | `"auto"` | Solver selection |
| `gpu_memory_cleanup` | `False` | CuPy pool cleanup after fit |

## CPU+GPU Examples

```python
from statgpu.linear_model import MCPRegression

# Basic usage
model = MCPRegression(alpha=0.1, gamma=3.0)
model.fit(X, y)
print(model.coef_)        # sparse coefficients
print(model.score(X, y))  # R-squared

# GPU acceleration
model_gpu = MCPRegression(alpha=0.1, device="cuda")
model_gpu.fit(X, y)

# Tuning gamma (concavity)
model_aggressive = MCPRegression(alpha=0.1, gamma=1.5)  # more aggressive thresholding
```

## MCP vs SCAD vs Lasso

| Property | Lasso | SCAD | MCP |
|---|---|---|---|
| Convexity | Convex | Non-convex | Non-convex |
| Oracle property | Not in general | Under regularity/tuning conditions | Under regularity/tuning conditions |
| Bias for large $\beta_j$ | Shrinks toward zero | Nearly unbiased | Nearly unbiased |
| Penalty continuity | Continuous | Continuous | Continuous |
| Penalty concavity | Linear (convex) | Piecewise linear-quadratic | Piecewise linear-quadratic |
| Default concavity param | — | $a = 3.7$ | $\gamma = 3.0$ |

## Outputs

- Coefficients: `intercept_`, `coef_`
- Methods: `fit`, `predict`, `score`
- Coefficient inference: use the explicit generic-estimator interface described above; `MCPRegression` does not expose `inference_method`.

## References

- Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. *Annals of Statistics*, 38(2), 894-942. [https://doi.org/10.1214/09-AOS729](https://doi.org/10.1214/09-AOS729)
- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
