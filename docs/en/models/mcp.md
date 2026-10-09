# MCP

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/mcp.md)

## Overview

`MCPRegression` provides MCP-penalized (Minimax Concave Penalty) linear regression (Zhang, 2010). MCP is a continuous non-convex penalty that reduces shrinkage of large coefficients. Its **oracle property** requires asymptotic regularity and tuning conditions; it is not a guarantee for every finite-sample fit.

Use this model for a continuous response when sparse prediction is useful and
Lasso's shrinkage of large slopes is a concern. Begin with [Lasso](lasso.md) or
[Elastic Net](elastic-net.md) when you want a convex objective and simpler tuning.
Non-convex fitting can converge to different local solutions; a selected feature
is not automatically significant or causal.

## Path

`statgpu.linear_model.MCPRegression`

## Objective Function

$$
\min_{b,\beta} \frac{1}{2n}\|y - b\mathbf{1} - X\beta\|_2^2 + \sum_{j=1}^p p_{\lambda,\gamma}(|\beta_j|)
$$

where the MCP penalty is defined as:

$$
p_{\lambda,\gamma}(\theta) = \begin{cases}
\lambda\theta - \frac{\theta^2}{2\gamma} & \text{if } \theta \le \gamma\lambda \\
\frac{\gamma\lambda^2}{2} & \text{if } \theta > \gamma\lambda
\end{cases}
$$

with concavity parameter $\gamma > 1$ (default 3.0, per Zhang's recommendation).

Here n is the observation count, X has p feature columns, b is the unpenalized
intercept, and `alpha` is $\lambda$. The scalar penalty argument is
$\theta=|\beta_j|\geq0$. `fit_intercept=False` fixes b=0. With finite,
nonnegative analytic weights of positive total, replace the data-fit term by
$\sum_i w_i(y_i-b-x_i^\top\beta)^2/(2\sum_i w_i)$; multiplying all weights
by the same positive constant leaves the objective unchanged. Features are not
automatically standardized. Learn any scaling on training rows only.

## A complete CPU example

The predictors below already have comparable scales. Reserve the final 40 rows
before fitting; only the first two features generate the signal.

Run the following steps in order in one Python session. Start with the imports.

<!-- learner-example: mcp-prediction -->
```python
import numpy as np
from statgpu.linear_model import MCPRegression
```

<a id="cpu-data"></a>

### Prepare training and test data

`X` has shape `(160, 5)`: each row is one observation and each column is a predictor. `y` is a one-dimensional continuous response with shape `(160,)`. Keep the last 40 rows out of fitting and tuning.

```python
rng = np.random.default_rng(64)
X = rng.normal(size=(160, 5))
y = 1.5 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.4, size=160)
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]
```

### Fit the prediction model

Use a fixed illustrative penalty first. Inference is disabled so this step only estimates the coefficients used for prediction.

```python
model = MCPRegression(
    alpha=0.1, gamma=3.0, device="cpu", compute_inference=False,
    max_iter=5000, tol=1e-8,
)
model.fit(X_train, y_train)
```

### Predict and inspect the fit

Pass the test features in the same column order. `score` evaluates R² against the held-out responses.

```python
prediction = model.predict(X_test)
print("Slopes:", np.round(model.coef_, 3))
print("Intercept:", round(model.intercept_, 3))
print("Test R2:", round(model.score(X_test, y_test), 3))
```
<!-- example-end: mcp-prediction -->

For this seed, slopes are approximately `[1.987, -0.982, 0, 0, 0]`, the
intercept is `1.430`, and held-out R² is `0.967`. Predictions have shape `(40,)`.
These are penalized prediction coefficients, not active-set refits or significance
results. Another sample can select noise or miss a real signal.

## Choosing parameters and checking results

- Choose `alpha` using training-only validation. The example's 0.1 is not a
  universal default, and the internal continuation path does not perform CV.
- `gamma=3.0` is a conventional starting point. Changing concavity changes
  the penalty and optimization difficulty; compare prediction and selected-set
  stability rather than interpreting greater sparsity as automatically better.
- Increase `max_iter` and tighten `tol` to assess numerical stability. `n_iter_`
  alone does not prove global optimality. Local minima remain possible.
- Keep the test set separate from tuning, and repeat learned preprocessing
  inside each training fold. These wrappers have no built-in CV method.

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
| `alpha` | `1.0` | Finite positive regularization strength ($\lambda$) |
| `gamma` | `3.0` | Finite concavity parameter, greater than 1 (conventional value 3.0) |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `1000` | Maximum FISTA iterations per LLA step |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto`; explicit GPU requests require a usable CUDA backend |
| `compute_inference` | `False` | Keep disabled on this wrapper; use the generic estimator above for an explicit inference method |
| `solver` | `"auto"` | Solver selection |
| `gpu_memory_cleanup` | `False` | CuPy pool cleanup after fit |

## Optional GPU use

After the [CPU data setup](#cpu-data), use `device="cuda"` for CuPy CUDA
or `device="torch"` for Torch CUDA in the same constructor. An unavailable
explicit backend raises; only `auto` can select another available backend.
See [device and memory](../guides/device-and-memory.md). No GPU is needed for
the CPU example.

## API and output shapes

The [complete MCPRegression method reference](../reference/linear-model-api.md#mcpregression)
includes fitting, formula/weight rules, prediction placement, score, inherited
helpers and diagnostic restrictions. The table above lists every constructor
parameter. Import this class from `statgpu.linear_model`; it is not exported
from top-level `statgpu`.

## MCP vs SCAD vs Lasso

| Property | Lasso | SCAD | MCP |
|---|---|---|---|
| Convexity | Convex | Non-convex | Non-convex |
| Oracle property | Not in general | Under regularity/tuning conditions | Under regularity/tuning conditions |
| Bias for large $\beta_j$ | Shrinks toward zero | Zero derivative above the threshold; no finite-sample guarantee | Zero derivative above the threshold; no finite-sample guarantee |
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
