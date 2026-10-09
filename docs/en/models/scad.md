# SCAD

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/scad.md)

## Overview

`SCADRegression` provides SCAD-penalized (Smoothly Clipped Absolute Deviation) linear regression (Fan & Li, 2001). SCAD is a non-convex penalty that reduces shrinkage of large coefficients. Its **oracle property** is an asymptotic result under regularity and tuning conditions, not a finite-sample guarantee for every fitted model.

Use this model for a continuous response when sparse prediction is useful and
Lasso's shrinkage of large slopes is a concern. Begin with [Lasso](lasso.md) or
[Elastic Net](elastic-net.md) when you want a convex objective and simpler tuning.
Non-convex fitting can converge to different local solutions; a selected feature
is not automatically significant or causal.

## Path

`statgpu.linear_model.SCADRegression`

## Objective Function

$$
\min_{b,\beta} \frac{1}{2n}\|y - b\mathbf{1} - X\beta\|_2^2 + \sum_{j=1}^p p_{\lambda,a}(|\beta_j|)
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

<!-- learner-example: scad-prediction -->
```python
import numpy as np
from statgpu.linear_model import SCADRegression
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
model = SCADRegression(
    alpha=0.1, a=3.7, device="cpu", compute_inference=False,
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
<!-- example-end: scad-prediction -->

For this seed, slopes are approximately `[1.987, -0.982, 0, 0, 0]`, the
intercept is `1.430`, and held-out R² is `0.967`. Predictions have shape `(40,)`.
These are penalized prediction coefficients, not active-set refits or significance
results. Another sample can select noise or miss a real signal.

## Choosing parameters and checking results

- Choose `alpha` using training-only validation. The example's 0.1 is not a
  universal default, and the internal continuation path does not perform CV.
- `a=3.7` is a conventional starting point. Changing concavity changes
  the penalty and optimization difficulty; compare prediction and selected-set
  stability rather than interpreting greater sparsity as automatically better.
- Increase `max_iter` and tighten `tol` to assess numerical stability. `n_iter_`
  alone does not prove global optimality. Local minima remain possible.
- Keep the test set separate from tuning, and repeat learned preprocessing
  inside each training fold. These wrappers have no built-in CV method.

## Algorithm

SCAD uses **LLA (Local Linear Approximation)** + FISTA:

1. **Continuation path**: Start from $\lambda_{max}$ and decrease along a geometric grid.
2. **LLA inner loop**:
   - Compute LLA weights: $w_j = p'_{\lambda,a}(|\beta_j|)$ (the subgradient of SCAD at current estimate)
   - Solve weighted L1 problem: $\min_{b,\beta} \frac{1}{2n}\|y - b\mathbf{1} - X\beta\|_2^2 + \sum w_j |\beta_j|$
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
| `alpha` | `1.0` | Finite positive regularization strength ($\lambda$) |
| `a` | `3.7` | Finite concavity parameter, greater than 2 (conventional value 3.7) |
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

The [complete SCADRegression method reference](../reference/linear-model-api.md#scadregression)
includes fitting, formula/weight rules, prediction placement, score, inherited
helpers and diagnostic restrictions. The table above lists every constructor
parameter. Import this class from `statgpu.linear_model`; it is not exported
from top-level `statgpu`.

## SCAD vs Lasso

| Property | Lasso | SCAD |
|---|---|---|
| Convexity | Convex | Non-convex |
| Oracle property | Not in general | Under regularity and tuning conditions |
| Bias for large $\beta_j$ | Shrinks toward zero | Zero derivative above the threshold; no finite-sample guarantee |
| Optimization | Convex objective; check numerical convergence | Multiple local minima possible |
| Sparsity | Yes | Yes; selected size depends on data and tuning |

## Outputs

- Coefficients: `intercept_`, `coef_`
- Methods: `fit`, `predict`, `score`
- Coefficient inference: use the explicit generic-estimator interface described above; `SCADRegression` does not expose `inference_method`.

## References

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360. [https://doi.org/10.1198/016214501753382273](https://doi.org/10.1198/016214501753382273)
- Wang, H., Li, R., & Tsai, C.-L. (2007). Tuning parameter selectors for the smoothly clipped absolute deviation method. *Biometrika*, 94(3), 553-568.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509-1533.
