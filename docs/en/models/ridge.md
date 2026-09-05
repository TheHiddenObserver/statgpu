# Ridge

> Language: English
> Last updated: 2026-09-05
> Switch: [简体中文](../../cn/models/ridge.md)

## What problem does it solve?

`Ridge` is linear regression with an L2 penalty. It is useful when ordinary least squares (OLS) fits the data but the individual coefficients are too unstable, especially because several predictors carry nearly the same information.

A common symptom is **multicollinearity**: two highly correlated features can make OLS coefficients swing dramatically even though predictions barely change. Ridge accepts a small amount of bias in exchange for smaller, more stable coefficients.

Typical questions include:

- Can I make a linear model less sensitive to correlated predictors?
- Can I reduce coefficient variance without forcing features to disappear?
- Can I improve prediction when OLS is overfitting noisy directions?

## A motivating example

Suppose two sensors measure almost the same physical quantity. Both are genuinely useful, but because their columns in `X` are nearly duplicates, OLS has many almost-equivalent ways to divide the signal between them.

You may see something like:

```text
               sensor 1   sensor 2
OLS coefficient    0.32        2.08
Ridge coefficient  1.09        1.09
```

Both models can make similar predictions. Ridge is often easier to trust because it avoids using a large positive coefficient on one correlated feature and a compensating coefficient on another.

## Intuition

OLS asks only:

> Which coefficients minimize prediction error on the training data?

Ridge adds a second preference:

> Among models with similar fit, prefer the one with smaller coefficients.

Geometrically, the L2 penalty discourages the coefficient vector from moving far from zero in any direction. It **shrinks** coefficients continuously but normally does not set them exactly to zero.

That distinction is important:

- Ridge is mainly a **stabilization / shrinkage** method.
- [Lasso](lasso.md) adds sparsity and can set coefficients exactly to zero.
- [Elastic Net](elastic-net.md) combines both behaviors.

## When to use it

Ridge is a strong default when:

- the response is continuous and a linear mean model is appropriate;
- predictors are strongly correlated;
- you have many predictors and want to reduce variance;
- prediction matters more than preserving the unbiased OLS estimator;
- you want all predictors to remain in the model rather than perform hard feature selection.

Choose another method when:

- you specifically need a sparse model with many exact zeros — start with Lasso or Elastic Net;
- the response is binary, count-valued, survival time, or otherwise poorly described by Gaussian linear regression — use the corresponding GLM or survival model;
- important relationships are nonlinear and have not been represented by suitable features;
- coefficient-level causal interpretation is your goal but the design does not support a causal claim.

## Model and objective

With an intercept, the linear model is

$$
y_i=b+x_i^\top\beta+\varepsilon_i.
$$

Ridge estimates the intercept $b$ and coefficient vector $\beta$ by minimizing

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\lVert\beta\rVert_2^2.
$$

Here:

- $n$ is the number of observations;
- $x_i$ is the feature vector for observation $i$;
- $\alpha\ge 0$ controls the amount of shrinkage;
- the intercept is not penalized.

As `alpha` increases, the model accepts more training error in exchange for smaller coefficients. At `alpha=0`, the objective reduces to OLS.

### Why L2 helps with correlated predictors

After centering, the Ridge normal equation is

$$
\left(X_c^\top X_c+n\alpha I\right)\hat\beta
=X_c^\top y_c.
$$

The added $n\alpha I$ term makes weak or nearly collinear directions less dominant. This is why Ridge can remain stable when $X^\top X$ is poorly conditioned.

## Minimal runnable example

This example deliberately creates two almost-duplicate predictors so you can see the difference from OLS.

```python
import numpy as np
from statgpu.linear_model import LinearRegression, Ridge

rng = np.random.default_rng(0)
n = 400

shared = rng.normal(size=n)
X = np.column_stack([
    shared + 0.03 * rng.normal(size=n),
    shared + 0.03 * rng.normal(size=n),
    rng.normal(size=n),
])
y = (
    1.0
    + 1.2 * X[:, 0]
    + 1.2 * X[:, 1]
    + 0.5 * X[:, 2]
    + rng.normal(scale=0.8, size=n)
)

ols = LinearRegression(
    device="cpu",
    compute_inference=False,
).fit(X, y)

ridge = Ridge(
    alpha=0.2,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print("OLS coefficients:  ", np.round(ols.coef_, 3))
print("Ridge coefficients:", np.round(ridge.coef_, 3))
print("Ridge R²:", round(ridge.score(X, y), 3))
```

With the fixed seed above, the first two OLS coefficients are roughly `0.32` and `2.08`, while Ridge gives values near `1.09` and `1.09`. The exact point is not that Ridge discovers a uniquely “true” split; it is that the split becomes much less sensitive to the near-duplicate columns.

## How to read the result

- `coef_[j]` is the fitted coefficient after shrinkage. It is **not** an OLS coefficient with a cosmetic adjustment; the penalty changes the estimator itself.
- `intercept_` is the fitted intercept and is not penalized.
- `predict(X_new)` returns predicted continuous responses.
- `score(X, y)` returns $R^2$.
- A smaller coefficient does not mean the feature became less scientifically important; part of the change may be the regularization bias introduced for stability.

If coefficient uncertainty matters, construct the model with `compute_inference=True`. Ridge exposes standard errors, test statistics, p-values, and confidence intervals under the supported covariance choices, but inference is conditional on the chosen `alpha`.

## Key parameters and how to choose them

| Parameter | Default | How to think about it |
|---|---:|---|
| `alpha` | `1.0` | Main modeling choice. Larger values shrink coefficients more strongly. For predictive work, choose it with `RidgeCV` or another validation procedure rather than from training fit alone. |
| `fit_intercept` | `True` | Keep it unless theory fixes the intercept at zero or your design already includes one. |
| `device` | `"auto"` | Use `"cpu"` for small/medium problems and compatibility; use `"cuda"` or `"torch"` when the workload is large enough to justify GPU transfer and setup cost. |
| `compute_inference` | `True` | Disable when you only need prediction/coefficients and want to avoid inference work. |
| `cov_type` | `"nonrobust"` | Use HC variants for heteroskedasticity and HAC when ordered observations may be serially correlated. |
| `solver` | `"exact"` | The dense direct Ridge path and the normal default. Change it mainly for controlled numerical experiments or special workloads. |

### Scale your predictors

Regularization acts directly on coefficient magnitude. If one feature is measured in meters and another in micrometers, equal predictive effects can require very different coefficients and therefore receive different penalties.

For most regularized workflows, standardize continuous predictors before fitting unless the feature scaling itself is part of the intended model.

## Compare with nearby methods

| Method | What the penalty does | Exact zero coefficients? | Prefer it when |
|---|---|:---:|---|
| OLS / `LinearRegression` | no shrinkage | no | the design is stable and you want the unpenalized linear estimator |
| **Ridge** | L2 shrinkage | usually no | predictors are correlated and you want stable prediction while retaining all variables |
| [Lasso](lasso.md) | L1 shrinkage | yes | a sparse model / feature selection is important |
| [Elastic Net](elastic-net.md) | L1 + L2 | yes | you want sparsity but correlated predictors should be treated more stably than pure Lasso |

A useful mental model is:

```text
OLS        : fit only
Ridge      : fit + shrink
Lasso      : fit + shrink + select
Elastic Net: fit + shrink + select, with extra stability for correlated features
```

## CPU, GPU, and weighted fitting

The same public estimator supports NumPy CPU, CuPy CUDA, and Torch CUDA paths where available.

```python
from statgpu.linear_model import Ridge

model = Ridge(
    alpha=0.2,
    device="cuda",
    compute_inference=False,
).fit(X, y)
```

For analytic sample weights, call:

```python
weighted = Ridge(alpha=0.2).fit(X, y, sample_weight=w)
```

statgpu normalizes the weighted data-fit term by `sum(sample_weight)`. Multiplying every weight by the same positive constant therefore leaves the fitted Ridge model unchanged.

## Advanced: solver support

`Ridge` is the regularized linear wrapper whose public default is the model-specific `solver="exact"` path.

| `solver` value | CPU | CuPy / Torch | Intended use |
|---|:---:|:---:|---|
| `exact` (default) | yes | yes | Dense L2 solve; first choice for ordinary Ridge |
| `auto` | exact | Newton | Backend-aware automatic dispatch |
| `fista` / `fista_bb` | yes | yes | Iterative comparison or controlled optimization experiments |
| `newton` / `lbfgs` | yes | yes | Smooth-objective alternatives |
| `admm` | yes | yes | Experimental split formulation; uniform sample weights only |
| `irls` | no | no | Squared-error loss does not expose the IRLS contract |

`coordinate_descent`, quantile coordinate descent, and L-BFGS-B are not Ridge estimator choices. General update equations are documented in the [solver algorithms guide](../guides/solver-algorithms.md).

## Advanced: inference and objective scaling

Supported covariance choices are `nonrobust`, `hc0`, `hc1`, `hc2`, `hc3`, and `hac`. With `compute_inference=True`, the established reporting surface includes `_bse`, `_tvalues`, `_pvalues`, and `_conf_int` along with fit diagnostics such as `rsquared`, `rsquared_adj`, `fvalue`, `aic`, and `bic` when defined.

For weighted inference, the numerical design uses the same analytic-weight convention as fitting. Numerical covariance and reference-distribution calculations remain on the executed NumPy/CuPy/Torch backend before the public reporting arrays are snapshotted to NumPy.

### Comparing `alpha` with scikit-learn

statgpu uses the average-loss objective shown above. scikit-learn Ridge uses an unnormalized residual sum of squares. For coefficient comparisons:

- unweighted: `sklearn_alpha = n_samples * statgpu_alpha`;
- weighted: `sklearn_alpha = sample_weight.sum() * statgpu_alpha`.

Using the same numerical `alpha` in both libraries therefore compares different objectives.

## Common pitfalls

- **Do not choose `alpha` from training $R^2$ alone.** Training fit almost always prefers less regularization; validation is what measures the bias-variance trade-off you care about.
- **Do not interpret shrinkage as feature deletion.** Ridge normally keeps every coefficient nonzero.
- **Do not compare coefficient magnitudes before considering feature scale.** Standardization is often essential.
- **Do not copy `alpha` directly from another library.** Check that the objective normalization matches.
- **Regularization does not repair model misspecification.** It does not make a nonlinear, dependent, or confounded model scientifically valid.
- **Small p-values after choosing `alpha` are not a substitute for a complete model-selection argument.** Treat inference as conditional on the regularization choice unless your inferential procedure explicitly accounts for tuning.

## API and validation

Import path:

```python
from statgpu.linear_model import Ridge
```

The constructor also exposes advanced controls including `max_iter`, `tol`, `cpu_solver`, `lipschitz_L`, `hac_maxlags`, and `gpu_memory_cleanup`. They are deliberately placed late on this learner page because most users should first decide whether Ridge is the right model and how to choose `alpha`.

Internal consistency is tested against the average-loss closed form and the shared penalized-linear engine. Weighted fitting, exact/FISTA consistency, formula alignment, inference, RidgeCV final-refit behavior, and backend-native inference contracts are covered by the maintained test suite and physical-GPU acceptance checks.

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
