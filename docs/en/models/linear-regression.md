# LinearRegression

> Language: English
> Last updated: 2026-09-03
> Switch: [简体中文](../../cn/models/linear-regression.md)

## What problem does it solve?

`LinearRegression` estimates how a continuous outcome changes with one or more predictors. It is ordinary least squares (OLS) with a unified CPU/GPU interface and optional classical, heteroskedasticity-robust, or autocorrelation-robust inference.

Typical questions include:

- How much does the expected outcome change when one feature increases by one unit?
- How well does a linear combination of the features predict a continuous target?
- Which coefficients remain distinguishable from zero after accounting for uncertainty?

## When to use it

Use linear regression when the response is continuous and a linear conditional-mean relationship is a reasonable first approximation. It is especially useful as an interpretable baseline.

Choose another method when:

- the response is binary, a count, positive and strongly skewed, or a survival time—start with a [generalized linear model](generalized-linear-model.md) or the Cox model;
- important relationships are nonlinear unless you add suitable transformed features;
- you need automatic feature selection—consider Ridge, Lasso, Elastic Net, or SCAD;
- your goal is a causal effect but the study design does not justify a causal interpretation.

## Intuition

OLS draws the hyperplane whose squared vertical distances from the observed responses are as small as possible. A coefficient is a partial association: it describes the expected change in `y` for a one-unit change in that feature while the other included features are held fixed.

## Model and assumptions

With an intercept, the model is

$$
y_i = \beta_0 + x_i^\top\beta + \varepsilon_i,
$$

and OLS estimates

$$
(\hat\beta_0,\hat\beta)
=
\arg\min_{\beta_0,\beta}
\sum_{i=1}^{n}
\left(y_i-\beta_0-x_i^\top\beta\right)^2.
$$

Here, $n$ is the number of observations, $x_i$ is the feature vector for observation $i$, and $\varepsilon_i$ is the part not explained by the included predictors.

For coefficient interpretation and conventional inference, check that:

- the conditional mean is adequately linear;
- observations and features are measured on meaningful scales;
- the design matrix is not rank deficient;
- errors have conditional mean zero;
- the selected covariance estimator matches the error structure.

Constant error variance is required only for classical (`nonrobust`) standard errors, not for HC or HAC covariance estimates.

## Minimal runnable example

This example creates its own data, so it can be copied and run as-is.

```python
import numpy as np
from statgpu.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
true_coef = np.array([2.0, -1.0, 0.5])
y = 1.5 + X @ true_coef + rng.normal(scale=0.7, size=500)

model = LinearRegression(
    device="cpu",
    cov_type="hc1",
    compute_inference=True,
).fit(X, y)

print("intercept:", model.intercept_)
print("coefficients:", model.coef_)
print("R²:", model.rsquared)
print("standard errors:", model._bse)
print("p-values:", model._pvalues)
print("first predictions:", model.predict(X[:3]))
model.summary()
```

The fitted coefficients should be close to `[2.0, -1.0, 0.5]` and the intercept close to `1.5`. Exact values vary because the response includes random noise.

`summary()` prints the fit statistics and coefficient table in one place. For this seeded example it begins like this:

```text
Linear Regression Results
Covariance Type:                        hc1
No. Observations:                       500
R-squared:                           0.9064
Adj. R-squared:                      0.9058
                        coef      std err          t      P>|t|
(Intercept)           1.4562       0.0327     44.512     0.0000
x1                    2.0288       0.0314     64.660     0.0000
x2                   -0.9906       0.0349    -28.382     0.0000
x3                    0.5440       0.0312     17.422     0.0000
```

`summary()` requires a fitted single-output model with `compute_inference=True`. The complete table also includes the F statistic, log-likelihood, AIC, BIC, and 95% confidence intervals.

## Formula and DataFrame example

Install the optional parser, then fit with column names:

```bash
pip install "statgpu[formula]"
```

```python
import pandas as pd
from statgpu.linear_model import LinearRegression

frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
frame["y"] = y

formula_model = LinearRegression(
    device="cpu",
    cov_type="hc3",
).fit(
    formula="y ~ x1 + x2 + x3",
    data=frame,
)

formula_model.summary()
pred = formula_model.predict(frame.iloc[:3])
```

The stored Patsy design information keeps categorical encodings and interaction columns aligned at prediction time. See the [Formula interface and support matrix](../guides/formula-interface.md) before assuming another estimator accepts `formula=`.

## How to read the result

- `coef_[j]` is the estimated change in the expected response for a one-unit increase in feature `j`, holding the other features fixed.
- `intercept_` is the expected response when every feature equals zero. It is useful only when that reference point is meaningful.
- `rsquared` is the fraction of sample variation explained by the fitted linear model; a high value does not establish causality or guarantee good out-of-sample predictions.
- `_bse`, `_tvalues`, `_pvalues`, and `_conf_int` describe coefficient uncertainty when `compute_inference=True`. With an intercept, inference arrays list the intercept first.
- `predict(X_new)` returns fitted outcomes; `score(X, y)` returns $R^2$.

## Key parameters and how to choose them

| Parameter | Default | Guidance |
|---|---:|---|
| `fit_intercept` | `True` | Keep it unless the scientific model genuinely passes through the origin or the design already contains an intercept |
| `device` | `"auto"` | Use `"cpu"` for small data and compatibility; use `"cuda"` or `"torch"` when the workload and installed backend justify transfer overhead |
| `compute_inference` | `True` | Disable it when you only need coefficients or predictions and want to avoid inference cost |
| `cov_type` | `"nonrobust"` | Use `"hc1"` or `"hc3"` for heteroskedastic data; use `"hac"` for ordered observations with serial correlation |
| `hac_maxlags` | `None` | Set the largest scientifically relevant lag for HAC; the default uses a sample-size heuristic |
| `gpu_memory_cleanup` | `False` | Enable only when returning cached CuPy memory after each fit is more important than repeated-fit speed |

## Compare with alternatives

| Method | Prefer it when |
|---|---|
| Ridge | predictors are strongly correlated and shrinkage may improve prediction |
| Lasso / Elastic Net | you need sparse coefficients or automatic feature selection |
| GLM | the response distribution is not well represented by a Gaussian continuous model |
| Robust regression | a small number of extreme outcomes dominate OLS |
| Panel OLS | repeated observations share entity or time effects |

## CPU, GPU, and inference support

The same estimator can fit on NumPy/CPU, CuPy/CUDA, or Torch where supported:

```python
gpu_model = LinearRegression(
    device="cuda",
    cov_type="hc1",
    compute_inference=True,
).fit(X, y)
```

Supported covariance choices are `nonrobust`, `hc0`, `hc1`, `hc2`, `hc3`, and `hac`. CPU/GPU results can differ slightly because of floating-point order and backend linear algebra. There is no separate public approximate-inference mode.

For the exact covariance formulas, assumptions, complete parameter list, output shapes, and backend behavior, continue to [Linear regression inference and complete API](linear-regression-inference.md). Source-level acceleration details are separated into the [CPU/GPU implementation guide](../guides/acceleration-internals.md).

## Common pitfalls

- Standardize or otherwise rescale features when coefficient magnitude comparisons matter.
- Do not interpret a small p-value as practical importance.
- Do not use HC standard errors to fix nonlinearity, omitted variables, or dependence; they only change uncertainty estimates.
- Use HAC only when observation order has a real meaning, and sort the data before fitting.
- Inspect residuals and validate predictions on held-out data.

## API and validation

Import path: `statgpu.linear_model.LinearRegression`

This beginner page intentionally covers the parameters used in the normal workflow. The [complete API page](linear-regression-inference.md#complete-api-reference) inventories every constructor and fit parameter, fitted statistic, inference array, method, failure condition, and multi-output shape.

Multi-output targets are supported, but `summary()` is single-output only. External consistency checks against `statsmodels.OLS` are in `dev/tests/test_external_consistency.py`.

## References

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
