# Ridge

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/ridge.md)

## Overview

`Ridge` provides L2-regularized linear regression for a single continuous response, with coefficient inference and robust covariance options. It is used to stabilize prediction or coefficient estimates under multicollinearity. With positive alpha, coefficient intervals describe the penalized fit and do not automatically remove shrinkage bias or adjust for choosing alpha.

Unlike [Lasso](lasso.md), Ridge does not aim to make slopes exactly zero. Use it
when many weak or correlated predictors may help prediction; start with
[LinearRegression](linear-regression.md) when an unpenalized low-dimensional
model is appropriate. Neither shrinkage nor a small p-value establishes causality.

## Objective Function

For unweighted observations, statgpu minimizes the average-loss objective

$$
\min_{b,\beta}
\frac{1}{2n}\sum_{i=1}^n
\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\|\beta\|_2^2.
$$

With `sample_weight=w`, the data-fit term is normalized by the total weight:

$$
\min_{b,\beta}
\frac{1}{2\sum_i w_i}\sum_{i=1}^n
w_i\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\|\beta\|_2^2.
$$

The intercept is not penalized. Multiplying every sample weight by the same positive constant therefore leaves the fitted model unchanged.

## A complete CPU example

This example uses comparable-scale predictors, an illustrative fixed alpha, and
analytic training weights. The last 40 rows are held out before fitting.

<!-- learner-example: ridge-weighted-prediction -->
```python
import numpy as np
from statgpu.linear_model import Ridge
```

<a id="cpu-data"></a>

### Prepare data and training weights

Run the blocks in order in one session. `X` has shape `(160, 5)`, with observations in rows and predictors in columns; `y` is a `(160,)` continuous response. Each of the 120 training rows has one positive analytic weight. The 40 test rows are kept separate.

```python
rng = np.random.default_rng(64)
X = rng.normal(size=(160, 5))
y = 1.5 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.4, size=160)
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]
weights = np.linspace(0.5, 2.0, 120)
```

### Fit the weighted model

The weights enter the fitting loss; larger weights give observations more influence. HC3 also requests leverage-adjusted heteroskedasticity-robust covariance for the later interval step. It does not change the Ridge fit.

```python
model = Ridge(
    alpha=0.1, device="cpu", cov_type="hc3", compute_inference=True,
)
model.fit(X_train, y_train, sample_weight=weights)
```

### Predict and evaluate

The test score here is unweighted. Training weights are not automatically carried into a new evaluation.

```python
prediction = model.predict(X_test)
print("Slopes:", np.round(model.coef_, 3))
print("Test R2:", round(model.score(X_test, y_test), 3))
```

The slopes are approximately `[1.821, -0.916, 0.017, -0.010, -0.020]` and held-out R² is about `0.957`. `prediction` has shape `(40,)`. These are shrunken prediction coefficients, not evidence that each feature is significant.

### Inspect coefficient intervals

The enabled inference reports one interval per fitted parameter, with the intercept first. These intervals describe coefficients rather than future responses; they do not correct shrinkage bias or tuning uncertainty.

```python
print("Interval shape:", model._conf_int.shape)
```
<!-- example-end: ridge-weighted-prediction -->

This prints `(6, 2)`: six parameters, each with a lower and upper bound.

Choose alpha using training-only validation, keeping the final test set separate.
Learn feature scaling inside each training fold, since Ridge penalizes coefficients
in their chosen units. The alpha value is illustrative; choose it for your data.

## Path

`statgpu.linear_model.Ridge`

## Estimating Equation

With an intercept, center the data using the corresponding ordinary or weighted means. The first-order condition is

$$
\left(X_c^\top W X_c + \alpha\,s_w I\right)\hat\beta
= X_c^\top W y_c,
$$

Here $W=I$ and $s_w=n$ without sample weights, while $W=\operatorname{diag}(w)$ and $s_w=\sum_iw_i$ for weighted fitting. `fit_intercept=False` instead uses the original X and y in this equation and fixes b=0.

`Ridge` defaults to `solver="exact"`. The same objective scale is used by the exact and FISTA paths, by `PenalizedLinearRegression(penalty="l2")`, and by `RidgeCV`.

scikit-learn uses an unnormalized residual sum of squares. For coefficient comparisons, use

- unweighted: `sklearn_alpha = n_samples * statgpu_alpha`;
- weighted: `sklearn_alpha = sample_weight.sum() * statgpu_alpha`.

Comparing the two libraries with the same numerical `alpha` compares different objectives.

<a id="large-feature-offsets"></a>

## Large feature offsets

The optimized CPU `solver="exact"` path with an intercept computes centered
cross-products by subtracting large raw moments. When feature means are much
larger than their variation, cancellation can give badly wrong coefficients
and predictions, despite finite output and a completed fit. Weighted and
unweighted fits are affected; enabling inference does not repair the fit.
For example, translating otherwise ordinary predictors by `1e8` can change
an estimated positive slope near 0.91 into a negative one near −0.40.

Subtract an origin learned from training rows before fitting and reuse that
same origin for every prediction. Do not independently center the test set.
The following example retains the fitted intercept, so this translation leaves
the Ridge statistical objective unchanged:

<!-- learner-example: ridge-training-origin -->
```python
import numpy as np
from statgpu.linear_model import Ridge
```

This separate dataset deliberately adds a large origin to three predictors. `X` has shape `(80, 3)` and `y` has shape `(80,)`; the last 20 rows are held out.

```python
rng = np.random.default_rng(113)
variation = rng.normal(size=(80, 3))
X = variation + 1e8
y = 0.4 + variation @ np.array([1.0, -0.5, 0.3]) + rng.normal(scale=0.1, size=80)
X_train, X_test = X[:60], X[60:]
y_train, y_test = y[:60], y[60:]
```

Learn the origin on training rows, then fit to the translated features. The intercept is still estimated.

```python
origin = X_train.mean(axis=0)
model = Ridge(alpha=0.1, device="cpu", compute_inference=True)
model.fit(X_train - origin, y_train)
```

Apply the same translation when predicting. The optional algebraic intercept conversion below expresses the equation in the original units; it is not needed for prediction.

```python
prediction = model.predict(X_test - origin)
original_intercept = model.intercept_ - origin @ model.coef_
print(np.round(model.coef_, 3))
print(round(float(np.mean((prediction - y_test)**2)), 3))
```
<!-- example-end: ridge-training-origin -->

The coefficients are approximately `[0.919, -0.430, 0.256]`; held-out MSE is
about `0.033`. `original_intercept` maps the fitted equation back to the original
feature coordinates, but evaluate predictions through the centered model to
avoid subtracting large terms. With training weights, a weighted training mean
is a suitable origin. Keep the weights and alpha unchanged. Inference for the
intercept now refers to the response at this origin; its interval is not an
interval for `original_intercept`. The unmodified FISTA fit also avoids this
particular raw-moment coefficient calculation, but still requires convergence
checks and does not make all large-offset numerical calculations safe.

## Covariance/Inference

- `cov_type="nonrobust"`: classical ridge covariance.
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`: sandwich-style robust covariance variants.
- `cov_type="hac"`: Newey-West (Bartlett) covariance with optional `hac_maxlags`.
- `compute_inference=True` returns `_bse`, `_tvalues`, `_pvalues`, `_conf_int`.
- Weighted inference uses the weighted design `[sqrt(w), sqrt(w) * X]`, so the intercept column, residuals, bread, and meat follow the same weighting convention as estimation.

The inference normal equations use the same average-loss penalty mapping as fitting: the numerical ridge term is `n * alpha` without weights and `sample_weight.sum() * alpha` with analytic weights. The intercept remains unpenalized.

Numerical covariance, standard errors, reference-distribution p-values and
intervals run on the fitted NumPy/CuPy/Torch backend. Reporting arrays are
converted to NumPy afterwards; this does not indicate CPU numerical fallback.
Nonrobust intervals use a Student-t reference; HC/HAC intervals use a normal
reference despite the `_tvalues` field name. There is no general guarantee that
these plug-in intervals cover an unpenalized population coefficient after
shrinkage or tuning on the same data.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L2 regularization strength on the average-loss scale |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | Shared worker configuration; this wrapper does not promise parallel fitting |
| `compute_inference` | `True` | Whether to compute inference stats (SE/t/p/CI) |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag for `cov_type="hac"`; default follows a Newey-West-style heuristic |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory cleanup after each fit |
| `solver` | `"exact"` | Exact L2 solution by default; `fista` uses the same objective |
| `max_iter` | `1000` | Iteration budget for iterative solvers; exact fitting needs one solve |
| `tol` | `1e-4` | Iterative convergence tolerance |
| `cpu_solver` | `"fista"` | Deprecated compatibility argument; use `solver` to select the algorithm |
| `lipschitz_L` | `None` | Optional smooth-gradient Lipschitz bound for compatible iterative solvers |

## Optional GPU use

Use `device="cuda"` for CuPy CUDA or `device="torch"` for Torch CUDA in the
complete CPU example once that backend is installed and usable. An unavailable
explicit backend raises; only `auto` may choose another available backend.
See [device and memory](../guides/device-and-memory.md).

## API reference

The [complete Ridge method reference](../reference/linear-model-api.md#ridge)
includes fit arguments, formulas, output placement, weighted scoring, diagnostics,
and inherited helpers. The parameter table above covers every constructor control.
Ridge accepts only a single response; it does not share LinearRegression's
multi-output API. Its prediction defaults to a NumPy array even after GPU fitting;
use `predict(X, return_cpu=False)` for backend-native output.

## strict/approx difference

No separate public approximate mode is exposed. Exact and FISTA fits optimize
the same objective; their difference is numerical rather than a different
statistical model. Use an explicit device when execution placement matters;
see [device and memory](../guides/device-and-memory.md).

## Outputs

- Coefficients: `intercept_`, `coef_`
- Inference: `_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- Diagnostics: `rsquared`, `rsquared_adj`, `fvalue`, `aic`, `bic`
- Methods: `fit`, `predict`, `score`, `summary`

## FAQ

- How should `alpha` be chosen? Use `RidgeCV` or a task-specific log grid on statgpu's average-loss scale.
- Why does the same `alpha` differ from sklearn? The residual term has a different normalization; apply the mapping above.
- Does rescaling all sample weights change the model? No. The weighted loss is divided by `sum(sample_weight)`.
- When should I set `hac_maxlags`? When using `cov_type="hac"` with time dependence; otherwise leave the default.
- Are GPU inference arrays exposed as CuPy/Torch objects? No. Numerical inference stays backend-native, but the established public reporting attributes remain NumPy snapshots after numerical inference completes.

Complete [RidgeCV controls and result schema](../reference/linear-model-api.md#ridgecv)
and a [standalone CPU tuning example](../reference/linear-model-api.md#ridgecv-and-lassocv-cpu-example)
are available in the API reference. Direct and CV constructor controls differ.

For non-complement custom training subsets, read the
[RidgeCV restriction](../reference/linear-model-api.md#custom-ridgecv-training-subsets)
and use an external CV loop.

## External Validation

When comparing with sklearn, apply the unweighted or weighted alpha mapping
above rather than using the same numerical alpha. Keep feature scaling,
intercept treatment and weights identical. Covariance and interval comparisons
also require the same ridge penalty, degrees of freedom, covariance choice and
reference distribution. Agreement for one solver, dtype or device does not
establish the accuracy or speed of another.

Contributors can consult the [validation reference](../../../dev/references/model-validation.md#ridge).

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
