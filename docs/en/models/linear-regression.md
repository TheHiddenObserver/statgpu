# LinearRegression

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/linear-regression.md)

## What problem does it solve?

Use ordinary least squares (OLS) when the outcome is continuous and you want a
simple additive model for prediction or for describing associations. Each
coefficient describes the change in the fitted outcome for a one-unit change in
one predictor, holding the other included predictors fixed. An association is
not automatically a causal effect.

`LinearRegression` fits an unpenalized model. Start here for a modest number of
predictors with a plausible linear relationship. For strongly correlated or
numerous predictors, consider [Ridge](ridge.md) for shrinkage or
[Lasso](lasso.md) for sparse coefficients. Binary outcomes need a classification
model rather than this continuous-outcome model.

## Model and intuition

With an intercept, the fitted mean is `intercept_ + X @ coef_`. OLS chooses the
line or plane that minimizes squared prediction errors:

$$
\min_{b,\beta}\sum_{i=1}^n (y_i-b-x_i^\top\beta)^2.
$$

Here n is the observation count, $x_i$ the p-vector of predictors, $b$ the intercept, and $\beta$ the p-vector of slopes. The intercept is unpenalized; `fit_intercept=False` fixes it to zero.

## A complete CPU example

After installing statgpu, this example needs only NumPy and the package. It
creates three predictors, reserves 60 observations for evaluation, and fits on
180 observations. The true coefficients are `[2, -1, 0]`.

<!-- learner-example: linear-prediction -->
```python
import numpy as np
from statgpu import LinearRegression
```

<a id="cpu-data"></a>

### Prepare the observations

Run these blocks in order in one session. `X` is a `(240, 3)` matrix: one observation per row and one predictor per column. `y` is a `(240,)` continuous response. Split the rows before fitting.

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(240, 3))
y = 1.5 + X @ np.array([2.0, -1.0, 0.0]) + rng.normal(scale=0.5, size=240)
X_train, X_test = X[:180], X[180:]
y_train, y_test = y[:180], y[180:]
```

### Fit the model

`fit` learns the intercept and slopes from the training rows. We also request HC3 coefficient uncertainty for the later interval step. HC3 adjusts covariance for heteroskedasticity and leverage; it does not change the OLS coefficients. Use `compute_inference=False` if you only need predictions.

```python
model = LinearRegression(device="cpu", cov_type="hc3", compute_inference=True)
model.fit(X_train, y_train)
```

### Predict on held-out rows

Use the same three feature columns in the training order. There is one predicted response for each of the 60 test rows.

```python
prediction = model.predict(X_test)
print("Intercept:", round(model.intercept_, 3))
print("Coefficients:", np.round(model.coef_, 3))
print("Prediction shape:", prediction.shape)
print("Test R2:", round(model.score(X_test, y_test), 3))
```

Expected rounded output:

```text
Intercept: 1.484
Coefficients: [ 2.049 -0.966 -0.088]
Prediction shape: (60,)
Test R2: 0.934
```

The first two slopes recover the positive and negative associations. Test
$R^2\approx0.934$ means the test residual sum of squares is about 6.6% of the
sum of squared deviations from the test outcome mean. It is not a probability
that the model is correct, and $R^2$ can be negative on new data.

### Read coefficient uncertainty

Prediction accuracy and coefficient uncertainty answer different questions. The enabled HC3 calculation gives marginal 95% coefficient intervals:

```python
print("95% coefficient intervals:\n", np.round(model._conf_int, 3))
```
<!-- example-end: linear-prediction -->

```text
95% coefficient intervals:
 [[ 1.410  1.557]
  [ 1.971  2.128]
  [-1.038 -0.894]
  [-0.167 -0.008]]
```

The interval rows are **intercept first, then input columns in order**. Notice
that the third interval happens to exclude zero even though its true slope is
zero. A nominal 95% interval is not guaranteed to cover the truth in every
sample; checking many coefficients adds a multiple-testing concern. HC3 changes
the covariance estimate, not the fitted OLS coefficients. These are marginal
coefficient intervals, not prediction intervals for new observations.

## Inputs and outputs

- Fitting array input: finite numeric `X` with shape `(n_samples, n_features)`
  and `y` with shape `(n_samples,)` or `(n_samples, n_targets)`. `fit` flattens
  a one-column `y` to single-output. Keep prediction columns in the training order.
- `fit(X, y, sample_weight=None)` returns the fitted estimator. Weights must be
  finite, nonnegative, have length `n_samples`, and have positive sum; they
  change the fitting objective, not just the reported errors.
- Formula input: `fit(formula="y ~ x1 + x2", data=df)` is an alternative to
  arrays and requires the optional pandas/patsy dependencies. Formula syntax
  controls the intercept (`~ 0 + ...` removes it); prediction from a DataFrame
  rebuilds the stored design. Account for rows dropped during formula parsing.
- Single-output `coef_` has shape `(n_features,)`, `intercept_` is a scalar,
  and `predict(X_new)` has shape `(n_new,)`.
- Multi-output `coef_` is `(n_targets, n_features)`, `intercept_` is
  `(n_targets,)`, and predictions are `(n_new, n_targets)`. `score` returns the
  mean per-target $R^2$, not an array of scores.

For a single target, pass a one-dimensional response to `score(X, y)`.
Unlike `fit`, `score` currently does not flatten `(n_samples, 1)` responses;
subtracting the one-dimensional predictions then broadcasts into a matrix and
can silently produce an incorrect $R^2$. Flatten only a single-target response
before scoring; do not flatten a genuine multi-target array.

The following independent six-row example isolates that shape issue; its noiseless response makes the correct score easy to check.

<!-- learner-example: linear-column-target -->
```python
import numpy as np
from statgpu.linear_model import LinearRegression

X = np.arange(6.0)[:, None]
y_column = 2.0 + 3.0 * X
model = LinearRegression(device="cpu", compute_inference=False).fit(X, y_column)
r2 = model.score(X, y_column.ravel())
print("R2:", round(float(r2), 3))
```
<!-- example-end: linear-column-target -->

This prints `R2: 1.0`: the line fits these noiseless observations exactly.

### Missing formula-prediction rows

Formula prediction currently drops rows with missing predictors and returns a
shorter unlabelled array. Resolve missing values and verify prediction length
before aligning results to the original observations. `score` is also unsafe
on such a query: one retained prediction can broadcast across several responses
and produce a finite but wrong R². Flattening y does not repair this separate
row-alignment problem. Filter deliberately with a retained row index, or reject
the query; see the [complete missing-row warning](../reference/linear-model-api.md#missing-prediction-rows-in-ordinary-glms).

## Choosing parameters and covariance

| Parameter | Default | How to choose |
|---|---|---|
| `fit_intercept` | `True` | Usually keep it; omit only when a zero intercept is justified. Do not add a second constant column. Formula syntax controls formula fits. |
| `device` | `"auto"` | Use `"cpu"` for these examples; `"cuda"` requests CuPy CUDA and `"torch"` requests Torch CUDA. Only `"auto"` may select another available backend. |
| `n_jobs` | `None` | Accepted as shared estimator configuration; does not control parallel fitting in this class. |
| `compute_inference` | `True` | Set `False` when only fitting/prediction is needed. `summary()` then raises. |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory cleanup; see the [device and memory guide](../guides/device-and-memory.md). |
| `cov_type` | `"nonrobust"` | Choose based on the error structure, using the table below. |
| `hac_maxlags` | `None` | For HAC, choose a nonnegative lag count using the sampling interval and dependence horizon. The automatic rule is sample-size based. |

| `cov_type` | Interpretation |
|---|---|
| `"nonrobust"` | Classical covariance; t-reference inference. Homoskedastic independent errors underlie the conventional OLS uncertainty calculation. |
| `"hc0"` | White heteroskedasticity-robust sandwich covariance. |
| `"hc1"` | HC0 with a residual-degrees-of-freedom correction. |
| `"hc2"` | Leverage-adjusted squared residuals. |
| `"hc3"` | Stronger leverage adjustment, useful when influential observations are a concern. |
| `"hac"` | Newey-West covariance with Bartlett weights for heteroskedasticity and serial correlation. Preserve meaningful observation/time order. |

HC/HAC coefficient p-values and intervals use a **normal reference**, despite
the historical `_tvalues` attribute name. With `hac_maxlags=None`, the lag rule
is `floor(4 * (n / 100)**(2 / 9))`, bounded to `[0, n - 1]`. Robust covariance
cannot repair omitted variables, nonlinear misspecification, or dependence
outside the chosen covariance model. The overall `fvalue`/`f_pvalue` diagnostic
is the residual-based F statistic, not a robust joint Wald test.

### How the covariance is constructed

Let $D=[\mathbf1,X]$ and $\theta=(b,\beta^\top)^\top$ when fitting an intercept; otherwise $D=X$ and $\theta=\beta$. With $W=\operatorname{diag}(w_i)$ (identity without weights), the weighted objective and normal equations are

$$
\min_\theta (y-D\theta)^\top W(y-D\theta),\qquad
D^\top W(y-D\hat\theta)=0.
$$

For a classical model with full-rank weighted design, writing
$r=\operatorname{rank}(W^{1/2}D)$,

$$
\hat\sigma^2=\frac{(y-D\hat\theta)^\top W(y-D\hat\theta)}{n-r},\qquad
\widehat{\operatorname{Var}}(\hat\theta)=\hat\sigma^2(D^\top WD)^{-1}.
$$

Standard error j is the square root of covariance diagonal j. A marginal interval is $\hat\theta_j\pm c\,\mathrm{SE}_j$, with a t critical value for classical inference and a normal critical value for HC/HAC. Robust choices replace the covariance construction, not the OLS/WLS fit. The rank reduces to that of D without weights; zero-weight rows can reduce the weighted design rank. These inverse expressions require a full-rank weighted design and positive residual degrees of freedom; users should not manually invert the Gram matrix to fit the estimator.

## Pitfalls and support boundaries

- Check residual patterns and evaluate predictions on held-out observations.
  Training $R^2$ alone is not a generalization check.
- Collinear columns make individual slopes poorly identified. `rank_` records
  fitted design rank; a numerical solution does not establish meaningful
  individual effects. Nonpositive residual degrees of freedom make inference
  unavailable, and `summary()` raises rather than supplying usable intervals.
- CPU supports multi-output inference. Multi-output fitting on CuPy/Torch CUDA
  requires `compute_inference=False`; requesting GPU multi-output inference
  raises `NotImplementedError`. `summary()` is single-output only, and
  multi-output `aic`/`bic` are `None`.
- There is no separate public strict/approx inference switch. Device requests
  and unsupported combinations still fail explicitly. Small floating-point
  differences across supported backends are possible.
- `score(X, y)` is unweighted even after a weighted fit; it does not accept a
  `sample_weight` argument. The training `rsquared` property uses fitted
  weights when provided.
- Ordinary coefficient intervals do not account for selecting predictors using
  the same responses. See [feature selection](feature-selection.md).

### Diagnostic limitations with weights or multiple targets

For a single weighted target, `llf` currently uses only the weighted residual sum of squares in an ordinary Gaussian expression. It omits the `0.5 * sum(log(sample_weight))` normalization required for the likelihood of independent Gaussian observations with variance proportional to `1 / sample_weight` (strictly positive weights). Consequently, multiplying every weight by a constant changes `llf`, `aic`, and `bic`, even though the fitted coefficients and classical standard errors are unchanged. Do not compare these values across weight normalizations or treat them as the normalized WLS likelihood. Comparisons among models using exactly the same rows and weights share the omitted constant.

Multi-output `fvalue`, `f_pvalue`, and `llf` are not joint multivariate tests or likelihoods. With weights, accessing `fvalue` or `f_pvalue` currently raises `TypeError`; without weights, their pooled calculation is not an individual-target F test. Fit each target separately for its diagnostics. The training `rsquared` uses centered total variation even without an intercept, so it is not the uncentered R² used by some no-intercept references.

## API inventory and advanced reference

The reader-facing [LinearRegression API reference](../reference/linear-model-api.md#linearregression) covers every constructor/method argument, default, shape and return. [Shared estimator helpers](../reference/estimator-api.md#inference-helpers) document inherited inference methods separately from module functions.

Import from `statgpu` or `statgpu.linear_model`. All constructor parameters are
listed above. The complete implementation/docstrings are in
[`LinearRegression`](../../../statgpu/linear_model/wrappers/_linear.py), with
inherited methods in [`BaseEstimator`](../../../statgpu/_base.py).

| Surface | Available names / contract |
|---|---|
| Fitting and reporting | `fit(X=None, y=None, sample_weight=None, formula=None, data=None)`, `predict(X)`, `score(X, y)`, `summary()` (prints a table) |
| Parameter management | `get_params(deep=True)`, `set_params(**params)` |
| Fitted estimates | `coef_`, `intercept_`, `rank_` |
| Diagnostics | `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`, `llf`, `aic`, `bic` |
| Inference arrays | `_bse`, `_tvalues`, `_pvalues`, `_conf_int`; populated only when inference is available |
| Inherited inference helpers | `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, `permutation_test`; see [inference API](../guides/inference-api.md) and base-method docstrings for their full arguments |

For single-output inference with `k` fitted parameters, the first three arrays
have shape `(k,)`, and `_conf_int` is `(k, 2)`. CPU multi-output inference uses
`(k, n_targets)` and `(k, n_targets, 2)`, respectively. The leading parameter
is the intercept when fitted. The inference helper methods are not automatic
post-selection corrections.

### External validation

When comparing with `statsmodels.OLS`, use the same observation rows, design
matrix and intercept convention. For HC/HAC inference, also align the covariance
type, small-sample correction, lag settings and normal-versus-t reference
distribution; matching coefficient estimates alone does not establish matching
standard errors or intervals. Weighted fits should be compared with the
corresponding WLS model using the same weights.

Contributors can consult the [validation reference](../../../dev/references/model-validation.md#linear-regression).

## References

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817–838. [DOI](https://doi.org/10.2307/1912934)
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325. [DOI](https://doi.org/10.1016/0304-4076(85)90158-7)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. [DOI](https://doi.org/10.2307/1913610)
