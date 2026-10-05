# Nonparametric Methods

> Language: English  
> Last updated: 2026-10-05  
> This page: Nonparametric overview  
> Switch: [Chinese](../../cn/models/nonparametric.md)

## Choose the question before the smoother

Nonparametric does not mean assumption-free. Kernel methods borrow information from nearby observations; the bandwidth defines how wide that neighborhood is.

- **Where are observations concentrated?** Use kernel density estimation (KDE), with samples `X` and no response `y`.
- **How does a continuous response vary with predictors?** Use kernel regression, with paired `X, y`. Nadaraya–Watson (`"nw"`) takes a local weighted average; `"local_linear"` fits a line in each neighborhood and can reduce boundary bias.
- **Do you need an additive curve for each feature?** See [GAM](semiparametric.md). For other approaches, see [kernel ridge regression](kernel-methods.md) and [spline bases](splines.md).

This page is a starting workflow for KDE and kernel regression, not a catalog of all nonparametric algorithms. These methods are most useful in low-dimensional continuous data. Sparse neighborhoods, many predictors, and extrapolation deserve particular care.

## What is being estimated?

For equally weighted, one-dimensional data and absolute bandwidth $h$, KDE estimates a density:

$$
\hat f(x)=\frac{1}{nh}\sum_{i=1}^{n}K\!\left(\frac{x-X_i}{h}\right).
$$

A density is not the probability of observing exactly `x`. It can exceed 1; probabilities come from integrating over a region. `pdf`/`predict` return density, while `logpdf`/`score_samples` return its logarithm. KDE `score(X)` is the **mean log density** of those observations, not classification accuracy or $R^2$.

Nadaraya–Watson estimates the conditional mean:

$$
\hat m(x)=\frac{\sum_i w_i K_H(x-X_i)y_i}{\sum_i w_i K_H(x-X_i)}.
$$

Here $w_i\ge0$ are normalized observation weights, $\sum_i w_i=1$; $H$ is the
positive-definite bandwidth matrix. Define the scaled kernel by

$$
K_H(u)=|H|^{-1/2}K(H^{-1/2}u),\qquad
\widehat f(x)=\sum_i w_iK_H(x-X_i).
$$

For a $p$-dimensional Gaussian kernel,
$K(v)=(2\pi)^{-p/2}\exp(-v^\top v/2)$. The weighted formula reduces to the
one-dimensional expression above for equal weights and $H=h^2$.
Local-linear regression chooses a local intercept $a$ and slope vector $b$:

$$
(\widehat a(x),\widehat b(x))=
\arg\min_{a,b}\sum_i w_iK_H(x-X_i)
\{y_i-a-b^\top(X_i-x)\}^2,\qquad \widehat m(x)=\widehat a(x).
$$

The fit is repeated at each query; neither regression method produces one global
slope vector or coefficient p-values. Singular local systems can use the
stabilization/NW fallback described below.

## CPU example 1: fit and evaluate a density

All CPU examples are standalone, seeded, and explicitly select NumPy. Reuse a fitted object for repeated evaluation; the one-shot helper is convenient but fits again on every call.

Here `bandwidth=0.35` is a dimensionless **bandwidth factor**, not the absolute bandwidth $h$ in the formula above. For one-dimensional equally weighted data, $h$ is approximately `0.35` times the training sample standard deviation; see “Bandwidth, kernels, and tuning boundaries” below for details.

<!-- example: kde-cpu -->
```python
import numpy as np
from statgpu.nonparametric import fit_kde, kde_pdf

rng = np.random.default_rng(42)
x_train = rng.normal(size=300)
x_test = rng.normal(size=80)
grid = np.linspace(-4, 4, 201)

kde = fit_kde(x_train, bandwidth=0.35, kernel="gaussian", backend="numpy")
density = kde.pdf(grid)
log_density = kde.score_samples(grid)
one_shot = kde_pdf(x_train, grid, bandwidth=0.35, backend="numpy")
mass_on_grid = np.sum((density[1:] + density[:-1]) * np.diff(grid) / 2)
print(density.shape)
print(f"Mass on grid: {mass_on_grid:.4f}")
print(f"Held-out mean log density: {kde.score(x_test):.4f}")
```

Rounded output is `(201,)`, grid mass `1.0000`, and held-out mean log density `-1.4740`. `density` and `one_shot` agree; `log_density` agrees with `np.log(density)` here. The integral is over a wide finite grid, not a guarantee of exact unit mass on every chosen interval. Gaussian kernels have tails outside the observed range.

When choosing bandwidth, compare held-out mean log density on the **same held-out points and measurement scale**; higher is better. Do not tune repeatedly on the final test set. A larger bandwidth merges small features; a smaller bandwidth reveals detail but may mainly reveal sampling noise.

## CPU example 2: predict a response

This synthetic example has a known nonlinear mean. The test responses include fresh noise, so zero test error is not expected. The absolute bandwidth is specified in the predictor's original units.

<!-- example: kernel-regression-cpu -->
```python
import numpy as np
from statgpu.nonparametric import KernelRegressionRegressor, kernel_regression_predict

rng = np.random.default_rng(42)
x_train = rng.uniform(-2, 2, 160)
y_train = np.sin(2 * x_train) + rng.normal(0, 0.1, 160)
x_test = np.linspace(-1.8, 1.8, 61)
mean_test = np.sin(2 * x_test)
y_test = mean_test + rng.normal(0, 0.1, x_test.size)

regressor = KernelRegressionRegressor(
    regression="local_linear", kernel="gaussian",
    kernel_metric="diagonal", bandwidth_per_feature=[0.18],
    backend="numpy", device="cpu",
).fit(x_train, y_train)
prediction = regressor.predict(x_test)
one_shot = kernel_regression_predict(
    x_train, y_train, x_test, regression="local_linear",
    kernel_metric="diagonal", bandwidth_per_feature=[0.18], backend="numpy",
)
mse = np.mean((prediction - y_test) ** 2)
baseline_mse = np.mean((y_train.mean() - y_test) ** 2)
print(prediction.shape)
print(f"Test MSE: {mse:.4f}; mean baseline: {baseline_mse:.4f}")
print(f"Test R2: {regressor.score(x_test, y_test):.4f}")
```

Rounded output is `(61,)`, test MSE `0.0126` versus a mean baseline of `0.4690`, and test $R^2$ `0.9730`. The fitted object and one-shot predictions agree. Here $R^2$ describes held-out response prediction; it is unrelated to KDE's mean-log-density score. For multi-target regression, this implementation's `score` flattens all targets before computing one $R^2$; evaluate each target separately when their scales differ.

The bandwidth `0.18` is an illustrative preset, not a universally optimal choice. In real data, split before selecting bandwidth, transformations, or kernel settings. Compare candidates on validation folds, refit the chosen configuration on training data, and evaluate the final test set once.

## CPU example 3: pointwise density intervals

Use independent, equally weighted observations for this introductory bootstrap. The numeric bandwidth **factor** is held fixed at `0.4`; each resample still has its own sample covariance, so its absolute bandwidth changes. The 100 resamples keep the demo small; use enough resamples to check stability of interval endpoints in an analysis.

<!-- example: kde-bootstrap-cpu -->
```python
import numpy as np
from statgpu.nonparametric import kde_bootstrap_confidence_interval

rng = np.random.default_rng(42)
samples = rng.normal(size=120)
points = np.array([-1.0, 0.0, 1.0])
ci = kde_bootstrap_confidence_interval(
    samples, points, bandwidth=0.4, kernel="gaussian", backend="numpy",
    n_resamples=100, confidence_level=0.95, random_state=17,
    method="percentile", return_bootstrap_samples=True,
)
print(ci.estimate.shape, ci.bootstrap_samples.shape)
print(np.column_stack([ci.estimate, ci.lower, ci.upper]).round(3))
```

```text
(3,) (100, 3)
[[0.249 0.165 0.315]
 [0.431 0.369 0.505]
 [0.239 0.190 0.305]]
```

Columns are estimate, lower bound, upper bound. These are **pointwise percentile bootstrap intervals** for the smoothed density at the chosen points, not a simultaneous confidence band, an interval containing future observations, or a probability interval. They do not correct smoothing bias, dependence, or a preceding bandwidth search. A percentile interval need not contain the original estimate; lower must not exceed upper.

Important scope details:

- The NumPy 1D Gaussian bootstrap fast path holds the originally selected factor fixed even when `bandwidth` is a string. Other paths refit the selector on each resample. To use the same bandwidth factor in every resample, specify a numeric factor as above. The sample covariance is still recomputed, so the absolute bandwidth can change; do not assume identical selector uncertainty across backends.
- With nonuniform `weights`, the implementation both samples in proportion to the weights and re-applies the sampled weights. This is not interchangeable with every frequency/survey/importance-weight bootstrap. The example and interpretation here are limited to equal weights; establish the resampling scheme for your design before using weighted intervals.
- The wrapper accepts only `method="percentile"`. The related `kde_confidence_interval` has `method="normal"` (default, asymptotic and only 1D Gaussian) or `"bootstrap"`; neither is a bias-corrected or simultaneous method.

For confidence level $1-\alpha$, percentile bootstrap bounds at each fixed query
are the empirical quantiles of the $B$ replicate density estimates:

$$
[L(x),U(x)]=[Q_{\alpha/2}\{\widehat f_b^*(x)\}_{b=1}^B,
Q_{1-\alpha/2}\{\widehat f_b^*(x)\}_{b=1}^B].
$$

The separate normal method uses the fitted absolute width $h$ and normalized
weights to form $n_{\mathrm{eff}}=1/\sum_i w_i^2$. With the Gaussian kernel's
$R(K)=\int K(u)^2du=1/(2\sqrt\pi)$,

$$
\widehat{\mathrm{SE}}(x)=\sqrt{\frac{\widehat f(x)R(K)}{n_{\mathrm{eff}}h}},
\qquad
[L(x),U(x)]=[\max\{0,\widehat f(x)-z_{1-\alpha/2}\widehat{\mathrm{SE}}(x)\},
\widehat f(x)+z_{1-\alpha/2}\widehat{\mathrm{SE}}(x)].
$$

$z_{1-\alpha/2}$ is the standard-normal quantile. Both formulas are pointwise;
the normal formula is an asymptotic variance approximation without bias
correction. See the [normal-interval example and complete arguments](../reference/survival-smoothing-api.md#density-confidence-intervals).

## Shapes and API choices

Import the following from `statgpu.nonparametric`:

| Need | API | Return / use |
|---|---|---|
| Reusable density fit | `fit_kde(samples, ...)` | Fitted `KDE` (a `KernelDensityEstimator` subclass). |
| Estimator-style density fit | `KernelDensityEstimator(...).fit(X)` | Reuse `pdf`, `logpdf`, `predict`, `score_samples`, `score`, or `__call__` (density). |
| One-shot density | `kde_pdf(samples, points, ...)` | Density vector; `return_log=True` returns log density. |
| Density intervals | `kde_bootstrap_confidence_interval(...)`, `kde_confidence_interval(...)` | `KDEBootstrapResult`. |
| Reusable regression fit | `fit_kernel_regression(samples, targets, ...)` | Fitted `KernelRegression`. |
| Estimator-style regression | `KernelRegressionRegressor(...).fit(X, y)` | Alias subclass of `KernelRegression`; use `predict`, `score`, `__call__`. |
| One-shot regression | `kernel_regression_predict(samples, targets, points, ...)` | Predicted response array. |

- Training samples: finite numeric `(n_samples,)` or `(n_samples, n_features)`, at least two observations. Regression targets: finite `(n_samples,)` or `(n_samples, n_targets)` with matching rows.
- Queries: `(n_query, n_features)`. A 1D query vector means many points for a single-feature fit, or one point if its length matches a multivariate fit's feature count. Feature count and ordering must agree with training.
- KDE output shape is `(n_query,)`; regression output is `(n_query,)` for a 1D target and `(n_query, n_targets)` for a 2D target, including `(n_query, 1)`. With NumPy the outputs are NumPy arrays; ordinary GPU predictions remain backend-native. Interval result arrays are converted to NumPy.
- `weights`, when supplied for fitting, must be finite, nonnegative, length `n_samples`, and have a positive sum; they are normalized. Concentrating all weight on one observation fails covariance estimation. Invalid shapes, nonfinite inputs, nonpositive bandwidths, and unknown kernel names raise errors. Call `fit` before prediction.

For large-offset coordinates, center training samples and queries with the same
training-derived offset before evaluation. Current distance calculations can
lose precision without centering, particularly log density and multivariate
density/regression; see the [API numerical limitation](../reference/survival-smoothing-api.md#kernel-density-estimation).

## Bandwidth, kernels, and tuning boundaries

| Control | Practical meaning |
|---|---|
| `bandwidth="scott"` / `"silverman"` | Starting rules based on effective sample size and dimension. Defaults use Scott. |
| Numeric `bandwidth` | Positive **dimensionless factor** $b$, with kernel covariance $H=b^2\widehat\Sigma$. In 1D the absolute width is approximately $b\,s_x$, not $b$ itself. |
| `kernel_metric="full"` | Regression default: use the full covariance to define neighborhoods. KDE also uses full covariance. |
| `kernel_metric="diagonal"` | Regression: remove covariance off-diagonal terms. This changes the metric, not an exact/approximation mode. |
| `bandwidth_per_feature` | Regression only, requires the diagonal metric. Positive **absolute widths** per feature, in feature units; a scalar broadcasts. Sets $H=\operatorname{diag}(h_j^2)$ and bypasses `bandwidth` selection. |
| `kernel` | `gaussian`, `rectangular`, `triangular`, `epanechnikov`, `biweight`, `triweight`, `cosine`, `optcosine`; last two are 1D-only. |
| `batch_size` | Positive query batch size, default `1024`; a NumPy 1D Gaussian density fast path may evaluate queries together. |

For $p$ features and normalized weights, the default factor rules are

$$
n_{\mathrm{eff}}=\frac1{\sum_i w_i^2},\qquad
b_{\mathrm{Scott}}=n_{\mathrm{eff}}^{-1/(p+4)},\qquad
b_{\mathrm{Silverman}}=\left(\frac{n_{\mathrm{eff}}(p+2)}4\right)^{-1/(p+4)}.
$$

They set $H=b^2\widehat\Sigma$, with the weighted sample covariance
$\widehat\Sigma$ (plus numerical stabilization). For equal weights,
$n_{\mathrm{eff}}=n$; in one dimension $h=\sqrt{H_{11}}$.

Other bandwidth names include `nrd0`, `nrd`, `ucv`, `bcv`, `sj`, `sj-ste`, and `sj-dpi`. They are not all prediction-loss optimizers. R-style selectors use Gaussian-reference rules, nonuniform weights may use quantile resampling, and multivariate extensions use a one-dimensional principal-axis projection. Inspect `bandwidth_info_` / `to_numpy_metadata()` for the selected factor and strategy; do not label these extensions exact multivariate R equivalents. Some data, including constant/sparse samples, can make a selector fail.

Kernel regression additionally accepts `"cv"`, `"cv_ls"`, `"cv-nw"`, and `"cv-ll"` for a leave-one-out MSE search over a scalar factor. The selector uses full covariance, and local-linear CV correction is implemented only for one feature; in multiple dimensions its objective uses NW predictions even for `"cv-ll"`. For a multivariate local-linear or diagonal-metric model, explicitly validate candidate widths against the **actual intended model**, rather than assuming this selector optimizes that exact configuration. These CV names are not KDE bandwidth options.

No unified strict/approx switch exists. With compact-support kernels, a query can have no supported observations: KDE returns zero density (`logpdf=-inf`), while regression falls back to its weighted training target mean when local effective weight is too small (`min_effective_weight=1e-12` by default). A failed/unstable local-linear solve can use stabilization or NW fallback. Treat distant-query predictions cautiously; they are not evidence of reliable extrapolation. Even Gaussian regression can reach the low-weight fallback far from the data.

## Complete API and diagnostics reference

The [complete reader-facing API reference](../reference/survival-smoothing-api.md#kernel-density-estimation)
collects constructors, function and method signatures, defaults, restrictions,
return shapes, and selector/interval result fields. In particular, KDE
`batch_size` is an evaluation argument, not a constructor parameter; regression
can set it in either place. Implementation links for further reading follow:

- [KDE and intervals](../../../statgpu/nonparametric/kernel_smoothing/_kde.py): `KernelDensityEstimator`, `KDE`, `fit_kde`, `kde_pdf`, `kde_confidence_interval`, `kde_bootstrap_confidence_interval`, `KDEBootstrapResult`. Estimator construction also accepts `weights=None`, `backend="auto"`, `device="auto"`, `n_jobs=None`, `gpu_memory_cleanup=False`. Interval controls include `n_resamples=200`, `confidence_level=0.95`, `random_state=None`, `return_bootstrap_samples=False`, and `batch_size=1024`; the general interval function uses `bootstrap_method="percentile"`.
- [Kernel regression](../../../statgpu/nonparametric/kernel_smoothing/_kernel_regression.py): `KernelRegression`, `KernelRegressionRegressor`, both functional helpers, all fit/predict controls, and `to_numpy_metadata()`. The constructor includes the same device/jobs/cleanup controls and `batch_size` / `min_effective_weight`; `predict` can override the latter two. One-shot functions select `backend` and do not take a `device` argument.
- [Bandwidth selection](../../../statgpu/nonparametric/kernel_smoothing/_bandwidth_selection.py): `select_bandwidth` returns `BandwidthSelectionResult` with diagnostics; `select_bandwidth_factor` returns a scalar. These lower-level functions require sample/covariance/weight/backend inputs; normal users can select through an estimator.
- [Shared validation and kernel definitions](../../../statgpu/nonparametric/kernel_smoothing/_kernel_common.py), [shared estimator parameters](../../../statgpu/_base.py), and the [complete nonparametric export inventory](../../../statgpu/nonparametric/__init__.py).

Useful fitted state includes `samples_`, normalized `weights_`, `bandwidth_factor_`, `bandwidth_info_` (`None` for explicit numeric factors), `covariance_`, `inv_covariance_`, `kernel_`, `backend_`, `n_samples_`, and `n_features_`. Regression adds `targets_`, `n_targets_`, `target_mean_`, `regression_`, `kernel_metric_`, and `bandwidth_per_feature_`. `to_numpy_metadata()` provides host-side diagnostics. Interval results expose `points`, `estimate`, `lower`, `upper`, `confidence_level`, `n_resamples`, `random_state`, `kernel`, `backend`, `metadata`, optional `(n_resamples, n_query)` `bootstrap_samples`, and `to_dict()`.

## Optional GPU execution and external comparisons

`backend` accepts `"numpy"`, `"cupy"`, `"torch"`, or `"auto"`. An explicit backend selects the array library; with `"auto"`, selection follows the estimator/global device configuration rather than just the input array type. `device` belongs to estimator constructors. Prefer matching device/backend settings and inspect returned array placement; selecting the Torch library alone is not evidence of CUDA execution. See [device and memory](../guides/device-and-memory.md).

The following is separate from the CPU workflows and requires working CuPy/CUDA. It does not run on a CPU-only installation, and a missing explicit backend is not silently replaced with NumPy.

<!-- example: kde-gpu -->
```python
import cupy as cp
from statgpu.nonparametric import fit_kde

samples_gpu = cp.linspace(-2, 2, 100)
points_gpu = cp.linspace(-3, 3, 41)
kde_gpu = fit_kde(samples_gpu, bandwidth=0.35, backend="cupy")
density_gpu = kde_gpu.pdf(points_gpu)  # CuPy output
```

Some bandwidth selection and interval work uses host arrays; do not assume an entirely GPU-resident pipeline or a speedup for small fits. Runtime depends on sample/query counts, dimension, batching, selector, and transfer costs.

For a SciPy `gaussian_kde` comparison, align data orientation, weights, and covariance bandwidth factor. For statsmodels kernel regression, align kernel, regression mode, diagonal metric, and **absolute per-feature widths**; the scalar factor here is not the same parameter. Dedicated comparison scripts are [SciPy KDE](../../../dev/benchmarks/benchmark_kde_vs_scipy.py), [statsmodels regression](../../../dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py), [R methods](../../../dev/benchmarks/benchmark_nonparametric_vs_r.py), and the [combined suite](../../../dev/benchmarks/benchmark_nonparametric_comparison_suite.py). Their existence alone does not prove every setting or backend.

## References

- Rosenblatt, M. (1956). Remarks on some nonparametric estimates of a density function. *Annals of Mathematical Statistics*, 27(3), 832–837. [DOI](https://doi.org/10.1214/aoms/1177728190).
- Parzen, E. (1962). On estimation of a probability density function and mode. *Annals of Mathematical Statistics*, 33(3), 1065–1076. [DOI](https://doi.org/10.1214/aoms/1177704472).
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141–142. [DOI](https://doi.org/10.1137/1109020).
- Watson, G. S. (1964). Smooth regression analysis. *Sankhya: The Indian Journal of Statistics, Series A*, 26(4), 359–372.
- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
