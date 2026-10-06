# Ridge

> Language: English  
> Last updated: 2026-10-06  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/ridge.md)

Language switch: [Chinese](../../cn/models/ridge.md)

## Overview

`Ridge` provides L2-regularized linear regression with the same inference surface as `LinearRegression` (including robust covariance options). It is used to stabilize prediction or coefficient estimates under multicollinearity. With positive alpha, coefficient intervals describe the penalized fit and do not automatically remove shrinkage bias or adjust for choosing alpha.

## Path

`statgpu.linear_model.Ridge`

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

## Estimating Equation

After centering the data using the corresponding ordinary or weighted means, the first-order condition is

$$
\left(X_c^\top W X_c + \alpha\,s_w I\right)\hat\beta
= X_c^\top W y_c,
$$

where $W=I$ and $s_w=n$ without sample weights, while $W=\operatorname{diag}(w)$ and $s_w=\sum_iw_i$ for weighted fitting.

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

rng = np.random.default_rng(113)
variation = rng.normal(size=(80, 3))
X = variation + 1e8
y = 0.4 + variation @ np.array([1.0, -0.5, 0.3]) + rng.normal(scale=0.1, size=80)
X_train, X_test = X[:60], X[60:]
y_train, y_test = y[:60], y[60:]
origin = X_train.mean(axis=0)
model = Ridge(alpha=0.1, device="cpu", compute_inference=True).fit(
    X_train - origin, y_train,
)
prediction = model.predict(X_test - origin)
original_intercept = model.intercept_ - origin @ model.coef_
print(np.round(model.coef_, 3))
print(round(float(np.mean((prediction - y_test)**2)), 3))
```

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
| `n_jobs` | `None` | Number of parallel jobs |
| `compute_inference` | `True` | Whether to compute inference stats (SE/t/p/CI) |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag for `cov_type="hac"`; default follows a Newey-West-style heuristic |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory cleanup after each fit |
| `solver` | `"exact"` | Exact L2 solution by default; `fista` uses the same objective |
| `max_iter` | `1000` | Iteration budget for iterative solvers; exact fitting needs one solve |
| `tol` | `1e-4` | Iterative convergence tolerance |
| `cpu_solver` | `"fista"` | Deprecated compatibility argument; use `solver` to select the algorithm |
| `lipschitz_L` | `None` | Optional smooth-gradient Lipschitz bound for compatible iterative solvers |

## CPU+GPU Examples

```python
from statgpu.linear_model import Ridge

# CPU
m_cpu = Ridge(alpha=1.0, device="cpu", cov_type="hc3", compute_inference=True)
m_cpu.fit(X, y, sample_weight=w)

# CuPy CUDA
m_gpu = Ridge(
    alpha=1.0,
    device="cuda",
    cov_type="hc3",
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y, sample_weight=w)
```

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

## External Validation

- Internal consistency is tested against the average-loss closed form and the generic penalized-linear estimator.
- sklearn comparisons use the explicit unweighted or weighted alpha mapping.
- Weighted exact/FISTA, formula-row alignment, inference, and RidgeCV weight-rescaling invariance are covered in `dev/tests/test_ridge_weighted_consistency.py`.
- Numerical covariance and reference-distribution comparisons must align the ridge penalty, weights, degrees of freedom and covariance choice. CPU checks do not establish GPU precision or performance.

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
