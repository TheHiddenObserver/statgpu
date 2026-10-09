# Adaptive Lasso

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/adaptive-lasso.md)

## When is Adaptive Lasso useful?

`AdaptiveLasso` fits a sparse linear model for a continuous response. Ordinary [Lasso](lasso.md) penalizes every slope equally; Adaptive Lasso uses an initial estimate to penalize small initial coefficients more strongly and large ones less strongly. This can reduce shrinkage of strong signals, but the initial fit and feature scales matter. Start with ordinary Lasso for a simpler baseline, or [Elastic Net](elastic-net.md) when correlated predictors make selection unstable.

The oracle property is an asymptotic result under regularity and tuning conditions. It does not guarantee that a finite sample recovers the true variables, that selected variables are significant, or that their associations are causal.

## Path

`statgpu.linear_model.AdaptiveLasso`

## Objective Function

$$
\min_{b,\beta} \frac{1}{2n}\|y - b\mathbf{1} - X\beta\|_2^2 + \alpha \sum_{j=1}^p w_j |\beta_j|
$$

where $w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$ are adaptive weights computed from an initial estimate (ridge regression by default).

Here n is the observation count, X contains p feature columns, b is the unpenalized intercept, and β contains the slopes. `fit_intercept=False` fixes b=0. The coordinate weights $w_j$ penalize features; they are not observation `sample_weight` values. Larger `nu` makes the penalty weights more sensitive to the initial coefficient magnitudes. Features are not automatically standardized for prediction.

## A complete CPU example

Run these steps in one Python session. The simulated predictors have comparable scales, so no learned scaling is needed here.

<!-- learner-example: adaptive-lasso-prediction -->
```python
import numpy as np
from statgpu.linear_model import AdaptiveLasso
```

<a id="cpu-data"></a>

### Prepare observations and a held-out set

`X` has shape `(160, 5)`: observations are rows and predictors are columns. `y` is a `(160,)` continuous response. Only the first two predictors generate a signal; keep the last 40 rows out of training and tuning.

```python
rng = np.random.default_rng(64)
X = rng.normal(size=(160, 5))
y = 1.5 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.4, size=160)
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]
```

### Fit the adaptive penalty

The estimator computes its initial coefficient estimate and adaptive weights from the training data. `alpha=0.1` and `nu=1.0` are illustrative choices; inference stays disabled.

```python
model = AdaptiveLasso(
    alpha=0.1, nu=1.0, device="cpu", compute_inference=False,
    max_iter=5000, tol=1e-8,
)
model.fit(X_train, y_train)
```

### Predict and interpret

Pass test columns in the same order as the training columns. Compare held-out R² with the selected coefficient pattern.

```python
prediction = model.predict(X_test)
print("Slopes:", np.round(model.coef_, 3))
print("Selected columns:", np.flatnonzero(np.abs(model.coef_) > 1e-8))
print("Test R2:", round(model.score(X_test, y_test), 3))
```
<!-- example-end: adaptive-lasso-prediction -->

For this seed, slopes round to `[1.982, -0.957, 0, 0, 0]`, selected columns are `[0, 1]`, and held-out R² is about `0.966`.

`coef_` has shape `(5,)`, `intercept_` is a separate scalar, and `prediction` has shape `(40,)`. Zero coefficients identify the variables omitted by this particular penalized fit. Selection can change with the sample and tuning; a high test R² does not validate selected variables as scientific discoveries.

## Choose parameters and check stability

- Choose `alpha` using training-only validation; keep the test set separate. This wrapper does not provide a dedicated Adaptive Lasso CV class.
- If choosing `nu` as well, compare both predictive performance and selected-set stability. The initial fit and adaptive weights must be recomputed within every training fold.
- Learn scaling inside each training fold and reuse it for validation or test rows. Do not compute adaptive weights from the full dataset before validation.
- Compare a tighter `tol` and larger `max_iter` when numerical accuracy matters. An iteration count alone does not establish optimality.

## Algorithm

1. **Initialization**: Compute initial coefficient estimates via ridge-penalized coordinate descent (matching R glmnet's ridge solver).
2. **Weight computation**: $w_j = 1/(|\hat{\beta}_j^{init}| + \varepsilon)^\nu$ with $\nu = 1$ (default).
3. **Weighted L1 solve**: Solve the weighted Lasso problem using FISTA with the computed weights.

## Oracle Property

Under regularity conditions (Zou 2006, Theorem 1):
- **Selection consistency**: $\Pr(\hat{S} = S_0) \to 1$ as $n \to \infty$
- **Asymptotic normality**: $\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

where $S_0$ is the true support set and $\Sigma_0$ is the limiting covariance of the oracle estimator. These asymptotic statements require the stated assumptions and tuning regime.

## Covariance and inference limits

Keep `compute_inference=False`, the default. Built-in post-fit inference is not supported for `adaptive_l1`; enabling it raises rather than producing usable coefficient intervals. The `inference_method` constructor argument does not remove this restriction.

A separate OLS fit on the selected columns can describe an active-set refit, but ordinary OLS intervals from the same responses do not correct for selecting those columns. The asymptotic oracle property is not permission to interpret such intervals as general selection-adjusted inference. See [inference modes](../guides/inference-modes.md) for these distinctions.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | Regularization strength |
| `nu` | `1.0` | Exponent controlling sensitivity of adaptive weights to initial coefficients |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `max_iter` | `1000` | Maximum iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `compute_inference` | `False` | Keep disabled; built-in inference is unsupported for this penalty |
| `inference_method` | `"debiased"` | Stored inference request; does not enable unsupported adaptive-Lasso inference |
| `solver` | `"auto"` | Solver selection |
| `gpu_memory_cleanup` | `False` | CuPy pool cleanup after fit |

## Optional GPU use

After the [CPU data setup](#cpu-data), use the same constructor and training arrays with `device="cuda"` for CuPy CUDA or `device="torch"` for Torch CUDA. An explicit unavailable backend raises; only `auto` may choose another available backend. Keep inference disabled. See [device and memory](../guides/device-and-memory.md).

## Outputs and method reference

The parameter table above lists every constructor control. Import `AdaptiveLasso` from `statgpu.linear_model`. `fit` returns the estimator; `predict` returns one response per row; `score` returns R². Prediction uses `coef_` and `intercept_`. `n_iter_` reports the numerical iteration count. `summary()` requires inference and is not available for this estimation-only use.

For the inherited `fit`, formula and weight arguments, prediction placement, weighted `score`, `get_params` and `set_params`, see the [PenalizedLinearRegression method reference](../reference/linear-model-api.md#penalizedlinearregression). Its generic inference options do not imply support for the adaptive penalty.

## Comparing implementations

When comparing adaptive-Lasso fits, align the initial estimator and its regularization, the adaptive-weight formula (including epsilon and `nu`), feature scaling, intercept treatment, loss normalization, final `alpha`, and convergence accuracy. Agreement for ordinary Lasso alone does not verify the adaptive-weight calculation.

## References

- Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476), 1418-1429. [https://doi.org/10.1198/016214506000000735](https://doi.org/10.1198/016214506000000735)
- Wang, H., Li, B., & Leng, C. (2009). Shrinkage tuning parameter selection with a diverging number of parameters. *Journal of the Royal Statistical Society: Series B*, 71(3), 671-683.
