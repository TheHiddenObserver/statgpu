# Lasso

> Language: English  
> Last updated: 2026-09-09  
> Switch: [简体中文](../../cn/models/lasso.md)

## What problem does it solve?

`Lasso` is linear regression with an L1 penalty. Its distinctive feature is that regularization can drive some coefficients **exactly to zero**, so the fitted model performs prediction and feature selection at the same time.

Suppose you have 100 candidate predictors but believe only a small subset carries useful signal. OLS gives every predictor a coefficient. Ridge shrinks all of them but normally keeps them nonzero. Lasso can return a much smaller active set.

Typical questions include:

- Which predictors can be removed while keeping a useful linear model?
- Can I reduce overfitting when many candidate features are mostly noise?
- Can I obtain a sparse model that is easier to inspect or deploy?

## A motivating example

Imagine a dataset with 12 measured variables, but the response was actually generated from only three of them.

An unregularized model can assign small noisy coefficients to many variables. Lasso instead tends to produce a pattern like:

```text
feature       0    1    2    3    4    5    6    7    8    9   10   11
true coef     0   2.0   0    0    0  -1.5   0    0    0   0.8   0    0
Lasso coef    0  ~2.0   0    0    0  ~-1.4  0    0    0  ~0.7   0    0
```

That is the central attraction of Lasso: it turns a continuous regression problem into a sparse representation without a separate hard feature-selection step.

## Intuition

Lasso makes coefficients expensive in proportion to their absolute magnitude:

$$
\lVert\beta\rVert_1=\sum_j|\beta_j|.
$$

This creates a **soft-thresholding** effect. A coefficient with weak evidence can be pulled all the way to zero instead of merely being made smaller.

A useful mental comparison is:

```text
OLS   : keep whatever improves fit
Ridge : shrink everything
Lasso : shrink, and remove weak coefficients entirely
```

The price of sparsity is that feature selection can become unstable when several predictors carry nearly the same information. Pure Lasso may keep one member of a correlated group and discard another almost arbitrarily. [Elastic Net](elastic-net.md) is often preferable in that situation.

## When to use it

Lasso is especially useful when:

- you expect the signal to be sparse;
- the number of candidate predictors is large relative to the amount of data;
- interpretability benefits from a small active feature set;
- storage, deployment, or downstream modeling benefits from dropping features;
- you are willing to tune the amount of regularization using validation.

Choose another method when:

- most predictors are expected to have small but real effects — [Ridge](ridge.md) may be better;
- predictors form strongly correlated groups and you would rather keep/shrink the group together — start with [Elastic Net](elastic-net.md);
- the response is not well modeled by Gaussian linear regression — use the corresponding penalized GLM, survival, or other model family;
- your primary goal is formal post-selection inference rather than prediction/selection — selection-aware inference requires additional assumptions and care.

## Model and objective

With intercept $b$, Lasso minimizes

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lVert\beta\rVert_1.
$$

Here:

- $n$ is the number of observations;
- $x_i$ is the feature vector for observation $i$;
- $\beta$ is the coefficient vector;
- $\alpha\ge 0$ controls regularization strength;
- the intercept is not penalized.

Larger `alpha` makes zero coefficients more likely. Very small `alpha` approaches an unregularized linear fit; very large `alpha` can remove almost all signal.

### Why can L1 produce exact zeros?

The absolute-value penalty has a sharp corner at zero. In the optimization step, small coefficient updates are soft-thresholded:

$$
\mathcal S_\lambda(z)
=
\operatorname{sign}(z)\max(|z|-\lambda,0).
$$

If $|z|\le\lambda$, the result is exactly zero. This is the computational mechanism behind Lasso sparsity.

## Minimal runnable example

The following data has 12 features but only features `1`, `5`, and `9` truly affect the response.

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(1)
X = rng.normal(size=(500, 12))

true_coef = np.zeros(12)
true_coef[[1, 5, 9]] = [2.0, -1.5, 0.8]
y = 0.7 + X @ true_coef + rng.normal(scale=0.7, size=500)

model = Lasso(
    alpha=0.08,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print("coefficients:", np.round(model.coef_, 3))
print("selected features:", np.flatnonzero(np.abs(model.coef_) > 1e-8))
print("R²:", round(model.score(X, y), 3))
```

With the fixed seed and this `alpha`, the clearly nonzero coefficients should be at indices `1`, `5`, and `9`, with values roughly `1.97`, `-1.41`, and `0.73`. The exact values are smaller in magnitude than the generating coefficients because shrinkage is part of the estimator.

## How to read the result

- `coef_[j] == 0` means Lasso removed that feature from the fitted linear predictor at the selected `alpha`.
- A nonzero coefficient is still **shrunken**; do not read it as the unpenalized OLS effect estimate.
- `intercept_` is fitted separately and is not part of the L1 penalty.
- `predict(X_new)` returns predicted continuous outcomes.
- `score(X, y)` returns $R^2$ and accepts `sample_weight=`.
- `n_iter_` reports optimization iterations for the selected numerical path.

Selection is data-dependent. A feature being zero in one sample does not prove its true effect is exactly zero, especially when predictors are correlated or the sample is small.

## Key parameters and how to choose them

This table is intentionally **curated** for the normal workflow. The exhaustive constructor inventory is in [Complete API reference](#complete-api-reference).

| Parameter | Default | How to think about it |
|---|---:|---|
| `alpha` | `1.0` | Main statistical choice. Larger values create more shrinkage and more zeros. Prefer `LassoCV` or another validation procedure for predictive selection. |
| `fit_intercept` | `True` | Usually keep it unless theory fixes the intercept or the design already contains one. |
| `device` | `"auto"` | CPU is usually simplest for small problems; GPU is useful when the optimization workload is large enough to amortize transfer/setup cost. |
| `solver` | `"fista"` | Authoritative direct-fit algorithm selector on every backend. Use `coordinate_descent` for the CPU CD path; use proximal solvers such as `fista` where appropriate. |
| `stopping` | `"coef_delta"` | `"kkt"` is useful when you want convergence judged against optimality conditions rather than coefficient movement alone. |
| `compute_inference` | `True` | Turn it off for pure prediction/selection. If inference matters, choose `inference_method` deliberately and read the dedicated inference reference. |

### Standardize before regularizing

L1 penalizes coefficient magnitude directly. Features measured on very different scales therefore receive effectively different penalties.

For most Lasso workflows, standardize continuous predictors before fitting. The synthetic example above already puts every feature on approximately the same scale.

## Compare with nearby methods

| Method | Keeps correlated groups? | Exact zeros? | Typical reason to choose it |
|---|:---:|:---:|---|
| OLS / `LinearRegression` | no regularization | no | unpenalized estimation when the design is stable |
| [Ridge](ridge.md) | tends to share signal | no | stabilize correlated predictors without feature deletion |
| **Lasso** | may select one member | yes | sparse prediction / automatic feature selection |
| [Elastic Net](elastic-net.md) | more group-friendly than Lasso | yes | sparse model with correlated predictors |
| Adaptive Lasso | data-dependent L1 weights | yes | reduce uniform-penalty bias under a stronger sparse-model assumption |

If your main uncertainty is “Ridge or Lasso?”, ask whether exact feature removal is actually valuable. If not, Ridge is often the safer low-variance choice.

## CPU, GPU, Formula, and weighted fitting

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.08,
    device="cuda",
    solver="fista",
    stopping="kkt",
    compute_inference=False,
).fit(X, y)
```

Explicit `device="cuda"` and `device="torch"` use their corresponding GPU backends when supported; an unavailable explicit device fails rather than silently changing the execution path.

`fit()` also accepts `sample_weight=` and the shared `formula=` / `data=` interface. Formula metadata is retained for DataFrame prediction.

## Advanced: solver support

| `solver` value | CPU | CuPy / Torch | Meaning |
|---|:---:|:---:|---|
| `fista` (default) | yes | yes | Stable proximal-gradient path for the L1 objective |
| `auto` | FISTA | FISTA | Current squared-error + L1 automatic destination |
| `fista_bb` | yes | yes | FISTA with Barzilai-Borwein step adaptation |
| `admm` | yes | yes | Alternative split solver; uniform sample weights only |
| `coordinate_descent` | yes | no | CPU-only direct-fit coordinate-descent path |

`solver` is the single backend-neutral direct-fit selector. `newton`, `lbfgs`, `irls`, and `exact` are rejected for the non-smooth L1 objective.

`cpu_solver` is retained as a compatibility control for legacy/CV behavior, but it does not select the algorithm for a direct `Lasso.fit`; use `solver` for direct-fit algorithm choice. The CV estimator has separate selection and final-refit stages, documented in the [cross-validation guide](../guides/cross-validation.md).

`admm_rho` controls the ADMM penalty parameter when the ADMM path is selected. `lipschitz_L` supplies a precomputed Lipschitz constant for compatible proximal paths.

## Advanced: inference after Lasso

Inference after data-driven selection is substantially harder than inference after a prespecified OLS model. statgpu exposes de-biased, OLS-style diagnostic, residual-bootstrap, and optional simultaneous max-|Z| paths, and they do **not** make the same statistical claim.

| `inference_method` | Intended use | Important limitation |
|---|---|---|
| `debiased` (constructor default) | de-biased/de-sparsified coefficient inference | marginal intervals require high-dimensional de-biasing assumptions; joint coverage needs the separate simultaneous procedure |
| `post_selection_ols` | active-set OLS/WLS diagnostic on the fit-resolved NumPy/CuPy/Torch backend | heuristic after selection; not a general selective-inference interval |
| `bootstrap` | residual-bootstrap alternative | materially more expensive and not a universal correction for selection uncertainty |

`post_selection_ols` is the canonical hardware-neutral spelling. Legacy `cpu_ols` / `gpu_ols` values are deprecated compatibility aliases that emit `FutureWarning` and normalize to the same method; device selection remains a separate concern.

For the actual node-wise-Lasso construction, coherent debiased intercept, marginal z inference, intercept-inclusive max-|Z| multiplier bootstrap, backend/reporting boundaries, multiple-testing distinction, and output fields, see **[Lasso inference](lasso-inference.md)**.

## Common pitfalls

- **Do not interpret “selected” as “proved causal” or even “certainly nonzero in the population.”** Lasso selection is sample- and tuning-dependent.
- **Do not ignore feature scaling.** An L1 penalty is not scale invariant.
- **Do not expect stable choices among nearly duplicate predictors.** Pure Lasso can arbitrarily prefer one correlated feature; Elastic Net is often more appropriate.
- **Do not choose `alpha` by maximizing training $R^2$.** Use held-out validation or cross-validation.
- **Do not attach ordinary OLS p-values after selection and treat them as if the model had been prespecified.** Use an inference method whose assumptions match your question; see [Lasso inference](lasso-inference.md).
- **Do not confuse numerical convergence with statistical correctness.** A tiny KKT residual only says the declared optimization problem was solved accurately.

## Complete API reference

The earlier parameter table is a decision guide. This section is the exhaustive constructor and model-method inventory for the current `Lasso` wrapper.

### Constructor

```python
Lasso(
    alpha=1.0,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    stopping="coef_delta",
    inference_method="debiased",
    n_bootstrap=200,
    bootstrap_random_state=None,
    enable_simultaneous_inference=False,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=None,
    simultaneous_include_intercept=False,
    device="auto",
    n_jobs=None,
    compute_inference=True,
    solver="fista",
    cpu_solver="coordinate_descent",
    lipschitz_L=None,
    admm_rho=1.0,
    gpu_memory_cleanup=False,
)
```

<!-- API-CONSTRUCTOR-START:Lasso -->
| Parameter | Default | Reference meaning |
|---|---:|---|
| `alpha` | `1.0` | L1 penalty strength. |
| `fit_intercept` | `True` | Fit an unpenalized intercept. |
| `max_iter` | `1000` | Maximum solver iterations. |
| `tol` | `1e-4` | Numerical convergence tolerance. |
| `stopping` | `"coef_delta"` | `coef_delta` or `kkt` convergence criterion where supported. |
| `inference_method` | `"debiased"` | Post-fit inference path: `debiased`, canonical `post_selection_ols`, or `bootstrap`. Legacy `cpu_ols` / `gpu_ols` aliases normalize to `post_selection_ols` with `FutureWarning`. |
| `n_bootstrap` | `200` | Number of residual-bootstrap draws for `inference_method="bootstrap"`. |
| `bootstrap_random_state` | `None` | RNG seed for the residual-bootstrap inference path. |
| `enable_simultaneous_inference` | `False` | Enable simultaneous max-|Z| intervals after de-biased inference. |
| `simultaneous_method` | `"maxz_bootstrap"` | Simultaneous interval calibration method; currently `maxz_bootstrap`. |
| `simultaneous_alpha` | `0.05` | Family-wise error level used for simultaneous intervals. |
| `simultaneous_n_bootstrap` | `1000` | Number of multiplier-bootstrap draws for max-|Z| calibration. |
| `simultaneous_random_state` | `None` | RNG seed for simultaneous bootstrap calibration. |
| `simultaneous_include_intercept` | `False` | Include the centered debiased intercept in the bootstrap max-|Z| target and reported joint interval family. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda` (CuPy), or `torch` (Torch CUDA). |
| `n_jobs` | `None` | Parallelism hint where a selected path uses it. |
| `compute_inference` | `True` | Compute the selected post-fit inference path. |
| `solver` | `"fista"` | Backend-neutral direct-fit solver; authoritative on CPU and GPU. |
| `cpu_solver` | `"coordinate_descent"` | Compatibility control retained for legacy/CV behavior; it does not replace `solver` for one direct `Lasso.fit`. |
| `lipschitz_L` | `None` | Optional precomputed Lipschitz constant for compatible proximal paths. |
| `admm_rho` | `1.0` | ADMM augmented-Lagrangian penalty parameter when ADMM is selected. |
| `gpu_memory_cleanup` | `False` | Best-effort release of cached GPU memory after fit. |
<!-- API-CONSTRUCTOR-END:Lasso -->

### `fit`

`Lasso` uses the shared penalized-linear fit signature:

```python
model.fit(
    X=None,
    y=None,
    sample_weight=None,
    formula=None,
    data=None,
)
```

| Argument | Meaning |
|---|---|
| `X` | Two-dimensional feature matrix for array-style fitting. |
| `y` | One-dimensional continuous response. |
| `sample_weight` | Optional non-negative analytic weights with positive finite total. Some solver paths have additional weight restrictions. |
| `formula` | Optional Patsy-style formula; use with `data`. |
| `data` | DataFrame used by the Formula interface. |

`fit()` returns `self`.

### Prediction, scoring, and reporting methods

| Method | Signature | Behavior |
|---|---|---|
| `predict` | `predict(X, return_cpu=True)` | Continuous predictions. `return_cpu=False` can keep GPU predictions on the executed backend. |
| `score` | `score(X, y, sample_weight=None)` | $R^2$, optionally weighted. |
| `summary` | `summary()` | Prints the coefficient/inference summary; requires a fitted model with inference enabled and available. |
| `get_params` / `set_params` | sklearn-style estimator utilities | Inspect or replace constructor state using the shared `BaseEstimator` contract. |

The inherited estimator-context utilities `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` are documented with their complete signatures and backend/fitted-state semantics in the [Inference API](../guides/inference-api.md). Lasso-specific coefficient inference is documented in [Lasso inference](lasso-inference.md).

### Fitted attributes and diagnostics

| Attribute | Availability / meaning |
|---|---|
| `coef_` | Penalized coefficients; exact zeros define the fitted active set. |
| `intercept_` | Fitted unpenalized intercept. |
| `n_iter_` | Iteration count for the selected numerical path. |
| `n_features_in_` | Number of fitted input features when published by the fit path. |
| `rsquared`, `rsquared_adj` | $R^2$ and adjusted $R^2$ when the required fitted state is available. |
| `fvalue`, `f_pvalue` | Classical joint fit statistic and p-value when defined. |
| `llf`, `aic`, `bic` | Gaussian fit diagnostics when the required reporting state is available. |
| `_bse` | Standard errors produced by the selected inference method. |
| `_tvalues` | Historical/statistic field used by compatible inference paths; de-biased inference has z semantics. |
| `_zvalues` | z-style statistics when populated through the structured inference result. |
| `_pvalues` | Coefficient p-values when inference succeeds. |
| `_conf_int` | Marginal coefficient intervals when inference succeeds. |
| `_conf_int_simultaneous` | Simultaneous intervals when explicitly enabled and successfully calibrated. |
| `_inference_result` | Structured inference result/metadata used by the reporting layer. |

Underscore-prefixed inference arrays are established reporting attributes in the current release. Their statistical interpretation depends on `inference_method`; do not treat all methods as interchangeable.

## Validation

Maintained validation covers solver convergence, CPU/GPU consistency, KKT stopping, de-biased inference, residual bootstrap, backend-native `post_selection_ols`, simultaneous inference, and physical-GPU behavior where required. Relevant entry points include `dev/tests/test_lasso_debiased_inference.py`, `dev/benchmarks/validate_post_selection_ols_gpu.py`, `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`, and `dev/comparisons/compare_lasso_kkt_stopping.py`.

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909.
