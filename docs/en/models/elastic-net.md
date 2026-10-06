# Elastic Net

> Language: English  
> Last updated: 2026-10-06<br>
> This page: Model documentation  
> Language switch: [Chinese](../../cn/models/elastic-net.md)

## Overview

`ElasticNet` combines L1 and L2 regularization for linear regression, balancing sparse feature selection (Lasso) and coefficient shrinkage (Ridge). It supports CPU, CuPy GPU, and PyTorch GPU execution. Direct fitting uses one backend-neutral `solver` interface; `device` controls where the computation runs.

## When is Elastic Net useful?

Use Elastic Net when you want some coefficients to be exactly zero, but correlated predictors make a pure Lasso fit unstable. Ridge is a simpler choice when shrinkage matters more than sparsity; ordinary [linear regression](linear-regression.md) is useful for a prespecified low-dimensional model without a shrinkage penalty. A selected variable is not automatically a causal effect.

## A complete CPU example

The example uses correlated predictors, standardizes them using training rows only, and evaluates predictions on held-out rows. The chosen tuning values illustrate the API; select them on validation data for a real application.

```python
import numpy as np
from statgpu.linear_model import ElasticNet

rng = np.random.default_rng(7)
X_raw = rng.normal(size=(400, 8))
X_raw[:, 1] = 0.8 * X_raw[:, 0] + 0.2 * rng.normal(size=400)
y = 0.4 + X_raw @ np.array([1.2, 0.8, -0.7, 0, 0, 0, 0, 0])
y += rng.normal(scale=0.5, size=400)

mean = X_raw[:300].mean(axis=0)
scale = X_raw[:300].std(axis=0)
X = (X_raw - mean) / scale
model = ElasticNet(
    alpha=0.08, l1_ratio=0.5, solver="fista", device="cpu",
    max_iter=5000, tol=1e-8, compute_inference=False,
).fit(X[:300], y[:300])

print("coef:", np.round(model.coef_, 3))
print("selected columns:", np.flatnonzero(np.abs(model.coef_) > 1e-8))
print("predictions:", np.round(model.predict(X[300:303]), 3))
print("held-out R2:", round(float(model.score(X[300:], y[300:])), 3))
```

For this seed, the CPU output is approximately: coefficients `[0.938, 0.834, -0.552, 0, 0, 0, 0, 0]`, selected columns `[0, 1, 2]`, predictions `[-1.575, 1.710, 1.476]`, and held-out R² `0.936`. Both correlated columns 0 and 1 remain in this fit, while the five noise columns are zero. The high test R² describes prediction on these simulated held-out rows; it does not turn selected variables into validated scientific discoveries. Small floating-point differences are expected.

### Reading the results and choosing parameters

- `coef_` has one coefficient per input column. In this example a unit increase means one training-set standard deviation because `X` was standardized. `intercept_` and `coef_` describe the prediction fit.
- Positive `l1_ratio` permits exact zeros. The displayed selected-column indices are a numerical summary of the penalized fit, not discoveries with guaranteed error control.
- `predict(X_new)` returns a one-dimensional response prediction; `score(X_new, y_new)` returns R². R² can be negative on new data, and training R² is not evidence of generalization.
- Larger `alpha` gives stronger total regularization. `l1_ratio` near 1 favors sparsity; closer to 0 gives more L2 shrinkage. For predictive tuning, use validation data or `ElasticNetCV` to choose both, keeping a final test set separate.
- Fit preprocessing only on each training split. Remove or otherwise handle zero-variance columns before dividing by their standard deviation. Do not center/scale the whole dataset before cross-validation.

## Input and prediction requirements

Use finite numeric `X` with shape `(n_samples, n_features)` and a one-dimensional response `y`. Prediction columns must match the training order and preprocessing. The fitting interface is `fit(X=None, y=None, sample_weight=None, initial_coef=None, **kwargs)`; optional nonnegative analytic weights enter the normalized weighted loss. `initial_coef` supplies a starting coefficient vector through `fit`; there is no `warm_start` constructor flag. The current implementation retains this starting vector on the estimator: omitting `initial_coef` on a later fit does not clear it. Create a fresh estimator when you want the default initialization, especially when changing the feature count; a retained vector with the old width can cause a dimension error. The shared optional formula interface accepts `formula=` and `data=` via fit keywords.

For weighted evaluation, pass a separate `score(X, y, sample_weight=weights)`
vector with one finite nonnegative weight per row and positive total weight.
The current squared-error `score` path does not reliably reject negative weights;
it can return an invalid R² above 1. Check these conditions yourself before
scoring. Training-weight validation does not validate a new evaluation vector.

### Weighted training diagnostics

After weighted `debiased` inference, `rsquared` currently uses the transformed
working response and centers it again. It can disagree substantially with R²
computed on the original weighted observations; `rsquared_adj` inherits that
problem. The residual-based `fvalue`/`f_pvalue` diagnostics also use that incorrect
total variation. Use `score(X, y, sample_weight=weights)` on the original response and
predictions, after validating the evaluation weights as described above. This
limitation does not change the fitted prediction coefficients. With inference
disabled, training diagnostic properties can instead be `None`.

<!-- learner-example: elasticnet-weighted-score -->
```python
import numpy as np
from statgpu import ElasticNet

rng = np.random.default_rng(25)
X = rng.normal(size=(20, 2))
y = np.arange(20.0) + 2 * X[:, 0]
weights = np.r_[np.ones(19), 1000.0]
model = ElasticNet(
    alpha=0.3, device="cpu", max_iter=5000, tol=1e-8, compute_inference=True,
).fit(
    X, y, sample_weight=weights,
)
weighted_r2 = model.score(X, y, sample_weight=weights)
print("Weighted training R2:", round(weighted_r2, 3))
```

This prints `Weighted training R2: 0.180`. It describes training fit, not
held-out performance. Enabling debiased inference does not change this score;
the current `rsquared` property would instead report about −2.262 on these data.

## Path

`statgpu.linear_model.ElasticNet`

## Objective Function

The Elastic Net optimization problem is:

$$
\min_{b,\beta}\frac{\sum_{i=1}^n w_i(y_i-b-x_i^\top\beta)^2}{2\sum_{i=1}^n w_i}+\alpha\lambda\|\beta\|_1+\frac{\alpha(1-\lambda)}{2}\|\beta\|_2^2
$$

Here n is the row count, p the feature count, $x_i$ the p-vector of predictors, $b$ the unpenalized intercept and $\beta$ the slopes. Set $w_i=1$ without weights; weights are nonnegative with positive sum. With `fit_intercept=False`, fix b=0.

where:
- `alpha` (α) controls overall regularization strength
- `l1_ratio` (λ) mixes L1 vs L2: λ=1 gives Lasso, λ=0 gives Ridge
- loss scaling by `1/(2n)` makes the public `alpha` use an average-loss convention

**Note on regularization scaling**: `ElasticNet` and `Ridge` use the same average-loss convention. Therefore, with `l1_ratio=0`, the Elastic Net objective at a given public `alpha` reduces to the corresponding L2 objective. The `ElasticNet` wrapper still retains its own solver/inference defaults; use `Ridge` when you specifically want the Ridge estimator contract.

## Estimating Equation

The KKT equation and optimization pseudocode below use unweighted centered X/y. For analytic weights, use the normalized square-root-weighted working arrays defined in the debiased section; centering alone does not remove unequal weights.

After eliminating the unpenalized intercept (equivalently, on centered data), the coefficient KKT condition is

$$
\frac{1}{n} X^\top (X\hat{\beta} - y) + \alpha(1-\lambda)\hat{\beta} + \alpha\lambda \cdot \partial\|\hat{\beta}\|_1 = 0.
$$

`solver` is the authoritative direct-fit algorithm selector on every backend. Use `device` separately to select CPU/CuPy/Torch execution. The historical `cpu_solver` argument is deprecated and no longer selects a second CPU-specific direct-fit algorithm in the unified engine. See the [penalized solver API migration guide](../guides/penalized-solver-api-migration.md).

## Estimation Algorithm

The normal default is **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm), a proximal-gradient method with Nesterov acceleration. Other solver values may be available according to the public solver compatibility contract.

### Key Optimization Insight

The L1 and L2 parts are handled by the Elastic Net proximal operator:

```text
# Gradient of the average squared-error term
grad = (X.T @ X @ w - X.T @ y) / n

# Elastic Net proximal step
w = soft_threshold(w_tilde, alpha * l1_ratio * step) / (
    1 + alpha * (1 - l1_ratio) * step
)
```

### Convergence checks and the stopping limitation

The current direct Gaussian fit stores `stopping="coef_delta"` or `"kkt"` but
ignores that choice. CPU FISTA and coordinate descent, and GPU FISTA, check
coefficient movement; ADMM uses primal/dual residuals. Setting `stopping="kkt"`
does not request an effective KKT check or certify optimality. Ill-scaled
features can stop moving while the KKT residual is still large. Scale features
using training data, compare tighter tolerances/budgets, and independently check
the objective or KKT residual when numerical accuracy matters. The separate
Lasso CV/path helper's stopping logic does not establish direct/final-refit
certification. Numerical optimality is separate from statistical validity.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `1.0` | Overall regularization strength |
| `l1_ratio` | `0.5` | L1 mixing proportion: 0=Ridge objective, 1=Lasso objective |
| `fit_intercept` | `True` | Fit an unpenalized intercept |
| `max_iter` | `1000` | Maximum solver iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `stopping` | `"coef_delta"` | Stored `coef_delta` / `kkt` request; currently ignored by direct Gaussian stopping checks (see above). |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` (CuPy), or `"torch"` |
| `n_jobs` | `None` | CPU parallelism where supported |
| `solver` | `"fista"` | Backend-neutral direct-fit optimization method |
| `cpu_solver` | `"fista"` | **Deprecated compatibility parameter**; use `solver` instead |
| `lipschitz_L` | `None` | Optional user-supplied Lipschitz constant |
| `gpu_memory_cleanup` | `False` | Release backend memory pools after fit where supported |
| `compute_inference` | `False` | Compute post-fit coefficient inference |
| `inference_method` | `"debiased"` | `"debiased"`, `"post_selection_ols"`, or `"bootstrap"`; deprecated `cpu_ols`/`gpu_ols` aliases remain temporarily accepted |
| `nodewise_alpha` | `None` | Node-wise Lasso penalty for `debiased` inference. Explicit positive values override the standardized design-side automatic rule. |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where supported |

The public wrapper does not accept separate `backend`, `warm_start`, or `random_state` constructor parameters. Backend selection is controlled by `device`; starting coefficients can be supplied through `fit(initial_coef=...)`, subject to the reuse limitation described above.

## Additional CPU/GPU examples

Run the data preparation above first. These examples reuse `X` and `y`; the inference example is separate from predictive tuning.

```python
from statgpu.linear_model import ElasticNet

# CPU: solver selects the algorithm; device selects the backend.
model_cpu = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="cpu",
    solver="fista",
)
model_cpu.fit(X, y)
print(f"R²: {model_cpu.score(X, y):.4f}")

# Explicit node-wise tuning changes debiased inference only.
model_db = ElasticNet(
    alpha=0.1,
    l1_ratio=0.7,
    nodewise_alpha=0.08,
    device="cpu",
    compute_inference=True,
    inference_method="debiased",
)
model_db.fit(X, y)
print(model_db.nodewise_alpha_)

# GPU with the same solver interface
model_gpu_cupy = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    gpu_memory_cleanup=True,
)
model_gpu_cupy.fit(X, y)

model_gpu_torch = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
    device="torch",
    solver="fista",
)
model_gpu_torch.fit(X, y)
```

Backend performance depends on sample size, feature dimension, dtype, hardware, data residency, and transfer costs. Benchmark the target workload before selecting a backend solely for speed.

## Covariance/Inference

`ElasticNet` is estimation-only by default. Set `compute_inference=True` to run post-fit inference through the shared penalized-linear inference engine. The default `inference_method="debiased"` uses the same standardized node-wise Lasso one-step correction framework as the sparse Gaussian Lasso path to construct an approximate precision matrix, corrected coefficients, standard errors, z statistics, p-values, and confidence intervals. Its statistical validity depends on the usual design, sparsity, regularization, and model assumptions; the Lasso literature is background for the construction rather than a blanket guarantee for every `l1_ratio`. `summary()` is available after inference succeeds.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `compute_inference` | `False` | Enable post-fit coefficient inference |
| `inference_method` | `"debiased"` | `"debiased"`, `"post_selection_ols"`, or `"bootstrap"` |
| `nodewise_alpha` | `None` | Node-wise precision tuning for `debiased`; explicit positive scalar or standardized design-side automatic rule |
| `cov_type` | `"nonrobust"` | Covariance convention where applicable |
| `hac_maxlags` | `None` | HAC lag count where the selected inference method supports HAC |

`nodewise_alpha` is separate from the main Elastic Net `alpha`: it never changes the penalized prediction fit. When omitted, statgpu standardizes the already centered/weighted working design and uses

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

with `n_nw=n` without analytic weights and a Kish-style effective sample size for non-uniform analytic weights. The rule is response-independent, so changing only the units of `y` does not change the design-side precision problem. Successful multi-feature debiased inference exposes the resolved value as `nodewise_alpha_` and records the tuning/KKT provenance in `_inference_result.metadata`. A one-feature problem uses analytic precision and does not consume a node-wise penalty.

`post_selection_ols` is the canonical hardware-neutral active-set diagnostic. The historical unified spellings `cpu_ols` and `gpu_ols` are deprecated together and normalize to `post_selection_ols` with `FutureWarning` during the compatibility window. They do not select a device.

For `post_selection_ols`, the penalized model first determines the active set. statgpu then refits an unpenalized OLS model, or WLS when `sample_weight` is supplied, on exactly that active set using the backend recorded by the successful fit. The original penalized `coef_` remains the prediction coefficient vector; the active-set refit is exposed through `_params` / `_inference_result` and related reporting fields.

Post-selection OLS remains heuristic and does not provide general selective-inference coverage. Inference is conditional on selected regularization parameters and does not alter the fitted penalized coefficients.

Device selection is orthogonal to the statistical method: explicit `cpu`/`cuda`/`torch` is authoritative, while only genuine `device="auto"` may preserve backend-native CuPy or Torch-CUDA input during automatic routing. `post_selection_ols` reuses the fit-resolved backend, and CuPy/Torch `debiased` routes keep numerical inference on the executed GPU backend, including scalar normal-reference critical values. Residual `bootstrap` also uses the fit-recorded NumPy/CuPy/Torch backend and concrete device for response construction and numerical child refits, preserving the fitted penalty and tuning configuration. NumPy generates the shared residual-index schedule and stores final reporting arrays; these boundaries do not make GPU numerical refits CPU-only. This path requires `sample_weight=None` and `cov_type="nonrobust"`; weighted or HC/HAC bootstrap requests fail explicitly. It describes the penalized coefficient distribution conditional on the chosen tuning, without selection adjustment.

For `debiased` inference with an intercept, public `coef_`/`intercept_` remain the **penalized prediction fit**. Inference reporting uses debiased slopes `_params[1:]` and their matching original-coordinate intercept `_params[0] = ybar_w - xbar_w @ _params[1:]`; the first SE/z/p-value/CI row therefore belongs to this debiased reporting intercept rather than prediction `intercept_`. The result metadata records `intercept_estimator="centered_debiased"` and `intercept_influence="centered_nodewise"`. Analytic weights use the same weighted-centered average-loss problem across NumPy/CuPy/Torch, so global positive weight rescaling leaves this inference unchanged.

For `ElasticNetCV`, `compute_inference=True` applies debiased inference only to the final full-data refit after alpha and `l1_ratio` have been selected. Fold models remain estimation-only. `nodewise_alpha` is final-refit inference configuration only: it does not enter the candidate grid or fold scoring, and the outer `nodewise_alpha_` reflects the final estimator when inference succeeds. The current `ElasticNetCV` API still fixes this final inference method to `debiased`; other inference methods are not selectable through this CV wrapper.

### Debiased reporting formula

Let $\widetilde X,\widetilde y$ denote the centered working design/response; analytic weights additionally multiply row i by $\sqrt{n w_i/\sum_j w_j}$. Omit centering without an intercept. Let $M$ be the approximate inverse Gram matrix estimated by node-wise regressions and $\widehat\Sigma=\widetilde X^\top\widetilde X/n$. The reported slopes and model-based covariance are

$$
\hat\theta_{\mathrm{db}}=\hat\beta+
\frac{M\widetilde X^\top(\widetilde y-\widetilde X\hat\beta)}{n},
\qquad
\widehat V_{\mathrm{db}}=\frac{\hat\sigma^2}{n}M\widehat\Sigma M^\top.
$$

The implementation estimates $\hat\sigma^2$ from working residual squares divided by $\max(n-s,1)$, where s is the number of nonzero penalized slopes. SEs are square roots of the covariance diagonal; z statistics and normal-reference 95% intervals use the corrected slopes. This is a model-based construction: current `cov_type="hc0"` through `"hc3"` or `"hac"` requests do not replace its covariance and must not be interpreted as robust debiased inference. The [method–covariance table](../reference/linear-model-api.md#covariance-and-inference-behavior) distinguishes `debiased`, `post_selection_ols`, and `bootstrap`.

## Solver and Inference Semantics

For a direct `ElasticNet.fit`, **use `solver` on both CPU and GPU**. `device` chooses the execution backend; `solver` chooses the optimization algorithm. `cpu_solver` is a deprecated compatibility argument from the earlier hardware-split API and should not be used for new code.

Likewise, use `inference_method="post_selection_ols"` when the active-set OLS/WLS diagnostic is wanted. Do not choose `cpu_ols` or `gpu_ols` based on hardware; both are deprecated aliases for the same statistical method.

`compute_inference=False` returns the penalized estimate only. With `compute_inference=True`, the same fitted coefficients are retained and the selected post-fit inference method runs afterward.

## Outputs

After fitting, the following attributes are available:

| Attribute | Description |
|-----------|-------------|
| `coef_` | Estimated penalized coefficients used for prediction |
| `intercept_` | Penalized fitted intercept used for prediction |
| `n_iter_` | Number of iterations performed; reaching the budget does not establish convergence |
| `nodewise_alpha_` | Resolved node-wise tuning after successful multi-feature `debiased` inference; otherwise `None` |
| `_params` | Inference/reporting parameter vector when inference succeeds; for `debiased`, contains the coherent debiased intercept plus debiased slopes; for `post_selection_ols`, contains the active-set OLS/WLS refit embedded in the full parameter layout |
| `_inference_result` | Structured inference result and numerical-backend / node-wise tuning metadata |
| `aic` | Compatibility plug-in fit diagnostic when available; not a penalty-aware effective-DoF criterion |
| `bic` | Compatibility plug-in fit diagnostic when available; not a penalty-aware effective-DoF criterion |

Methods: `fit(X, y)`, `predict(X)`, `score(X, y)`, `summary()`

## Common pitfalls and complete API

- `summary()` requires successful inference; with the default `compute_inference=False`, use prediction/score and coefficient outputs instead.
- Post-selection OLS is a diagnostic on the selected variables, not a general correction for having searched for them. Debiased reporting coefficients can differ from prediction coefficients; see the inference section above.
- Hitting `max_iter` can indicate insufficient numerical accuracy. Inspect warnings and iteration count, check feature scaling, and compare results under tighter tolerance/larger iteration budget before interpretation.
- Explicit `device="cuda"` or `device="torch"` requires a usable corresponding GPU backend and does not silently switch to CPU. See [device and memory](../guides/device-and-memory.md).
- Check the [solver–penalty matrix](../guides/solver-penalty-matrix.md) before requesting a different solver; `device` does not replace `solver`.

The [complete ElasticNet API reference](../reference/linear-model-api.md#elasticnet) includes `predict(X, return_cpu=True)`, weighted `score(X, y, sample_weight=None)`, formula input, reporting fields and inference restrictions. Inherited methods have a [shared reference](../reference/estimator-api.md). Use the separate [ElasticNetCV constructor](../reference/linear-model-api.md#elasticnetcv) and [CV workflow/results](../reference/linear-model-api.md#cv-methods-and-results), including a runnable example; direct and CV parameters differ. Source inspection is supplementary, not a substitute for these contracts.

### Tuning a Ridge-like mixture

With `ElasticNetCV(l1_ratio=0)` or a very small positive ratio, supply an explicit
positive `alphas` grid spanning the shrinkage strengths you want to compare.
The automatic grid currently divides a weighted-average centered design-response cross-product
by `max(l1_ratio, 1e-6)`. At zero it can therefore contain only extremely large
penalties and miss useful Ridge fits. This is a grid-construction limitation;
direct `ElasticNet(l1_ratio=0, alpha=...)` still fits the Ridge objective.
The automatic rule also centers X/y even with `fit_intercept=False`; use an
explicit grid for that case and inspect the candidate range and validation losses.

## Numerical Validation

The maintained regression suite checks agreement across supported backends and reference implementations at tolerances appropriate to each dtype and solver path. Solver API migration behavior is covered by `dev/tests/test_penalized_solver_api_cleanup.py`; node-wise tuning is covered by `dev/tests/test_nodewise_alpha_inference_contract.py`; the post-selection OLS migration and active-set OLS/WLS behavior are covered by `dev/tests/test_post_selection_ols_inference_api.py`.

## References

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
