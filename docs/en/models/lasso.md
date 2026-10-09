# Lasso

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/lasso.md)

## Why use Lasso?

Lasso fits a linear prediction model while shrinking coefficients toward zero.
Its L1 penalty can make some coefficients exactly zero, producing a sparse model
when many candidate predictors may contribute little. The selected variables
can change substantially when predictors are strongly correlated; use
[Elastic Net](elastic-net.md) when retaining correlated groups matters, or
[ordinary least squares](linear-regression.md) for an unpenalized low-dimensional
model. A nonzero Lasso coefficient alone is not a significance test or a causal effect.

Import `Lasso` from `statgpu` or `statgpu.linear_model`. For prediction, begin with
`compute_inference=False`; the constructor default is `True` and requests
additional debiased inference. Choosing a useful prediction penalty and making
coefficient-uncertainty claims are separate tasks. Complete method signatures,
parameter defaults and outputs are in the [Lasso API reference](../reference/linear-model-api.md#lasso).

## Model and objective

For n observations, p features, response y and design X of shape `(n,p)`, Lasso solves

$$
\min_{b,\beta}\frac{1}{2n}\sum_{i=1}^n(y_i-b-x_i^\top\beta)^2
+\alpha\|\beta\|_1.
$$

The intercept b is unpenalized; `fit_intercept=False` fixes b=0. Larger `alpha`
means stronger shrinkage under this **average-loss** convention. With analytic
weights, the squared-loss term becomes

$$
\frac{1}{2\sum_{i=1}^n w_i}\sum_{i=1}^n w_i(y_i-b-x_i^\top\beta)^2,
$$

with finite nonnegative weights of positive total. Multiplying every weight by
the same positive constant does not change the objective. Feature units affect
the penalty: learn any scaling on training rows only, and apply it unchanged to
held-out rows. Do not assume Lasso automatically standardizes prediction features.

## A complete CPU example

The simulated predictors already have comparable scales. The last 60 rows are
held out before fitting; only the first two columns generate the signal.

Run the following steps in order in one Python session. Start with the imports.

<!-- learner-example: lasso-prediction -->
```python
import numpy as np
from statgpu.linear_model import Lasso
```

<a id="cpu-data"></a>

### Prepare training and test data

`X` has shape `(240, 6)`: each row is one observation and each column is a predictor. `y` is a one-dimensional continuous response with shape `(240,)`. Keep the last 60 rows out of fitting and tuning.

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(240, 6))
y = 1 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.3, size=240)
X_train, X_test = X[:180], X[180:]
y_train, y_test = y[:180], y[180:]
```

### Fit the prediction model

Use a fixed illustrative penalty first. Inference is disabled so this step only estimates the coefficients used for prediction.

```python
model = Lasso(
    alpha=0.1, device="cpu", solver="coordinate_descent",
    max_iter=5000, tol=1e-10, compute_inference=False,
)
model.fit(X_train, y_train)
```

### Predict and inspect the fit

Pass the test features in the same column order. `score` evaluates R² against the held-out responses.

```python
prediction = model.predict(X_test)
print("Slopes:", np.round(model.coef_, 3))
print("Test R2:", round(model.score(X_test, y_test), 3))
```
<!-- example-end: lasso-prediction -->

For this seed, the slopes round to `[1.887, -0.880, 0, 0, 0, 0]` and held-out
R² is about `0.980`. The first two slopes are shrunk relative to the generating
values 2 and −1. `prediction` has shape `(60,)`; `coef_` has one entry per input
feature and `intercept_` is a separate scalar. Other datasets can select noise
or omit real signals. This test score measures prediction, not interval coverage.

## Choosing alpha and checking the fit

- Pick `alpha` using training-only validation or `LassoCV`; keep the test set out
  of tuning. The value 0.1 above is illustrative, not a universal default choice.
- Compare held-out error and coefficient stability across plausible penalties.
  If preprocessing is learned, repeat it within each validation training fold.
- `tol` and `max_iter` control numerical work, not sparsity or statistical
  uncertainty. Check stability under a tighter tolerance and larger budget;
  an iteration count alone does not certify optimality.
- For uncertainty, choose an inference method deliberately. Tuning alpha or
  selecting variables on the same data adds uncertainty that the methods below
  do not automatically remove. See [cross-validation](../guides/cross-validation.md).

## Solvers and current control limitations

`solver` selects the algorithm; `device` selects its execution location. Use
`solver="coordinate_descent"` for CPU coordinate descent or `solver="fista"`
for a supported CPU/GPU proximal fit. Other combinations are described in the
[solver–penalty reference](../guides/solver-penalty-matrix.md).

The current direct Gaussian fit stores `stopping="coef_delta"` or `"kkt"`, but
**does not use this setting to select its stopping check**. CPU FISTA and
coordinate descent check the sum of absolute coefficient changes; the GPU
FISTA paths also check coefficient movement. ADMM uses primal/dual residuals.
Thus `stopping="kkt"` does not certify a direct fit's KKT residual. Ill-scaled
features can make movement small while optimality error remains large; scale
features appropriately and check the objective/KKT residual when accuracy matters.
The separate Lasso CV/path helper has its own stopping behavior; a successful
path check does not certify the final direct-model refit.

`admm_rho` is currently stored but ignored by unified ADMM, which starts with
rho=1.0. Whether rho adapts depends on the solver path: the direct squared-error
Cholesky solve keeps it fixed, while other paths can adapt it. The constructor
setting is not an effective tuning lever, including when forwarded to a
`LassoCV` final refit. `cpu_solver` is a deprecated
compatibility argument; use `solver` to select the direct algorithm. See the
[solver API migration guide](../guides/penalized-solver-api-migration.md).

## Covariance/Inference

`Lasso` inference is controlled by `inference_method`. Ordinary `auto` resolves
to `debiased`; when enabling simultaneous inference, use the explicit
`debiased` spelling because the current constructor rejects `auto` in that
combination. The default remains `debiased`:

- `post_selection_ols`: hardware-neutral active-set OLS/WLS refit diagnostic.
- `debiased`: de-biased (de-sparsified) Lasso inference with z-statistic semantics.
- `bootstrap`: unweighted Gaussian residual bootstrap for the penalized coefficient distribution; typically slower and not a universal correction for model-selection uncertainty.

The unified spellings `cpu_ols` and `gpu_ols` are deprecated together. During the compatibility window they emit `FutureWarning` and normalize to `post_selection_ols`; they are **not** separate CPU and GPU statistical procedures. `LassoCV` additionally accepts the older `cpu_ols_inference` / `gpu_ols_inference` spellings at its compatibility boundary and normalizes them to the same method.

### What `post_selection_ols` computes

The penalized fit first chooses an active feature set. statgpu then refits an **unpenalized OLS model on exactly those selected columns**, or WLS when `sample_weight` is supplied, using the backend/device recorded by the successful penalized fit. Gaussian covariance and reference-distribution inference are computed on that same numerical backend before the established NumPy reporting snapshot is taken.

The original penalized `coef_` remains the coefficient vector used for prediction. The active-set OLS/WLS refit is exposed for inference/reporting through `_params`, `_inference_result`, `_bse`, `_tvalues`/`_zvalues`, `_pvalues`, and `_conf_int`.

Validity notes:

- `post_selection_ols` is a heuristic post-selection diagnostic. Its intervals should not be interpreted as general selective-inference confidence intervals after choosing variables from the same data.
- The ordinary `debiased` `_conf_int` is marginal per coefficient. Simultaneous/joint family-wise coverage requires the dedicated simultaneous inference path.
- Rank-deficient active refits use the effective design rank for residual degrees of freedom and a design-level Moore-Penrose/SVD refit rather than squaring the condition number through normal equations.

### Device/backend rule

`inference_method` describes **what statistical procedure is computed**; it does not choose hardware.

- explicit `device="cpu"` -> NumPy CPU;
- explicit `device="cuda"` -> CuPy CUDA only, raising an error if unavailable;
- explicit `device="torch"` -> Torch CUDA only, raising an error if unavailable;
- only genuine estimator/global `device="auto"` may preserve an already backend-native CuPy or Torch-CUDA input during automatic routing.

`post_selection_ols` and marginal `debiased` inference reuse the successful
fit's numerical backend. With `fit_intercept=True`, simultaneous debiased
inference also computes on that backend and returns NumPy reporting arrays.
With `fit_intercept=False`, the simultaneous calculation uses a NumPy host
helper even after a GPU fit; it is not GPU-native. See the
[Lasso reference](../reference/linear-model-api.md#lasso) for result metadata.

Residual `bootstrap` uses a fixed-design residual-refit procedure. For each draw,
statgpu resamples the fitted residuals with replacement, forms a bootstrap
response around the fitted values, and refits the same Lasso configuration.
`n_bootstrap` controls the number of refits and `bootstrap_random_state` controls
reproducibility.

Bootstrap refits follow the backend and concrete device of the successful fit:
CPU fits use NumPy, while CuPy or Torch CUDA fits keep the refits on the same GPU
device. This changes **where** the bootstrap runs, not its statistical definition.
Final inference arrays use the standard NumPy reporting boundary.

Residual bootstrap requires `sample_weight=None` and the nonrobust covariance
configuration (`cov_type="nonrobust"`, the Lasso default; it is not a Lasso
constructor option).
Weighted residual bootstrap, robust/HC or HAC/block bootstrap, non-Gaussian
bootstrap, and Cox bootstrap are unsupported and raise an error. The resulting
intervals describe this fixed-design, fixed-tuning penalized-estimator bootstrap;
they do not automatically account for variable-selection uncertainty.

With analytic weights, direct Lasso and debiased inference use the same
weighted-centered average-loss convention on NumPy/CuPy/Torch, so multiplying all
weights by one positive constant does not change the statistical problem.
The training `rsquared`/`rsquared_adj` properties have a separate limitation:
after weighted debiased inference they re-center the transformed working
response, so they need not describe R² on the original weighted observations.
The residual-based `fvalue`/`f_pvalue` diagnostics use the same incorrect total.
Use `model.score(X, y, sample_weight=weights)` with original data and finite,
nonnegative evaluation weights of the correct length and positive sum. Validate
those weights yourself; the shared score path does not reliably reject negative
weights. See the [weighted-score example](elastic-net.md#weighted-training-diagnostics).

`LassoCV` uses the same convention for the default alpha grid, every weighted
training fold, validation MSE, and final refit. Constant positive weights take the
exact unweighted CV path. Once AUTO resolves a concrete backend for CV, the final
selected-alpha `Lasso` refit remains on that backend.

### Node-wise tuning for debiased inference

`alpha` controls the main penalized Lasso fit. `nodewise_alpha` is a **different** tuning parameter used only by `inference_method="debiased"` to construct the node-wise Lasso approximation to the design precision matrix.

If `nodewise_alpha` is supplied, that positive scalar is used exactly on the standardized node-wise design. If it is omitted (`None`), statgpu standardizes the centered/weighted working design and uses the response-independent default

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

where `n_nw=n` without analytic weights and a Kish-style effective sample size is used with non-uniform analytic weights. The `sqrt(log p / n)` order is theory-motivated; the exact constant and weighted effective-sample-size convention are statgpu defaults, not a unique theorem-mandated choice. In particular, changing only the units of `y` no longer changes the design-side precision construction.

The node-wise problems are solved on a standardized design and the resulting precision estimate is transformed back to the original working-feature scale. An independent KKT check is required before inference is published. `nodewise_alpha_` records the resolved value after successful multi-feature debiased inference, and `_inference_result.metadata` records the requested/resolved value, source, effective sample size, solver settings, and maximum KKT residual. For a one-feature problem there is no nuisance node-wise regression: statgpu uses the analytic univariate precision and leaves `nodewise_alpha_` as `None`.

For `LassoCV`, `nodewise_alpha` is final-refit inference configuration only. It does not participate in the main `alpha` grid, fold scoring, or alpha selection.

### Debiased intercept ownership

With `inference_method="debiased"`, prediction and inference intentionally expose
different intercept estimates when an intercept is fitted. Public `coef_` and `intercept_` remain the
**penalized prediction fit**. Inference reporting uses
`theta_db = _params[1:]` and the matching original-coordinate intercept
`_params[0] = ybar_w - xbar_w @ theta_db`.

Consequently, `_bse[0]`, the first z-statistic/p-value, and `_conf_int[0]` describe
the debiased reporting intercept, not `intercept_`. Shifting every feature by a
constant vector `c` leaves the debiased slopes unchanged and shifts `_params[0]`
by `-c @ theta_db`, preserving one coherent parameterization. The structured
intercept-fit result records `intercept_estimator="centered_debiased"` and
`intercept_influence="centered_nodewise"` in metadata. Without an intercept,
all `_params` entries are slopes and `intercept_` is zero.

For `LassoCV(compute_inference=True, inference_method="debiased")`, the outer CV
estimator exposes the same final-refit `_inference_result` and matching
`_params`/SE/statistic/p-value/CI reporting surface as `estimator_`. Its public
`coef_`/`intercept_` still belong to the penalized selected-alpha prediction
refit.

### Simultaneous debiased inference

With `enable_simultaneous_inference=True`, Lasso calibrates a multiplier-bootstrap
max-|Z| critical value. The ordinary `_conf_int` remains marginal; the joint
intervals are stored separately in `_conf_int_simultaneous`.

`simultaneous_alpha` must lie strictly in `(0, 1)` and
`simultaneous_n_bootstrap` must be a positive integer. These controls are validated before
NumPy/CuPy/Torch backend dispatch.

`simultaneous_include_intercept=False` calibrates the family over feature
coefficients only. With `simultaneous_include_intercept=True`, the same centered-
nodewise original-coordinate intercept influence used by the marginal debiased
SE is part of the bootstrap maximum itself. It is therefore not merely an extra
output row receiving a feature-only critical value. On CuPy/Torch with
`fit_intercept=True`, this centered simultaneous calculation is backend-native as
described above. Every successful refit clears any previous simultaneous critical
value, target mask, intervals, and precision/influence state before publishing
the new result.

## Parameters

This table is the complete public constructor inventory for `statgpu.linear_model.Lasso`. Method signatures, input/output shapes, fitted attributes, formula inputs and inherited helpers are in the [complete Lasso API reference](../reference/linear-model-api.md#lasso). Unlike ElasticNet, Lasso has no `l1_ratio`, `cov_type`, or `hac_maxlags` constructor controls.

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | Nonnegative L1 regularization strength under average squared loss. |
| `fit_intercept` | `True` | Whether to fit an intercept. |
| `max_iter` | `1000` | Maximum optimization iterations. |
| `tol` | `1e-4` | Convergence tolerance. |
| `stopping` | `"coef_delta"` | Stored `coef_delta` / `kkt` request; currently ignored by direct-fit stopping checks. See the solver limitations above. |
| `inference_method` | `"debiased"` | `post_selection_ols` / `debiased` / `bootstrap`; ordinary `auto` resolves to `debiased`. With `enable_simultaneous_inference=True`, explicitly use `debiased`: the constructor currently rejects `auto`. Deprecated `cpu_ols` and `gpu_ols` aliases remain temporarily accepted. |
| `nodewise_alpha` | `None` | Node-wise Lasso penalty for `debiased` inference. Explicit positive values override the standardized design-side automatic rule. |
| `n_bootstrap` | `200` | Residual-bootstrap refit count; use at least 2 draws. |
| `bootstrap_random_state` | `None` | RNG seed for residual-bootstrap inference. |
| `enable_simultaneous_inference` | `False` | Enable simultaneous inference (debiased only). |
| `simultaneous_method` | `"maxz_bootstrap"` | Simultaneous-inference method; currently `maxz_bootstrap`. |
| `simultaneous_alpha` | `0.05` | Simultaneous family-wise error level; must be strictly in `(0, 1)` when simultaneous inference is enabled. |
| `simultaneous_n_bootstrap` | `1000` | Positive integer multiplier-bootstrap draw count for max-\|Z\| calibration when simultaneous inference is enabled. |
| `simultaneous_random_state` | `None` | RNG seed for simultaneous bootstrap. |
| `simultaneous_include_intercept` | `False` | Whether the debiased intercept is included in both the simultaneous target set and max-\|Z\| calibration family. |
| `device` | `"auto"` | Execution device: `auto`, `cpu`, `cuda` (CuPy), or `torch` (Torch CUDA). |
| `n_jobs` | `None` | Shared CPU worker setting; not a solver selector or parallel-fit guarantee. |
| `compute_inference` | `True` | Whether to compute post-fit inference. |
| `solver` | `"fista"` | Backend-neutral direct-fit solver; use `coordinate_descent` for the CPU CD path or another supported solver as appropriate. |
| `cpu_solver` | `"coordinate_descent"` | **Deprecated compatibility parameter.** It does not select the current direct-fit algorithm; use `solver` instead. |
| `lipschitz_L` | `None` | Optional user-supplied Lipschitz constant for compatible iterative solvers. |
| `admm_rho` | `1.0` | Stored request; unified ADMM currently ignores it and starts with rho=1.0. Adaptation depends on the solver path; the direct Cholesky solve keeps rho fixed. |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory cleanup after fit where supported. |

## A complete simultaneous-inference example

Continue after the complete [CPU example](#a-complete-cpu-example). This section reuses its `X_train` and `y_train`; no test responses enter inference. The intervals condition on fixed alpha and do not account for selecting alpha on the same responses.

<!-- example-requires: lasso-prediction -->
<!-- learner-example: lasso-simultaneous -->
```python
m_sim = Lasso(
    alpha=0.1, device="cpu", solver="coordinate_descent",
    max_iter=5000, tol=1e-10, inference_method="debiased",
    enable_simultaneous_inference=True, simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=200, simultaneous_random_state=7,
    simultaneous_include_intercept=True,
)
m_sim.fit(X_train, y_train)
```

The ordinary intervals remain marginal. Read the separate simultaneous array for the family that includes all six slopes and the intercept:

```python
ci_marginal = m_sim._conf_int
ci_simul = m_sim._conf_int_simultaneous
print(ci_marginal.shape, ci_simul.shape)
```
<!-- example-end: lasso-simultaneous -->

Both shapes are `(7,2)` here: the intercept precedes six slopes. The 200 draws
keep this demonstration small; more draws improve Monte Carlo precision.
For optional GPU execution, use an installed CuPy/Torch CUDA backend and follow
the [device guide](../guides/device-and-memory.md); the no-intercept boundary
described above still applies.

## Choosing an inference method

`debiased` is the main high-dimensional coefficient-inference path. `post_selection_ols` is a lighter active-set OLS/WLS diagnostic, while `bootstrap` is a more expensive resampling path. Their statistical claims are different and should not be treated as interchangeable.

## Outputs

- Penalized prediction fit: `intercept_`, `coef_`, `n_iter_`
- Inference (if enabled): `_params`, `_bse`, `_tvalues` / `_zvalues`, `_pvalues`, `_conf_int`, `_inference_result`
- Successful multi-feature `debiased` inference: `nodewise_alpha_` records the resolved node-wise tuning value; p=1 and non-node-wise inference leave it `None`.
- Under `inference_method="post_selection_ols"`, `coef_` remains penalized while `_params` contains the active-set OLS/WLS refit embedded in the full parameter layout.
- Under `inference_method="debiased"` with an intercept, `_params[1:]` contains debiased slopes and `_params[0]` contains their matching original-coordinate debiased intercept. Without an intercept, all rows correspond to input features. `_conf_int` is marginal per reported parameter.
- With simultaneous inference enabled, `_conf_int_simultaneous` stores joint intervals over the configured target family (`maxz_bootstrap`); when the debiased intercept is included it also participates in the max-|Z| calibration.
- Methods: `fit`, `predict`, `score`, `summary`, `get_params`, `set_params`, plus inherited `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic` and `permutation_test`; see the [Lasso API reference](../reference/linear-model-api.md#lasso) for arguments and returns.
- Common diagnostics include `aic` and `bic` when available.

## FAQ

- Why can CPU and GPU iteration counts differ under the same `tol`? Different numerical backends and solver implementations can converge differently; compare under fixed `solver`, tolerance and data scaling. The direct `stopping` setting is currently ineffective.
- Should CPU users set `cpu_solver`? No. Use `solver`; `cpu_solver` is a deprecated compatibility argument from the previous CPU/GPU-split API.
- What is the difference between `alpha` and `nodewise_alpha`? `alpha` defines the penalized prediction fit. `nodewise_alpha` is used only to estimate the approximate precision matrix for debiased inference.
- Should I choose `cpu_ols` or `gpu_ols` based on hardware? No. Both are deprecated aliases for `post_selection_ols`. Choose the statistical method with `inference_method` and the execution location with `device`.
- Does `post_selection_ols` change `coef_`? No. Prediction keeps the penalized coefficients; the active-set refit lives in inference/reporting fields such as `_params` and `_inference_result`.
- Why can `intercept_` differ from `_params[0]` under `debiased` with an intercept? `intercept_` belongs to the penalized prediction fit, while `_params[0]` is the intercept paired with the debiased slope vector used by statistical reporting.
- When should I use `debiased`? Prefer it when you need coefficient-level inference in high-dimensional sparse settings, subject to the method's assumptions.
- Is `post_selection_ols` a valid selective-inference confidence procedure? No. Treat it as a post-selection diagnostic.
- Are ordinary `debiased` intervals simultaneous/joint confidence regions? No. Ordinary `_conf_int` values are marginal. Enable the dedicated simultaneous path when family-wise intervals are required.
- How do I include the intercept in simultaneous coverage? Set `simultaneous_include_intercept=True`; the debiased intercept then participates in the bootstrap max-|Z| calibration as well as the reported joint interval set.

Complete [LassoCV controls and result schema](../reference/linear-model-api.md#lassocv)
and a [standalone CPU tuning example](../reference/linear-model-api.md#ridgecv-and-lassocv-cpu-example)
are available in the API reference. Direct and CV constructor controls differ.

## External Validation

For a comparison with another Lasso implementation, align the average-loss
objective, feature scaling, intercept treatment, sample weights, `alpha` and
solver tolerance. Compare prediction coefficients separately from debiased or
post-selection reporting estimates: they target different quantities. For
intervals, also match the inference method and covariance assumptions; numerical
agreement does not correct uncertainty from selecting variables or tuning alpha.

Contributors can consult the [validation reference](../../../dev/references/model-validation.md#lasso-and-elastic-net).

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Buhlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- van de Geer, S., Buhlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166-1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217-242. [https://doi.org/10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869-2909. [https://jmlr.org/papers/v15/javanmard14a.html](https://jmlr.org/papers/v15/javanmard14a.html)
