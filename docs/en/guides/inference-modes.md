# Inference Modes

> Language: English  
> Last updated: 2026-09-08  
> This page: Guide  
> Switch: [Chinese](../../cn/guides/inference-modes.md)

Language switch: [Chinese](../../cn/guides/inference-modes.md)

## Gaussian linear-model inference

For the shared Gaussian inference path used by squared-error L2/Ridge consumers,
numerical covariance and reference-distribution inference run on the backend that
executed the fitted model: NumPy, CuPy, or Torch. The numerical path includes the
bread/covariance calculation, standard errors, test statistics, p-values, and
confidence-interval critical values.

The established public reporting contract is intentionally unchanged: after all
numerical inference is complete, the inference result and estimator reporting
attributes (`_params`, `_bse`, `_tvalues`, `_pvalues`, `_conf_int`) take a final
NumPy snapshot. This is a reporting boundary, not a CPU inference fallback.
`_inference_result.metadata` records `numerical_backend`, `numerical_device`,
`reporting_backend="numpy"`, and
`reporting_boundary="post_numerical_inference"` for this shared path.

Explicit `device="cuda"` and `device="torch"` requests do not silently downgrade
Gaussian inference to NumPy. Missing or invalid executed-backend provenance fails
closed. `device="auto"` is the only mode that may select among available backends
automatically.

Supported covariance choices on the Gaussian path are:

- `nonrobust` — classical covariance with Student-t reference inference.
- `hc0`, `hc1`, `hc2`, `hc3` — heteroskedasticity-consistent sandwich covariance
  with normal reference inference.
- `hac` — Bartlett-kernel HAC covariance with normal reference inference.

The backend-native reference helper also preserves the maintained stable
Student-t identities at one and two residual degrees of freedom, avoiding
subtractive cancellation or an avoidable `t**2` overflow in representable
extreme tails.

## Sparse penalized-linear inference

For `Lasso`, `ElasticNet`, and the public generic
`PenalizedGeneralizedLinearModel(loss="squared_error", penalty="l1" | "elasticnet")`
entry point, statistical method identity and execution hardware are separate
controls. The maintained inference methods are:

- `debiased` — de-biased/de-sparsified coefficient inference.
- `post_selection_ols` — heuristic OLS/WLS refit on the active set selected by
  the penalized fit.
- `bootstrap` — residual-bootstrap inference where supported.

`post_selection_ols` is the canonical hardware-neutral spelling. The unified
aliases `cpu_ols` and `gpu_ols` are deprecated together and are accepted for one
compatibility cycle with `FutureWarning`; both normalize to
`post_selection_ols`. `LassoCV` also accepts the older
`cpu_ols_inference` / `gpu_ols_inference` spellings at its compatibility boundary
and normalizes them to the same method.

The method name does **not** choose a device. Device/backend routing follows the
estimator contract:

- explicit `device="cpu"` runs the NumPy CPU route;
- explicit `device="cuda"` requires CuPy CUDA and fails closed when unavailable;
- explicit `device="torch"` requires Torch CUDA and fails closed when unavailable;
- only genuine estimator/global `device="auto"` may preserve an already
  backend-native CuPy or Torch-CUDA input as part of automatic routing.

String and `Penalty`-object forms of the sparse Gaussian penalty participate in
the same migration and AUTO-routing contract.

Backend reuse is method-specific. `post_selection_ols` always reuses the
successful fit's recorded `_selected_backend_name` / `_selected_backend_device`,
and the maintained CuPy/Torch `debiased` routes keep their numerical inference on
the executed GPU backend. This includes scalar normal-reference critical values:
inside debiased GPU inference, scalar distribution calls are pinned to the
executed CuPy/Torch backend (and Torch concrete device) instead of re-resolving a
Python scalar to NumPy. Residual `bootstrap`, by contrast, currently uses a
CPU-native residual-refit implementation. An explicit GPU `device` therefore
controls the penalized fit but must not be interpreted as making residual
bootstrap GPU-native.

With analytic `sample_weight`, the maintained NumPy/CuPy/Torch `debiased` paths
use the same weighted-centered average-loss working problem. Multiplying every
weight by the same positive constant therefore leaves both the penalized fit and
the debiased inference unchanged.

Debiased inference also has explicit parameter ownership when an intercept is
fitted. Public `coef_` and `intercept_` remain the **penalized prediction fit**.
Inference reporting uses the debiased slope vector `theta_db = _params[1:]` and
the matching original-coordinate intercept
`_params[0] = ybar_w - xbar_w @ theta_db`. Therefore `_bse[0]`, the first
z-statistic/p-value, and `_conf_int[0]` refer to that debiased reporting
intercept, not to prediction `intercept_`. This preserves the ordinary feature-
translation identity: shifting the design by a constant vector `c` shifts the
reported debiased intercept by `-c @ theta_db` while leaving the debiased slopes
unchanged. Metadata records `intercept_estimator="centered_debiased"` and
`intercept_influence="centered_nodewise"`.

Weighted `LassoCV` uses that same analytic-weight convention for its default
alpha grid, every training-fold objective, weighted validation MSE, and the final
selected-alpha refit. Positive constant weights are treated as the exact
unweighted statistical problem, avoiding artificial floating-point differences.
Once AUTO routing resolves a concrete CPU/CuPy/Torch backend for CV, the final
`Lasso` refit stays on that same backend; explicit CPU also converts heterogeneous
GPU-resident inputs to NumPy before entering the dedicated CV selector. If the
final refit produces inference, the outer `LassoCV` exposes the same structured
`_inference_result` and matching `_params`/SE/statistic/p-value/CI reporting
surface. Its public `coef_`/`intercept_` still belong to the penalized prediction
refit, so the same ownership distinction applies there too.

For debiased simultaneous inference, ordinary `_conf_int` remains marginal.
`enable_simultaneous_inference=True` uses multiplier-bootstrap max-|Z|
calibration. When `simultaneous_include_intercept=True`, the same centered-
nodewise original-coordinate intercept influence used by the marginal SE is part
of the bootstrap maximum itself, not merely an extra reported interval row. A
successful refit clears the previous fit's simultaneous critical value, target
mask, joint intervals, and precision/influence state before computing the new
result.

### What `post_selection_ols` computes

The penalized model first selects an active set. statgpu then refits an
**unpenalized OLS or WLS model on exactly that active set** on the fit-resolved
backend and computes covariance/reference-distribution inference there. The
original penalized `coef_` remains the coefficient vector used for prediction;
the active-set refit is an inferential/reporting object in `_params` /
`_inference_result`.

The two fits have separate diagnostic ownership. In `summary()`, the coefficient
table and `Post-selection Refit DoF` belong to the active-set refit, while
R-squared, adjusted R-squared, F statistic, log-likelihood, AIC, BIC, and
`Penalized-fit Residual DoF` continue to describe the penalized prediction fit.
When the active design is rank deficient, the refit residual degrees of freedom
are `n - effective_rank`, not `n - active_column_count`; the coefficient refit
and covariance bread use a design-level Moore-Penrose/SVD calculation rather
than squaring the condition number through normal equations. Metadata records
`refit_rank`, `refit_parameter_count`, and `refit_rank_deficient`.

For `cov_type="nonrobust"`, this path preserves the established classical
**Student-t** reporting convention. Robust covariance choices exposed by the
estimator use the shared Gaussian robust-covariance layer and its normal-reference
reporting convention. If a no-intercept fit selects no features, all coefficient
entries are inactive compatibility placeholders; the result still preserves the
requested covariance/reference family (`nonrobust` -> Student-t, robust/HAC ->
normal) rather than silently rewriting the request.

The full reporting arrays preserve one compatibility detail from the old
`cpu_ols` surface: coordinates that were not selected are represented with
`SE=0`, statistic `0`, `p=1`, and `[0, 0]` confidence-interval placeholders.
Those values are **not inferential claims that the coefficient is known exactly**.
Use `_inference_result.metadata["selected_feature_indices"]` to identify the
coordinates that actually received the active-set OLS/WLS calculation.

This remains a post-selection diagnostic. Ordinary OLS/WLS intervals formed
after choosing variables from the same data are not general selective-inference
confidence intervals.

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    compute_inference=True,
    inference_method="post_selection_ols",
)
model.fit(X, y)

# Prediction still uses the penalized fit.
penalized_coef = model.coef_

# Reporting/inference uses the active-set OLS/WLS refit.
post_selection_params = model._params
```

For high-dimensional coefficient inference rather than an engineering
post-selection diagnostic, prefer `inference_method="debiased"` and check its
statistical assumptions. Lasso's simultaneous max-|Z| path is a separate
procedure from ordinary marginal intervals and from p-value adjustment.

## Related robust covariance support

- `LinearRegression(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
- `Ridge(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
- sparse Gaussian `post_selection_ols` uses the same Gaussian covariance layer
  for covariance choices exposed by the estimator.
- `LogisticRegression(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
