# Quantile Regression

> Language: English  
> Last updated: 2026-10-09<br>
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/quantile.md)

## When to use quantile regression

Quantile regression models a location in the conditional response distribution: `quantile=0.5` estimates the conditional median, and `0.9` estimates its 90th percentile. Use it to study tails or asymmetric responses. If the scientific question concerns the conditional mean, consider a mean model such as ordinary linear regression.

`QuantileRegression` provides unpenalized fitting and inference under the conditions below; `PenalizedQuantileRegression` adds regularization. Both use the asymmetric absolute loss called check loss or pinball loss. Median regression penalizes large response residuals linearly rather than quadratically; that does not automatically protect against unusual predictor values.

<a id="cpu-example"></a>

## A complete CPU example

Run these steps in order to fit a conditional median and evaluate it on rows not used for fitting.

### 1. Import

<!-- learner-example: quantile-basic -->
```python
import numpy as np
from statgpu.linear_model import QuantileRegression
```

### 2. Prepare a continuous response

`X` is a `(320, 2)` matrix, with observations in rows and numeric features in columns; `y` is a length-320 continuous response. The symmetric noise has conditional median zero, so the true conditional median is `1 + X @ [1.5, -0.7]`. Train on the first 240 rows and hold out 80. Handle nonfinite values in real data and preserve feature order for prediction.

```python
rng = np.random.default_rng(23)
X = rng.normal(size=(320, 2))
y = 1.0 + X @ np.array([1.5, -0.7]) + rng.normal(scale=0.6, size=320)
```

### 3. Fit the conditional median

`quantile=0.5` selects the median. Leave inference off while learning the fitted coefficients and predictions.

```python
model = QuantileRegression(
    quantile=0.5, device="cpu", max_iter=3000, tol=1e-6,
).fit(X[:240], y[:240])
print("Slopes:", np.round(model.coef_, 3))
```

Slopes round to `[1.476, -0.636]`. Holding the other feature fixed, a unit increase in the first feature increases the fitted conditional median by about 1.476 response units. This interpretation concerns the selected quantile, not automatically a causal or mean effect.

### 4. Predict and evaluate

```python
prediction = model.predict(X[240:])
pinball_loss = -model.score(X[240:], y[240:])
print("Predicted medians:", np.round(prediction[:3], 3))
print("Held-out pinball loss:", round(float(pinball_loss), 3))
```

Predictions have shape `(80,)`; the first three round to `[1.429, 1.455, 1.632]`. Held-out pinball loss is about `0.255`; lower is better on fixed evaluation data at the same target quantile. `score` returns its negative, not R². A quantile prediction is not a confidence interval for an individual future observation.
<!-- example-end: quantile-basic -->

## Choosing settings and checking results

- Choose `quantile` from the question, rather than changing the target to whichever quantile has the lowest training loss. Extreme quantiles need more data.
- Feature scale affects shrinkage in penalized fits. Learn scaling and `alpha` within training folds and keep a final test set separate.
- Separately fitted quantiles can cross; this API does not automatically impose noncrossing constraints.
- Inspect numerical warnings and iteration stability. `n_iter_` is not an accuracy guarantee, and a larger budget cannot fix an unidentified design.

## Objective function

For quantile $\tau\in(0,1)$, the check / pinball loss is

$$
\ell(\eta,y)=\rho_\tau(y-\eta),
\qquad
\rho_\tau(u)=u\left(\tau-\mathbf 1\{u<0\}\right).
$$

Equivalently,

$$
\rho_\tau(u)=
\begin{cases}
\tau u, & u\ge 0,\\
(\tau-1)u, & u<0.
\end{cases}
$$

The asymmetric linear slopes select the requested conditional quantile. “Check loss” is the traditional quantile-regression term; “pinball loss” is a common modern name referring to the same piecewise-linear shape.

At $\tau=0.5$,

$$
\rho_{0.5}(u)=\frac12|u|,
$$

so median regression differs from least absolute deviations only by a constant scale factor.

A per-observation subgradient is

$$
\frac{\partial\ell}{\partial\eta}
=-\tau+\mathbf 1\{y-\eta<0\},
$$

with the usual subgradient interpretation at zero residual. The gradient is a step function, so `has_hessian=False` and `smooth_gradient=False`.

## Key parameter

The table selects the target-quantile control only. Complete standalone parameters, inference controls, and methods are in the [QuantileRegression public implementation](../../../statgpu/linear_model/wrappers/_quantile.py) or `help(QuantileRegression)`. For the penalized class, see its [public implementation](../../../statgpu/linear_model/penalized/_penalized_quantile.py) and [shared penalized API](../reference/linear-model-api.md#penalizedgeneralizedlinearmodel).

| Parameter | Default | Description |
|---|---:|---|
| `quantile` | `0.5` | Target quantile in `(0,1)`; `0.5` is median regression |

## Advanced usage

### Standalone coefficient inference

Standalone kernel/bootstrap inference is defined only for omitted or uniform `sample_weight`; genuinely non-uniform analytic weights are estimation-only on this class and raise when `compute_inference=True`. The bootstrap implementation is an **i.i.d. residual bootstrap**: fitted residuals are first centered at their empirical target-quantile so the bootstrap error distribution has empirical τ-quantile zero, then the centered residuals are resampled as exchangeable draws and refitted with backend-native batched Quantile IRLS/MM. It is not a wild/multiplier bootstrap and does not claim heteroscedastic-robust coverage; heteroscedastic quantile-regression bootstrap inference requires a different resampling construction. Bootstrap inference also requires `n_bootstrap >= 2`. Kernel inference additionally requires the selected bandwidth rule to keep `q ± h` inside `(0, 1)` and to produce a finite positive residual-density estimate at zero; otherwise inference raises instead of publishing non-finite standard errors.

Reuse the imports and training data from the completed [CPU example](#cpu-example). The kernel-density method below estimates coefficient uncertainty; its intervals are for coefficients, not future responses.

<!-- example-requires: quantile-basic -->
<!-- learner-example: quantile-inference -->
```python
inference_model = QuantileRegression(
    quantile=0.5, device="cpu", max_iter=3000, tol=1e-6,
    compute_inference=True, inference_method="kernel",
    kernel="epa", bandwidth="hsheather",
).fit(X[:240], y[:240])
print("Standard errors:", inference_model._bse)
print("Intervals:", inference_model._conf_int)
```
<!-- example-end: quantile-inference -->

Here the interval array has shape `(3, 2)`: intercept first, then two slopes. `_bse` and `_pvalues` use the same order.

### Add an L2 penalty

Reuse `X`, `y` from the completed [CPU example](#cpu-example). `alpha` controls shrinkage; the example value is illustrative and should be selected using training-only validation. For L2/no penalty, `solver="auto"` selects IRLS.

<!-- example-requires: quantile-basic -->
<!-- learner-example: quantile-penalized -->
```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

penalized_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="l2", alpha=0.1, device="cpu",
).fit(X[:240], y[:240])
penalized_prediction = penalized_model.predict(X[240:])
```
<!-- example-end: quantile-penalized -->

### Optional: a SCAD penalty

Complete the [CPU example](#cpu-example) and “Add an L2 penalty” first, then reuse their import and training data. Scalar SCAD/MCP uses specialized Proximal IRLS-CD continuation for nonconvex penalties; assess tuning and numerical stability separately.

<!-- example-requires: quantile-penalized -->
<!-- learner-example: quantile-scad -->
```python
scad_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="scad", alpha=0.1, device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: quantile-scad -->

### Explicit IRLS or FISTA

Complete the [CPU example](#cpu-example) and “Add an L2 penalty” first, then reuse their import and training data. Keep the target and penalty fixed while changing the algorithm. `auto` and explicit IRLS use the same route; explicit ordinary FISTA really runs FISTA, rather than redirecting to IRLS. Nonsmooth penalties such as ElasticNet already use ordinary FISTA.

<!-- example-requires: quantile-penalized -->
<!-- learner-example: quantile-explicit-solvers -->
```python
irls_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="l2", alpha=0.01, solver="irls", device="cpu",
).fit(X[:240], y[:240])
```

Request FISTA for the same data and objective:

```python
fista_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="l2", alpha=0.01, solver="fista", device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: quantile-explicit-solvers -->

### Weighted quantile regression

Complete the [CPU example](#cpu-example) and “Add an L2 penalty” first, then reuse their import and training data. This length-240 finite nonnegative weight vector gives the first 50 training observations more influence in the fitting objective; the weight sum must be positive.

<!-- example-requires: quantile-penalized -->
<!-- learner-example: quantile-weighted -->
```python
sample_weight = np.ones(240)
sample_weight[:50] = 5.0
weighted_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="l2", alpha=0.01, device="cpu",
).fit(X[:240], y[:240], sample_weight=sample_weight)
```
<!-- example-end: quantile-weighted -->

This demonstrates an estimation path that supports analytic weights, not universal support across low-level solvers or inference methods.

### GPU (Torch CUDA)

Complete the [CPU example](#cpu-example) and “Add an L2 penalty” first, then reuse the `PenalizedQuantileRegression` import and `X`, `y`. Request Torch CUDA explicitly; an installed Torch package and usable CUDA device are required, and an unavailable explicit device raises.

```python
gpu_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="scad", alpha=0.1, device="torch",
).fit(X[:240], y[:240])
```

## Solver compatibility

The support column below first describes unweighted algorithm availability. With `sample_weight`, the selected route must also satisfy the corresponding weighted capability contract; unweighted solver support does not imply arbitrary non-uniform weight support.

| Solver | Support | Notes |
|--------|:---:|-------|
| Proximal IRLS-CD | ✅ | Specialized IRLS majorization + LLA for scalar SCAD/MCP; supports analytic `sample_weight` |
| Group Proximal IRLS-LLA | ✅ | Automatic Group SCAD/MCP route; Quantile IRLS/MM plus a convex Adaptive-Group-Lasso weighted least-squares solve on the selected backend |
| IRLS | ✅ | **Default `solver="auto"` route for L2/no-penalty Quantile objectives**; `QuantileLoss.irls()` has an explicit `sample_weight` path |
| FISTA | ✅ | Used for L1/ElasticNet and related proximal routes; explicit L2/no-penalty `solver="fista"` also executes ordinary FISTA, while `auto` continues to prefer IRLS there |
| FISTA-BB | ❌ | BB step sizes use smooth-gradient differences as local-curvature estimates. Quantile has a step-function subgradient, so estimator/CV requests and public low-level `fista_bb_solver(QuantileLoss, ...)` raise an error |
| L-BFGS | ✅ (unweighted/uniform at the low-level boundary) | Public `PenalizedQuantileRegression` / `PenalizedGLM_CV` explicit L-BFGS requests are unsupported. Direct low-level `lbfgs_solver(QuantileLoss, ...)` retains its historical unweighted/uniform compatibility; genuine non-uniform weights are rejected |
| ADMM | ❌ as a direct Quantile solver | Public `admm_solver(QuantileLoss, ...)` is unsupported because its generic w-update assumes a smooth loss. The automatic Group Proximal IRLS-LLA route may internally use ADMM only after Quantile IRLS has produced a smooth weighted least-squares surrogate |
| Newton | ❌ | Quantile loss has no Hessian |
| Proximal Newton | ❌ | Quantile loss has no Hessian |

For L2/no-penalty Quantile objectives, `PenalizedQuantileRegression(..., solver="auto")` resolves to IRLS and explicit `solver="irls"` requests the same algorithm. Explicit ordinary `solver="fista"` is also supported: it executes the generic FISTA engine and is never redirected to IRLS. IRLS remains the automatic/default choice because the pinball loss is non-smooth and ordinary Quantile FISTA is a first-order proximal/subgradient route rather than a claim of textbook smooth-FISTA convergence. FISTA-BB remains unsupported because its BB curvature update specifically requires meaningful smooth-gradient differences.

## Penalty compatibility

| Penalty | Main `solver="auto"` route | Notes |
|---------|----------------------------|-------|
| l2 / none | IRLS | `none` is canonicalized to `L2(alpha=0)`; explicit `irls` selects the same route and explicit ordinary `fista` is available when requested |
| l1 / elasticnet | FISTA | Proximal/subgradient route |
| SCAD / MCP | Proximal IRLS-CD | Specialized Quantile IRLS majorization + LLA |
| adaptive_l1 | FISTA | Adaptive weights are prepared first, then the Quantile FISTA route is used |
| group_lasso / adaptive group | Group FISTA | Group-aware proximal route |
| group_scad / group_mcp | Group Proximal IRLS-LLA | Group LLA with Quantile IRLS/MM; each convex Adaptive-Group-Lasso weighted least-squares surrogate is solved on the selected backend |

The table describes the automatic route. An explicit Group SCAD/MCP `solver="fista"` request remains an explicit proximal-FISTA request; it is not silently rewritten into Group Proximal IRLS-LLA. The public low-level `fista_lla_path` performs the same dedicated Proximal IRLS-LLA solve for scalar Quantile SCAD/MCP; group penalties and warm-started/path-reporting low-level calls keep the fused FISTA-LLA engine. The delegated route counts IRLS iterations in `n_iter` and reports its own budget-exhaustion warning.

## `sample_weight` semantics

On quantile routes that explicitly support non-uniform analytic weights, the data-fit objective is the weighted check/pinball loss. With per-observation loss $\rho_\tau(r_i)$,

$$
L_w(\beta)
=\frac{\sum_i w_i\rho_\tau(y_i-x_i^\top\beta)}{\sum_i w_i}.
$$

But `sample_weight` is **not one universal solver capability**. In particular:

- Quantile IRLS / Proximal IRLS-CD have explicit weighted implementations;
- Group Proximal IRLS-LLA normalizes the same analytic weights and carries them into each Quantile IRLS/MM majorization before solving the convex group surrogate;
- ordinary FISTA, including explicitly selected L2/no-penalty FISTA, uses the loss-layer normalized weighted objective where supported;
- Adaptive L1 uses the same analytic training weights when it must learn adaptive penalty weights from its initialization fit; user-supplied fixed adaptive weights are used directly and do not trigger a redundant initializer;
- generic `LossBase` shared value/gradient primitives can evaluate the normalized weighted objective;
- FISTA-BB and direct public ADMM do not support Quantile because their generic algorithms rely on smooth-gradient structure that check loss does not provide;
- direct low-level Quantile L-BFGS retains omitted/uniform-weight compatibility, while genuine non-uniform weights raise an error; estimator/CV explicit L-BFGS remains unsupported.

For weighted support across other losses and solver families, see the [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) and [Solver Algorithms](../guides/solver-algorithms.md).

## Algorithm details

### Proximal IRLS-CD (SCAD/MCP)

See [Solver Algorithms](../guides/solver-algorithms.md#1-proximal-irls-cd) for the full update equations. The method combines an IRLS quadratic majorization of the check loss with local linear approximation of SCAD/MCP.

### Group Proximal IRLS-LLA (Group SCAD/MCP)

For the automatic Group SCAD/MCP route, the outer LLA step converts the non-convex group penalty into a convex weighted Group-Lasso surrogate. If $D_g^{(k)}$ denotes the current derivative of the group penalty with respect to $\|\beta_g\|_2$, the Quantile-specific inner iteration first constructs

$$
w_i^{(t)}
=
\widetilde s_i
\frac{\tau+(1-2\tau)\mathbf 1\{r_i^{(t)}<0\}}
{\max(|r_i^{(t)}|,\varepsilon)},
\qquad
\widetilde s_i=\frac{n s_i}{\sum_j s_j},
$$

where $s_i=1$ in the unweighted case. It then solves the convex weighted least-squares surrogate

$$
\min_\beta
\frac{1}{2n}
\sum_i w_i^{(t)}
\left(y_i-x_i^\top\beta\right)^2
+
\sum_g D_g^{(k)}\|\beta_g\|_2.
$$

The convex subproblem is solved with a backend-native splitting method whose quadratic update uses the weighted least-squares system and whose proximal update is the exact Adaptive Group Lasso block shrinkage. The intercept is part of the quadratic model but remains unpenalized. If all $D_g^{(k)}$ are zero, the LLA target is exactly unpenalized Quantile regression, so it is solved directly with ordinary weighted Quantile IRLS.

This automatic route is separate from explicit FISTA control: `solver="fista"` remains an explicit FISTA request, and direct low-level `fista_lla_path(...)` performs the same dedicated Proximal IRLS-LLA solve for scalar Quantile SCAD/MCP (group penalties and warm-started/path-reporting calls keep the fused FISTA-LLA engine).

### IRLS (L2/none)

IRLS is the `auto` route for L2/no-penalty Quantile objectives and can also be requested explicitly. Let

$$
r_i=y_i-x_i^\top\beta.
$$

The quantile IRLS weight is

$$
w_i^{\mathrm{IRLS}}
=\frac{\tau+(1-2\tau)\mathbf1\{r_i<0\}}
{\max(|r_i|,\varepsilon)}.
$$

With analytic weights $s_i$, the implementation first normalizes them as

$$
\widetilde s_i=\frac{n s_i}{\sum_j s_j},
$$

then uses

$$
w_i=\widetilde s_i w_i^{\mathrm{IRLS}}.
$$

For $W=\operatorname{diag}(w)$, the unpenalized update solves

$$
(X^\top W X+\varepsilon I)\beta_{\mathrm{new}}
=X^\top W y.
$$

The L2 route adds the corresponding ridge diagonal term, excluding the intercept coordinate from the penalty. See the [IRLS solver reference](../guides/solver-algorithms.md#6-irls-iteratively-reweighted-least-squares) for the complete behavior.

### Ordinary FISTA (explicit L2/none and sparse convex routes)

Quantile/check loss is non-smooth, so this route should not be interpreted as satisfying the classical smooth-gradient assumptions of FISTA. statgpu uses the registered Quantile subgradient together with the existing first-order step/Lipschitz policy and the requested penalty proximal operator. This option is available for explicit algorithm control and for the convex sparse Quantile paths; the automatic L2/no-penalty policy remains IRLS.

## Outputs

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | `(p,)` float | Estimated coefficients |
| `intercept_` | float | Estimated intercept |
| `n_iter_` | int | Number of iterations |
| `quantile` | float | Target quantile |

## Notes

- `QuantileRegression.score()` and the typed `PenalizedQuantileRegression.score()` use check/pinball loss and return its negative to follow sklearn's “higher is better” convention. NumPy, CuPy, and Torch response/weight containers are accepted on supported routes; scoring takes only the final reporting snapshot needed to return a Python scalar. The generic `PenalizedGeneralizedLinearModel(loss="quantile")` keeps the shared scalar-response `score()` contract and therefore reports response-scale R²; `PenalizedGLM_CV.score()` delegates to that selected refit. For CV, `best_score_` is negative validation loss and is intentionally a different metric from the post-fit `score()`.
- `sample_weight` support is a **loss × solver × estimator** route capability, not an automatic property of every solver.
- Strict Quantile cross-validation requires complete finite fold evidence for every penalty family. A fold that its solver route explicitly marks as a target-level convergence failure is not scored, and an alpha is eligible only when every fold has a finite score. If any fold fit fails or carries that explicit target-level failure signal, the whole candidate column is treated as missing rather than averaging the remaining folds. A generic solver `ConvergenceWarning` alone does not erase an otherwise finite candidate result. Two-stage screening remains intentionally relaxed; the complete-evidence rule applies to strict refinement/selection. The selected full-data refit follows direct-estimator convergence reporting and may emit `ConvergenceWarning` rather than being treated as a CV candidate failure.
- Quantile non-convex continuation routes reject stopping-control coercion. `max_iter` must be a positive integer and `tol` a finite positive real number; direct scalar SCAD/MCP and automatic Group SCAD/MCP also require boolean `lla=True`, integer `max_lla_iters`, and finite-positive `lla_tol`. The current automatic Quantile continuation has three alpha steps, so `max_lla_iters` must be at least 3 to give every step one LLA update. Intermediate continuation steps use a reduced IRLS budget that never exceeds the public `max_iter`; the target step may use the full budget. Explicit Group SCAD/MCP `solver="fista"` does not enter an LLA continuation route, so `lla`, `max_lla_iters`, and `lla_tol` do not govern that explicit algorithm.
- Public low-level Quantile solver calls—including ordinary `fista_solver`, the direct `lbfgs_solver` compatibility row, `QuantileLoss.irls()`, `proximal_irls_quantile_solver()`, and Quantile `fista_lla_path()`—reject malformed supervised shapes before numerical work: `X` must be two-dimensional, `y` one-dimensional, and their row counts must agree; both arrays must also contain real finite values. The specialized IRLS/continuation boundaries also reject invalid intercept, stopping, path, and weight controls instead of relying on broadcasting or implicit coercion. For direct continuation calls, `alpha_path` must be a non-empty one-dimensional sequence of finite positive values in non-increasing order from the continuation start to the target.
- Explicit ordinary L2/no-penalty Quantile FISTA is supported and is authoritative: it executes FISTA rather than silently substituting IRLS. Explicit Group SCAD/MCP FISTA is likewise not rewritten into the automatic Group Proximal IRLS-LLA route.
- Quantile FISTA-BB, direct ADMM, Newton, Proximal Newton, and L-BFGS-B are unsupported and raise before numerical iteration. Estimator/CV ordinary L-BFGS remains unsupported, while the existing low-level unweighted/uniform `lbfgs_solver` compatibility boundary is preserved. The historical `quantile_cd_solver` name remains import-compatible but raises immediately when called: its old implementation ignored `sample_weight` and could not represent an unpenalized intercept reliably, so scalar SCAD/MCP fitting uses Proximal IRLS-CD instead.
- Unsupported explicit weighted-solver combinations raise an error before numerical iteration rather than silently substituting another solver.
- Supported GPU routes (`cuda`/`torch`) do not silently fall back to CPU.

For a low-level loss object, import `QuantileLoss` from `statgpu.losses`. For an external unpenalized comparison with R `quantreg::rq()`, align the quantile, design, intercept, weights, and optimization accuracy; inference also needs compatible density or resampling assumptions.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Feng, X., He, X. & Hu, J. (2011). Wild bootstrap for quantile regression. *Biometrika*, 98(4), 995-999.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.