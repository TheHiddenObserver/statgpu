# Quantile Regression

> Language: English  
> Last updated: 2026-09-18  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/quantile.md)

## Overview

`QuantileLoss` implements the **check loss (also called pinball loss)** used in quantile regression. These are two names for the same asymmetric absolute-loss objective, not two different losses. `PenalizedQuantileRegression` adds penalized estimation, including ordinary IRLS for the automatic L2/no-penalty route, an explicitly selectable ordinary FISTA route for convex objectives, the specialized Proximal IRLS-CD route for scalar SCAD/MCP, and a Group Proximal IRLS-LLA automatic route for Group SCAD/MCP.

| Component | Path |
|-----------|------|
| Loss | `statgpu.losses.QuantileLoss` |
| Standalone Model | `statgpu.linear_model.QuantileRegression` |
| Penalized Model | `statgpu.linear_model.penalized.PenalizedQuantileRegression` |
| Specialized Solver | `statgpu.solvers._proximal_irls_quantile.proximal_irls_quantile_solver` |
| R equivalent | `quantreg::rq()` |

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

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `quantile` | `0.5` | Target quantile in `(0,1)`; `0.5` is median regression |

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

The table describes the automatic route. An explicit Group SCAD/MCP `solver="fista"` request remains an explicit proximal-FISTA request; it is not silently rewritten into Group Proximal IRLS-LLA. Likewise, the public low-level `fista_lla_path` retains its FISTA-LLA meaning rather than aliasing this automatic estimator route.

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

## Examples

### Standalone model with inference

Standalone kernel/bootstrap inference is defined only for omitted or uniform `sample_weight`; genuinely non-uniform analytic weights are estimation-only on this class and raise when `compute_inference=True`. The bootstrap implementation is an **i.i.d. residual bootstrap**: fitted residuals are first centered at their empirical target-quantile so the bootstrap error distribution has empirical τ-quantile zero, then the centered residuals are resampled as exchangeable draws and refitted with backend-native batched Quantile IRLS/MM. It is not a wild/multiplier bootstrap and does not claim heteroscedastic-robust coverage; heteroscedastic quantile-regression bootstrap inference requires a different resampling construction. Bootstrap inference also requires `n_bootstrap >= 2`. Kernel inference additionally requires the selected bandwidth rule to keep `q ± h` inside `(0, 1)` and to produce a finite positive residual-density estimate at zero; otherwise inference raises instead of publishing non-finite standard errors.

```python
from statgpu.linear_model import QuantileRegression

model = QuantileRegression(
    quantile=0.5,
    compute_inference=True,
    inference_method="kernel",
    kernel="epa",
    bandwidth="hsheather",
)
model.fit(X, y)
print(model.coef_)
print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

### Penalized quantile regression

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

# solver="auto" resolves to IRLS for this L2-penalized Quantile problem.
model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.1,
    solver="auto",
)
model.fit(X, y)

# Scalar SCAD/MCP use the specialized Proximal IRLS-CD continuation path.
scad_model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="scad",
    alpha=0.1,
)
scad_model.fit(X, y)
```

### Explicit Quantile IRLS or FISTA

```python
irls_model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.01,
    solver="irls",
)
irls_model.fit(X, y)

fista_model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.01,
    solver="fista",
)
fista_model.fit(X, y)
```

For L2/no penalty, `auto` and explicit IRLS use the same IRLS route. Explicit ordinary FISTA is an algorithm-control option and truly executes FISTA; it does not alias or fall back to IRLS. Non-smooth penalties such as ElasticNet already use ordinary FISTA rather than IRLS.

### GPU (Torch CUDA)

```python
import torch

X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="scad",
    alpha=0.1,
)
model.fit(X_t, y_t)
```

### Weighted quantile

```python
sample_weight = np.ones(n)
sample_weight[:50] = 5.0

model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.01,
)
model.fit(X, y, sample_weight=sample_weight)
```

This example uses an estimator path that supports analytic weights. It does not imply that every explicitly selected low-level solver supports the same non-uniform weights.

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

This automatic route is separate from explicit FISTA control: `solver="fista"` and direct low-level `fista_lla_path(...)` continue to mean FISTA-based algorithms.

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

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Feng, X., He, X. & Hu, J. (2011). Wild bootstrap for quantile regression. *Biometrika*, 98(4), 995-999.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.