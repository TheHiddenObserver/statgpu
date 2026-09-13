# Quantile Regression

> Language: English  
> Last updated: 2026-09-12  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/quantile.md)

## Overview

`QuantileLoss` implements the **check loss (also called pinball loss)** used in quantile regression. These are two names for the same asymmetric absolute-loss objective, not two different losses. `PenalizedQuantileRegression` adds penalized estimation, including the specialized Proximal IRLS-CD route for SCAD/MCP.

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
| Proximal IRLS-CD | ✅ | Specialized IRLS majorization + LLA for SCAD/MCP; maintained route has explicit analytic-weight handling |
| FISTA | ✅ | Proximal/non-smooth route; weighted behavior follows the maintained FISTA route |
| FISTA-BB | ✅ | Available on supported sparse routes; weighted capability is loss/solver-route specific |
| IRLS | ✅ | L2/none; `QuantileLoss.irls()` has an explicit `sample_weight` path |
| L-BFGS | ✅ (unweighted/uniform weights) | Genuine non-uniform direct weighted L-BFGS is fail-closed for generic `LossBase`; see Issue #153 |
| ADMM | ✅ (unweighted/uniform weights) | Shared `admm_solver` currently rejects genuine non-uniform `sample_weight` |
| Newton | ❌ | Quantile loss has no Hessian |
| Proximal Newton | ❌ | Quantile loss has no Hessian |

## Penalty compatibility

| Penalty | Main `solver="auto"` route | Notes |
|---------|----------------------------|-------|
| l2 / none | IRLS | Quantile-specific IRLS |
| l1 / elasticnet | FISTA | Proximal/subgradient route |
| SCAD / MCP | Proximal IRLS-CD | IRLS majorization + LLA |
| adaptive_l1 | FISTA-LLA | Weighted-L1 proximal surrogate |
| group_* | FISTA-LLA / group route | Corresponding group proximal operator |

## `sample_weight` semantics

On quantile routes that explicitly support non-uniform analytic weights, the data-fit objective is the weighted check/pinball loss. With per-observation loss $\rho_\tau(r_i)$,

$$
L_w(\beta)
=\frac{\sum_i w_i\rho_\tau(y_i-x_i^\top\beta)}{\sum_i w_i}.
$$

But `sample_weight` is **not one universal solver capability**. In particular:

- maintained Quantile IRLS / Proximal IRLS-CD routes have explicit weighted implementations;
- generic `LossBase` shared value/gradient primitives can evaluate the normalized weighted objective;
- direct `lbfgs_solver` remains fail-closed for genuine non-uniform Quantile weights;
- shared `admm_solver` currently accepts omitted or uniform weights only.

GitHub Issue #153 tracks a unified, auditable weighted-capability contract for `LossBase`.

## Examples

### Standalone model with inference

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

model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="scad",
    alpha=0.1,
)
model.fit(X, y)
```

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

This example uses a maintained estimator-level weighted route. It does not imply that every explicitly selected low-level solver supports the same non-uniform weights.

## Algorithm details

### Proximal IRLS-CD (SCAD/MCP)

See [Solver Algorithms](../guides/solver-algorithms.md#1-proximal-irls-cd) for the full update equations. The method combines an IRLS quadratic majorization of the check loss with local linear approximation of SCAD/MCP.

### IRLS (L2/none)

Let

$$
r_i=y_i-x_i^\top\beta.
$$

The quantile IRLS weight is

$$
w_i^{\mathrm{IRLS}}
=\frac{\tau+(1-2\tau)\mathbf1\{r_i<0\}}
{\max(|r_i|,\varepsilon)}.
$$

With analytic weights $s_i$, the maintained implementation first normalizes them as

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

The L2 route adds the corresponding ridge diagonal term, excluding the intercept coordinate from the penalty. See the [IRLS solver reference](../guides/solver-algorithms.md#6-irls-iteratively-reweighted-least-squares) for the complete maintained behavior.

## Outputs

| Attribute | Type | Description |
|-----------|------|-------------|
| `coef_` | `(p,)` float | Estimated coefficients |
| `intercept_` | float | Estimated intercept |
| `n_iter_` | int | Number of iterations |
| `quantile` | float | Target quantile |

## Notes

- `score()` uses check/pinball loss and returns its negative to follow sklearn's “higher is better” convention.
- `sample_weight` support is a **loss × solver × estimator** route capability, not an automatic property of every solver.
- Unsupported explicit weighted-solver combinations should fail before numerical iteration rather than silently substitute another solver.
- Maintained GPU routes (`cuda`/`torch`) must not silently fall back to CPU.

## References

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.