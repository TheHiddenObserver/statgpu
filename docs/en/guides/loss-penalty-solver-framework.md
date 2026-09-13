# Loss × Penalty × Solver Framework

> Language: English
>
> Last updated: 2026-09-13

## Overview

statgpu separates its public API from its numerical-computation interfaces: **model classes face users and orchestrate a fit; Loss + Penalty define the optimization problem; Solvers consume that problem and perform the numerical optimization; Backend is a cross-cutting execution dimension across those steps.**

“Loss functions × penalty types × solvers × backends” form the composable computation structure used inside model fitting, independently of the model-class inheritance hierarchy. This page documents the current runtime call graph, dispatch logic, and coverage matrix.

## Architecture

### User-facing runtime call graph

```text
User
  │
  │  model = Estimator(...)
  │  model.fit(X, y, sample_weight=...)
  ▼
Model class / public API
  │
  ├── formula / X,y parsing and validation
  ├── backend / device selection
  ├── _resolve_loss()      → LossBase subclass instance
  ├── _resolve_penalty()   → Penalty subclass instance
  ├── _select_solver()     → solver name (auto or explicit)
  ├── sample_weight / intercept / initialization handling
  │
  ▼
Optimization problem
  │
  │      F(β) = L(β) + P(β)
  │       ▲           ▲
  │       │           │
  │      Loss      Penalty
  │
  ▼
Solver
  │
  ├── exact / IRLS / Newton / L-BFGS
  ├── FISTA / FISTA-BB / FISTA-LLA
  ├── proximal IRLS / proximal Newton
  └── ADMM / specialized paths
  │
  │  returns coef / intercept / n_iter / convergence state
  ▼
Fitted-model state
  │
  ├── coef_ / intercept_
  ├── inference / fitted-state metadata
  ├── backend / solver provenance
  └── predict() / summary()
```

On the current penalized-model path, `_PenalizedFitMixin.fit()` performs this orchestration: it constructs `self._loss` and `self._penalty`, chooses the backend and solver, then enters `_fit_loss_backend()`, `_dispatch_irls()`, or a specialized SCAD/MCP path. `LossBase` describes the optimization objective at this layer and is constructed by the model before being consumed by the solver.

### Responsibilities

| Component | Primary responsibility | Normally user-facing? |
|---|---|:---:|
| Model class | Public API, formula/data validation, backend/solver selection, state management, inference, prediction | ✅ |
| Loss | Defines data-fit term `L(β)` and value/gradient/Hessian-style primitives | Usually no |
| Penalty | Defines `P(β)` and regularization primitives such as gradient/proximal/LLA | Usually no |
| Solver | Reads Loss + Penalty capabilities and runs the numerical algorithm | No |
| Backend | NumPy/CuPy/Torch arrays, device, and numerical primitives; cuts across all layers above | Selected through the model class |

Loss and Penalty compose the objective in parallel:

$$
F(\beta)=L(\beta)+P(\beta).
$$

A solver then consumes primitives such as

$$
L(\beta),\quad \nabla L(\beta),\quad \nabla^2L(\beta),\quad
P(\beta),\quad \nabla P(\beta),\quad
\operatorname{prox}_{\gamma P}(v)
$$

to implement Newton, L-BFGS, FISTA, ADMM, and related algorithms.

### Backend is a cross-cutting execution dimension

NumPy, CuPy, and Torch span model preparation, objective evaluation, penalty operations, and solver iterations. The model first resolves the actual backend/device; `X`, `y`, `sample_weight`, Loss derivatives, Penalty proximal operations, and Solver iterations then remain on that execution backend whenever the interface contract permits. Only explicitly allowed metadata or final small results cross to host.

```text
                  NumPy / CuPy / Torch
                ┌───────────────────────┐
Model      ──────┤ backend/device choice │
Loss       ──────┤ value/grad/Hessian    │
Penalty    ──────┤ value/grad/prox       │
Solver     ──────┤ numerical iterations  │
                └───────────────────────┘
```

This page focuses on model paths that construct `Loss + Penalty` and pass that objective to the generic Solver layer. Panel models organize computation around panel-data transformations, OLS/GLS/period-wise regressions, and Panel-specific inference; see [Panel Architecture](../panel/architecture.md).

## 1. Loss Functions

### LossBase

Abstract base class at `statgpu/losses/_base.py`. Subclasses implement `per_sample_value()` and `per_sample_gradient()`. The base class derives `value()`, `gradient()`, and `fused_value_and_gradient()` automatically.

`LossBase` is an **optimization-problem definition interface**. Models normally construct a Loss object through `_resolve_loss()` or a registry/factory and pass it to a solver together with a Penalty object.

```python
class LossBase:
    name: str               # "quantile", "huber", etc.
    y_type: str             # "continuous" or "survival"
    smooth_gradient: bool   # whether the per-sample gradient is smooth
    has_hessian: bool       # whether Hessian primitives are provided
    _supports_irls: bool    # whether maintained IRLS dispatch is declared
```

These fields describe **numerical primitives or dispatch capability**. They do not by themselves determine a complete solver × penalty support route.

### All Implemented Losses

| Loss | Class | `has_hessian` | `smooth_gradient` | `_supports_irls` | R Equivalent |
|------|-------|:---:|:---:|:---:|--------------|
| Squared Error | `GLMLoss` (squared_error) | ✅ | ✅ | ✅ | `lm()` |
| Logistic | `GLMLoss` (logistic) | ✅ | ✅ | ✅ | `glm(…, binomial)` |
| Poisson | `GLMLoss` (poisson) | ✅ | ✅ | ✅ | `glm(…, poisson)` |
| Gamma | `GLMLoss` (gamma) | ✅ | ✅ | ✅ | `glm(…, Gamma)` |
| Inverse Gaussian | `GLMLoss` (inverse_gaussian) | ✅ | ✅ | ✅ | `glm(…, inverse.gaussian)` |
| Negative Binomial | `GLMLoss` (negative_binomial) | ✅ | ✅ | ✅ | `glm.nb()` |
| Tweedie | `GLMLoss` (tweedie) | ✅ | ✅ | ✅ | `glm(…, tweedie)` |
| Quantile | `QuantileLoss` | ❌ | ❌ | ✅ | `quantreg::rq()` |
| Huber | `HuberLoss` | ✅ | ✅ | ❌ | `MASS::rlm()` |
| Bisquare | `BisquareLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="bisquare")` |
| Fair | `FairLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="fair")` |
| Cox PH | `CoxPartialLikelihoodLoss` | ✅ | ✅ | ❌ | `survival::coxph()` |

Huber's current `_supports_irls=False` means public dispatch does not enter a Huber IRLS route. Restoring and validating that path is tracked by Issue #156.

### Per-Sample Formulas

**Quantile (check, also called pinball)**:
$$\ell(u) = u \cdot (\tau - \mathbf{1}_{u<0}), \quad u = y - \eta$$

**Huber** (effective threshold $\delta$):
$$\ell(u) = \begin{cases} \frac{1}{2}u^2 & |u| \leq \delta \\ \delta|u| - \frac{1}{2}\delta^2 & |u| > \delta \end{cases}$$

**Bisquare (Tukey biweight)** (c = 4.685):
$$\ell(u) = \begin{cases} \frac{c^2}{6}[1 - (1-(u/c)^2)^3] & |u| \leq c \\ c^2/6 & |u| > c \end{cases}$$

**Cox Partial Likelihood** (Breslow / Efron ties in `CoxPartialLikelihoodLoss`):
$$L(\beta) = \prod_{i:\delta_i=1} \frac{\exp(X_i\beta)}{\sum_{j:T_j \geq T_i} \exp(X_j\beta)}$$

The high-level `CoxPH` estimator additionally provides Exact ties, delayed-entry/counting-process risk sets, strata, and the corresponding survival-model data structures.

## 2. Penalty Functions

### All Implemented Penalties

| Penalty | `is_convex` | `is_smooth` | Proximal Operator | LLA Support | P(β) |
|---------|:---:|:---:|:---:|:---:|------|
| None / Null | ✅ | ✅ | identity | ❌ | 0 |
| L2 (Ridge) | ✅ | ✅ | — | ❌ | α·‖β‖²₂ |
| L1 (Lasso) | ✅ | ❌ | soft-threshold | ❌ | α·‖β‖₁ |
| ElasticNet | ✅ | ❌ | soft-threshold | ❌ | α(r‖β‖₁+(1-r)‖β‖²₂) |
| SCAD | ❌ | ❌ | 3-region | ✅ | piecewise |
| MCP | ❌ | ❌ | 3-region | ✅ | piecewise |
| Adaptive L1 | ✅ | ❌ | weighted soft-threshold | ✅ | α/|β̂|^ν · |β| |
| Group Lasso | ✅ | ❌ | block soft-threshold | ❌ | · |
| Group MCP | ❌ | ❌ | block proximal | ✅ | · |
| Group SCAD | ❌ | ❌ | block proximal | ✅ | · |

### SCAD Formula
$$P(|\beta|) = \begin{cases} \alpha|\beta| & |\beta| \leq \alpha \\ \frac{-(|\beta|^2 - 2a\alpha|\beta| + \alpha^2)}{2(a-1)} & \alpha < |\beta| \leq a\alpha \\ \frac{(a+1)\alpha^2}{2} & |\beta| > a\alpha \end{cases}$$

### LLA (Local Linear Approximation)
Non-convex penalties (SCAD, MCP) are solved via LLA:
1. Compute weights `w_j = P'(|\beta_j|)` at the current iterate
2. Solve the weighted L1 problem: `min L(β) + Σ w_j·|β_j|`
3. Repeat until convergence (typically 2-5 iterations)

## 3. Solvers

### Solver Dispatch Table

The main `solver="auto"` dispatch can be summarized as follows:

| Priority | Solver | Condition |
|----------|--------|-----------|
| 1 | `exact` | squared_error + L2 + NumPy |
| 2 | `newton` | squared_error + L2 + GPU |
| 3 | `fista` + LLA | non-convex penalties such as SCAD/MCP/adaptive routes |
| 4 | quantile-specific FISTA/IRLS paths | quantile loss |
| 5 | `fista` / `fista_bb` | squared_error/GLM/robust + sparse penalties |
| 6 | `lbfgs` / `newton` | CV + L2 + loss-specific routing |
| 7 | `newton` | maintained smooth L2/no-penalty GLM/robust/Cox paths with Hessian support |

The `exact` solver in this table is the closed-form squared-error/L2 solver; it is unrelated to `CoxPH(ties="exact")`.

### All Solvers

`sample_weight` support depends on the solver, the statistical semantics of the selected loss, and its value/gradient/curvature capabilities. The table below summarizes the maintained main paths; #153 tracks the complete support contract.

| Solver | Loss Constraints | Penalty Constraints | `sample_weight` | warm_start |
|--------|:-----------------|:---------------------|:------------|:----------:|
| `exact` | squared_error only | L2 only | ✅ | ❌ |
| `irls` | losses declaring maintained IRLS dispatch | L2 / none | available where the IRLS loss supports it | ❌ |
| `newton` | losses with Hessian support | L2 / none | loss-dependent; ordinary GLM ✅ | ❌ |
| `lbfgs` | smooth losses | L2 / none | capability-gated; ordinary GLM ✅ | ❌ |
| `lbfgs_b` | smooth box-constrained problems | L2 / none | no generic non-uniform-weight contract declared | ❌ |
| `fista` | losses supporting gradient/proximal routes | supported proximal penalties | loss-dependent | ✅ |
| `fista_bb` | losses supporting gradient/proximal routes | supported sparse penalties | loss-dependent | ✅ |
| `fista_lla` | losses supporting the maintained LLA route | SCAD/MCP/adaptive | loss-dependent | ✅ |
| `proximal_irls_cd` | quantile only | SCAD/MCP | ✅ | ✅ |
| `proximal_newton` | smooth losses with Hessian support | L2 / none | loss-dependent | ✅ |
| `admm` | maintained ADMM losses | supported proximal forms | omitted/uniform only; genuine non-uniform weights fail closed | ✅ |

### Specialized Solvers

**Proximal IRLS-CD** (quantile + SCAD/MCP):
1. Compute IRLS weights: `w_i = τ_i / max(|r_i|, ε)`
2. Quadratic majorization: `Q(β) = ½ Σ w_i(y_i - X_iβ)²`
3. Parallel diagonal-majorization step + LLA threshold
4. GPU convergence checks remain on device except for the final boolean synchronization

**Proximal Newton** (maintained smooth route):
1. Construct the full smooth objective gradient and Hessian
2. Solve the stabilized Newton linear system
3. Use Armijo backtracking to accept `β - t d`
4. A non-smooth penalty request delegates explicitly to FISTA instead of using an incorrect Euclidean-prox Newton approximation

**FISTA-LLA** (generic non-convex path):
1. Continuation path: λ_max → target α (3-5 steps)
2. LLA outer loop (2-5 iterations per step)
3. The maintained generic composite route uses a weighted-convex FISTA inner solve. A Proximal-Newton inner route should be enabled only if a loss explicitly provides the correct Hessian-metric proximal subproblem. Cox currently remains on FISTA-LLA.

## 4. Backend Coverage

| Solver / Path | NumPy | CuPy | Torch |
|:---------------|:---:|:---:|:---:|
| Proximal IRLS-CD | ✅ | ✅ | ✅ |
| Proximal Newton (smooth route) | ✅ | ✅ | ✅ |
| FISTA (weighted) | ✅ | ✅ | ✅ |
| FISTA-BB (weighted) | ✅ | ✅ | ✅ |
| FISTA-LLA (weighted) | ✅ | ✅ | ✅ |
| Quantile IRLS (smooth penalty) | ✅ | ✅ | ✅ |
| Cox partial likelihood (Breslow/Efron) | ✅ native | ✅ native | ✅ native |
| CoxPH counting process / strata / Exact | ✅ native | ✅ native | ✅ native |
| DBSCAN | ✅ | GPU dist + host-sync CC | ✅ on-device |
| UMAP | yes | supported with explicit SciPy host graph boundary | supported with explicit SciPy host graph boundary |

## 5. User-Facing Penalized Models

These are the public model classes users normally construct and call with `.fit()`; internally they resolve Loss, Penalty, Solver, and Backend objects/policies.

| Class | Loss | Penalties | Main solver routes |
|-------|------|-----------|---------|
| `PenalizedGeneralizedLinearModel` | any registered loss | registered penalties | auto-dispatched from the full loss × penalty × backend combination, or explicitly selected |
| `PenalizedLinearRegression` | squared_error | l1/l2/elasticnet/scad/mcp/adaptive_l1 | exact / Newton / FISTA / LLA |
| `PenalizedLogisticRegression` | logistic | l1/l2/elasticnet/scad/mcp/adaptive_l1 | Newton / FISTA / LLA |
| `PenalizedPoissonRegression` | poisson | l1/l2/elasticnet/scad/mcp/adaptive_l1 | Newton / FISTA / LLA |
| `PenalizedQuantileRegression` | quantile | scad/mcp/l2 and related supported penalties | quantile IRLS / Proximal IRLS-CD / FISTA |
| `PenalizedRobustRegression` | huber/bisquare/fair | l1/l2/elasticnet/scad/mcp and related penalties | Newton / FISTA / FISTA-LLA; maintained IRLS additionally exists for Bisquare/Fair |
| `PenalizedCoxPHModel` | cox_ph | l1/l2/elasticnet/scad/mcp | FISTA; FISTA-LLA for SCAD/MCP |

`PenalizedCoxPHModel` provides penalized Cox coefficient estimation; use `statgpu.survival.CoxPH` when covariance, significance tests, baseline hazard, or survival curves are required.

## 6. Quick Reference

```python
# Quantile regression with SCAD
from statgpu.linear_model.penalized import PenalizedQuantileRegression
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)

# Robust regression with MCP
from statgpu.linear_model.penalized import PenalizedRobustRegression
model = PenalizedRobustRegression(loss='huber', penalty='mcp', alpha=0.1)
model.fit(X, y)

# Cox PH with SCAD penalty
import numpy as np
from statgpu.linear_model.penalized import PenalizedCoxPHModel

y_surv = np.column_stack([time, event])
model = PenalizedCoxPHModel(
    penalty='scad', alpha=0.1,
    fit_intercept=False, compute_inference=False,
)
model.fit(X, y_surv)

# All penalties + losses via PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
model = PenalizedGeneralizedLinearModel(loss='gamma', penalty='scad', alpha=0.1)
model.fit(X, y)
```

## References

- Fan & Li (2001): Variable selection via nonconcave penalized likelihood (SCAD)
- Zhang (2010): Nearly unbiased variable selection under minimax concave penalty (MCP)
- Wu & Liu (2009): Variable selection in quantile regression
- Hunter & Li (2005): MM algorithms for nonconvex penalized estimation
- Barzilai & Borwein (1988): Two-point step size gradient methods (BB)
- O'Donoghue & Candes (2015): Adaptive restart for accelerated gradient schemes