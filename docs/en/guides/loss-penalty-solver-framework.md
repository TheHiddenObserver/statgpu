# Loss × Penalty × Solver Framework

> Language: English
>
> Last updated: 2026-09-12

## Overview

statgpu separates its public API from its numerical-computation interfaces: **Estimators face users and orchestrate a fit; Loss + Penalty define the optimization problem; Solvers consume that problem and perform the numerical optimization; Backend is a cross-cutting execution dimension across those steps.**

Accordingly, “loss functions × penalty types × solvers × backends” describes the composable computation space inside an estimator, not one inheritance tree. This page documents the actual runtime call graph, dispatch logic, and coverage matrix.

## Architecture

### User-facing runtime call graph

```text
User
  │
  │  model = Estimator(...)
  │  model.fit(X, y, sample_weight=...)
  ▼
Estimator / public API
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
Estimator post-fit
  │
  ├── coef_ / intercept_
  ├── inference / fitted-state metadata
  ├── backend / solver provenance
  └── predict() / summary()
```

On the current penalized-estimator path, `_PenalizedFitMixin.fit()` performs this orchestration: it constructs `self._loss` and `self._penalty`, chooses the backend and solver, then enters `_fit_loss_backend()`, `_dispatch_irls()`, or a specialized SCAD/MCP path. `LossBase` is not a parent class of the estimator and is not a “lower estimator layer”; it is an objective object constructed by the estimator and consumed during numerical fitting.

### Responsibilities

| Component | Primary responsibility | Normally user-facing? |
|---|---|:---:|
| Estimator | Public API, formula/data validation, backend/solver selection, state management, inference, prediction | ✅ |
| Loss | Defines data-fit term `L(β)` and value/gradient/Hessian-style primitives | Usually no |
| Penalty | Defines `P(β)` and regularization primitives such as gradient/proximal/LLA | Usually no |
| Solver | Reads Loss + Penalty capabilities and runs the numerical algorithm | No |
| Backend | NumPy/CuPy/Torch arrays, device, and numerical primitives; cuts across all layers above | Selected through estimator |

Loss and Penalty define the objective **in parallel** rather than by inheritance:

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

NumPy, CuPy, and Torch should not be read as a layer “below the solver.” The estimator first resolves the actual backend/device; `X`, `y`, `sample_weight`, Loss derivatives, Penalty proximal operations, and Solver iterations should then remain on that execution backend whenever the contract permits. Only explicitly allowed metadata or final small results should cross to host.

```text
                  NumPy / CuPy / Torch
                ┌───────────────────────┐
Estimator  ──────┤ backend/device choice │
Loss       ──────┤ value/grad/Hessian    │
Penalty    ──────┤ value/grad/prox       │
Solver     ──────┤ numerical iterations  │
                └───────────────────────┘
```

### Why Panel is not part of the `LossBase` hierarchy

Current Panel estimators follow a separate estimator-level pipeline because their defining structure is panel metadata, transformations, effect recovery, and panel-specific inference rather than a new per-sample loss.

```text
User
  │
  ▼
Panel estimator / BasePanelModel
  │
  ├── formula + entity/time metadata
  ├── within / between / difference / quasi-demeaning transforms
  ▼
transformed (X*, y*) or estimator-specific intermediate state
  │
  ├── current: OLS / GLS / repeated cross-sectional solve / specialized path
  ▼
beta-hat
  │
  ├── fixed/random-effect recovery
  ├── covariance / diagnostics
  └── prediction / summary
```

For example, `PanelOLS` first constructs a within-transformed `(X*, y*)`; `RandomEffects` must estimate variance components before quasi-demeaning; and `FamaMacBeth` performs period-by-period cross-sectional regressions and aggregates the resulting `beta_t`. Those semantics are not represented correctly by making a Panel estimator inherit `LossBase`.

If penalized panel estimators are added later, the natural reuse mechanism is **composition**:

```text
Panel transformation
      │
      ▼
   (X*, y*)
      │
      ├── LossBase object
      ├── Penalty object
      ▼
     Solver
      │
      ▼
  beta-hat
      │
      ▼
Panel inference / effects / diagnostics
```

The Panel estimator would therefore continue to own panel semantics while reusing `LossBase + Penalty + Solver` only for transformed optimization stages where that statistical abstraction is valid.

## 1. Loss Functions

### LossBase

Abstract base class at `statgpu/losses/_base.py`. Subclasses implement `per_sample_value()` and `per_sample_gradient()`. The base class derives `value()`, `gradient()`, `fused_value_and_gradient()` automatically.

`LossBase` is an **optimization-problem definition interface**, not the public estimator base class. Estimators normally construct a Loss object through `_resolve_loss()` or a registry/factory and pass it to a solver together with a Penalty object.

```python
class LossBase:
    name: str               # "quantile", "huber", etc.
    y_type: str             # "continuous" or "survival"
    smooth_gradient: bool   # True ≈ Newton-friendly
    has_hessian: bool       # True ≈ can use proximal Newton
    _supports_irls: bool    # True ≈ has irls() method
```

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
| Huber | `HuberLoss` | ✅ | ✅ | ✅ | `MASS::rlm()` |
| Bisquare | `BisquareLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="bisquare")` |
| Fair | `FairLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="fair")` |
| Cox PH | `CoxPartialLikelihoodLoss` | ✅ | ✅ | ❌ | `survival::coxph()` |

### Per-Sample Formulas

**Quantile (check, also called pinball)**:
$$\ell(u) = u \cdot (\tau - \mathbf{1}_{u<0}), \quad u = y - \eta$$

**Huber** (delta-k = 1.345):
$$\ell(u) = \begin{cases} \frac{1}{2}u^2 & |u| \leq k \\ k|u| - \frac{1}{2}k^2 & |u| > k \end{cases}$$

**Bisquare (Tukey biweight)** (c = 4.685):
$$\ell(u) = \begin{cases} \frac{c^2}{6}[1 - (1-(u/c)^2)^3] & |u| \leq c \\ c^2/6 & |u| > c \end{cases}$$

**Cox Partial Likelihood** (Breslow / Efron ties in `CoxPartialLikelihoodLoss`):
$$L(\beta) = \prod_{i:\delta_i=1} \frac{\exp(X_i\beta)}{\sum_{j:T_j \geq T_i} \exp(X_j\beta)}$$

The high-level `CoxPH` estimator additionally implements Exact ties,
delayed-entry/counting-process risk sets, and strata. Its Exact path is not a
generic `LossBase` solver combination.

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
1. Compute weights `w_j = P'(|\beta_j|)` at current iterate
2. Solve weighted L1 problem: `min L(β) + Σ w_j·|β_j|`
3. Repeat until convergence (typically 2-5 iterations)

## 3. Solvers

### Solver Dispatch Table

The `solver="auto"` dispatch follows priority:

| Priority | Solver | Condition |
|----------|--------|-----------|
| 1 | `exact` | squared_error + l2 + numpy |
| 2 | `newton` | squared_error + l2 + GPU |
| 3 | `fista_lla` | nonconvex SCAD/MCP paths, including penalized Cox |
| 4 | `fista` | quantile (has no Hessian) |
| 5 | `fista` / `fista_bb` | squared_error/GLM + sparse penalties |
| 6 | `lbfgs` / `newton` | CV + L2 + loss-specific |
| 7 | `newton` / `irls` | smooth penalties + smooth losses |

The `exact` solver in this table is the closed-form squared-error/L2 solver; it
is unrelated to `CoxPH(ties="exact")`.

### All Solvers

`sample_weight` is not a solver-only property: support also depends on the statistical semantics and value/gradient/curvature capabilities of the selected loss. The table below summarizes the maintained main paths; #153 tracks the explicit complete capability contract.

| Solver | Loss Constraints | Penalty Constraints | `sample_weight` | warm_start |
|--------|:-----------------|:---------------------|:------------|:----------:|
| `exact` | squared_error only | l2 only | ✅ | ❌ |
| `irls` | losses with IRLS support | l2 / none | available where the IRLS loss supports it | ❌ |
| `newton` | losses with Hessian support | l2 / none | loss-dependent; ordinary GLM ✅ | ❌ |
| `lbfgs` | smooth losses | l2 / none | capability-gated; ordinary GLM ✅ | ❌ |
| `lbfgs_b` | smooth box-constrained problems | l2 / none | no generic non-uniform-weight contract declared | ❌ |
| `fista` | losses supporting gradient/proximal path | all | loss-dependent | ✅ |
| `fista_bb` | losses supporting gradient/proximal path | all (except nonconvex groups) | loss-dependent | ✅ |
| `fista_lla` | losses supporting the maintained LLA path | SCAD/MCP/adaptive | loss-dependent | ✅ |
| `proximal_irls_cd` | quantile only | SCAD/MCP | ✅ | ✅ |
| `proximal_newton` | selected Hessian losses | SCAD/MCP/adaptive (via LLA) | loss-dependent | ✅ |
| `admm` | maintained ADMM losses | all | omitted/uniform only; genuine non-uniform weights fail closed | ✅ |

### Specialized Solvers

**Proximal IRLS-CD** (quantile + SCAD/MCP):
1. Compute IRLS weights: `w_i = τ_i / max(|r_i|, ε)`
2. Quadratic majorization: `Q(β) = ½ Σ w_i(y_i - X_iβ)²`
3. Parallel diagonal majorization step + LLA threshold
4. GPU: convergence check stays on device, only syncs bool

**Proximal Newton** (Huber/Bisquare + SCAD/MCP):
1. Compute Hessian `H = ∇²ℓ(β)` and gradient `g = ∇ℓ(β)`
2. Newton direction: `d = -H⁻¹·g`
3. Armijo line search with proximal step
4. Typically converges in 5-10 iterations

**FISTA-LLA** (generic nonconvex path):
1. Continuation path: λ_max → target α (3-5 steps)
2. LLA outer loop (2-5 iterations per step)
3. Weighted-L1 FISTA inner solve

`PenalizedCoxPHModel` uses this FISTA-LLA continuation for SCAD and MCP. Its
convex L1/L2/ElasticNet paths use the corresponding FISTA/Newton routing.

## 4. Backend Coverage

| Solver / Path | NumPy | CuPy | Torch |
|:---------------|:---:|:---:|:---:|
| Proximal IRLS-CD | ✅ | ✅ | ✅ |
| Proximal Newton | ✅ | ✅ | ✅ |
| FISTA (weighted) | ✅ | ✅ | ✅ |
| FISTA-BB (weighted) | ✅ | ✅ | ✅ |
| FISTA-LLA (weighted) | ✅ | ✅ | ✅ |
| Quantile IRLS (smooth) | ✅ | ✅ | ✅ |
| Cox partial likelihood (Breslow/Efron) | ✅ native | ✅ native | ✅ native |
| CoxPH counting process / strata / Exact | ✅ native | ✅ native | ✅ native |
| DBSCAN | ✅ | GPU dist + host-sync CC | ✅ on-device |
| UMAP | yes | supported with explicit SciPy host graph boundary | supported with explicit SciPy host graph boundary |

## 5. User-Facing Penalized Estimators

These are the public classes users normally construct and call with `.fit()`; internally they resolve Loss, Penalty, Solver, and Backend objects/policies.

| Class | Loss | Penalties | Solvers |
|-------|------|-----------|---------|
| `PenalizedGeneralizedLinearModel` | any | all 10 | all 10 |
| `PenalizedLinearRegression` | squared_error | l1/l2/elasticnet/scad/mcp/adaptive_l1 | exact/fista |
| `PenalizedLogisticRegression` | logistic | l1/l2/elasticnet/scad/mcp/adaptive_l1 | irls/fista |
| `PenalizedPoissonRegression` | poisson | l1/l2/elasticnet/scad/mcp/adaptive_l1 | irls/fista |
| `PenalizedQuantileRegression` | quantile | scad/mcp/l2 | proximal_irls_cd/fista/irls |
| `PenalizedRobustRegression` | huber/bisquare | scad/mcp/l2 | proximal_newton/irls |
| `PenalizedCoxPHModel` | cox_ph | l1/l2/elasticnet/scad/mcp | fista/newton; fista_lla for SCAD/MCP |

The penalized Cox wrapper never fits an intercept and is estimation-only.
Passing `compute_inference=True` raises `NotImplementedError` rather than
falling through to generic GLM inference.

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
