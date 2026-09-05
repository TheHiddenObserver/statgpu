# Loss × Penalty × Solver Framework

> Language: English
>
> Last updated: 2026-09-04

## Overview

statgpu supports a combinatorial space of **loss functions × penalty types × solvers × backends**. This page documents the complete framework architecture, dispatch logic, and coverage matrix.

## Architecture

```
fit(X, y, sample_weight)
  ├── _resolve_loss()   → LossBase subclass
  ├── _resolve_penalty() → Penalty subclass
  ├── _select_solver()   → solver name (auto or explicit)
  ├── _pre_fit()         → backend conversion, intercept augmentation
  └── _fit_loss_backend() → route to specific solver path
       ├── fista / fista_bb              → generic proximal paths
       ├── newton / irls / lbfgs        → smooth paths
       ├── quantile refinement          → quantile IRLS or proximal IRLS + LLA
       ├── robust refinement            → Newton, FISTA, or FISTA-LLA
       ├── Cox refinement               → Cox-specific Newton/FISTA/FISTA-LLA
       └── admm                         → supported split paths
```

## 1. Loss Functions

### LossBase

Abstract base class at `statgpu/losses/_base.py`. Subclasses implement `per_sample_value()` and `per_sample_gradient()`. The base class derives `value()`, `gradient()`, `fused_value_and_gradient()` automatically.

```python
class LossBase:
    name: str               # "quantile", "huber", etc.
    y_type: str             # "continuous" or "survival"
    smooth_gradient: bool   # smoothness capability
    has_hessian: bool       # Hessian contract, not a public solver guarantee
    _supports_irls: bool    # low-level IRLS contract
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
| Huber | `HuberLoss` | ✅ | ✅ | ❌ | `MASS::rlm()` |
| Bisquare | `BisquareLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="bisquare")` |
| Fair | `FairLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="fair")` |
| Cox PH | `CoxPartialLikelihoodLoss` | ✅ | ✅ | ❌ | `survival::coxph()` |

These capability flags describe loss objects. They do not by themselves make a
name valid for an estimator's `solver=` argument. In particular, Bisquare and
Fair expose an IRLS contract at the loss layer but are not documented as
high-level IRLS estimator paths. Use the corresponding model page below.

### Per-Sample Formulas

**Quantile (Pinball)**:
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

::: warning Estimator boundary
The solver library contains more callable functions than the public estimator
API contains `solver=` values. Choose a solver on the model page first; use
this framework page to understand the internal route.
:::

### Public controls and internal paths

| Layer | Names | Meaning |
|---|---|---|
| Shared penalized selector | `auto`, `fista`, `fista_bb`, `irls`, `newton`, `lbfgs`, `admm`, `exact` | Candidate values; each estimator accepts only a subset |
| CPU squared-error compatibility path | `coordinate_descent` | L1/Elastic Net only; not the quantile CD routine |
| Internal/direct low-level functions | `fista_lla_path`, `proximal_irls_quantile_solver`, `quantile_cd_solver`, `proximal_newton_solver`, `lbfgs_b_solver` | Implementation APIs, not universal estimator keywords |
| Fixed estimators | no selector | Logistic regression, ordinary quantile regression, ordinary Cox PH, and ordered models choose their own algorithms |

Model-specific routing runs before some generic solver branches. For SCAD, MCP,
and Adaptive Lasso, an explicit generic label that passes common validation
does not necessarily select a distinct algorithm. Do not use those labels to
benchmark algorithms unless the model page says that they are distinct.

### Current automatic routes

| Model or objective | Automatic/fixed route |
|---|---|
| ordinary GLM | IRLS |
| Ridge | exact by wrapper default; shared `auto` uses exact on NumPy and Newton on CuPy/Torch |
| Lasso / Elastic Net | FISTA |
| Adaptive Lasso | initial estimate followed by weighted-L1 FISTA |
| SCAD / MCP | model-specific FISTA-LLA continuation |
| penalized quantile, L2/none | quantile IRLS refinement |
| penalized quantile, SCAD/MCP | proximal IRLS + LLA |
| penalized robust, L2/none | Newton |
| penalized robust, sparse penalties | FISTA; SCAD/MCP use FISTA-LLA |
| ordinary Cox PH / ordered models | fixed Newton variants with no public selector |

The `exact` squared-error/L2 solver is unrelated to
`CoxPH(ties="exact")`. Exact accepted and rejected combinations are listed in
the [solver × penalty matrix](solver-penalty-matrix.md).

### Specialized internal paths

- Quantile SCAD/MCP uses a quadratic IRLS majorization with an LLA thresholding
  step. The estimator exposes the high-level penalty-aware route, not the
  low-level function name.
- SCAD/MCP FISTA-LLA follows a continuation path, updates local-linear weights,
  and solves weighted-L1 inner problems.
- Proximal Newton remains a low-level solver-library facade. The current robust
  SCAD/MCP estimator route is FISTA-LLA, not Proximal Newton.

## 4. Backend scope

The shared FISTA, FISTA-BB, weighted-L1/FISTA-LLA, quantile IRLS, smooth
Newton/L-BFGS, and Cox-specific paths have NumPy, CuPy, and Torch
implementations where the corresponding estimator exposes them. Backend
coverage does not widen the model's public selector. For example,
`LogisticRegression` remains fixed to IRLS on every backend.

## 5. Model-specific references

| Model family | Authoritative solver section |
|---|---|
| GLM wrappers | [Generalized linear model](../models/generalized-linear-model.md#solver-support), [Logistic](../models/logistic-regression.md#solver-support), [Poisson](../models/poisson-regression.md#solver-support) |
| Convex regularization | [Ridge](../models/ridge.md#solver-support), [Lasso](../models/lasso.md#solver-support), [Elastic Net](../models/elastic-net.md#solver-support) |
| Weighted/non-convex regularization | [Adaptive Lasso](../models/adaptive-lasso.md#solver-support), [SCAD](../models/scad.md#solver-support), [MCP](../models/mcp.md#solver-support) |
| Specialized regression | [Quantile](../models/quantile.md#solver-support), [Robust](../models/robust.md#solver-support), [Cox PH](../models/coxph.md#solver-support) |
| Other model-specific algorithms | [Ordered models](../models/ordered.md#solver-support), [Kernel methods](../models/kernel-methods.md#solver-support), [PCA](../unsupervised/pca.md#solver-support), [NMF](../unsupervised/nmf.md#solver-support) |

The penalized Cox wrapper never fits an intercept and is estimation-only.
Passing `compute_inference=True` raises `NotImplementedError`; use ordinary
`CoxPH` when inference, baseline hazard, or survival curves are required.

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
