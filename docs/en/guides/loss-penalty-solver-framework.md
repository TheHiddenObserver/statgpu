# Loss × Penalty × Solver Framework

> Language: English  
> Last updated: 2026-07-01

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
       ├── fista / fista_bb / fista_lla → FISTA family
       ├── newton / irls                  → smooth paths
       ├── proximal_irls_cd              → quantile + SCAD/MCP
       ├── proximal_newton               → Huber/Bisquare/Cox + SCAD/MCP
       └── lbfgs / admm                 → quasi-Newton / augmented Lagrangian
```

## 1. Loss Functions

### LossBase

Abstract base class at `statgpu/losses/_base.py`. Subclasses implement `per_sample_value()` and `per_sample_gradient()`. The base class derives `value()`, `gradient()`, `fused_value_and_gradient()` automatically.

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

**Quantile (Pinball)**:
$$\ell(u) = u \cdot (\tau - \mathbf{1}_{u<0}), \quad u = y - \eta$$

**Huber** (delta-k = 1.345):
$$\ell(u) = \begin{cases} \frac{1}{2}u^2 & |u| \leq k \\ k|u| - \frac{1}{2}k^2 & |u| > k \end{cases}$$

**Bisquare (Tukey biweight)** (c = 4.685):
$$\ell(u) = \begin{cases} \frac{c^2}{6}[1 - (1-(u/c)^2)^3] & |u| \leq c \\ c^2/6 & |u| > c \end{cases}$$

**Cox Partial Likelihood** (Breslow / Efron ties):
$$L(\beta) = \prod_{i:\delta_i=1} \frac{\exp(X_i\beta)}{\sum_{j:T_j \geq T_i} \exp(X_j\beta)}$$

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
| 3 | `fista` (LLA) | all nonconvex penalties (SCAD/MCP/adaptive) |
| 4 | `fista` | quantile (has no Hessian) |
| 5 | `fista` / `fista_bb` | squared_error/GLM + sparse penalties |
| 6 | `lbfgs` / `newton` | CV + L2 + loss-specific |
| 7 | `newton` / `irls` | smooth penalties + smooth losses |

### All Solvers

| Solver | Loss Constraints | Penalty Constraints | sample_weight | warm_start |
|--------|:-----------------|:---------------------|:------------:|:----------:|
| `exact` | squared_error only | l2 only | ✅ | ❌ |
| `irls` | any with IRLS | l2 / none | ✅ | ❌ |
| `newton` | any with Hessian | l2 / none | ❌ | ❌ |
| `lbfgs` | any | l2 / none | ❌ | ❌ |
| `lbfgs_b` | any (box-constrained) | l2 / none | ❌ | ❌ |
| `fista` | any | all | ✅ | ✅ |
| `fista_bb` | any | all (except nonconvex groups) | ✅ | ✅ |
| `fista_lla` | any (SCAD/MCP path) | SCAD/MCP/adaptive | ✅ | ✅ |
| `proximal_irls_cd` | quantile only | SCAD/MCP | ✅ | ✅ |
| `proximal_newton` | any with Hessian | SCAD/MCP/adaptive (via LLA) | ✅ | ✅ |
| `admm` | any | all | ❌ | ✅ |

### Specialized Solvers

**Proximal IRLS-CD** (quantile + SCAD/MCP):
1. Compute IRLS weights: `w_i = τ_i / max(|r_i|, ε)`
2. Quadratic majorization: `Q(β) = ½ Σ w_i(y_i - X_iβ)²`
3. Parallel diagonal majorization step + LLA threshold
4. GPU: convergence check stays on device, only syncs bool

**Proximal Newton** (Huber/Bisquare/Cox + SCAD/MCP):
1. Compute Hessian `H = ∇²ℓ(β)` and gradient `g = ∇ℓ(β)`
2. Newton direction: `d = -H⁻¹·g`
3. Armijo line search with proximal step
4. Typically converges in 5-10 iterations

**FISTA-LLA** (generic nonconvex path):
1. Continuation path: λ_max → target α (3-5 steps)
2. LLA outer loop (2-5 iterations per step)
3. FISTA or Proximal Newton inner solve

## 4. Backend Coverage

| Solver / Path | NumPy | CuPy | Torch |
|:---------------|:---:|:---:|:---:|
| Proximal IRLS-CD | ✅ | ✅ | ✅ |
| Proximal Newton | ✅ | ✅ | ✅ |
| FISTA (weighted) | ✅ | ✅ | ✅ |
| FISTA-BB (weighted) | ✅ | ✅ | ✅ |
| FISTA-LLA (weighted) | ✅ | ✅ | ✅ |
| Quantile IRLS (smooth) | ✅ | ✅ | ✅ |
| CoxPH Efron GPU | ✅ | ✅ (kernel) | ✅ (DLPack→CuPy) |
| DBSCAN | ✅ | GPU dist + host-sync CC | ✅ on-device |
| UMAP | ✅ | backend-aware + known host transfer | backend-aware + known host transfer |

## 5. Penalized Model Classes

| Class | Loss | Penalties | Solvers |
|-------|------|-----------|---------|
| `PenalizedGeneralizedLinearModel` | any | all 10 | all 10 |
| `PenalizedLinearRegression` | squared_error | l1/l2/elasticnet/scad/mcp/adaptive_l1 | exact/fista |
| `PenalizedLogisticRegression` | logistic | l1/l2/elasticnet/scad/mcp/adaptive_l1 | irls/fista |
| `PenalizedPoissonRegression` | poisson | l1/l2/elasticnet/scad/mcp/adaptive_l1 | irls/fista |
| `PenalizedQuantileRegression` | quantile | scad/mcp/l2 | proximal_irls_cd/fista/irls |
| `PenalizedRobustRegression` | huber/bisquare | scad/mcp/l2 | proximal_newton/irls |
| `PenalizedCoxRegression` | cox_ph | scad/mcp/l2 | proximal_newton |

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
from statgpu.linear_model.penalized import PenalizedCoxRegression
model = PenalizedCoxRegression(penalty='scad', alpha=0.1)
model.fit(X, (time, event))

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
