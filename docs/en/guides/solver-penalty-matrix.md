# Solver × Penalty Compatibility Matrix

> Language: English  
> Last updated: 2026-06-12  
> This page: Reference guide  
> Switch: [Chinese](../../guides/solver-penalty-matrix.md)

## Overview

`PenalizedGeneralizedLinearModel` supports a combinatorial space of **7 loss families × 9 penalties × 9 solvers**. This page documents which combinations are supported, how `solver='auto'` dispatches, and what happens when you explicitly request a solver.

**Key rule**: Every loss × penalty combination works with `solver='auto'`. Restrictions only apply when you explicitly specify a solver.

## 1. Auto-Dispatch Table

When `solver='auto'` (the default), the model selects the best solver for each loss × penalty pair:

| Loss | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | exact | fista | fista | irls_cd → fista_lla | irls_cd → fista_lla | fista | fista (CD) | fista_lla | fista_lla |
| **logistic** | irls | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **poisson** | irls | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **gamma** | newton | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **inverse_gaussian** | newton | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **negative_binomial** | irls | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **tweedie** | irls | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |

**Dispatch notes**:
- `fista_lla` is not a user-facing solver keyword. It is invoked internally for nonconvex penalties (SCAD, MCP, group_scad, group_mcp). The outer `solver` keyword controls only the inner loop (fista, fista_bb, or irls_cd).
- `irls_cd` is preferred for squared_error + SCAD/MCP (Gauss-Seidel CD is faster for OLS). GLM + SCAD/MCP uses `fista_lla` with FISTA inner loop.
- GPU paths may substitute `fista_bb` for `fista` when the Barzilai-Borwein step is beneficial.

## 2. Explicit Solver Constraints

When you set `solver=` explicitly, these constraints apply:

| Solver | Accepts | Rejects | Notes |
|--------|---------|---------|-------|
| `exact` | l2 only, squared_error only | everything else | Eigendecomposition closed-form |
| `irls` | l2 only (any loss) | all non-smooth | Iteratively Reweighted Least Squares |
| `newton` | l2 / none (any loss) | l1, elasticnet, scad, mcp, adaptive_l1, group_* | Newton-Raphson with line search |
| `lbfgs` | l2 / none (any loss) | l1, elasticnet, scad, mcp, adaptive_l1, group_* | L-BFGS with line search |
| `fista` | all penalties (any loss) | — | FISTA with Nesterov momentum |
| `fista_bb` | all penalties (any loss) | — | FISTA + Barzilai-Borwein step size |
| `admm` | all penalties (any loss) | — | ADMM with proximal z-update |
| `irls_cd` | scad, mcp, adaptive_l1 | l1, elasticnet, group_* | IRLS outer + coordinate descent inner |

**Attempting an unsupported combination raises `ValueError`** with a message indicating which solver–penalty pairs are valid.

## 3. Solver Capabilities

| Solver | sample_weight | warm_start | Inference | Best for |
|--------|:------------:|:----------:|:---------:|----------|
| `exact` | ✅ | ❌ | ✅ (OLS) | squared_error + l2 (small p) |
| `irls` | ✅ | ❌ | ❌ | GLM + l2 (canonical link) |
| `newton` | ❌ | ❌ | ❌ | GLM + l2 (non-canonical link) |
| `lbfgs` | ❌ | ❌ | ❌ | GLM + l2 (large p) |
| `fista` | ✅ | ✅ | ❌ | Smooth + non-smooth penalties |
| `fista_bb` | ✅ | ✅ | ❌ | GLM + non-smooth (adaptive step) |
| `admm` | ✅ | ✅ | ❌ | Any penalty (augmented Lagrangian) |
| `irls_cd` | ✅ | ✅ | ❌ | squared_error + SCAD/MCP (fast CD) |

## 4. CV Support (`PenalizedGLM_CV`)

The CV estimator uses specialized fast paths where available and falls back to per-fold `fit()` for the rest:

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_* |
|------|:--:|:---------------:|:----------:|:-----------:|:-------:|
| **squared_error** | eig-batch (O(p³)) | sparse FISTA path | LLA + FISTA/CD | general fit | general fit |
| **logistic** | general fit | logistic sparse path | LLA + FISTA | general fit | general fit |
| **poisson** | general fit | fold-batched GPU | LLA + FISTA | general fit | general fit |
| **gamma** | general fit | fold-batched GPU | LLA + FISTA | general fit | general fit |
| **inverse_gaussian** | general fit | fold-batched GPU | LLA + FISTA | general fit | general fit |
| **negative_binomial** | general fit | fold-batched GPU | LLA + FISTA | general fit | general fit |
| **tweedie** | general fit | fold-batched GPU | LLA + FISTA | general fit | general fit |

**Fast path descriptions**:
- **eig-batch**: Precomputes X'X eigendecomposition once, solves all alphas/folds in one batch. O(p³) setup + O(p·n_alphas·n_folds) solve.
- **sparse FISTA path**: Specialized FISTA loop for squared_error + l1/elasticnet with sparse matrix operations.
- **logistic sparse path**: Specialized FISTA loop for logistic + l1/elasticnet.
- **fold-batched GPU**: All folds × all alphas evaluated in one GPU kernel launch. Used for GLM + l1/elasticnet on GPU.
- **LLA + FISTA**: Local Linear Approximation (LLA) continuation path for nonconvex penalties. Traces solution from λ_max down to target α.
- **general fit**: Falls back to per-fold `PenalizedGeneralizedLinearModel.fit()`. Works for all combinations but is slower.

## 5. Penalty Reference

| Penalty | Formula | Proximal | Parameters |
|---------|---------|----------|------------|
| `l2` | ½α‖β‖² | β/(1+α·step) | `alpha` |
| `l1` | α‖β‖₁ | soft_threshold(β, α·step) | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft_threshold / (1+α(1-λ)step) | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD thresholding | `alpha`, `a` (default 3.7) |
| `mcp` | MCP(β; α, γ) | MCP thresholding | `alpha`, `gamma` (default 3.0) |
| `adaptive_l1` | α·w·‖β‖₁ | weighted soft_threshold | `alpha`, `_weights` |
| `group_lasso` | αΣ_g‖β_g‖₂ | block soft_threshold | `alpha`, `groups` |
| `group_scad` | SCAD group | SCAD block thresholding | `alpha`, `groups`, `a` |
| `group_mcp` | MCP group | MCP block thresholding | `alpha`, `groups`, `gamma` |

**Nonconvex penalty notes**:
- SCAD and MCP are solved via **LLA (Local Linear Approximation)**: at each continuation step, the nonconvex penalty is linearized around the current estimate, producing a weighted L1 problem that FISTA/CD can solve.
- The continuation path traces from `λ_max` (where all coefficients are zero) down to the target `α`, using 20-100 steps. This avoids bad local minima.
- `a=2.0` for SCAD and `gamma=1.0` for MCP are numerically singular. The code clamps these to safe values (`a ≥ 2+1e-6`, `gamma ≥ 1+1e-6`).

## 6. Inference Support

| Penalty | Inference method | Status |
|---------|-----------------|--------|
| `l2` | Standard OLS/GLS inference | ✅ Available |
| `l1` | Debiased Lasso (nodewise regression) | ✅ Available via `compute_inference=True` |
| `elasticnet` | Debiased Lasso (adapted) | 待实现 |
| `scad` / `mcp` | Debiased nonconvex | 待实现 |
| `adaptive_l1` | Debiased adaptive Lasso | 待实现 |
| `group_*` | Group debiased | 待实现 |

## 7. Choosing a Solver

```
                    ┌─ squared_error + l2? ─── Yes ──→ exact (closed-form)
                    │
                    ├─ smooth penalty only? ── Yes ──→ irls / newton / lbfgs
                    │
solver='auto' ──────├─ nonconvex (SCAD/MCP)? ─ Yes ──→ fista_lla (auto)
                    │
                    ├─ l1 / elasticnet? ────── Yes ──→ fista / fista_bb
                    │
                    └─ group penalty? ───────── Yes ──→ fista with block CD
```

**Manual solver selection guidelines**:
- Use `solver='fista_bb'` for GLM + non-smooth when you want adaptive step sizes (often faster than fixed-step FISTA).
- Use `solver='admm'` when you need a specific augmented Lagrangian formulation or when the proximal operator is cheap.
- Use `solver='irls_cd'` for squared_error + SCAD/MCP when you want Gauss-Seidel CD (faster convergence than Jacobi-style block CD for small p).
