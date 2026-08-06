# Solver × Penalty Compatibility Matrix

> Language: English  
> Last updated: 2026-08-03  
> This page: Reference guide  
> Switch: [Chinese](../../cn/guides/solver-penalty-matrix.md)

## Overview

`PenalizedGeneralizedLinearModel` supports **7 loss families × 9 registered penalty names × 9 solvers**. `AdaptiveGroupLassoPenalty` is additionally available as a public penalty object; it intentionally has no string-registry alias because callers must supply explicit group weights.

**Key rule**: supported loss × penalty combinations work with `solver='auto'`. Explicit solver requests are validated before numerical work.

## 1. Auto-Dispatch Table

| Loss | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | exact | fista | fista | irls_cd → fista_lla | irls_cd → fista_lla | fista | fista | group fista_lla | group fista_lla |
| **logistic** | irls | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **poisson** | irls | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **gamma** | newton | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **inverse_gaussian** | newton | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **negative_binomial** | irls | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **tweedie** | irls | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |

**Dispatch notes**:
- `AdaptiveGroupLassoPenalty` follows the `group_lasso` column, using its supplied per-group weights.
- `fista_lla` is not a user-facing `solver=` keyword. It is invoked internally for nonconvex penalties. The exported `fista_lla_path()` function enforces the same surrogate when called directly.
- Scalar squared-error SCAD/MCP may use coordinate-descent continuation. Group SCAD/MCP always use a weighted Group Lasso surrogate with a group-aware FISTA inner solve.
- Every Group Lasso or Adaptive Group Lasso estimator uses the advertised loss gradient and the exact Euclidean group proximal operator. This includes squared error, robust/GLM losses, `sample_weight`, CV folds, and the selected-alpha final refit.
- The former Gaussian block update is not public-routed. Solving a group Gram system and then applying Euclidean block thresholding is exact only for orthonormal group blocks, which the public design matrix does not require.

## 2. Explicit Solver Constraints

| Solver | Accepts | Rejects | Notes |
|--------|---------|---------|-------|
| `exact` | l2 only, squared_error only | everything else | Eigendecomposition closed-form |
| `irls` | l2 only (any loss) | all non-smooth | Iteratively Reweighted Least Squares |
| `newton` | l2 / none (any loss) | l1, elasticnet, scad, mcp, adaptive_l1, all group penalties | Newton-Raphson with line search |
| `lbfgs` | l2 / none (any loss) | l1, elasticnet, scad, mcp, adaptive_l1, all group penalties | L-BFGS with line search |
| `fista` | all proximal penalties (any supported loss) | — | FISTA with Nesterov momentum |
| `fista_bb` | supported sparse penalties | unsupported combinations fail explicitly | FISTA + Barzilai-Borwein step size |
| `admm` | supported proximal penalties | unsupported combinations fail explicitly | ADMM with proximal z-update |
| `irls_cd` | scalar scad, mcp, adaptive_l1 | l1, elasticnet, all group penalties | IRLS outer + coordinate descent inner |
| `proximal_irls_cd` | scalar scad, mcp (quantile only) | group penalties and non-quantile losses | IRLS majorization + LLA |
| `proximal_newton` | l2 / none use Newton; non-smooth direct calls delegate visibly to FISTA | group penalties and unsupported penalties | no silent Euclidean-prox approximation |

Unsupported combinations raise `ValueError` before numerical work.

## 3. Solver Capabilities

| Solver | sample_weight | warm_start | Inference | Best for |
|--------|:------------:|:----------:|:---------:|----------|
| `exact` | ✅ | ❌ | ✅ (OLS) | squared_error + l2 |
| `irls` | ✅ | ❌ | ❌ | GLM + l2 |
| `newton` | loss dependent | ❌ | ❌ | smooth objectives |
| `lbfgs` | loss dependent | ❌ | ❌ | large smooth objectives |
| `fista` | ✅ | ✅ | ❌ | convex group/sparse objectives and LLA inner solves |
| `fista_bb` | ✅ | ✅ | ❌ | supported sparse objectives with adaptive steps |
| `admm` | ✅ | ✅ | ❌ | supported proximal objectives |
| `irls_cd` | ✅ | ✅ | ❌ | squared_error + scalar SCAD/MCP |

Group warm starts carry the coefficient and intercept components together for one fit call and are cleared after success or failure.

## 4. CV Support (`PenalizedGLM_CV`)

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | eig-batch | sparse FISTA | LLA + FISTA/CD | general fit | Group FISTA | Group FISTA-LLA |
| **logistic** | general fit | sparse FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **poisson** | general fit | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **gamma** | general fit | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **inverse_gaussian** | general fit | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **negative_binomial** | general fit | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **tweedie** | general fit | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |

Group validation occurs before alpha-grid generation, fold construction, or candidate fitting. Groups are interpreted against the final design width, including formula-expanded columns. Missing unweighted features are completed as singleton groups once; out-of-range indices and incomplete adaptive weighted groups fail transactionally.

CV uses fit-local penalty state. It does not mutate a caller's penalty object or `penalty_kwargs` dictionary. For penalty objects, every candidate is rebuilt at the candidate alpha; the selected final estimator exposes an unmarked penalty snapshot whose alpha and groups match the resolved objective. The top-level CV estimator retains its original constructor parameter.

## 5. Penalty Reference

| Penalty | Formula | Proximal | Parameters |
|---------|---------|----------|------------|
| `l2` | ½α‖β‖² | β/(1+α·step) | `alpha` |
| `l1` | α‖β‖₁ | soft threshold | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft threshold / L2 scale | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD thresholding | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP thresholding | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | weighted soft threshold | `alpha`, weights |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | block soft threshold | `alpha`, `groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | weighted block soft threshold | `alpha`, `groups`, `weights`; object-only |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | SCAD block threshold | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | MCP block threshold | `alpha`, `groups`, `gamma` |

For Group SCAD/MCP, let `D_g` denote the derivative with respect to `‖β_g‖₂`. The exact convex surrogate is `Σ_g D_g‖β_g‖₂`, represented internally by `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`. Neither target alpha nor group size is multiplied twice. Group LLA uses FISTA rather than the generic proximal-Newton branch because the latter can reject all Armijo steps without exposing a failure status.

Group inputs are strict: alpha and other hyperparameters must be finite numeric scalars rather than booleans or coercible strings; indices/IDs must be non-negative integer-valued numerics representable as signed `int64`; explicit groups must be nonempty and duplicate-free; flat IDs must be contiguous from zero; and numerical penalty methods require exactly the grouped feature dimension. The fused group-LLA surrogate alone has a private one-coordinate allowance for its unpenalized intercept.

## 6. Inference Support

| Penalty | Inference method | Status |
|---------|-----------------|--------|
| `l2` | Standard OLS/GLS inference | ✅ Available |
| `l1` | Debiased Lasso | ✅ Supported paths |
| `elasticnet` | method dependent | See estimator contract |
| `scad` / `mcp` | oracle/bootstrap where implemented | See estimator contract |
| `adaptive_l1` | method dependent | See estimator contract |
| Group Lasso / Adaptive Group Lasso / Group SCAD / Group MCP | Group-preserving covariance/bootstrap | Not implemented; every inference request fails explicitly before fitting |

## 7. Choosing a Solver

```
                    ┌─ squared_error + l2? ─── Yes ──→ exact
                    │
                    ├─ smooth penalty only? ── Yes ──→ irls / newton / lbfgs
                    │
solver='auto' ──────├─ scalar nonconvex? ───── Yes ──→ scalar LLA path
                    │
                    ├─ convex group penalty? ─ Yes ──→ exact Group FISTA
                    │
                    └─ group SCAD/MCP? ─────── Yes ──→ Group FISTA-LLA
```
