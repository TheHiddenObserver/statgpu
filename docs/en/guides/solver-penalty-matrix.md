# Solver × Penalty Compatibility Matrix

> Language: English  
> Last updated: 2026-09-04
> This page: Reference guide  
> Switch: [Chinese](../../cn/guides/solver-penalty-matrix.md)

## Overview

The shared penalized engine supports several losses, penalties, and solver paths, but there is no universal list of values that applies to every estimator. Wrapper defaults and model-specific routes can narrow the shared engine.

::: warning Source of truth
Use the corresponding model page to select `solver=`. This page explains shared compatibility; it does not turn low-level functions into estimator keywords. Start from the [model solver lookup](../models/#solver-lookup).
:::

## 1. Auto-dispatch rules

`auto` depends on the loss, penalty, resolved backend, CV mode, and sometimes problem size. The main current rules are:

| Model objective | Automatic dispatch or refined path |
|---|---|
| squared error + L2 | exact on CPU; Newton on CuPy/Torch |
| squared error + L1 / Elastic Net | FISTA |
| ordinary `GeneralizedLinearModel` | IRLS |
| smooth penalized GLM / robust / Cox objective | Newton; selected CV cases use L-BFGS |
| sparse GLM | FISTA or FISTA-BB according to family, backend, CV mode, and size |
| Adaptive Lasso | weighted-L1 FISTA |
| scalar SCAD / MCP | FISTA dispatch refined to model-specific FISTA-LLA; quantile uses proximal IRLS + LLA |
| group penalties | model-specific group proximal/LLA path |
| quantile + L2/none | FISTA dispatch refined internally to quantile IRLS |

Wrapper defaults can be more specific than the shared `auto` policy: `Ridge` defaults to `exact`, Lasso and Elastic Net default to `fista`, `LogisticRegression` exposes no selector and uses fixed IRLS, and ordered models use fixed trust-region Newton. Follow the links from the model catalog rather than inferring a keyword from this table.

## 2. Explicit Solver Constraints

| Solver | Accepts | Rejects | Notes |
|--------|---------|---------|-------|
| `exact` | l2 only, squared_error only | everything else | Eigendecomposition closed-form |
| `irls` | l2/none and a loss declaring the IRLS contract | all non-smooth; squared error and Huber on the shared penalized surface | Model-specific IRLS |
| `newton` | l2 / none with a Hessian-equipped loss | non-smooth penalties and quantile loss | Newton-Raphson with line search |
| `lbfgs` | l2 / none with a smooth loss | non-smooth penalties and quantile loss | L-BFGS with line search |
| `fista` | all proximal penalties (any supported loss) | — | FISTA with Nesterov momentum |
| `fista_bb` | supported sparse penalties | unsupported combinations fail explicitly | FISTA + Barzilai-Borwein step size |
| `admm` | supported loss/penalty pairs | non-uniform sample weights | ADMM with proximal z-update |
| `coordinate_descent` | CPU squared-error L1/Elastic Net compatibility path | GPU and non-squared losses | Estimator compatibility path, not quantile CD |

`irls_cd`, `proximal_irls_quantile_solver`, `fista_lla_path`, `proximal_newton_solver`, and `lbfgs_b_solver` are internal or direct low-level APIs, not universal estimator `solver=` values. Some model-specific penalties refine a generic dispatch label into a fixed internal path; the model page states that boundary explicitly.

## 3. Solver Capabilities

| Solver | sample_weight | warm_start | Inference | Best for |
|--------|:------------:|:----------:|:---------:|----------|
| `exact` | ✅ | ❌ | ✅ (OLS) | squared_error + l2 |
| `irls` | ✅ | ❌ | ❌ | GLM + l2 |
| `newton` | uniform only | ❌ | ❌ | smooth objectives |
| `lbfgs` | uniform only | ❌ | ❌ | large smooth objectives |
| `fista` | ✅ | ✅ | ❌ | convex group/sparse objectives and LLA inner solves |
| `fista_bb` | ✅ | ✅ | ❌ | supported sparse objectives with adaptive steps |
| `admm` | uniform only | ✅ | ❌ | supported proximal objectives |
| `coordinate_descent` | ✅ | ✅ | ❌ | CPU squared-error L1/Elastic Net |

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
