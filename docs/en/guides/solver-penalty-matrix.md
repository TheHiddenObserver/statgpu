# Solver × Penalty Compatibility Matrix

> Language: English  
> Last updated: 2026-09-12  
> This page: Reference guide  
> Switch: [Chinese](../../cn/guides/solver-penalty-matrix.md)

## Overview

This page is the model-level reference for choosing a solver in `PenalizedGeneralizedLinearModel` and `PenalizedGLM_CV`.

The most important distinction is between **direct fitting** and **cross-validation**:

- direct `solver="auto"` uses the direct-fit dispatch table in section 1;
- `PenalizedGLM_CV` has a related but intentionally different smooth-L2 policy, shown in section 4;
- an explicit solver request is validated before numerical work and is never silently replaced because `sample_weight` is present.

`AdaptiveGroupLassoPenalty` is available as a public penalty object but intentionally has no string-registry alias because callers must supply explicit group weights.

## 1. Direct-fit `solver="auto"`

| Loss | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | l2: exact on CPU / Newton on GPU; none: FISTA | FISTA | FISTA | IRLS-CD → FISTA-LLA | IRLS-CD → FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **logistic** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **poisson** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **gamma** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **inverse_gaussian** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **negative_binomial** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **tweedie** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |

### How to read the table

- `fista_lla` is an internal continuation path, not a public `solver=` keyword. The exported `fista_lla_path()` function uses the same surrogate when called directly.
- Scalar squared-error SCAD/MCP may use coordinate-descent continuation. Group SCAD/MCP use a weighted Group-Lasso surrogate with a group-aware FISTA inner solve.
- Group Lasso and Adaptive Group Lasso use the advertised loss gradient and the Euclidean group proximal operator, including maintained `sample_weight` and CV routes.
- `sample_weight` does not rewrite an explicit solver request. Supported weighted Newton/L-BFGS routes use the same normalized weighted objective throughout optimization; unsupported loss/solver/weight combinations raise.

### Inverse-power Gamma boundary

For ordinary `GammaRegression(link="inverse_power")`, genuinely non-uniform weighted explicit Newton/L-BFGS requires `fit_intercept=True` so statgpu can construct a strictly positive family-valid starting predictor.

Only that **non-uniform weighted + no-intercept + explicit Newton/L-BFGS** combination is rejected. Omitted, uniform, or effectively-uniform weights retain the historical no-intercept behavior.

## 2. Explicit solver constraints

| Solver | Accepts | Rejects / limits | Notes |
|--------|---------|------------------|-------|
| `exact` | L2 + squared error only | everything else | closed-form/eigendecomposition path |
| `irls` | L2 on supported IRLS losses | non-smooth penalties | loss/family-specific IRLS |
| `newton` | L2 / none on smooth losses with Hessian support | L1, ElasticNet, non-convex and group penalties | Newton + Armijo line search |
| `lbfgs` | L2 / none on smooth losses | L1, ElasticNet, non-convex and group penalties | limited-memory BFGS + line search |
| `fista` | supported proximal penalties | unsupported model combinations | Nesterov proximal gradient |
| `fista_bb` | supported sparse penalties | unsupported combinations | FISTA + BB step adaptation |
| `admm` | supported proximal formulations | unsupported combinations | variable splitting + proximal update |
| `irls_cd` | scalar SCAD/MCP/adaptive-L1 routes | L1/ElasticNet and group penalties | IRLS outer + coordinate descent inner |
| `proximal_irls_cd` | quantile + scalar SCAD/MCP | non-quantile losses and group penalties | quantile majorization + LLA |
| `proximal_newton` | L2 / none uses Newton; non-smooth direct calls visibly use FISTA | unsupported penalty structures | no silent Euclidean-prox approximation |

Unsupported explicit combinations fail before numerical fitting.

## 3. Solver capabilities

| Solver | `sample_weight` | `warm_start` | Inference | Best for |
|--------|:---------------:|:------------:|:---------:|----------|
| `exact` | ✅ on its maintained route | ❌ | ✅ (OLS path) | squared error + L2 |
| `irls` | estimator/loss dependent | ❌ | estimator dependent | maintained IRLS GLMs |
| `newton` | maintained GLMs support analytic weights | ❌ | estimator dependent | smooth objectives with Hessian support |
| `lbfgs` | maintained GLMs support analytic weights; other losses are route-specific | ❌ | estimator dependent | smooth objectives without forming a full Hessian |
| `fista` | ✅ on maintained weighted routes | ✅ | estimator dependent | convex sparse/group objectives and LLA inner solves |
| `fista_bb` | ✅ on maintained weighted routes | ✅ | estimator dependent | supported sparse objectives with adaptive steps |
| `admm` | combination dependent | ✅ | estimator dependent | supported proximal formulations |
| `irls_cd` | ✅ on maintained routes | ✅ | estimator dependent | scalar non-convex continuation routes |

For Newton and L-BFGS, `sample_weight` support is a **loss/estimator contract**, not a property that can be inferred from the solver signature alone. Maintained GLM losses use

`sum(w_i * loss_i) / sum(w_i)`

for the data-fit term. Generic robust, quantile, and Cox direct L-BFGS consumers retain their own weight boundaries.

Group warm starts carry coefficient and intercept state together for one fit call and are cleared after success or failure.

## 4. CV support (`PenalizedGLM_CV`)

`PenalizedGLM_CV` keeps the public `solver="auto"` request but uses a family-specific smooth-L2 policy for candidate fits and the selected full-data final refit.

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | eig-batch / exact-style path | sparse FISTA | LLA + FISTA/CD | general fit | Group FISTA | Group FISTA-LLA |
| **logistic** | Newton | sparse FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **poisson** | Newton | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **gamma** | L-BFGS | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **inverse_gaussian** | L-BFGS | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **negative_binomial** | L-BFGS | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **tweedie** | Newton | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |

Weights do not silently substitute another solver. For the three L-BFGS smooth-L2 families above, weighted candidate/final-refit support follows the maintained GLM weighted-objective contract.

Group validation happens before alpha-grid generation, fold construction, or candidate fitting. Groups are interpreted against the final design width, including formula-expanded columns. Missing unweighted features are completed as singleton groups once; out-of-range indices and incomplete adaptive weighted groups fail before candidate fitting.

CV uses fit-local penalty state and does not mutate a caller's penalty object or `penalty_kwargs` dictionary. For penalty objects, each candidate is rebuilt at the candidate alpha; the selected final estimator exposes a penalty snapshot whose alpha and groups match the resolved objective. The top-level CV estimator retains its original constructor parameter.

## 5. Penalty reference

| Penalty | Formula | Proximal form | Main parameters |
|---------|---------|---------------|-----------------|
| `l2` | ½α‖β‖² | ridge scaling | `alpha` |
| `l1` | α‖β‖₁ | soft threshold | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft threshold + L2 scaling | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD thresholding / LLA route | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP thresholding / LLA route | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | weighted soft threshold | `alpha`, weights |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | block soft threshold | `alpha`, `groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | weighted block soft threshold | `alpha`, `groups`, `weights`; object-only |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | group LLA surrogate | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | group LLA surrogate | `alpha`, `groups`, `gamma` |

For Group SCAD/MCP, let `D_g` denote the derivative with respect to `‖β_g‖₂`. The exact convex surrogate is `Σ_g D_g‖β_g‖₂`, represented internally by `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`. Target alpha and group size are not multiplied twice. Group LLA uses FISTA rather than the generic proximal-Newton branch.

Group inputs use a strict contract: hyperparameters must be finite numeric scalars; group indices/IDs must be non-negative integer-valued numerics representable as signed `int64`; explicit groups must be non-empty and duplicate-free; flat IDs must be contiguous from zero; and public numerical penalty methods require exactly the grouped feature dimension.

## 6. Inference support

| Penalty | Inference method | Status |
|---------|------------------|--------|
| `l2` | standard / M-estimation where exposed by the estimator | ✅ Available on maintained routes |
| `l1` | Debiased Lasso | ✅ Supported routes |
| `elasticnet` | method dependent | See estimator contract |
| `scad` / `mcp` | oracle/bootstrap where implemented | See estimator contract |
| `adaptive_l1` | method dependent | See estimator contract |
| Group Lasso / Adaptive Group Lasso / Group SCAD / Group MCP | group-preserving covariance/bootstrap | Not implemented; inference requests fail before fitting |

## 7. Choosing a solver

For most users, start with `solver="auto"` and override it only when you have a reason to require a particular algorithm.

```
direct solver="auto"
├── squared_error + L2?                 → CPU exact / GPU Newton
├── squared_error + none?               → FISTA
├── smooth non-Gaussian GLM + L2/none?  → Newton
├── scalar non-convex penalty?          → scalar LLA path
├── convex group penalty?               → Group FISTA
└── group SCAD/MCP?                     → Group FISTA-LLA
```

For `PenalizedGLM_CV`, use the separate CV table above. Gamma, Inverse-Gaussian, and Negative-Binomial L2 rows intentionally use L-BFGS during CV/final refit even though direct-fit `auto` uses Newton.
