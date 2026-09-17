# Solver × Penalty Compatibility Matrix

> Language: English  
> Last updated: 2026-09-17  
> This page: Compatibility reference  
> Switch: [Chinese](../../cn/guides/solver-penalty-matrix.md)

## Overview

This page is the compact model-level reference for **which solver statgpu selects or permits for a given loss × penalty combination**. It is intended to be read as a matrix, not as model-specific algorithm documentation.

Use this page to answer three questions:

1. What does direct-fit `solver="auto"` select?
2. What numerical conditions constrain an explicit solver request?
3. What does `PenalizedGLM_CV` select under `solver="auto"`?

Detailed model behavior belongs on the model pages; update equations and algorithmic assumptions belong in [Solver Algorithms](solver-algorithms.md).

General conventions:

- `none` / `null` is canonicalized to `L2(alpha=0)` before solver selection;
- an explicit solver request is validated before numerical fitting and is not silently replaced because weights are present;
- backend-specific entries are shown only when the backend changes the effective route;
- internal resolved labels such as FISTA-LLA or Proximal IRLS-CD may appear in the matrix even when they are not public `solver=` keywords.

`AdaptiveGroupLassoPenalty` is available as a public penalty object but has no string-registry alias because callers must supply explicit group weights.

## 1. Direct-fit `solver="auto"`

| Loss | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | CPU `exact` / GPU Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **logistic** | Newton | FISTA-BB | FISTA-BB | FISTA-LLA | FISTA-LLA | FISTA-BB | Group FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **poisson** | Newton | FISTA-BB | FISTA-BB | FISTA-LLA | FISTA-LLA | FISTA-BB | Group FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **gamma** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **inverse_gaussian** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **negative_binomial** | Newton | FISTA-BB | FISTA-BB | FISTA-LLA | FISTA-LLA | FISTA-BB | Group FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **tweedie** | Newton | CPU FISTA-BB / GPU FISTA | CPU FISTA-BB / GPU FISTA | FISTA-LLA | FISTA-LLA | CPU FISTA-BB / GPU FISTA | Group FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **quantile** | IRLS | FISTA | FISTA | Proximal IRLS-CD | Proximal IRLS-CD | FISTA | Group FISTA | Group FISTA-LLA | Group FISTA-LLA |

### Reading the table

- The cells show the **effective automatic route**, not every explicit solver that may be valid.
- FISTA-LLA denotes the non-convex continuation route used for scalar SCAD/MCP objectives; Group FISTA-LLA is the corresponding group route.
- Proximal IRLS-CD is a specialized resolved route rather than a public explicit solver keyword.
- Group Lasso and Adaptive Group Lasso use the group-aware FISTA path; Group SCAD/MCP use a group-aware LLA path.
- Model-specific reasons for a cell belong in the corresponding model page rather than in this matrix.

For family/link domain restrictions, weighting semantics, or special initialization rules, see [GeneralizedLinearModel](../models/generalized-linear-model.md). For Quantile-specific solver choices and non-smooth behavior, see [Quantile Regression](../models/quantile.md).

## 2. Explicit solver constraints

This table summarizes the numerical prerequisites for **model-level explicit solver requests**. Low-level solver functions can have narrower or different contracts and should be read from their API/algorithm documentation.

| Solver | Main numerical prerequisite | Typical penalty scope | Public-request notes |
|--------|-----------------------------|-----------------------|----------------------|
| `exact` | quadratic squared-error objective | L2 / none | squared-error route only |
| `irls` | the loss exposes an estimator-level IRLS route | L2 / none | family/loss specific |
| `newton` | smooth objective with Hessian support | L2 / none | uses Newton + line search |
| `lbfgs` | smooth objective with a consistent gradient | L2 / none | avoids forming a full Hessian |
| `fista` | compatible first-order loss primitive plus proximal penalty step | convex proximal routes and selected explicit routes | route-specific support |
| `fista_bb` | smooth-gradient differences suitable for BB step adaptation | supported sparse proximal routes | excluded when the loss lacks meaningful smooth-gradient differences |
| `admm` | supported splitting with a smooth w-subproblem | supported proximal formulations | route-specific support |
| `irls_cd` | specialized scalar IRLS/coordinate-descent formulation | specialized routes | not a general-purpose fallback |
| `proximal_irls_cd` | specialized proximal IRLS majorization | specialized non-convex routes | internal resolved label; not a public explicit `solver=` keyword |
| `proximal_newton` | compatible Newton/proximal structure | route specific | behavior depends on loss and penalty structure |

Unsupported explicit estimator combinations raise an error before numerical fitting. The matrix intentionally states the shared numerical conditions here; model-specific exclusions and alternatives are documented on the corresponding model page.

## 3. Solver capability summary

| Solver | Core requirement | Typical use | `sample_weight` | `warm_start` |
|--------|------------------|-------------|-----------------|:------------:|
| `exact` | quadratic closed form / eigensystem | squared error + L2 | supported on its declared route | ❌ |
| `irls` | loss-specific reweighted least-squares update | supported L2/no-penalty routes | loss/estimator dependent | ❌ |
| `newton` | gradient + Hessian | smooth L2/no-penalty objectives | loss/estimator dependent | ❌ |
| `lbfgs` | consistent smooth gradient | smooth L2/no-penalty objectives | loss/estimator dependent | ❌ |
| `fista` | first-order loss primitive + proximal step | convex proximal objectives and LLA inner solves | route dependent | ✅ |
| `fista_bb` | smooth-gradient differences + proximal step | sparse objectives with adaptive BB steps | route dependent | ✅ |
| `admm` | compatible splitting and smooth w-update | proximal formulations | route dependent | ✅ |
| `irls_cd` | specialized IRLS + coordinate descent | specialized scalar routes | route dependent | ✅ |

`sample_weight` support is a **loss × solver × estimator** contract; it cannot be inferred from a solver signature alone. Weight semantics and unsupported combinations are documented on the relevant model page and in [Loss × Penalty × Solver Framework](loss-penalty-solver-framework.md).

## 4. CV `solver="auto"` (`PenalizedGLM_CV`)

Cross-validation may intentionally choose a different numerical route from direct fitting because candidate fitting, backend execution, and final refitting have different performance tradeoffs.

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | eig-batch / exact-style CPU path; Newton on GPU final fits where applicable | FISTA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **logistic** | Newton | FISTA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **poisson** | Newton | CPU FISTA; GPU L1 may use size-gated FISTA-BB and GPU ElasticNet uses FISTA-BB | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **gamma** | L-BFGS | FISTA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **inverse_gaussian** | L-BFGS | FISTA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **negative_binomial** | L-BFGS | FISTA-BB, except a GPU ElasticNet size band uses FISTA | FISTA-LLA | FISTA-BB | Group FISTA | Group FISTA-LLA |
| **tweedie** | Newton | CPU FISTA-BB / GPU FISTA | FISTA-LLA | CPU FISTA-BB / GPU FISTA | Group FISTA | Group FISTA-LLA |
| **quantile** | IRLS | FISTA | Proximal IRLS-CD | FISTA | Group FISTA | Group FISTA-LLA |

### CV notes

- The table records the actual automatic route; backend- or size-dependent policies are shown directly in the affected cells.
- An explicit solver request remains authoritative when that loss × penalty × solver combination is supported; CV does not silently replace it simply because folds or weights are present.
- Candidate fits and the selected full-data refit preserve the resolved loss, penalty, groups, and solver contract.
- Group validation is performed before candidate fitting; detailed group-input rules are documented in [Loss × Penalty × Solver Framework](loss-penalty-solver-framework.md).
- Strict/two-stage CV semantics and model-specific validation behavior are documented on the relevant model pages rather than duplicated here.

## 5. Penalty reference

| Penalty | Formula | Proximal / surrogate form | Main parameters |
|---------|---------|---------------------------|-----------------|
| `l2` | ½α‖β‖² | ridge scaling | `alpha` |
| `l1` | α‖β‖₁ | soft threshold | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft threshold + L2 scaling | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD thresholding / LLA | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP thresholding / LLA | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | weighted soft threshold | `alpha`, weights |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | block soft threshold | `alpha`, `groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | weighted block soft threshold | `alpha`, `groups`, `weights`; object-only |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | group LLA surrogate | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | group LLA surrogate | `alpha`, `groups`, `gamma` |

For Group SCAD/MCP, the convex LLA surrogate is represented through an adaptive group-lasso problem. Group metadata must match the final design width; exact validation rules are documented in [Loss × Penalty × Solver Framework](loss-penalty-solver-framework.md).

## 6. Related references

This page intentionally does not reproduce model-specific derivations, optimization safeguards, inference contracts, or validation procedures.

- [Solver Algorithms](solver-algorithms.md) — update equations, convergence/stopping behavior, and algorithmic assumptions
- [Loss × Penalty × Solver Framework](loss-penalty-solver-framework.md) — computation architecture and dispatch concepts
- [Loss Functions](../models/losses.md) — loss-layer mathematics and numerical primitives
- [GeneralizedLinearModel](../models/generalized-linear-model.md) — GLM family/link behavior, weights, CV, and inference
- [Quantile Regression](../models/quantile.md) — Quantile-specific solver, penalty, weighting, and inference behavior
- [Robust Regression](../models/robust.md) — robust-loss estimator behavior
- [Penalized GLM inference](penalized-glm-inference.md) and [Inference Modes](inference-modes.md) — inference support and interpretation
