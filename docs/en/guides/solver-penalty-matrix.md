# Solver × Penalty Compatibility Matrix

> Language: English  
> Last updated: 2026-09-15  
> This page: Reference guide  
> Switch: [Chinese](../../cn/guides/solver-penalty-matrix.md)

## Overview

This page is the model-level reference for choosing a solver in `PenalizedGeneralizedLinearModel` and `PenalizedGLM_CV`.

The most important distinction is between **direct fitting** and **cross-validation**:

- direct `solver="auto"` uses the direct-fit dispatch table in section 1;
- `PenalizedGLM_CV` has a related but intentionally different policy, shown in section 4;
- an explicit solver request is validated before numerical work and is never silently replaced because `sample_weight` is present.

`none` / `null` penalties are canonicalized to `L2(alpha=0)` before solver selection. Therefore an unpenalized smooth route follows the same auto-dispatch branch as L2.

`AdaptiveGroupLassoPenalty` is available as a public penalty object but intentionally has no string-registry alias because callers must supply explicit group weights.

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

### How to read the table

- The table describes the **effective maintained route**, not only an internal dispatch string.
- Smooth Quantile L2/no-penalty `auto` resolves to ordinary Quantile IRLS. Sparse Quantile L1/ElasticNet remains on FISTA-family routes. Quantile SCAD/MCP is different again: it uses the dedicated Proximal IRLS-CD continuation path rather than ordinary Quantile IRLS.
- `fista_lla` is an internal continuation path, not a public `solver=` keyword. For squared-error SCAD/MCP, `fit()` enters the fused `fista_lla_path()` directly.
- Direct logistic, Poisson, and Negative-Binomial sparse convex rows reach the default FISTA-BB rule. Gamma and Inverse-Gaussian sparse rows are explicitly pinned to FISTA. Tweedie sparse rows use FISTA on CuPy/Torch and the default FISTA-BB route on CPU.
- Group Lasso and Adaptive Group Lasso use the group-aware FISTA path. Group SCAD/MCP use a weighted Group-Lasso LLA surrogate with a group-aware FISTA inner solve.
- `sample_weight` does not rewrite an explicit solver request. Unsupported loss/solver/weight combinations raise rather than silently selecting another algorithm.

### Inverse-power Gamma smooth-domain contract

For inverse-power Gamma,

\[
\eta_i=x_i^\top\beta>0,
\qquad
\ell_i(\eta_i)=y_i\eta_i-\log\eta_i.
\]

The maintained explicit `newton` and `lbfgs` paths construct a backend-native interior start and keep every **active training** predictor inside the unclipped numerical interval where the implemented value, gradient, and Hessian describe one smooth objective. The Armijo step cap is computed only after the solver has finalized its actual post-fallback search direction.

`fit_intercept=False` is therefore not categorically rejected. It is supported when statgpu can certify an interior start for the executed design and the optimizer converges without being pinned to the numerical-domain boundary. Omitted, uniform, effectively-uniform, and genuinely non-uniform analytic weights use the same domain policy; with genuine weighting, rows whose analytic weight is exactly zero do not constrain the training domain.

If a finite design cannot be numerically certified, or optimization reaches the domain boundary before convergence, the explicit smooth path fails visibly instead of publishing a clipped surrogate fit. Public prediction and held-out validation retain their existing clipping semantics, so this training-domain guarantee is not a claim that every unseen design row must remain in the training interval.

## 2. Explicit solver constraints

| Solver | Accepts | Rejects / limits | Notes |
|--------|---------|------------------|-------|
| `exact` | L2 + squared error only | everything else | closed-form/eigendecomposition path |
| `irls` | L2/no penalty on losses declaring maintained IRLS support | non-smooth penalties | loss/family-specific IRLS; smooth Quantile `auto` also resolves to this route |
| `newton` | L2 / none on smooth losses with Hessian support | L1, ElasticNet, non-convex and group penalties | Newton + Armijo line search |
| `lbfgs` | L2 / none on smooth losses | L1, ElasticNet, non-convex and group penalties | limited-memory BFGS + line search |
| `fista` | supported proximal penalties | smooth Quantile L2/no penalty and unsupported model combinations | explicit smooth Quantile FISTA fails instead of silently executing IRLS |
| `fista_bb` | supported sparse penalties | smooth Quantile L2/no penalty and unsupported combinations | FISTA + BB step adaptation |
| `admm` | supported proximal formulations | unsupported combinations | variable splitting + proximal update |
| `irls_cd` | specialized scalar routes | unsupported combinations | not the current squared-error SCAD/MCP public auto route |
| `proximal_irls_cd` | **not a public explicit solver keyword** | all user-supplied explicit requests | internal resolved label for Quantile SCAD/MCP selected through `solver="auto"`; Proximal IRLS-CD majorization + LLA |
| `proximal_newton` | L2 / none uses Newton; non-smooth direct calls visibly use FISTA | unsupported penalty structures | no Euclidean-prox approximation |

Unsupported explicit combinations fail before numerical fitting. In particular, users request Quantile SCAD/MCP through `solver="auto"`; `proximal_irls_cd` is published only as internal/executed solver provenance.

## 3. Solver capabilities

| Solver | `sample_weight` | `warm_start` | Inference | Best for |
|--------|:---------------:|:------------:|:---------:|----------|
| `exact` | ✅ on its maintained route | ❌ | ✅ (OLS path) | squared error + L2 |
| `irls` | estimator/loss dependent | ❌ | estimator dependent | maintained smooth Quantile and GLM IRLS routes |
| `newton` | maintained GLMs support analytic weights | ❌ | estimator dependent | smooth objectives with Hessian support |
| `lbfgs` | maintained GLMs support analytic weights; other losses are route-specific | ❌ | estimator dependent | smooth objectives without forming a full Hessian |
| `fista` | ✅ on maintained weighted routes | ✅ | estimator dependent | convex sparse/group objectives and LLA inner solves |
| `fista_bb` | ✅ on maintained weighted routes | ✅ | estimator dependent | supported sparse objectives with adaptive steps |
| `admm` | shared `admm_solver`: omitted/uniform weights only | ✅ | estimator dependent | supported proximal formulations |
| `irls_cd` | route-specific | ✅ | estimator dependent | specialized scalar coordinate-descent routes |

For Newton and L-BFGS, `sample_weight` support is a **loss/estimator contract**, not a property that can be inferred from the solver signature alone. Maintained GLM losses use

`sum(w_i * loss_i) / sum(w_i)`

for the data-fit term. Generic robust, quantile, and Cox direct L-BFGS consumers retain their own weight boundaries. The shared `admm_solver` requires `sample_weight` to be omitted or uniform; genuinely non-uniform analytic weights fail before numerical iteration.

Group warm starts carry coefficient and intercept state together for one fit call and are cleared after success or failure.

## 4. CV support (`PenalizedGLM_CV`)

`PenalizedGLM_CV` keeps the public `solver="auto"` request but uses a family-, backend-, and sometimes problem-size-specific policy for candidate fits and the selected full-data final refit.

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | eig-batch / exact-style CPU path; Newton on GPU final fits where applicable | FISTA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **logistic** | Newton | FISTA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **poisson** | Newton | CPU FISTA; GPU L1 may use size-gated FISTA-BB and GPU ElasticNet uses FISTA-BB | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **gamma** | L-BFGS | FISTA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **inverse_gaussian** | L-BFGS | FISTA | FISTA-LLA | FISTA | Group FISTA | Group FISTA-LLA |
| **negative_binomial** | L-BFGS | FISTA-BB, except a maintained GPU ElasticNet size band uses FISTA | FISTA-LLA | FISTA-BB | Group FISTA | Group FISTA-LLA |
| **tweedie** | Newton | CPU FISTA-BB / GPU FISTA | FISTA-LLA | CPU FISTA-BB / GPU FISTA | Group FISTA | Group FISTA-LLA |
| **quantile** | IRLS | FISTA | Proximal IRLS-CD | FISTA | Group FISTA | Group FISTA-LLA |

For Quantile CV, the same policy is used for candidate fitting and the selected full-data refit: smooth L2/no-penalty rows report and execute IRLS; convex sparse rows remain FISTA-family. SCAD/MCP remains its separate Proximal IRLS-CD continuation algorithm.

The Poisson GPU L1 FISTA-BB rule is size-gated: the maintained fast path applies below roughly two million design elements; larger rows use FISTA. Negative-Binomial GPU ElasticNet uses FISTA in the maintained medium-size band (roughly 200k–1M design elements) and FISTA-BB outside that band. These thresholds are internal dispatch policy, not universal performance guarantees.

Weights do not substitute another solver. For the three L-BFGS smooth-L2 families above, weighted candidate/final-refit support follows the maintained GLM weighted-objective contract. For `loss="gamma"` with `loss_kwargs={"link": "inverse_power"}`, smooth-L2 CV preserves that actual loss object through candidate fitting, validation scoring, alpha selection, and final refit rather than using the log-link-only Gamma validation shortcut.

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

For Group SCAD/MCP, let `D_g` denote the derivative with respect to `‖β_g‖₂`. The exact convex surrogate is `Σ_g D_g‖β_g‖₂`, represented internally by `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`. Target alpha and group size are not multiplied twice. Group LLA uses FISTA rather than the generic Proximal Newton branch.

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

```text
direct solver="auto"
├── squared_error + L2/none?             → CPU exact / GPU Newton
├── quantile + L2/none?                  → IRLS
├── quantile + L1/ElasticNet?            → FISTA
├── quantile + SCAD/MCP?                 → Proximal IRLS-CD
├── smooth non-Gaussian GLM + L2/none?  → Newton
├── squared_error sparse convex?         → FISTA
├── gamma / inverse-Gaussian sparse?     → FISTA
├── logistic / poisson / NB sparse?      → FISTA-BB
├── tweedie sparse?                      → CPU FISTA-BB / GPU FISTA
├── scalar SCAD/MCP?                     → FISTA-LLA
├── convex group penalty?                → Group FISTA
└── group SCAD/MCP?                      → Group FISTA-LLA
```

For `PenalizedGLM_CV`, use the separate CV table above. Gamma, Inverse-Gaussian, and Negative-Binomial L2 rows intentionally use L-BFGS during CV/final refit even though direct-fit `auto` uses Newton.
