# Technical Debt — PR #49 Code Review Residuals

> Generated from PR #49 code review (2026-06-10). All P1/P2 bugs and easy P3
> items have been fixed. This document tracks remaining P3 items that require
> larger refactoring efforts.

## Status Summary

| Severity | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| P1 (must fix) | 10 | 10 | 0 |
| P2 (should fix) | 26 | 26 | 0 |
| P3 (nice to have) | ~35 | ~20 | ~15 |

---

## Remaining P3 Items

### 1. Code Duplication — `_penalized_cv.py` Backend Branches

**Location**: `_penalized_cv.py` (5+ functions)
**Issue**: Nearly every function has a 3-way `if torch / elif cupy / else` block
that repeats the same logic with slightly different API calls. The `_fb_*`
helpers exist for the fold-batch path but are not used in
`_logistic_sparse_cv_path`, `_squared_error_sparse_cv_path`,
`_glm_sparse_cv_path`, or `_scad_mcp_cv_path`.
**Effort**: High — requires extending the `_fb_*` helper set or adopting a
lightweight backend abstraction.
**Impact**: Adding a new backend or fixing a bug requires editing 5+ identical
code blocks.

### 2. Code Duplication — `_elasticnet_cv.py` Alpha Grid Functions

**Location**: `_elasticnet_cv.py:128-252`
**Issue**: `_default_elasticnet_alpha_grid` (numpy) and
`_default_elasticnet_alpha_grid_backend` (GPU) do the same thing with
different array backends. The numpy version has slightly different logic for
`l1_ratio=0` (pure Ridge), meaning the alpha grid could differ between CPU
and GPU paths.
**Effort**: Medium — merge into a single backend-agnostic function using
`_xp()` helpers.
**Impact**: Potential CPU/GPU result divergence.

### 3. Code Duplication — `_logistic_cv.py` Log-Loss Functions

**Location**: `_logistic_cv.py:154-227`
**Issue**: `_batch_log_loss` (numpy) and `_batch_log_loss_backend` (GPU) are
nearly identical. Same pattern as the elasticnet alpha grid.
**Effort**: Medium — consolidate using backend dispatch.
**Impact**: Maintenance burden, risk of divergence.

### 4. Code Duplication — `_hash_logistic_data` vs `_hash_data`

**Location**: `_logistic_cv.py:44-62` vs `_elasticnet_cv.py:93-114`
**Issue**: Near-identical hash functions for cache keys.
**Effort**: Low — extract to `_cv_base.py` as shared utility.
**Impact**: Minor maintenance burden.

### 5. Code Duplication — `_penalized.py` Debiased Inference

**Location**: `_penalized.py:927-1211`
**Issue**: `_compute_inference_debiased_gpu()` and
`_compute_inference_debiased_torch()` are ~280 lines of nearly identical code.
They differ only in CuPy vs Torch API calls.
**Effort**: High — refactor into a single backend-agnostic method.
**Impact**: Major maintenance burden.

### 6. Performance — `_elasticnet_cv.py` XtX Redundant Computation

**Location**: `_elasticnet_cv.py:621-625`
**Issue**: `XtX_fold` and eigenvalues are recomputed for every l1_ratio inside
the fold loop. Since `XtX_fold` depends only on the fold (not on l1_ratio),
it could be computed once per fold and reused across l1_ratios.
**Effort**: Medium — requires restructuring the loop.
**Impact**: O(n_l1_ratios) redundant O(n_features^3) work per fold.

### 7. Performance — `_solver.py` Hardcoded Loss Names

**Location**: `_solver.py:626`
**Issue**: The fused GLM value+gradient check uses a hardcoded set of loss
names. If a new loss is added to `_GLM_FUSED_REGISTRY`, the caller must also
update this list.
**Effort**: Low — use `if _loss_name in _GLM_FUSED_REGISTRY:`.
**Impact**: Silent fallback to slower path if not updated.

### 8. Performance — `_penalized.py` `_fit_initial` Always CPU

**Location**: `_penalized.py:1387-1443`
**Issue**: `_fit_initial()` always converts to numpy and runs on CPU, even
when the main fit will run on GPU. For large datasets, this CPU init can be
a bottleneck.
**Effort**: Medium — run init on the same backend.
**Impact**: Unnecessary D2H + H2D transfers for GPU users.

### 9. Performance — `_penalized.py` Node-wise Lasso Loop

**Location**: `_penalized.py:849-865`
**Issue**: The debiased inference node-wise Lasso loop creates a new
`PenalizedLinearRegression` instance for each feature j (up to p times).
Each instance goes through the full `__init__` and `fit()` machinery.
**Effort**: High — reuse a single instance with `warm_start=True` or use
a lower-level solver directly.
**Impact**: Per-instance overhead for p=1000 features.

### 10. Performance — `_logistic_cv.py` GPU Probability Not Vectorized

**Location**: `_logistic_cv.py:616-624`
**Issue**: The probability computation loops over each C value individually.
Could be vectorized by stacking all coefs and doing a single matrix multiply.
**Effort**: Medium — similar to `_batch_mse_elasticnet` approach.
**Impact**: Unnecessary GPU kernel launches.

### 11. Code Duplication — `_penalized.py` Local SelectivePenalty

**Location**: `_penalized.py:4558-4623`
**Issue**: A local `SelectivePenalty` class is defined inside
`_fit_cpu_loss()`, duplicating the module-level class at line 206.
**Effort**: Low — reuse `_get_selective_penalty_singleton().configure()`.
**Impact**: Minor — two copies to maintain.

### 12. Style — `_elasticnet_cv.py` Two-Step Candidate Mask

**Location**: `_penalized_cv.py:91-111`
**Issue**: `_two_stage_candidate_mask` marks all finite scores when none are
finite. This is a safe fallback but could log a warning (already fixed).
**Effort**: N/A (already addressed).
**Impact**: None.

### 13. Style — `_cv_base.py` `folds_are_complements` Dead Code

**Location**: `_cv_base.py:71-78`
**Issue**: `folds_are_complements` is defined but never imported anywhere
in production code.
**Effort**: Low — remove or mark as test utility.
**Impact**: None.

### 14. Style — `_cv_engine.py` is Reference Implementation

**Location**: `_cv_engine.py` (entire file)
**Issue**: `run_cv` is only called from test files. The module docstring
acknowledges this ("The production CV paths use their own optimized loops").
This entire file is effectively a reference implementation shipped as
production code.
**Effort**: Low — move to `dev/` or mark as reference.
**Impact**: Confusion about which code path is actually used.

---

## Priority Order (Recommended)

1. **#4** `_hash_logistic_data` consolidation (Low effort, Low risk)
2. **#7** `_solver.py` loss name registry check (Low effort, Low risk)
3. **#11** Local SelectivePenalty reuse (Low effort, Low risk)
4. **#13** Remove `folds_are_complements` dead code (Low effort, No risk)
5. **#2** Alpha grid function consolidation (Medium effort, Medium risk)
6. **#3** Log-loss function consolidation (Medium effort, Medium risk)
7. **#6** XtX redundant computation (Medium effort, Medium risk)
8. **#8** `_fit_initial` GPU support (Medium effort, Medium risk)
9. **#10** GPU probability vectorization (Medium effort, Medium risk)
10. **#1** Backend branch abstraction (High effort, High risk)
11. **#5** Debiased inference consolidation (High effort, High risk)
12. **#9** Node-wise Lasso optimization (High effort, High risk)
