# Cross-Validation Implementation Architecture

This document owns internal implementation details for statgpu cross-validation. It is an engineering reference, not a user guide. Public CV semantics belong in `docs/en/guides/cross-validation.md` and `docs/cn/guides/cross-validation.md`.

## 1. Ownership boundary

The public CV contract is selection followed by a full-data refit. This file owns the implementation details behind that contract:

- fold materialization and validation;
- automatic alpha-grid construction;
- CV-device heuristics;
- specialized scoring paths;
- backend batching and synchronization choices;
- selection-only caches;
- implementation provenance and validation entry points.

Model-specific statistical semantics still belong to the corresponding model implementation. In particular, Cox risk-set mathematics is owned by `statgpu/survival/`, while generic loss/penalty/solver composition is owned by the penalized fit layer.

## 2. Scalar-response sequence

The generic scalar-response flow is conceptually:

```text
PenalizedGLM_CV._fit_standard(X, y)
  |
  +-- validate or generate the complete alpha grid
  |
  +-- materialize generated/custom folds exactly once
  |
  +-- resolve the CV device
  |
  +-- score the candidate grid
  |     +-- specialized fast path when available
  |     +-- general per-fold/per-alpha path otherwise
  |
  +-- choose the best eligible candidate
  |
  +-- refit the selected configuration on all observations
```

Generated/custom folds are normalized to reusable arrays before scoring so one-shot generators are not consumed differently by device selection, scoring, or diagnostics.

## 3. Penalized-Cox sequence

Penalized Cox needs a different preparation order because fold event support and backend-native Cox preprocessing affect which work is evaluable:

```text
fit_penalized_cox_cv(estimator, X, (time, event))
  |
  +-- normalize survival targets and materialize folds
  |
  +-- validate the alpha-grid request
  |
  +-- validate event support for train/validation partitions
  |
  +-- resolve the CV device from effective work
  |
  +-- convert/preprocess on the selected backend
  |
  +-- generate an automatic grid when requested
  |
  +-- score evaluable folds and retain diagnostics
  |
  +-- require finite candidate evidence, select, and refit
```

Do not collapse the scalar and Cox preparation orders into one implementation diagram unless the runtime is actually unified: automatic-grid construction occurs at different points.

## 4. Scoring-path families

Current implementations use several scoring-path families. These are implementation choices, not independent public statistical methods.

### Squared-error L2

CPU CV can reuse a fold-level eigendecomposition across the alpha grid. The selected full-data Ridge result may use the exact float64 CPU eigensolve while retaining the public prediction/output-device behavior defined by the estimator.

### Sparse squared-error paths

L1/ElasticNet paths can reuse fold-level Gram/cross-product state and warm-start coefficients across descending alpha values. The concrete algorithm depends on the resolved CV solver and backend.

### Batched GPU paths

Some GLM loss/penalty combinations batch multiple folds or candidate work on an accelerator to replace repeated matrix-vector operations with larger matrix operations. Backend synchronization is deliberately reduced in hot loops where correctness permits it.

### SCAD/MCP continuation

Non-convex paths use continuation/local-linear-approximation machinery appropriate to the selected estimator contract. Quantile SCAD/MCP uses its specialized proximal IRLS-CD route rather than being inferred from a generic smooth-loss fast path.

### General fallback

When no specialized scoring implementation applies, CV constructs the appropriate estimator for each fold/candidate, fits it under the requested public solver/device contract, evaluates validation evidence, and then performs the ordinary selection/refit sequence.

## 5. Device heuristics

`device="auto"` is allowed to choose a backend from workload and backend availability. The concrete thresholds live in code (including `_effective_cv_device()` and specialized path predicates) and are implementation tuning parameters, not stable public API guarantees.

Engineering changes to these heuristics must preserve:

- explicit `device="cpu"`, `"cuda"`, or `"torch"` authority;
- no silent CPU fallback for an unavailable explicit accelerator request;
- equivalent statistical candidate sets and scoring semantics;
- the selected configuration and final-refit contract.

Benchmark-derived thresholds should be updated together with their evidence rather than copied into the public CV guide.

## 6. LassoCV selection cache

`LassoCV` has a selection-only LRU cache in `statgpu.linear_model.wrappers._lasso`. The cache reuses alpha-selection evidence; it does not cache the final full-data estimator.

The current implementation represents each of `X`, `y`, and `sample_weight` through `_array_identity_token(...)`. The selection key is assembled by `_make_lasso_cv_auto_cache_key(...)` and includes the pieces that can change candidate scoring:

- the `X`, `y`, and `sample_weight` identity tokens;
- a digest of the complete evaluated alpha grid;
- complete train and validation index arrays for every fold;
- intercept mode and whether CV execution is GPU-backed;
- `max_iter` and `tol`;
- the resolved CV solver and normalized method controls;
- `cd_kkt_check_every`;
- `gpu_cv_mixed_precision`.

The final-refit `solver` is intentionally absent when it cannot change CV scoring. Cache hits return cloned NumPy arrays for the selection payload so callers cannot mutate stored evidence.

The LRU capacity defaults to **64** and is controlled at import time by `STATGPU_LASSO_CV_CACHE_SIZE`. Capacity, hashing strategy, sampled-content identity, and helper names are internal performance contracts and should remain here rather than in the public CV guide.

## 7. Performance implementation notes

Performance work may include:

- fold batching;
- warm starts across the alpha path;
- precomputed Gram matrices;
- backend-native validation-score accumulation;
- reduced host synchronization;
- compiled/fused accelerator kernels where supported.

These optimizations are acceptable only when they preserve the public objective, folds, candidate set, selection rule, explicit device semantics, and final-refit behavior.

## 8. Validation ownership

Exact benchmark numbers, physical-GPU acceptance thresholds, commit-specific evidence, and PR/Issue history belong in `dev/benchmarks/`, `dev/reviews/`, CI logs, or the corresponding issue/PR. They should not be duplicated into the ordinary public CV guide.

When a new CV route changes numerical/statistical behavior, tests should cover the active axes: candidate construction, folds, weights, scoring, selection, final refit, solver provenance, backend/device behavior, and inference-after-selection where applicable.

## 9. Documentation boundary

Use the following ownership rule when changing CV documentation:

- `docs/*/guides/cross-validation.md`: how users configure and interpret CV;
- model pages: model-specific CV targets and restrictions;
- `solver-penalty-matrix.md`: compatibility of explicit solver/loss/penalty combinations;
- inference guides: statistical meaning of inference after tuning;
- this file: implementation sequences, fast paths, caches, heuristics, and engineering validation ownership.
