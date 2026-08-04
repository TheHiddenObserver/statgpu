# PR #80 Independent Code Review — 2026-08-04

> Runtime reviewed through: `ed3719ceeada36abd5e153d613c6139db7301506`  
> Hosted validation: workflow `#907` (`30862287008`)  
> Status: `PARTIAL_REMOTE_PENDING`

## Review Scope

This was a fresh independent audit of the latest PR head, not a re-use of the
previous approval conclusion. The review followed the repository's full
code-review contract and re-evaluated correctness, public API behavior,
three-backend execution, CV selection, fitted-state transactions, cache
semantics, documentation, tests, and exact-source physical evidence.

The highest-risk cross-module surfaces audited were:

- canonical `CoxPHCV` custom penalty grids, continuation, two-stage search, and
  successive halving;
- cache miss/hit behavior under permutations of the same grid;
- selected penalty and full-data final refit;
- canonical versus penalized Cox grid validation;
- CuPy and Torch candidate fitting and held-out scoring;
- public diagnostics, class introspection, and failed-refit state;
- interaction with the previously approved Group Lasso compatibility layer;
- physical artifact source manifests and exact-head promotion rules.

## Impact Matrix

| Surface | Impact | Final contract |
|---|---|---|
| Numerical coefficients | Potentially affected before the fix because positional continuation could change convergence and the selected penalty | Candidate fitting is always strongest-to-weakest; final refit uses the permutation-invariant selected penalty |
| Selected hyperparameter | Affected for unordered custom grids, especially with two-stage/halving | Equivalent grids select identically; numerically tied candidates prefer stronger regularization deterministically |
| NumPy backend | Affected orchestration and reference path | End-to-end permutation and cache-hit tests pass |
| CuPy backend | Candidate fit/scoring traversal affected | Exact-head physical gate required |
| Torch CUDA backend | Candidate fit/scoring traversal affected | Exact-head physical gate required |
| Cross-validation | Directly affected | Public result axes remain in caller order while evaluation order is explicitly reported |
| Cache | Directly affected | Cache keys use canonical numerical order; each call restores its own public order without mutating cached arrays |
| Formula | No semantic change | Existing formula and side-array alignment contracts remain applicable |
| Inference | No statistical change | Cox inference policy and covariance definitions are unchanged |
| Serialization / clone | No constructor-state change | User-supplied grid object remains constructor state; fitted diagnostics are separate |
| Public API | Validation and diagnostics strengthened | Boolean, textual, byte, complex, nested, and non-scalar grids fail before candidate work |
| Documentation | Affected | `CoxPHCV` and `PenalizedCoxPHModel` introspection now expose their actual contracts |
| Benchmarks / artifacts | Affected | Previous exact-head artifacts are stale; one final promotion suite now covers both impacted physical chains |

## Findings and Fixes

### 1. HIGH — Custom CoxPHCV grids were position-dependent

The public API accepted an arbitrary one-dimensional penalty grid, but the
candidate continuation path iterated its raw positions. Two-stage screening
sampled raw positions and defined the refinement window by positional
neighbors. The default generated grid is descending, which hid the defect.
Permuting an otherwise identical custom grid could therefore change warm
starts, candidate eligibility, refinement, convergence, and the selected
penalty.

**Fix:** custom grids are strictly validated and stably sorted from strongest to
weakest regularization before continuation, coarse screening, halving, or
refinement. The original solver and scoring implementation remains the single
numerical implementation. Candidate-axis diagnostics are copied and restored to
the caller's original grid order. `cv_results_['penalty_evaluation_order']`
records the internal order explicitly.

### 2. MEDIUM — Lossy numeric coercion accepted invalid grids

Canonical Cox CV and penalized Cox CV converted custom arrays directly to
`float64`. Values such as `True` and `"0.1"` could silently become numeric
candidates, unlike the stricter contracts elsewhere in the project.

**Fix:** a shared strict real-scalar grid validator rejects booleans, strings,
bytes, complex values, nested values, and non-scalar objects before numerical
work. The original constructor object is not rewritten.

### 3. MEDIUM — Scalar and detailed selector calls could disagree

After introducing deterministic near-tie handling, the detailed selector path
applied the stronger-regularization rule while `return_details=False` returned
the historical selector result directly.

**Fix:** custom-grid calls always obtain the diagnostic result internally,
perform one deterministic selection, and then return either the scalar or the
full remapped result according to the caller's request.

### 4. MEDIUM — Public introspection was incomplete

`PenalizedCoxPHModel` had no effective Python class docstring because a class
attribute preceded the long literal. The new custom-grid evaluation versus
presentation ordering also needed an explicit public contract.

**Fix:** public introspection now documents penalized Cox estimation-only
semantics and the `CoxPHCV` custom-grid validation, evaluation order, public
order, and diagnostic field.

### 5. MEDIUM — Physical evidence could certify only part of the change

The first new GPU runner tested cache behavior but did not hash the shared
`CVCache` implementation. In addition, this review changed
`statgpu/linear_model/penalized/__init__.py`, which is part of the previously
approved group-penalty artifact boundary. The prior physical artifact therefore
cannot certify the final exact head.

**Fix:**

1. a Cox CV inner runner exercises sorted and permuted cache misses plus a third
   permutation cache hit on both CuPy and Torch;
2. a Cox CV canonical suite hashes the cache, runtime, validation boundaries,
   all relevant tests, and the inner runner;
3. a final promotion suite runs both the existing group canonical suite and the
   new Cox CV canonical suite from one clean exact head.

## Regression Coverage

New hosted tests cover:

- stable strongest-to-weakest ordering, including duplicate penalty values;
- complete candidate-axis result remapping;
- deterministic near-tie selection;
- identical scalar and detailed selector results;
- real CPU solver invariance across independent cache misses;
- real CPU cache hits across grid permutations;
- selected penalty and full-data coefficient equality;
- strict canonical and penalized Cox grid validation before solver work;
- failed-refit cleanup after a prior successful fitted state;
- constructor object identity after invalid input;
- public class documentation contracts;
- exact-source manifest existence, uniqueness, and coverage;
- final promotion-suite composition.

## Hosted Validation

Workflow `#907` passed all jobs at runtime commit
`ed3719ceeada36abd5e153d613c6139db7301506`:

- complete CPU suite: `1857 passed, 662 skipped, 10 warnings`;
- static contracts, maintained-source/script compilation, high-signal checks,
  Cox behavior checks, and complete test collection;
- documentation contracts;
- regression matrices on Python 3.9, 3.10, 3.11, and 3.12.

A final independent pass over the review delta found no remaining locally
reproducible `CRITICAL`, `HIGH`, or actionable `MEDIUM` issue.

## Remaining Physical Gate

Run the following command from a clean checkout of the exact final PR head on a
machine where both CuPy CUDA and Torch CUDA are available:

```bash
python dev/benchmarks/benchmark_pr80_final_gpu_suite.py \
  --output results/benchmark_frontend_sources/pr80_final_gpu_suite_schema1.json
```

Promotion to `COMPLETE` requires:

- outer `source_commit` equals the final PR head;
- outer `source_clean=true` and `source_clean_after=true`;
- both child suites use the same exact commit and clean tree;
- both child-suite return codes are zero and `passed=true`;
- the group canonical suite and all five group sub-runners pass;
- the Cox CV canonical suite and its inner runner pass;
- CuPy passes every group and Cox-CV-order case;
- Torch CUDA passes every group and Cox-CV-order case;
- all inner, child, and outer `gate_failures` arrays are empty.

Until that final artifact exists, the correct formal state is
`REQUEST_CHANGES` / `PARTIAL_REMOTE_PENDING`. No additional local code fix is
currently identified.
