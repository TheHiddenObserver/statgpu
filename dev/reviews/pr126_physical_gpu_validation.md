# PR #126 Panel Stage C physical GPU validation

## Current physical acceptance status

**PARTIAL_REMOTE_PENDING / LOCAL FIX VALIDATED / NOT MERGE-READY**

Validation tier: local gates complete; fresh remote CUDA correctness/performance pending.

A 2026-08-12 strict re-review found that the previously accepted implementation did not use identified rank consistently for rank-deficient residual/Swamy-Arora degrees of freedom and used a unit-dependent negative-variance tolerance. Production behavior has therefore changed after the previously accepted numerical measurement `3dc7df19176f8fb881a8d37e9d75b4f75e71b058`; that P100 evidence and its canonical v2 sources remain immutable **historical** evidence, not current acceptance evidence.

## Current local fix contract

- historical full-column-rank coefficient solvers and df formulas remain algebraically unchanged;
- supported rank-deficient `PanelOLS`, `BetweenOLS`, `FirstDifferenceOLS`, and `RandomEffects` use identified fit-space rank for residual degrees of freedom;
- `RandomEffects` between/within auxiliary df are rank-aware, so exact redundant columns do not change `sigma2_e`, `sigma2_a`, theta, identified fitted values, or fit-space inference;
- strict inference rejects every strictly negative final covariance diagonal; only IEEE signed zero may be normalized to zero;
- maintained column-space invariance tests cover nonrobust and HC1/`robust`, including unbalanced RandomEffects and entity-effect PanelOLS;
- the physical correctness runner now records each estimator case's `fit_rank` and `parameter_count`.

Focused validation completed before commit:

- rank/inference/covariance/external review matrix: **86 passed**;
- strict-variance/entity-FE re-review matrix: passed;
- expanded physical-runner/Torch contract matrix: **45 passed**;
- syntax and `git diff --check`: passed.

## Fresh remote acceptance target

Fresh evidence must be measured on an exact-clean post-fix numerical head.

Correctness/backend provenance target:

- CuPy: **47/47** = 35 estimator integrations + 12 direct public primitives;
- Torch: **47/47** = 35 estimator integrations + 12 direct public primitives;
- eight dedicated rank-deficient estimator cases cover PanelOLS(entity effects), BetweenOLS, FirstDifferenceOLS, and RandomEffects under both `nonrobust` and historical HC1/`robust`;
- each of those eight cases must record `fit_rank < parameter_count`;
- requested backend must equal executed backend for every estimator/primitive;
- all prior numerical-rank-boundary primitives and `panel_rank_boundary_dk` remain required;
- no numerical CPU fallback is accepted.

Synchronized performance target:

- **58/58** maintained rows = 54 base + 4 high-T QS;
- synchronized end-to-end estimator fit;
- CuPy/Torch × PooledOLS/PanelOLS QS at `N=10,000,k=2,T=200` remains included;
- raw samples must be finite/positive and stored medians must match raw samples;
- the updated performance runner records CUDA-suffixed CuPy package distributions (`cupy-cuda11x`/`cupy-cuda12x`) when the unsuffixed distribution is absent;
- no speedup or CPU-baseline claim is made.

## Historical immutable evidence

The following remain registered exactly as historical audit evidence and must not be overwritten:

- measurement `3dc7df19...`, raw commit `be679c13...`: CuPy/Torch 39/39 and 58 performance rows;
- canonical v2 correctness source `panel-stage-c-rank-policy-validation-pr126-20260811-c67ada7ec59f`;
- canonical v2 performance source `panel-stage-c-rank-policy-performance-pr126-20260811-f27bef0b7c55`;
- earlier `ec511f53...` canonical v1 correctness/performance sources.

Any fresh post-fix artifacts require new immutable source/parser identities rather than mutating v1/v2 registrations.

## Exit rule

Current technical status remains **PARTIAL_REMOTE_PENDING / NOT MERGE-READY** until fresh exact-head P100 correctness and performance evidence is audited, promoted under new immutable identities, the promoted head passes the permanent hosted matrix, and a final strict re-review finds no unresolved CRITICAL/HIGH/relevant MEDIUM issue. PR #126 remains Draft and unmerged unless the user explicitly requests a lifecycle transition.
