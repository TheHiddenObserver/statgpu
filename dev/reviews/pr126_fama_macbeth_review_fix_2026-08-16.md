# PR #126 Fama-MacBeth review-fix follow-up — 2026-08-16

## Scope

This follow-up closes four review findings against the current PR #126 head:

1. explicit no-intercept Fama-MacBeth formulas were silently reinterpreted as intercept models;
2. Fama-MacBeth Newey-West chronology did not preserve ordered pandas categorical time labels;
3. retained rank-deficient period regressions could feed non-unique coordinate vectors into coefficient-series inference;
4. panel covariance documentation incorrectly advertised the internal `ols_covariance` dispatcher as public and described 12 physical primitive cases as public helpers.

Impact axes: **FORMULA**, **INFER**, **MATRIX/RANK**, **BACKEND**, **DOC/API**, and **ARTIFACT/VALIDATION**. No loss, penalty, solver, CV, or new covariance-family capability is introduced.

## Capability decision

The conservative contract is intentional:

- `FamaMacBeth` continues to add one period-specific intercept on the array API.
- Formula input therefore rejects explicit no-intercept specifications (`0 +` / `-1`) instead of silently changing the requested model.
- Fama-MacBeth still supports only its model-specific `nonrobust` and coefficient-series `newey-west` covariance choices; it is not routed through the Stage-C residual-OLS covariance registry.
- Every period that survives the existing observation-count filter must have full column rank under the shared panel SVD cutoff. A retained rank-deficient period raises before coefficient averaging or coordinate-level inference.
- The Stage-C minimum-norm/rank-deficient fit-space extension remains a contract for the residual-OLS estimator families covered by the Stage-C covariance matrix; it is not extended to Fama-MacBeth coefficient-series inference.

The historical Stage-C plan row that called Fama-MacBeth “unchanged” should therefore be read as **unchanged covariance-family scope**, not as a claim that every historical Fama-MacBeth edge behavior is frozen. This follow-up is correctness hardening required by accepted public formula/time metadata inputs.

## Implementation

- `statgpu/panel/_fama_macbeth.py`
  - uses `factorize_panel_metadata()` for time labels/codes, preserving ordered-categorical chronology and the shared missing-label contract;
  - rejects explicit no-intercept formula requests;
  - uses `panel_matrix_rank()` before every retained period regression and fails closed on rank deficiency;
  - retains the historical full-rank solve/fallback path and beta-series covariance formulas.
- `dev/tests/test_fama_macbeth_review_fixes.py`
  - covers both `0 +` and `-1` formula rejection;
  - compares ordered categorical `t1 -> t2 -> t10` with numeric chronology and a deliberately different lexical-order negative control;
  - exercises both array and formula/missing-row-alignment paths;
  - covers retained-period rank rejection on NumPy and Torch CPU.
- `.github/workflows/panel-stage-c-torch-cpu.yml`
  - explicitly includes the new Fama-MacBeth regression file in the maintained Torch 2.0 CPU gate.
- EN/CN Fama-MacBeth and covariance pages synchronize the strict rank/formula/time-order contracts and the actual public covariance-helper surface.

## Validation status

The source-changing follow-up invalidates any claim that the previously accepted P100 artifact is exact-head physical acceptance. The prior P100 measurements remain immutable historical evidence for the preceding numerical tree, but exact-head CuPy/Torch physical validation is required before this follow-up can be called `COMPLETE` under the Stage-C remote-full plan.

Hosted CI and the post-fix review are authoritative for the current head until that physical rerun is supplied. If hosted gates are green but exact-head physical GPU evidence is still missing, the correct review exit is `PARTIAL_REMOTE_PENDING`, not `COMPLETE`.
