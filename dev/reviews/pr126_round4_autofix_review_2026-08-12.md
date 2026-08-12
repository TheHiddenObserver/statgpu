# PR #126 round-4 auto-fix review — 2026-08-12

Review standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Checkpoint status

`LOCAL_REVIEW_CLEAN / HOSTED_FINAL_PENDING / NOT MERGE-READY`

Technical candidate before this review-record-only commit:

`8ce6f80a1292265b92d5cf53b11f27b652291432`

This record is intentionally the only branch delta after the technical candidate. After the exact checkpoint hosted workflows complete, live PR metadata may record the resulting lifecycle state without mutating the branch and creating a CI/status-record recursion.

## Active review axes

- correctness / public estimator behavior
- inference
- NumPy / CuPy / Torch backend behavior
- formula and panel metadata alignment
- benchmark / performance
- docs / artifacts

Inactive: loss, penalty, generic solver, CV.

## Findings closed in this review-fix loop

1. **CRITICAL / BUG — two-way FE alternating projection could stop when only `y` had converged.**
   - Convergence now checks both outcome and every design column.
   - The criterion is scale-equivariant using original-variable reference scales.
   - One backend scalar reduction is copied to host per iteration.
   - `max_iter` exhaustion fails closed instead of returning an unconverged transformed design.
   - Adversarial tests cover `y` already in the FE-orthogonal subspace while `X` still needs iterations, and response/regressor unit rescaling.

2. **HIGH / INFER — exact rank-deficient fits published ordinary coordinate-wise inference.**
   - Supported rank-deficient fits retain Moore-Penrose minimum-norm coefficients, prediction, residuals, fit statistics, rank-aware df, variance components/theta where applicable, and auditable fit-space covariance.
   - Ordinary coordinate-wise `bse_`, `tvalues_`, `pvalues_`, and `conf_int_` are unavailable because the original coefficient coordinates are not uniquely identified.
   - `_inference_result` records `applicable=False` with an auditable reason; `summary()` fails closed.
   - Full-column-rank inference behavior remains unchanged.

3. **HIGH / BACKEND — `PanelOLS.predict()` silently executed linear prediction in NumPy.**
   - Linear prediction now executes on the estimator-selected backend and returns the historical NumPy-visible output.
   - Entity/time effect lookup remains an explicit CPU metadata boundary; only compact effect values are transferred to the numerical backend.
   - Historical omitted-explicit-constant prediction compatibility is preserved.
   - Fresh physical correctness evidence must persist `prediction_backend` on a representative PanelOLS case.

4. **HIGH / BACKEND — re-review found the same CPU-only prediction defect in `RandomEffects.predict()`.**
   - RandomEffects now uses the same backend-native linear prediction helper and preserves omitted-explicit-constant compatibility.
   - Fresh physical correctness evidence must persist `prediction_backend` on a representative RandomEffects case.

5. **HIGH / FORMULA-DATA — `FirstDifferenceOLS` silently differenced duplicate `(entity,time)` rows.**
   - Duplicate panel cells now fail explicitly when `time_ids` are supplied.
   - The retained time contract is consecutive-observed differencing: calendar gaps are allowed, not filled, and not divided by gap length.
   - Ordered categorical chronology remains preserved.

6. **MEDIUM / PERF — joint two-way FE convergence initially added two GPU scalar synchronizations per iteration.**
   - The final implementation keeps both relative changes on backend and performs one combined scalar host reduction per iteration.

7. **MEDIUM / API — first backend-native PanelOLS prediction draft dropped omitted-constant compatibility.**
   - Restored without returning to host-side linear algebra.

8. **MEDIUM / DOC — durable EN/CN model pages and reviewed plan were stale about rank-deficient coefficient inference and physical lifecycle state.**
   - Long-lived docs now distinguish identified fit-space quantities from unavailable coordinate-wise inference and describe FirstDifference time semantics.
   - PR lifecycle state stays in review records / PR metadata.
   - The reviewed plan is synchronized with the final statistical/backend/remote-evidence contract.

9. **MEDIUM / ARTIFACT — fresh correctness evidence semantics changed while the runner still emitted schema v1.**
   - Fresh correctness runner now emits schema v2; historical schema-v1 evidence remains immutable.

10. **MEDIUM / PERF-MATRIX — the physical timing matrix did not observe the newly changed iterative two-way FE path.**
    - Fresh performance runner now emits schema v3 and adds an incomplete/unbalanced `PanelOLS(entity_effects=True,time_effects=True,cov_type='nonrobust')` case at `N=10,000`, `k=2`, `T=20` for both GPU backends.
    - Fresh performance target is 60 synchronized rows: 54 historical base rows + 4 bounded high-T QS rows + 2 two-way-unbalanced rows.

11. **HIGH / TEST-CONTRACT — the first exact hosted checkpoint exposed one stale legacy rank-deficiency test.**
    - `dev/tests/test_pr79_final_review_fixes.py::test_pooled_rank_deficiency_uses_effective_rank_for_df` still required statsmodels-style coordinate BSE on an exact-collinear `PooledOLS` design.
    - The test now keeps its effective-rank/df assertions, compares the identifiable fitted-space covariance `X V X'`, and requires explicit coefficient-inference unavailability.
    - The targeted failing contract plus the current rank-deficient/inference matrix passed before the fix was committed.

## Validation already completed before this checkpoint

The fix loop used fail-before-commit helper workflows. Relevant completed gates include:

- initial source fix matrix: 152/155 passed; the three failures were stale maintained tests that still required coordinate-wise rank-deficient inference; no production commit was made from that failing run;
- updated source/inference/formula/backend matrix: **155/155 passed**;
- scale-equivariant convergence + prediction backward-compatibility re-review matrix: passed;
- final single-sync / bilingual docs / physical-contract refinement matrix: **155 tests passed**, plus compile/static/docs contracts and `git diff --check`;
- fresh remote-runner contract matrix: **92/92 passed**, plus compile/static checks and `git diff --check`;
- RandomEffects prediction-backend focused matrix: passed with static checks;
- reviewed-plan synchronization contract: passed;
- first hosted checkpoint `1a5f7138...`: all regression matrices, static/docs contracts, Stage-B external alignment, Torch Stage-C workflow, maintenance/release gates reached green, but the complete CPU suite found the one stale PR79 BSE expectation above (`1 failed, 2535 passed, 737 skipped`);
- hosted-failure follow-up: the exact failing PR79 test plus current rank-deficient/inference matrix and static checks passed before committing technical candidate `8ce6f80a...`.

Final read-only review of the updated technical candidate found no unresolved CRITICAL, HIGH, or relevant MEDIUM finding.

## Fresh remote acceptance contract

Previous exact-clean P100 evidence from `f154647665788df2570439a1cc154a43f509aa45` and its immutable v3 parser/source lineage remain historical evidence only because production numerical behavior changed again.

Fresh exact-head evidence must use new immutable v4 source/parser identities and must not overwrite v1/v2/v3 registrations.

### Correctness — schema v2

Per requested GPU backend:

- **35 estimator integrations + 12 public covariance primitives = 47/47 checks**;
- requested fit backend must equal persisted executed backend;
- every exact rank-deficient estimator case, including the rank-boundary case when applicable, must have `fit_rank < parameter_count`, `coefficient_inference_applicable=false`, and an explicit rank-deficiency reason;
- representative `panel_entity_hc0` and `random_effects_explicit_constant_hc0` predictions must persist `prediction_backend == requested_backend`;
- no numerical CPU fallback;
- all numerical comparison fields that are applicable must be finite and within the maintained tolerance contract.

### Performance — schema v3

- **60/60 synchronized end-to-end fit rows**;
- 54 base rows + 4 bounded `N=10,000, k=2, T=200` QS rows + 2 `panel_two_way_nonrobust` / `two_way_unbalanced` rows at `N=10,000, k=2, T=20`;
- three finite positive raw samples per row under the default runner, with the stored median exactly equal to the sample median;
- requested backend must equal persisted executed fit backend;
- CUDA-specific CuPy package provenance must be recorded when applicable;
- no CPU speedup claim.

## Hosted and lifecycle boundary

The seven permanent workflows must complete successfully on the exact review-record checkpoint before the branch can be classified `PARTIAL_REMOTE_PENDING / LOCAL REVIEW CLEAN`.

Even after hosted and fresh physical acceptance, this PR remains Draft unless the user explicitly requests Ready-for-review. No merge is authorized by this review record.
