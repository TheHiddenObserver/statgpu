# PR #80 Review-Fix Cycle Addendum — 2026-07-28

This addendum supersedes the `Current ... SHA-256` metadata and the unconditional
`COMPLETE` status at the top of `dev/reviews/pr80_review_fix.md` for source
changes made after its recorded boundary/workspace artifact.

## Reviewed source

- Earlier user-updated head: `c967e6f08b976f7f1df0df63ec58efda528df438`
- Exact-source P100 evidence runner head: `b42ab0ade37e9fc7c5abf159089da195220680df`
- Latest user-updated head for the complete review: `fe7f8e72364e405dc96ed45520addabaa0440bdf`
- Final production/test head from the complete review: `1e1a139770d33e27692f6f1cf2719e0815d9ec68`
- Compatibility target: `master` at `7ccf6163d30a9078bf80107c4d16e08943a56a1e`

## Findings closed before the complete review

1. **Constructor boolean coercion.** `CoxPH` converted truthy strings such as
   `compute_cindex="False"` and `gpu_memory_cleanup="False"` with `bool(...)`,
   making them `True` before fit-time validation could reject them. Public
   `CoxPH` and `CoxPHCV` constructors now accept only actual booleans or integer
   `0`/`1` controls and reject truthy strings before constructor coercion.
2. **Wide delayed-entry workspace underestimation.** The dense Breslow/Efron
   workspace estimator counted row-scalar and `p x p` tensors but omitted the
   possible `n x p` weighted-design intermediate used by optimized three-operand
   `einsum` contraction paths. The estimate now conservatively includes two
   row-feature buffers so wide models select the row-streaming fallback before
   exceeding `STATGPU_COX_GROUP_MAX_BYTES`.
3. **Coverage gaps.** Regression gates cover constructor boundaries, signature
   preservation, the wide-model estimate, and forced row-streaming parity for
   Breslow/Efron with delayed entry, multiple failure times, multiple strata,
   score residuals, and log-likelihood-only evaluation.

## Findings closed in the complete review

1. **[MEDIUM][MEMORY] Ordinary public concordance workspace.**
   `statgpu/survival/_cox_score.py` allowed one chunk to contain 128 million
   event-by-row pair entries. Several boolean pair matrices can coexist, so a
   public `score()` call could allocate several hundred MiB even though the
   shared counting-process concordance path used a two-million-entry bound.
   The ordinary path now uses the same two-million-entry ceiling through a
   separately tested batch-size helper.
2. **[MEDIUM][API/CORRECTNESS] All-censored scoring inconsistency.** Ordinary
   right-censored `CoxPH.score()` returned the neutral C-index `0.5` for an
   all-censored scoring set, while start-stop, stratified, subject-grouped, and
   penalized-Cox scoring reached fit-oriented validation and raised
   `at least one observed event is required`. The counting-process input
   normalizer now retains event-required validation by default but allows the
   concordance-only caller to set `require_event=False`. Likelihood, fitting,
   baseline, and loss APIs still require at least one event.
3. **[MEDIUM][PERFORMANCE] Hidden final-refit C-index in `CoxPHCV`.** Fold
   candidates already disabled training concordance, but the selected full-data
   refit inherited `CoxPH(compute_cindex=True)`. This added an unrequested
   pairwise pass after cross-validation even though `CoxPHCV.score()` computes
   evaluation concordance on demand. The final estimator now explicitly uses
   `compute_cindex=False`; its public score contract is unchanged.
4. **[MEDIUM][API] Penalized-Cox truthy-string booleans.**
   `PenalizedCoxPHModel` still accepted strings for `gpu_memory_cleanup`,
   `compute_inference`, and `lla`, interpreting nonempty strings as true in
   downstream control flow. Constructor and `set_params` boundaries now accept
   only actual booleans or integer `0`/`1`, while preserving the supplied 0/1
   objects for sklearn clone identity checks. The no-intercept contract uses the
   same validation.
5. **[TEST] New complete-review gates.**
   `dev/tests/test_pr80_complete_review_cycle.py` covers the bounded ordinary
   pair workspace, neutral all-censored scoring through ordinary/counting and
   penalized APIs, NumPy/CuPy/Torch counting-concordance parametrization,
   `CoxPHCV` final-refit behavior, penalized constructor/set-parameter rejection,
   and sklearn clone compatibility. The file is part of the maintained Python
   3.9–3.12 regression matrix.

## Validation

GitHub Actions run `30336940628` (run number 719) passed after the production
fixes, and run `30337146649` (run number 720) passed after adding the explicit
three-backend concordance regression:

- full CPU test tree;
- Python 3.9, 3.10, 3.11, and 3.12 regression matrices;
- static/compile contracts and complete test collection;
- documentation contracts.

Temporary write-enabled patch workflow, patch script, and trigger files were
removed before these final validation heads. The net complete-review diff from
`fe7f8e72364e405dc96ed45520addabaa0440bdf` contains only the four production
fixes, their regression tests, the maintained CI registration, and this report.

## Physical-GPU evidence

The schema-v2 P100 artifact
`results/benchmark_frontend_sources/coxph_boundary_workspace_pr80_20260728_refresh.json`
continues to cover the unchanged Cox likelihood, score/information moments,
row-streaming workspace route, public device normalization, and constructor
boundary implementation on CuPy and Torch. It records a clean detached source,
`gate_failures=[]`, 104 passed targeted tests, and the wide `n=4096`, `p=128`
route switching from the old dense estimate to streaming with maximum NumPy
error `3.997e-15`.

The complete-review changes do not modify likelihood, Hessian, baseline, Newton,
or workspace kernels. They do change the shared file hash by adding the
concordance-only zero-event validation option. The new all-censored concordance
test is parameterized for NumPy, CuPy, and Torch; repository-hosted CPU CI runs
NumPy and explicitly skips unavailable CUDA backends. A release process that
requires every final source hash to be reproduced on physical CUDA should rerun
that targeted test, but there is no known GPU numerical defect or remaining
code-review blocker.

## Exit status

**COMPLETE REVIEW: APPROVE FROM SOURCE/CORRECTNESS PERSPECTIVE.**

No unresolved CRITICAL, HIGH, or actionable MEDIUM finding remains in the
reviewed scope. PR merge and release remain outside this review-fix cycle.
