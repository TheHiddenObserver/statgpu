# PR #80 Review-Fix Cycle Addendum — 2026-07-28

This addendum supersedes the `Current ... SHA-256` metadata and the unconditional
`COMPLETE` status at the top of `dev/reviews/pr80_review_fix.md` for source
changes made after its recorded boundary/workspace artifact.

## Reviewed source

- Earlier complete-review head: `1e1a139770d33e27692f6f1cf2719e0815d9ec68`
- Latest user-updated head for this follow-up: `26dc26ea38fe2abf16b7fe9fc4659049572275cc`
- Exact-source production, test, and P100 runner head: `d551039b43ab4f95026e61bf2ccb3e7a112a1450`
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
6. **[MEDIUM][PERFORMANCE] The concordance pair ceiling was not a hard cap.**
   The prior event-only batching forced a minimum batch size of one. For more
   than two million scoring rows, one event could therefore still allocate a
   comparison matrix wider than the documented two-million-entry limit in both
   ordinary and counting-process scoring. A shared two-dimensional event/row
   tile helper now guarantees `event_tile * sample_tile <= 2,000,000`, including
   the one-event, `n > 2,000,000` case. Forced tiny-tile tests verify numerical
   parity for both scoring implementations, and structural tests cover up to
   20 million rows without allocating the full comparison matrix.
7. **[MEDIUM][ARTIFACT] Final-source physical-GPU evidence was stale.**
   The earlier artifact predated the wide-workspace estimate, public boundary
   wrapper, final-refit C-index change, and concordance hard-cap correction. The
   schema-v3 refresh now hashes every affected Cox source, runner, and targeted
   regression file; records the exact clean commit and the physical pytest
   command/result; and exercises the hard-cap boundary on an actual
   2,000,001-row GPU scoring input.

## Validation

GitHub Actions run `30336940628` (run number 719) passed after the production
fixes, and run `30337146649` (run number 720) passed after adding the explicit
three-backend concordance regression:

- full CPU test tree;
- Python 3.9, 3.10, 3.11, and 3.12 regression matrices;
- static/compile contracts and complete test collection;
- documentation contracts.

Temporary write-enabled patch workflow, patch script, and trigger files were
removed before these final validation heads. The follow-up diff from
`26dc26ea38fe2abf16b7fe9fc4659049572275cc` through the tested source commit
contains only the shared concordance tile helper, its two scoring integrations,
regressions, maintained CI registration, and the expanded evidence runner.

The exact-source follow-up additionally passed the full local CPU suite
(`1425 passed, 414 skipped`), the focused PR #80/risk/penalized suite
(`205 passed, 135 skipped`), the Cox phase/CV suite (`103 passed, 10 skipped`),
documentation contracts for 122 maintained files, compilation, and diff checks.

## Physical-GPU evidence

The schema-v3 P100 artifact
`results/benchmark_frontend_sources/coxph_concordance_boundary_pr80_20260728.json`
was generated from clean detached commit
`d551039b43ab4f95026e61bf2ccb3e7a112a1450` on a Tesla P100-SXM2-16GB with
CuPy 13.6.0 and Torch 2.0.0+cu117. It records `gate_failures=[]`, source hashes
for every affected Cox implementation and test file, and a machine-readable
physical-GPU pytest result of `121 passed in 8.89s` under
`STATGPU_REQUIRE_PHYSICAL_GPU=1`.

Both GPU backends selected row streaming for the wide `n=4096`, `p=128`,
8 MiB workspace case and matched NumPy within `4.441e-15`. Public ordinary,
counting-process, and penalized all-censored scoring each returned `0.5`; the
CV final refit skipped hidden training concordance; complex prediction and
truthy-string boundaries were rejected. The actual `n=2,000,001` concordance
case used one event tile and two row tiles, with a maximum of exactly 2,000,000
pair entries per tile. The local artifact SHA-256 is
`682f6e8507d082a07393c68641079ff4df007963f4b20bccaeb28a28b5fdd536`.

## Exit status

**COMPLETE REVIEW: APPROVE FROM SOURCE/CORRECTNESS PERSPECTIVE.**

No unresolved CRITICAL, HIGH, or actionable MEDIUM finding remains in the
reviewed scope. PR merge and release remain outside this review-fix cycle.
