# PR #80 Review-Fix Cycle — 2026-07-28

This report supersedes the unconditional completion statement in the earlier
PR #80 addendum for changes made after its recorded physical-GPU artifact.

## Hard exit status

**BLOCKED_NEEDS_USER_APPROVAL.** The legacy extraction passes the complete local
CPU and static gates. Its exact-source physical-GPU artifact refresh requires a
clean commit and the user-authorized remote/push workflow; no CRITICAL or HIGH
finding remains open.

## Reviewed source and mode

- Remote and local starting head:
  `f17d83fcce3c17fdb7546ac1d844e56efe66935a`.
- Review mode: `.claude/skills/code-review.md` `auto-fix`.
- Development contract: `.claude/workflows/new-module-dev.md`.
- Repository conventions: `dev/AGENTS.md`.
- Exact-source production, test, and P100 runner commit:
  `fe06a4cf1e96e0dc5e8c74de2c763bf92b5ebdb6`.

## Impact classification

| Axis | Status | Reason |
| --- | --- | --- |
| Public API | active | prediction/scoring cleanup, summary metadata, errors |
| Backend | active | shared backend helpers, dtype/device normalization, synchronization |
| Survival | active | Cox risk-set and concordance primitives |
| Inference | active | shared result container and estimator state |
| CV | active | `CoxPHCV` final-refit inference propagation |
| Formula | active | summary must preserve the fitted formula contract |
| Benchmark/performance | active | ordinary concordance synchronization changed |
| Docs/process | active | changelog categories and completion matrix |
| Loss, penalty, solver | inactive | no loss definition, penalty rule, or optimizer step changed |

## Capability decisions

| Component | Backend | CV | Inference | Formula | Benchmark |
| --- | --- | --- | --- | --- | --- |
| `CoxPH` | three-backend | non-tunable | supported | supported | required |
| `CoxPHCV` | three-backend | supported | supported after final refit | matrix-facing wrapper | required |
| counting-process concordance | three-backend | non-tunable | estimation metric | not formula-facing | required |

## Public and architecture-specific matrix

| Contract | NumPy | CuPy | Torch | Evidence |
| --- | --- | --- | --- | --- |
| public predict/score cleanup, success and failure | passed | passed on P100 | passed on P100 | `test_pr80_completion_contract_followup.py` |
| truthful matrix/formula summary | passed | backend-neutral | backend-neutral | matrix plus `Surv(start, stop, event)`, categorical interaction, strata |
| `ParameterInferenceResult` state | passed | passed on P100 | passed on P100 | direct Cox and CV final refit |
| fractional/non-finite/overflow `subject_id` rejection | passed | passed on P100 | passed on P100 | low-level concordance tests |
| representable host `uint64` codes | passed | passed on P100 | passed on P100 | low-level concordance tests |
| one post-loop C-index scalar synchronization | passed | passed on P100 | passed on P100 | forced 1-by-2 tile counter |
| public fit isolation from legacy methods | passed | same canonical dispatcher | same canonical dispatcher | legacy methods inherited unchanged from private mixin and monkeypatched to fail |

CuPy and Torch are unavailable on the local Windows runner. Their active tests
were therefore executed through the maintained Paramiko P100 runner from the
clean detached exact-source commit recorded above.

## Objective scaling, precision, convergence, and formula

- The Cox partial log-likelihood remains the sum of event contributions from
  the shared counting-process objective. This review does not change its
  statistical definition.
- Ridge fitting remains
  `log_partial_likelihood - penalty * ||beta||^2`, with score adjustment
  `-2 * penalty * beta`. There is no new sample-count normalization or external
  penalty remapping in this cycle.
- Formula construction remains Patsy-based with the Cox intercept removed.
  The added summary regression preserves the exact
  `Surv(start, stop, event)` formula string, categorical term, interaction,
  strata flag, and counting-process flag.
- Existing coefficient, objective, information, KKT, R-alignment, and
  convergence gates remain green in the complete local suite. New inference
  tests require finite parameters, standard errors, z statistics, p-values,
  confidence intervals, and shared-result parity.

## Findings

[MEDIUM][API/PERF][fixed] statgpu/survival/_cox.py:35 - GPU cleanup did not cover public Cox prediction and scoring.
Impact: allocator pools could retain large temporary tiles during the estimator lifetime.
Fix: a shared `try/finally` decorator now covers hazard, risk, survival, and score methods; `predict()` delegates to the decorated hazard method.
Evidence: success and exception-path mock tests at `dev/tests/test_pr80_completion_contract_followup.py:43` and `:63`.

[MEDIUM][BUG/API][fixed] statgpu/survival/_cox.py:5438 - `summary()` printed a synthetic fixed R call.
Impact: matrix, start-stop, strata, subject, and formula fits produced misleading reproduction metadata.
Fix: fit stores structured call metadata and summary renders only the actual interface and supported flags.
Evidence: matrix and formula summary tests at `dev/tests/test_pr80_completion_contract_followup.py:83` and `:108`.

[MEDIUM][BACKEND/MAINT][fixed] statgpu/survival/_risk_sets.py:35 - survival duplicated generic backend primitives.
Impact: device, dtype, scalar, and array rules could drift from the shared backend layer.
Fix: risk-set normalization now reuses shared backend resolution, namespace, scalar, zeros, eye, cast, and integer-code helpers; canonical fit uses one `BackendBase` object.
Evidence: complete local suite, three-backend parameterization, and source guard against direct CuPy/Torch imports in the canonical dispatcher.

[MEDIUM][INFER/MAINT][fixed] statgpu/survival/_cox.py:1474 - canonical inference did not publish the shared result contract.
Impact: `_params`, `_tvalues`, serialization, and final-CV-refit state diverged from other inferential estimators.
Fix: Cox constructs and applies `ParameterInferenceResult`, uses `norm.ppf(0.975)`, and CV copies the shared result and public inference arrays.
Evidence: direct and CV inference tests at `dev/tests/test_pr80_completion_contract_followup.py:138` and `:167`.

[MEDIUM][BUG/INTERNAL][fixed] statgpu/survival/_risk_sets.py:2112 - low-level concordance silently truncated fractional `subject_id` values.
Impact: distinct subjects could be merged and valid comparison pairs incorrectly removed.
Fix: shared integer-code normalization rejects complex, nonnumeric, fractional, non-finite, and out-of-int64 inputs before conversion while accepting safe host `uint64` values.
Evidence: three-backend cases at `dev/tests/test_pr80_completion_contract_followup.py:217` and `:238`.

[MEDIUM][MAINT/EXT][fixed] statgpu/survival/_cox.py:121 - canonical and legacy Cox reference implementations coexisted.
Impact: the 5,500-line module remains difficult to extend and inactive code can attract misplaced fixes.
Fix: the public estimator and canonical dispatcher remain in `_cox.py`; the
historical CPU, CuPy, and Torch implementations moved mechanically to the
private `_LegacyCoxReferenceMixin` in `_cox_legacy.py`. Existing private
regression entry points are inherited unchanged, without import-time method
replacement.
Evidence: the structural regression verifies the mixin MRO, method identity,
origin module, absence of legacy fit definitions from the canonical source,
and canonical dispatcher ownership. The complete CPU tree passes.

[MEDIUM][DOC/PROCESS][fixed] dev/reviews/pr80_review_fix_cycle_2026-07-28.md:1 - the completion report lacked required workflow decisions and used invalid changelog categories.
Impact: prior approval wording did not demonstrate the active capability and validation matrix.
Fix: this report records impact, capability, backend, CV, inference, formula, scaling, performance, review, skipped work, and hard status; bilingual changelogs use only Fixed/Optimized/Validation categories for this cycle.
Evidence: documentation contracts pass for 122 maintained files.

[LOW][PERF][fixed] statgpu/survival/_cox_score.py:181 - ordinary concordance synchronized three scalars per tile.
Impact: many-event GPU scoring could become synchronization-bound.
Fix: all three counts remain backend-native through the tile loop and `_sync_scalars` performs one stacked device-to-host transfer after the loop.
Evidence: forced tiny tiles preserve the exact result and record one three-value synchronization at `dev/tests/test_pr80_completion_contract_followup.py:261`.

## Performance and physical evidence

No new comparative timing claim is made. The performance contract is
structural: one ordinary-concordance host synchronization per score call. The
schema-v4 maintained runner executed the physical `completion_contract` case
for both CuPy and Torch, covering cleanup, complex rejection, summary metadata,
`subject_id`, one-sync scoring, inference results, backend reuse, and absence of
the import-time adapter.

Executed from clean detached commit `fe06a4cf1e96`:

```text
/root/miniconda3/envs/myconda/bin/python dev/benchmarks/benchmark_cox_boundary_gpu.py \
  --output results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260728.json \
  --run-targeted-tests
```

The resulting artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260728.json`.
It records schema 4, `source_clean=true`, 19 Git-blob-verified source hashes,
CuPy 13.6.0 and Torch 2.0.0+cu117 on Tesla P100-SXM2-16GB, `154 passed` targeted
tests, every backend case passed, and `gate_failures=[]`. Its SHA-256 is
`823df8aff42bb238ae3e3575207e535da7da3c403ac23b16beb16d21243c09bb`.

## Local validation

- Complete CPU tree, split only to stay inside the command time limit:
  `900 passed, 235 skipped` plus `539 passed, 199 skipped`; aggregate
  `1439 passed, 434 skipped`.
- Focused Cox/PR80 matrix: `324 passed, 162 skipped`.
- Documentation links: zero affected files.
- Documentation contracts: 122 maintained files passed.
- `compileall`, benchmark CLI parsing, `git diff --check`, and `pyflakes` on all
  changed Python files passed.
- The earlier one-command full-tree run also reached `1438 passed, 434 skipped`
  before the shell wrapper timeout; the two split runs provide clean exit codes.

## Changed files

- `.github/workflows/test.yml`
- `CHANGELOG.md`
- `dev/benchmarks/benchmark_cox_boundary_gpu.py`
- `dev/reviews/pr80_review_fix_cycle_2026-07-28.md`
- `dev/tests/test_pr80_completion_contract_followup.py`
- `docs/en/changelog.md`, `docs/cn/changelog.md`
- `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260728.json`
- `statgpu/backends/_array_ops.py`, `statgpu/backends/_utils.py`
- `statgpu/survival/__init__.py`, `_cox.py`, `_cox_cv.py`,
  `_cox_fit_adapter.py`, `_cox_legacy.py`, `_cox_score.py`, `_risk_sets.py`

## Skipped and deferred work

- The committed schema-v4 P100 artifact predates the mechanical legacy split.
  The maintained runner now hashes `_cox_legacy.py` and checks mixin isolation;
  refresh it from the clean source commit before restoring `remote-full`.
- `inference_mode="approx"` remains the documented compatibility-only no-op;
  changing or removing that public option is outside this cycle.
