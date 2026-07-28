# PR #80 Review-Fix Cycle — 2026-07-28

This report supersedes the unconditional completion statement in the earlier
PR #80 addendum for changes made after its recorded physical-GPU artifact.

## Hard exit status

**PARTIAL_REMOTE_PENDING.** The current post-closure review fixes and all local
CPU/documentation gates pass, and the user has authorized the exact-source
workflow. The physical CuPy/Torch artifact below still predates this delta;
P100 refresh, evidence write-back, push, and hosted-CI follow-up are in progress.

## Current post-closure delta

- Starting local and remote head: `fd6d7952c7f7810506395ade46953eacda98f91c`.
- Review mode remains `.claude/skills/code-review.md` `auto-fix` under
  `dev/AGENTS.md` and `.claude/workflows/new-module-dev.md`.
- `CoxPHCV.full_host_transfer_performed_` now covers host-orchestrated CV as
  well as final refit. Separate `cv_full_host_transfer_performed_`,
  `final_refit_full_host_transfer_performed_`, `orchestration_device_`, and
  invocation-specific cached provenance make the data movement auditable.
- Public `CoxPH` raises `CoxCandidateNumericalError` only when a finite-input fit
  returns non-finite fitted coefficients or likelihood. CV catches only this
  subtype, records the failed penalty/fold, and continues; input, OOM, CUDA,
  backend, and unexpected runtime exceptions keep their original type.
- CV factorizes strata once, moves train/test codes through `BackendBase` once
  per fold evaluation, and passes an internal preencoded-label carrier so each
  candidate skips `unique` and H2D label work. Candidate fits omit cluster and
  subject labels because inference and training concordance are disabled. The
  unstratified right-censored fast kernel is enabled when otherwise eligible;
  stratified candidates correctly retain the shared stratified objective rather
  than falsely claiming that the unstratified kernel supports strata.
- Canonical `CoxPH` now initializes all fitted state through `_reset_fit_state()`.
  Legacy Efron/Breslow/entry caches are owned only by the explicit test adapter,
  and the unused statsmodels convergence extractor was removed.
- Local validation for this delta: `1454 passed, 436 skipped`; documentation
  links affected `0` files; contracts passed for `122` maintained files;
  `compileall`, benchmark CLI parsing, `git diff --check`, and `pyflakes` pass.
- The maintained P100 runner is upgraded to schema 5, hashes the new exception
  and CV test sources, runs `test_cox_cv.py`, and records per-backend transfer
  provenance plus label-preparation counts. The schema-5 JSON will be generated
  from the first clean source commit in the authorized remote step.

The sections below retain the exact-source evidence for the preceding schema-4
cycle as historical baseline; they are not evidence for the uncommitted delta.

## Reviewed source and mode

- Remote and local starting head for this follow-up:
  `fadfce7d5c010a09b90b6e18aede5bb0c8c39636`.
- Review mode: `.claude/skills/code-review.md` `auto-fix`.
- Development contract: `.claude/workflows/new-module-dev.md`.
- Repository conventions: `dev/AGENTS.md`.
- Exact-source production, test, and P100 runner commit:
  `2daf5c6178562308d5e4a6fc41e24c96a0e73e25`.
- Pushed evidence commit:
  `89e4307c4015b60db375b7a49cf4d819ba4a57e7`.

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
| public fit and inference isolation from legacy methods | passed | same canonical dispatcher | same canonical dispatcher | inference-enabled fit passes while every legacy method rejects; mixin absent from public MRO and import graph |

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

### Follow-up findings and selected designs

[MEDIUM][API/PERF][fixed] `CoxPHCV` and its final `CoxPH` both owned public GPU cleanup.
Impact: one CV prediction or score could flush both allocators and synchronize
CUDA twice. Option A disables cleanup on the delegated final model and retains
the outer CV `try/finally`; option B deletes the outer boundary and delegates.
Selected: option A. It covers delegation, validation, and future CV-owned work,
while giving one class unambiguous ownership of the complete public call.
Evidence: hook counters require one outer round and zero guarded inner calls for
prediction and scoring; the physical runner repeats this check per GPU backend.

[MEDIUM][MAINT/EXT/INFER][fixed] canonical inference inherited inversion helpers
from the historical mixin. Options considered were moving only those methods to
another mixin or using stateless backend functions. Selected: stateless helpers
in `_cox_inference.py`, because they carry no estimator state, avoid MRO name
collisions, and can be reused by both implementations. `CoxPH` now inherits
only `BaseEstimator`; regression-only legacy execution uses an explicit
composition adapter. An inference-enabled isolation test makes every legacy
method fail and a fresh-process test verifies that public import does not load
`_cox_legacy` or its optional probes.

[MEDIUM][MAINT/REUSE][fixed] CPU held-out Breslow/Efron likelihood duplicated
the shared risk-set definition. Directly routing it through the pre-existing
general row/group implementation was statistically clean but exploratory
benchmarks showed an avoidable 8--23x continuous-case slowdown. Selected: move
the stable log-likelihood-only suffix implementation into `_risk_sets.py` and
dispatch it from `cox_counting_process_objective(compute_derivatives=False)`.
CV now has one statistical owner across all ties/backends while retaining a
formally owned fast path. Parity covers ordinary, tied, delayed-entry, strata,
and `1e8` common offsets.

[LOW][READ/MAINT][fixed] formula side arrays passed through a nested aligner and
then a module-level aligner. The nested implementation was removed; entry,
cluster, strata, and subject ID each pass once through the existing
backend-preserving helper. NumPy/CuPy/Torch tests verify retained rows, dtype,
backend, and device.

[LOW][DOC/API][fixed] `CoxPHCV.predict()` documented only `ndarray`. Its return
contract now explicitly lists NumPy, CuPy, and Torch native arrays.

[LOW][EVIDENCE][fixed] the first refreshed runner stored a live cleanup-counter
dictionary after evaluating its gate. `CoxPHCV.__del__` later mutated that same
object, so the JSON could report `single_cleanup_owner=true` beside a count of
two. The runner now snapshots counters at the public-call boundary. The
accepted rerun records outer CUDA/Torch hooks exactly once and inner hooks zero
times for both CuPy and Torch.

### Held-out likelihood performance comparison

The selected shared-owner fast path was compared with the exact pre-change
`_compute_partial_likelihood` at 4,096 and 16,384 rows using three-repeat
medians. Maximum absolute likelihood difference was `2.910e-11`. Ratios below
are new/previous, so lower is faster:

| Rows and scenario | Breslow | Efron |
| --- | ---: | ---: |
| 4,096 ordinary | 0.848x | 1.007x |
| 4,096 heavy ties | 1.640x | 0.750x |
| 4,096 delayed entry + strata | 1.214x | 0.801x |
| 16,384 ordinary | 1.459x | 0.976x |
| 16,384 heavy ties | 2.007x | 1.532x |
| 16,384 delayed entry + strata | 1.035x | 0.928x |

The largest ratios occur on millisecond heavy-tie cases; at the larger
delayed-entry/strata workload Breslow is within 3.5% and Efron is faster. This
supports centralizing the fast path without accepting the general-loop
regression.

## Performance and physical evidence

The schema-v4 maintained runner executed every physical case for both CuPy and
Torch, covering public and CV cleanup ownership, complex rejection, summary
metadata, `subject_id`, one-sync scoring, inference results, backend reuse,
bounded row streaming, wide-workspace routing, and public legacy isolation.

Executed through Paramiko from clean detached commit `2daf5c617856`:

```text
/root/miniconda3/envs/myconda/bin/python dev/benchmarks/benchmark_cox_boundary_gpu.py \
  --output results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260728.json \
  --run-targeted-tests
```

The resulting artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260728.json`.
It records schema 4, `source_clean=true`, 21 Git-blob-verified source hashes,
CuPy 13.6.0 and Torch 2.0.0+cu117 on Tesla P100-SXM2-16GB, `159 passed` targeted
tests, every backend case passed, and `gate_failures=[]`. Both backend CV cases
record outer CUDA/Torch cleanup once and guarded inner cleanup zero times; both
completion cases record legacy isolation. Its SHA-256 is
`aea9931b7dd00cfdccf8bdf18060244b929eb3392234c1c23f8ee2d178d9f0c5`.

## Local validation

- Complete CPU tree: `1450 passed, 436 skipped`. The first run exposed one
  parity diagnostic that still assumed legacy inheritance; after moving that
  script to the explicit composition adapter, its focused two-test rerun and
  the complete tree both passed.
- Focused Cox/PR80 matrix: `340 passed, 164 skipped` before the final diagnostic
  adapter adjustment; the reference/history matrix passed `122` with `26`
  expected optional-backend skips.
- Documentation links: zero affected files.
- Documentation contracts: 122 maintained files passed.
- `compileall`, benchmark CLI parsing, `git diff --check`, and `pyflakes` on all
  changed Python files passed.
- The earlier one-command full-tree run also reached `1438 passed, 434 skipped`
  before the shell wrapper timeout; the split runs include the new isolation test.

## Hosted CI (preceding schema-4 baseline)

GitHub Actions run
`https://github.com/TheHiddenObserver/statgpu/actions/runs/30369118924`
completed successfully for evidence commit `89e4307c4015`. The required
`docs-contracts`, `static-contracts`, `full-cpu-suite`, and Python 3.9, 3.10,
3.11, and 3.12 regression-matrix jobs all reached successful terminal states.

## Changed files

- `.github/workflows/test.yml`
- `CHANGELOG.md`
- `dev/benchmarks/benchmark_cox_boundary_gpu.py`
- `dev/benchmarks/pr79/diagnose_cox_pen.py`
- `dev/reviews/pr80_review_fix_cycle_2026-07-28.md`
- `dev/tests/test_pr80_completion_contract_followup.py`
- `docs/en/changelog.md`, `docs/cn/changelog.md`
- `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260728.json`
- `statgpu/backends/_array_ops.py`, `statgpu/backends/_utils.py`
- `statgpu/survival/__init__.py`, `_cox.py`, `_cox_cv.py`,
  `_cox_fit_adapter.py`, `_cox_inference.py`, `_cox_legacy.py`, `_cox_score.py`,
  `_risk_sets.py`

## Skipped and deferred work

- For the preceding schema-4 cycle, no physical-GPU or hosted-CI gate was
  skipped: its artifact passed all CuPy/Torch cases and 159 required tests, and
  hosted run `30369118924` passed all seven required jobs.
- For the current post-closure delta, exact-source P100 JSON, evidence commit,
  push, and hosted CI are in progress under the user's authorization.
- `inference_mode="approx"` remains the documented compatibility-only no-op;
  changing or removing that public option is outside this cycle.
