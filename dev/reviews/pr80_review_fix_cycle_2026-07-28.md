# PR #80 Review-Fix Cycle — through 2026-07-29

This report supersedes the unconditional completion statement in the earlier
PR #80 addendum for changes made after its recorded physical-GPU artifact.

## Current hard exit status

**COMPLETE.** The post-schema-6 prepared-state, packed-target, constructor, and
backend-reuse fixes pass the complete local suite and the exact-source
schema-7 P100 refresh. The read-only re-review found no remaining CRITICAL,
HIGH, or active MEDIUM issue. The machine-readable artifact has been
independently verified against all 29 Git blobs from source commit
`28d7b367c364d4b64e24b6b36662724b6b1c9a86`; evidence commit
`98d8ef3b762e7acd95049320d614bcc09721b454` was pushed, and all seven hosted
jobs passed in run `30422694780`.

## 2026-07-29 impact classification

| Axis | Status | Reason |
| --- | --- | --- |
| Backend | active, three-backend | packed-target provenance and shared prediction preparation |
| Performance | active | safe prepared-state validation must retain fold reuse |
| CV/cache | active | packed target unpacking and reusable loss capability |
| Public API | active boundary | clone-safe constructor round trips |
| Inference | unchanged/shared | canonical inference remains in `_cox_inference.py` |
| Formula | active maintenance | side-array alignment now uses `BackendBase` |
| Benchmark/artifact | physical passed | schema-7 exact-source CuPy/Torch artifact verified |
| Documentation | active | bilingual numerical and provenance contracts |

## Post-schema-6 independent findings and fixes

| Finding | Status | Resolution |
| --- | --- | --- |
| Same-shape prepared state could fit stale data and a current baseline | fixed; physical GPU passed | The low-level solver now verifies current centered/sorted `X`, time, and event against the cached state on the active backend. Same-shape foreign content and in-place mutation are rejected before optimization; only one boolean scalar is transferred. Preparation helpers and the solver were removed from module `__all__`. |
| Packed CuPy/Torch `CoxPHCV` target lost provenance before selection | fixed; physical GPU passed | `_unpack_survival_target()` preserves native column views, so selector input backends and full-host-transfer fields see the original device residency. |
| `CoxPH.__init__()` clone-safety comment contradicted canonicalization | fixed | Cox-specific public constructor objects are stored unchanged and normalized by the existing fit-time boundary, matching `CoxPHCV`. |
| Penalized Cox and formula alignment duplicated backend branches | fixed; physical GPU passed | Penalized prediction/score now use `BaseEstimator._get_backend()`, `BackendBase.asarray()/to_numpy()`, and shared validators. Formula side-array indices use the backend factory rather than direct CuPy/Torch imports. |

The exact content check was selected over identity-only or lossy aggregate
fingerprints because either alternative can silently accept legal in-place
mutation or a checksum collision. The check performs one backend scan per
reused solver invocation but no full design-matrix D2H. At `n=4096`, `p=12` on
the local NumPy path it measured 0.68 ms versus 69.86 ms for complete Efron
preparation (0.97%), so it preserves the material preprocessing reuse.

Current-delta files are `statgpu/backends/_utils.py`,
`statgpu/survival/_cox.py`, `_cox_counting.py`, `_cox_cv.py`,
`statgpu/linear_model/penalized/_penalized_cox.py`,
`dev/tests/test_pr80_constructor_boundaries.py`,
`dev/tests/test_pr80_target_transfer_overflow_cache.py`, the schema-7 physical
runner, and the three changelog/review documents.

## 2026-07-29 findings and fixes

| Finding | Status | Resolution |
| --- | --- | --- |
| Ordinary GPU fast-path target D2H and per-penalty rebuild | fixed; physical GPU passed | Added reusable immutable right-censored loss state once per valid fold for the complete selector invocation, direct target-vector transfer counters, and truthful public/CV provenance. |
| Canonical/penalized hazard-ratio overflow mismatch | fixed; physical GPU passed | Added one strict NumPy/CuPy/Torch exp boundary. Fit raises public `CoxFitNumericalError`; prediction raises `FloatingPointError`; raw log-risk remains available. |
| Cache-hit diagnostics retained old invocation work | fixed | Added cache-hit, origin device, requested device, and `*_this_call` fields; cache hits report zero preparation/target transfers without rewriting selection origin. |
| Duplicate Cox fitted-state initialization and ambiguous exception status | fixed | Removed `_fit_impl()` reset and the contradictory history sentinel; renamed and exported public `CoxFitNumericalError` from both API levels. |
| GPU entry/strata/subject vectors were copied to host without complete provenance | fixed; physical GPU passed | Counted every retained full side-vector transfer, stopped copying a synthetic zero start vector, and updated grouped CV/refit expectations. |
| Staged CV could prepare a fold after every requested penalty was already evaluated | fixed | Filters pending penalty indices before backend/loss preparation, so an empty staged overlap performs no transfer or metadata work. |
| Staged/halving passes rebuilt non-empty fold state | fixed | Lifted backend arrays and right-censored metadata into one selector-level fold cache; later full-precision passes reuse the exact prepared state. |
| Selector-level fold reuse could retain unbounded multi-fold GPU state | fixed | Enabled cross-stage retention only below an explicit 512 MiB estimated workspace gate; larger workloads use the counted stage-local fallback. |
| Unused GPU cluster/scoring unique labels crossed to host | fixed; physical GPU passed | Label encoding now materializes host labels only for fitted strata prediction mapping; cluster and scoring paths retain only backend-native inverse codes. |
| Public dispatch and solver repeated counting-input normalization | fixed | Public dispatch marks its validated arrays as prepared; direct solver calls retain validation, while public/CV candidates avoid the second scalar-sync round. |
| Penalized raw-risk prediction could cast complex input before validation | fixed; physical GPU passed | Added a pre-cast real-valued guard and three-backend regression coverage, so `predict_risk_score()` cannot silently discard an imaginary component. |
| Backend exp and summary inverse-HR edges were not fully covered by theoretical range checks | fixed; physical GPU passed | The shared boundary now validates the actual exp result as finite and positive, promotes inputs to float64, and applies the same strict rule to inverse hazard ratios and confidence intervals. |
| Ordinary survival prediction discarded its centered log-baseline state | fixed; physical GPU passed | Preserved ordinary baseline reference/centered-log fields without changing the historical `_baseline_by_stratum is None` contract; extreme finite log-risk no longer re-enters direct `exp(Xβ)`. |

## Selected designs and tradeoffs

- For CV preprocessing, a fully backend-native failure-group builder would
  remove the remaining once-per-fold target D2H, but it would replace audited
  loss metadata logic on all three backends. This cycle instead reuses the
  existing numerically validated sorted loss state across penalties, reducing
  ordinary CV from `folds * penalties` preprocessing passes to one per valid
  fold for the entire selector invocation, including staged/halving passes,
  when the estimated retained state fits a 512 MiB workspace gate. Larger
  cases repeat the stage-local preparation rather than risk a multi-fold OOM;
  diagnostics report the route and remaining transfers. This is the
  lower-risk correctness/performance choice; backend-native grouping remains a
  possible later optimization.
- Hazard-ratio clipping would avoid an exception but would silently change the
  reported statistical quantity and preserve the previous canonical/penalized
  mismatch. Strict range errors were selected. `predict_risk_score()` is the
  lossless API for extreme finite log-risk.
- The fit-specific numerical failure is a supported user-visible CV contract,
  so it is public as `CoxFitNumericalError` rather than described as internal.
  CV catches only that subtype; OOM, CUDA, input, and programming failures keep
  propagating.

## Current local evidence

- Complete CPU tree: **1486 passed, 439 skipped**, 0 failed.
- Maintained target list: **226 passed, 56 skipped**, 0 failed.
- New focused regressions: **46 passed, 3 skipped** locally; the skips are the
  physical CuPy and Torch CUDA cases.
- Documentation links affected 0 files; documentation contracts passed for
  122 maintained files.
- `py_compile`, `pyflakes`, benchmark `--help`, and `git diff --check` pass.
  Ruff is not installed in the local Windows environment; the hosted static
  workflow now includes `_numeric.py` and the new regression file.
- Starting source head: `cb1b60c383021b5fec7dd067d21fa2245d96ebca`.
- Base of the current local follow-up:
  `0ca376e314afed22c66f7f37a9ff94b82a42de91`.
- Frozen schema-7 source commit:
  `28d7b367c364d4b64e24b6b36662724b6b1c9a86`.
- Schema-7 evidence commit:
  `98d8ef3b762e7acd95049320d614bcc09721b454`.
- Hosted CI for the source plus evidence:
  `https://github.com/TheHiddenObserver/statgpu/actions/runs/30422694780`;
  **7/7 jobs passed**.
- Frozen schema-6 source commit:
  `e26c21e2d1ed373fb0fd2d40169c99a31abdc82d`.

## Exact-source physical evidence (schema 7)

Schema 7 adds direct per-backend gates for the two post-schema-6 MEDIUM
findings. It rejects same-shape GPU data that do not match a reusable prepared
right-censored state, and fits a NumPy design with a packed CuPy/Torch target
to verify that native target slicing preserves device provenance until the
intentional CV orchestration transfer. The maintained physical pytest matrix
also covers shared backend prediction/scoring, constructor round trips, and
the existing completion contract.

- Exact clean source commit:
  `28d7b367c364d4b64e24b6b36662724b6b1c9a86`.
- Paramiko remote worktree:
  `/root/statgpu-pr80-28d7b36-schema7-20260729-1228`.
- Environment: Python 3.9.16, NumPy 1.24.2, CuPy 13.6.0, Torch
  2.0.0+cu117, Tesla P100-SXM2-16GB.
- Command: `/root/miniconda3/envs/myconda/bin/python
  dev/benchmarks/benchmark_cox_boundary_gpu.py --output
  results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260729_schema7.json
  --run-targeted-tests`.
- Targeted physical matrix: **282 passed**, 5 expected convergence warnings,
  0 failed in 16.95 seconds. All 18 CuPy/Torch case gates passed and
  `gate_failures=[]`.
- Both backends rejected prepared-state content mismatch. The packed-target
  cases recorded `cupy` or `torch-device` in `input_backends` and set both
  `cv_full_host_transfer_performed_` and `full_host_transfer_performed_` true.
- Independent local verification matched all 29 recorded source SHA-256 values
  to the exact commit's Git blobs and confirmed `source_clean=true`.
- Artifact:
  `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260729_schema7.json`;
  SHA-256
  `bb125ff584e9275ff2a31d197d02ca779d6bfd9f19148e8931024863a0c20d02`.

## Exact-source physical evidence (schema 6)

Schema 6 directly instruments `statgpu.losses._cox_ph._to_numpy`, adds an
ordinary unstratified CuPy/Torch CV case, checks one preprocessing pass per fold
across the complete selector instead of per candidate/stage when its bounded
cache is active, records target-vector copy shapes and counts, validates
CPU fitting from device-resident input provenance, and checks strict canonical
plus penalized hazard-ratio overflow, stable ordinary survival, and raw
log-risk preservation. Its source
hash manifest includes the production modules, workflow, runner, and affected
tests.

- Exact clean source commit:
  `e26c21e2d1ed373fb0fd2d40169c99a31abdc82d`.
- Paramiko remote worktree:
  `/root/statgpu-pr80-e26c21e-20260729T0129Z`.
- Environment: Python 3.9.16, NumPy 1.24.2, CuPy 13.6.0, Torch
  2.0.0+cu117, Tesla P100-SXM2-16GB.
- Command: `/root/miniconda3/envs/myconda/bin/python
  dev/benchmarks/benchmark_cox_boundary_gpu.py --output
  results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260729_schema6.json
  --run-targeted-tests`.
- Targeted physical matrix: **270 passed**, 5 expected convergence warnings,
  0 failed in 16.28 seconds. All 16 CuPy/Torch case gates passed and
  `gate_failures=[]`.
- Independent local verification matched all 29 recorded source SHA-256 values
  to the exact commit's Git blobs and confirmed `source_clean=true`.
- Artifact:
  `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260729_schema6.json`;
  SHA-256
  `2d07b7b2db98b709a4ed27c7690ef2a7c7d6d5397fdfaee43c57be5fc6d17cdf`.
- For both GPU backends, ordinary CV records two fold preparations rather than
  one rebuild per candidate/stage, exact target-vector copy shapes, and
  truthful full-transfer provenance. The public boundary rejects complex
  prediction input, clears failed-refit state, and preserves extreme survival
  in the log domain. Canonical and penalized hazard-ratio APIs both reject
  log-risk `-800` and `800`, while raw log-risk remains available.
- The wide `n=4096`, `p=128`, 8 MiB workspace case proves the refreshed route:
  the old 1,056,768-byte estimate would select dense, the corrected
  9,445,376-byte estimate selects streaming, and streaming was observed with
  maximum objective/derivative differences below `5.4e-15`. The forced 4 KiB
  single-group `n=8192`, `p=3` case also passed on both backends.

## Prior schema-5 closure delta

- Starting local and remote head: `fd6d7952c7f7810506395ade46953eacda98f91c`.
- Review mode remains `.claude/skills/code-review.md` `auto-fix` under
  `dev/AGENTS.md` and `.claude/workflows/new-module-dev.md`.
- `CoxPHCV.full_host_transfer_performed_` now covers host-orchestrated CV as
  well as final refit. Separate `cv_full_host_transfer_performed_`,
  `final_refit_full_host_transfer_performed_`, `orchestration_device_`, and
  invocation-specific cached provenance make the data movement auditable.
- Public `CoxPH` raises `CoxFitNumericalError` only when a finite-input fit
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
  provenance plus label-preparation counts. The exact-source schema-5 refresh
  passed and is recorded below; evidence commit, push, and hosted CI remain.

The next section records the prior schema-5 evidence. Later schema-4 sections
remain as the exact historical baseline for the preceding cycle.

## Prior schema-5 physical-GPU evidence

- Exact clean source commit:
  `7b4a33820b4acd80313df0c97a0127c18d219e3c`.
- Pushed evidence commit:
  `eef4010db37925e64f6422ffda93e074785e8d47`.
- Paramiko remote worktree:
  `/root/statgpu-pr80-7b4a338-20260728T185325Z`.
- Environment: Python 3.9.16, NumPy 1.24.2, CuPy 13.6.0, Torch
  2.0.0+cu117, Tesla P100-SXM2-16GB.
- Command:
  `/root/miniconda3/envs/myconda/bin/python dev/benchmarks/benchmark_cox_boundary_gpu.py --output results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260729.json --run-targeted-tests`.
- Targeted physical matrix: `225 passed, 1 warning in 16.06s`; every CuPy and
  Torch case passed and `gate_failures=[]`.
- The artifact contains 23 source SHA-256 values. Independent local comparison
  with the exact commit's Git blobs found zero mismatches; `source_clean=true`.
- Artifact SHA-256:
  `e66abaa3782c15e50218bd83be5200d4f8f2db1806e8a17fec9b8349895d90c5`.
- Both GPU backends report truthful full-transfer provenance: CV `true`, final
  refit `false`, whole fit `true`, orchestration `cpu`. Each reports two fold
  preparations, preencoded strata, no candidate cluster/subject use, one outer
  cleanup round, zero inner cleanup rounds, and finite coefficients.

## Current hosted CI

GitHub Actions run
`https://github.com/TheHiddenObserver/statgpu/actions/runs/30414822901`
completed successfully for schema-6 evidence commit `0ca376e314af`. The required
`docs-contracts`, `static-contracts`, `full-cpu-suite`, and Python 3.9, 3.10,
3.11, and 3.12 regression-matrix jobs all reached successful terminal states.
This run predates the post-schema-6 local fixes recorded at the top of this
report and therefore does not replace their required hosted rerun.

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
- For the prior schema-5 post-closure delta, the exact-source P100 JSON,
  evidence commit, push, and all seven hosted-CI jobs pass. The schema-6 delta
  also has exact-source CuPy/Torch P100 evidence, its evidence commit is pushed,
  and all seven hosted jobs pass. The later prepared-state and packed-target
  fixes now have exact-source schema-7 P100 evidence, the evidence commit is
  pushed, and all seven jobs in hosted run `30422694780` pass.
- `inference_mode="approx"` remains the documented compatibility-only no-op;
  changing or removing that public option is outside this cycle.
