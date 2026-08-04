# PR #80 CoxPHCV Staged-Screening Review/Fix — 2026-08-04

> Runtime implementation validated through: `d74cfe2d75a182d11c40b3d89e867d9b9183ad2a`  
> Hosted workflow: `#914` (`30868211509`)  
> Runtime test result: `1864 passed, 662 skipped, 14 warnings`  
> Status at report creation: `PARTIAL_REMOTE_PENDING`

## Scope and active gates

This review/fix cycle followed `.claude/skills/code-review.md` in auto-fix mode. The active axes were:

- canonical `CoxPHCV` selection correctness;
- custom penalty-grid order and near-tie behavior;
- two-stage and successive-halving candidate screening;
- NumPy, CuPy, and Torch CUDA behavior;
- fold-workspace/cache lifecycle;
- scalar versus detailed selector behavior;
- concurrency and failed-call restoration;
- public diagnostics and documentation;
- exact-head physical GPU evidence.

Penalized Cox CV was checked as an adjacent capability. It supports only `cv_strategy="strict"` and does not enter the experimental staged-screening branch.

## Findings and resolution

### [CRITICAL][BUG][fixed] `statgpu/survival/_cox_cv.py` — halving could remove the stronger near-tied penalty

The historical fast-pass finalist ranking used reversed `argsort`. For equal scores, it visited larger candidate indices first. Because the canonical custom-grid boundary evaluates penalties from strongest to weakest, a tie could therefore prefer the weakest penalty and remove stronger candidates before full-precision evaluation. Final near-tie selection could not recover candidates that had already been screened out.

**Impact:** the selected regularization parameter and full-data refit coefficients could depend on preliminary numerical ties.

**Fix:** environment-controlled screening is now safety-disabled. Every requested staged run evaluates the complete grid at full solver precision. No candidate can be removed by a preliminary score.

### [HIGH][BACKEND][fixed] staged behavior was silently CuPy-specific

The experimental optimization was active only for explicit CuPy CUDA. CPU and Torch CUDA used exhaustive CV without a user-visible effective-mode contract.

**Fix:** all backends now expose one statistical contract:

- every candidate is evaluated at full precision;
- no candidate is screened out;
- final selection uses the complete grid.

Explicit CuPy runs retain the staged fold-workspace machinery, but coarse, refinement, and finalist sets are expanded to the complete grid and fast solver controls are set equal to full controls. CPU and Torch use a single exhaustive pass.

### [HIGH][TEST][fixed] the prior physical runner did not enter the staged branch

The existing penalty-order runner used only three penalties, while staged execution required at least eight. It therefore could not certify the affected path.

**Fix:** a new physical runner uses eight unsorted penalties, enables both environment switches, and checks CuPy/Torch selected-penalty, score-path, coefficient, evaluation-order, public-order, and all-candidate masks.

### [MEDIUM][API][fixed] requested versus effective mode was not observable

**Fix:** `cv_results_` now records:

- `two_stage_requested`;
- `two_stage_enabled`;
- `successive_halving_requested`;
- `successive_halving_enabled`;
- `staged_execution_mode`;
- `staged_safety_strategy`;
- `staged_fallback_reason`;
- `fast_pass_candidate_mask`;
- `full_precision_candidate_mask`;
- `screened_out_candidate_mask`.

A `RuntimeWarning` states that exhaustive full-precision CV over all candidates is being used.

### [HIGH][BUG][fixed] first safety wrapper had a concurrent-call race

The first wrapper version read staged flags before entering the lock. A concurrent call could observe the temporary disabled environment reader and later enter the raw selector after the reader was restored.

**Fix:** when a staged environment request is present, every selector invocation enters one `RLock` before reading or replacing module-level environment readers. A dedicated concurrent regression test verifies that staged calls cannot overlap inside the temporary compatibility boundary.

### [MEDIUM][PERF][fixed] first fallback removed existing fold-workspace coverage

The initial single-pass fallback made historical staged fold-cache tests fail because the cache is intentionally retained only across multiple staged passes.

**Fix:** explicit CuPy requests keep the staged fold-workspace lifecycle while expanding every candidate set to the full grid. This preserves cache-on/cache-off resource behavior without permitting screening. CPU and Torch remain single-pass exhaustive.

### [MEDIUM][API][fixed] wrapper conversion could preempt public validation

An intermediate version converted `max_iter`, `tol`, and `n_penalties` before the raw selector's established validation boundary.

**Fix:** conversions are delayed until the validated raw selector requests the relevant environment controls. Invalid public inputs retain their existing error contracts.

### [MEDIUM][DOC][fixed] fallback semantics were documented only by class introspection

**Fix:** added English and Chinese guides plus hosted documentation contracts:

- `docs/en/guides/cox-cv-staged-safety.md`;
- `docs/cn/guides/cox-cv-staged-safety.md`;
- `dev/tests/test_pr80_cox_cv_staged_safety_docs.py`.

## Changed files

Runtime and import boundary:

- `statgpu/survival/_cox_cv_staged_safety_contract.py`;
- `statgpu/survival/__init__.py`.

Hosted regression coverage:

- `dev/tests/test_pr80_cox_cv_staged_safety_contract.py`;
- `dev/tests/test_pr80_cox_cv_staged_safety_suite_contract.py`;
- `dev/tests/test_pr80_cox_cv_staged_safety_docs.py`;
- `dev/tests/test_pr80_final_gpu_suite_contract.py`.

Physical evidence:

- `dev/benchmarks/benchmark_cox_cv_staged_safety_gpu.py`;
- `dev/benchmarks/benchmark_cox_cv_staged_safety_suite.py`;
- `dev/benchmarks/benchmark_pr80_final_gpu_suite.py`.

Documentation:

- `docs/en/guides/cox-cv-staged-safety.md`;
- `docs/cn/guides/cox-cv-staged-safety.md`.

## Review/fix iterations

1. Added an explicit single-pass exhaustive fallback, diagnostics, tests, and a physical runner.
2. Independent re-review found a concurrent-call race; serialized staged requests and added a concurrency test.
3. Hosted workflow `#911` found two legacy fold-workspace assertions failing (`1862 passed, 662 skipped, 2 failed`).
4. Preserved explicit-CuPy staged workspace machinery while expanding all candidate sets and full-precision controls.
5. Independent re-review found premature public-control coercion; delayed conversion to preserve validation behavior.
6. Workflow `#914` passed all hosted gates.
7. Final documentation audit added synchronized English/Chinese user guidance and a docs contract.

## Hosted evidence

Workflow `#914` passed at runtime head `d74cfe2d75a182d11c40b3d89e867d9b9183ad2a`:

- complete CPU tree: `1864 passed, 662 skipped, 14 warnings`;
- static contracts and maintained-script compilation;
- Cox behavior checks;
- documentation contracts;
- Python 3.9, 3.10, 3.11, and 3.12 regression matrices.

A final exact-head hosted run is required after this report and the EN/CN documentation commits.

## Remaining physical GPU gate

Run from a clean checkout of the final PR head on a machine with both CuPy CUDA and Torch CUDA:

```bash
python dev/benchmarks/benchmark_pr80_final_gpu_suite.py \
  --output results/benchmark_frontend_sources/pr80_final_gpu_suite_schema2.json
```

The final promotion suite now contains three exact-head child suites:

1. Group penalty canonical suite;
2. Cox CV custom-grid order suite;
3. Cox CV staged-safety suite.

Promotion to `COMPLETE / APPROVE` requires:

- outer and all child/nested reports use the same final commit;
- source trees are clean before and after every suite;
- every return code is zero;
- every `gate_failures` array is empty;
- Group cases pass on CuPy and Torch;
- Cox custom-grid order/cache cases pass on CuPy and Torch;
- staged safety reports both switches requested and both effective screening flags false;
- every staged candidate is full precision and no candidate is screened out;
- CuPy/Torch selected penalty, score path, and final coefficients satisfy the runner tolerances.

## Exit status

No unresolved locally reproducible `CRITICAL`, `HIGH`, or actionable `MEDIUM` finding remains after the final pure audit pass. The correct state remains `PARTIAL_REMOTE_PENDING` until the final exact-head hosted run and physical GPU promotion suite pass.
