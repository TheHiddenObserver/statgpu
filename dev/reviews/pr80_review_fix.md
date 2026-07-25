# PR #80 Review-Fix Report

> Review date: 2026-07-25<br>
> PR head reviewed: `d6f798c1834fd6318c8257eed334f84a198fa8ad`<br>
> Original merge base: `a4879fb` (0.2.1 line)<br>
> Compatibility target: `origin/master` at `7ccf616` (0.2.2 line)<br>
> Status: `COMPLETE`

## Review Contract

This review follows `dev/AGENTS.md` and the complete
`.claude/skills/code-review.md` workflow: inspect before editing, assess impact
across correctness/API/backend/inference/performance/docs/tests, fix all relevant
critical/high/medium findings, rerun affected gates, and re-review the resulting
diff. No PR merge, release, commit, or push is part of this report.

## Compatibility Result

PR #80 was developed from the 0.2.1 tree. A real three-way merge with the 0.2.2
`master` was reviewed instead of resolving whole files from either side. The
result preserves the 0.2.2 version declarations and the PR #79 final-KKT,
inference-provenance, strict-inference, documentation, and validation contracts
while retaining PR #80's counting-process implementation.

## Impact Matrix

| Area | Reviewed impact | Result |
|---|---|---|
| Risk sets and ties | Breslow, Efron, Exact; `(start, stop]`; strata | fixed and locally validated |
| Optimization | objective monotonicity, line search, final normalized KKT | fixed and locally validated |
| Inference | observed information, HC0/HC1/cluster, Exact restriction | fixed and locally validated |
| Backends | NumPy/CuPy/Torch fit and prediction boundaries | fixed; physical P100 validation passes |
| Cross-validation | penalty completeness, held-out likelihood, subject grouping | fixed and locally validated |
| Compatibility | 0.2.1 PR head against 0.2.2 and PR #79 contracts | fixed |
| Benchmark evidence | synchronization, transfer scope, source version, schema | fixed; local and remote quick/full artifacts pass |
| Documentation | English-first/Chinese-follow capability and limitation contracts | fixed; contracts pass |

## Findings and Fixes

- [CRITICAL][VERSION/API][fixed] `pyproject.toml:7` and
  `statgpu/__init__.py:7`: a direct PR-head integration would have restored the
  0.2.1 version over the 0.2.2 release line. The reviewed merge retains 0.2.2;
  both files have no diff from `origin/master`.
- [CRITICAL][MERGE/INFERENCE][fixed] `statgpu/survival/_cox.py:1203`: whole-file
  conflict selection would have discarded PR #79 KKT, fitted-state, strict
  inference, and provenance fixes. The conflict was reconstructed as a
  three-way merge and the public state is synchronized from the final iterate.
- [CRITICAL][RISK_SET/CORRECTNESS][fixed] `statgpu/survival/_cox.py:4932`,
  `statgpu/survival/_cox.py:4969`, and `statgpu/survival/_cox.py:5014`:
  delayed-entry baseline risk sets used the wrong equality boundary. They now
  use open-left membership (`start < event_time`) on NumPy, CuPy, and Torch.
- [CRITICAL][BASELINE/BACKEND][fixed] `statgpu/survival/_cox.py:4978` and
  `statgpu/survival/_cox.py:5027`: merged CuPy/Torch baseline code referenced
  undefined hazard variables. Backend-local hazard and cumulative-hazard values
  are now constructed before the explicit metadata transfer.
- [HIGH][CONVERGENCE/CORRECTNESS][fixed]
  `statgpu/survival/_cox_counting.py:98` and
  `statgpu/survival/_cox_counting.py:156`: a small Newton step could previously
  certify convergence despite an excessive KKT residual. Every accepted/final
  coefficient is now checked against the normalized penalized KKT condition.
- [HIGH][SOLVER/COMPATIBILITY][fixed] `statgpu/solvers/_fista_lla.py:489`: the
  generic proximal-Newton Armijo path rejected Cox risk-set steps. Cox is routed
  through the stable FISTA-LLA path until a Cox-specific line search exists.
- [HIGH][INFERENCE/BACKEND][fixed] `statgpu/survival/_cox.py:1420` and
  `statgpu/survival/_cox.py:1574`: delayed-entry/start-stop HC0, HC1, and cluster
  inference now use exact internal counting-process residuals, aggregate repeat
  rows by subject, report provenance, and do not depend on statsmodels.
- [HIGH][PREDICTION/BACKEND][fixed] `statgpu/survival/_cox.py:5235` and
  `statgpu/survival/_cox.py:5287`: formula transforms, feature validation,
  stratified survival, log-domain accumulation, risk prediction, and scoring
  preserve the active array backend for array inputs.
- [HIGH][TORCH/PREDICTION][fixed] `statgpu/survival/_cox.py:5359`: the physical
  GPU run exposed `torch.minimum(tensor, float)` in survival prediction. The
  scalar cap now goes through the backend abstraction, which materializes the
  correct dtype/device scalar for Torch without changing NumPy or CuPy.
- [HIGH][SKLEARN/COMPATIBILITY][fixed] `statgpu/survival/_cox.py:414` and
  `statgpu/penalties/_base.py:157`: scikit-learn 1.2.2 clone checks exposed
  constructor-identity and serialization-only parameter incompatibilities.
  Normalized `inference_mode` now preserves an already-normalized constructor
  object, and the five maintained simple penalties implement
  `get_params(deep=False)` as constructor-only parameters while retaining the
  existing serialization form for the default call.
- [HIGH][CV/API][fixed] `statgpu/survival/_cox_cv.py:871` and
  `statgpu/survival/_cox_cv.py:1731`: stale CPU delayed-entry penalty rejection
  was removed; start/strata/subject data, inference policy, candidate-complete
  selection, and subject leakage checks reach every fold and the final refit.
- [HIGH][EVIDENCE/PERFORMANCE][fixed]
  `dev/benchmarks/benchmark_survival_completion.py:363` and
  `dev/benchmarks/benchmark_survival_completion.py:843`: prediction timing now
  uses backend inputs and synchronizes before/after the measured region, while
  result conversion is excluded. The artifact records the imported source
  version rather than unrelated editable-install metadata.
- [MEDIUM][TEST/COMPATIBILITY][fixed] `dev/tests/test_cox_cv.py:49`,
  `dev/tests/test_cox_phase1_completion.py:280`, and
  `dev/tests/test_pr79_remaining_review_fixes.py:237`: 0.2.1/PR #79 tests that
  expected removed delayed-entry limitations or old validation text now assert
  the PR #80 capability and 0.2.2 public error contract.
- [MEDIUM][TEST/BACKEND][fixed] `dev/tests/test_cox_core_completion.py:8` and
  `dev/tests/test_cox_phase1_completion.py:11`: three assertions implicitly
  converted backend-native CuPy results through NumPy. Tests now make the host
  conversion explicit at the assertion boundary and continue to require native
  return types before comparison.
- [MEDIUM][DOC/API][fixed] `docs/en/models/coxph.md:8` and
  `docs/cn/models/coxph.md:8`: the bilingual support matrix now documents Exact
  ties, counting-process boundaries, strata, time-varying rows, robust
  inference, subject-grouped CV, backend behavior, and the remaining Exact
  robust-covariance limitation.
- [HIGH][GPU/VALIDATION][fixed] Paramiko validation in the remote `myconda`
  environment on a Tesla P100-SXM2-16GB first reproduced 11 failures across
  backend-native test boundaries, Torch scalar prediction, and scikit-learn
  1.2.2 cloning. After the fixes above, all 15 failed and adjacent nodes passed,
  the complete matrix passed with 379 tests and two expected availability
  skips, and both quick and full benchmark schemas passed with no gate failure.

## Review-Fix Cycles

1. Reconstructed the 0.2.1-to-0.2.2 merge and reviewed survival, solver, CV,
   penalty, formula, tests, benchmark, and adjacent PR #79 contracts.
2. Fixed risk-set, baseline, convergence, inference, prediction, CV, and solver
   defects; the first targeted run exposed two stale PR #80 expectations and
   four stale PR #79 delayed-entry expectations, which were updated.
3. Re-ran affected gates, repaired benchmark evidence/version reporting,
   synchronized English/Chinese documentation, and performed a final marker,
   whitespace, version, and diff audit.
4. Uploaded the exact reviewed source bundle to an isolated remote worktree and
   ran 381 physical-GPU tests under `myconda`; the first run produced 368
   passes, two expected skips, and 11 actionable failures.
5. Fixed the three remote root causes (Torch scalar minimum, scikit-learn 1.2.2
   clone compatibility, and implicit test-side host conversion), then reran the
   11 failures plus four adjacent CuPy/Torch contracts: **15 passed**.
6. Re-ran the complete physical-GPU matrix and quick/full benchmarks, reviewed
   their machine-readable artifacts, and synchronized the final evidence into
   the bilingual public documentation.

## Validation Evidence

- `pytest` survival core target: **143 passed, 14 skipped, 0 failed** (157 total).
- Legacy `dev/tests/test_cox.py`: **8 passed, 4 skipped, 0 failed**.
- Penalized/PR79 compatibility run: **103 passed, 25 skipped** before the four
  stale capability assertions were corrected; all four corrected CPU nodes then
  passed. The remaining 11 failures were CuPy driver-initialization failures,
  not assertion or numerical failures.
- `dev/tests/test_core_contracts.py`: **7 passed, 0 failed**.
- Documentation contracts: **122 files passed**; deterministic link check:
  **0 affected files**.
- `py_compile` passed for the Cox, counting-process, CV, solver, and benchmark
  modules.
- Local quick benchmark: `schema_status="ok"`, no `gate_failures`, source version
  `0.2.2`; NumPy ordinary-heavy-ties, delayed-entry, Exact, stratified
  start-stop, inference, and subject-grouped CV scenarios all passed.
- Remote environment: Tesla P100-SXM2-16GB; Python 3.9.16, NumPy 1.24.2,
  CuPy 13.6.0, Torch 2.0.0+cu117, scikit-learn 1.2.2, and statsmodels 0.14.6.
- Initial remote physical-GPU matrix: **368 passed, 2 skipped, 11 failed**.
  All failures were reviewed and fixed; no failure was waived.
- Remote targeted review-fix rerun: **15 passed, 0 failed**.
- Remote complete current-source rerun with
  `STATGPU_REQUIRE_PHYSICAL_GPU=1`: **379 passed, 2 expected skips, 0 failed**
  in 57.19 seconds. The skips are the CuPy/Torch-unavailable negative tests,
  which cannot execute when both GPU backends are available.
- Remote quick and full benchmarks: `validation_tier="remote-full"`,
  `schema_status="ok"`, zero `gate_failures`; all four compatibility and
  inference scenarios plus subject-grouped CV passed on NumPy, CuPy, and Torch.
- The remote-tested source bundle before evidence-only documentation updates
  has SHA-256
  `5bfe538ab4d3b9ad5c3f74c9e4e979bb56eb3e801947222a40440e6d85451121`.
- Full PR delta `git diff --check origin/master`: passed.
- Version compatibility: `git diff origin/master -- pyproject.toml
  statgpu/__init__.py` is empty.

## Remaining Gate

None for the reviewed PR #80 scope. The benchmark explicitly does not invoke R,
does not estimate a deployment crossover threshold from one workload, and keeps
Exact ties at a bounded size; these are declared evidence boundaries rather than
failed gates.
