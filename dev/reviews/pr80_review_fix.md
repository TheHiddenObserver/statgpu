# PR #80 Review-Fix Report

> Review date: 2026-07-26<br>
> Original PR head reviewed: `d6f798c1834fd6318c8257eed334f84a198fa8ad`<br>
> Performance-fix base: `ad3c0026eb682ac6394369a3318e9fb806e631b8`<br>
> Final Exact risk-set SHA-256: `190567fbbc7ae40f24e9e1506ce8ac1fca5a58118a2afb800f7dec2fa05a10d8`<br>
> Final counting-solver SHA-256: `9684867f90b153c23675d8804698f76092765a3d96da05c7a3d989528782d501`<br>
> Final Cox dispatch SHA-256: `efe199e7bb40112f882109efbe8b462ab8050f52349d939d33a611f819f81e6c`<br>
> Final R/performance artifact SHA-256: `85e7c72d736b859564e598e8e6e26b26b05a6fe06a076c39645083af80ea896e`<br>
> Physical-GPU matrix SHA-256: `09cdcc9e900ba7eccae7a5d7e389c7ff6ddcbabdf5f4a648ce776b52ff8d78c6`<br>
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
| Optimization | objective monotonicity, line search, final normalized KKT, nested Exact, Torch channel scans, baseline prefixes, objective reuse | fixed; local and physical-P100 validation passes |
| Inference | observed information, HC0/HC1/cluster, Exact restriction | fixed and locally validated |
| Backends | NumPy/CuPy/Torch fit and prediction boundaries | fixed; physical P100 validation passes |
| Cross-validation | penalty completeness, held-out likelihood, subject grouping | fixed and locally validated |
| Compatibility | 0.2.1 PR head against 0.2.2 and PR #79 contracts | fixed |
| Benchmark evidence | synchronization, transfer scope, source version, schema, Exact scaling, R external alignment | fixed; local and remote artifacts pass with zero gate failures |
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
- [HIGH][PERF][fixed] `statgpu/survival/_cox.py:3921`,
  `statgpu/survival/_cox.py:3957`, `statgpu/survival/_cox.py:4154`,
  `statgpu/survival/_cox.py:4494`, `statgpu/survival/_risk_sets.py:182`, and
  `statgpu/survival/_risk_sets.py:739` -
  dense Efron ties launched small kernels once per failure group and tie
  substep, while Exact ties nested GPU loops over failure groups, risk rows,
  and subset sizes.
  Impact: Exact remained slower than NumPy at small sizes and scaled poorly as
  the number of failure groups grew.
  Fix: CuPy/Torch Efron moments remain in bounded cumulative tensors; Exact now
  carries a failure-group batch dimension and evaluates all active subset sizes
  in one row-wise DP scan. Launch-bound iterations fall from the sum of risk-set
  sizes to the sample count, and six redundant state copies per row are gone.
  Evidence: synchronized P100 fits are 2.59x/4.12x faster than NumPy at n=960
  and 5.14x/8.36x faster at n=1920 for CuPy/Torch.
- [HIGH][PERF][fixed] `statgpu/survival/_risk_sets.py:455` and
  `statgpu/survival/_risk_sets.py:855` - the batched GPU DP removed inner
  scalar launches but still recomputed every nested right-censored risk set;
  NumPy retained the same failure-group factor.
  Impact: at `n=1920`, R completed in 0.048 s while the reviewed
  NumPy/CuPy/Torch paths took 54.222/11.032/6.621 s.
  Fix: ordinary one-stratum right-censored fits now sort rows by decreasing stop
  time and reuse a single elementary-symmetric prefix DP across all failure
  groups on NumPy, CuPy, and Torch. Sorted event-time segment prefix sums also
  remove the remaining quadratic failure-group-by-sample numerator mask. A
  pre-allocation 512 MiB gate and conservative exponent/combination/moment bounds
  fall back to the normalized implementation;
  delayed entry, strata, and score-residual requests keep their existing paths.
  Evidence: on the final synchronized P100 `n=1920` case, full-fit times are
  0.0396/0.0915/0.0589 s, about 1368x/121x/112x faster than the reviewed
  pre-prefix NumPy/CuPy/Torch implementation. All R precision gates pass.
- [HIGH][PERF][fixed] `statgpu/survival/_risk_sets.py:1299` - after the Exact
  objective was optimized, ordinary right-censored baseline-hazard inference
  still built a full risk mask for every failure time.
  Impact: phase profiling at `n=61,440` measured 6.847/5.988/3.328 s in the
  NumPy/CuPy/Torch baseline phases, substantially more than the optimized Exact
  solver for NumPy and CuPy and therefore hiding the large-sample GPU benefit.
  Fix: sort each stratum by descending stop time and compute every ordinary
  right-censored risk denominator from one log-risk prefix. NumPy uses
  `logaddexp.accumulate`, Torch uses `logcumsumexp`, and CuPy uses a shifted
  exponential cumulative sum inside a conservative predictor-range gate.
  Extreme CuPy predictors and delayed entry retain the stable backend-native
  per-group implementation.
  Evidence: the same baseline phases now take 0.0202/0.00701/0.00265 s. At
  `n=122,880`, synchronized full-fit CuPy time is 0.1518 s versus 2.589 s for R
  and 3.023 s for NumPy, with zero convergence or R precision gate failures.
- [HIGH][PERF/GPU][fixed] `statgpu/survival/_risk_sets.py:129`,
  `statgpu/survival/_risk_sets.py:532`, and
  `statgpu/survival/_risk_sets.py:684` - PyTorch 2.0's CUDA implementation of
  a long multidimensional `cumsum(dim=0)` dominated the nested Exact moment
  updates even though the equivalent one-dimensional scan was fast.
  Impact: on the P100 at `n=122,880`, Torch took 3.0308 s versus 3.0230 s for
  NumPy, while an operator probe measured 43.725 ms for an `(n, 4)` Torch scan
  versus 0.164 ms for CuPy; splitting four Torch channels into one-dimensional
  scans reduced that operator from 10.034 ms to 0.103 ms in the controlled
  layout probe.
  Fix: eligible Torch CUDA moment arrays are laid out channel-major, scanned as
  bounded one-dimensional channels, and stacked back on device. Conservative
  defaults require at least 2,048 rows and at most 64 trailing channels;
  environment variables expose both gates. CPU, small, and wide inputs retain
  the native scan. The extra transpose/output workspace is included in the
  nested memory decision; if only that extra space is unavailable, the nested
  DP stays active with the native scan instead of falling back to general Exact.
  Evidence: final synchronized Torch time is 0.1000 s at `n=122,880`, 30.32x
  faster than the prior Torch result, 30.44x faster than NumPy, 1.43x faster
  than CuPy, and 26.92x faster than R. Dedicated CUDA scan and full-objective
  parity tests pass, as do the memory-gate regression and all R precision gates.
- [MEDIUM][PERF/MAINT][fixed] `statgpu/survival/_cox_counting.py:95`,
  `statgpu/survival/_cox_counting.py:154`, and
  `statgpu/survival/_cox.py:1484` - the solver recomputed the accepted final
  objective, recomputed the default zero-init null objective, and the estimator
  evaluated the null score/information a third time.
  Impact: every Exact fit paid for up to two avoidable objective evaluations.
  Fix: reuse the initial null state, reuse the accepted final state unless score
  residuals are requested, and return null score/information for the estimator's
  score test. A call-recording regression locks the reuse contract.
- [HIGH][PERF][fixed] `statgpu/survival/_risk_sets.py:855` - the first batched
  Exact prototype allocated dense group-by-sample masks before enforcing its
  workspace cap.
  Impact: an oversized workload could OOM before reaching the documented
  backend-native memory-bounded fallback.
  Fix: unique failure counts and the full workspace estimate are now computed
  first. A zero or exceeded `STATGPU_EXACT_BATCH_MAX_BYTES` ceiling returns
  before any dense mask allocation.
  Evidence: the forced-fallback regression verifies numerical parity and that
  no group-by-sample float conversion occurs before fallback; local and P100
  matrices pass.
- [MEDIUM][EVIDENCE/EXTERNAL][fixed]
  `dev/benchmarks/benchmark_exact_ties_scaling.py`: the Exact benchmark compared
  StatGPU backends only with NumPy, so it did not establish agreement with an
  independent Exact implementation or expose shape-dependent external timing.
  The reusable benchmark now optionally runs R
  `survival::coxph(ties="exact", robust=FALSE, timefix=FALSE)`, records R and
  package versions, compares coefficients, exact partial log likelihood,
  model-based covariance, convergence, and synchronized timings, and fails
  closed on numerical or convergence gate violations. Separate cases cover
  right-censoring, delayed entry, strata, and delayed entry plus strata. The
  reviewed script is explicitly unignored so a normal commit cannot omit the
  reusable evidence entry point while retaining the repository's blanket ignore
  for ad-hoc benchmarks.
  Evidence: R 4.4.1/survival 3.8.9 and all NumPy/CuPy/Torch cases passed with
  zero gate failures; maximum coefficient/log-likelihood/covariance differences
  were `1.30e-09`/`5.12e-09`/`5.01e-12`.
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
  1.2.2 cloning. After the fixes above, all failed and adjacent nodes passed.
  The final current-source matrix passed with 392 tests and two expected
  availability skips, including CuPy/Torch baseline parity, the extreme-CuPy
  stability fallback, Torch channel-scan parity, and its memory gate; quick/full
  benchmark schemas have no gate failure.

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
7. Profiled the reported heavy/Exact tie slowdown, rejected slower raw-CuPy,
   grouped-Torch, Triton, and `segment_reduce` alternatives, and vectorized the
   launch-bound Efron path plus Exact subset-size updates.
8. Re-reviewed the first optimization, included group-by-substep memory in the
   Efron workspace gate, applied the same gate to Torch log-likelihood, and
   closed its local and remote matrices.
9. Reproduced Exact scaling separately, batched failure groups into one row-wise
   DP scan, removed redundant per-row state copies, moved the 512 MiB gate ahead
   of dense allocation, and reran precision, 13-file physical-GPU, and n=960/
   n=1920 scaling gates to closure.
10. Added an optional R `survival::coxph(ties="exact")` reference to the same
    machine-readable benchmark, fixed its fail-closed R diagnostics and direct
    covariance extraction, ran ordinary/delayed-entry/strata alignment on the
    remote P100 host, independently audited the artifact, and synchronized the
    shape-specific findings into English-first and Chinese-follow documentation.
11. Re-reviewed R's right-censored advantage, replaced repeated nested risk-set
    work with a backend-native prefix DP, added pre-allocation/numerical gates,
    and removed redundant final/null/score-test objective evaluations.
12. Compared the prefix path against the forced normalized fallback, reran the
    local Cox matrix and exact-source physical-GPU matrix, regenerated R timing
    and precision evidence with all three source hashes, and completed another
    English-first/Chinese-follow review-fix pass.
13. Phase-profiled the large-sample complete fit, isolated the remaining
    failure-time-by-sample baseline scan, and replaced the ordinary
    right-censored path with backend-native descending-stop log-risk prefixes.
14. The targeted P100 rerun exposed CuPy 13.6's missing
    `logaddexp.accumulate`; added the shifted-cumulative implementation plus an
    extreme-predictor stable fallback, then reran local, physical-GPU, R
    alignment, and large-scale timing gates to closure.
15. Microprofiled the remaining Torch/R/NumPy gap on the P100, isolated the
    PyTorch 2.0 multidimensional long-axis scan, measured its row/channel
    crossover, and implemented gated per-channel one-dimensional CUDA scans.
16. Re-reviewed numerical ordering and peak workspace, replaced an invalid
    bit-equality expectation with strict floating-point tolerances, and ensured
    scan-workspace pressure disables only the split scan rather than the nested
    DP. Targeted, full local, full P100, and R/performance gates then closed.
17. Audited the final source/artifact hashes and synchronized the algorithm,
    performance, compatibility, and limitation evidence English-first and then
    Chinese-follow.

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
  CuPy 13.6.0, Torch 2.0.0+cu117, scikit-learn 1.2.2, statsmodels 0.14.6,
  R 4.4.1, and survival 3.8.9.
- Initial remote physical-GPU matrix: **368 passed, 2 skipped, 11 failed**.
  All failures were reviewed and fixed; no failure was waived.
- Remote targeted review-fix rerun: **15 passed, 0 failed**.
- Remote complete current-source rerun with
  `STATGPU_REQUIRE_PHYSICAL_GPU=1`: **379 passed, 2 expected skips, 0 failed**
  in 57.19 seconds. The skips are the CuPy/Torch-unavailable negative tests,
  which cannot execute when both GPU backends are available.
- Prior Exact-batched physical-GPU matrix, including the pre-allocation batch
  workspace-fallback regression: **381 passed, 2 expected skips, 0 failed** in
  48.26 seconds.
- Prior nested-Exact/objective-reuse physical-GPU matrix: **384 passed, 2
  expected skips, 0 failed** in 45.74 seconds.
- Prior baseline-prefix current-source physical-GPU matrix: **388 passed, 2
  expected skips, 0 failed** in 45.55 seconds.
- Final Torch-channel-scan current-source physical-GPU matrix: **392 passed, 2
  expected skips, 0 failed** in 46.98 seconds. The final dedicated CUDA gate
  passed the scan and complete-objective parity tests; the adjacent targeted
  selection passed 7 tests with 47 deselected.
- Local 13-file affected Cox matrix: **297 passed, 97 skipped, 0 failed** in
  44.68 seconds. The focused risk-set file passed **45 tests, 9 skipped** in
  31.57 seconds.
- Remote quick and full benchmarks: `validation_tier="remote-full"`,
  `schema_status="ok"`, zero `gate_failures`; all four compatibility and
  inference scenarios plus subject-grouped CV passed on NumPy, CuPy, and Torch.
- Final synchronized full benchmark (`repeats=3`, `warmups=1`) measured
  heavy-ties medians of 0.477 s NumPy, 0.179 s CuPy, and 0.212 s Torch. Against
  the pre-optimization GPU medians, CuPy improved 8.36x and Torch 24.31x; both
  are now faster than NumPy for this 20,000-by-32 workload.
- The final synchronized Exact/Torch-channel-scan scaling benchmark (`p=4`,
  maximum tie size 8, full fit plus inference) measured R/NumPy/CuPy/Torch
  medians of 0.0460/0.0354/0.0838/0.0571 s at `n=1,920`,
  0.295/0.273/0.0949/0.0558 s at `n=15,360`,
  1.323/1.465/0.1114/0.0662 s at `n=61,440`, and
  2.691/3.043/0.1430/0.1000 s at `n=122,880`. At the largest size Torch is
  30.32x faster than the previous Torch implementation, 26.92x faster than R,
  30.44x faster than NumPy, and 1.43x faster than CuPy. The `n=1,920` GPU paths
  remain launch-bound.
- Phase profiling at `n=61,440` measured baseline-hazard construction before
  the final optimization at 6.847/5.988/3.328 s on NumPy/CuPy/Torch and after it
  at 0.0202/0.00701/0.00265 s. No implicit CPU fallback was added.
- R external-alignment artifact: status `complete`, zero gate failures, and all
  three recorded source hashes match the uploaded worktree. It covers bounded
  scaling plus separate right-censored, delayed-entry, strata, and combined
  delayed-entry/strata cases. Across NumPy/CuPy/Torch, the largest differences
  from R were `1.30e-09` for coefficients, `5.12e-09` for exact partial log
  likelihood, and `5.01e-12` for model-based covariance; every fit converged.
- On the separate `n=160` delayed-entry case, R/NumPy/CuPy/Torch medians were
  59.116/0.174/0.603/0.358 seconds. The artifact keeps all timings as
  shape-specific evidence and records the unequal process boundary: StatGPU
  includes conversion and inference, while R includes the `coxph` call and
  inference but excludes startup, package loading, and CSV parsing.
- Final heavy-ties coefficient differences versus NumPy are at most
  `8.88e-16` (CuPy) and `1.22e-15` (Torch); Exact coefficient differences are
  at most `8.33e-17` and `5.55e-17`, with zero log-likelihood difference.
- The final remote-loaded risk-set/counting-solver/Cox-dispatch SHA-256 values
  are `190567fbbc7ae40f24e9e1506ce8ac1fca5a58118a2afb800f7dec2fa05a10d8`,
  `9684867f90b153c23675d8804698f76092765a3d96da05c7a3d989528782d501`, and
  `efe199e7bb40112f882109efbe8b462ab8050f52349d939d33a611f819f81e6c`.
  `exact-torch-channel-scan-memory-final.json` records the same hashes, benchmark
  SHA-256 `a3c1ed48d03ff3d9d557a478d9ec832cd0a303b85d64fcbe1a397d7e6a649b39`,
  P100 metadata, and the R/performance evidence; its SHA-256 is
  `85e7c72d736b859564e598e8e6e26b26b05a6fe06a076c39645083af80ea896e`.
  The final matrix XML `exact-torch-channel-scan-memory-final-matrix.xml` has
  SHA-256
  `09cdcc9e900ba7eccae7a5d7e389c7ff6ddcbabdf5f4a648ce776b52ff8d78c6`.
- Full PR delta `git diff --check origin/master`: passed.
- Version compatibility: `git diff origin/master -- pyproject.toml
  statgpu/__init__.py` is empty.

## Remaining Gate

None for the reviewed PR #80 scope. External R alignment closes the
independent-implementation accuracy gate, but timings remain shape-specific.
Both GPU backends are faster than R from the measured `n=15,360` ordinary
right-censored case through `n=122,880`; Torch is the fastest measured backend
on that low-dimensional large-sample shape. Small GPU fits remain launch-bound,
and wide Torch moment tensors keep the native scan. Large individual tie blocks
remain combinatorial; delayed-entry, score-residual, and multi-stratum Exact use
the backend-native normalized paths. These are explicit evidence boundaries
rather than failed gates.
