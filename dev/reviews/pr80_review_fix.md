# PR #80 Review-Fix Report

> Review date: 2026-07-28<br>
> Original PR head reviewed: `d6f798c1834fd6318c8257eed334f84a198fa8ad`<br>
> Performance-fix base: `ad3c0026eb682ac6394369a3318e9fb806e631b8`<br>
> Current risk-set SHA-256: `eee6900332526d5e68815e46d6d43a0f52e981760b724c10f98740fc56eeb3da`<br>
> Current Cox-loss SHA-256: `7220b6de7ebc0a72e0468b86e56617252eff1722339b9bd1c937976a1f41a7ea`<br>
> Current FISTA-LLA SHA-256: `3c9a665d0d46bebc32c6e43dbd2f777d989fe09114f73a2c7ae1e9bdb1642536`<br>
> Current penalized-fit mixin SHA-256: `56fcaa3667afc27935a73a363e77ca940560ce9beb3019b809c2544998b6062d`<br>
> Current penalized-Cox estimator SHA-256: `8349b9a9a3d80f254db06bdd2e7601aa68c1d36b83e112973fc85ef8afa3ea55`<br>
> Trusted-gradient artifact source commit: `98de333d5be17715a2cafa0c560aa78a9c92b3e1`<br>
> Final counting-solver SHA-256: `466bdc86891bc41749e2272d2566344cd28c112b7234fb5d1e104df25c61e2da`<br>
> Final Cox dispatch SHA-256: `17738770458ae986037f5e1209a8da51e1bad41a1869d5d5518886c15ad348d0`<br>
> Final R/performance artifact SHA-256: `85e7c72d736b859564e598e8e6e26b26b05a6fe06a076c39645083af80ea896e`<br>
> Final stratified-Exact artifact SHA-256: `0bc0325240b64e1a957f0597a969233374ca4696571c0fcc6229a8ea0986e2c6`<br>
> Follow-up delayed-entry+strata artifact SHA-256: `b3c9cadb3235b8280fc0c338d81302d4929d109da6506208868782d2fac01c1b`<br>
> Follow-up strata-count artifact SHA-256: `c7465368a66f748a5f1e410795c5ff3acb64ca6e43efcb6cdeec63ee22de335f`<br>
> Penalized-Cox trusted-gradient artifact SHA-256: `8956b71e09ac5036e726f913e4665767919edb6ae497d00dc0f34f83da35d51c`<br>
> Ordinary-Cox stability artifact SHA-256: `29855aa68b78f93dfc233b4fa45ff813ccf7875e2eb197ab22ae753c551b6f3e`<br>
> Exact-kernel physical-GPU matrix SHA-256: `09cdcc9e900ba7eccae7a5d7e389c7ff6ddcbabdf5f4a648ce776b52ff8d78c6`<br>
> Boundary/workspace artifact source commit: `16695feec8d4187b591d8a24d8977de543fd33c3`<br>
> Boundary/workspace artifact SHA-256: `e876d0cc8760486259aff967c1ed6de0a4fc3915cd9aac8c745ec2940b9ca41d`<br>
> Boundary adapter SHA-256: `c6742e20dd57c8dc5a36dbe594e7ce040effae4217939538ab39df0fb338f9d3`<br>
> Original merge base: `a4879fb` (0.2.1 line)<br>
> Compatibility target: `origin/master` at `7ccf616` (0.2.2 line)<br>
> Status: `COMPLETE` for source review; external GPU-CI wiring remains an infrastructure action

## Review Contract

This review follows `dev/AGENTS.md` and the complete
`.claude/skills/code-review.md` workflow: inspect before editing, assess impact
across correctness/API/backend/inference/performance/docs/tests, fix all relevant
critical/high/medium findings, rerun affected gates, and re-review the resulting
diff. PR merge and release remain outside this report; reviewed commits are
pushed only after the local and physical-GPU gates pass.

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

- [HIGH][PERF][fixed] `statgpu/survival/_risk_sets.py`: multi-stratum Exact
  objectives bypassed both optimized one-stratum kernels and executed one
  elementary-symmetric DP per failure time. Exact likelihood additivity now
  composes the bounded nested/batched kernels once per stratum; NumPy also uses
  the memory-gated batched path for eligible delayed-entry workloads. No device
  fallback or statistical-definition change was introduced.
- [MEDIUM][TEST/ARTIFACT][fixed]
  `dev/benchmarks/benchmark_exact_ties_scaling.py` and
  `dev/tests/test_survival_risk_sets.py`: the maintained scaling benchmark only
  generated ordinary right-censored data, and no regression forced the
  multi-stratum fast paths on all three backends. The benchmark now accepts
  `--scaling-scenario strata`, and tests compare nested/batched results with the
  forced memory-bounded reference on NumPy, CuPy, and Torch.

## 2026-07-27 Follow-up Review-Fix Cycle

This follow-up applies the additional P1/P2/P3 review list against commit
`99a43881ffe50d8116702bde150ea8a69c1f8881` and re-runs the complete affected
axes from `.claude/skills/code-review.md`: survival semantics, loss/solver,
penalty continuation, NumPy/CuPy/Torch behavior, memory lifetime, performance,
tests, benchmark provenance, and bilingual documentation.

- [HIGH][P1-1][LOSS/SOLVER][fixed] Cox SCAD/MCP computed the zero-score from a
  sorted/centered loss cache but then passed the original `X`/`y` into
  `fista_lla_path`; that function discarded its own `X_proc`/`y_proc`, so every
  LLA/FISTA gradient could sort and transfer the response again. Cox preprocessing
  now returns an opaque active-cache token, FISTA-LLA carries the exact
  preprocessed pair through continuation, and the estimator reuses it for the
  zero-score and fit. The three-backend SCAD/MCP regression requires exactly one
  preprocessing call per fit and makes the obsolete fused objective raise if it
  is called.
- [HIGH][P1-2][GPU/PERF][fixed] the nonquadratic inner loop requested a fused
  objective although it used only the gradient; Cox converted that unused value
  to a Python float, validated coefficients with a device scalar on every step,
  rebuilt failure-group indices/fractions on the device, and range-checked every
  segment synchronously. The solver now uses a Cox gradient-only trusted path,
  metadata is cached by backend/device once, the fused public value stays a
  backend scalar, fixed bounded segments avoid hot-loop host reads, and finite
  state is checked together with the existing periodic convergence transfer.
  Public loss calls and periodic Hessian/Lipschitz evaluations retain the full
  numerical validation path.
- [HIGH][P1-3][MEMORY][fixed] the estimator's loss retained sorted design,
  response, order, and device metadata after fit, so freeing allocator pools
  could not release active training allocations. `release_fit_cache()` now
  invalidates the preprocessing token and clears every host/device training
  reference. Fit success/failure, refit reset, prediction/score cleanup, and
  destruction all release loss state before allocator cleanup. Physical-P100
  active-byte tests keep the input arrays alive and verify that fitting does not
  retain an additional training-sized allocation.
- [MEDIUM][P2-1][VALIDATION][fixed] low-level counting-process input conversion
  cast floating strata directly to integer labels. NumPy, CuPy, and Torch now
  reject fractional/non-finite labels before conversion and continue to accept
  integral floating labels.
- [MEDIUM][P2-2][TORCH/PERF][fixed] the Torch channelwise Exact scan was enabled
  on every CUDA architecture from evidence collected only on Torch 2.0/P100.
  `STATGPU_TORCH_EXACT_SCAN_STRATEGY={auto,native,channelwise}` now exposes the
  policy; `auto` enables only the evidenced Torch 2.0 + compute-capability 6.0
  combination and conservatively uses native scans elsewhere. Explicit strategy
  tests continue to exercise channelwise numerical parity on any GPU.
- [MEDIUM][P2-3][STRATA/PERF][fixed] centering covariates looped over strata with
  a scalar device read, ordinary multi-stratum Exact called one prefix DP per
  stratum, and a failed fast path discarded completed strata before a global
  reference recomputation. Centering now uses `unique`/inverse codes plus
  `add.at` or `index_add_`; ordinary right-censored Exact uses one segmented
  prefix DP across all strata; delayed-entry Exact with at least eight GPU strata
  first tries one global batched path, while smaller GPU cases and NumPy retain
  per-stratum batches; and only the failing stratum reaches the per-group
  reference. No statistical boundary or CPU fallback changed.
- [MEDIUM][P2-4][DOC/BACKEND][fixed] the loss guide overstated that every Cox
  object remained on the selected device. The English-first and Chinese-follow
  text now records the one-time sorted `time`/`event` host copy used to construct
  deterministic group metadata, while distinguishing it from iterative matrix,
  predictor, objective, gradient, or Hessian transfers.
- [MEDIUM][P2-5][BENCHMARK][fixed]
  `benchmark_exact_ties_scaling.py` now accepts all four right-censored,
  delayed-entry, strata, and combined scenarios; records the complete command;
  uses five ordinary repeats by default; supports explicit reduced-repeat and R
  repeat policies; reports R timeout rather than aborting the artifact; verifies
  finiteness/convergence for every repeat; and takes numerical fields from the
  actual median-ranked run. The committed delayed-entry+strata artifact is
  generated by this maintained CLI rather than an ad-hoc driver.
- [MEDIUM][P2-6][CI][external infrastructure pending] repository-hosted CI still
  has no CUDA runner. Adding an unconfigured `self-hosted` label would leave PR
  checks queued indefinitely, so no fictitious gate was added. The maintained
  GPU tests are complete and pass under `STATGPU_REQUIRE_PHYSICAL_GPU=1`; wiring
  them into nightly/required CI needs a repository CUDA runner or equivalent
  external CI credential, which is outside this source-only PR.
- [LOW][P3-1][CONFIG][fixed] every Exact workspace/scan integer environment
  variable now uses bounded, non-negative parsing with safe defaults for invalid
  strings and caps for unreasonable values.
- [LOW][P3-2][MAINT][deferred] consolidating all local backend helper functions
  into `statgpu.backends` is a broad internal refactor with no current defect and
  would enlarge the survival-risk regression surface. The follow-up reuses new
  helpers within `_risk_sets.py` but leaves cross-module consolidation for a
  dedicated maintenance PR.

Follow-up performance evidence on the remote Tesla P100-SXM2-16GB:

- penalized Cox at `n=4096`, `p=12`, 64 tie bins: SCAD NumPy/CuPy/Torch medians
  `0.4416/0.1749/0.1317` s (CuPy 2.52x, Torch 3.35x); MCP medians
  `0.4564/0.1874/0.1462` s (2.44x, 3.12x). Every fit performed one preprocess;
  maximum coefficient difference from NumPy was `5.0e-16`.
- vectorized centering at 100,000 rows remained approximately constant from 3
  through 1,000 strata: NumPy `0.045-0.051` s, CuPy `0.00122-0.00148` s, and
  Torch `0.00084-0.00121` s.
- for a controlled right-censored Exact workload with one tied failure group per
  stratum, the 1,000-stratum NumPy/CuPy/Torch medians fell from
  `0.2643/4.3872/1.7104` s before segmented DP to
  `0.00653/0.00825/0.00417` s after it. Torch is 1.56x faster than NumPy; CuPy is
  within 1.26x, and maximum log-likelihood difference is `4.10e-12`.
- the maintained delayed-entry + 3-strata fit benchmark completed at 320 through
  10,240 rows with zero gate failures. At 10,240 rows the NumPy/CuPy/Torch
  medians were `136.02/36.50/21.95` s, so CuPy and Torch were 3.73x and 6.20x
  faster than NumPy. Torch crossed NumPy by 2,560 rows and CuPy by 5,120 rows;
  smaller cases remain launch-bound. R survival completed 320 and 640 rows in
  `0.327/24.173` s and was recorded as an explicit 30-second timeout at larger
  sizes rather than producing a synthetic timing or aborting the artifact.

## 2026-07-27 Trusted-Gradient Underflow Follow-up

- [HIGH][P1][NUMERICAL/PERF][fixed] `gradient_preprocessed()` skipped duplicate
  device-scalar finite checks by passing `validate_numerics=False`, but that flag
  also disabled the adaptive predictor-range segmentation required for stable
  suffix risk sets. A finite two-row example with centered predictors
  `(500, -500)` therefore underflowed the second denominator to zero and returned
  `NaN`, while the public gradient returned zero. Finite-state validation is now
  a separate `validate_finite_state` concern; adaptive segmentation is
  unconditional and cannot be disabled by the trusted solver path. Breslow and
  Efron trusted gradients match the public and shared counting-process gradients
  on NumPy, CuPy, and Torch for the deterministic departing-maximum example.
- [MEDIUM][P2][VALIDATION][fixed] integral floats and unsigned integers were
  accepted without checking the signed-int64 domain, so values such as `1e30`
  or `uint64.max` could overflow and merge distinct strata. All three backends
  now reject values below `-2**63`, floats at or above `2**63`, and unsigned
  values above `2**63 - 1` before casting. Signed-int64 boundary labels remain
  accepted and unchanged.
- [MEDIUM][P2][GPU EVIDENCE][fixed for source evidence; CI infrastructure still
  external] `benchmark_penalized_cox_trusted_gradient.py` generates a
  machine-readable artifact from a clean detached worktree. It records exact
  Git commit and source/script hashes, complete command argv, P100/CUDA/CuPy/
  Torch metadata, six trusted/public/shared gradient comparisons, and twelve
  SCAD/MCP fit results with coefficients, objective components, KKT residuals,
  finite state, iterations, and timing. This closes the requested auditable
  physical-GPU evidence without pretending that repository CI has a CUDA runner.

Follow-up validation and performance evidence:

- remote Tesla P100 matrix with physical NumPy, CuPy CUDA, and Torch CUDA:
  **343 passed, 0 failed** in 22.30 seconds; the focused remaining-finding
  selection passed **32 tests**;
- local affected Cox/survival matrix: **253 passed, 90 optional GPU skips, 0
  failed**; documentation contracts still pass for 122 files;
- at the superseded zero-sync head, `n=4096`, `p=12`, 64 time bins, SCAD
  NumPy/CuPy/Torch medians were
  `0.121/0.0318/0.0341` seconds and MCP medians were
  `0.105/0.0314/0.0324` seconds. These measurements are retained as review
  history and are not a claim about the later cancellation-safe head;
- the earlier `penalized_cox_trusted_gradient_pr80_20260727.json` was `complete` with zero
  gate failures from clean commit `bbbf4b9`. Across its 6 gradient and 12 fit
  cases, every trusted gradient records zero host-scalar conversions, every fit
  records five iterations, maximum gradient/coefficient/KKT differences are
  zero, and maximum cross-backend objective difference is `7.13e-12`. It is
  superseded by the signed-moment-cancellation artifact described below.

## 2026-07-27 Remaining P2/P3 Follow-up

- [MEDIUM][PERF][superseded] `statgpu/losses/_cox_ph.py`: trusted gradients still
  called `_stable_segment_boundaries()`, synchronizing every predictor-range
  decision. NumPy now uses reverse `logaddexp.accumulate`, Torch uses
  `logcumsumexp`, and CuPy uses a fixed-topology parallel RawKernel. A maintained
  counter requires zero `_to_float_scalar` calls per trusted gradient; public
  validation retains its independently stable adaptive implementation. The next
  review found that separating positive and negative log moments is not stable
  under strong cancellation; the zero-sync implementation was removed rather
  than preserving a performance claim with an incorrect gradient.
- [MEDIUM][BACKEND][fixed] `statgpu/survival/_risk_sets.py`: Torch 2.0 rejected
  every NumPy `uint64` strata input before inspecting its range. Representable
  host unsigned labels are now range-checked and converted to int64 before
  `torch.as_tensor`; oversized values remain rejected on all three backends.
- [MEDIUM][SOLVER/API][fixed] `statgpu/solvers/_fista_lla.py`: the generic path
  incremented `total_iter` after the convergence break. It now counts each
  completed proximal update first. The regression verifies cumulative path
  counts `[1, 2, 3, 4, 5]` across NumPy, CuPy, and Torch.
- [MEDIUM][PERF/BACKEND][fixed]
  `statgpu/linear_model/penalized/_penalized_cox.py`: estimator validation copied
  the complete packed GPU target to the host before loss preprocessing. It now
  validates event values on the selected backend and transfers only the two
  booleans `invalid` and `has_event`.
- [LOW][DOC][fixed] `statgpu/losses/_cox_ph.py` now states that public Hessian
  evaluation uses the specialized right-censored implementation; the shared
  counting-process objective is an independent compatibility baseline.
- [LOW][DOC][fixed] the root PR #80 follow-up changelog is reduced to four
  one-line bullets. Detailed algorithms, validation, and timings remain in the
  EN/CN changelogs and this report.

Physical-P100 evidence for this follow-up:

- the focused correctness/synchronization/iteration/uint64/event-transfer
  selection passed **32 tests**, and scan boundary sizes 1/255/256/257/513
  passed **30 tests** across NumPy, CuPy, and Torch;
- at `n=4096`, `p=12`, and 64 time bins, SCAD NumPy/CuPy/Torch medians were
  `0.121/0.0318/0.0341` seconds and MCP medians were
  `0.105/0.0314/0.0324` seconds. The trusted-gradient medians under an extreme
  predictor range were `0.00851/0.00175/0.00342` seconds, with maximum gradient
  difference from the public stable path below `1.59e-13`. These timings belong
  to the superseded signed-log implementation.

## 2026-07-27 Signed-Moment Cancellation Follow-up

- [CRITICAL][BUG][fixed] `statgpu/losses/_cox_ph.py`: the zero-sync trusted
  gradient stored positive and negative first moments as separate log-sums and
  reconstructed their difference. When both moments were approximately `1e15`
  but their signed difference was order one, the two logs rounded together and
  the valid zero gradient became `-0.125`. The trusted path now reuses the
  cancellation-safe scaled direct-moment calculation. Regressions cover scales
  `1e8`, `1e12`, and `1e15`, Breslow/Efron, trusted/public/shared parity, and
  actual SCAD/MCP coefficient, finite-state, iteration, and KKT behavior on all
  three backends.
- [HIGH][API/BACKEND][fixed] public Cox normalization boundaries cast complex
  arrays to float before validation, silently discarding their imaginary parts.
  `X`, time, event, start, stop, and coefficient/beta inputs are now rejected
  before casting for NumPy, CuPy, and Torch. Penalized-Cox event validation and
  initialization follow the same contract.
- [MEDIUM][PERF][fixed] the removed signed-log path materialized multiple
  full-length positive/negative/log-scan buffers and had no byte ceiling when
  `p=1`. The retained direct-moment path caps each scan at 65,536 rows and two
  million moment elements. Structural tests enforce both limits; physical-GPU
  tests gate additional allocator/active workspace at 64 MiB for `n=300,000`.
- [LOW][ARTIFACT/DOC][fixed]
  `benchmark_penalized_cox_trusted_gradient.py` no longer labels its fit timing
  as an unspecified latency. It explicitly records that fresh-process cold start
  is unmeasured, whether gradient work ran before fitting, the first fit in that
  warmed process, a repeated steady-state fit, and whether CuPy RawKernel JIT is
  applicable. The regenerated artifact also records predictor-range sync counts
  rather than requiring a misleading zero.

Final evidence for this follow-up:

- focused cancellation, complex-boundary, and bounded-block selection:
  **31 passed, 60 optional GPU skips, 0 failed** locally;
- CPU artifact dry run: 8 trusted/public/shared comparisons and 16 SCAD/MCP fit
  cases had zero gradient, coefficient, objective, and KKT gate differences;
  its only expected failure was the dirty-worktree audit used during development;
- physical Tesla P100 follow-up file: **148 passed, 0 failed**; expanded
  Cox/loss/solver matrix: **413 passed, 102 expected skips, 0 failed**;
- clean-source artifact from commit `98de333d5be17715a2cafa0c560aa78a9c92b3e1`:
  `status="complete"`, zero gate failures, 24 gradient comparisons, 48 fit/KKT
  cases, six performance cases, and two workspace cases. Every deterministic
  gradient, coefficient, and KKT difference is zero; maximum cross-backend
  objective difference is `7.13e-12`; every trusted call records the one
  documented predictor-range scalar check;
- physical workspace delta at `n=300,000`, `p=1` was 6,758,400 bytes for CuPy
  and 6,076,928 bytes for Torch, below the 64 MiB gate;
- synchronized `n=4096`, `p=12`, 64-bin Efron medians after one excluded warmup
  were SCAD NumPy/CuPy/Torch `0.08350/0.03148/0.02137` seconds and MCP
  `0.08469/0.03100/0.02133` seconds. CuPy/Torch speedups were 2.65x/3.91x for
  SCAD and 2.73x/3.97x for MCP, with zero coefficient difference from NumPy.

## Ordinary-Cox Stability and Resource-Safety Follow-up

- [CRITICAL][BUG/BACKEND][fixed] ordinary unpenalized Breslow/Efron fits could
  still evaluate raw `exp(X @ beta)`. Centered `X=[-1000, 0, 1000]` with
  `init_coef=[1]` overflowed in the legacy NumPy/CuPy/Torch paths. Every public
  fit now enters the stable shared solver; the ordinary nonrobust case uses the
  cancellation-safe bounded suffix-moment kernel rather than the dense
  group-by-row reference. The deterministic case is finite for both tie rules
  and all three backends.
- [HIGH][PERF][fixed] legacy Breslow Hessian strategies could allocate two or
  more `(n,p,p)` buffers without considering `n`. The conservative workspace
  estimate includes simultaneously live row/group moments and is capped by
  `STATGPU_BRESLOW_HESSIAN_MAX_BYTES` (default 512 MiB). CPU selects the
  incremental strategy and CuPy selects bounded streaming GEMM when the cap is
  exceeded; the stable ordinary public path already uses bounded row blocks.
- [HIGH][API/BACKEND][fixed] `CoxPH.fit`, `CoxPH.score`, `CoxPHCV.fit/score`,
  and held-out partial likelihood could discard complex components before the
  low-level guards ran. They now share the pre-cast real-valued validator for
  `X`, packed targets, time/event/start/entry, coefficients, and initial
  coefficients. CPU plus physical CuPy/Torch regressions cover the boundaries.
- [HIGH][FALLBACK/BACKEND][fixed] counting, legacy Torch/CuPy Newton, and the
  fused CuPy Hessian used broad exception fallbacks that could retry a larger
  least-squares solve after OOM and report a device failure as singular
  information. Device/runtime failures now propagate unchanged; only explicit
  singular, rank-deficient, or non-positive-definite solves may use the
  least-squares fallback. Sentinel tests ensure the fallback is not entered for
  CUDA OOM/illegal-memory errors.
- [MEDIUM][API/INFER][fixed] ordinary concordance returned `NaN` for no
  comparable pair while counting-process concordance returned `0.5`. Both now
  use the existing neutral `0.5` convention. Score-test availability and a
  singular-null-information reason are exposed explicitly, while device errors
  are not converted to `NaN`.
- [LOW][MAINT][fixed] removed the unused counting-solver `delta_norm`, removed
  the obsolete feature-offset dispatch heuristic and unreachable public fit
  branches, and kept the legacy numerical primitives only for private
  compatibility tests.

Local evidence: the focused review file passed **34 tests with 4 optional
backend skips**; the Cox core/phase/CV matrix passed **101 tests with 15
optional-backend skips**. The complete CPU gate passed **1359 tests**, with
357 optional-backend skips and 43 marker deselections. A warm CPU smoke at continuous
event times completed `n=500`, `1000`, and `5000`, `p=4` ordinary fits in
0.0184, 0.0311, and 0.1447 seconds, confirming the stable dispatch does not use
the quadratic dense risk-set reference.

The exact clean commit `0214e701c68f12be15dddaad4667ce519b491898`
passed **41 focused physical-P100 tests** and the expanded three-backend Cox
matrix passed **460 tests**, both with zero failures. At `n=4096`, `p=12`, one
excluded warmup and three synchronized repeats, continuous Breslow medians were
0.1003/0.0367/0.0373 seconds and continuous Efron medians were
0.2316/0.0501/0.0488 seconds for NumPy/CuPy/Torch. Heavy-ties Breslow medians
were 0.0234/0.0184/0.0187 seconds and Efron medians were
0.1858/0.0214/0.0198 seconds. Every run converged and stayed finite; every
backend/tie combination also passed the centered `[-1000,0,1000]` nonzero-init
case. The machine-readable artifact is
`results/benchmark_frontend_sources/coxph_stability_resource_pr80_20260727.json`.

## Validation Evidence

- Final follow-up local Cox/survival matrix: **253 passed, 90 optional GPU
  skips, 0 failed**; core/static contracts add **15 passed**.
- Final physical-P100 follow-up matrix under
  `STATGPU_REQUIRE_PHYSICAL_GPU=1`: **343 passed, 0 failed** in 22.30 seconds,
  including NumPy, CuPy CUDA, and Torch CUDA execution.
- Final maintained delayed-entry+strata and strata-count artifacts: status
  `complete`, zero gate failures, exact source/benchmark hashes, synchronized
  GPU timing, finite/converged StatGPU repeats, and explicit R timeout states.
- Final documentation contracts: **122 files passed**; deterministic link check:
  **0 affected files**.
- `pytest` survival core target: **143 passed, 14 skipped, 0 failed** (157 total).
- Legacy `dev/tests/test_cox.py`: **8 passed, 4 skipped, 0 failed**.
- Penalized/PR79 compatibility run: **103 passed, 25 skipped** before the four
  stale capability assertions were corrected; all four corrected CPU nodes then
  passed. The remaining 11 failures were CuPy driver-initialization failures,
  not assertion or numerical failures.
- `dev/tests/test_core_contracts.py`: **7 passed, 0 failed**.
- Documentation contracts: **122 files passed**; deterministic link check:
  **0 affected files**.
- Current stratified-Exact local affected matrix: **162 passed, 26 skipped, 0
  failed**. The complete local suite reached **1272 passed, 298 skipped, 0
  failed** with four unrelated/pre-existing warnings.
- Current physical-P100 risk-set and Cox public-API matrix: **103 passed, 0
  failed** on Python 3.9.16, CuPy 13.6.0, and Torch 2.0.0+cu117.
- The synchronized three-stratum Exact benchmark (`p=4`, full fit plus
  inference) measured R/NumPy/CuPy/Torch medians of
  0.0180/0.0143/0.1742/0.0747 s at `n=160`,
  0.258/0.2263/0.2181/0.1341 s at `n=15,360`, and
  1.118/0.9874/0.2285/0.1384 s at `n=61,440`. CuPy and Torch are 4.89x and
  8.08x faster than R at the largest size; the artifact reports zero alignment
  gate failures and hashes matching the exact source and benchmark files.
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

No source-code or numerical gate remains for the reviewed PR #80 scope. External
R alignment closes the
independent-implementation accuracy gate, but timings remain shape-specific.
Both GPU backends are faster than R from the measured `n=15,360` ordinary
right-censored case through `n=122,880`; Torch is the fastest measured backend
on that low-dimensional large-sample shape. Small GPU fits remain launch-bound,
and wide Torch moment tensors keep the native scan. Large individual tie blocks
remain combinatorial. Ordinary multi-stratum Exact fits use one segmented prefix
DP; delayed-entry fits use either a memory-gated global GPU batch or bounded
per-stratum kernels. Score-residual requests and shapes rejected by numerical or
memory gates retain the backend-native normalized reference. These are explicit
evidence boundaries rather than failed gates.

Repository-hosted CI still lacks a CUDA runner. The maintained physical-GPU gate
is ready and passes under `STATGPU_REQUIRE_PHYSICAL_GPU=1`, but making it nightly
or required needs repository-level runner/credential provisioning; no
unconfigured self-hosted job was added to this PR.

## Public Boundary and Extreme Single-Group Follow-up

Impact classification: backend=`three-backend`; survival objective=Breslow/Efron
counting process; CV=`CoxPHCV`; inference=`compatibility contract`; formula=
unchanged; performance/memory=`active`; documentation=`active`; validation tier=
`remote-full`.

- [HIGH][TEST][fixed] `dev/tests/test_pr80_fit_boundary.py:13` - The new CuPy
  boundary tests called `getDeviceCount()` without handling an installed CuPy
  package paired with an unavailable driver.
  Impact: a legitimate CPU-only validation environment failed instead of
  skipping physical-CUDA cases.
  Fix: CuPy runtime failures now produce explicit skips; the same robustness was
  added to adjacent PR79/PR80 test helpers.
  Evidence: the full no-CuPy CPU tree passed **1391 tests**, with 412 optional
  skips and zero failures.
- [MEDIUM][API][fixed] `docs/en/models/coxph.md:154` -
  `inference_mode="approx"` was documented as selecting a legacy approximate
  Efron sandwich although the unified public fit always uses exact
  counting-process residuals.
  Impact: users could infer an algorithm choice that no longer occurs.
  Fix: `approx` remains a backward-compatible alias; both modes use exact
  inference and successful fits report `inference_approximate_=False`.
  Evidence: existing strict/approx provenance regressions and the expanded
  physical-GPU matrix passed.
- [MEDIUM][DOC][fixed] `docs/en/models/coxph.md:47` - The strata text described
  the low-level signed-int64 contract as if it applied to public estimators.
  Impact: documented support was narrower than actual `CoxPH`/`CoxPHCV`
  factorization of host strings/objects and finite backend numeric labels.
  Fix: public factorization and low-level numeric-code contracts are now
  distinguished in English and Chinese.
  Evidence: string-host and fractional-device strata regressions remain green.
- [MEDIUM][PERF][fixed] `statgpu/survival/_risk_sets.py:298` - A dense
  Breslow/Efron delayed-entry batch was clamped to at least one failure group,
  so an extreme single stratum could exceed the intended temporary-workspace
  ceiling.
  Impact: the selected GPU backend could OOM on a statistically valid, very
  large single risk set.
  Fix: `STATGPU_COX_GROUP_MAX_BYTES` (512 MiB default) now estimates live masks,
  weights, residual buffers, and second moments before allocation. Oversized
  groups use a stable multi-pass row-streaming moment calculation on the same
  backend.
  Evidence: forced 256-byte Torch CPU parity covers Breslow/Efron likelihood,
  score, information, and residuals; the P100 audit forced 4096 bytes at
  `n=8192`, `p=3`, with maximum information differences of `1.33e-14` for
  CuPy and `1.20e-14` for Torch.

Physical validation used the clean detached source commit
`16695feec8d4187b591d8a24d8977de543fd33c3` on a Tesla
P100-SXM2-16GB with Python 3.9.16, CuPy 13.6.0, and Torch 2.0.0+cu117.
The focused boundary matrix passed **85 tests**; the expanded Cox,
counting-process, CV, and penalized matrix passed **504 tests**. The
machine-readable artifact has `gate_failures=[]`, exact source hashes, commands,
timings, and device metadata:
`results/benchmark_frontend_sources/coxph_boundary_workspace_pr80_20260728.json`.

Exit status: `COMPLETE`. No unresolved CRITICAL/HIGH finding remains in this
follow-up. Repository-hosted CUDA CI remains an infrastructure item; the
maintained physical-GPU runner and artifact close the PR-level evidence gate.

## Parameter and Prepared-Capability Follow-up

Impact classification: backend=`three-backend`; survival objective=
`ordinary Breslow/Efron`; CV=`CoxPHCV`; inference=`unchanged`; formula=
`unchanged`; performance/memory=`active`; public API=`parameter stability`;
validation tier=`local-full`.

- [LOW][API/MAINT][fixed] `CoxPH.set_params()` validated then lowercased choice
  controls and converted penalties to Python floats. Constructor and fit had
  already moved to representation-stable public parameters plus immutable
  active controls. The setter now validates without rewriting the caller's
  valid object; `_CoxFitControls` remains the sole computational normalization
  boundary.
- [LOW][PERF][fixed] Strict reusable metadata validation reconstructed and
  compared a complete centered-and-sorted design for every CV candidate.
  Hashing would retain the O(np) scan, and arbitrary backend arrays have no
  reliable mutation generation counter. CV now creates a distinct
  `_PreparedImmutableFoldRightCensoredCox` capability for its privately owned
  fold arrays, so the complete penalty path skips both the scan and temporary
  matrix. Direct low-level state remains strict and still rejects changed or
  in-place-mutated inputs.
- [LOW][MAINT/EXT][fixed] The canonical call chain previously combined
  `right_censored_fast_path`, `right_censored_prepared`, and
  `_inputs_prepared`. Public dispatch now supplies either
  `_PreparedCountingProcessInputs` or
  `_PreparedOrdinaryRightCensoredState`; the concrete type determines the
  canonical objective path. A direct low-level prepared object also selects
  the fast path without a second boolean, while explicit fast-path requests
  remain backward compatible.

Targeted evidence covers public parameter representation, strict direct-state
mutation rejection, CV reuse with the full content validator replaced by a
fail-fast sentinel, direct type-selected fast-path parity, and duplicate input
normalization prevention across the Cox boundary.

Local validation passed **1509 tests**, with 455 optional-backend skips and 10
expected warnings. Pyflakes, compileall, `git diff --check`, benchmark CLI
loading, deterministic documentation links, and all **122 documentation
contracts** also pass. Exact clean source commit
`94b1a4be2c87416275e247eb8bff245b478cef8d` was then validated in remote
`myconda` on a Tesla P100-SXM2-16GB. CuPy 13.6.0 and Torch 2.0.0+cu117 each
passed all 10 structured cases; both recorded zero strict content-validation
calls for internally owned ordinary GPU CV folds and stable `set_params()`
representation. The physical targeted suite passed **321 tests** with 5
expected warnings, all 29 recorded Git-blob hashes match, and
`gate_failures=[]`. The schema-9 artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260729_schema9.json`
with SHA-256
`c0971df86347f8baf4350f8ba4500e07b94b8f6b059dc5e68e325655941b8fc2`.
