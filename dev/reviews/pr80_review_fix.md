# PR #80 Review-Fix Report

> Review date: 2026-07-28<br>
> Latest follow-up: 2026-08-03<br>
> Original PR head reviewed: `d6f798c1834fd6318c8257eed334f84a198fa8ad`<br>
> Performance-fix base: `ad3c0026eb682ac6394369a3318e9fb806e631b8`<br>
> Current risk-set SHA-256: `eee6900332526d5e68815e46d6d43a0f52e981760b724c10f98740fc56eeb3da`<br>
> Current Cox-loss SHA-256: `7220b6de7ebc0a72e0468b86e56617252eff1722339b9bd1c937976a1f41a7ea`<br>
> Current FISTA-LLA SHA-256: `3c9a665d0d46bebc32c6e43dbd2f777d989fe09114f73a2c7ae1e9bdb1642536`<br>
> Current penalized-fit mixin SHA-256: `56fcaa3667afc27935a73a363e77ca940560ce9beb3019b809c2544998b6062d`<br>
> Current penalized-Cox estimator SHA-256: `8349b9a9a3d80f254db06bdd2e7601aa68c1d36b83e112973fc85ef8afa3ea55`<br>
> Current shared-CV boundary SHA-256: `c5cff1c47d78c34a491007386c6412ced9250bc006c8f89ce4aca776af63e1cc`<br>
> Current penalized-CV orchestration SHA-256: `ac19fa0dd0754872f0f86ddd1fb2f4433222895b17aad92847a54cd8c5b5763c`<br>
> Current penalized-Cox CV SHA-256: `d9ca5923deb07452b6e2c158c0a3d0808894f3376a1a96cdb928333e6ac4c151`<br>
> Current canonical-Cox CV SHA-256: `98b9ba1a0274381f93b34ce09165d52021d195bb3cabc762648df05aa822e810`<br>
> Current schema-19 runner SHA-256: `bf2b12d5aaf5de1af18e41ee2f13b91457477f8b1b91436551351d066e914f0d`<br>
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
> Penalized-Cox CV/backend artifact source commit: `d688f760d8a0678c3c52c657a50178dad1b5ab3d`<br>
> Penalized-Cox CV/backend artifact SHA-256: `f0b47df704d2a0895cd1d66019c8676ff8a525d0f85e827d90ba816ad02b4837`<br>
> Penalized-Cox fold/grid artifact source commit: `f9e974b33c080c36a1a0cf1ca3508baca09f4939`<br>
> Penalized-Cox fold/grid artifact SHA-256: `e3ef1327b97755ebf1ea98482d7e274797a223aadff89842f1cb5505e67dfd7b`<br>
> Actual-fold auto-device artifact source commit: `a2d6a97d092d51a506421b67eea90fa71b5f8ac4`<br>
> Actual-fold auto-device artifact SHA-256: `2a70bac745e6114fce9c0f548538f54b53c8749f2c1df735b48e63169e19cde8`<br>
> Evaluable-fold routing artifact source commit: `0bc131767bef1eeec45805073431e666f690b78c`<br>
> Evaluable-fold routing artifact SHA-256: `4cc0cfb896d472cca601963f2cb6e86c6e1c5d9925fcba321df2f41942f2962c`<br>
> Original merge base: `a4879fb` (0.2.1 line)<br>
> Compatibility target: `origin/master` at `7ccf616` (0.2.2 line)<br>
> Status: `COMPLETE`; scalar/evaluable-fold routing passes local-full and schema-19 exact-source P100 validation at tier `remote-full`

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
| Backends | NumPy/CuPy/Torch fit, prediction, CV selection, operational fallback, and evaluable-fold workload | fixed; local-full and schema-19 P100 validation pass |
| Cross-validation | scalar-response and Cox custom-fold routing, plus canonical/penalized Cox selection contracts | fixed; local-full and schema-19 P100 validation pass |
| Compatibility | 0.2.1 PR head against 0.2.2 and PR #79 contracts | fixed |
| Benchmark evidence | synchronization, transfer scope, source version, schema, Exact scaling, R external alignment | historical artifacts remain scoped; schema-19 exact-source evidence passes |
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
  `_PreparedCVOwnedRightCensoredCox` trusted capability for its privately owned
  fold arrays, so the complete penalty path skips both the scan and temporary
  matrix. The name and documentation explicitly state that backend arrays are
  mutable and safety follows from private CV ownership. Direct low-level state
  remains strict and still rejects changed or in-place-mutated inputs.
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
Evidence commit `41e4040702a16d98341b913c3c62d5060f916915` is pushed, and all
seven hosted jobs passed in GitHub Actions run `30440782646`.

## Strict Robust-Inference Unit Follow-up

Impact classification: backend=`NumPy/CuPy/Torch`; survival objective=
`Breslow/Efron inference`; CV=`final refit inherits contract`; inference=
`HC0/HC1/cluster`; formula=`unchanged`; performance=`negligible pre-meat gate`;
validation tier=`remote-full`.

- [MEDIUM][BUG/INFERENCE][fixed] Robust covariance previously formed sandwich
  meat even when aggregation left a single independent subject or cluster, and
  HC1 silently replaced `n_units - n_features <= 0` with a denominator of one.
  A shared strict gate now requires at least two independent units for HC0,
  HC1, and cluster covariance, and requires `n_units > n_features` for HC1.
  The finite-unit multiplier is exactly
  `n_units / (n_units - n_features)`; failures remain inference errors and
  transactionally clear public fitted state.
- [MEDIUM][BUG/INFERENCE][fixed] Covariance diagonals were unconditionally
  clipped at zero. The shared inference helper now uses a scale-aware roundoff
  tolerance, rejects materially negative entries, and rejects non-positive
  robust marginal variances instead of publishing zero standard errors and
  misleading extreme significance.
- [LOW][VALIDATION][fixed] The cluster benchmark labelled an ordinary
  statsmodels model-based fit as HC1 and never ran its R helper. Statsmodels HC1
  is now explicitly `unsupported`; the R path executes
  `survival::coxph(robust=TRUE)` and applies the documented HC1 correction.
  Every result records the independent-unit count, covariance contract,
  correction, and unsupported reason, and JSON output replaces non-finite
  placeholders with `null`. A dynamically non-finite PHReg coefficient,
  standard-error, or p-value vector is likewise labelled unsupported rather
  than presented as external inference evidence.
- [LOW][MAINT/EXT][fixed] The trusted fold capability is now named
  `_PreparedCVOwnedRightCensoredCox`. Its documentation states that contained
  backend arrays and loss caches are structurally mutable and that bypassing
  content validation is safe only under private CV ownership for the complete
  penalty path.

The complete local suite passes 1519 tests with 467 optional-backend skips and
10 expected warnings. The maintained physical runner is schema 10 and adds direct CuPy/Torch
cases for one cluster, one subject, `n_units == n_features`, and the valid
`n_units == n_features + 1` correction ratio. Exact-source P100 JSON and the R
comparison were then executed from clean detached commit
`4570b9dca4cb771edfb1c29efb564c0e5340227f` in remote `myconda` on a Tesla
P100-SXM2-16GB. CuPy 13.6.0 and Torch 2.0.0+cu117 each passed all 11 structured
cases; the targeted matrix passed 343 tests with 5 expected warnings. All 31
Git-blob source hashes match, `source_clean=true`, and `gate_failures=[]`.

The external `n=3000`, `p=10` Breslow and Efron artifacts record 3000 HC1
independent units and the exact correction `1.0033444816053512`, plus 120
cluster units. R HC1 maximum coefficient/SE/p-value differences are
`5.55e-16`/`1.39e-16`/`8.00e-19`; R cluster differences are
`5.55e-16`/`1.32e-16`/`2.22e-16` for both ties methods. Statsmodels HC1 is
explicitly unsupported, and the remote PHReg cluster result is likewise marked
unsupported because its SE and p-value vectors are non-finite.

Machine-readable evidence:

- `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260729_schema10.json`
  (SHA-256 `9bd141082fad06151e57952f537e5afff2306d05642dd84e9b27e8f20b501fa0`);
- `results/benchmark_frontend_sources/coxph_robust_inference_breslow_pr80_20260729_schema10.json`
  (SHA-256 `a7667402371aac7ca7cd3ef128dbdac66bce22cdf12504439682b4f97cade0ed`);
- `results/benchmark_frontend_sources/coxph_robust_inference_efron_pr80_20260729_schema10.json`
  (SHA-256 `53f43f7fc4cc2d3fd29443aff318ba07c7b3dc3c1cf8931db593c1546fda74c2`).

Evidence commit `8cb02c0e782b8719f86efea172059f5e801ab685` is pushed. All seven
hosted jobs (`docs-contracts`, `static-contracts`, `full-cpu-suite`, and the
Python 3.9–3.12 regression matrix) passed in GitHub Actions run `30451833466`;
PR #80 reported `mergeable=true` and `mergeable_state=clean`.

## Joint Robust-Wald and External-Validation Follow-up

Impact classification: backend=`NumPy/CuPy/Torch`; inference=
`HC0/HC1/cluster joint Wald`; marginal inference=`preserved`; summary=
`robust/classical labels`; benchmark=`R/statsmodels strict`; validation tier=
`remote-full`.

- [MEDIUM][BUG/INFERENCE][fixed] A robust covariance with positive diagonal but
  deficient full-parameter rank could reach `np.linalg.solve`, producing a
  bare NaN or unstable finite Wald statistic. `_joint_wald_from_covariance()`
  now symmetrizes the covariance and applies a relative, scale-aware
  eigenvalue rank threshold before solving. Rank deficiency sets
  `wald_test_available_=False`, records the failure reason, and leaves valid
  coefficient SE/z/p/CI and the fitted model intact. CV propagates the final
  refit's availability metadata.
- [LOW][INFERENCE/DOC][fixed] Cox summaries now label likelihood-ratio and score
  tests as classical model-based tests, and label the covariance-dependent
  joint test as robust or classical Wald. An unavailable robust Wald is printed
  with its reason rather than a formatted `nan` result.
- [LOW][VALIDATION][fixed] R vectors now require exact feature length and finite
  values before a result is supported; `safe_diff()` rejects unequal shapes
  instead of truncating. R `coxph.control()` and statsmodels Newton fits receive
  explicit `max_iter`/`tol`, and every JSON row records those solver controls.

Focused NumPy tests cover two-cluster `p=3`, subject-HC0 with `G=p`, full-rank
`G=p+1`, summary output, CV propagation, near-singular helper behavior,
truncated/non-finite R output, and comparison-shape rejection. Schema 11 extends
the maintained physical CuPy/Torch case with the same rank-deficient joint-Wald
and summary contracts. The complete local suite passes 1525 tests with 471
optional-backend skips and 10 expected warnings.

Exact clean detached source commit
`f7215093c342c296ac3a1299117d8aea7baa33e1` passed schema 11 in remote
`myconda` on a Tesla P100-SXM2-16GB. CuPy 13.6.0 and Torch 2.0.0+cu117
each passed all 11 structured cases, including rank-deficient cluster and
subject-HC0 joint-Wald gates; the targeted physical matrix passed 353 tests
with 5 expected warnings. All 31 Git-blob hashes match, `source_clean=true`,
and `gate_failures=[]`.

The exact-source `n=3000`, `p=10` R comparison passed for Breslow and
Efron with aligned Newton `max_iter=80` and `tol=1e-8`. R HC1 maximum
coefficient/SE/p-value differences are
`5.55e-16`/`1.39e-16`/`8.00e-19`; R cluster differences are
`5.55e-16`/`1.32e-16`/`2.22e-16` for both ties methods.

Machine-readable evidence:

- `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260729_schema11.json`
  (SHA-256 `080ee60a7bf7e4e1a073698e5316a0b06b42c4ce7fec907af6845a2e31349c4e`);
- `results/benchmark_frontend_sources/coxph_robust_inference_breslow_pr80_20260729_schema11.json`
  (SHA-256 `8131b9f2e06ef7c08f75ba1c0b032ca1e5d1b53caacdb02fee1b01bc3c37744d`);
- `results/benchmark_frontend_sources/coxph_robust_inference_efron_pr80_20260729_schema11.json`
  (SHA-256 `b95eb0475a50e9e339d6f5cb337c9ce9557683c18ae85b6b0ef0fa457814a728`).

Evidence commit `d61d4f26dbe03960cc6cf47fd92c82d97cecaddb` is pushed. All seven
hosted jobs (`docs-contracts`, `static-contracts`, `full-cpu-suite`, and the
Python 3.9–3.12 regression matrix) passed in GitHub Actions run `30461628851`;
PR #80 reported `mergeable=true` and `mergeable_state=clean`.

## Strict Covariance-PSD Follow-up

Impact classification: backend=`NumPy/CuPy/Torch`; inference=`Cox`;
CV=`strict final-refit propagation`; objective=
`unchanged`; formula=`unchanged`; benchmark=`external failure schema only`;
performance=`one p-by-p eigendecomposition reused`; validation tier=
`remote-full`; exact-source physical GPU completed in schema 12.

- [MEDIUM][BUG/INFERENCE][fixed] Positive covariance diagonals could previously
  publish SE/z/p/CI even when the complete covariance had a materially negative
  eigenvalue. The shared classifier now distinguishes positive definite,
  rank-deficient PSD, and materially indefinite spectra. Indefinite covariance
  raises strict `RuntimeError` before any inference result is published, and
  Cox/CoxPHCV fit boundaries clear state transactionally. PSD rank deficiency
  still preserves valid marginal inference and marks only joint Wald
  unavailable; roundoff-level negative eigenvalues follow that PSD path.
- [LOW][MAINT/REUSE][deferred] Spectrum validation, marginal standard errors,
  and joint-Wald computation now live in
  `statgpu/inference/_covariance.py`, and Cox reuses that policy with one
  eigendecomposition. Generic sandwich migration remains separate because its
  GLM/penalized refit boundaries first need transactional state cleanup; doing
  only the numerical swap here could create stale fitted state after a new
  strict failure.
- [LOW][VALIDATION][fixed] Unsupported statsmodels/R benchmark rows now use
  `covariance_contract="unsupported"` and separately retain
  `requested_covariance_contract` plus `unsupported_reason`; successful rows
  continue to report the actual contract.

Focused local tests cover the three spectrum classes, Cox state cleanup,
CoxPHCV final-refit cleanup, external failure metadata, and NumPy/CuPy/Torch
routing. The complete local suite passes `1528 passed, 473 skipped`, with 10
expected warnings.

Exact clean detached source commit
`3e4d9bd3159ea329c16bd761197e3ad371f64893` passed schema 12 in remote
`myconda` on a Tesla P100-SXM2-16GB. CuPy 13.6.0 and Torch 2.0.0+cu117 each
passed all 11 structured cases, including materially indefinite Cox/CoxPHCV
strict-failure and state-cleanup gates; the targeted physical matrix passed
358 tests with 5 expected warnings. All 32 Git-blob hashes match,
`source_clean=true`, and `gate_failures=[]`.

Machine-readable evidence:

- `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260730_schema12.json`
  (SHA-256 `26ae5d47c3c5f9e447c4350ef7c599e06b8ee3fd8f56890a00c1c06d7f9b8b12`).

## Single-Stratum Prediction and Stop-Provenance Follow-up

Impact classification: backend=`NumPy/CuPy/Torch`; public API=
`CoxPH/CoxPHCV prediction and convergence diagnostics`; objective/inference=
`unchanged`; formula=`unchanged`; documentation=`EN/CN synchronized`;
validation tier=`remote-full`.

- [MEDIUM][BUG/API][fixed] An explicitly stratified fit with only one observed
  stratum previously bypassed prediction-label validation because
  `predict_survival()` inferred stratification from the number of stored
  baselines. The public boundary now uses explicit fitted state: every
  explicitly stratified fit requires one known label per prediction row,
  including the single-stratum case. Missing and unseen labels fail before a
  baseline is selected; the delegated `CoxPHCV` path inherits the same rule.
- [MEDIUM][DOC][fixed] The EN/CN Cox model pages no longer claim that schema-6
  evidence is pending while the repository contains newer physical evidence.
  Each page now presents one concise, commit-pinned schema-13 table with hardware,
  software, test counts, source hashes, gate failures, scope, and exclusions.
  Detailed historical timing and review chronology remain in this developer
  report rather than accumulating on the user-facing model page.
- [MEDIUM][API/DOC][fixed] `termination_reason_` remains the intentionally
  interpreted three-category user result. The new
  `optimization_stop_reason_` exposes the raw solver exit, including
  `max_iter`, and is reset transactionally, copied by `CoxPHCV`, and printed by
  `summary()` so warnings and fitted diagnostics can be reconciled.
- [REVIEW][remote delta][clean] Remote HEAD `488ab5b0dc144146e4b0274fb15ac8d9c7848ae0`
  adds Cox architecture and estimator-selection documentation only. The
  updated ownership descriptions are consistent with the canonical risk-set,
  inference, CV/refit, and penalized-estimator boundaries; no additional code
  correctness finding was identified in that delta.

Regression coverage includes direct NumPy/CuPy/Torch single-stratum prediction
cases, the CPU `CoxPHCV` delegated boundary, converged/line-search/max-iteration
raw-stop publication, summary output, and failed-refit cleanup. The focused
matrix passes `137 passed, 23 skipped`; the complete local tree passes
`1533 passed, 479 skipped`, with 11 expected warnings. Documentation links,
the maintained 122-file documentation contract, full package/dev compileall,
changed-file pyflakes, and `git diff --check` pass. Local ruff is unavailable;
the hosted static-contract job installs and executes it.

Exact clean detached source commit
`a7655904ea05fd9ce700d35832c44f90b0176251` passed schema 13 in remote
`myconda` on a Tesla P100-SXM2-16GB. CuPy 13.6.0 and Torch 2.0.0+cu117 each
passed all 11 structured cases, including direct and delegated single-stratum
label rejection/acceptance and interpreted/raw stop provenance. The targeted
physical matrix passed 432 tests with 7 expected warnings. All 34 recorded
Git-blob hashes independently match the source commit, `source_clean=true`,
and `gate_failures=[]`.

Machine-readable evidence:

- `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260730_schema13.json`
  (SHA-256 `799d439ceae1b25b582b5180573c390ada9438a86c7045dc3fb538611b1ac474`).

Evidence commit `28cc5857545951e492a652fbaf8c514ab58e1f5a` is pushed. All
seven hosted jobs (`docs-contracts`, `static-contracts`, `full-cpu-suite`, and
the Python 3.9-3.12 regression matrix) passed in GitHub Actions run
`30517578257`; PR #80 reported `mergeable=true` and
`mergeable_state=clean`.

## Fixed-Penalty Inference and Shared Strata-Scoring Follow-up

Impact classification: backend=`NumPy/CuPy/Torch`; public API=
`CoxPH/CoxPHCV inference, summary, scoring, and survival prediction`;
objective/optimizer=`unchanged`; inference=`fixed-penalty nonrobust covariance`;
formula=`unchanged`; documentation=`EN/CN synchronized`; validation tier=
`remote-full`; exact-source physical refresh completed in schema 14.

- [MEDIUM][BUG/INFERENCE][fixed] The L2 solver correctly optimized
  `loglik(beta) - penalty * ||beta||^2`, but positive-penalty nonrobust
  inference published `A^-1`, where `A=J+2*penalty*I_p`, as if it were a
  frequentist sampling covariance. That matrix is a penalized curvature or
  Laplace-style quantity. The fixed-penalty frequentist estimating-equation
  covariance is now `A^-1 J A^-1`, because the deterministic penalty changes
  the bread but adds no sampling variation to the unpenalized Cox score meat.
  Robust penalized paths already had the corresponding `bread @ meat @ bread`
  structure and retain it.
- [CONTRACT][decision] Three remedies were compared. Treating `A^-1` as a
  Bayesian posterior covariance would require a prior-scale contract, credible
  interval naming, and removal of frequentist p/Wald outputs. Disabling all
  positive-penalty inference would break the default `CoxPHCV` final refit and
  would first require separating baseline computation from coefficient
  inference. The fixed-penalty sandwich was selected because it supplies the
  requested frequentist contract without either unrelated API break. It remains
  conditional on the chosen penalty and does not correct shrinkage bias or CV
  selection uncertainty.
- [API/DOC][fixed] Positive-penalty fits now report the concise
  `inference_method_="m_estimation"`, matching the result-method vocabulary of
  `PenalizedGLM`, while
  `inference_target_="penalized_estimating_equation"`,
  `penalty_conditioning_="fixed_penalty"`, and
  `penalty_selection_adjusted_=False`. `CoxPHCV` copies these fields from the
  final refit. Machine-readable inference metadata identifies bread, meat, and
  covariance convention. Only the naming pattern was reused: PenalizedGLM's
  current nonrobust `penalized_information` convention still publishes the
  curvature inverse and is not the statistical implementation used here; that
  broader inference-engine issue remains outside this Cox-scoped change.
  Classical likelihood-ratio, score, AIC, and BIC diagnostics are suppressed;
  summary labels the remaining fixed-penalty coefficient/Wald inference and
  states its limitations.
- [MEDIUM][BUG/API/BACKEND][fixed] `CoxPH.score()` previously mapped fitted
  strata labels before validating shape, so scalar and two-dimensional inputs
  leaked backend/Python `TypeError`s. `score()` and `predict_survival()` now
  reuse `_encode_prediction_strata()`, which validates `(n_samples,)`, maps
  known training labels, and emits backend-independent `ValueError`s for scalar,
  two-dimensional, wrong-length, and unseen labels.

Deterministic regression coverage computes the unpenalized observed information
at the fitted coefficient and verifies the complete covariance identity for
Breslow/Efron/Exact and NumPy/CuPy/Torch. It also proves the result is not the old
curvature inverse, preserves the zero-penalty inverse-information contract,
checks `CoxPHCV` provenance propagation, checks summary/test suppression, and
exercises valid plus malformed scoring/prediction strata across all three
backends. The current local PR #80 targeted matrix passes **355 passed,
113 skipped**
with seven expected warnings; the complete local suite passes **1544 passed,
487 skipped** with eleven expected warnings. Package/dev compileall,
changed-file pyflakes, 122 maintained documentation contracts, deterministic
bilingual links, and `git diff --check` pass.

The schema-14 physical runner records the same covariance identity, metadata,
CV propagation, and strata errors for CuPy and Torch. Exact-source implementation
commit `0e48291de3c78dcfa6063e11947c43274e70c6c9` passed all 12/12 CuPy and
12/12 Torch cases on a Tesla P100 plus 468 physical-GPU targeted tests. The
machine-readable artifact records `source_clean=true`, 39/39 matching Git-blob
hashes, and `gate_failures=[]`.

## CoxPH Model-Documentation Completeness Follow-up

Impact classification: runtime=`unchanged`; public API=`documented only`;
objective/inference=`made explicit`; examples=`NumPy/CuPy/Torch CUDA`;
external validation=`R artifacts linked`; EN/CN=`synchronized`.

- [DOC][fixed] Both model pages now have named Objective Function / Estimating
  Equation sections, define the total partial-likelihood penalty scale, and
  separate fixed-penalty estimating-equation inference from Bayesian,
  debiased, and post-CV claims.
- [DOC][fixed] Deterministic NumPy, CuPy CUDA, and Torch CUDA fit/prediction
  examples share one data setup. The CV section includes both GPU backends and
  states that explicit device requests never silently fall back.
- [DOC][fixed] External R HC1/cluster evidence and exact-source physical-GPU
  evidence are separately scoped. A FAQ covers unavailable CUDA, missing
  baselines/strata, robust-unit gates, singular information, exponential range,
  nonconvergence, Exact workspace, and no-pair concordance.
- [DOC][fixed] Page dates are 2026-07-31 and the English reference ranges use
  Unicode en dashes instead of corrupted question marks.

## Canonical Cox Validator Contract Follow-up

Impact classification: production runtime=`unchanged`; canonical numerical
reference=`corrected`; delayed-entry convention=`start < t <= stop`;
fixed-penalty inference=`A^-1 J A^-1`; EN/CN=`synchronized`; physical GPU
source audit=`expanded`.

- [MEDIUM][VALIDATION/INFERENCE][fixed] The canonical PR79 final-state
  recomputation now keeps the unpenalized observed information `J` as the meat,
  adds `2 * penalty * I` only to the bread information `A`, and validates the
  fixed-penalty frequentist covariance `A^-1 J A^-1`. A solve-first,
  pseudoinverse-fallback policy matches the production contract without
  importing production inference code.
- [MEDIUM][VALIDATION/RISK-SET][fixed] Delayed-entry risk membership now uses
  the documented open-left boundary `entry < failure_time`. A deterministic
  valid interval with `entry == failure_time < stop` proves that the row is not
  admitted to that failure risk set.
- [TEST][fixed] The covariance regression uses an analytic two-feature
  information matrix derived independently from the validator. It asserts both
  `A^-1 J A^-1` parity and detectable disagreement with the former `A^-1`
  expectation, closing the circular expected-value gap in the previous smoke
  test.
- [DOC][fixed] The Cox model FAQ now distinguishes the single-explicit-stratum
  survival-prediction contract from scoring: prediction always needs labels
  after an explicitly stratified fit, whereas scoring only requires labels for
  a multi-stratum fit and validates any labels that are supplied.

Local evidence: the canonical accuracy-pipeline tests pass 17/17; the clean
implementation commit passes the CI-equivalent canonical smoke with 2/2
numerical/final-state checks and `gate_verdict="PASS"`; the expanded PR80
targeted matrix passes 355 tests with 113 expected GPU skips and seven expected
warnings. The complete CPU test tree passes 1,544 tests with 487 expected GPU
skips and eleven expected warnings. Documentation links, all 122 maintained documentation
contracts, compileall, changed-file pyflakes, and diff whitespace checks pass.

The schema-14 runner records 39 source files, including the canonical validator
and its accuracy-pipeline test, and executes 16 targeted test files. The final
clean implementation commit `0e48291de3c78dcfa6063e11947c43274e70c6c9`
passed all 12/12 CuPy and 12/12 Torch structured cases plus 468 targeted tests on
a Tesla P100-SXM2-16GB in remote `myconda`. The audited artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260731_schema14.json`;
all 39 source hashes match Git blobs and `gate_failures=[]`.

## Eventless-Stratum Survival Prediction Follow-up

Impact classification: runtime prediction=`corrected`; coefficient estimation,
risk-set objective, baseline construction, and inference=`unchanged`; public
result contract=`survival exactly one for a fitted stratum with no failures`;
EN/CN=`synchronized`; exact-source physical evidence=`schema 15 complete`.

- [MEDIUM][CORRECTNESS/PREDICTION][fixed] `cox_baseline_hazard()` intentionally
  stores empty arrays for a fitted stratum with no observed failures. The
  prediction consumer now distinguishes that valid zero-hazard state from a
  mismatched stored time/hazard shape. Empty knots leave the prefilled survival
  row exactly one without allocation; incompatible shapes still raise.
- [TEST][fixed] The pre-fix regression deterministically failed in both direct
  `CoxPH` and delegated `CoxPHCV` calls at the former empty-knot guard. The
  NumPy/CuPy/Torch matrix now covers explicit times, automatic union times,
  mixed eventful/eventless rows, CV delegation, finite output, exact ones, and
  continued rejection of corrupted baseline shapes.
- [VALIDATION][updated] The physical runner is schema 15 and adds a structured
  `eventless_stratum_survival` case for each GPU backend. It records the empty
  producer state, explicit/automatic/mixed prediction results, and CV
  delegation. The source and targeted-test manifests remain 39 and 16 files.
- [LOW][VALIDATION/MAINT][fixed] The machine-readable field formerly named
  `direct_backend_imports_absent` only inspected
  `_fit_counting_process_dispatch`; schema 15 narrows it to
  `dispatch_direct_backend_imports_absent` instead of implying a model-wide AST
  audit.
- [LOW][MAINT/EXT][deferred] Splitting prediction/baseline responsibilities out
  of `_cox.py`, validating CoxPHCV side-array dimensions before flattening,
  centralizing final-estimator state adoption, and moving label factorization
  into the backend layer remain worthwhile follow-ups. They are not required to
  correct this established baseline-state contract and are intentionally not
  mixed into the runtime patch.

Local evidence: the focused file passes 12 tests with 12 expected GPU skips;
the schema-targeted matrix passes 358 tests with 117 expected GPU skips and
seven expected warnings; the complete CPU tree passes 1,547 tests with 491
expected GPU skips and eleven expected warnings. Documentation links, all 122
maintained documentation contracts, package/dev compileall, changed-file
pyflakes, and `git diff --check` pass.

Exact-source implementation commit `0d33a4fa64e7bf023407c4f691d008995ae67493`
passed all 13/13 CuPy and 13/13 Torch structured cases plus 475 targeted tests
with seven expected warnings on a Tesla P100-SXM2-16GB in remote `myconda`.
The eventless case reports zero error from survival one for explicit times,
automatic times, mixed rows, and `CoxPHCV` delegation. The audited artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260801_schema15.json`
(SHA-256 `56386d7d0a51e73423939dacc0238a31bfdaa929f6e4f24cc3de73053a5e8ff0`);
all 39 source hashes match Git blobs, `source_clean=true`, and
`gate_failures=[]`.

## Penalized-Cox CV, Fitted-Backend, and Clone-Contract Follow-up

Impact classification: selected regularization=`correctness-critical`;
public API=`PenalizedGLM_CV, PenalizedCoxPHModel, CoxPH, CoxPHCV,
CompositePenalty`; backends=`NumPy/CuPy/Torch`; inference=`unchanged,
estimation-only for PenalizedGLM_CV`; formula=`unchanged`; exact-source physical
evidence=`schema 16 remote-full`.

### Capability decisions by public estimator family

The table uses the allowed capability values from the review skill. A canonical
L2 Cox CV result is not used as evidence for the separate penalized-model
family.

| Public family | Backend | CV | Inference | Formula | Benchmark |
|---|---|---|---|---|---|
| `CoxPH` | `three-backend` | `supported` through `CoxPHCV` for L2 | `supported` | `supported` | `required` |
| `CoxPHCV` | `three-backend` | `supported` | `supported` on final refit, conditional after selection | `not-formula-facing` | `required` |
| `PenalizedCoxPHModel` (L1/L2/ElasticNet/SCAD/MCP) | `three-backend` | `supported` through `PenalizedGLM_CV(loss="cox_ph")` | `estimation-only` | `supported` | `required` |
| `PenalizedGLM_CV(loss="cox_ph")` | `three-backend` | `supported` with strict held-out partial likelihood | `estimation-only` | `not-formula-facing` | `required` |
| `PenalizedGLM_CV` scalar-response families | `three-backend` | `supported` | `estimation-only` | `not-formula-facing` | `required` |

`CompositePenalty` is a supporting public penalty object rather than an
estimator family. Its applicable decision is constructor/clone compatibility;
Cox explicitly supports only the five validated simple penalty families above.

- [CRITICAL][CV/CORRECTNESS][fixed] Two remedies were compared. A hard
  rejection of `loss="cox_ph"` would prevent the false first-alpha selection,
  but would leave every public tunable penalized-Cox family without the CV
  capability required by the review contract. The selected implementation is a
  separate survival-aware branch: it preserves `(n, 2)` targets, forbids an
  intercept, scores unpenalized held-out Cox partial likelihood per row,
  requires finite evidence from every evaluable fold, hard-fails transactionally
  when no alpha is supported, and refits `PenalizedCoxPHModel`. L1, L2,
  ElasticNet, SCAD, and MCP share this path; `two_stage`, sample weights,
  dictionary targets, and post-selection coefficient inference are explicitly
  unsupported.
- [HIGH][BACKEND/API][fixed] A successful `CoxPH(device="auto")` fit
  records `_fitted_backend_name` and public `effective_device_`. Prediction and
  scoring construct that exact backend directly, so later global device changes
  cannot migrate an existing model. Failed refits clear both fields.
- [HIGH][SKLEARN/API][fixed] `CompositePenalty.get_params(deep=False)`
  now returns only its real constructor inputs, with component penalty objects
  intact. The default/deep call retains the descriptive serialization contract.
  The first schema-16 P100 run exposed an additional sklearn <=1.2 identity
  check because the constructor rebuilt `weights`; the final constructor
  preserves an already-normalized float tuple by identity while still
  normalizing ordinary user inputs. Direct reconstruction, legacy/current
  sklearn clone, and cloning an estimator that contains the composite pass.
- [MEDIUM][API/VALIDATION][fixed] `CoxPHCV` side arrays must have the
  exact public shape `(n_samples,)` before cache hashing, fold construction,
  label grouping, or candidate fitting. `(n, 1)` and `(1, n)` fail at the public
  boundary for time, event, entry, cluster, strata, and subject ID.
- [MEDIUM][MATRIX/REVIEW-CONTRACT][fixed] The per-family decision matrix above
  replaces the former area-level assertion that conflated canonical L2 CV with
  the complete penalized-Cox model family.

Focused local coverage passes 27 tests with 14 expected physical-GPU skips. The
17-file schema-targeted local matrix passes 385 tests with 131 expected GPU
skips and seven expected warnings; the complete CPU tree passes 1,574 tests
with 505 expected GPU skips and eleven expected warnings. Documentation links,
all 122 maintained documentation contracts, package/validation/benchmark
compileall, new-file/runner pyflakes, and `git diff --check` pass.

The schema-16 runner records 43 source files and all five penalized-Cox
penalties, complete-fold selection evidence, direct final-refit coefficient
parity, and fitted-backend pinning for both CuPy and Torch. Exact clean
implementation commit `d688f760d8a0678c3c52c657a50178dad1b5ab3d` passed all
14/14 CuPy and 14/14 Torch structured cases plus 516 targeted tests with seven
expected warnings on a Tesla P100-SXM2-16GB in remote `myconda`. The audited
artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260802_schema16.json`
(SHA-256 `f0b47df704d2a0895cd1d66019c8676ff8a525d0f85e827d90ba816ad02b4837`);
all 43 source hashes match Git blobs, `source_clean=true`, and
`gate_failures=[]`. That schema-16 follow-up is `COMPLETE` at validation tier
`remote-full`; the later runtime section below supersedes the report's current
overall status.

## Penalized-Cox CV Fold, Grid, and Auto-Device Follow-up

Impact classification: selected regularization=`correctness-critical`;
public API=`PenalizedGLM_CV(loss="cox_ph"), PenalizedCoxPHModel`;
backends=`NumPy/CuPy/Torch`; inference=`unchanged, estimation-only`;
formula=`unchanged`; exact-source physical evidence=`schema 17 remote-full`.

### Capability decisions by public family

| Public family | Backend | CV | Inference | Formula | Benchmark |
|---|---|---|---|---|---|
| `PenalizedCoxPHModel` L1/L2/ElasticNet/SCAD/MCP | `three-backend` | `supported` through survival-aware `PenalizedGLM_CV` | `estimation-only` | `supported` | `required` |
| `PenalizedGLM_CV(loss="cox_ph")` | `three-backend` | `supported` for general non-empty disjoint splits with strict finite evidence | `estimation-only` | `not-formula-facing` | `required` |
| Penalized Cox with `"none"`, `"null"`, or `""` | `three-backend` | `non-tunable` and rejected before candidate fitting | `estimation-only` through direct fit | `supported` on the direct model | `not-applicable` |
| Canonical `CoxPH` / `CoxPHCV` | `three-backend` | `supported` for canonical L2 selection | `supported` | `supported` / `not-formula-facing` | `required` |

- [CRITICAL][CV/CORRECTNESS/API][fixed] Custom fold indices were cast to
  `int64` before validation, so fractional floats, booleans, numeric strings,
  non-finite values, overflowing unsigned integers, and higher-dimensional
  arrays could silently change the requested split. The strict canonical-Cox
  policy is now a shared `_coerce_cv_indices()` utility and both Cox CV paths use
  it before candidate or backend work. Tests assert each malformed class fails
  transactionally and candidate fitting is never entered.
- [CRITICAL][CV/CORRECTNESS][fixed] ElasticNet automatic Cox grids formerly
  reused `||gradient L(0)||_inf` without accounting for its L1 mixing weight.
  For `l1_ratio=rho>0`, the first alpha is now the independent zero-model KKT
  boundary `||gradient L(0)||_inf / rho`. String penalties use the estimator
  ratio and `ElasticNetPenalty` objects use their own ratio. `rho=0` is pure L2
  and has no finite all-zero KKT threshold, so the raw zero-score norm remains
  only an explicit, machine-readable grid heuristic.
- [HIGH][BACKEND/FALLBACK][fixed] Two approaches were compared: importing
  CuPy, or querying the shared backend health contract. Import presence cannot
  establish a working CUDA driver/device, so auto CV now selects Torch or CuPy
  only when `get_backend(..., device="cuda").is_available()` succeeds. Large
  automatic searches fall back to CPU when neither backend is operational;
  explicit CUDA requests remain strict and propagate the backend failure.
- [MEDIUM][CV/API/MATRIX][fixed] No-penalty aliases always resolve to alpha zero
  and therefore cannot support a parameter-selection claim. The Cox CV boundary
  rejects them as non-tunable before any candidate fit and directs callers to a
  single direct `PenalizedCoxPHModel` fit.
- [MEDIUM][DOC/API][fixed] Two remedies were compared for custom splitters:
  restrict Cox documentation to partition-style K-fold, or support general
  disjoint train/validation pairs. The candidate/scoring loop has no statistical
  need for complement or exactly-once coverage, so the implementation now
  accepts forward `TimeSeriesSplit`, repeated holdout, and other non-empty
  disjoint designs. EN/CN guides document the exact index, event-support, and
  overlap contracts.

Focused local coverage passes 46 tests with 20 expected physical-GPU skips. The
17-file schema-targeted matrix passes 404 tests with 137 expected GPU skips and
seven expected warnings. The complete CPU tree passes 1,593 tests with 511
expected GPU skips and eleven expected warnings. Documentation links, all 122
maintained documentation contracts, package/validation/benchmark compileall,
changed new-path/runner pyflakes, benchmark CLI parsing, and `git diff --check`
pass. Local `ruff` is unavailable; the hosted static-contract job remains the
authoritative execution of its selected rules.

The schema-17 runner adds the shared CV source file to its 44-file exact-source
hash manifest and extends both physical GPU cases with independently recomputed
ElasticNet string/object KKT boundaries, the pure-L2 heuristic, and general
non-complementary disjoint folds. Exact clean implementation commit
`f9e974b33c080c36a1a0cf1ca3508baca09f4939` passed all 14/14 CuPy and 14/14
Torch structured cases plus 541 targeted tests with seven expected warnings on
a Tesla P100-SXM2-16GB in remote `myconda`. The audited artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260802_schema17.json`
(SHA-256 `e3ef1327b97755ebf1ea98482d7e274797a223aadff89842f1cb5505e67dfd7b`);
all 44 recorded hashes match the exact Git blobs, `source_clean=true`, and
`gate_failures=[]`. The string/object ElasticNet KKT boundaries and pure-L2
heuristic match independently recomputed values on both GPU backends, and each
records two general non-complementary disjoint splits. This follow-up is
`COMPLETE` at validation tier `remote-full`.

## Actual-Fold Auto-Device and Penalty-Documentation Follow-up

Impact classification: numerical result=`unchanged`; selected alpha=`unchanged`;
performance/backend placement=`affected`; public documentation=`corrected`;
backends=`NumPy/CuPy/Torch`; exact-source physical evidence=`schema 18 remote-full`.

- [MEDIUM][PERF/BACKEND][fixed] Auto-device fallback work formerly used
  `self.cv`, even after the survival-aware path had normalized a custom split
  generator. Two scopes were compared: patch only Cox, or make the shared
  device estimator accept the actual fold count. The shared fix is preferable:
  both Cox and scalar-response CV now materialize/normalize folds before device
  selection and pass `len(folds)`. Default/internal callers retain the
  constructor count only when no explicit count is supplied. Tests cover one
  custom holdout, four repeated folds, and both sides of the 100-million-work
  break-even while `cv=99` proves the constructor value is not reused.
- [MEDIUM][DOC/API][fixed] The module capability text now lists only the
  five public Cox penalties: L1, L2, ElasticNet, SCAD, and MCP. EN/CN generic
  alpha-grid text is restricted to scalar-response estimators and documents the
  Cox exception: user grids are not filtered or replaced; non-finite/negative
  values fail, SCAD/MCP require strictly positive alpha, and L1/L2/ElasticNet
  permit zero.

The schema-18 runner extends the physical penalized-Cox CV case with an
actual-fold workload gate: at `n*p=200,000` and 100 alphas, one normalized fold
must remain on CPU while five folds meet the 100-million-work threshold and
select operational Torch CUDA. Focused Cox-CV coverage passes 49 tests with 20
expected physical-GPU skips; the affected scalar-CV safety set passes 89 tests
with seven optional-backend skips. The 17-file schema-targeted matrix passes
407 tests with 137 expected GPU skips and seven expected warnings, while the
complete CPU tree passes 1,596 tests with 511 expected GPU skips and eleven
expected warnings. Documentation links, all 122 maintained documentation
contracts, package/validation/benchmark compileall, changed-path/runner
pyflakes, benchmark CLI parsing, and `git diff --check` pass. Exact clean
implementation commit `a2d6a97d092d51a506421b67eea90fa71b5f8ac4` then
passed all 14/14 CuPy and 14/14 Torch structured cases plus 544 targeted
tests with seven expected warnings on a Tesla P100-SXM2-16GB in remote
`myconda`. Both backend cases record one fold on CPU and five folds on
operational Torch at the exact break-even boundary. The audited artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260802_schema18.json`
(SHA-256 `2a70bac745e6114fce9c0f548538f54b53c8749f2c1df735b48e63169e19cde8`);
all 44 recorded hashes independently match the exact Git blobs,
`source_clean=true`, and `gate_failures=[]`. This follow-up is `COMPLETE` at
validation tier `remote-full`.

## Scalar-Response and Evaluable-Fold Device-Sizing Follow-up

Impact classification: numerical result=`unchanged`; selected alpha=`unchanged`;
backend placement and transfer cost=`affected`; public CV families=`scalar and
Cox`; documentation=`affected`; exact-source physical evidence=`schema 19
remote-full`.

### Capability decisions by touched public family

| Public family | Backend | CV | Inference | Formula | Benchmark |
|---|---|---|---|---|---|
| Scalar-response `PenalizedGLM_CV` | `three-backend` | `supported`; list/generator folds size device work after one materialization | `estimation-only` | `not-formula-facing` | `required` |
| `PenalizedGLM_CV(loss="cox_ph")` L1/L2/ElasticNet/SCAD/MCP | `three-backend` | `supported`; only event-supported folds enter candidate work and evidence counts | `estimation-only` | `not-formula-facing` | `required` |
| Direct `PenalizedCoxPHModel` | `three-backend` | supplied by the survival-aware CV family | `estimation-only` | `supported` | `required` through family CV |

- [HIGH][TEST/MATRIX][fixed] The shared scalar-response `_fit_standard()`
  path changed public backend routing but only Cox fit and a private selector
  had end-to-end coverage. New squared-error/L2 public-fit regressions set
  `cv=99`, exercise one and four custom folds as both lists and one-shot
  generators, capture the selector's `n_folds`, and assert the generator is
  consumed exactly once. The public `cv_results_` now records
  `device_sizing_fold_count` for auditable routing.
- [HIGH][BUG/BACKEND][fixed] Adjacent re-review found that scalar auto-
  device CV published `cv_selected_device_` but `_refit_best()` read the
  nonexistent `_cv_selected_device_`. The final refit could therefore resolve
  `device="auto"` again instead of using the CV-selected backend. The refit now
  reads the public fitted routing state, and an end-to-end regression forces a
  Torch selection without requiring CUDA, then proves that the same device
  reaches the refit estimator.
- [MEDIUM][PERF/BACKEND][fixed] Two policies were compared for Cox folds
  without training or validation events: reject the complete split design, or
  retain diagnostic skipping while excluding those folds from device work.
  Rejection would narrow the documented general-disjoint/repeated-split API and
  discard useful `failure_path` reasons. The selected policy therefore computes
  the one-time host event summary before auto-device resolution, rejects
  non-finite/non-binary event input before backend work, preserves each
  skipped fold, and passes only `n_effective_folds` to generic fallback sizing.
  A threshold regression proves five normalized folds with one evaluable fold
  remain on CPU even though sizing the same input by five folds would select
  Torch.
- [MEDIUM][DOC/BACKEND][fixed] EN/CN device tables no longer claim an
  unconditional CPU default. They document the 100-million aggregate-work
  fallback, operational Torch/CuPy ordering, scalar normalized-fold count, Cox
  evaluable-fold count, the SCAD/MCP continuation factor, and the precedence of
  empirical per-loss `n*p`/feature rules that do not use a fold multiplier.

Focused coverage passes 58 tests with 20 expected physical-GPU skips; the
affected scalar-response safety set passes 89 tests with seven optional-backend
skips. The 17-file schema-targeted matrix passes 416 tests with 137 expected GPU
skips and seven expected warnings, while the complete CPU tree passes 1,605
tests with 511 expected GPU skips and eleven expected warnings. Documentation
links, all 122 maintained documentation contracts, package/validation/benchmark
compileall, changed-path/runner pyflakes, benchmark CLI parsing, and
`git diff --check` pass. The schema-19 runner adds structured public-fit
routing evidence for scalar list
and one-shot-generator folds on both GPU backends, plus a Cox design with five
normalized folds and one evaluable fold. Exact clean implementation commit
`0bc131767bef1eeec45805073431e666f690b78c` passed all 14/14 CuPy and
14/14 Torch structured cases plus 553 targeted tests with seven expected
warnings on a Tesla P100-SXM2-16GB in remote `myconda`. Both backends record
scalar list count 1, one-shot-generator count 4 with one consumption, and Cox
normalized/evaluable/device-sizing counts 5/1/1. The audited artifact is
`results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260803_schema19.json`
(SHA-256 `4cc0cfb896d472cca601963f2cb6e86c6e1c5d9925fcba321df2f41942f2962c`);
all 44 recorded hashes independently match the exact Git blobs,
`source_clean=true`, and `gate_failures=[]`. This follow-up is `COMPLETE` at
validation tier `remote-full`.
