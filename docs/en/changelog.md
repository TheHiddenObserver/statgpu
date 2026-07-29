# Changelog

> Language: English<br>
> Last updated: 2026-07-29<br>
> This page: Changelog<br>
> Switch: [Chinese](../cn/changelog.md)

## 2026-07

### Fixed (2026-07-29) — PR #80 final follow-up

- Ordinary GPU Breslow/Efron fits now report their complete sorted time/event
  device-to-host transfer. `CoxPHCV` prepares the corresponding sorted design,
  failure groups, event indices, and Efron fractions once per fold for the
  complete selector invocation and reuses that immutable loss state across all
  staged penalty passes when the bounded fold cache fits its workspace gate.
  Larger staged workloads repeat preparation rather than retaining an
  unbounded multi-fold GPU cache. This replaces
  repeating target transfers and metadata construction for every candidate.
  Delayed-entry, strata, and subject fits likewise disclose complete retained
  side-vector transfers, while side-array-free paths avoid copying a synthetic
  all-zero start vector. Unused unique cluster/scoring labels now remain on the
  selected backend instead of being materialized on the host.
- Hazard-ratio outputs now share one strict numerical contract across `CoxPH`,
  `CoxPHCV`, and `PenalizedCoxPHModel`: finite log-risk outside the finite,
  positive float64 exponential range raises `FloatingPointError` rather than
  returning infinity, zero, or an estimator-specific clipped value. Raw
  log-risk remains available through `predict_risk_score()`. Ordinary
  unstratified survival prediction now retains the fitted centered log-baseline
  state instead of falling back to a direct `exp(X @ coef)` product.
- CV cache diagnostics now distinguish the immutable selection origin from the
  current invocation through `selection_cache_hit`,
  `selection_origin_device`, `requested_fit_device`, and per-call preparation
  and transfer counts, including separate preparation and physical vector-copy
  totals. Canonical fitted state uses one reset per public fit,
  and the public `CoxFitNumericalError` is exported from both `statgpu` and
  `statgpu.survival`.
- Reused right-censored loss state now verifies the current `X`, time, and
  event contents on the active backend before a low-level solve, so same-shape
  foreign data or in-place source mutation cannot combine stale coefficients
  with a new baseline. Packed CuPy/Torch `CoxPHCV` targets remain backend-native
  through column unpacking, making their full host transfer visible in CV
  provenance. Cox constructors preserve clone-sensitive inputs until fit-time
  normalization, while penalized prediction and scoring reuse `BackendBase`
  conversion and the shared Cox boolean/real-value validators.
- The schema-7 exact-source physical refresh passed 282 targeted tests and all
  18 CuPy/Torch case gates on a Tesla P100. Its machine-readable artifact
  records the clean source commit and 29 source hashes, plus direct gates for
  prepared-state content mismatch and packed-GPU-target transfer provenance.
- Penalized and canonical Cox prediction now share one backend-neutral matrix
  normalization contract: a one-dimensional input is one complete row for a
  multi-feature model or multiple observations for a one-feature model, while
  wrong feature counts and higher-rank inputs fail before backend matmul.
  The low-level right-censored fast path rejects nonzero entry times or
  multiple strata instead of mixing an ordinary objective with a different
  baseline definition. `CoxPH` and `CoxPHCV` now use immutable private active
  controls during fitting, so fit-time normalization no longer rewrites public
  constructor parameters. The maintained physical runner is advanced to
  schema 8 for an exact-source GPU refresh of these boundaries.

### Fixed (2026-07-27) — PR #80 follow-up review

- Penalized Cox SCAD/MCP now preprocesses, sorts, and transfers survival-group
  metadata once per fit. FISTA-LLA uses a gradient-only hot path, performs its
  finite/convergence transfer periodically, and releases loss-held training
  arrays before allocator cleanup.
- The trusted gradient uses cancellation-safe scaled direct first moments in
  adaptively bounded row blocks. This preserves both suffix denominators after
  a maximum predictor departs and signed first moments near `1e15`; it retains
  explicit predictor-range scalar checks instead of claiming a zero-sync path.
  A row block is capped at 65,536 rows and two million moment elements, so the
  removed signed-log scan cannot create an `O(n)` temporary workspace.
- FISTA-LLA counts every completed proximal update, including the converged
  update, and its per-alpha path records cumulative work accurately. GPU event
  validation transfers one two-boolean status vector instead of the packed
  target; valid host `uint64` strata are normalized before Torch 2.0 conversion.
  Complex `X`, time, event, start, stop, and coefficient inputs are rejected
  before any real-valued cast on NumPy, CuPy, and Torch.
- Every public `CoxPH.fit()` now uses the stable shared risk-set objective.
  Ordinary nonrobust Breslow/Efron fits use its bounded suffix-moment fast path,
  retaining near-linear row scaling while keeping finite objectives and
  gradients for centered predictors such as `[-1000, 0, 1000]` with a nonzero
  initial coefficient. Start-stop, strata, robust, and Exact cases retain the
  corresponding backend-native shared kernels.
- Explicit Breslow `(n, p, p)` Hessian buffers are gated by
  `STATGPU_BRESLOW_HESSIAN_MAX_BYTES` (512 MiB by default); CPU falls back to
  incremental grouped moments and CuPy to bounded grouped GEMM updates. CUDA
  OOM/runtime failures are no longer swallowed by the fused kernel or relabeled
  as singular information, and least-squares is attempted only for recognized
  singular/ill-conditioned solves.
- `CoxPH`, `CoxPHCV`, public scoring, and held-out partial likelihood all reject
  complex values before real conversion. A score test now exposes
  `score_test_available_` and `score_test_failure_reason_`; device failures
  propagate, while singular null information is recorded explicitly. Both
  ordinary and counting-process concordance return `0.5` when no comparable
  pair exists.

- `CoxPH(gpu_memory_cleanup=True)` now runs both allocator cleanup hooks after
  every public prediction and scoring call, including exceptional exits.
  `CoxPHCV` owns that cleanup at its outer public boundary and disables it on
  the delegated final estimator, so one CV prediction or score performs only
  one allocator flush and synchronization round.
  Summaries report the actual matrix or formula interface and the fitted
  counting-process, strata, subject, cluster, and ties metadata instead of a
  synthetic R call. Canonical Cox and final `CoxPHCV` refits now publish the
  shared `ParameterInferenceResult` contract and its parameter, z, p-value,
  and confidence-interval fields.
- `CoxPHCV` now reports full host transfers across the complete selection plus
  refit workflow, with separate CV/refit provenance. A dedicated candidate
  numerical exception lets CV exclude a non-finite penalty without swallowing
  input, CUDA, allocator, or programming errors. Fold strata are factorized and
  moved through the shared backend once per fold evaluation, while cluster and
  subject labels no longer enter candidate fits that compute neither inference
  nor concordance. Canonical `CoxPH` fitted state is initialized through one
  reset contract; historical risk-set caches now live only on the test adapter.
- Low-level concordance validates `subject_id` as finite, exactly integral
  int64 codes before conversion. Survival risk-set normalization reuses the
  shared backend array, scalar, zeros, eye, and integer-code helpers; public
  fit boundary handling is defined directly on the estimator rather than
  installed by an import-time adapter.
- The canonical `CoxPH` estimator and public dispatch remain in `_cox.py` and
  no longer inherit or import the historical mixin. Backend-specific
  information inversion is stateless in `_cox_inference.py`; inactive CPU,
  CuPy, and Torch reference kernels remain test-only through an explicit
  composition adapter in `_cox_legacy.py`, so optional legacy probes are not
  loaded by a public survival import.
- `CoxPHCV` now routes NumPy, CuPy, and Torch held-out Breslow, Efron, and Exact
  likelihoods through the shared counting-process objective. A stable NumPy
  log-likelihood-only specialization lives with the risk-set implementation,
  retaining the previous suffix-path performance without duplicating the
  statistical definition in the CV module. Formula side arrays use one
  backend-preserving alignment helper, and CV prediction documents its native
  NumPy/CuPy/Torch return type.

### Validation (2026-07-27) — PR #80 follow-up review

- The exact clean-commit P100 artifact at `n=4096`, `p=12` records synchronized
  NumPy/CuPy/Torch medians of 0.1003/0.0367/0.0373 seconds for continuous
  Breslow, 0.2316/0.0501/0.0488 for continuous Efron,
  0.0234/0.0184/0.0187 for heavy-ties Breslow, and
  0.1858/0.0214/0.0198 for heavy-ties Efron. All runs converged and all six
  extreme-predictor backend/tie cases were finite:
  `results/benchmark_frontend_sources/coxph_stability_resource_pr80_20260727.json`.
- A machine-readable physical-P100 artifact records its exact clean source
  commit, Cox/FISTA/fit source hashes, 24 synchronization/gradient comparisons,
  48 SCAD/MCP coefficient/objective/KKT/finite-state results, six synchronized
  performance cases, and two physical GPU workspace measurements:
  `results/benchmark_frontend_sources/penalized_cox_trusted_gradient_pr80_20260727.json`.
  It labels fresh-process cold-start timing as unmeasured, records both the
  first fit in the warmed process and the immediately repeated steady-state
  fit, and states which initialization or compilation costs are excluded.

### Optimized (2026-07-27) — PR #80 follow-up review

- Ordinary right-censored Exact ties now use one segmented prefix DP across all
  strata. Delayed-entry GPU workloads with at least eight strata can use one
  memory-gated global batch; smaller cases use bounded per-stratum batches.

### Fixed (2026-07-27) — PR #80 follow-up review

- Fractional, non-finite, or out-of-int64-range strata are rejected before
  integer conversion, including oversized unsigned labels; representable
  `uint64` labels are accepted consistently by NumPy, CuPy, and Torch.
  `STATGPU_TORCH_EXACT_SCAN_STRATEGY` selects `auto`, `native`, or
  `channelwise`; conservative `auto` enables the split scan only on the
  benchmarked Torch 2.0 + Pascal/P100 combination.

- Public Cox fit boundaries preserve packed CuPy/Torch targets, revalidate mutable
  device and boolean controls, reject complex prediction inputs before casting,
  and transactionally clear failed-refit state. `inference_mode="approx"` is
  documented as a compatibility-only alias for the exact unified inference
  path, and public estimators document their broader factorized host-label
  support for strata.

### Optimized (2026-07-27) — PR #80 follow-up review

- `STATGPU_COX_GROUP_MAX_BYTES` now gates the Breslow/Efron delayed-entry
  failure-group workspace before allocation. An oversized single risk set uses
  a stable backend-native row-streaming moment fallback rather than allocating
  an unbounded minimum-size dense batch.
  The final schema-v3 exact-source P100 refresh passed 121 targeted tests. At `n=4096`,
  `p=128`, and an 8 MiB limit, the old 1,056,768-byte estimate selected dense
  while the corrected 9,445,376-byte estimate selected streaming. CuPy and
  Torch both recorded the streaming route and matched NumPy within `4.441e-15`.
  Concordance now tiles both event and sample axes under a hard two-million-pair
  ceiling; the physical `n=2,000,001` boundary used two sample tiles, while
  ordinary, counting-process, and penalized all-censored scoring returned `0.5`:
  `results/benchmark_frontend_sources/coxph_concordance_boundary_pr80_20260728.json`.
- Ordinary concordance now accumulates all tile counts on the active backend
  and performs one batched scalar transfer after the loop, instead of three
  host synchronizations per tile.

### Validation (2026-07-27) — PR #80 follow-up review

- The maintained delayed-entry + 3-strata P100 benchmark reached
  NumPy/CuPy/Torch medians of 136.02/36.50/21.95 seconds at 10,240 rows, or
  3.73x/6.20x GPU speedups over NumPy. The corresponding artifact and the new
  strata-count artifact completed with zero gate failures.
- On the same P100 at `n=4096`, `p=12`, and 64 time bins, the direct-moment
  SCAD NumPy/CuPy/Torch medians were 0.08350/0.03148/0.02137 seconds and MCP
  medians were 0.08469/0.03100/0.02133 seconds after one excluded warmup.
  CuPy/Torch were 2.65x/3.91x faster than NumPy for SCAD and 2.73x/3.97x for
  MCP. The artifact labels these as warm, synchronized timings rather than
  fresh-process latency.
- The refreshed schema-v4 exact-source completion artifact passed 159 targeted tests on
  CuPy 13.6.0 and Torch 2.0.0+cu117 on a Tesla P100. It verifies public cleanup
  on success and failure, single outer `CoxPHCV` cleanup ownership, truthful
  summaries, shared stateless inference results,
  integer subject codes, one ordinary-concordance scalar transfer, direct
  backend reuse, absence of import-time method replacement, and private
  legacy composition isolation, with `source_clean=true`, 21 Git-blob-verified
  hashes, and `gate_failures=[]`:
  `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260728.json`.

### Optimized (2026-07-26) — PR #80 stratified Exact composition

- Multi-stratum Exact fits previously bypassed both optimized one-stratum
  kernels and fell back to a Python/device loop over every stratum and failure
  time. The new path composes the nested right-censored or bounded batched
  counting-process objective once per stratum; NumPy can now use the same
  memory-gated batched Exact kernel for delayed-entry workloads.
- On a Tesla P100-SXM2-16GB (`p=4`, three strata, full fit plus inference),
  R/NumPy/CuPy/Torch medians were 0.0180/0.0143/0.1742/0.0747 s at `n=160`,
  0.258/0.2263/0.2181/0.1341 s at `n=15,360`, and
  1.118/0.9874/0.2285/0.1384 s at `n=61,440`. The GPU paths overtake R by the
  measured `n=15,360` point; at `n=61,440`, CuPy and Torch are 4.89x and 8.08x
  faster than R. Small stratified fits remain launch-bound on explicit GPUs.
- R 4.4.1/survival 3.8.9 alignment reports zero gate failures. Maximum
  coefficient, exact partial-log-likelihood, and covariance differences are
  `5.84e-10`, `8.15e-10`, and `4.45e-12`.
- Reusable benchmark: `dev/benchmarks/benchmark_exact_ties_scaling.py` with
  `--scaling-scenario strata`; auditable artifact:
  `results/benchmark_frontend_sources/coxph_exact_strata_pr80_20260726.json`.

### Optimized (2026-07-26) — PR #80 Torch Exact channel scans

- Profiling the nested Exact implementation on a Tesla P100 with PyTorch
  2.0.0+cu117 showed that one-dimensional CUDA prefix sums were fast, while
  long `cumsum(dim=0)` calls over 4 or 16 trailing moment channels dominated
  the Torch runtime.
- For Torch CUDA inputs with at least 2,048 rows and at most 64 trailing
  channels, Exact now transposes each channel into contiguous storage, executes
  efficient one-dimensional scans, and stacks the results back on device.
  `STATGPU_TORCH_EXACT_SCAN_MIN_ROWS` and
  `STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS` configure the gates. Small, wide, and
  CPU cases keep the native scan.
- The extra channel-scan workspace is included in the existing 512 MiB nested
  Exact memory decision. If the base DP fits but the extra scan workspace does
  not, the nested algorithm remains active with the native Torch scan.
- On the synchronized bounded-tie workload (`p=4`, maximum tie size 8, full fit
  plus inference), R/NumPy/CuPy/Torch medians were
  0.295/0.273/0.0949/0.0558 s at `n=15,360`,
  1.323/1.465/0.1114/0.0662 s at `n=61,440`, and
  2.691/3.043/0.1430/0.1000 s at `n=122,880`. At the largest size Torch is
  30.32x faster than its previous result, 26.92x faster than R, 30.44x faster
  than NumPy, and 1.43x faster than CuPy.
- R 4.4.1/survival 3.8.9 alignment reports zero gate failures; the maximum
  coefficient, exact partial-log-likelihood, and covariance differences are
  `1.30e-09`, `5.12e-09`, and `5.01e-12`. The local 13-file matrix passed
  **297 tests** with 97 optional-dependency skips, and the physical-P100 matrix
  passed **392 tests** with 2 expected skips.
- Reusable entry point: `dev/benchmarks/benchmark_exact_ties_scaling.py`, which
  writes `results/exact_ties_scaling.json`; final artifact hashes are recorded
  in `dev/reviews/pr80_review_fix.md`.

### Optimized (2026-07-26) — PR #80 right-censored Exact full fit

- Large-sample phase profiling showed that the Exact likelihood prefix was no
  longer the full-fit bottleneck: Breslow baseline inference still performed a
  failure-group-by-sample risk-mask scan for ordinary right-censored data.
- Replaced that common path with a per-stratum descending-stop log-risk prefix:
  NumPy uses `logaddexp.accumulate`, Torch uses `logcumsumexp`, and CuPy uses a
  shifted cumulative sum inside a conservative predictor-range gate. Delayed
  entry and extreme CuPy predictors retain stable backend-native fallbacks.
- At `n=61,440`, NumPy/CuPy/Torch baseline phases fell from
  6.847/5.988/3.328 s to 0.0202/0.00701/0.00265 s. The final local affected
  matrix passed with **226 passed, 37 skipped, 0 failed**; the complete 13-file
  physical-P100 matrix passed with **388 passed, 2 expected skips, 0 failed**.
- On the synchronized P100 bounded-tie workload (`p=4`, maximum tie size 8,
  full fit plus inference), R/NumPy/CuPy/Torch medians were
  0.305/0.282/0.0971/0.361 s at `n=15,360`,
  1.293/1.469/0.113/1.510 s at `n=61,440`, and
  2.589/3.023/0.1518/3.031 s at `n=122,880`. CuPy was 17.05x faster than R and
  19.91x faster than NumPy at the largest measured size; small `n=1920` GPU
  fits remain launch-bound.
- R 4.4.1 survival 3.8.9 alignment still reports zero gate failures. Maximum
  coefficient, exact partial-log-likelihood, and model-covariance differences
  across the comprehensive cases are `1.30e-09`, `5.46e-12`, and `5.01e-12`.
- Reusable validation entry point:
  `dev/benchmarks/benchmark_exact_ties_scaling.py`, which writes
  `results/exact_ties_scaling.json`.


### Improved (2026-07-25) — v0.2.2 release preparation

- **Version and packaging**:
  - Updated `pyproject.toml` and `statgpu/__init__.py` from 0.2.1 to 0.2.2.
  - Retained the tag-triggered PyPI workflow and `STATGPU_NO_EXT=1` build policy,
    which produces a universal `py3-none-any` wheel plus a source distribution.
  - Kept Python 3.9 through 3.12 in the maintained CI matrix.
- **Included maintained scope**:
  - Carries the PR #79 correctness, backend-contract, inference, and validation
    work summarized in the entries below and the linked auditable artifacts.
  - Includes PR #84's release-facing README, documentation portals, method
    inventory, bilingual model/backend guides, and deterministic docs contracts.
- **Release files**:
  - `pyproject.toml`
  - `statgpu/__init__.py`
  - `CHANGELOG.md`
  - `docs/en/changelog.md`
  - `docs/cn/changelog.md`

### Validation (2026-07-25) — v0.2.2 release candidate

- The version declarations agree at 0.2.2; live PyPI metadata reported 0.2.1 as
  the latest release, and the remote repository had no `v0.2.2` tag.
- The documentation link check and maintained-document contracts passed for
  all 122 maintained documentation files.
- The complete CPU-only suite passed with **1051 passed, 257 skipped, 0 failed**.
- `STATGPU_NO_EXT=1` produced `statgpu-0.2.2-py3-none-any.whl` and
  `statgpu-0.2.2.tar.gz`; both artifacts passed `twine check`.
- Wheel and sdist metadata, archive paths, and contents were audited, with no
  local configuration, credentials, caches, or unrelated result bundles found.
- Fresh wheel and sdist environments both imported statgpu 0.2.2 from their
  installed `site-packages` and passed a CPU `LinearRegression` smoke test.

### Added and fixed (2026-07-25) — PR #80 Cox Phase-1 completion

- Reconciled the PR #80 head, originally based on 0.2.1, with the 0.2.2 release
  tree while preserving the 0.2.2 version and PR #79 inference/KKT contracts.
- Added a shared counting-process risk-set engine for Breslow, Efron, and Exact
  ties, delayed entry, `(start, stop]` time-varying rows, strata, penalties,
  robust/cluster inference, and subject-aware concordance.
- Extended `CoxPHCV` with start/strata/subject propagation, subject-preserving
  folds, Exact held-out likelihood, backend-consistent refit, and inference-mode
  provenance.
- Fixed final-KKT convergence, the open-left `start < event_time` boundary,
  baseline-hazard construction, backend-native prediction/scoring, and
  synchronized GPU benchmark timing and source-version reporting.
- Vectorized dense Efron cumulative moments and log-likelihood substeps on CuPy
  and Torch. For one-stratum ordinary right-censored Exact fits, NumPy/CuPy/Torch
  now reuse an elementary-symmetric prefix DP across nested risk sets, while
  sorted event-time segment sums remove the dense failure-group-by-sample mask.
  Delayed entry, multiple strata, score residuals, excessive workspace, and conservative
  numerical-range gates retain the normalized backend-native batch/per-group
  fallbacks. Both Exact workspace limits default to 512 MiB and are checked
  before dense allocation.
- Reused the default zero-initial objective for null-model inference, the
  accepted final objective when score residuals are not requested, and the
  solver's null score/information in `CoxPH`, removing redundant Exact fits.
- The 2026-07-25 local NumPy quick gate passed all executable correctness,
  inference, CV, schema, and external-comparison checks. Paramiko validation of
  the exact reviewed source in remote `myconda` on a Tesla P100 exposed and
  fixed Torch prediction, scikit-learn 1.2.2 cloning, and test-boundary issues.
  The final physical-GPU matrix passed with **384 passed, 2 expected skips, 0
  failed**; quick/full benchmark schemas passed without gate failures on NumPy,
  CuPy, and Torch.
- The synchronized full benchmark measured heavy-ties median fit time at
  0.477 s for NumPy, 0.179 s for CuPy, and 0.212 s for Torch; the earlier Efron
  optimization remains 8.36x/24.31x faster on CuPy/Torch. The final nested-Exact
  benchmark on the same Tesla P100 (`p=4`, maximum tie size 8, full fit plus
  inference) measured R/NumPy/CuPy/Torch at 0.029/0.0253/0.1686/0.0941 s for
  `n=960` and 0.047/0.0585/0.2690/0.1590 s for `n=1920`. At `n=1920`, the
  StatGPU paths improved about 928x/41.0x/41.6x over the reviewed pre-prefix
  NumPy/CuPy/Torch implementation, with no implicit CPU fallback. The reusable
  benchmark is `dev/benchmarks/benchmark_exact_ties_scaling.py`.
- Extended that benchmark with R 4.4.1 survival 3.8.9
  `coxph(ties="exact")` alignment. Right-censored, delayed-entry, strata, and
  combined delayed-entry/strata cases passed coefficient, exact log-likelihood,
  covariance, and convergence gates on all three StatGPU backends. Maximum
  differences from R were `1.30e-09`, `4.55e-13`, and `5.01e-12`, respectively.
  At `n=1920` on the bounded right-censored shape, R/NumPy/CuPy/Torch took
  0.047/0.0585/0.2690/0.1590 s; on the separate `n=160` delayed-entry shape they
  took 57.079/0.167/0.544/0.353 s, demonstrating that Exact performance depends
  strongly on risk-set shape.

### Validation (2026-07-24) — PR #79 exact-head closure

The final reviewed production head is
`c85750d63d4e6dbc9d988847566c20f5fa862e91`.

- GitHub Actions Tests run #545 passed on the exact head.
- Python 3.9, 3.10, 3.11, and 3.12 regression jobs passed.
- The complete CPU suite passed with **1074 passed, 275 skipped, 0 failed**.
- The clean-head canonical smoke pipeline passed with `canonical_eligible=True` and a
  `PASS` verdict.
- The maintained Tesla P100 suite passed **33 executed checks**, with two expected skips
  and zero failures.
- Maintained CoxPH, Linear, and Panel paths passed their PR79 acceptance contracts.

The six ignored legacy GPU diagnostic scripts executed separately are not part of the
maintained pytest Gate. Their conversion, replacement, or retirement is tracked in
[Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83).

### Fixed (2026-07-24) — final public-contract synchronization

- Corrected the CoxPH delayed-entry support matrix. Robust or cluster covariance with
  `compute_inference=True` raises explicitly; the same fit with
  `compute_inference=False` is allowed as estimation-only and leaves inference fields
  unset.
- Documented `CoxPHCV` as applying the same inference guard during final refit.
- Documented PooledOLS backend-preserving prediction, stable HAC `time_index` ordering,
  and effective-rank residual degrees of freedom.
- Clarified rank-deficient PooledOLS behavior: fitted values, prediction, RSS, rank, and
  fitted-space checks remain valid, while coefficient-level inference is
  `NOT_COMPARABLE` because it is not uniquely identified.
- Synchronized README, English/Chinese CoxPH and Panel pages, release summaries, and the
  auditable PR79 report.
- Removed stale hard-coded final accuracy artifacts. A new full canonical report may be
  committed only after a full exact-head raw campaign is validated by the current
  aggregator and renderer.

### Fixed (2026-07-23) — PR #79 complete review closure

- Unified CoxPH final KKT, line search, termination, and public result fields on
  CPU/CuPy/Torch.
- Added strict-by-default robust inference with explicit approximate opt-in,
  provenance fields, and the `statgpu[survival]` optional dependency.
- Kept Cox prediction and scoring backend-native, vectorized baseline hazards, removed
  the affected Torch Hessian materialization, and avoided unconditional GPU training-data
  host copies for nonrobust inference.
- Hardened PR79 diagnostics and canonical-report generation against missing, failed,
  duplicate, non-finite, dirty, and wrong-SHA evidence.
- Added behavioral regressions and synchronized the bilingual Cox support matrix.

### Validation history (2026-07-21)

The earlier complete Tesla P100 campaign passed on code head
`2f18e5dec9195da1a12e5eea89ee2d832557b3ad`:

- Gate A: 160 passed, 0 failed, 2 expected skips;
- Gate B: 1100 passed, 0 failed, 124 skipped, 1 strict XFAIL;
- Gate C: 10/10 metamorphic checks passed;
- Gate D: no audited full-design GPU-to-CPU transfer;
- Gate E: no leak over 15 repeated CuPy and Torch cycles;
- Gate F: synchronized Tesla P100 baselines recorded at three scales;
- Gate G: Ridge/scikit-learn and linear-regression/statsmodels parity passed.

A subsequent exact-head campaign on `786af9e2eb4742a56e5203b4380b03aec63a3ac8`
passed 17/17 focused physical-GPU checks. These historical SHAs remain auditable evidence,
but the 2026-07-24 entry above is the final PR head closure.

### Performance baseline — Tesla P100

These hardware-specific measurements remain regression baselines, not portable guarantees.

| Shape | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

Environment: Tesla P100-SXM2-16GB, Python 3.9, CuPy 13.6.0,
PyTorch 2.0.0+cu117.

### Known non-blocking follow-ups

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81): shared
  backend-native NaN/Inf validation.
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82): coordinated
  public-constructor refactor for scikit-learn <=1.2 clone identity.
- [Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83): convert or retire
  ignored legacy GPU diagnostic scripts.

## Historical entries

Detailed entries through 2026-07-14 are retained in
[the archived changelog](changelog-history-through-2026-07-14.md).
