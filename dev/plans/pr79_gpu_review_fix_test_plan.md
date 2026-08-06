# PR #79 Physical-GPU Review/Fix Validation Plan

Date: 2026-07-21  
Target branch: `agent/code-review-fixes`  
Current status before execution: `PARTIAL_REMOTE_PENDING`

## 1. Goal and exit decision

This plan closes the physical-GPU evidence gap for PR #79 through repeated
**review -> reproduce -> fix -> targeted retest -> affected-suite retest -> full
GPU re-review** cycles.

The plan is not a one-shot benchmark. A run is complete only when:

1. all mandatory CuPy CUDA and Torch CUDA correctness/device tests pass;
2. no explicit GPU mode silently transfers a complete numerical design to CPU;
3. numerical and inference differences satisfy the declared tolerances;
4. repeated fit/predict/score cycles do not show unbounded GPU-memory growth;
5. performance measurements are synchronized, warmed up, repeated, and archived;
6. every discovered defect has a minimal regression test and a recorded fix cycle;
7. CPU CI and the complete physical-GPU gate pass on the final clean commit;
8. documentation and changelogs match the final capability boundary.

Do not mark PR #79 ready solely because a smoke test passes.

## 2. Non-negotiable repository rules

- Explicit `device="cuda"` means CuPy CUDA and must not silently fall back to CPU.
- Explicit `device="torch"` means Torch CUDA and must not silently use Torch CPU.
- Formula parsing and string/categorical label factorization may remain CPU metadata
  boundaries; complete numerical X/y arrays must remain on the selected backend.
- GPU timing must synchronize before and after the measured region.
- A failure must be reproduced before it is fixed.
- Do not weaken a tolerance or skip a case merely to make the gate pass.
- Do not store passwords, SSH keys, tokens, or machine-specific credentials in the
  repository or result bundle.

## 3. Execution model

Use a dedicated validation worktree or clone on the GPU server.

```bash
git fetch origin
git checkout agent/code-review-fixes
git reset --hard origin/agent/code-review-fixes
export STATGPU_GPU_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export STATGPU_GPU_RESULT_DIR="results/pr79_gpu_validation/${STATGPU_GPU_RUN_ID}"
mkdir -p "${STATGPU_GPU_RESULT_DIR}"/{environment,logs,junit,parity,memory,performance,failures,iterations}
```

Use the existing remote configuration mechanism when connecting from a local
machine. Credentials must come from `STATGPU_REMOTE_*`, an SSH agent/key, or the
gitignored `dev/scripts/remote_config_local.py`.

The remote environment should be treated as immutable during a validation cycle.
If a dependency must change, start a new run ID and archive both environments.

## 4. Phase 0: freeze and record the environment

Record before importing statgpu:

```bash
git rev-parse HEAD | tee "${STATGPU_GPU_RESULT_DIR}/environment/git_sha.txt"
git status --short | tee "${STATGPU_GPU_RESULT_DIR}/environment/git_status.txt"
python -V | tee "${STATGPU_GPU_RESULT_DIR}/environment/python.txt"
python -m pip freeze > "${STATGPU_GPU_RESULT_DIR}/environment/pip_freeze.txt"
nvidia-smi -q > "${STATGPU_GPU_RESULT_DIR}/environment/nvidia_smi_q.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap \
  --format=csv,noheader > "${STATGPU_GPU_RESULT_DIR}/environment/gpus.csv"
```

Create `environment/backend_probe.json` containing at least:

- Python, NumPy, SciPy, scikit-learn, statsmodels versions;
- CuPy version, CUDA runtime/driver, selected device ID and compute capability;
- Torch version, `torch.version.cuda`, cuDNN version and selected device;
- `cp.cuda.runtime.getDeviceCount()` and `torch.cuda.device_count()`;
- default dtype and TF32/determinism settings;
- available and total memory immediately before testing.

Hard preflight failures:

- CuPy cannot allocate, synchronize, and round-trip a float64 tensor;
- Torch CUDA cannot allocate, synchronize, and round-trip a float64 tensor;
- CuPy and Torch resolve to different physical GPUs when a single-GPU comparison
  was requested;
- the checkout is dirty before testing;
- the recorded commit differs from the intended PR head.

## 5. Phase 1: immutable CPU baseline

Before GPU testing, run the permanent CPU gate on the same checkout:

```bash
python -m pytest dev/tests -q --tb=short \
  --junitxml="${STATGPU_GPU_RESULT_DIR}/junit/cpu_full.xml" \
  2>&1 | tee "${STATGPU_GPU_RESULT_DIR}/logs/cpu_full.log"
```

Any CPU failure blocks GPU interpretation. Fix CPU failures first, add a regression,
and restart from Phase 0 on a new commit.

## 6. Phase 2: backend and device smoke gate

Run these first because later results are meaningless if dispatch is wrong:

```bash
python -m pytest \
  dev/tests/test_backends.py \
  dev/tests/test_core_contracts.py \
  dev/tests/test_v10_import_smoke.py \
  dev/tests/test_three_backend_native_followup.py \
  dev/tests/test_third_full_review.py \
  -q --tb=short \
  --junitxml="${STATGPU_GPU_RESULT_DIR}/junit/gpu_smoke.xml" \
  2>&1 | tee "${STATGPU_GPU_RESULT_DIR}/logs/gpu_smoke.log"
```

Add physical-GPU assertions where existing tests only exercise Torch CPU:

- CuPy outputs are `cupy.ndarray` on the requested CUDA device;
- Torch outputs are CUDA tensors on the requested device;
- fitted state used by predict/score remains on the same backend;
- explicit `device="cuda"` fails clearly when CuPy is unavailable;
- explicit `device="torch"` fails clearly when Torch CUDA is unavailable;
- `device="auto"` records the backend it selected;
- float32 and float64 are not silently changed unless documented;
- non-contiguous Torch inputs either work or fail with a clear contract error.

## 7. Phase 3: numerical parity matrix

For every mandatory model/case, generate the data once in NumPy, copy the exact
bytes to CuPy and Torch, fit all three implementations, and compare canonical
NumPy copies of results.

### 7.1 Default tolerances

Use these as defaults, not as permission to ignore known tighter references:

| Quantity | float64 direct/deterministic | float64 iterative | float32 |
|---|---:|---:|---:|
| coefficients/predictions | `atol=1e-8, rtol=1e-6` | `atol=1e-6, rtol=1e-5` | `atol=1e-4, rtol=1e-4` |
| objective/KKT/gradient | `1e-8` | `1e-5` | `1e-3` |
| covariance/standard errors | `1e-6` | `1e-3` | `5e-3` |
| p-values | `5e-3` preferred, `5e-2` hard ceiling | `5e-2` | case-specific |

For clustering, rankings, support sets, and embeddings, use invariant metrics rather
than raw coordinate equality:

- adjusted Rand index / normalized mutual information;
- pairwise-distance or Gram-matrix agreement;
- subspace projection error and explained variance;
- selected support equality or symmetric difference;
- objective value and KKT residual.

Every tolerance exception must be justified in the result record.

### 7.2 Mandatory module coverage

#### Core linear, GLM, penalties, solvers and inference

Test at least:

- LinearRegression, Ridge and RidgeCV;
- Lasso, ElasticNet and their CV paths;
- LogisticRegression;
- squared-error SCAD/MCP through FISTA-LLA;
- quantile/Huber paths that claim GPU support;
- weighted and unweighted fits;
- intercept/no-intercept;
- full rank, rank deficient, `p > n`, sparse truth and multi-target where supported;
- inference fields: coef, bse, statistic, p-value, CI, likelihood/AIC/BIC where exposed;
- predict and score after fitting;
- sklearn clone and repeated fit on the same object.

For Ridge, preserve the repository's average-loss penalty normalization and apply the
explicit external-alpha mapping when comparing with scikit-learn.

#### Cox survival

Cover:

- Breslow and Efron ties;
- low, moderate and heavy tie rates;
- censored, low-event and high-event data;
- rank-deficient and moderately high-dimensional X;
- inference on/off;
- score-test/information calculations;
- prediction/risk score;
- repeated fits;
- numerical comparison with the NumPy path and the configured external reference.

The Torch Hessian path must receive an explicit peak-memory scaling test because it
still contains an `O(n*p*p)` intermediate.

#### ANOVA and post-hoc

Cover one-way, balanced two-way full/additive, Welch, Tukey, Bonferroni and effect
sizes, including:

- equal and unequal variance;
- zero-variance groups;
- fractional Welch denominator df;
- rejected unbalanced two-way designs;
- large groups where reductions should remain on-device.

#### Covariance

Cover EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS, MinCovDet,
GraphicalLasso and GraphicalLassoCV:

- centered and non-centered data;
- `p < n`, `p ~= n`, and selected `p > n` cases;
- rank deficiency;
- covariance/precision inverse identities;
- symmetry and positive-(semi)definiteness;
- MCD support and Mahalanobis distances;
- Graphical Lasso objective, sparsity pattern, convergence and CV selection.

#### Panel

Cover PanelOLS, RandomEffects, PooledOLS, BetweenOLS, FirstDifferenceOLS and
FamaMacBeth:

- numeric and string entity/time labels;
- balanced and unbalanced panels;
- array and formula interfaces;
- Patsy missing-row deletion with aligned side arrays;
- entity/time effects;
- nonrobust, robust, one/two-way clustered and HAC/Newey-West covariance where
  supported;
- rank-deficient designs;
- Fama-MacBeth coefficient path, inference and prediction;
- device preservation through sorting, grouping and differencing.

#### Kernel, smoothing, spline and GAM

Cover:

- RBF/polynomial/sigmoid/chi-square kernels;
- KernelRidge/CV, KernelPCA and Nystroem;
- KDE and kernel regression;
- B-spline, natural, cyclic and thin-plate bases;
- SplineTransformer `error`, `constant`, `linear`, `continue` modes;
- GAM fit/predict;
- rank-deficient kernel/Gram matrices;
- knot/device metadata transfer;
- large kernel matrices near the memory limit, with a safe precomputed size cap.

#### Unsupervised

Cover PCA, IncrementalPCA, TruncatedSVD, KMeans/MiniBatchKMeans, GMM, NMF,
MiniBatchNMF, DBSCAN, NNDescent, t-SNE and UMAP where supported:

- deterministic seed replay;
- empty/singleton clusters where relevant;
- duplicated points and zero-variance columns;
- cluster-label invariants rather than label-number equality;
- embedding pairwise distances/subspace invariants;
- fit/transform consistency;
- NaN/Inf rejection;
- repeated fit and cache cleanup.

#### Feature selection, metrics and resampling

Cover:

- knockoff construction and selection statistics;
- FDR/power calibration over multiple seeded simulations;
- Stepwise state reset and feature order;
- classification metrics on CuPy/Torch inputs;
- bootstrap/permutation RNG determinism, device purity and cleanup;
- multiple-testing/distribution output parity.

## 8. Phase 4: adversarial and metamorphic tests

For each supported model family include applicable transformations:

- row permutation invariance;
- feature permutation with inverse-permuted coefficients;
- duplicated rows;
- global sample-weight scaling invariance;
- adding a constant to a feature when intercept handling should absorb it;
- response scaling identities for Gaussian models;
- identical rerun with the same random seed;
- different seed produces a valid but not byte-identical randomized result;
- fit twice on different shapes using the same estimator object;
- failed fit followed by a valid fit does not retain partial state;
- read-only/non-contiguous inputs;
- NaN, +Inf and -Inf fail before low-level CUDA errors;
- invalid labels/shapes/parameters produce consistent public exceptions.

A metamorphic failure is treated as a correctness finding even when CPU and GPU
agree on the same wrong result.

## 9. Phase 5: host-transfer and device-purity audit

Mandatory checks:

1. inspect output/state device and dtype after fit, predict, score and summary;
2. instrument or monkeypatch known boundaries (`cp.asnumpy`, `torch.Tensor.cpu`,
   backend `to_numpy`) to count transferred bytes;
3. distinguish allowed scalar/metadata transfers from complete numerical designs;
4. run representative cases under `torch.profiler` and, when available, Nsight
   Systems (`nsys profile`) to identify synchronization and D2H/H2D copies;
5. record every allowed transfer boundary in `parity/host_transfer_manifest.json`.

Blockers:

- full X/y transfer in an explicit GPU numerical path;
- output unexpectedly returned on CPU when the contract promises backend arrays;
- repeated hidden transfers proportional to `n*p` inside an iterative loop;
- CuPy and Torch silently selecting different devices.

## 10. Phase 6: repeated-fit and GPU-memory tests

Use both backend-native counters and `nvidia-smi`.

### CuPy

Record before/after each iteration:

- memory-pool `used_bytes()` and `total_bytes()`;
- pinned-pool statistics;
- device free/total memory;
- allocated object/device ownership.

### Torch

Record:

- `memory_allocated`, `memory_reserved`;
- `max_memory_allocated`, `max_memory_reserved`;
- memory summary on failure;
- device synchronization before reading counters.

For each representative estimator:

1. warm up twice;
2. run fit -> predict/transform/score -> delete for at least 20 cycles;
3. test `gpu_memory_cleanup=False` and `True` where exposed;
4. run `gc.collect()` plus the backend's documented cleanup between normalized
   measurements;
5. repeat with changing input shapes;
6. repeat after an intentional validation failure.

Hard failures:

- monotonic unbounded allocated-memory growth;
- OOM at a size that succeeds on the first iteration;
- stale fitted tensors retained after object deletion/cleanup;
- cleanup corrupts state required for predict/score.

Investigate as a probable leak when normalized used/allocated memory grows by more
than both 10% and 128 MiB after warmup. Pool `reserved/total` growth alone is not a
leak unless live allocation or free device memory continues to deteriorate.

Reuse and extend `dev/benchmarks/benchmark_gpu_memory_cleanup.py`; it currently
covers CuPy linear, Ridge, Lasso, logistic and Cox and should be extended to Torch
and the newly corrected panel/kernel/spline paths.

## 11. Phase 7: synchronized performance and scaling

Performance is measured only after correctness passes.

For each case:

- generate data before timing;
- transfer data before timing unless transfer cost is the explicit metric;
- warm up at least two runs;
- synchronize immediately before starting and after finishing;
- use at least five measured repeats, report median, IQR, min and max;
- record dtype, n, p, solver, tolerance, iterations and convergence status;
- run backends in alternating order to reduce thermal/order bias;
- record GPU utilization, clocks, temperature and memory;
- separate fit, inference, predict/transform, CV and end-to-end timings;
- compare objective accuracy before interpreting speedup.

Required scale tiers:

- small: correctness/crossover, expected GPU overhead;
- medium: representative workload;
- large: GPU-beneficial workload;
- stress: largest safe case below 80% of available VRAM.

Do not require GPU speedup for small problems. Treat an unexplained median regression
above 20% versus a same-machine saved baseline as review-required, not automatically
as a correctness failure. Never publish a speedup without the hardware, workload,
synchronization and reference identity.

Useful existing starting points:

- `dev/benchmarks/benchmark_torch_vs_cupy_comprehensive.py`;
- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`;
- `dev/benchmarks/benchmark_gpu_warmup.py`;
- module-specific benchmark and external-comparison scripts.

Do not treat old ad-hoc scripts as authoritative until their synchronization,
correctness checks and JSON output are reviewed.

## 12. Phase 8: external statistical validation

Run after three-backend parity:

- sklearn for estimators, predictions, CV selection and transforms;
- statsmodels for Gaussian/GLM/panel/inference quantities;
- lifelines or the repository's configured Cox reference;
- SciPy for ANOVA/distributions/KDE where applicable;
- R baselines for methods already covered by maintained comparison scripts.

External settings must explicitly align:

- objective normalization and penalty mapping;
- intercept and centering;
- sample weights;
- ties method;
- solver, tolerance and maximum iterations;
- covariance/inference convention;
- random seed and data split.

Store coefficients, predictions, objective/KKT values, standard errors, p-values,
confidence intervals and model-selection decisions where applicable.

## 13. Required review/fix loop

For every failure create an iteration directory:

```text
iterations/iteration-N/
  finding.md
  reproducer.py
  before.json
  patch-summary.md
  targeted-test.log
  affected-suite.log
  gpu-smoke.log
  after.json
```

### Finding template

- ID and severity: CRITICAL / HIGH / MEDIUM / LOW;
- module, estimator, backend, dtype, device and data shape;
- exact commit/environment;
- observed behavior and expected contract;
- minimal reproduction command;
- whether CPU, CuPy and Torch agree;
- correctness, inference, device, memory or performance impact;
- suspected root cause;
- proposed fix and affected capability axes.

### Fix protocol

1. reproduce twice on a clean process;
2. reduce to a minimal deterministic test;
3. add the failing regression before or with the fix;
4. implement the smallest architecture-consistent fix;
5. run the new targeted test on CPU, CuPy CUDA and Torch CUDA;
6. run the complete affected module suite;
7. run the GPU smoke gate;
8. rerun memory/performance cases if the code path changed;
9. run the permanent CPU CI suite;
10. re-review adjacent shared helpers and callers;
11. start another full GPU cycle if a shared backend/solver helper changed.

Do not close a finding when only one GPU backend passes.

## 14. Severity and blocking policy

### CRITICAL

- statistically wrong result with plausible output;
- silent CPU fallback for explicit GPU mode;
- data corruption, device mix-up or nondeterministic wrong state;
- security/credential leakage.

### HIGH

- backend crash on a documented supported path;
- parity/inference mismatch above hard tolerance;
- unbounded memory growth or reproducible OOM regression;
- failed convergence reported as success;
- repeated-fit state contamination;
- formula/metadata misalignment.

### MEDIUM

- clear but inconsistent validation/error behavior;
- material performance regression with correct output;
- avoidable whole-array transfer outside an inner loop;
- documentation/test coverage gap.

### LOW

- readability, duplication, minor diagnostics or non-blocking ergonomics.

PR readiness requires zero unresolved CRITICAL/HIGH findings. MEDIUM findings must be
fixed or explicitly deferred with user-visible behavior and an issue/plan.

## 15. Result bundle and machine-readable records

Every JSON result should include:

- schema version;
- run ID, UTC timestamp, git SHA and dirty status;
- host/GPU UUID, driver/runtime and package versions;
- module/model/case/backend/device/dtype/shape/seed;
- parameters and solver/convergence metadata;
- numerical reference and tolerance;
- pass/fail plus exception/traceback;
- elapsed samples and summary statistics;
- peak/live/reserved memory;
- transferred bytes when measured;
- artifact paths.

Produce at minimum:

```text
environment/backend_probe.json
parity/parity_matrix.json
parity/host_transfer_manifest.json
memory/repeated_fit_memory.json
performance/timing_matrix.json
failures/findings.json
review_summary.md
```

Do not commit raw large logs automatically. Commit the final concise review summary,
small JSON evidence required by documentation, permanent regression tests, and any
maintained benchmark result explicitly approved for the repository.

## 16. Recommended execution order

### Gate A: fast physical-GPU smoke

Environment probe, backend dispatch, import, shared helpers, third-review regressions,
and one representative estimator from each major family.

### Gate B: complete correctness/parity

All mandatory module cases, adversarial/metamorphic tests, formula and inference.

### Gate C: memory/device purity

Repeated fit, cleanup, host-transfer profiling and failed-fit recovery.

### Gate D: performance/scaling

Synchronized medium/large/stress benchmarks only after A-C pass.

### Gate E: external references

sklearn/statsmodels/SciPy/lifelines/R comparisons with aligned settings.

### Gate F: final re-review

Run the complete CPU test tree, complete physical-GPU gate and documentation audit on
the final clean commit. Update PR status from `PARTIAL_REMOTE_PENDING` only when the
physical evidence is complete.

## 17. Suggested first commands on the server

```bash
# 1. Activate the existing prepared environment; do not mutate it mid-cycle.
source <conda-path>/etc/profile.d/conda.sh
conda activate myconda

# 2. Update the clean validation checkout.
cd <remote-statgpu-workdir>
git fetch origin
git checkout agent/code-review-fixes
git reset --hard origin/agent/code-review-fixes
find . -name __pycache__ -type d -prune -exec rm -rf {} +

# 3. Record environment and run CPU baseline.
export STATGPU_GPU_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export STATGPU_GPU_RESULT_DIR="results/pr79_gpu_validation/${STATGPU_GPU_RUN_ID}"
mkdir -p "${STATGPU_GPU_RESULT_DIR}"/{environment,logs,junit,parity,memory,performance,failures,iterations}
nvidia-smi -q > "${STATGPU_GPU_RESULT_DIR}/environment/nvidia_smi_q.txt"
python -m pytest dev/tests -q --tb=short \
  --junitxml="${STATGPU_GPU_RESULT_DIR}/junit/cpu_full.xml" \
  2>&1 | tee "${STATGPU_GPU_RESULT_DIR}/logs/cpu_full.log"

# 4. Run the first focused GPU gate.
python -m pytest \
  dev/tests/test_backends.py \
  dev/tests/test_core_contracts.py \
  dev/tests/test_three_backend_native_followup.py \
  dev/tests/test_third_full_review.py \
  dev/tests/test_ordered_cross_backend.py \
  dev/tests/test_distributions_backend.py \
  -q --tb=short \
  --junitxml="${STATGPU_GPU_RESULT_DIR}/junit/gpu_gate_a.xml" \
  2>&1 | tee "${STATGPU_GPU_RESULT_DIR}/logs/gpu_gate_a.log"
```

Before running the full plan, review the collected test list and confirm that each
mandatory physical-GPU case actually executes rather than skips. A green suite with
all CUDA cases skipped is a failed validation run.
