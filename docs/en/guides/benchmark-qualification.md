# Auditable benchmark qualification campaigns

Benchmark coverage has three different denominators: public capabilities,
registered workloads, and the finite configurations actually selected for a run.
Passing one denominator does not complete the others. The maintained runners are
**candidate evidence producers**, not certification of every public API, every
CUDA operation, or release-ready speedups.

`dev/benchmarks/run_full_benchmark_matrix.py` is the common plan-first entry
point for registered configurations and independent protocols.
`dev/benchmarks/run_qualification_campaign.py` provides opt-in serial execution
of seven registered suites, with exact workload inventories, separate CPU thread
profiles, and failure-preserving archives. Its default **21 invocations and 1692
rows are the registered representative matrix, not the full benchmark surface**.
Planning a broader matrix must also account for independent runners and axes that
are not represented by those default fixtures.

## Choose the coverage denominator

| Denominator | What it counts | What it does not establish |
| --- | --- | --- |
| 151 public capability mappings in the coverage audit | Direct, runner-smoke, indirect, alias and infrastructure relationships | 151 independent numerical passes or physical GPU proofs |
| 275 registered case identities | 2 Gaussian + 7 classification + 74 model + 138 statistical + 44 learning + 4 workflow + 6 CV identities | A complete solver, dtype, weight, shape or backend matrix |
| 1692 default expanded rows | The exact registered identities at default shapes, seeds, input dtypes, scenarios and three backends | Exhaustive API coverage; 16 rows are expected host-only CUDA `not-applicable` dispositions |
| 41 additional parameterized runner files in the repository audit | Independent timing entry points outside the seven registered runners | 41 mutually disjoint, safe-to-run suites or 41 additional passes |
| Selected finite matrix | The exact configuration and evidence identities declared in the generated plan | Every possible input or configuration, or any unexecuted/blocked dimension |

The repository audit inspected 202 tracked Python files: 158 top-level and 44
nested. This is a source-inventory snapshot, not a live file total or a suite
count. It includes helpers, wrappers, validators, diagnostics, legacy programs
and remote launchers. Wrapper and child execution must not be counted twice.
Likewise, indirect or alias capability mappings must not become additional
independent numerical evidence.

A useful completion report separates **planned, executed, passed, failed,
blocked, not applicable and not run**. A blocked axis stays visible in the
coverage denominator; it is not silently removed to obtain a green result.
Completing all selected executable jobs is different from closing every declared
axis. Preserve exact case, configuration, backend, shape, input/computed dtype,
seed, thread profile and timing protocol when comparing or deduplicating results.
Overlapping representative, medium and large points do not add unique coverage
merely because they came from different batches.

## Start with the unified plan

The common entry point combines registered configuration plans with the
independent-runner inventory:

```bash
python -m dev.benchmarks.run_full_benchmark_matrix \
  > /tmp/statgpu-full-plan.json
```

The default only plans: it includes the unchanged registered representative
matrix and all 41 independent inventory entries with their dispositions. It does
not launch benchmark workloads, import numerical runners or query a GPU. The
name “full” means a unified accounting surface, not that every entry is runnable
or every API is covered. Inspect the plan before choosing execution lanes.

For a selected plan, save the following JSON as an external configuration file,
for example `/tmp/statgpu-matrix-config.json`:

```json
{
  "schema_version": "statgpu-full-benchmark-config-v1",
  "registered": {
    "argv": ["--suites", "six-family-cv", "--seeds", "17", "42"]
  },
  "independent": {
    "lanes": ["public-native-kde-bootstrap"],
    "configurations": {
      "public-native-kde-bootstrap": [
        {"n":128,"dimension":1,"bandwidth":0.4,"weights":"none","seed":17},
        {"n":256,"dimension":3,"bandwidth":0.4,"weights":"unequal","seed":42}
      ]
    },
    "validation_tier": "remote-full",
    "environment_id": "my-benchmark-machine",
    "expected_sha": "<full-commit-sha>"
  }
}
```

Replace `<full-commit-sha>` with the reviewed checkout's exact 40-character
commit and use the actual environment identifier before execution. The displayed
placeholder is not a valid execution value. `remote-full` labels the evidence
tier; it does not connect to a server or launch SSH. Run the command in the
intended environment. The example selects two bounded public KDE configurations, each using all three
backends. Omitted fields retain adapter defaults; `registered.argv` backend
options do not filter independent lanes.

```bash
python -m dev.benchmarks.run_full_benchmark_matrix \
  --config /tmp/statgpu-matrix-config.json > /tmp/statgpu-selected-plan.json

# Only after reviewing the selected workloads, dependencies and resource budget:
python -m dev.benchmarks.run_full_benchmark_matrix \
  --config /tmp/statgpu-matrix-config.json --execute \
  --output-dir /tmp/statgpu-selected-evidence-NEW --timeout 1800
```

Omit `registered` or `independent` to select only the other lane. A registered
lane accepts an argv list for the campaign CLI below; nested `--execute` and
`--output-dir` are forbidden because the coordinator owns execution and evidence
locations. An independent lane requires explicit unique lane IDs for execution;
an inventory-all plan is not executable. Unknown/duplicate JSON keys are
rejected. Output must be fresh and outside the checkout. The coordinator records
its configuration hash and source snapshots, keeps lane evidence in `registered/`
and `independent/`, and freezes the collected report and archive. A
`complete_for_selected_protocols` result is not whole-library certification.

## Inspect first: planning does not run benchmarks

From the source checkout:

```bash
python -m dev.benchmarks.run_qualification_campaign > /tmp/statgpu-plan.json
```

Without `--execute`, the script imports only the standard library, reads the
registry and CPU allocation, and prints the plan. It does not import statgpu,
NumPy, CuPy or Torch; launch subprocesses; query a GPU; or create an output directory.
`--output-dir` alone does not authorize execution.

The default `representative` profile preserves the existing seven-suite fullscan
shapes, cases, input dtypes, scenarios, seeds, and runner-default warmup/repeats.
It is not a new reduced smoke profile, but it is still a fixed set of fixtures:

| Suite | Default dimensions | Rows per backend |
| --- | --- | ---: |
| Gaussian refresh | 1024×16, 8192×64, 32768×128; OLS/Ridge; float32/float64 inputs; unweighted/analytic weights | 24 |
| Classification metrics | n=1000/10000/100000; 7 operations; 2 input dtypes; 3 scenarios | 126 |
| Model families | All 74 cases; nominal n=500, p=8 | 74 |
| Statistical families | All 138 cases; n=128/1024, p=4 | 276 |
| Learning families | All 44 cases; n=256, p=4; knockoff p=20 | 44 |
| Workflow families | 4 cases; n=256/1024, p=6, targets=4 | 8 |
| Six-family CV | 6 models; n=240/1024, p=16; 3 folds and fixed grids | 12 |

There are 21 serial invocations: seven suites × NumPy/CuPy/Torch, with one CPU
thread requested for every invocation. The exact total is **1692 rows**, including
16 permitted host-only `not-applicable` rows in the CUDA statistical suites.
These 16 rows are expected dispositions, not GPU execution successes. The current
producer omits their problem dimensions, so the assessor preserves and checks
that omission and their exact multiplicity instead of inventing measured shapes.
The plan records all nominal selected shapes. Balanced-panel effective n and
one-dimensional learning fixtures are normalized to the producer's actual layout.

## Explicit execution on the intended benchmark machine

Use a reviewed source checkout and an environment with the runner dependencies,
`threadpoolctl`, and the requested physical backends already installed. For GPU
or numerical benchmark work, follow the project's configured remote-environment
policy; do not interpret local harness tests as physical CUDA evidence.

Run the targeted regression gates **before** starting a campaign:

```bash
python -m pytest -q -rs \
  dev/tests/test_cv_clean_timing.py \
  dev/tests/test_learning_compute_qualification.py
```

Their current physical single-GPU matrix comprises 12 CV nodes (six models ×
CuPy/Torch) and 88 learning nodes (44 cases × CuPy/Torch). These are physical
matrix dimensions, not fixed totals for all tests in those files. To inspect or
run just these physical gates on the intended machine:

```bash
python -m pytest -q -rs \
  dev/tests/test_cv_clean_timing.py::test_physical_cv_clean_lane_matches_observed_cuda_workload \
  dev/tests/test_learning_compute_qualification.py::test_physical_gpu_complete_learning_compute_contract
```

A local run without GPU can validate harness behavior but cannot pass these
physical gates by skipping them. Even on a machine with a GPU, review every
skip reason and verify that the intended backend nodes actually executed;
skipped or unavailable nodes leave physical closure open.

```bash
OUT="/tmp/statgpu-qualification-$(date -u +%Y%m%dT%H%M%SZ)"
python -m dev.benchmarks.run_qualification_campaign \
  --execute --output-dir "$OUT" --timeout 1800
```

The output directory must not exist, its parent must already exist, and its
resolved location must be outside the checkout. Existing output directories,
archives, and archive checksums are refused, even if the directory is empty.
The timeout is positive and capped at 1800 seconds **per invocation**, including
imports and preflight. It is not a whole-campaign time budget; the 21 default timeouts sum to 10.5
hours, before scheduling and finalization overhead. This is neither a duration
prediction nor a strict whole-campaign wall-clock bound. There is no `--allow-incomplete` escape hatch.
Use `--suites` and `--backends` to request a smaller explicit campaign.

Ordinary nonzero exits, missing dependencies, malformed/missing JSON, timeout,
failed numerical checks, and incomplete artifacts are retained and the next
invocation runs. Source changes or resource guards stop the remaining campaign.
Resource guards monitor the selected CPU quota/affinity, a 512 MiB output disk
reserve, and a readable cgroup memory reserve of 256 MiB. They are conservative
stop conditions, not a guarantee against every host or GPU OOM. A guard terminates
only this campaign's current child process group, never other GPU workloads.
The owned group is cleaned even if its leader exits first; surviving descendants
make the invocation fail rather than continuing to write after evidence is frozen.

A complete exit requires all selected expanded rows and the reported checks,
process exits, source identities, and thread verification to pass. Dirty/unknown
source remains incomplete. Success does **not** establish exhaustive API coverage.

## CPU thread profiles and bounded scaling

Additional thread settings are separate **NumPy-only** profiles, never silent
reruns of all GPU jobs:

```bash
python -m dev.benchmarks.run_qualification_campaign --cpu-threads 1 2 4 8
```

Baseline 1 is mandatory. Each optional profile must fit both scheduler affinity
and the floor of the actual cgroup CPU quota. A fractional quota below one may
throttle the single baseline thread, but does not permit extra threads. An unreadable Linux quota permits only the
single-thread baseline. Thread environment variables are set before starting the child interpreter. The child
then checks actual `threadpoolctl` pools after importing the runner and again
after running it; Torch profiles also set and verify intra-op/inter-op counts.
Requested thread counts are resource ceilings, not an assertion that every pool
used exactly that many threads. Actual counts and equality flags are retained.
Never combine profiles by selecting the fastest CPU result as an unstated baseline.

Medium and large profiles require explicit suites and at most eight exact case
IDs per selected suite. CV has no per-model selector, so its six-model contract
stays intact. Example plan, which still does not execute:

```bash
python -m dev.benchmarks.run_qualification_campaign \
  --scale large --suites learning-families six-family-cv \
  --cases learning-families=pca,kde --backends numpy cupy torch
```

All 44 learning identities can now be selected at explicit shapes, but they do
not share one scale envelope. The campaign rejects requests beyond a case's
resource/oracle contract; it does not silently shrink them:

| Learning work class | Maximum n | Maximum nominal p | Representative members |
| --- | ---: | ---: | --- |
| Linear/streamed fixtures | 2000 | 64 | PCA, k-means, Nystroem and other bounded fixtures |
| Dense factorization | 1024 | 16 | Kernel ridge, KernelPCA, thin-plate spline, agglomerative, DBSCAN |
| Iterative embedding | 512 | 16 | t-SNE, UMAP, UMAP NNDescent |
| Iterative factorization | 2000 | 32 | NMF and minibatch NMF |
| Bandwidth search | 1024 | 4 | UCV, BCV, SJ variants and bandwidth CV |
| Kernel CV | 512 | 16 | KernelRidgeCV |
| Bootstrap | 2000 | 4 | Registered KDE bootstrap fixture |
| Selection | 2000 | 64 | Fixed/model knockoff; selected p comes from `--knockoff-p` |

These are reviewed harness bounds, not claims about estimator scalability or
hard peak-memory guarantees. Learning also requires n ≥ 2p and an n² bound of
4,000,000 elements; `--max-pairwise-elements` can tighten but not enlarge that
bound. One-dimensional fixtures retain their actual effective p in row identity.
Model/statistical/workflow n is capped at 2048; quantile references stop at 2000.
CV n is capped at 2048, not an implicit 5000/10000 run. Gaussian and
classification retain their existing upper bounds of n=32768 and n=100000.
Optional `--warmup` and `--repeats` are explicit, bounded and recorded.

### Expanded profiles, explicit seeds and shapes

`--profile expanded` plans three candidate shapes per suite and three independent
seeds (17, 42, 20261020), splitting non-CV cases into at most eight per job. It
retains the six-model CV contract. Its exact job and row totals come from the
plan, not from multiplying the default 1692 rows. Candidate case/shape pairs
outside the reviewed bounds appear in `capability_exclusions`, not as passes.
Explicit unsupported shapes or case requests fail rather than silently removing
requested work.

```bash
python -m dev.benchmarks.run_qualification_campaign \
  --profile expanded > /tmp/statgpu-expanded-plan.json

python -m dev.benchmarks.run_qualification_campaign \
  --suites learning-families --cases learning-families=tsne,umap \
  --shapes learning-families=256x4,512x4 --seeds 17 42 \
  > /tmp/statgpu-embedding-plan.json
```

`--seeds` accepts up to 32 distinct seeds and `--shapes` up to eight distinct
`NxP` pairs per selected suite. Classification uses `Nx0`. Explicit shapes cannot
be combined with `--scale medium` or `--scale large`. Dtypes, weights, resamples,
targets, Gaussian alpha, knockoff feature counts/selection trials and device
ordinal have suite-specific contracts; a flag is not evidence that every suite
supports that axis. Unsupported requests are rejected. Knockoff selection seeds
are independent trials, not timing repeats, and must not be multiplied again by
multiple campaign seeds.

For deterministic partitions, use the same options with `--batch-count K` and
`--batch-index I` (zero-based). Each batch records both its selected totals and
the all-batch denominator. Job identities stay stable across partitions. Use a
fresh output directory per batch, retain every plan, and reconcile all expected
slots once; a successful individual batch is not completion of the whole matrix.
Each expected slot carries its row, configuration and source-content identity.
These plans still do not execute any benchmark without `--execute`.

### Named runner configurations

The campaign's repeated `--config SUITE:NAME=JSON` option enumerates up to 16
named profiles per suite. This is a JSON object passed as one CLI argument,
not the unified coordinator's configuration-file option. Profile names are
unique within a suite; values are checked against each selected case's supported
fixture and oracle. Unknown keys, duplicate keys and unsupported combinations
fail before execution. Named dtype/weight values cannot be combined with the
same global `--dtypes`/`--weights` axis.

```bash
python -m dev.benchmarks.run_qualification_campaign \
  --suites model-families --cases model-families=glm:gaussian,glm:poisson \
  --config 'model-families:baseline64={"dtype":"float64","solver":"irls"}' \
  --config 'model-families:weighted64={"dtype":"float64","weights":"analytic","solver":"irls"}' \
  --seeds 17 42 > /tmp/statgpu-glm-config-plan.json

python -m dev.benchmarks.run_qualification_campaign \
  --suites six-family-cv \
  --config 'six-family-cv:fivefold={"folds":5,"grid_size":3}' \
  > /tmp/statgpu-cv-config-plan.json
```

Current named-profile keys and boundaries are:

- Model families: `dtype`, `weights`, `solver`, `sparse`. Float32 inputs are
  limited to GLM/GLM-inference/OLS-inference fixtures. Analytic weights are
  aligned for coefficient-only ordinary GLM and OLS/WLS inference, not weighted
  GLM covariance. Explicit solver menus depend on the case and penalty.
- Learning families: `dtype`, `weights`, `solver`, `sparse`, `covariance_type`.
  Float32 fixtures cover PCA, incremental PCA, truncated SVD, k-means/minibatch
  k-means, DBSCAN, agglomerative and Gaussian mixture. Explicit PCA solvers are
  full/covariance/randomized; Gaussian mixture covariance choices are
  diag/full/tied/spherical. Analytic weights are not admitted by this runner;
  public KDE has its separate weighted protocol.
- Six-family CV: `dtype`, `weights`, `sparse`, `folds`, `grid_size`. This remains
  a float64, unweighted, dense accuracy baseline. Folds range from 2 to 5 and
  grid size from 2 to 9, with at most 27 candidate-fold fits before final refit.
  Changing folds/grid does not create a mixed-precision default-performance lane.

The reviewed configuration adapters currently require dense inputs. A sparse
request is an explicit unsupported capability, not permission to densify it or
claim CSR/CSC coverage. Input precision is always separate from observed compute
precision. These menus expand selected fixtures; they do not imply that every
public solver, covariance, family or weight combination is covered.
A schedulable configuration is not a promised numerical pass. For example,
randomized PCA is still tested against the exact rank-2 reconstruction gate;
higher-dimensional approximate results may fail that unchanged accuracy target.
Convergence warnings in weighted solver configurations likewise invalidate timing,
even when coefficient errors alone pass. Such rows remain failed, not excluded
after observing their result.

## Independent runners and configuration gaps

The audited 41 independent files currently have 25 executable adapters (14 ready and 11 conditional on external dependencies),
nine blocked admissions, six unsupported contracts and one deduplicated CV-models entry already used by the six-family
campaign. “Ready” means an execution/validation contract exists, not that the
workload has passed on the target hardware or covers its entire parameter space.
`independent_benchmark_catalog.json` retains audit hashes and wrapper lineage;
current execution uses fresh source fingerprints rather than treating historical
audit hashes as new-run evidence. New support files are reported separately
under `new_since_audit`, not added to the historical 202-file denominator.

Inspect all independent dispositions or a selected lane without executing:

```bash
python -m dev.benchmarks.run_independent_benchmarks \
  > /tmp/statgpu-independent-plan.json
python -m dev.benchmarks.run_independent_benchmarks \
  --lanes public-native-kde-bootstrap > /tmp/statgpu-kde-pilot-plan.json
```

Independent execution requires explicit `--lanes`, `--execute`, a fresh external
`--output-root`, `--expected-sha` containing the exact reviewed 40-character
commit, and `--environment-id`. `--validation-tier` is one of `local-minimal`,
`local-full` or `remote-full`; it records context and never causes a remote
connection. Required packages and physical backends must already be available.
The executor does not install dependencies, use SSH or silently fall back to CPU.

`independent.configurations` in the unified JSON maps selected lane IDs to lists
of bounded overrides, as shown above. The standalone equivalent is
`--configurations /path/to/configurations.json`, where the file contains only
that mapping. Each lane accepts 1–128 configurations; exact duplicates are
rejected and each normalized configuration gets its own identity and expected
cells. Defaults are bounded pilots, not automatic full parameter sweeps.

Public KDE overrides include n, query/bootstrap counts, dimension, kernel,
numeric/string bandwidth, weight pattern, seed and warmup/repeats. Here the
weight label `sparse` means zero-containing weights, not a sparse input matrix.
Other adapters expose only their reviewed axes, such as Exact ties sizes/scenario,
strata counts, panel shapes, hybrid cases or compile shapes/cases. Unknown keys,
unsupported values and work beyond combined resource bounds are rejected; there
is no arbitrary command or extra-arguments passthrough. A larger plan still
needs review of its total workload and statistical/oracle meaning.

The adapters run serially with one CPU thread, 900-second per-lane budgets or
1800 seconds for the compile/Fama–MacBeth protocols, further limited by the
requested `--timeout`. They check a 1 GiB free GPU/available-host-memory reserve
in addition to the inherited cgroup/disk guards. These are admission/stop guards,
not peak-memory or OOM guarantees. Inspect every adapter's actual shape and
protocol: a bounded Fama–MacBeth lane is not a tiny KDE workload.

Each lane retains `raw.json` when produced, `qualified.json`, process logs,
source snapshots and available runtime reports. A complete timing-only protocol
can still have `correctness_status=unavailable`; device availability is not
executed numerical-device proof. Blocked/failed lanes remain in the summary and
all results remain performance-ineligible. The raw artifacts are never rewritten
to make validation pass.

Independent lanes include public KDE bootstrap, hybrid-phase diagnostics,
Cox ties/strata/cluster and stability studies, panel covariance and Fama–MacBeth
scaling, inference and quantile checks, group/penalized GLM comparisons,
unsupervised and external-package comparisons, Torch compile cold/steady-state
work, memory-cleanup studies, and repeated-seed statistical calibration. A lane
must declare its output contract, dependencies, expected rows and resource bounds
before it becomes executable through a common adapter. A CLI or a `main` guard
alone is not a safety or completeness contract.

Do not loop over all benchmark files, import legacy programs to discover their
CLI, or invoke a default SSH launcher as if it were a local runner. Programs with
fixed output locations, import-time workloads, unbounded defaults, missing
structured artifacts or unreviewed subprocess behavior need adaptation or an
explicit blocked disposition. External R/package comparisons and multi-GPU/MPS
checks also require their own dependencies and supported hardware.

Configuration coverage is finite and algorithm-specific, not an indiscriminate
Cartesian product. Track legal combinations and intentional unsupported cases
separately. In particular:

- Solver, loss/family, penalty, covariance/inference, weights, sparse format,
  folds/grid/refit, backend/device, seed and scale require explicit axes. A fixed
  registered case does not prove all public parameter combinations.
- Repeated seeds exercise numerical and stochastic behavior, but a small seed
  matrix is not a coverage, power or false-discovery-rate calibration study.
- A dense workload says nothing about CSR/CSC support. Zero-containing weights
  are not sparse-matrix coverage. Input float32 does not prove float32 compute.
- Scale bounds and oracle bounds are independent. In particular, a large
  quantile fixture cannot silently cross its dense-LP reference limit of n=2000;
  it needs an appropriate oracle or must remain blocked.
- Learning algorithms excluded from a larger profile remain coverage gaps.
  Pairwise, quadratic, bootstrap and optimization workloads need algorithm-aware
  budgets, not only a shared n limit. Larger CV, query counts, bootstrap counts
  and Monte Carlo repetition counts are separate workload decisions.

### Keep public KDE and experimental protocols separate

The public native KDE bootstrap runner measures the public interval workflow.
Controlled-index developer kernels, RNG-inclusive experiments and hybrid-phase
measurements have different scopes; their timings are not interchangeable or
three independent proofs of the same public API. Phase timings must not be added
or subtracted to manufacture an unmeasured end-to-end result.

Kernel, dimension, numeric/string bandwidth, weight pattern, query count,
bootstrap count and workspace policy are separate axes. Cosine and optcosine
kernels are one-dimensional only. A numeric bandwidth is a covariance factor
(H = b²Σ); native string selectors refit per draw, while the NumPy
one-dimensional Gaussian fast path fixes its initial factor. The same seed does
not produce paired draws across backend RNGs. Treat selector and backend
algorithm differences as distinct protocols rather than equal-work speedups.

## What the evidence can and cannot establish

Every job has a raw artifact plus an independent qualification sidecar. The
sidecar compares the exact expanded identity **multiset**, not only case names:
backend, shape, input dtype, seed, weights/scenario where applicable, and every
expected duplicate count. It checks timing sample counts against the selected
CLI repeats. Failed/partial dispositions cannot be replaced by a successful
process exit. The original JSON is never rewritten to repair a result.

Input float32 is not evidence of float32 computation. Classification rates,
Gaussian promotion, host orchestration, diagnostic accessors, learning host/hybrid
stages, and the CV explicit-float64 accuracy baseline remain distinct. Output on
CUDA is not proof that the whole algorithm ran natively on CUDA. Scoped numerical
traces and synchronized devices must be reviewed with each producer's contract.

Clean public-fit CV timings remain separate from instrumented selector/refit
profiles. Process peak RSS is recorded in bytes using the platform-normalized
child high-water mark. It covers imports, reference work and runner execution,
not an individual algorithm's memory. **GPU allocator peaks are `not_measured`**;
pairwise safety estimates are not measured VRAM. Peak RSS may be unavailable if a
child is killed before its finalizer, and is not fabricated from a parent reading.

All performance results remain `diagnostic_only`, with
`UNVERIFIED_EXCLUSIVITY`, no speedup comparisons, and no canonical dashboard
promotion. Unknown or Docker-namespace-hidden GPU PIDs do not block correctness,
but do not prove exclusivity either. The campaign does not kill NVML-listed PIDs,
mount host namespaces, or try to establish ownership through privileged access.

## Failure-preserving evidence layout

The fresh output contains:

- `plan.json`: exact selected commands, shapes, expanded identities and limits;
- `source-before.json`, `source-after.json`, and source/script copies with hashes;
- `child_bootstrap.py` and parent runtime metadata;
- one job directory containing `invocation.json`, original `raw.json` when produced,
  combined stdout/stderr, process exit/timeout information, source snapshots,
  actual thread/runtime reports when available, and `qualification.json`;
- `summary.json` and `SHA256SUMS.json`.

Normal completion, ordinary failure, source mutation, resource stop, and caught
exceptions all finalize the collected evidence. Files are made read-only and a
sibling `OUT.tar.gz` plus `OUT.tar.gz.sha256` is produced without overwriting
existing evidence. Hashes make later changes detectable; read-only permissions
are not a cryptographic or administrator-proof storage guarantee. A machine
power loss, forced parent kill, disk failure, or inability to write the archive
can still prevent finalization.

Harness validation, without any numerical or physical GPU campaign:

```bash
python -m pytest dev/tests/test_qualification_campaign.py -q
```

These tests use synthetic artifacts and pure-Python subprocess stand-ins. Real
correctness/performance evidence still needs a separately authorized run on the
intended source and hardware. Historical successful runs belong to their recorded
source and environment: they do not validate changed code, a larger plan, a new
adapter, or an unexecuted backend. See the [benchmark guide](../benchmarks.md) and
[device and memory guide](device-and-memory.md) for the wider context.

## Regression gates for the configuration extension

Before executing newly configured protocols, run the ordinary control/oracle tests:

```bash
python -m pytest -q -rs dev/tests/test_campaign_configuration_matrix.py dev/tests/test_registered_benchmark_axes.py dev/tests/test_full_benchmark_matrix.py dev/tests/test_independent_benchmark_registry.py
```

The additional physical suite has 58 single-GPU parameter nodes, covering CuPy
and Torch weighted model input precision, PCA solvers, GMM covariance modes, and
all six CV families under two folds/four candidates (six families checked inside
each of the two backend test calls). This includes 20 additional inference
combinations across covariance type, input precision, weights and backend; the
existing four weighted HC3 combinations remain in the original group:

```bash
python -m pytest -q -rs -x dev/tests/test_registered_benchmark_axes_physical.py
```

Every one of these 58 call phases must pass in the intended two-backend CUDA
environment. A missing-backend skip is useful local feedback, not physical
acceptance. This suite supplements the earlier device/clean-timing tests; it is
not a replacement or proof of all configurations. Pin the collected node list
to the actual source before running. Passing this suite does not establish that
the expanded configuration campaign or independent runners have been executed.

## Explicit execution-count budgets

Plan-only may enumerate a large matrix. Execution defaults to at most 21
invocations and never accepts a per-dispatch budget above 64. Use bounded batches
instead of launching the whole expanded plan at once:

```bash
python -m dev.benchmarks.run_qualification_campaign --profile expanded \
  --batch-count 48 --batch-index 0 --execute \
  --output-dir /tmp/statgpu-expanded-batch-00
```

Use a new output path for each batch; reconcile all global slot identities before
claiming completion. `--max-invocations` is an explicit count budget, not runtime
or memory proof. The independent dispatcher has the same default21/maximum64
count bound; the unified JSON uses `independent.max_invocations` for that lane.
Each child also remains subject to its planned timeout/resource bounds. Statistical
resamples are restricted to 2–1000. CV n=5000/10000 and unreviewed large quadratic
work remain blocked, not enabled by increasing the invocation count.
