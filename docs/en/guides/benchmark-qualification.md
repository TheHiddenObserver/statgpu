# Auditable benchmark qualification campaigns

The seven maintained coverage suites are **candidate evidence producers**, not a
certification of every public API, every CUDA operation, or release-ready speedups.
`dev/benchmarks/run_qualification_campaign.py` adds an opt-in serial campaign with
exact workload inventories, separate CPU thread profiles, and failure-preserving
archives. It does not change the numerical implementations or canonical dashboard.

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
It is not a new reduced smoke profile:

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
imports and preflight. It is not a whole-campaign time budget; 21 invocations can
take up to 10.5 hours at the maximum. There is no `--allow-incomplete` escape hatch.
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

The larger learning menu is limited to KDE, NW kernel regression, PCA,
incremental PCA, truncated SVD, k-means, minibatch k-means, Nystroem, B-spline,
natural/cyclic splines, and Scott/Silverman/NRD/NRD0 bandwidth rules. Use the exact
registry case IDs. Learning n is 1024/2000 for medium/large, with an explicit
4,000,000-element pairwise cap; no automatic all-44-case large run is available.
Model/statistical/workflow n is capped at 2048. CV n is 1024/2048, never an implicit
5000/10000 CV run. Gaussian profiles use 8192×64/32768×128, and classification
uses n=10000/100000. These selected profiles overlap some fullscan points; they
are not claims that every default case has been scaled. Optional `--warmup` and
`--repeats` overrides are explicit, bounded, and included in the inventory.

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
intended source and hardware. See the [benchmark guide](../benchmarks.md) and
[device and memory guide](device-and-memory.md) for the wider context.
