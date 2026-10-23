# Native CuPy KDE RNG: physical acceptance

This recipe validates the new native-RNG candidate, not a public-default switch.
No physical result is claimed by this document. Earlier PR #252 fixed-index
results remain historical and cannot certify RNG generation or this runner.
See the [contract and resource plan](../plans/native_gpu_kde_rng.md).

## Freeze identity and environment

Use the exact published PR head in a clean, non-shallow checkout. Record full
HEAD/tree/parent, all tracked file hashes, git status and exact candidate/statgpu
import paths before and after execution. Put logs and result files outside the
checkout so source identity stays clean. Save Python/NumPy/SciPy/CuPy versions,
CUDA runtime/driver, GPU model/UUID/ordinal, CPU affinity, cgroup quota and actual
BLAS thread pools. For the core comparison use an exclusive single GPU and
THREADS=1, setting OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS before
Python starts. Verify CuPy float64 arithmetic and synchronize the actual device.
Do not silently change dependencies, inputs, tolerances or resource limits.

## Core pytest gate, then benchmarks

```sh
python -m pytest -q \
  dev/tests/test_native_kde_bootstrap.py \
  dev/tests/test_benchmark_native_kde_bootstrap.py \
  dev/tests/test_native_kde_rng.py \
  dev/tests/test_benchmark_native_kde_rng.py \
  dev/tests/test_inference_kde.py \
  --junitxml="$OUT/core.xml"
```

Stop at any failure. Collect the exact head's test IDs and audit JUnit: every
required single-device CuPy parameter must execute with no skip. Only explicitly
named mixed-device rejection tests may skip on a single GPU; missing CuPy/CUDA is
not success. Preserve warnings and all failing outputs. New RNG gates include
realized-index algebra, seed repeatability/query-chunk invariance, unchanged
global RNG stream, zero support/weighted frequencies, CDF boundary cases,
nondefault stream, retained results and fail-before-RNG budget rejection.

After all applicable tests pass, run all four points without smoke/incomplete
flags, with distinct new output filenames (the runner refuses overwrites):

```sh
for preset in small gpu; do
  for weights in none unequal; do
    python -m dev.benchmarks.benchmark_native_kde_rng \
      --preset "$preset" --weights "$weights" --device 0 \
      --warmup 2 --repeats 7 --workspace-bytes 67108864 \
      --output "$OUT/rng-${preset}-${weights}.json" || exit "$?"
  done
done
```

Small is n=256,q=64,B=30; GPU is n=4096,q=512,B=100. The 64 MiB managed workspace
includes generated indices/RNG/CDF buffers, but is not a total CUDA memory cap.
Record actual separate pool diagnostics. Do not expand n/q/B, add devices or
T=2/T=4 profiles without the applicable resource decision. Do not infer GPU
suitability only from the small point. The existing fixed-index runner remains
available as a separate diagnostic; its RNG-excluded timing is not interchangeable
with the new end-to-end timing.

## Validate JSONs and preserve evidence

Each core JSON must have `schema_version=statgpu-benchmark-v2`,
`validation.status=complete`, `validation.physical_gpu_claim=true`,
`source_changed=false`, and `performance.eligible=true`. Require exactly three
passing variants, seven positive finite uninstrumented timing samples per row,
concrete synchronization device matching actual execution, unchanged data and
source identities, and all source files tracked/imported from this checkout.
RNG diagnostic evidence, native generation/operation/host-conversion guards and
separate native memory diagnostics must pass. Use the runner's `finalize` logic
and inspect the reported uncovered reasons, not merely process exit status.

Public NumPy and public CuPy calls include RNG and return bootstrap draws; the
NumPy baseline is the actual specialized public fast path. Native timing includes
RNG construction/generation, necessary input transfers and the same four host
output arrays. Correctness uses each implementation's own actual realized indices
from an untimed diagnostic capture, followed by an independent Gaussian formula.
Do not require equal streams from NumPy/CuPy same seeds or claim that their
individual random draws are paired. Timed calls must be unpatched/uninstrumented;
CPU reference checks happen outside measured intervals. Sampling-law testing is
separate from fixed-realization algebra and does not prove confidence coverage.

Archive original command logs, exits, JUnit, all four JSONs, environment/probe and
before/after identities even on failure; include archive byte count and SHA256.
Fresh-pool memory high-water marks exclude CUDA context, cuRAND internal state
and allocations bypassing that allocator. Report numerical/device acceptance
separately from speed. All core points passing supports only this CuPy fixed-
numeric-factor experimental RNG pipeline. Native Torch, selector refits, public
opt-in/default promotion and canonical dashboard publication remain separate.
