# Experimental native CuPy KDE: physical acceptance recipe

This is a reusable validation recipe, not a claim that every later head has
been physically tested. The candidate is developer-only and has not replaced
public bootstrap defaults. See the exact-head core validation record and
[scope and statistical contract](../plans/native_gpu_kde_bootstrap.md).

## 1. Freeze and identify the target

Use a clean, non-shallow checkout of the new PR's exact published head, record
`git rev-parse HEAD`, tree, status, Python/package versions, CPU affinity/thread
limits, GPU model/UUID, driver/runtime, and confirm `statgpu.__file__` belongs to
that checkout. Preserve identity before and after tests and benchmarks. Do not
reuse PR #251 GPU results as proof of this new candidate.

Run an actual float64 CuPy arithmetic/synchronization probe on the requested
single CUDA device. An import or a device-resident input array alone is not
physical computation evidence. Do not install a new GPU runtime or use a remote
server without the applicable environment/resource authorization.

## 2. Correctness before timing

```sh
python -m pytest -q dev/tests/test_native_kde_bootstrap.py \
  dev/tests/test_benchmark_native_kde_bootstrap.py \
  dev/tests/test_inference_kde.py --junitxml=native-kde-tests.xml
```

Check JUnit parameter identities, not just the total passed count: all applicable
single-CuPy cases must execute without skip. The explicit two-device rejection
test may skip on a single P100 and is outside this single-device gate. Missing
CuPy/CUDA means incomplete, not PASS. Keep failures and warnings unchanged.
The CPU algebra checks are useful but do not constitute physical GPU evidence.

## 3. Controlled, same-index comparison

Keep the GPU exclusive and record actual CPU/BLAS thread limits. Start with a
small correctness workload, then a representative GPU-scale workload. No
`--allow-incomplete` exists. Default 2 warmups and 7 repetitions are required for
performance eligibility; never lower thresholds to turn a failure into PASS.

```sh
python -m dev.benchmarks.benchmark_native_kde_bootstrap \
  --preset small --weights unequal --warmup 2 --repeats 7 \
  --output native-kde-small.json
python -m dev.benchmarks.benchmark_native_kde_bootstrap \
  --preset gpu --weights unequal --warmup 2 --repeats 7 \
  --output native-kde-gpu.json
```

The GPU preset is n=4096, q=512, B=100. If already within the approved physical
run budget, a second useful point matches the earlier query-size advantage
region: `--n 2000 --queries 1024 --resamples 100`. Do not infer that small defines
GPU suitability. For a CPU comparison claim, repeat independent CPU-thread
profiles within the machine's quota and compare to a defensible best CPU
baseline, not arbitrarily to its slowest thread count. Preserve each profile's
identity and samples rather than mixing environment groups.

All three variants use exactly the same realized row indices and weighted
refit semantics. The runner checks every original estimate, draw and percentile
bound. It measures controlled-index workflows, not the unchanged public
bootstrap API. RNG is excluded consistently; required data/index transfers and
host outputs are included. CPU and current CuPy references call `fit_kde` for
every replicate, so do not equate the CPU reference with the public NumPy
bootstrap's specialized fast path. Consequently these ratios alone are not a
claim of public-API speedup or evidence for replacing its default implementation.

The JSON must distinguish the separate numerical-operation trace and memory
instrumentation from uninstrumented synchronized timing. Require unchanged
source/input identities, real concrete-device primitive evidence, all variants
passing and a complete validation result before interpreting paired ratios.
Memory values are fresh-pool allocation high-water diagnostics, not total CUDA
process peak or proof of a hard workspace cap.

## 4. Decision boundary

Report failures and missing evidence explicitly. Near-parity ratios are not a
stable crossover. No canonical dashboard publication or production migration
follows automatically from a successful experiment. The next decision requires
numerical/device acceptance, representative-scale timing, memory observations,
and a separately reviewed native-RNG/public-API integration contract. This experiment
preserves existing bandwidth/knockoff routes and does not expand Torch bootstrap.
