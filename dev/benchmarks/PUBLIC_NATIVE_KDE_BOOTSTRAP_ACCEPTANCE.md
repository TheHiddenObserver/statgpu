# Public native KDE bootstrap acceptance

This runner evaluates the production `kde_bootstrap_confidence_interval` API.
It does not substitute the earlier development-only fixed-index or native-RNG
candidates. Those experiments remain separate and their historical GPU results
must not be presented as evidence for the public implementation.

## Active public capability

- NumPy, CuPy, and Torch; Torch CPU is a separate numerical tier, never CUDA proof.
- Numeric bandwidth factors and `scott`, `silverman`, `nrd`, `nrd0`, `ucv`, `bcv`,
  `sj`, `sj-ste`, and `sj-dpi` string selectors.
- Gaussian, rectangular, triangular, Epanechnikov, biweight, triweight, cosine,
  and optcosine kernels. Cosine and optcosine remain one-dimensional only.
- One and multiple dimensions, absent/unequal/zero-containing weights, list and
  NumPy ingress, backend arrays, and noncontiguous views.
- Host NumPy result arrays, retained or omitted bootstrap samples, and the
  general `kde_confidence_interval(method='bootstrap')` wrapper.

Weights are sampling probabilities and are applied again as fitting weights on
sampled rows. Every native replicate refits a string selector. Native R-style selection uses
stable sorting for ties and a canonical principal-axis sign; legacy CPU sorting
and eigensolver orientation remain unchanged. Weighted interpolation can therefore
produce materially different selector factors even for the same rows. The historical
NumPy 1D Gaussian shortcut instead fixes the originally selected factor; numeric
factors remain fixed everywhere, although resampled covariance changes absolute
kernel widths. These are pointwise percentile intervals, not simultaneous bands,
bias-corrected intervals, or a universal validity guarantee for weighted designs.

Native RNG uses a local generator. `None`, nonnegative integers (not restricted
to 32 bits), and `numpy.random.Generator` are supported by the seed bridge. The
bridge consumes exactly four uint32 words from `numpy.random.default_rng`;
a supplied generator advances by that amount. Backend streams need not match
one another or the historical NumPy stream. CDF inversion uses native float64
uniforms and `searchsorted`; zero-mass rows are excluded, but probabilities below
float64 cumulative resolution can collapse. Query chunk size does not change
the RNG call schedule.

## Numerical acceptance

`dev/tests/test_public_native_kde_bootstrap.py` includes:

1. Fixed-index comparisons to NumPy refits for all kernels, selectors,
   dimensions, and weight families. Selector × kernel interactions cover Gaussian
   and Epanechnikov. Native selector oracles independently recompute centered
   projection, variance and stable-tie resampling from the captured principal
   axis, then use NumPy scalar rules. They verify the axis is a normalized
   principal eigenvector and do not use the captured selected factor as truth.
   Historical CPU selector behavior is preserved, not forced to agree. Native production's `_draw_indices` hook is patched only in
   fixed-index tests, not in the public API or timing runner.
2. Independent centered weighted covariance/jitter/Gaussian mixture algebra for
   numeric factors. No production covariance or density implementation is used
   by that oracle.
3. Actual RNG reproducibility, global-state isolation, arbitrary-width seeds,
   four-word Generator consumption, and chunk invariance.
4. Public result types/shapes, wrappers, input immutability, invalid arguments,
   singleton percentiles, workspace rejection, and an isolated copied-package run without `dev` (not a built-wheel installation test).
5. Harness rejection checks: missing sampling, wrong device, altered native
   primitives, host-array conversion, incorrect output, and absent physical GPU
   evidence must not become a successful GPU/performance claim.

The broad refit tolerance is `rtol=3e-5, atol=3e-7` for selectors, allowing small
optimizer/PCA reduction differences; numeric test tolerances are tighter
(`rtol=3e-8, atol=2e-10`). Benchmark output records its own explicit thresholds
(`rtol=3e-5, atol=3e-8`), metrics, and reference type. These are fixture-specific
acceptance thresholds, not universal statistical error bounds.

## Timing and provenance

Default protocol is two warmups plus seven repeats, reporting the median and all
raw samples. The timed region includes input H2D conversion, original fitting,
all native RNG/index generation, each bootstrap refit, quantiles, and public
NumPy returns. Data generation, independent oracle construction, realized-index
capture, operation traces, and output validation are outside the measured region.
CUDA is synchronized immediately before and after timing on the concrete device
recorded by the executed runtime and verified by diagnostic numerical operands.

Each backend is validated against its own actual captured indices. Equal seeds
are not assumed to mean equal draws. A second capture checks exact within-backend
reproducibility, and an uninstrumented call must reproduce those densities and
bounds. Sampling support and one seeded empirical-CDF DKW diagnostic are checked;
this is not full RNG certification.

Operation evidence observes genuine installed-backend calls to RNG generation,
CDF summation, searchsorted, weighted covariance reductions, inversion, and
quantiles inside source-identified invocations. Gaussian density additionally
requires observed native `exp`. Host-array numerical calls and listed full-array export hooks are guarded
during native stages. Torch hooks include cpu/numpy/tolist/__array__/to(cpu).
CuPy guards asnumpy and probes whether direct ndarray.get is observable by the
installed extension's profiling protocol; that capability is recorded, not
assumed. A complementary direct-export-name source audit covers selected
production numerical functions. This finite guard/audit is not an exhaustive
transitive data-flow proof. Scalar diagnostics and the small seed bridge are allowed. This is an operation observer, not a CUDA profiler or proof about
every overloaded operator/fused kernel. Non-Gaussian density operators are not
exhaustively traced; their numerical output still requires the refit oracle.

Source records include commit, clean/dirty state, exact imported paths, hashes
of production and validator sources, and a deterministic content fingerprint.
Source and data identities must remain unchanged throughout execution. Canonical
performance evidence additionally requires a clean fully tracked tree, all three
backends including real CuPy/Torch CUDA, and the full timing protocol. A partial
run exits 2 and emits no eligible speedup ratios. String-selector 1D Gaussian rows and all R-style-selector rows never emit
ratios because CPU/native refit or tie/projection policies can differ.

The optional `workspace_bytes` argument estimates managed bootstrap buffers,
not a total VRAM/process memory cap. Known selector arrays have an explicit conservative reserve. Caller inputs,
the original fitted model, allocator caches, and library/RNG/eigensolver scratch
remain outside that estimate. The benchmark neither claims peak-memory measurements nor silently
reduces the requested workload.

## Commands

Local harness and Torch-CPU smoke, explicitly not physical GPU evidence:

```bash
python -m pytest dev/tests/test_public_native_kde_bootstrap.py -q
python -m dev.benchmarks.benchmark_public_native_kde_bootstrap \
  --preset tiny --cpu-only --torch-cpu \
  --output results/public-native-kde-local.json
```

Mandatory physical acceptance before any performance claim:

1. Use the exact clean, fully tracked checkout. Record `git rev-parse HEAD`,
   `git status --porcelain`, package versions, and the result JSON source hashes.
   The status output must be empty; do not substitute a stale local patch.
2. Probe the actual target device, including its ordinal, model (P100 if that is
   the designated target), UUID, total memory and driver. Verify both CuPy and
   Torch see and execute on that concrete CUDA device. A successful import or
   Torch CPU run is not the probe. Keep probe output with the evidence.
3. Record a bounded CPU/BLAS thread budget before starting Python, for example
   `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Keep the same
   budget across the tests and CPU/GPU timing rows; the JSON records these
   environment variables and available thread-pool information.
4. Run all five capability suites and retain JUnit plus the terminal summary:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q -ra -o junit_family=xunit1 \
  dev/tests/test_native_kde_selectors.py \
  dev/tests/test_public_native_kde_bootstrap.py \
  dev/tests/test_torch_weighted_kde_native.py \
  dev/tests/test_native_kde_projection_warning.py \
  dev/tests/test_torch_cpu_cupy_kernel_ingress.py \
  --junitxml=results/public-native-kde-physical-tests.xml
```

The cross-library ingress suite is mandatory: its real CuPy cases exercise
Torch CPU samples, queries and weights, including logical views and dtype
conversion. Its CPU conversion spies are boundary contract tests, not evidence
of GPU execution. Its two-device case may skip only under the hardware rule
below. Keep the existing public bootstrap ingress cases in the same run.

Inspect test identities and skip reasons, not just the aggregate count. Every
required single-GPU CuPy and Torch-CUDA case must actually execute with zero
unexpected skips or failures. Multi-device tests may skip on a single-device
host under their explicit hardware contract. Deliberate NumPy/CPU-tier contract
skips are recorded separately and do not excuse skipped GPU capability cases.
Do not predeclare fixed pass/skip counts before the exact source has executed.

The only expected new `RuntimeWarning` is the explicitly defined native PCA
ambiguity diagnostic: a repeated or numerically indistinguishable largest
eigenvalue makes the selected principal axis backend-dependent. The diagnostic
covers exact, near, and zero-scale eigenspaces; it retains native computation
rather than silently switching to CPU. Its relative threshold is the recorded
`64 * float64 epsilon * dimension`, without an absolute scale floor.

An affected successful public bootstrap call emits one aggregate warning, not
one warning for every replicate. Retain its text and the result's
`metadata["projection_diagnostics"]`: status, total count, mutually exclusive
exact/near/zero-scale counts, and minimum relative eigengap. The warning suite
records backend, emitted warning text, and JSON diagnostics as JUnit properties;
retain these properties with the `xunit1` report above. Verify the count
matches the original fit plus affected replicate projections and resets between
calls. Calls whose projected fits all have distinct leading eigenvalues,
numeric-bandwidth calls, and legacy NumPy cases must not acquire this new warning.
Accept this diagnostic only when warning-specific assertions or recorded
eigengap/count metadata substantiate it. Do not blanket-ignore `RuntimeWarning`,
use a broad warning whitelist, or reinterpret unrelated warnings as expected
success. Any unexpected warning requires investigation and resolution before
acceptance.

Only after this gate passes, run the synchronized public benchmarks from the
same source, environment, concrete device and recorded thread budget:

```bash
python -m dev.benchmarks.benchmark_public_native_kde_bootstrap \
  --preset small --device 0 --weights unequal \
  --output results/public-native-kde-small.json
python -m dev.benchmarks.benchmark_public_native_kde_bootstrap \
  --preset small --device 0 --dimension 2 --bandwidth sj-dpi --weights sparse \
  --output results/public-native-kde-sj-dpi-2d.json
python -m dev.benchmarks.benchmark_public_native_kde_bootstrap \
  --preset gpu --device 0 --weights none \
  --output results/public-native-kde-gpu.json
```

Repeat representative cases with compact kernels and singleton queries. Never
move a requested scale upward merely to manufacture a GPU speedup. Physical
GPU commands above are an acceptance recipe, not a claim that they ran here.
No new physical GPU performance result is asserted by this document.
