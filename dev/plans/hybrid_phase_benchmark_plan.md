# CPU / current-hybrid phase benchmark

## Scope and honest interpretation

This standalone optimization-decision suite does not change production kernels,
replace PR #251 repair acceptance, or establish canonical dashboard eligibility.
It compares existing public CPU and current GPU/hybrid workflows. No native-GPU
prototype has been implemented or timed. Do not migrate implementations on the
basis of GPU residency alone.

Matrix: KDE bootstrap; bandwidth NRD, NRD0, UCV, BCV, SJ, SJ-STE, SJ-DPI and
kernel-regression CV; fixed-X and model-X knockoffs with corr_diff. Float64 only.
KDE is univariate; bandwidth and knockoff honor --p. Seed, n, p, query count,
bootstrap count and data fingerprint are recorded per row. Small/medium/large
presets are configurable, never silently reduced. Model-X preserves the public
corr_diff default of three native random draws; Fixed-X uses one construction.

CPU and GPU bootstrap use numeric factor 0.4 and identical NumPy resampling
indices/weights. CPU has an inline Gaussian-density fast path whereas GPU
refits each replicate. Both are checked against SciPy densities for every same
realized resample and interval quantiles. This checks computation, not statistical
interval coverage. String bandwidth bootstrap is deliberately excluded because
CPU and GPU refit semantics can differ. Torch weighted bootstrap remains an
explicit unsupported row, not PASS or CPU fallback.

Bandwidth selected factors and predictions are checked against the current CPU
public implementation using the same input and method. This is a trusted CPU
parity gate, not independent global optimality certification of UCV/BCV/SJ/CV.

Knockoff backend RNG streams differ. Validate every realized draw's corr_diff,
average the three model-X statistics, independently enumerate the knockoff+
threshold, and check Fixed-X Gram identities. Do not compare W between backends.
Neither empirical FDR nor full model-X sampler distribution is certified. All
knockoff timings remain diagnostic and no knockoff speedup ratio is emitted.

## Timing and evidence

Three separate executions are used:

1. Native primitive/device evidence via NumericalTrace, outside performance timing.
2. Instrumented stage/selected transfer-helper diagnostics, with synchronization
   on the concrete verified runtime device at each observed boundary.
3. Uninstrumented end-to-end public calls, warmup then repeated measurements.
   Input conversion, public computation, internal transfers and result extraction
   to host are included. Data generation, reference calculations and checks are
   excluded. Every measured result is checked after stopping its timer.

Phase spans are inclusive and overlap. Their totals must NEVER be summed into an
end-to-end decomposition or subtracted to infer a residual/core time. Boundary
sync samples are observer-added waits, not production synchronization overhead.
Transfer-helper durations include allocation, dispatch and waits, not pure copy
cost. Python hooks do not enumerate every transfer or kernel. CPU bootstrap's
inline fast density loop, host RNG and quantiles are not separately isolated.
This limitation is explicitly machine-readable; use a dedicated GPU profiler
later if finer attribution is necessary. Instrumented values are not speedup
inputs. Missing arithmetic/device evidence blocks the row.

Ratios require passing CPU and GPU rows with the same workload in the same run,
matching end-to-end scopes and unchanged source. GPU-only invocation gives no
ratio. There is no cross-machine aggregation or crossover interpolation. Runtime
and convergence warnings fail rows. No completed row means no successful claim.
A run including unsupported Torch bootstrap exits 2 (incomplete), by design.

## Commands

Run from repository root with the package/dependencies already installed. Keep
thread settings identical between comparison runs. No remote execution is
performed by these commands.

```bash
python -m dev.benchmarks.benchmark_hybrid_phases --inventory
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m dev.benchmarks.benchmark_hybrid_phases \
  --preset tiny --backends numpy --warmup 1 --repeats 3 \
  --output results/hybrid-tiny-cpu.json
python -m pytest dev/tests/test_benchmark_hybrid_phases.py -q
```

On the physical P100 host, explicitly select all backends for the support matrix:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m dev.benchmarks.benchmark_hybrid_phases \
  --preset small --backends numpy cupy torch --device 0 \
  --warmup 2 --repeats 7 --output results/hybrid-small-matrix.json
```

Expected overall incomplete because Torch bootstrap is unsupported. To obtain a
complete supported KDE comparison, select --cases kde_bootstrap --backends numpy
cupy. For other families all three backends can be explicitly requested. Repeat
at medium and large with separate output paths; inspect resource budgets first.
The default --max-work 500000000 blocks large CV (n*n*p*100 estimate exceeds it).
Review available memory/time before explicitly increasing this ceiling; no
budget estimates guarantee actual memory safety, no OOM fallback/downscale exists.
Use --n, --p, --queries, --resamples for measured crossover points, retaining
identical workload across backends. Existing outputs are protected unless
--overwrite is explicitly provided.

## Validation boundary

Local CPU tests and tiny CLI runs validate harness behavior, not P100 performance.
Optional physical tests skip when CuPy/Torch CUDA is unavailable; skips are never
physical evidence. New source, helper and trace dependencies are fingerprinted;
source mutation invalidates all performance eligibility. GPU measurements on this
new runner remain pending until actually executed. Do not reuse PR #251's earlier
9a6145d physical results as evidence for this new benchmark.
