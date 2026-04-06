# Remote Inference Feature Suite Verification (2026-04-05)

## Remote Environment

- Host: hz-4.matpool.com:27613
- User: root
- Remote workspace: /tmp/statgpu_inference_full_1775387750
- Python: 3.9.16 (/root/miniconda3/envs/myconda/bin/python)
- CUDA available: true

## Scope Required by User

- Unified resampling engine: full feature coverage
- Multiple-testing/FDR module: full feature coverage

## Test Files Executed

1. dev/tests/test_inference_resampling.py
2. dev/tests/test_inference_multiple_testing.py

## Coverage Checklist

### Unified resampling engine (statgpu.inference._resampling)

- Bootstrap strategies:
  - iid
  - stratified
  - cluster
  - block
- Permutation strategies:
  - iid
  - stratified
  - grouped
- Backend modes:
  - auto
  - numpy
  - cupy
- Alternatives:
  - two-sided
  - greater
  - less
- Validation/guard paths:
  - invalid n_resamples
  - invalid confidence_level
  - missing strata/clusters/groups
  - mismatched dimensions
  - invalid strategy names
- Result object features:
  - to_dict()
  - to_dataframe() (or expected ImportError when pandas is unavailable)
- BaseEstimator thin wrappers:
  - bootstrap_statistic(...)
  - permutation_test(...)
  - cached-training-array bootstrap path
  - CUDA-array path with backend='auto'
  - explicit backend='numpy' fallback path on CUDA models

### Multiple-testing/FDR (statgpu.inference._multiple_testing)

- Supported methods:
  - bh
  - by
  - holm
  - bonferroni
- Alias paths:
  - fdr_bh, benjamini-hochberg
  - fdr_by
  - holm-bonferroni
  - bonf
- Axis behavior:
  - axis=None (flatten)
  - explicit axis (including negative axis)
- Validation/guard paths:
  - invalid p-values
  - invalid alpha
  - invalid method
  - scalar with axis != None
  - empty-input behavior
- BaseEstimator thin wrapper:
  - internal _pvalues path
  - explicit pvalues path

## Command and Result

Command:

/root/miniconda3/envs/myconda/bin/python -m pytest dev/tests/test_inference_resampling.py dev/tests/test_inference_multiple_testing.py -q -ra

Result:

- 36 passed in 4.33s
- exit code: 0

## Precision Comparison (Remote)

Data source:

- results/remote_inference_precision_timing_2026-04-05.json

### 1) Multiple-testing/FDR vs statsmodels

Vector input (axis=None):

- BH: max abs adjusted-p diff = 1.11e-16, reject mismatch = 0
- BY: max abs adjusted-p diff = 0.0, reject mismatch = 0
- Holm: max abs adjusted-p diff = 0.0, reject mismatch = 0
- Bonferroni: max abs adjusted-p diff = 0.0, reject mismatch = 0

Matrix input (axis=0):

- BH: max abs adjusted-p diff = 1.11e-16, reject mismatch = 0
- BY: max abs adjusted-p diff = 0.0, reject mismatch = 0
- Holm: max abs adjusted-p diff = 0.0, reject mismatch = 0
- Bonferroni: max abs adjusted-p diff = 0.0, reject mismatch = 0

Interpretation:

- FDR/FWER outputs are numerically aligned with statsmodels (floating-point noise level only for BH).

### 2) Resampling engine vs SciPy

Bootstrap (percentile CI, iid):

- observed abs diff = 0.0
- CI low abs diff = 0.0
- CI high abs diff = 0.0

Permutation (iid, two-sided, correlation statistic):

- statgpu p-value = 4.9975e-4
- SciPy p-value = 9.9950e-4
- absolute difference = 4.9975e-4

Interpretation:

- Difference is at one-count resolution of permutation p-values ((k+1)/(n+1) correction), still in the same significance regime.

### 3) Lasso bootstrap CPU/GPU precision

- coef max abs diff = 1.78e-15
- intercept abs diff = 4.16e-17
- bse max abs diff = 4.68e-17
- p-value max abs diff = 0.0
- conf_int max abs diff = 1.78e-15

Interpretation:

- CPU and GPU bootstrap inference outputs are numerically consistent.

### 4) Unified resampling dual-backend precision (NumPy vs CuPy)

Data source:

- results/remote_resampling_dual_backend_2026-04-05.json

Bootstrap:

- observed abs diff = 0.0
- CI low abs diff = 1.33e-3
- CI high abs diff = 4.38e-4

Permutation:

- p-value abs diff = 0.0

Interpretation:

- For the same random-seed setup, CPU/GPU outputs are numerically aligned; CI differences remain small and are compatible with backend-level RNG/ordering differences.

## Timing Comparison (Remote)

### 1) Test suite runtime

- pytest (resampling + multiple-testing): 36 passed in 4.33s

### 2) FDR timing by backend and framework

Data source:

- results/remote_inference_backend_benchmark_2026-04-05.json
- results/remote_inference_backend_benchmark_v2_2026-04-05.json
- results/remote_inference_backend_benchmark_v3_2026-04-05.json
- results/remote_inference_backend_benchmark_v4_2026-04-05.json

Workload: n=200000 p-values, warmup=1, repeats=5.

- BH:
  - statgpu NumPy: 18.842 ms
  - statgpu CuPy: 2.346 ms
  - statsmodels: 20.568 ms
- BY:
  - statgpu NumPy: 19.543 ms
  - statgpu CuPy: 2.470 ms
  - statsmodels: 21.313 ms
- Holm:
  - statgpu NumPy: 19.543 ms
  - statgpu CuPy: 2.321 ms
  - statsmodels: 37.858 ms
- Bonferroni:
  - statgpu NumPy: 18.003 ms
  - statgpu CuPy: 0.791 ms
  - statsmodels: 19.276 ms

Interpretation:

- FDR no longer has a single timing result; backend and framework timing are now both included.
- On this CUDA host, statgpu CuPy is ~8.6x to ~24.4x faster than statsmodels depending on method.

### 3) Resampling engine timing by backend and strategy

Data source:

- results/remote_resampling_strategy_external_benchmark_v1_2026-04-05.json

Workload: bootstrap n=600 / n_resamples=1200, permutation n=500 / n_resamples=2000, warmup=1, repeats=3.

Bootstrap:

- iid:
  - NumPy: 12.591 ms
  - CuPy: 1.810 ms
- stratified:
  - NumPy: 92.343 ms
  - CuPy: 2198.408 ms
- cluster:
  - NumPy: 69.444 ms
  - CuPy: 646.809 ms
- block:
  - NumPy: 30.809 ms
  - CuPy: 479.568 ms

Permutation:

- iid:
  - NumPy: 197.207 ms
  - CuPy: 3.577 ms
- stratified:
  - NumPy: 217.869 ms
  - CuPy: 2688.234 ms
- grouped:
  - NumPy: 485.961 ms
  - CuPy: 17734.204 ms

Interpretation:

- GPU implementation is completed for unified resampling (both bootstrap/permutation and all listed strategies have CuPy timings).
- Current optimization focus is IID path; with vectorized fastpath enabled, IID on CuPy is much faster than NumPy.
- Non-IID strategies (stratified/cluster/grouped) are gather-heavy and still slower on this small strategy workload; these are the next optimization targets.

### 4) Unified resampling dual-backend timing (same workload, baseline vs optimized)

Data source:

- results/remote_resampling_dual_backend_2026-04-05.json
- results/remote_resampling_dual_backend_optimized_2026-04-05.json
- results/remote_resampling_dual_backend_optimized_v2_2026-04-05.json
- results/remote_resampling_dual_backend_optimized_v3_2026-04-05.json

Bootstrap (mean statistic, n=2500, n_resamples=1200):

- Baseline NumPy backend: 33.605 ms
- Baseline CuPy backend: 1194.110 ms
- Optimized NumPy backend: 32.638 ms
- Optimized CuPy backend: 572.760 ms
- Optimized v2 NumPy backend: 32.987 ms
- Optimized v2 CuPy backend: 285.837 ms
- Optimized v3 NumPy backend: 33.647 ms
- Optimized v3 CuPy backend: 247.889 ms

Permutation (correlation stat, n=2000, n_resamples=1500):

- Baseline NumPy backend: 183.873 ms
- Baseline CuPy backend: 1498.616 ms
- Optimized NumPy backend: 183.101 ms
- Optimized CuPy backend: 1416.032 ms
- Optimized v2 NumPy backend: 177.861 ms
- Optimized v2 CuPy backend: 1221.439 ms
- Optimized v3 NumPy backend: 181.439 ms
- Optimized v3 CuPy backend: 1165.922 ms

Ratio notes:

- bootstrap CuPy speedup (baseline -> v2): 4.178x
- bootstrap CuPy speedup (baseline -> v3): 4.817x
- bootstrap CuPy speedup (v2 -> v3): 1.153x
- permutation CuPy speedup (baseline -> v2): 1.227x
- permutation CuPy speedup (baseline -> v3): 1.285x
- permutation CuPy speedup (v2 -> v3): 1.048x

Interpretation:

- CuPy remains slower on this small-to-medium workload because each resample launches GPU kernels and relies on gather-heavy random indexing; launch/synchronization overhead is still substantial relative to arithmetic cost.
- The implemented v2/v3 optimization (batched IID index/permutation generation plus batch gather usage) further reduced RNG and launch overhead, giving substantial additional gains for bootstrap and measurable gains for permutation while preserving p-value consistency.

### 4.1) Full benchmark trend on canonical workload (v0 -> v6)

Data source:

- results/remote_inference_backend_benchmark_2026-04-05.json
- results/remote_inference_backend_benchmark_v2_2026-04-05.json
- results/remote_inference_backend_benchmark_v3_2026-04-05.json
- results/remote_inference_backend_benchmark_v4_2026-04-05.json
- results/remote_inference_backend_benchmark_v5_2026-04-05.json
- results/remote_inference_backend_benchmark_v6_2026-04-05.json

CuPy bootstrap mean time (n=5000, n_resamples=1500):

- v0: 507.389 ms
- v2: 146.461 ms
- v3: 102.092 ms
- v4: 101.698 ms
- v5: 100.280 ms
- v6: 106.082 ms

CuPy permutation mean time (n=3000, n_resamples=2000):

- v0: 1350.619 ms
- v2: 950.569 ms
- v3: 881.965 ms
- v4: 879.175 ms
- v5: 871.900 ms
- v6: 884.364 ms

Ratio notes:

- bootstrap CuPy speedup (v0 -> v6): 4.783x
- permutation CuPy speedup (v0 -> v6): 1.527x
- bootstrap CuPy speedup (v5 -> v6): 0.945x
- permutation CuPy speedup (v5 -> v6): 0.986x

### 4.2) Large-scale CPU/GPU crossover (v6, with CPU vectorized baseline)

Data source:

- results/remote_inference_backend_benchmark_v6_2026-04-05.json

Workload (resampling_large_scale config):

- bootstrap: n=30000, n_resamples=2500
- permutation: n=18000, n_resamples=2000
- repeats=3, warmup=1

Bootstrap mean time:

- NumPy scalar callback: 372.619 ms
- NumPy vectorized callback: 388.271 ms
- CuPy scalar callback: 171.903 ms
- CuPy vectorized callback: 16.303 ms

Permutation mean time:

- NumPy scalar callback: 2594.905 ms
- NumPy vectorized callback: 2440.481 ms
- CuPy scalar callback: 919.434 ms
- CuPy vectorized callback: 72.754 ms

Ratio notes:

- bootstrap CPU scalar vs GPU scalar: 2.361x
- bootstrap CPU scalar vs CPU vectorized: 1.045x
- bootstrap CPU vectorized vs GPU vectorized: 23.816x
- bootstrap CPU scalar vs GPU vectorized: 24.893x
- permutation CPU scalar vs GPU scalar: 2.822x
- permutation CPU scalar vs CPU vectorized: 1.063x
- permutation CPU vectorized vs GPU vectorized: 33.544x
- permutation CPU scalar vs GPU vectorized: 35.667x
- bootstrap GPU scalar vs GPU vectorized: 10.544x
- permutation GPU scalar vs GPU vectorized: 12.638x

Interpretation:

- CPU vectorized baseline is now included for direct apples-to-apples comparison.
- On this large-scale workload, GPU outperforms CPU for bootstrap and permutation even on scalar callback.
- With vectorized path enabled on both sides, GPU remains decisively faster (bootstrap ~23.8x, permutation ~33.5x vs CPU vectorized), satisfying the large-scale crossover objective.

### 4.3) CI-style performance guard assertion

Data source:

- dev/tests/test_inference_performance_guard.py

Guard behavior:

- default: skipped (requires environment variable `STATGPU_RUN_PERF_GUARD=1`)
- assertion target: on large-scale vectorized path, GPU time < CPU time for both bootstrap and permutation

Remote execution:

- command: `STATGPU_RUN_PERF_GUARD=1 PYTHONPATH=. /root/miniconda3/envs/myconda/bin/python -m pytest dev/tests/test_inference_performance_guard.py -q -ra`
- result: 1 passed in 7.32s

### 4.4) Large-scale external framework comparison (SciPy)

Data source:

- results/remote_resampling_strategy_external_benchmark_v1_2026-04-05.json

Workload: bootstrap n=30000 / n_resamples=1200, permutation n=18000 / n_resamples=1000, warmup=1, repeats=3.

Bootstrap mean time:

- statgpu NumPy vectorized: 196.234 ms
- statgpu CuPy vectorized: 8.587 ms
- SciPy bootstrap: 507.635 ms

Permutation mean time:

- statgpu NumPy vectorized: 1229.125 ms
- statgpu CuPy vectorized: 37.983 ms
- SciPy permutation_test: 2965.755 ms

Ratio notes:

- bootstrap: SciPy vs statgpu NumPy vectorized: 2.587x slower
- permutation: SciPy vs statgpu NumPy vectorized: 2.413x slower
- bootstrap: SciPy vs statgpu CuPy vectorized: 59.117x slower
- permutation: SciPy vs statgpu CuPy vectorized: 78.081x slower

Interpretation:

- Large-scale crossover now includes existing framework comparison (SciPy).
- On this CUDA host and workload, statgpu outperforms SciPy on CPU vectorized path, and CuPy vectorized path provides a further large speedup.

### 5) Bootstrap end-to-end framework comparison

Data source:

- results/remote_bootstrap_benchmark_2026-04-05.md
- results/remote_bootstrap_benchmark_2026-04-05.json

Mean timing (n=3000, p=24, n_bootstrap=100, warmup=1, repeats=5):

- statgpu bootstrap CPU: 89.961 ms
- statgpu bootstrap GPU: 86.346 ms
- sklearn residual bootstrap CPU: 160.997 ms
- statsmodels residual bootstrap CPU: 3631.271 ms

Ratios:

- statgpu CPU vs GPU: 1.042x
- sklearn vs statgpu CPU: 1.790x slower
- statsmodels vs statgpu CPU: 40.365x slower

## Notes

- The same expanded suite was executed locally before remote run: 28 passed, 3 skipped (CUDA-specific tests skipped on local non-CUDA environment).
- Remote run used an uploaded snapshot of the current workspace to ensure test-code and implementation consistency.
- Rscript is unavailable on this remote machine, so no R-side bootstrap timing was produced.
- A single bootstrap/permutation call returns one backend-specific result object by design; CPU/GPU side-by-side comparison is produced by invoking both backends within the same script and aggregating outputs.
- Detailed backend/framework benchmark payload is saved at results/remote_inference_backend_benchmark_v6_2026-04-05.json.
