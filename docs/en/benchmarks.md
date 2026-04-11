# Benchmark Index

> Language: English  
> Last updated: 2026-04-10  
> This page: Benchmark index  
> Switch: [Chinese](../benchmarks.md)

Language switch: [Chinese](../benchmarks.md)

## Inference

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
  - Compares `cpu_ols_inference` vs `gpu_ols_inference`

## Multiple-testing and Global P-value Combination

- `dev/benchmarks/benchmark_inference_backends.py`
  - Includes `combine_pvalues` benchmarks for `fisher/cauchy/acat`
  - Includes consistency checks:
    - Fisher vs `scipy.stats.combine_pvalues`
    - Cauchy vs independent NumPy reference
    - statgpu NumPy vs CuPy
  - Outputs structured JSON under `results/`

Remote supplement artifacts:
- `results/remote_fisher_cauchy_benchmark_2026-04-05.json`
- `results/remote_fisher_cauchy_benchmark_2026-04-05.md`

## GPU Memory

- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
  - Compares `gpu_memory_cleanup=False/True`

## Large-scale All-method Runtime

- `dev/benchmarks/benchmark_all_methods_large_scale.py`
  - Covers `LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`
  - Separates data construction from fit timing
  - Supports CPU/GPU, warmup, repeats, and JSON output

Recommended command:

```bash
python dev/benchmarks/benchmark_all_methods_large_scale.py \
  --devices cpu,cuda \
  --include-external \
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out results/bench_all_large_results.json
```

To include inference-statistics computation time in measurements, add:

```bash
--compute-inference
```

## External Framework Comparison (accuracy + runtime)

- `dev/benchmarks/benchmark_external_frameworks.py`
  - Primary comparison: `statsmodels`, `sklearn`
  - Optional comparison: `R` (if `Rscript` and required packages are available)
  - Outputs: `fit_ms` + coefficient/inference differences (+ JSON option)

Recommended command (statsmodels + sklearn):

```bash
python dev/benchmarks/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow \
  --skip-r
```

Recommended command (including R):

```bash
python dev/benchmarks/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow
```

Comparison gate recommendations:
- Explicitly use the same feature set across frameworks (avoid accidental `y ~ .` leakage)
- Explicitly fix Cox tie method (`breslow` or `efron`)
- Explicitly log regularization and convergence settings (`alpha/C/max_iter/tol`)

## Unified Covariance Tri-Comparison (statsmodels / statgpu CPU / statgpu GPU)

- Runner script: `tmp_remote_covariance_full_compare.py`
- Result artifact: `results/remote_covariance_full_compare_2026-04-10.json`
- Aligned setup:
  - `cov_type`: `hc2/hc3/hac`
  - `linear`: `n=8000, p=24`
  - `logistic`: `n=12000, p=16`
  - `timing_repeats=2` (with warmup)

Latest rerun snapshot (2026-04-10, aligned setup):
- Linear-HAC: `statsmodels=9.9158ms`, `statgpu CPU=10.3402ms`, `statgpu GPU=3.8064ms`
- Logistic-HAC: `statsmodels=14.6619ms`, `statgpu CPU=10.2583ms`, `statgpu GPU=7.4366ms`
- Linear-HAC precision: `statgpu CPU vs statsmodels` has `max_abs_bse_diff=1.3817e-09`

## Cox Covariance Benchmark

- `dev/benchmarks/benchmark_cox_cluster.py`
  - Compares `CoxPH cov_type=nonrobust/hc1/cluster` on runtime and numerical differences
  - Covers `statgpu CPU/GPU` and `statsmodels.PHReg` when available

## Knockoff Feature Selection

- `dev/benchmarks/benchmark_knockoff_fixedx.py`
  - Runs fixed-X knockoff at multiple `q` values and reports selected-set diagnostics.

- `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
  - Compares fixed-X/model-X knockoff with baseline selectors:
    - marginal-correlation top-k
    - statgpu lasso top-k
    - sklearn `LassoCV` (if installed)
    - `knockpy` Gaussian knockoff + lasso statistic (if installed)
  - Supports configurable knockoff statistic via `config.knockoff_method` (current default: `ols_coef_diff`).
  - Model-X path uses covariance-shrinkage plus multi-draw W aggregation (draw count depends on statistic).
  - Run output includes model-X calibration metadata (`modelx_n_draws`, `modelx_covariance_shrinkage`).
  - Environment and method blocks include optional availability flags and pairwise deltas for `knockpy` when present.
  - Reports precision/recall/FDP/F1/Jaccard and timing in one JSON file.
  - Additional environment controls:
    - `STATGPU_KNOCKOFF_COMPAT_MODE`: `statgpu` or `knockpy`
    - `STATGPU_KNOCKOFF_LASSO_CV_IMPL`: `auto` / `statgpu` / `sklearn`

- `dev/benchmarks/benchmark_knockoff_same_xk_parity.py`
  - Compares `statgpu` and `knockpy` using the exact same `Xk` generated once by knockpy.
  - Key outputs: `W` correlation, `W` error, threshold difference, and selected-set Jaccard.
  - Useful for correctness diagnostics when sampler randomness must be held constant.
