# Benchmark Index

> Language: English  
> Last updated: 2026-07-20  
> This page: Benchmark index  
> Switch: [Chinese](../../cn/guides/benchmarks.md)

## Benchmark Dashboard

- **Interactive dashboard**: [Open Dashboard](../../assets/benchmarks/index.html)
- **Dashboard guide**: [Filters, charts, metrics, and reproduction](statgpu_benchmark_dashboard.md)

The canonical dashboard is restricted to benchmark sources dated **2026-06-01 or later**. It currently contains **8 registered sources, 1,774 normalized runs, and 36 models**.

The current bundle connects penalized GLM and GLM, recent linear models, robust/quantile regression, survival analysis, unsupervised learning, ordered models, nonparametric methods, panel models, covariance estimation, and ANOVA. The feature-selection category remains reserved until a June 2026-or-later structured benchmark is available.

April 2026 ElasticNet, LassoCV, comprehensive-validation, Cox package-comparison, and knockoff results are intentionally not registered. A June distribution report is also excluded until it is rerun or converted into a structured source with raw repeat and precision provenance.

Current capabilities:

- Environment and multi-category navigation.
- Metric-scope filtering for Fit, CV, Inference, Prediction, and Selection.
- Progressive model, variant, penalty, solver, and scale filters.
- NumPy, CuPy, and Torch backend selection.
- Context-aware external comparisons with scikit-learn, SciPy, statsmodels, linearmodels, and pyGAM.
- Focused and Full matrix chart modes.
- Timing and speedup charts with distinct computed and runner-reported semantics.
- Sortable run-level table with explicit scope labels.
- Validation, accuracy, inference, prediction, convergence, and selection panels.
- Source provenance, parse-report metadata, and source-inventory coverage.

Generate and validate the canonical bundle:

```bash
python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources
```

Build and test the deployed dashboard:

```bash
cd frontend
npm ci
npm run typecheck
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

---

## Inference

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
  - Compares `cpu_ols_inference` vs `gpu_ols_inference`

## Nonparametric

- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`
  - Compares `statgpu` vs `statsmodels.nonparametric.kernel_regression.KernelReg`
  - Supports `regression=nw/local_linear` and multidimensional settings
  - Supports fair parity mode via `--kernel-metric diagonal`
  - Reports `statgpu CPU/GPU` and `statsmodels` accuracy/runtime comparisons
  - Outputs precision and runtime JSON under `results/`

- `dev/benchmarks/benchmark_kde_vs_scipy.py`
  - Compares `statgpu` vs `scipy.stats.gaussian_kde`
  - Reports `statgpu CPU/GPU` and SciPy accuracy/runtime comparisons

- `dev/benchmarks/benchmark_nonparametric_vs_r.py`
  - Compares `statgpu` with R `density()` / `ksmooth()` / `KernSmooth::locpoly()`
  - Supports `--statgpu-backend numpy/cupy`
  - Supports `--ci-method normal/bootstrap`
  - Reports `statgpu CPU/GPU`, R, and KDE CI vs SciPy comparisons

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

## Elastic Net Benchmarks

### sklearn Comparison

- `dev/benchmarks/benchmark_elasticnet_sklearn.py`
  - Compares `statgpu` (CPU/CuPy/Torch) vs `sklearn.linear_model.ElasticNet`
  - Tests 6 datasets: n=200~5,000, p=20~100
