# Benchmark Index

> Language: English  
> Last updated: 2026-05-07
> This page: Benchmark index  
> Switch: [Chinese](../cn/benchmarks.md)

Language switch: [Chinese](../cn/benchmarks.md)

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

## Unsupervised Learning

- Detailed model docs: [docs/en/unsupervised/](unsupervised/README.md)
- `dev/benchmarks/benchmark_unsupervised.py`
  - Compares `PCA` and `KMeans` across `statgpu` CPU/CuPy/Torch and sklearn when available.
  - Outputs timing and numerical differences to JSON.
- `dev/benchmarks/benchmark_unsupervised_phase2.py`
  - Compares `DBSCAN`, `GaussianMixture`, `NMF`, and `AgglomerativeClustering` against available sklearn/SciPy/R baselines.
  - Records `umap-learn` and `openTSNE` smoke/runtime baselines for future UMAP/t-SNE work.
- `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`
  - Validates the optional statgpu-owned DBSCAN Cython CPU fast path against the exact fallback, sklearn CPU, CuPy, and Torch.
- `dev/benchmarks/benchmark_unsupervised_phase3.py`
  - Compares `TruncatedSVD`, `MiniBatchKMeans`, `UMAP`, and `TSNE` across statgpu CPU/CuPy/Torch and available external baselines.
  - Records warmup/repeat timings, precision or quality metrics, and skipped optional frameworks.
  - GPU measurements use backend-resident input arrays for the main timing path.
- `dev/benchmarks/benchmark_unsupervised_phase3b.py`
  - Compares `GaussianMixture` covariance variants and `AgglomerativeClustering` linkage variants.
  - Covers statgpu CPU/CuPy/Torch where supported plus sklearn, SciPy, and R `cluster::agnes` where available.
  - Outputs JSON and Markdown summaries under `results/`.
- `dev/benchmarks/benchmark_unsupervised_phase3c.py`
  - Compares `IncrementalPCA` and `MiniBatchNMF` across statgpu CPU/CuPy/Torch and sklearn when available.
  - Records reconstruction quality, explained variance metrics, warmup/repeat timings, and skipped optional frameworks.

Remote Phase 2 artifacts:
- `results/unsupervised_phase2_remote_20260502_142727.json`
- `results/unsupervised_phase2_full_comparison_20260502_142727.md`
- `results/unsupervised_phase2_dbscan_cython_benchmark_20260502_153927.json`
- `results/unsupervised_phase2_dbscan_cython_summary_20260502_153927.md`
- `results/unsupervised_phase2_dbscan_cython_final_20260502_160719.json`
- `results/unsupervised_phase2_final_20260502_160719.json`
- `results/unsupervised_phase2_final_summary_20260502_160719.md`
- `results/unsupervised_phase2_dbscan_cython_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_summary_20260502_210000.md`

Remote Phase 3 artifacts:
- `results/unsupervised_phase3_remote_finalopt_20260505_084444.json`
- `results/unsupervised_phase3_remote_finalopt_20260505_084444.md`
- `results/unsupervised_phase3_remote_perfopt_mediumlarge_20260505_131617.json`
- `results/unsupervised_phase3_remote_perfopt_mediumlarge_20260505_131617.md`
- `results/unsupervised_phase3_remote_perfopt2_large_tabular_20260505_132223.json`
- `results/unsupervised_phase3_remote_perfopt2_large_tabular_bs4096_20260505_132359.json`

Remote Phase 3B artifacts:
- `results/unsupervised_phase3b_verify_20260507_003957.json`
- `results/unsupervised_phase3b_verify_summary_20260507_003957.md`

Remote Phase 3C artifacts:
- `results/unsupervised_phase3c_opt7_20260507_185500.json`
- `results/unsupervised_phase3c_opt7_summary_20260507_185500.md`
- `results/unsupervised_phase3c_opt7_large_bs4096_20260507_185500.json`
- `results/unsupervised_phase3c_opt7_large_bs4096_summary_20260507_185500.md`
- `results/unsupervised_phase3c_opt7_xlarge_20260507_185500.json`
- `results/unsupervised_phase3c_opt7_xlarge_summary_20260507_185500.md`

DBSCAN CPU Cython note:
- The optional `_dbscan_cpu` extension is a statgpu-owned implementation, not a sklearn wrapper.
- Compact dense CPU cases use the extension when it is built and selected; fallback remains available for variable-density, sparse/all-noise, or no-compiler environments.

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
  - Outputs: coefficient difference, R², fit time (ms)
  - Key finding: All backends match sklearn with max coef diff < 3e-8

### R glmnet Comparison

- `dev/benchmarks/benchmark_glmnet_full.R` (R script)
- `dev/benchmarks/benchmark_statgpu_full.py` (Python script)
- `dev/benchmarks/run_full_benchmark.py` (unified runner)
  - Compares `statgpu CPU` vs `R glmnet::glmnet()`
  - Tests 6 datasets: small/medium/large/high_dim/sparse_coef/high_noise
  - Key findings:
    - statgpu CPU wins 4/6 comparisons
    - Coefficient norm difference due to regularization scaling conventions
    - Both implementations are correct Elastic Net

### Large-Scale Performance (n ≥ 10,000)

- `dev/benchmarks/benchmark_large_scale.py`
- `dev/benchmarks/run_large_scale.py` (remote runner)
  - Tests 6 configurations: n=10k~100k, p=100~500
  - Compares sklearn vs statgpu (CPU/CuPy/Torch)
  - Key findings:
    - statgpu Torch fastest in 5/6 tests (83%)
    - Max speedup: **4.36x** vs sklearn (n=100k, p=500)
    - GPU advantage visible at n ≥ 10,000

### Backend Selection Recommendations

| Data Scale | Recommended Backend | Expected Speedup |
|------------|---------------------|------------------|
| n < 1,000 | CPU (NumPy) | 0.7x - 1.0x |
| 1,000 ≤ n < 10,000 | CPU (NumPy) | 1.5x - 4x |
| 10,000 ≤ n < 50,000 | GPU (Torch) | 2x - 3x |
| n ≥ 50,000 | GPU (Torch) | 3x - 4.4x |

### Result Artifacts

- `results/benchmark_elasticnet_sklearn_2026-04-18.json` - sklearn comparison
- `results/benchmark_elasticnet_sklearn_2026-04-18.md` - sklearn summary
- `results/benchmark_full/benchmark_glmnet_all.json` - R glmnet comparison
- `results/benchmark_full/benchmark_complete_report.md` - full report
- `results/large_scale/benchmark_elasticnet_large_scale_2026-04-18.json` - large scale
- `results/large_scale/benchmark_elasticnet_large_scale_2026-04-18.md` - large scale summary
- `results/benchmark_complete_summary.md` - comprehensive summary

---

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
