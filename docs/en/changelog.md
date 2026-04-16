# Changelog

> Language: English  
> Last updated: 2026-04-15  
> This page: Changelog  
> Switch: [Chinese](../changelog.md)

Language switch: [Chinese](../changelog.md)

## 2026-04

### Added

- Knockoff feature-selection API (fixed-X + model-X Gaussian second-order path):
  - `statgpu.knockoff_filter`
  - `statgpu.fixed_x_knockoff_filter`
  - `statgpu.model_x_knockoff_filter`
  - `statgpu.KnockoffSelector` / `statgpu.FixedXKnockoffSelector`
  - Knockoff statistics now include `method='corr_diff'` and `method='ols_coef_diff'`
  - Model-X calibration now includes covariance shrinkage and multi-draw W aggregation for improved cross-seed stability
- Lasso inference rename:
  - `cpu_ols_inference` (alias `naive_ols`)
  - `gpu_ols_inference` (alias `gpu_naive_ols`)
- `gpu_memory_cleanup` for all current models
- `LinearRegression` robust covariance: `nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU, with `hac_maxlags`)
- `Ridge` robust covariance: `nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU, with `hac_maxlags`)
- `LogisticRegression` robust covariance: `nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU, with `hac_maxlags`)
- `CoxPH` covariance support: `nonrobust/hc0/hc1/cluster` (cluster is CPU path)
- Exported CV estimator interface skeletons:
  - `RidgeCV`
  - `LogisticRegressionCV`
  - `CoxPHCV`
  - Current status: interface-only scaffolding; CV training logic is not implemented yet and currently raises `NotImplementedError`.
- New benchmark: `dev/benchmarks/benchmark_all_methods_large_scale.py`
- New external comparison benchmark: `dev/benchmarks/benchmark_external_frameworks.py`
- Nonparametric exports and API coverage:
  - KDE: `fit_kde`, `kde_pdf`, `kde_bootstrap_confidence_interval`
  - KDE kernel options: `gaussian/rectangular/triangular/epanechnikov/biweight/cosine/optcosine/triweight`
  - KDE bandwidth rules: `nrd0` and `nrd`
  - Kernel regression: `fit_kernel_regression`, `kernel_regression_predict`, `KernelRegression`
  - Kernel regression API added `kernel_metric='full'|'diagonal'` and `bandwidth_per_feature`
- New benchmark: `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`
- Nonparametric benchmark coverage expanded:
  - `dev/benchmarks/benchmark_kde_vs_scipy.py` now reports statgpu CPU/GPU vs SciPy
  - `dev/benchmarks/benchmark_nonparametric_vs_r.py` supports `--statgpu-backend numpy/cupy`
  - `dev/benchmarks/benchmark_nonparametric_vs_r.py` KDE CI supports `--ci-method normal/bootstrap`
  - Unified CPU/GPU/R/SciPy/statsmodels comparisons now cover KDE, KernelReg NW, KernelReg Local Linear, and KDE CI
- New knockoff benchmarks:
  - `dev/benchmarks/benchmark_knockoff_fixedx.py`
  - `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
  - `benchmark_knockoff_vs_baselines.py` now supports optional `knockpy` baseline comparison when available
- New multiple-testing guide:
  - `docs/en/guides/multiple-testing-combine-pvalues.md`

### Validation

- Added consistency tests against `statsmodels` for robust covariance in:
  - `LinearRegression`
  - `LogisticRegression` (CPU+GPU)
- Added nonparametric validation coverage:
  - `dev/tests/test_inference_kde.py` (9 passed, 1 skipped)
  - `dev/tests/test_nonparametric_kernel_regression.py` (13 passed, 1 skipped)
- Remote kernel-regression parity run (`run_id=20260415_103036`) confirmed machine-precision alignment with statsmodels in diagonal metric mode.
- Added Cox consistency checks vs `statsmodels.PHReg` (`breslow/efron`) for coefficients
- Refreshed unified tri-backend covariance benchmark artifact:
  - `results/remote_covariance_full_compare_2026-04-10.json`
  - covers `statsmodels` / `statgpu CPU` / `statgpu GPU` under aligned `hc2/hc3/hac` settings

### Improved

- `LinearRegression` CPU HAC path now uses adaptive precision selection (mixed vs float64 probe + shape-bucket cache) to reduce large-scale runtime regressions.
- Kernel regression local-linear multidim path now uses batched vectorized solves; remote run (`run_id=20260415_120903`) preserved parity and improved runtime substantially (dim3: CPU ~4.81x, GPU ~115.5x; dim5: CPU ~5.39x, GPU ~116.4x).
- KDE 1D Numba fast path improved local SciPy-relative runtime from ~1.39x slower to ~0.58x faster.
