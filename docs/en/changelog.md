# Changelog

> Language: English  
> Last updated: 2026-04-11  
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
- Added Cox consistency checks vs `statsmodels.PHReg` (`breslow/efron`) for coefficients
- Refreshed unified tri-backend covariance benchmark artifact:
  - `results/remote_covariance_full_compare_2026-04-10.json`
  - covers `statsmodels` / `statgpu CPU` / `statgpu GPU` under aligned `hc2/hc3/hac` settings

### Improved

- `LinearRegression` CPU HAC path now uses adaptive precision selection (mixed vs float64 probe + shape-bucket cache) to reduce large-scale runtime regressions.
