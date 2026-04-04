# Changelog

> Language: English  
> Last updated: 2026-04-02  
> This page: Changelog  
> Switch: [中文](../changelog.md)

Language switch: [中文](../changelog.md)

## 2026-04

### Added

- Lasso inference rename:
  - `cpu_ols_inference` (alias `naive_ols`)
  - `gpu_ols_inference` (alias `gpu_naive_ols`)
- `gpu_memory_cleanup` for all current models
- `LinearRegression` robust covariance: `nonrobust/hc0/hc1` (CPU+GPU)
- `LogisticRegression` robust covariance: `nonrobust/hc0/hc1` (CPU+GPU)
- `CoxPH` covariance support: `nonrobust/hc0/hc1/cluster` (cluster is CPU path)
- New benchmark: `dev/benchmarks/benchmark_all_methods_large_scale.py`
- New external comparison benchmark: `dev/benchmarks/benchmark_external_frameworks.py`

### Validation

- Added consistency tests against `statsmodels` for robust covariance in:
  - `LinearRegression`
  - `LogisticRegression` (CPU+GPU)
- Added Cox consistency checks vs `statsmodels.PHReg` (`breslow/efron`) for coefficients
