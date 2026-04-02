# Changelog

## 2026-04

### Added

- Lasso inference method semantic rename:
  - `cpu_ols_inference` (alias: `naive_ols`)
  - `gpu_ols_inference` (alias: `gpu_naive_ols`)
- GPU memory cleanup switch `gpu_memory_cleanup` for:
  - `LinearRegression`
  - `Ridge`
  - `Lasso`
  - `LogisticRegression`
  - `CoxPH`
- `LinearRegression(cov_type=...)`:
  - `nonrobust`
  - `hc0`
  - `hc1`
  with CPU + GPU robust inference path
- `LogisticRegression(cov_type=...)`:
  - `nonrobust`
  - `hc0`
  - `hc1`
  with CPU + GPU robust inference path
- New benchmark script:
  - `examples/benchmark_all_methods_large_scale.py`
  for large-scale runtime comparison across all current methods

### Improved

- Lasso `gpu_ols_inference` path computes more inference steps on GPU,
  reducing CPU transfer and external SciPy dependency.
- Documentation structure expanded into:
  - `docs/getting-started`
  - `docs/guides`
  - `docs/models`
  - `docs/benchmarks`

### Fixed

- `LogisticRegression.fit()` implicit NumPy conversion issue when `y` is a CuPy array.

### Validation

- Added external consistency tests with `statsmodels`:
  - `LinearRegression` robust covariance (`HC0/HC1`) consistency
  - `LogisticRegression` robust covariance (`HC0/HC1`) consistency (CPU + GPU)
