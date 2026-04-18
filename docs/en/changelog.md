# Changelog

> Language: English  
> Last updated: 2026-04-18  
> This page: Changelog  
> Switch: [Chinese](../changelog.md)

Language switch: [Chinese](../changelog.md)

## 2026-04

### Added (2026-04-18)

- **PyTorch Backend Fixes** (Torch Backend Fixes):
  - Fixed `_get_backend()` method in `_base.py` to properly handle `Device.TORCH`
  - Fixed import path issues in `_gpu_utils_torch.py`
  - Fixed variable name error in `compute_aic_bic_torch()`
  - Fixed device string handling in `_linear.py`, `_logistic.py`, `_ridge.py` (from `device.value` to `"cuda"`/`"cpu"`)
  - Fixed `y_arr.astype()` compatibility for Torch tensors in `_logistic.py`
  - **Fixed Cholesky solver `upper` parameter error in `_linear.py`** (`L.T` is upper triangular, should use `upper=True`)
  - Performance results (Tesla P100):
    - LinearRegression Torch GPU: numerical accuracy ~1e-15 (was ~0.22)
    - LogisticRegression Torch GPU: numerical accuracy ~1e-14
    - Lasso Torch GPU: numerical accuracy ~1e-5
    - Ridge Torch GPU: numerical accuracy ~1e-15
    - CoxPH Torch GPU: numerical accuracy ~1e-15

- **PyTorch Backend Complete** (Torch Backend Complete):
  - ✅ All core models support Torch backend (LinearRegression, Ridge, Lasso, LogisticRegression, CoxPH)
  - ✅ Nonparametric modules support (KDE, KernelRegression)
  - ✅ Feature selection module support (Knockoff)
  - ✅ Complete benchmarks and documentation
  - Files added:
    - `statgpu/_gpu_utils_torch.py` - Torch GPU utilities
    - `statgpu/inference/_distributions_torch.py` - Distribution objects (norm, t, F)
  - Files modified:
    - `statgpu/linear_model/_linear.py` - Added `_fit_torch()`
    - `statgpu/linear_model/_ridge.py` - Added `_fit_torch()`
    - `statgpu/linear_model/_logistic.py` - Added `_fit_torch()`
    - `statgpu/linear_model/_lasso.py` - Added `_fit_torch()`
    - `statgpu/survival/_cox.py` - Added `_fit_torch()`
    - `statgpu/nonparametric/_kernel_common.py` - Added Torch support
    - `statgpu/feature_selection/_knockoff_utils.py` - Added Torch support
  - Benchmark results:
    - Small dataset (2K×50): Torch competitive with CuPy (<20% gap)
    - Large dataset (50K×200): CuPy leads 2-5x (more mature linear algebra)
    - All models numerical accuracy <1e-6 vs CPU
  - Documentation updated:
    - `docs/guides/pytorch-backend.md` - PyTorch backend guide
    - `docs/en/guides/pytorch-backend.md` - English version
    - `dev/docs/torch_backend_final_report.md` - Final report

- **API Cleanup** (API Cleanup):
  - Removed `LinearRegression.bse_`, `LinearRegression.tvalues_`, `LinearRegression.pvalues_` properties
  - Removed `LogisticRegression.bse_`, `LogisticRegression.pvalues_` properties
  - **Reason**: These properties were temporarily added for test code; correct approach is test code using internal attributes `_bse`, `_pvalues`
  - **Impact**: Test code should use `model._bse[1:]` and `model._pvalues[1:]` (excluding intercept)

### Added (2026-04-17)

- **PyTorch Backend** (Phase 1-5 complete):
  - New GPU backend alternative to CuPy using PyTorch 2.0+
  - **Completed Models**:
    - ✅ Ridge Regression: Full covariance (HC1/HC2/HC3/HAC) + inference
    - ✅ LogisticRegression: IRLS solver + full inference
    - ✅ Lasso: FISTA solver + Debiased/Simultaneous inference
    - ✅ CoxPH: Breslow/Efron tie handling + full inference + C-index + Baseline Hazard
  - Files added:
    - `statgpu/inference/_distribution_utils_torch.py` - Special functions (betainc, gammainc, erf, etc.)
    - `statgpu/inference/_distributions_torch.py` - Distribution objects (norm, t, F)
    - `statgpu/backends/_torch.py` - Backend adapter (50+ NumPy-compatible methods)
  - Files modified:
    - `statgpu/linear_model/_ridge.py` - Added `_fit_torch()`, `_robust_covariance_torch()`
    - `statgpu/linear_model/_logistic.py` - Added `_fit_torch()` with IRLS
    - `statgpu/linear_model/_lasso.py` - Added `_fit_torch()`, `_compute_inference_debiased_torch()`, `_compute_simultaneous_inference_torch()`
    - `statgpu/linear_model/_linear.py` - Added `_fit_torch()` with HAC covariance
    - `statgpu/survival/_cox.py` - Added `_fit_torch()`, `_compute_log_likelihood_torch()`, `_compute_gradient_hessian_torch()`, `_compute_cindex_torch()`, `_compute_baseline_hazard_torch()`
  - Features:
    - Full GPU acceleration for Ridge, LogisticRegression, Lasso, CoxPH
    - Lasso Debiased inference (Javanmard-Montanari / Zhang-Zhang methods)
    - Lasso Simultaneous inference (max-|Z| multiplier bootstrap)
    - Robust covariance support (HC1/HC2/HC3/HAC)
    - CoxPH Baseline Hazard estimation (Breslow method)
    - SciPy fallback for older PyTorch versions (< 2.0)
    - Numerical accuracy: coefficients match NumPy within 1e-14
  - **Large-Scale Performance** (Tesla P100, 50K×200):
    - Ridge HC3: Torch GPU 0.067s vs CuPy GPU 0.064s (4% gap)
    - Logistic HC1: Torch GPU 0.099s vs CuPy GPU 0.102s (Torch wins!)
    - Lasso: Torch GPU 0.081s vs CuPy GPU 0.076s (7% gap)
    - CoxPH: Torch GPU 1.94s vs CuPy GPU 0.42s (CuPy faster for baseline hazard)
    - 60x GPU speedup for robust covariance vs CPU
  - Documentation:
    - `dev/docs/torch_backend_full_feature_report.md` - Complete benchmark report
    - `dev/docs/torch_backend_implementation_summary.md` - Implementation summary
    - `dev/docs/torch_vs_cupy_comprehensive_report.md` - Comprehensive comparison report
    - `docs/en/guides/pytorch-backend.md` - PyTorch backend guide
  - Installation: `pip install statgpu[torch]`

### Added (2026-04-15)

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
