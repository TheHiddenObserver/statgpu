# Changelog

> Language: English  
> Last updated: 2026-04-18  
> This page: Changelog  
> Switch: [Chinese](../changelog.md)

Language switch: [Chinese](../changelog.md)

## 2026-04

### Added (2026-04-26)

- **Phase 1: Ordered Model Cross-Backend Precision Fixes**:
  - CuPy convergence tolerance aligned: `gtol = 1e-6` → `gtol = self.tol` (matches scipy)
  - CuPy min iterations reduced from 30 to 5 (avoids forced extra iterations on small samples)
  - Removed CuPy warm-start branch, always initialize from zero (matches scipy/torch)
  - PyTorch captures real iteration count from `optimizer.state_dict()` instead of falsely reporting `max_iter`
  - PyTorch `strong_wolfe` failure now raises `RuntimeError` instead of silently degrading
  - Regression tests: `dev/tests/test_ordered_cross_backend.py` (10 cross-backend cases, all passed)
  - Files modified: `statgpu/linear_model/_glm_base.py`, `dev/tests/test_ordered_cross_backend.py`

- **Phase 2a: New hochberg (adjust_pvalues) + stouffer (combine_pvalues) across 3 backends**:
  - `adjust_pvalues` new `method='hochberg'` (step-up FDR), aliases `fdr_hochberg` / `step_up` / `stepup`
  - `combine_pvalues` new `method='stouffer'` (weighted Z-test), aliases `ztest` / `weighted_z`
  - Stouffer supports weights, consistent with cauchy weight interface
  - Batched support with `axis` parameter (arbitrary shape arrays)
  - Dependency: added `norm` distribution proxy (alongside existing `chi2`)
  - Files modified: `statgpu/inference/_multiple_testing.py`, `statgpu/inference/_distributions_backend.py`

- **Phase 2b: Test Expansion**:
  - New `TestHochberg` (4 tests): closed-form verification, aliases, vs BH, axis batching
  - New `TestStouffer` (6 tests): vs scipy, weights, aliases, axis, edge cases
  - New `TestCauchyNoWeights` (2 tests): cauchy without weights, default weight equivalence
  - New `TestTorchBackend` (6 tests): adjust/combine Torch vs NumPy consistency
  - Fixed `np._core.numeric` compatibility (NumPy 1.x vs 2.x), added `_normalize_axis_index` helper
  - Test file grew from 339 to 519 lines
  - Remote validation: 40/40 passed (Tesla P100)
  - Files modified: `dev/tests/test_inference_multiple_testing.py`

- **Phase 3: Package Structure Audit & Reorganization**:
  - Moved `_gpu_utils.py` → `backends/_gpu_inference_cupy.py`
  - Moved `_gpu_utils_torch.py` → `backends/_gpu_inference_torch.py`
  - Merged `evaluation/` → `metrics/`, deleted `evaluation/` directory
  - Merged `glm_core/_backend.py` → `backends/_array_ops.py`
  - Moved `_cv_base.py` → `linear_model/_cv_base.py`
  - Fixed `core/__init__.py` docstring (removed references to non-existent modules)
  - Added `survival/__init__.py` naming convention docs (`_cuda` / `_cupy` / `_triton`)
  - Updated 18 import sites across the codebase
  - Deleted files: `_gpu_utils.py`, `_gpu_utils_torch.py`, `_cv_base.py`, `glm_core/_backend.py`, `evaluation/` directory
  - All moves verified with `import statgpu` smoke test

### Added (2026-04-21)

- **CoxPHCV upgraded from skeleton to trainable implementation**:
  - Implemented K-fold penalty search and final refit on full data
  - Supports `ties='breslow'/'efron'` with existing `device` paths (executed via `CoxPH` backends)
  - Current boundary: `entry` and `cluster` are not yet supported in `CoxPHCV.fit()` (explicit `NotImplementedError`)
  - Files:
    - `statgpu/survival/_cox_cv.py`
    - `dev/tests/test_coxph_cv.py`

- **RidgeCV and LogisticRegressionCV Full Implementation**:
  - Upgraded from interface scaffolding to full-featured implementation with GPU-accelerated cross-validation
  - `RidgeCV` new features:
    - K-fold cross-validation (custom folds or fold generator support)
    - Automatic alpha grid generation (log-spaced grid)
    - Cross-validation result caching (Blake2b hash key, LRU cache maxsize=64)
    - Support for `sample_weight` and `scoring` parameters
    - Backend support: CPU (NumPy), GPU (CuPy), GPU (PyTorch)
  - `LogisticRegressionCV` similar enhancements
  - Files modified:
    - `statgpu/linear_model/_ridge_cv.py` - Full implementation (~1000 lines)
    - `statgpu/linear_model/_logistic_cv.py` - Full implementation
  - Core API:
    ```python
    from statgpu.linear_model import RidgeCV, LogisticRegressionCV

    # RidgeCV with automatic alpha grid
    ridge_cv = RidgeCV(alphas=100, cv=5, device='cuda')
    ridge_cv.fit(X, y)
    print(f"Best alpha: {ridge_cv.best_alpha_}")
    print(f"CV scores: {ridge_cv.cv_results_['mean_test_score']}")

    # LogisticRegressionCV with custom alphas
    logit_cv = LogisticRegressionCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5, device='cuda')
    logit_cv.fit(X, y)
    ```

### Added (2026-04-20)

- **CoxPH Efron Implementation Fix and Performance Optimization**:
  - Fixed numerical overflow in Cython Efron gradient/Hessian computation with clipping protection (`MAX_LINPRED=700`, `MIN_LINPRED=-700`)
  - Identified correctness issues in compiled Cython version, temporarily using Python fallback (verified against numeric gradient)
  - CoxPH comprehensive benchmark (vs statsmodels/lifelines/R survival):
    - statgpu-Torch GPU achieves **15.44x** speedup on n=5000, p=20 (vs statsmodels)
    - All statgpu backends match statsmodels coefficients (Max Diff < 4e-12)
    - C-index calculation fixed: CPU/CuPy/Torch now use identical exact blockwise vectorized algorithm
  - Files modified:
    - `statgpu/survival/_cox_efron_cy.pyx` - Added exp() clipping protection
    - `statgpu/survival/_cox.py` - Use Python fallback for Efron gradient computation
  - Benchmark results:
    - n=1000, p=10: statgpu-Torch 2.05x, lifelines 3.33x, R survival 21.6x (vs statsmodels)
    - n=5000, p=20: statgpu-Torch **15.44x**, lifelines 3.42x (vs statsmodels)
  - Test scripts:
    - `dev/scripts/test_coxph_fit.py` - CoxPH fit with lifelines comparison
    - `dev/scripts/final_verification.py` - Comprehensive verification script
  - Report:
    - `results/coxph_benchmark_report_2026-04-20.md` - Comprehensive benchmark report

### Added (2026-04-18)

- **Elastic Net Implementation and Benchmarks**:
  - New `ElasticNet` class combining L1 and L2 regularization with FISTA solver
  - Supports CPU (NumPy), GPU (CuPy), and GPU (PyTorch) backends
  - Files added:
    - `statgpu/linear_model/_elasticnet.py` - Elastic Net implementation
    - `dev/benchmarks/benchmark_elasticnet_sklearn.py` - sklearn comparison
    - `dev/benchmarks/benchmark_glmnet_full.R` - R glmnet comparison
    - `dev/benchmarks/benchmark_statgpu_full.py` - statgpu vs glmnet
    - `dev/benchmarks/benchmark_large_scale.py` - large-scale performance tests
    - `dev/benchmarks/run_full_benchmark.py` - unified benchmark runner
    - `dev/benchmarks/run_large_scale.py` - remote runner
    - `dev/benchmarks/generate_complete_report.py` - report generator
    - `dev/scripts/remote_elasticnet_smoke.py` - basic validation
    - `dev/scripts/remote_stability_en.py` - numerical stability tests
  - Benchmark results:
    - All backends match sklearn with max coef diff < 3e-8
    - statgpu CPU wins 4/6 vs R glmnet
    - statgpu Torch fastest in 5/6 large-scale tests (83%)
    - Maximum speedup: **4.36x** vs sklearn (n=100k, p=500)
  - Documentation:
    - `docs/models/elastic-net.md` - Chinese documentation
    - `docs/en/models/elastic-net.md` - English documentation
    - `results/benchmark_complete_summary.md` - comprehensive benchmark summary

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
