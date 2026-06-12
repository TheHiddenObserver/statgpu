# Changelog

> Language: English  
> Last updated: 2026-06-12
> This page: Changelog  
> Switch: [Chinese](../changelog.md)

Language switch: [Chinese](../changelog.md)

## 2026-06

### Fixed (2026-06-10 ~ 2026-06-12)

- **PR #49 Code Review: 110+ fixes across 16 files**:
  - Fixed 26 P1 bugs (merge conflict, NameError, numerical formula errors, GPU path crashes)
  - Fixed 55 P2 bugs (cache thread safety, backend consistency, edge cases, API compatibility)
  - Fixed ~30 P3 improvements (dead code cleanup, magic numbers, performance)
  - Added 428 test cases (all passing on remote GPU Tesla P100)
  - Cross-backend precision deviation < 0.02% (same random_state)
  - No performance regression (RidgeCV CuPy 6.8x speedup, PenalizedGLM_CV Torch 3.1x)
  - Removed ~1300 lines of dead code
  - Unified `best_score_` to negative MSE (sklearn convention)
  - Merged PLAN_UNIFIED.md gates + PR #49 coding conventions into TO_DO.md

### Optimized (2026-06-05)

- **Strict sparse GLM CV GPU squeeze pass**:
  - Reduced GPU synchronization in `fista_bb_solver` CV paths by clipping gradients on device and reusing the norm already synchronized by safeguarded backtracking.
  - Avoided repeated full-vector GPU-to-CPU transfers for sparse GLM CV objective tracking and positive-family `y` scaling; CV wrappers now keep L1/ElasticNet penalty tracking and `mean/max(abs(y))` reductions on device until scalar synchronization.
  - Reduced logistic sparse GPU CV convergence synchronization after the early-iteration window, and added a low-dimensional squared-error sparse CV early-stop check where it is faster than deferred GPU checks.
  - Added a GPU batched-alpha score path for squared-error L1/ElasticNet CV, solving the alpha grid as one coefficient matrix to amortize small-kernel launches; final refit remains strict single-alpha.
  - Added a strict single-alpha sparse-GLM final refit fast path for Poisson/Gamma-style sparse CV; it still uses the original `max_iter`, original `tol`, and `cv_mode=False`.
  - Reused the fold-level initial Lipschitz estimate across sparse GLM alpha paths, including the `fista_bb_solver` burn-in checks, avoiding repeated Hessian/power-iteration setup without changing strict `max_iter`/`tol`.
  - Batched CuPy validation scoring for sparse GLM CV in the same style as the Torch score path; solver trajectories and final refits are unchanged.
  - Added a Torch fold-batched strict logistic sparse CV path: all folds share `X @ coef_matrix` and `X.T @ residual_matrix` updates while keeping per-fold Lipschitz constants, convergence checks, warm starts, and validation scores equivalent to the previous per-fold helper.
  - Added a CuPy fold-batched strict logistic sparse CV path with the same per-fold Lipschitz, warm-start, convergence-freezing, and batched validation semantics as the Torch helper; explicit `device="cuda"` remains CuPy-only and falls back only to the previous CuPy per-fold path if this helper fails.
  - Refined `solver="auto"` for Poisson sparse CV: GPU `poisson+elasticnet` uses `fista_bb`, while high-dimensional `poisson+l1` uses `fista` to preserve alpha agreement and avoid the slower BB pocket.
  - Refined `device="auto"` CV routing for sparse GLMs using the Matpool P100 break-even matrix; explicit `device="cuda"` and `device="torch"` are still never overridden. Logistic sparse auto routing now includes the high-dimensional `p>=500`, `n*p>=1e6` Torch fold-batched break-even.
  - Remote P100 validation (`cv=3`, `n_alphas=8`, `max_iter=1000`, `tol=1e-4`) showed Torch faster than CPU for all `5000x500` logistic/Poisson/Gamma L1/ElasticNet strict-CV rows, with all alpha selections matching CPU.
  - Larger GLM sparse validation (`10000x500` and `20000x500`) showed Torch faster than CPU for 12/12 logistic/Poisson/Gamma L1/ElasticNet rows, with all alpha selections matching CPU.
  - After the squared-error batched-alpha path, the mid/high matrix has Torch faster than CPU in 16/32 rows overall and 12/16 `p=500` rows, with all alpha selections matching CPU.
  - After the sparse-GLM Lipschitz cache and CuPy score batching, the aligned mid/high strict matrix still had Torch faster than CPU in 16/32 rows, with all CPU/CuPy/Torch alpha selections matching. The main improvement was in Poisson sparse CV: representative Torch runtimes improved by about `0.83x`-`0.89x`, and CuPy Poisson score-heavy rows by about `0.76x`-`0.82x`, versus the previous round-3 matrix.
  - After Torch fold-batched logistic CV, the same mid/high strict matrix has Torch faster than CPU in 18/32 rows, with all CPU/CuPy/Torch alpha selections matching. Logistic Torch runtimes improved to roughly `0.46x`-`0.53x` of the previous round-6 timings, and logistic Torch is faster than CPU in 6/8 tested rows.
  - `device="auto"` on the same matrix selected CPU for 14 rows and Torch for 18 rows; it was faster than explicit CPU in 27/32 rows, with all alpha selections matching CPU. One low-dimensional squared-error row still shows a one-time Torch initialization outlier under `warmup=0`.
  - A follow-up auto-routing pass keeps low-dimensional squared-error sparse CV (`p<256`) on CPU, avoiding that Torch cold-start outlier while preserving the high-dimensional Torch batched-alpha route. In the round-8 auto matrix, all alpha selections still match CPU and the remaining auto-vs-CPU slow rows are within roughly 3% timing noise.
  - After CuPy fold-batched logistic CV, the round-9 mid/high strict matrix (`warmup=1`) kept all CPU/CuPy/Torch/auto alpha selections matching CPU. Explicit Torch was faster than CPU in 18/32 rows, explicit CuPy in 8/32 rows, and `device="auto"` in 27/32 rows while selecting CPU for 16 rows and Torch for 16 rows. Targeted logistic CuPy validation matched the previous CuPy per-fold scores to numerical precision and made CuPy faster than CPU on the larger `10000x100` and `5000x500` logistic rows, but `2000x100` and `2000x500` remain explicit-CuPy hotspots.
  - Remaining strict hotspots are small/low-dimensional explicit GPU cases and Gamma/Poisson `p=100` pockets; strict mode still preserves the requested `max_iter` and `tol`.
  - Validation artifacts: `results/cv_mid_high_after_sqerr_batch_round3.json`, `results/cv_squared_error_batched_alpha_gpu_probe.json`, `results/cv_squared_error_auto_batched_alpha_round3.json`, `results/cv_large_glm_cpu_torch_round2.json`, `results/cv_poisson_gamma_lipcache_round5.json`, `results/cv_poisson_gamma_cupy_score_batch_round6.json`, `results/cv_mid_high_after_lipcache_scorebatch_round6.json`, `results/cv_auto_after_lipcache_scorebatch_round6.json`, `results/cv_logistic_foldbatch_round7.json`, `results/cv_mid_high_after_logistic_foldbatch_round7.json`, `results/cv_auto_after_logistic_foldbatch_round7.json`, `results/cv_auto_lowp_sqerr_cpu_round8.json`, `results/cv_logistic_cupy_foldbatch_round9.json`, `results/cv_mid_high_after_cupy_foldbatch_round9.json`.

### Added (2026-06-04)

- **Strict-first PenalizedGLM_CV strategy controls**:
  - `PenalizedGLM_CV` now defaults to `cv_strategy="strict"` and exposes opt-in `cv_strategy="two_stage"` alpha screening.
  - Two-stage CV uses relaxed screening solves, strict candidate refinement, and a strict final refit.
  - Added `ApproximateCVWarning`, `acknowledge_approx`, `refine_top_k`, and CV diagnostics (`cv_strategy_`, `cv_selected_device_`, `refined_mask`, stage-1 score arrays).
  - Benchmark scripts can run strict or two-stage CV via `--cv-strategy`.

### Fixed (2026-06-04)

- **Poisson sparse `PenalizedGLM_CV` cross-backend precision**:
  - Strict GPU FISTA no longer uses the asynchronous CV-only update loop; that fast path is reserved for approximate screening.
  - Poisson L1/ElasticNet CV now uses a deterministic near-tie rule for flat CV curves, preferring the stronger regularization when backend score differences are at numerical-noise scale.
  - Remote P100 validation for `poisson+l1/elasticnet`, `n=500`, `p=20`, `cv=3`, `n_alphas=8` selected the same alpha on CPU, CuPy, and Torch with coefficient L2 differences around `1.6e-05`.

### Optimized (2026-06-04)

- **Small sparse-CV GPU transfer reduction**:
  - Squared-error sparse CV now skips unnecessary coefficient-path host transfers when only validation scores are needed.
  - On Matpool P100 (`n=500`, `p=20`, `cv=3`, `n_alphas=8`), `squared_error+l1` strict CV improved from `820ms` to `190ms` on CuPy and from `266ms` to `97ms` on Torch, with unchanged alpha selection and coefficient L2 differences around `6.9e-06` versus CPU.
  - Logistic sparse CV remains a strict-mode hotspot; the existing iteration cap is intentionally not applied to strict CV because strict mode preserves the requested `max_iter` and `tol`.
  - Added `dev/tests/benchmark_glm_penalty_external_small.py` for small sklearn/statsmodels/R accuracy and runtime comparisons with explicit penalty-parameter mappings.
  - Validation artifacts: `results/cv_strict_sparse_sync_opt_v2_500x20.json` and `results/external_glm_penalty_small_gpu_sync_opt_v2.json`.

- **GPU sparse GLM CV solver policy**:
  - `solver="auto"` now uses backend-aware strict-CV choices for sparse GLMs: GPU `poisson+l1` and `negative_binomial+l1` use `fista_bb` on the benchmarked small strict-CV matrix, while Gamma and inverse-Gaussian sparse CV use conservative `fista`; explicit solver choices are unchanged.
  - The sparse GLM CV path initializes the intercept at `log(mean(y))`, matching the regular positive-family fit initialization.
  - Remote P100 strict matrix (`n=500`, `p=20`, `cv=3`, `n_alphas=8`) kept 90/90 alpha matches across CPU, CuPy, and Torch; targeted speedups included `negative_binomial+l1` Torch `0.37x` and CuPy `0.55x` runtime, `poisson+l1` Torch `0.57x` and CuPy `0.83x`, relative to the prior strict baseline.
  - Validation artifacts: `results/cv_strict_500x20_gpu_policy_opt_v3.json` and `results/cv_two_stage_sparse_auto_policy_opt_500x20.json`.

### Optimized (2026-06-01)

- **Backend transfer helpers and benchmark parser**:
  - CuPy <-> Torch CUDA conversions now prefer DLPack zero-copy sharing and fall back to the previous safe conversion path when unavailable.
  - NumPy -> Torch CUDA transfers try pinned host memory with `non_blocking=True`.
  - Added `dev/tests/_bench_report_parser.py` to summarize full-matrix benchmark text logs into JSON or Markdown.
  - Benchmark summaries include backend/family/penalty row counts and support `--fail-on-alerts` for scriptable benchmark gates.
  - CoxPH/CoxPHCV now expose Torch cleanup hooks consistently with the GPU memory cleanup contract.

## 2026-05

### Fixed (2026-05-20)

- **v23c: L-BFGS fused penalty gradient fix**:
  - Root cause: `lbfgs_solver` fused GLM path computed loss-only gradient, missing penalty gradient
  - L-BFGS converged to unregularized solution (`loss_grad ≈ 0`) instead of `loss_grad + α·coef = 0`
  - Fix: add `_smooth_penalty_gradient(penalty, coef)` after each `_fused_glm_value_and_gradient` call
  - Affected: all GLM families (logistic, poisson, gamma, NB, tweedie, inv_gauss) + smooth penalties (L2, ElasticNet)
  - Impact: 9 MISMATCH cases fixed (max|diff| from 1e-01~1e-02 down to 1e-04~1e-08)
  - Full benchmark: 1043/1043 ALL PASS (Section A: 816, B: 13, D: 68, E: 146)
  - Files modified: `statgpu/glm_core/_solver.py`

### Optimized (2026-05-20)

- **v22g: Async FISTA and GPU optimizations**:
  - Async FISTA for non-smooth penalties: 2-5.5x speedup on GLM+non-smooth at n=5000
  - Lipschitz recomputation, y-scaling cap, NB momentum cap, gamma conservative momentum
  - Backtracking optimization, gradient clipping unification
  - GPU sync optimizations for CuPy/Torch backends
  - Files modified: `statgpu/glm_core/_solver.py`, `statgpu/glm_core/_negative_binomial.py`, `statgpu/backends/_array_ops.py`

- **v23c: Full matrix benchmark (1043 tests)**:
  - 7 families x 10 penalties x 3 scales x multiple solvers x 3 backends
  - Section A timing: CPU avg 953ms/3995ms/2875ms, Torch at n=5000: 2.19x speedup
  - Section B: 13/13 vs sklearn ALL PASS
  - Section D: 68/68 vs statsmodels ALL PASS
  - Section E: 146/146 cross-solver ALL PASS
  - Report: `dev/tests/_bench_v23c_report.md`

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
