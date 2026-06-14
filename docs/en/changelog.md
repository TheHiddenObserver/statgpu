# Changelog

> Language: English  
> Last updated: 2026-06-14
> This page: Changelog  
> Switch: [Chinese](../changelog.md)

Language switch: [Chinese](../changelog.md)

## 2026-06

### Added (2026-06-13 ~ 2026-06-14)

> PR #55~#58 were split from the original PR #36 (GLM+Penalty full module). PR #36 delivered the complete GLM + Penalty system achieving 1043/1043 ALL PASS (100%) in full-matrix benchmark.

- **PR #55 — Core GLM solver, backends, penalties, inference (PR-A, from PR #36)**:
  - 7 GLM families: squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie
  - 10 penalties: none, l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
  - 6 solvers: irls, fista, fista_bb, admm, lbfgs, newton — dispatched per family+penalty combination
  - 3 backends: NumPy (CPU), CuPy (CUDA), PyTorch (CUDA) with auto device selection
  - Unified inference: 15 distributions, p-value adjustment, bootstrap, permutation test
  - Key technical features: LLA routing for non-convex penalties (SCAD/MCP), augmented intercept for log-link GLMs, iterate-dependent Lipschitz computation, kernel fusion for loss+gradient

- **PR #56 — Penalized models + CV framework (PR-B, from PR #36)**:
  - 7 Penalized estimators: PenalizedLinearRegression, PenalizedLogisticRegression, PenalizedPoissonRegression, PenalizedGammaRegression, PenalizedInverseGaussianRegression, PenalizedNegativeBinomialRegression, PenalizedTweedieRegression
  - PenalizedGLM_CV: full CV over families x penalties x solvers
  - Lasso, Ridge, ElasticNet with full inference
  - LogisticRegression, LinearRegression with GPU

- **PR #57 — New modules (PR-C, from PR #36)**:
  - ANOVA: `f_oneway` — GPU-accelerated one-way ANOVA, float32/float64 support
  - Covariance: `EmpiricalCovariance`, `LedoitWolf`, `OAS` — covariance estimation with shrinkage
  - Panel Data: `PanelOLS` (one/two-way fixed effects), `RandomEffects` (Swamy-Arora), `PanelSummary`, clustered covariance
  - Splines: `bspline_basis`, `natural_cubic_spline_basis`, penalized regression with GCV
  - Semiparametric: `GAM` (penalized B-splines + GCV smoothing parameter selection)
  - Kernel Methods: `KernelRidge`, `KernelRidgeCV`, 6 kernel functions (rbf, polynomial, linear, laplacian, sigmoid, cosine)

- **PR #58 — Infrastructure, exports, backward compatibility (PR-D, from PR #36)**:
  - Unified `statgpu/__init__.py` exports (~60 public names)
  - `BaseEstimator` with device management and sklearn-compatible `get_params`/`set_params`
  - `Device` enum (CPU/CUDA/TORCH/AUTO) with auto-detection
  - Backward-compat shims for `kernel_methods/` and `splines/` old import paths

- **PR #48 — Module reorganization**:
  - Moved kernel_methods/ and splines/ under nonparametric/ subpackage
  - Created kernel_smoothing/ subpackage for KDE + kernel regression
  - Extracted GAM to semiparametric/ package for future extensibility
  - Backward-compat shims for old import paths

- **PR #59 — Documentation, changelog, guides (PR-E)**:
  - Complete model documentation for all new modules
  - Updated docs/en/ and docs/cn/ indexes

- **PR #60, #61 — README cleanup**:
  - Cleaned up README Implemented Methods with tables
  - Compressed README GLM section + removed redundancy

- **PR #62 — Dev folder reorganization**:
  - Archived 241 old/temp files from tests/, benchmarks/, scripts/ to _archive/
  - Updated remote_config.py: environment variables now override local config

- **PR #63 — Dev workspace documentation**:
  - Added dev/README.md (directory structure, remote GPU testing setup)
  - Added dev/tests/TESTING.md (test categories, remote workflow)
  - Added dev/benchmarks/RESULTS.md (GPU speedup data, version history)
  - Added dev/design/ARCHITECTURE.md (backend abstraction, GLM solver architecture)

- **PR #64 — Plans and changelog updates**:
  - Reorganized root files (USAGE.md → docs/, AGENTS.md → dev/, plans → dev/plans/)
  - Added module completion percentages to TO_DO.md
  - Updated plan files with implementation status

- **GPU Performance: Async FISTA (v22e)**:
  - Eliminated per-iteration GPU->CPU synchronization in FISTA loop
  - logistic + L1: 2.22x -> **5.41x** (n=5000, p=500)
  - logistic + ElasticNet: 2.18x -> **5.17x**
  - Poisson + L1: 1.90x -> **4.55x**
  - Smaller scale: logistic + Adaptive L1 now beats CPU (0.56x -> **1.12x**)

- **GPU Performance: v23c Full Matrix (1043/1043 ALL PASS)**:
  - 7 families x 13 penalties x 5 solvers x 3 backends
  - L-BFGS fused penalty gradient fix
  - Section A timing: CPU avg 953ms/3995ms/2875ms, Torch at n=5000: **2.19x** speedup
  - Section B: 13/13 vs sklearn ALL PASS
  - Section D: 68/68 vs statsmodels ALL PASS
  - Section E: 146/146 cross-solver ALL PASS
  - Report: `dev/tests/_bench_v23c_report.md`

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

### Added (2026-06-07 ~ 2026-06-09)

- **PR #50 — Add val_sample_weight to GLM sparse CV path**:
  - Validation sample weight support for sparse GLM cross-validation
  - Enables weighted CV folds for imbalanced datasets

- **PR #53 — Fix weighted Ridge inference**:
  - Correct scale calculation for weighted Ridge regression
  - Preserve bse/pvalues/conf_int with sample weights

- **PR #54 — Refactor CV dispatch table**:
  - Dispatch table for _compute_cv_scores
  - Cleaner separation of CV scoring logic

### Optimized (2026-06-05)

- **Strict sparse GLM CV GPU squeeze pass**:
  - Reduced GPU synchronization in `fista_bb_solver` CV paths
  - Batched CuPy validation scoring for sparse GLM CV
  - Torch fold-batched strict logistic sparse CV path
  - CuPy fold-batched strict logistic sparse CV path
  - Remote P100 validation: all alpha selections matching CPU

### Added (2026-06-04)

- **Strict-first PenalizedGLM_CV strategy controls**:
  - `PenalizedGLM_CV` now defaults to `cv_strategy="strict"`
  - Two-stage CV uses relaxed screening solves, strict candidate refinement
  - Added `ApproximateCVWarning`, CV diagnostics

### Fixed (2026-06-04)

- **Poisson sparse PenalizedGLM_CV cross-backend precision**:
  - Strict GPU FISTA no longer uses asynchronous CV-only update loop
  - Poisson L1/ElasticNet CV uses deterministic near-tie rule

### Optimized (2026-06-04)

- **Small sparse-CV GPU transfer reduction**:
  - Squared-error sparse CV skips unnecessary coefficient-path host transfers
  - `squared_error+l1` strict CV improved from 820ms to 190ms on CuPy

- **GPU sparse GLM CV solver policy**:
  - `solver="auto"` uses backend-aware strict-CV choices for sparse GLMs

### Optimized (2026-06-01)

- **Backend transfer helpers and benchmark parser**:
  - CuPy <-> Torch CUDA conversions prefer DLPack zero-copy
  - NumPy -> Torch CUDA transfers try pinned host memory

## 2026-05

### Added (2026-05-03 ~ 2026-05-11)

- **PR #27~#29 — Unsupervised learning Phase 3/3B/3C**:
  - Added 12 estimators: PCA, KMeans, DBSCAN, GaussianMixture, NMF, AgglomerativeClustering, UMAP, TSNE, MiniBatchKMeans, MiniBatchNMF, IncrementalPCA, TruncatedSVD
  - GPU exact paths for agglomerative clustering
  - Documentation and validation benchmarks

- **PR #30, #32 — Agglomerative GPU exact paths**:
  - GPU-accelerated exact linkage for all distance metrics

- **PR #33 — Nonparametric module review**:
  - GPU memory fixes for KDE
  - Bandwidth selection GPU化
  - Log-sum-exp stabilization

- **PR #34, #35 — Documentation**:
  - Clarified runtime device selection
  - Explicit Torch backend docs

### Added (2026-05-24 ~ 2026-05-29)

- **PR #37 — GLM penalty correctness + auto GPU routing**:
  - Fixed penalized GLM predict() to return inverse-link mean-scale predictions
  - Auto GPU routing for penalized models

- **PR #38 — Gamma inverse-power FISTA**:
  - Link-aware Gamma FISTA support across CPU/CuPy/Torch

- **PR #39~#42 — GLM solver refactoring**:
  - Fixed GLM GPU dtype and review regressions
  - Refactored GLM solver backend helpers

- **PR #43, #44 — Linear inference result fixes**:
  - Refactored linear inference result containers

- **PR #47 — CuPy cummin/cummax fix**:
  - Fixed CuPy cummin/cummax CUDA kernels on non-contiguous arrays
  - adjust_pvalues BH/BY/Hochberg now returns correct results

### Fixed (2026-05-20)

- **v23c: L-BFGS fused penalty gradient fix**:
  - L-BFGS converged to unregularized solution instead of loss_grad + α·coef = 0
  - 1043/1043 ALL PASS

### Optimized (2026-05-20)

- **v22g: Async FISTA and GPU optimizations**:
  - Async FISTA for non-smooth penalties: 2-5.5x speedup at n=5000
  - GPU sync optimizations for CuPy/Torch backends

- **v23c: Full matrix benchmark (1043 tests)**:
  - 7 families x 10 penalties x 3 scales x multiple solvers x 3 backends

## 2026-04

### Added (2026-04-26)

- **PR #24 — Precision fixes, hochberg/stouffer, package restructure**:
  - Ordered Model Cross-Backend Precision Fixes
  - New hochberg (adjust_pvalues) + stouffer (combine_pvalues) across 3 backends
  - Package Structure Audit & Reorganization

- **PR #26 — README refresh**:
  - Reorganized features, added models, recommended editable install
  - Exported combine_pvalues

### Added (2026-04-21)

- **PR #19 — Cython Efron optimization**:
  - Cython-optimized Efron gradient and Hessian computation
  - Comprehensive CoxPH accuracy and runtime benchmarks

- **PR #21 — Distribution backends unification**:
  - Consolidated into single `_distributions_backend.py`
  - 15 distributions across 3 backends

- **PR #22 — Backend utility consolidation**:
  - Consolidated duplicated backend utility functions

- **CoxPHCV upgraded from skeleton to trainable**:
  - Implemented K-fold penalty search and final refit

- **RidgeCV and LogisticRegressionCV Full Implementation**:
  - Upgraded from scaffolding to full-featured implementation

### Added (2026-04-20)

- **PR #18 — Remote config + backend enhancements**:
  - Removed hardcoded SSH credentials
  - Added remote config module

- **PR #20 — CoxPHCV CuPy optimization**:
  - Optimized CoxPHCV CuPy Hessian path

- **CoxPH Efron Implementation Fix**:
  - Fixed numerical overflow with clipping protection
  - statgpu-Torch GPU achieves **15.44x** speedup (vs statsmodels)

### Added (2026-04-18)

- **PR #16 — Torch backend support**:
  - Comprehensive PyTorch backend integration
  - Feature parity with NumPy and CuPy backends

- **PR #17 — Elastic Net implementation**:
  - New `ElasticNet` class combining L1 and L2 regularization
  - Maximum speedup: **4.36x** vs sklearn (n=100k, p=500)

- **PyTorch Backend Fixes**:
  - Fixed multiple Torch compatibility issues
  - Numerical accuracy: coefficients match NumPy within 1e-14

### Added (2026-04-17)

- **PyTorch Backend (Phase 1-5 complete)**:
  - New GPU backend alternative to CuPy using PyTorch 2.0+
  - Ridge, LogisticRegression, Lasso, CoxPH all supported
  - Large-Scale Performance: Ridge HC3 Torch 0.067s vs CuPy 0.064s

### Added (2026-04-15)

- Knockoff feature-selection API
- Lasso inference rename
- `gpu_memory_cleanup` for all current models
- Robust covariance: `nonrobust/hc0/hc1/hc2/hc3/hac` for all models
- CV estimator interface skeletons: RidgeCV, LogisticRegressionCV, CoxPHCV
- Nonparametric exports and API coverage

### Validation

- Consistency tests against `statsmodels` for robust covariance
- Cox consistency checks vs `statsmodels.PHReg`
- Nonparametric validation: KDE, KernelRegression

### Improved

- LinearRegression CPU HAC path uses adaptive precision selection
- Kernel regression local-linear multidim path uses batched vectorized solves
- KDE 1D Numba fast path improved
