# Changelog

All notable changes to statgpu are documented here, organized by date and PR.

## 2026-06-14

### PR #63 — Dev workspace documentation
- Added dev/README.md (directory structure, remote GPU setup, archive policy)
- Added dev/tests/TESTING.md (test categories, remote workflow)
- Added dev/benchmarks/RESULTS.md (GPU speedup data, version history)
- Added dev/design/ARCHITECTURE.md (backend abstraction, GLM solver architecture)

### PR #62 — Dev folder reorganization
- Archived 241 old/temp files from tests/, benchmarks/, scripts/ to _archive/
- Updated remote_config.py: env vars now override local config
- Removed plaintext password from remote_config_local.py

### PR #61, #60, #59 — Documentation cleanup
- Compressed README GLM section, cleaned up Implemented Methods tables
- Documentation, changelog, and guides (PR-E)

### PR #58 — Infrastructure, exports, backward compatibility (PR-D)
- Unified statgpu/__init__.py exports
- BaseEstimator with device management
- Device enum (CPU/CUDA/TORCH/AUTO)
- nonparametric/__init__.py re-exports
- CoxPH/CoxPHCV updated backend integration

### PR #57 — New modules (PR-C)
- **ANOVA**: `f_oneway` — GPU-accelerated one-way ANOVA, float32/float64
- **Covariance**: `EmpiricalCovariance`, `LedoitWolf`, `OAS`
- **Panel Data**: `PanelOLS` (fixed effects), `RandomEffects`, `PanelSummary`, clustered covariance
- **Splines**: `bspline_basis`, `natural_cubic_spline_basis`, penalized regression with GCV
- **Semiparametric**: `GAM` (penalized B-splines + GCV smoothing)
- **Kernel Methods**: `KernelRidge`, `KernelRidgeCV`, 6 kernel functions

### PR #56 — Penalized models + CV framework (PR-B)
- 7 Penalized estimators: PenalizedLinearRegression, PenalizedLogisticRegression, PenalizedPoissonRegression, PenalizedGammaRegression, PenalizedInverseGaussianRegression, PenalizedNegativeBinomialRegression, PenalizedTweedieRegression
- PenalizedGLM_CV: full CV over families x penalties x solvers
- Lasso, Ridge, ElasticNet with full inference
- LogisticRegression, LinearRegression with GPU

### PR #55 — Core GLM solver, backends, penalties, inference (PR-A)
- 7 GLM families: squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie
- 10 penalties: none, l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
- 6 solvers: irls, fista, fista_bb, admm, lbfgs, newton
- 3 backends: NumPy (CPU), CuPy (CUDA), PyTorch (CUDA)
- Unified inference: distributions, p-value adjustment, bootstrap, permutation test

## 2026-06-07 ~ 2026-06-13

### PR #49 — Unified CV framework + bug fixes
- `_cv_base.py`: shared kfold_indices, CVCache, batch_mse
- `_cv_engine.py`: generic CV loop engine
- `_penalized_cv.py`: PenalizedGLM_CV with full family x penalty x solver matrix
- 110+ bug fixes across 16 files, 428 test cases added
- Cross-backend precision < 0.02%

### PR #48 — Module reorganization
- Moved kernel_methods/ and splines/ under nonparametric/
- Created kernel_smoothing/ subpackage for KDE + kernel regression
- Extracted GAM to semiparametric/ package
- Backward-compat shims for old import paths

### PR #54 — Refactor CV dispatch table
- Dispatch table for _compute_cv_scores

### PR #53 — Fix weighted Ridge inference
- Correct scale, preserve bse/pvalues/conf_int

### PR #50 — Add val_sample_weight to GLM sparse CV path

## 2026-05-24 ~ 2026-05-29

### PR #47 — CuPy cummin/cummax fix + Poisson IRLS precision
- Fixed CuPy cummin/cummax CUDA kernels on non-contiguous arrays
- adjust_pvalues BH/BY/Hochberg now returns correct results
- Poisson IRLS precision improvements

### PR #44, #43 — Linear inference result fixes
- Refactored linear inference result containers
- Merged fixes into GPU feature branch

### PR #42, #41, #40, #39 — GLM solver refactoring
- IRLS solve backend aliases
- Refactored GLM solver backend helpers
- Fixed GLM GPU dtype and review regressions

### PR #38 — Gamma inverse-power FISTA
- Link-aware Gamma FISTA support across CPU/CuPy/Torch
- Fixed objective mismatch for inverse-power link

### PR #37 — GLM penalty correctness + auto GPU routing
- Fixed penalized GLM predict() for positive families
- Auto GPU routing for penalized models

## 2026-05-03 ~ 2026-05-15

### PR #35, #34 — Documentation
- Clarified runtime device selection
- Explicit Torch backend docs
- README installation and requirements

### PR #33 — Nonparametric module review
- GPU memory fixes for KDE
- Bandwidth selection GPU化
- Log-sum-exp stabilization

### PR #32, #30, #29, #28, #27 — Unsupervised learning
- Phase 3/3B/3C estimators: PCA, KMeans, DBSCAN, GaussianMixture, NMF, AgglomerativeClustering, UMAP, TSNE, MiniBatchKMeans, MiniBatchNMF, IncrementalPCA, TruncatedSVD
- GPU exact paths for agglomerative clustering
- Documentation and validation benchmarks

## 2026-04-20 ~ 2026-04-29

### PR #26 — README refresh
- Reorganized features, added models, recommended editable install
- Exported combine_pvalues

### PR #24 — Precision fixes + new methods
- Ordered model cross-backend precision fixes
- Hochberg/Stouffer methods added
- Package restructure
- GPU kernel fixes

### PR #22, #21 — Backend refactoring
- Consolidated duplicated backend utility functions
- Unified distribution backends (numpy/cupy/torch) into single _distributions_backend.py
- 15 distributions across 3 backends

### PR #20 — CoxPHCV CuPy optimization
- Optimized CoxPHCV CuPy Hessian path and defaults

### PR #19 — Cython Efron optimization
- Cython-optimized Efron gradient and Hessian computation
- Comprehensive CoxPH accuracy and runtime benchmarks

### PR #18 — Remote config + backend enhancements
- Removed hardcoded SSH credentials
- Added remote config module
- Backend enhancements

## 2026-04-13 ~ 2026-04-18

### PR #17 — Elastic Net implementation
- Optimized Elastic Net with benchmarks
- statgpu vs sklearn comparison

### PR #16 — Torch backend support
- Comprehensive PyTorch backend integration
- Feature parity with NumPy and CuPy backends
- Memory management improvements

### PR #15 — Lasso inference GPU support
- Lasso debiased inference with GPU support
- Ridge inference GPU/CPU comparison tolerance relaxed
- Enhanced CoxPH and Knockoff documentation

### PR #14 — Kernel regression + Lasso GPU optimization
- Nonparametric kernel methods: KDE, kernel regression
- Lasso GPU computation optimization
- Extensive validation and benchmarks

### PR #13 — F-test p-value handling
- Perfect fit F-test p-value handling
- Lasso p-value calculation edge cases

### PR #12 — Distribution compatibility layer
- Legacy distribution function compatibility
- Refactored inference methods

## 2026-04-03 ~ 2026-04-11

### PR #11 — Documentation for new models
- Knockoff feature selection documentation
- New model documentation

### PR #10 — HAC covariance support
- HAC covariance for LinearRegression and LogisticRegression
- Newey-West bandwidth selection

### PR #6 — Logistic Regression evaluation metrics
- Comprehensive evaluation metrics
- ROC, AUC, confusion matrix

### PR #5 — Ridge inference support
- Full inference parity with LinearRegression
- cov_type: nonrobust/hc0/hc1 (CPU + GPU)
- summary(), rsquared_adj, fvalue, f_pvalue, llf, aic, bic

### PR #4 — Pluggable backends abstraction
- BackendBase ABC with NumPy/CuPy/Torch implementations
- Removed redundant model implementations
- Clean path for multi-backend support

### PR #1 — CoxPH cluster-robust covariance
- cluster-robust covariance for CoxPH
- Breslow tie handling
- Benchmarking scripts

---

## GPU Performance Milestones

### v23c — 1043/1043 ALL PASS (100%)
- Full matrix: 7 families x 13 penalties x 5 solvers x 3 backends
- L-BFGS fused penalty gradient fix

### v22e — Async FISTA
- Eliminated per-iteration GPU->CPU synchronization
- logistic + L1: 2.22x -> **5.41x** (n=5000, p=500)
- logistic + ElasticNet: 2.18x -> **5.17x**

### v20b — Kernel fusion + D2H batching
- Reduced kernel launch overhead

### v17f — Torch SCAD/MCP fix
- GPU sync optimizations

### v15 — 531/533 (99.6%)
- 2 remaining FISTA+L2 edge cases
