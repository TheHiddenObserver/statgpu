# Changelog

All notable changes to statgpu are documented here, organized by date and PR.

### PR #73 — LossBase Extraction, Proximal IRLS-CD, CoxPH Efron optimization
- Extracted LossBase from GLMLoss; added QuantileLoss, HuberLoss, BisquareLoss, CoxPartialLikelihoodLoss
- New penalized models: PenalizedQuantileRegression, PenalizedRobustRegression, PenalizedCoxPHModel
- Proximal IRLS-CD solver: quantile+SCAD/MCP, ~3x CPU/49x GPU speedup (Tesla P100, n=10K, p=500)
- CoxPH: vectorized Efron gradient/Hessian, multi-block CUDA kernel, statsmodels reference parity
- UMAP sparse COO graph (O(n·k)), NNDescent module, DBSCAN CuPy label propagation
- Sample weight global backend handling; GPU convergence optimization; 13 bug fixes

## 2026-06-26

### Unsupervised Benchmark — 12 Algorithms × 3 Backends

**Complete benchmark** (PCA, KMeans, GMM, NMF, TruncatedSVD, IncrementalPCA, DBSCAN, Agglomerative, UMAP, TSNE, MiniBatchKMeans, MiniBatchNMF):
- Best GPU speedups: TruncatedSVD **28.6x**, IncrementalPCA **21.9x**, DBSCAN **21.0x**, NMF **19.9x**
- vs sklearn: IncrementalPCA **39.0x**, TruncatedSVD **21.6x**, DBSCAN **7.5x**
- Results: `results/unsupervised_bench_2026-06-26.json`

### Unsupervised — DBSCAN Optimization

- **Cython fast path** (`_dbscan_cy_fast.pyx`): two entry points — `dbscan_labels_from_pairs` (from `query_pairs`) and `dbscan_labels_from_csr` (from CSR graph). Both run counting, Union-Find, and label assignment entirely in C.
- **CPU hybrid strategy**: low-dim (p≤12) uses cKDTree `query_pairs` + Cython; high-dim (p>12) uses sklearn `radius_neighbors_graph` + Cython CSR.
- **Fully GPU pipeline** (PyTorch CUDA): distance → sparse graph → label propagation → border assignment, all on-device. Zero GPU→CPU transfer until final labels. Single-pass distance computation avoids OOM.
- **GPU label propagation**: connected components via `scatter_reduce_(amin)`, fully parallel over edges. Typically converges in 2-5 iterations.
- CPU effect: p=5 3-4x faster than sklearn, p=50 matches sklearn.
- GPU effect (Tesla P100): p=5 **14-17x** faster than sklearn, p=50 **3-4x** faster. ARI=1.0 for all cases.

### Unsupervised — UMAP Optimization

- **Sparse graph + negative sampling**: replaced dense n×n epoch loop with sparse edge iteration
- **GPU-native scatter-add**: no CPU transfers in optimization loop
- Effect: 10K GPU from 325s → 19.4s (**16.7x**), 1K torch from 3.7s → 0.81s (**4.6x**)
- `nn_method` parameter: `"auto"`, `"exact"`, `"nndescent"` for NNDescent support
- Epoch reduction for large data (10K: 500→200, >10K: 500→100)

### Unsupervised — IncrementalPCA & MiniBatchNMF

- **IncrementalPCA**: default `batch_size` changed from `min(n, 5*p)` to `n` (process all at once)
  - Effect: GPU 0.4x → **21.9x**
- **MiniBatchNMF**: auto-size batch, pre-compute HtH per epoch, throttle convergence check
  - Effect: GPU 0.1x → **3.2x**

### CuPyBackend — Missing Methods

Added 30+ methods to match NumpyBackend/TorchBackend:
- `qr`, `svd`, `solve`, `norm`, `bool`, `nan`, `inf`, `pi`
- `zeros_like`, `ones_like`, `full_like`, `isnan`, `isinf`, `nan_to_num`
- `count_nonzero`, `any`, `all`, `unique`, `sort`
- `reshape`, `flatten`, `squeeze`, `astype`, `cat`, `concatenate`
- `einsum`, `tensordot`, `meshgrid`, `item`, `empty_cache`
- Effect: TruncatedSVD, IncrementalPCA, DBSCAN GPU backends now functional

### TorchBackend — Missing Methods

Added `qr`, `svd`, `solve` methods.

### Backend Utilities — Unified Scatter-Add

- **`scatter_add_1d`**: 1D scatter-add across numpy/cupy/torch
- **`scatter_add_2d`**: 2D scatter-add (row-wise) across numpy/cupy/torch
- Used by UMAP optimization loop; available for other modules

### Build System — Consolidated setup.py

- Merged 7 separate setup files into single `setup.py`
- All 5 Cython extensions: `_cox_efron_cy`, `_dbscan_cpu`, `_dbscan_cy_fast`, `_kdtree`, `_unionfind`
- Deleted: `setup_cython.py`, `setup_dbscan_cy.py`, `setup_dbscan_fast.py`, `setup_kdtree.py`, `setup_kdtree_cy.py`, `setup_unionfind.py`

## 2026-06-24

### Benchmark Suite — GLM Solver, New Modules, Unsupervised

**GLM Solver Benchmark** (7 families × 10 penalties × 7 solvers × 3 backends):
- Complete 3D matrix: 70 family×penalty combinations, all valid solver choices
- 3 backends: numpy, cupy, torch (Tesla P100-SXM2-16GB)
- Top speedups: NB+none+irls **101.8x**, tweedie+none+newton **88.7x**, gamma+none+newton **82.8x**
- Results: `results/glm_solver_benchmark_2026-06-23.json`

**New Modules Benchmark** (Panel Data, GAM, ANOVA):
- Panel: 8 estimators × 3 backends × 3 scales; best: PanelOLS_two_way **19.9x** (torch)
- GAM: 3 scales; best: **22.7x** at 100K obs (torch); aligned with pygam (0.25% pred diff)
- ANOVA: 5 functions × 3 backends; f_oneway **3.4x** (cupy) after vectorization
- vs external: PanelOLS **16.7x** vs linearmodels, GAM **51.3x** vs pygam, f_oneway **2.2x** vs scipy
- Results: `results/new_modules_full_2026-06-24.json`

**Unsupervised Benchmark** (12 algorithms × 3 backends):
- Best: IncrementalPCA **21.1x**, TruncatedSVD **27.6x**, NMF **20.6x**, GMM **13.0x**
- vs sklearn: IncrementalPCA **37.7x**, DBSCAN **23.8x**, TruncatedSVD **21.7x**
- Results: `results/unsupervised_bench_2026-06-24.json`

### CuPyBackend — Missing Methods

Added 30+ methods to match NumpyBackend/TorchBackend:
- **Linear algebra**: `qr`, `svd`, `solve`, `norm`
- **Dtype properties**: `bool`, `nan`, `inf`, `pi`
- **Array creation**: `zeros_like`, `ones_like`, `full_like`
- **Element-wise**: `isnan`, `isinf`, `nan_to_num`, `square`, `log1p`, `sign`
- **Reduction**: `count_nonzero`, `any`, `all`, `unique`, `sort`
- **Manipulation**: `reshape`, `flatten`, `squeeze`, `astype`, `cat`, `concatenate`, `einsum`, `tensordot`, `meshgrid`, `item`
- **Memory**: `empty_cache`
- Effect: TruncatedSVD, IncrementalPCA, DBSCAN GPU backends now functional

### TorchBackend — Missing Methods

Added `qr`, `svd`, `solve` methods for TruncatedSVD/IncrementalPCA support.

### Unsupervised — IncrementalPCA

- **Fix**: Default `batch_size` changed from `min(n, 5*p)` to `n` (process all at once)
- **Effect**: GPU speedup from 0.4x → **21.1x** at 100K scale
- Old behavior forced 200+ batch iterations with SVD each time

### Unsupervised — MiniBatchNMF

- **Fix**: Default `batch_size` auto-sized to `min(n, max(20000, n//5))` (was 1024)
- **Fix**: Pre-compute HtH once per epoch (was recomputed 3× per batch)
- **Fix**: In-place multiply/divide for W updates (reduced allocations)
- **Fix**: Throttle convergence check to every 5 epochs on GPU (was every epoch)
- **Effect**: GPU speedup from 0.1x → **3.2x** at 100K scale

### Unsupervised — UMAP

- **New**: `nn_method` parameter: `"auto"` (default), `"exact"`, `"nndescent"`
  - `"auto"`: uses NNDescent for n > 5000 (if pynndescent installed), exact otherwise
  - `"nndescent"`: requires `pip install pynndescent`
- **Optimization**: Reduced epochs for large data (10K: 500→200, >10K: 500→100)
- **Optimization**: Float32 for distance matrix computation (2x memory savings)
- **Optimization**: Torch `topk` instead of `argsort` for nearest neighbor search

### ANOVA — f_oneway Vectorization

- **Fix**: Vectorized group statistics (concatenate + scatter-add instead of Python loop)
- **Effect**: cupy speedup from 0.7x → **3.4x** at 2M observations

### ANOVA — f_twoway Torch Fix

- **Fix**: `np.asarray` → `xp.asarray` for torch dtype compatibility
- **Fix**: `arr.size` → `arr.numel()` for torch tensor compatibility
- **Effect**: f_twoway now works with torch backend; cupy **3.9x** speedup

### Panel — BetweenOLS

- **Fix**: Added `time_ids=None` parameter to `fit()` for API consistency
- Other panel models (PanelOLS, RandomEffects, FirstDifferenceOLS, FamaMacBeth) already accept `time_ids`

### GAM — Parameter Alignment

- **New**: `knot_method` parameter: `"quantile"` (default), `"uniform"`
  - `"uniform"`: matches pygam's knot placement for fair comparison
- **New**: `gamma` parameter for GCV (default 1.0, use 1.4 to match pygam Wood 2006)
- **Precision**: With aligned params, pred rel_diff from 2.5% → **0.25%** vs pygam

## 2026-06-19

### LossBase Architecture — Phase 1

- **LossBase**: Extracted generic base class from `GLMLoss`; all loss types share penalty/solver infrastructure
- **QuantileLoss**: Pinball loss for quantile regression (R `quantreg::rq()`)
- **HuberLoss**: Robust M-estimator loss (R `MASS::rlm()`)
- **CoxPartialLikelihoodLoss**: Cox PH negative log partial likelihood (R `survival::coxph()`), Breslow+Efron ties
- **Loss Registry**: `register_loss()`, `get_loss()`, `list_losses()` — 10 total losses registered
- `GLMLoss` inherits `LossBase` (backward compatible); solver type hints updated
- 64 tests, all passing; model docs in English + Chinese

## 2026-06-17

### PR #72 — P2 modules: ANOVA, Covariance, Panel, Splines, Kernel methods

- **ANOVA**: `f_twoway` (two-way with/without interaction), `f_welch` (unequal variances), `tukey_hsd`, `bonferroni` (post-hoc), `cohens_f`, `partial_eta_squared` (effect sizes)
- **Covariance**: `ShrunkCovariance`, `MinCovDet` (FAST-MCD, matches sklearn corr=1.0), `GraphicalLasso`, `GraphicalLassoCV`
- **Panel**: `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, `FamaMacBeth`, `hac_covariance` (Newey-West HAC)
- **Splines**: `SplineTransformer` (sklearn API), `cyclic_cubic_spline_basis`, `thin_plate_spline_basis`
- **Kernel**: `chi2_kernel`, `Nystroem`, `KernelPCA`; RBF kernel optimized (3.5-13x faster than sklearn on CPU)
- 112 new tests, all passing; 3-backend benchmark on Tesla P100

## 2026-06-15

### PR #66 — Code review round 10: final bug fixes

**Bug fixes:**
- **`fista_lla_path` ignored `sample_weight` in XtX fast paths**: both the fused GPU path and the numpy path used unweighted Gram matrix for squared_error gradient, silently ignoring sample_weight — fixed by gating on `sample_weight is None`
- **Missing `xp_ones` import in `_fit_gpu_backend`**: NameError when `n_features >= 1000` on GPU (power-iteration Lipschitz path)
- **Removed stale `t_k = t_new` after Nesterov refactor**: `t_new` was undefined after `_nesterov_momentum` already updates `t_k` via tuple unpacking
- **Removed dead debiased inference block in torch exact solver**: exact solver = L2 only, debiased = L1/ElasticNet only, mutually exclusive

**Tests added:**
- `TestWeightedSCADMCP`: verifies weighted SCAD produces different coefficients than unweighted (regression for XtX sample_weight bug)
- `TestFitGpuBackendImports`: verifies `_fit_gpu_backend` imports `xp_ones` (regression for large-feature GPU path)

**Review coverage:**
- Round 10a: solvers, penalized mixin, CV, glm_core, backends (found 1 bug)
- Round 10b: inference, predict, penalties, cross_validation (found 1 bug)
- Round 10c: wrappers, glm_base, metrics, feature_selection, nonparametric, panel, survival (no bugs)

### PR #66 — Code review round 9: bug fixes, performance, cleanup

**Critical bug fixes:**
- **Newton solver convergence**: `_norm2_dev` returns L2 norm, not squared norm — convergence check was 10,000x too strict (`tol²` instead of `tol`), causing excessive iterations
- **Import path crash**: `_resolve_loss_name` imported from `_base.py` where it doesn't exist — CV fold pipeline (`_cv_fold_general`) would crash with `ImportError`

**High-severity bug fixes:**
- **ElasticNet Lipschitz**: `_smooth_penalty_lipschitz` returned 0 for the `"en"` alias (missing smooth L2 component), causing incorrect FISTA step sizes for ElasticNet via alias
- **Inference attribute cleanup**: Debiased inference cleared `_resid`, `_X_design`, `_y` after completion, breaking downstream properties (`rsquared`, `rsquared_adj`, `fvalue`, `aic`, `bic`) — now preserved
- **Missing `_df_resid`**: Debiased inference paths never set `_df_resid`, breaking `rsquared_adj` and `fvalue` — now set as fallback

**Medium bug fixes:**
- **FISTA-BB divergence return**: `_coef_best` returned without `_copy_arr()`, inconsistent with all other return paths

**Performance fixes:**
- **Deleted `_solver_utils.py`** (442 lines): complete duplicate of `solvers/_utils.py` + `_linesearch.py` + `_fused.py` — no imports referenced it
- **IRLS `_to_backend(y)` recomputation**: hoisted outside `_dev_val` closure — was called on every Armijo backtracking step (up to 30x per iteration)
- **IRLS redundant `X @ params_old`**: reuse `eta_raw` computed at top of iteration instead of recomputing O(n*p) matmul
- **Fused dispatch dict**: promoted from per-call construction to module-level `_FUSED_DISPATCH` constant

**Cleanup:**
- Removed unused `import numpy as np` from `_fused.py`
- Moved `_resolve_backend` import inside the function that uses it (consistent lazy import pattern)
- Removed dead `if solver_name != "lbfgs": return` code in `_validate_solver_penalty`
- Removed duplicate entries in top-level `__init__.py` (PenalizedGLM_CV, ApproximateCVWarning, Gamma/Tweedie/InvGauss/NegBin imported twice)
- Deleted dead `_linesearch.py` (compiled step functions never imported by any solver)
- Fixed legacy `_penalized_legacy.py` crash: referenced `_get_selective_penalty_singleton()` without importing it
- Replaced `SelectivePenalty` thread-local singleton with fresh-per-call instance (avoids same-thread conflicts in nested CV)
- Cached `_family_for_loss()` result (avoids re-creating Family objects on every `predict()`/`score()`)
- Replaced `xp.sum(sw * ps)` with `xp.dot(sw, ps)` in `GLMLoss.value()`/`fused_value_and_gradient()` and `_weighted_mean()` (avoids O(n) temporary allocation)
- Replaced 8x `try/except TypeError` blocks in `_fista_bb.py` and `_fista.py` with `_call_with_weight()` helper (DRY, no longer swallows internal TypeErrors)

**Refactoring:**
- **Unified `_fit_gpu`/`_fit_torch` into single `_fit_gpu_backend` method** (-468 lines): uses `_get_xp()`, `xp_asarray`, `xp_zeros`, `xp_copy`, `_to_numpy` for backend-agnostic operations; `getattr` dispatch for backend-specific exact solver and cleanup methods
- Extracted `_nesterov_momentum(t_k, beta_cap)` and `_nesterov_update(coef, coef_old, t_k, beta_cap)` helpers — replaced 12 duplicated Nesterov momentum sites across 6 files
- Extracted gradient clipping constants (`_GRAD_CLIP_COEF_FACTOR`, `_GRAD_CLIP_ABS_FLOOR`, `_GRAD_CLIP_MAX`) to `solvers/_constants.py` — replaced magic numbers in 4 files
- Added `_soft_threshold_gpu(w, thresh, xp)` static method for backend-agnostic soft-thresholding
- Added type hints to all public solver function signatures (`fista_solver`, `fista_bb_solver`, `newton_solver`, `lbfgs_solver`, `admm_solver`)

## 2026-06-14

### Refactor: Top-level structure reorganization (Phases 0-6)
- **Phase 1**: Extracted `solvers/` as top-level generic module (6 solvers: fista, fista_bb, fista_lla, newton, lbfgs, admm)
- **Phase 2**: Extracted `cross_validation/` module (CVEstimatorBase, kfold_indices, hash_cv_data, run_cv)
- **Phase 3**: Moved wrappers into `linear_model/wrappers/` (10 model files, renamed _gamma_glm → _gamma etc.)
- **Phase 4**: Split `_penalized.py` (3968 lines) into mixin architecture (_base + _fit_mixin + _inference_mixin + _predict_mixin)
- **Phase 5**: Moved CV wrappers into `linear_model/cv/` (LassoCV, RidgeCV, ElasticNetCV, LogisticRegressionCV)
- **Phase 6**: Cleaned up nonparametric/ duplicate files, added DeprecationWarning to kernel_methods/ and splines/ shims
- **refactor**: Moved GLM-specific fused functions to `glm_core/_fused.py`
- **refactor**: Added optimization hint attributes to GLMLoss base class
- **refactor**: All old import paths preserved as backward-compatible shims (DeprecationWarning, remove in v0.3.0)
- **test**: Added 45 safety net tests + Phase 1-6 verification stubs
- New modules: `solvers/`, `cross_validation/`, `linear_model/wrappers/`, `linear_model/penalized/`, `linear_model/cv/`

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
