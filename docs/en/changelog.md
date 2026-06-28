# Changelog

> Language: English  
> Last updated: 2026-06-28
> This page: Changelog  
> Switch: [Chinese](../changelog.md)

Language switch: [Chinese](../changelog.md)

## 2026-06

### Added (2026-06-28) — PR #73

- **Loss Architecture — LossBase Extraction**:
  - Extracted `LossBase` from `GLMLoss` for quantile/robust/survival losses
  - `LossBase`: abstract base with `per_sample_value()`, `per_sample_gradient()` as single source of truth; derives `value()`, `gradient()`, `fused_value_and_gradient()` automatically
  - `GLMLoss` inherits from `LossBase` for GLM-specific features (canonical link, IRLS)
  - New loss classes: `QuantileLoss`, `HuberLoss`, `BisquareLoss`, `CoxPartialLikelihoodLoss`
  - New modules: `PenalizedQuantileRegression`, `PenalizedRobustRegression`, `PenalizedCoxRegression`

- **Proximal IRLS-CD Solver**: New solver for quantile + SCAD/MCP
  - Algorithm: IRLS quadratic majorization + LLA nonconvex penalty + parallel diagonal majorization
  - CPU (numpy): ~3x faster than FISTA-LLA (60-120 iterations vs 1800+)
  - GPU (torch-CUDA): ~36x faster than CPU numpy for large problems (n=10K, p=500)
  - Three-backend: numpy, cupy, torch — fully GPU-native, no CPU round-trips

- **CoxPH Efron Optimization**:
  - Vectorized Efron: prefix-sum based gradient/Hessian computation (no Python loops)
  - Multi-block CUDA kernel: fused loglik+grad+hess for Efron on GPU
  - DLPack bridge: torch-CUDA uses CuPy Efron kernel via DLPack
  - Performance: 3-6x faster than statsmodels at n=5000; GPU 6x faster than CPU
  - Removed Numba dependency, pure numpy implementation

- **GLM Fused Value+Gradient**: Integrated `_fused.py` into `GLMLoss.fused_value_and_gradient()`
- **FISTA GPU Sync Optimization**: Batch GPU syncs (convergence+divergence+lipschitz in one transfer)
- **Quantile IRLS Solver**: `QuantileLoss.irls()` for fast convergence with smooth penalties (5-15 iterations)
- **Huber Hessian Support**: `has_hessian = True`, enables proximal Newton (5-10 iterations)
- **Bisquare + SCAD/MCP Fix**: Empty active sets for alpha >= 0.1

- **Refactoring**:
  - Extracted `_compute_lla_path()` shared helper
  - Renamed `_NON_IRLS_LOSSES` → `_SPECIAL_LLA_LOSSES`
  - Renamed `_cd_sweep_batch` → `_parallel_majorization_step`
  - Added `_dispatch_irls()` method for IRLS backend routing

- **Numerical Stability**: IRLS weight clamping, SCAD denominator zero protection, CoxPH Efron `inv_d1_sq` clamping

- **Bug Fixes**:
  - Group penalties: cupy compatibility, device-aware cache
  - Huber: correct `per_sample_value` formula
  - Quantile IRLS: skip intercept column penalty
  - Proximal Newton: pass `sample_weight`
  - DBSCAN: `min_samples` off-by-one, indices/distances swap, GPU propagation to convergence
  - NNDescent: exclude self-candidates
  - Cox C-index: exclude censored shorter times
  - CV scoring: pass loss kwargs
  - ANOVA: torch device mismatch

### Added (2026-06-26)

- **Unsupervised Benchmark**: 12 algorithms × 3 backends, vs sklearn
  - Best: TruncatedSVD 28.6x, IncrementalPCA 21.9x, DBSCAN 21.0x, NMF 19.9x

- **DBSCAN Optimization**:
  - Cython `_dbscan_cy_fast.pyx`: `dbscan_labels_from_pairs` + `dbscan_labels_from_csr` — full pipeline in C
  - CPU: p≤12 cKDTree query_pairs + Cython (3-4x sklearn); p>12 sklearn BLAS + Cython CSR (matches sklearn)
  - GPU (PyTorch CUDA): fully on-device pipeline — distance, sparse graph, label propagation, border — zero GPU→CPU transfer
  - GPU label propagation via `scatter_reduce_(amin)`, 2-5 iterations to converge
  - GPU (P100): p=5 **14-17x** faster than sklearn, p=50 **3-4x** faster

- **UMAP Optimization**:
  - Sparse graph + negative sampling (16.7x GPU speedup)
  - GPU-native scatter-add (no CPU transfers)
  - `nn_method` parameter for NNDescent support

- **IncrementalPCA**: batch_size default → n (GPU 0.4x → 21.9x)
- **MiniBatchNMF**: auto batch, HtH pre-compute, throttled sync (GPU 0.1x → 3.2x)

- **CuPyBackend**: Added 30+ missing methods (qr, svd, bool, zeros_like, etc.)
- **TorchBackend**: Added qr, svd, solve
- **Backend Utils**: Unified `scatter_add_1d` and `scatter_add_2d`
- **Build**: Consolidated 7 setup files into single `setup.py`

### Added (2026-06-24)

### Added (2026-06-24)

- **Comprehensive Benchmark Suite**:
  - GLM Solver: 7 families × 10 penalties × 7 solvers × 3 backends (70 combos)
  - New Modules: Panel (8 estimators), GAM, ANOVA (5 functions) — 3 backends × 3 scales
  - Unsupervised: 12 algorithms × 3 backends vs sklearn
  - External comparison: statgpu vs linearmodels, pygam, scipy, sklearn

- **CuPyBackend**: Added 30+ missing methods (qr, svd, bool, zeros_like, solve, norm, etc.)
  - TruncatedSVD, IncrementalPCA, DBSCAN GPU backends now functional

- **TorchBackend**: Added qr, svd, solve methods

- **Unsupervised Optimizations**:
  - IncrementalPCA: batch_size default → n (GPU 0.4x → 21.1x)
  - MiniBatchNMF: batch auto-sizing + HtH pre-compute + throttled sync (GPU 0.1x → 3.2x)
  - UMAP: `nn_method` parameter (auto/exact/nndescent), epoch reduction, float32

- **ANOVA Fixes**:
  - f_oneway: vectorized group statistics (cupy 0.7x → 3.4x)
  - f_twoway: torch dtype compatibility fix

- **Panel**: BetweenOLS accepts `time_ids` parameter for API consistency

- **GAM**: `knot_method` (quantile/uniform) and `gamma` parameters for pygam alignment

### Added (2026-06-19)

- **LossBase Architecture** (Phase 1):
  - Extracted `LossBase` from `GLMLoss` as generic base class for all loss functions
  - `GLMLoss` now inherits from `LossBase` (backward compatible)
  - New loss types automatically get all 10 penalties and 6 solvers
  - Solver type hints updated from `GLMLoss` to duck-typed `LossBase` (fista, newton, lbfgs, admm)

- **New Loss Types**:
  - `QuantileLoss`: Pinball loss for quantile regression (matches R `quantreg::rq()`)
    - `smooth_gradient=False` for FISTA proximal handling
    - Supports all quantiles in (0, 1)
  - `HuberLoss`: Robust M-estimator loss (matches R `MASS::rlm()`)
    - `smooth_gradient=True`, `has_hessian=False`
    - Recovers OLS for large delta; robust to outliers for small delta
  - `CoxPartialLikelihoodLoss`: Cox PH negative log partial likelihood (matches R `survival::coxph()`)
    - Breslow and Efron tie handling
    - `has_hessian=True` for Newton solver
    - CPU-only (numpy); for GPU use `statgpu.survival.CoxPH` directly
    - Fused `fused_value_and_gradient()` avoids redundant X @ beta computation

- **Loss Registry** (`statgpu.losses._registry`):
  - `register_loss(name)`: Decorator to register custom loss classes
  - `get_loss(name, **kwargs)`: Factory function for loss instantiation
  - `list_losses()`: Lists all registered losses (GLM + non-GLM)
  - GLM losses auto-registered via `register_glm_loss` cross-registration

- **Files Created**: `statgpu/losses/__init__.py`, `_base.py`, `_registry.py`, `_quantile.py`, `_huber.py`, `_cox_ph.py`
- **Files Modified**: `statgpu/glm_core/_base.py`, `statgpu/solvers/_fista.py`, `_newton.py`, `_lbfgs.py`, `_admm.py`, `statgpu/__init__.py`
- **Tests**: 64 tests in `dev/tests/test_losses.py` (all passing)

### Added (2026-06-17)

- **P2 Module Expansion** (PR #72):
  - 5 modules upgraded: ANOVA (15%→60%), Covariance (30%→60%), Panel (45%→70%), Splines (35%→60%), Kernel Methods (60%→80%)
  - All new functions support numpy/cupy/torch three-backend computation
  - 17 new source files, 112 new tests (all passing)
  - External validation against scipy, sklearn, statsmodels (precision: coef diff ≤ 1e-14)

- **ANOVA**:
  - `f_twoway`: Two-way ANOVA with/without interaction term (Type I SS decomposition)
  - `f_welch`: Welch ANOVA for unequal variances (Welch 1951, Welch-Satterthwaite df)
  - `tukey_hsd`: Tukey HSD post-hoc test with studentized range distribution
  - `bonferroni`: Bonferroni-corrected pairwise t-tests (uses `statgpu.inference.adjust_pvalues`)
  - `cohens_f`: Cohen's f effect size (sqrt(eta²/(1-eta²)))
  - `partial_eta_squared`: Partial eta-squared from sum of squares
  - Files: `_twoway.py`, `_welch.py`, `_posthoc.py`, `_effect_size.py`

- **Covariance**:
  - `ShrunkCovariance`: Generic shrinkage estimator with user-specified intensity (matches sklearn)
  - `MinCovDet`: Robust Minimum Covariance Determinant (FAST-MCD, Rousseeuw & Van Driessen 1999)
    - Multi-stage algorithm: 30 random starts → top 10 → full C-steps
    - Consistency correction factor (Croux & Haesbroeck 1999)
    - Log-determinant for numerical stability
    - Matches sklearn MinCovDet with correlation = 1.000000
  - `GraphicalLasso`: Sparse inverse covariance via graphical lasso (Friedman et al. 2008)
  - `GraphicalLassoCV`: Cross-validated graphical lasso with log-likelihood scoring
  - Files: `_robust.py`, `_graphical_lasso.py`, `_shrinkage.py` (extended)

- **Panel**:
  - `PooledOLS`: Pooled OLS without demeaning (supports nonrobust/robust/clustered/HAC)
  - `BetweenOLS`: OLS on entity-level group means
  - `FirstDifferenceOLS`: OLS on first-differenced data (Δy_t = y_t - y_{t-1})
  - `FamaMacBeth`: Two-pass regression (cross-sectional OLS → time-series average with NW SE)
  - `hac_covariance`: Newey-West HAC estimator with Bartlett kernel (auto bandwidth, NW 1994 rule)
  - Files: `_pooled.py`, `_between.py`, `_first_diff.py`, `_fama_macbeth.py`, `_covariance.py` (extended)

- **Splines**:
  - `SplineTransformer`: sklearn-compatible fit/transform API (n_knots, degree, knots, extrapolation)
  - `cyclic_cubic_spline_basis`: Periodic cubic splines (null-space projection, 3 periodicity constraints)
  - `thin_plate_spline_basis`: Multi-dimensional smoothing splines (φ(r) = r²log(r) for d=1, m=2)
  - Files: `_transformer.py`, `_cyclic.py`, `_thin_plate.py`

- **Kernel Methods**:
  - `chi2_kernel`: Exponentiated chi-squared kernel (uses sklearn Cython for numpy backend)
  - `Nystroem`: Kernel approximation via random landmark sampling (SVD-based normalization, matches sklearn)
  - `KernelPCA`: Kernel PCA via eigendecomposition of centered kernel matrix
  - RBF kernel optimized: float32 chunked computation, 3.5-13x faster than sklearn on CPU
  - Files: `_nystroem.py`, `_kpca.py`, `_kernels.py` (extended + optimized)

### Optimized (2026-06-17)

- **RBF kernel numpy performance**:
  - Large matrices (n>2000) automatically use float32 (halves memory bandwidth)
  - Chunked computation for very large matrices (avoids OOM at n=50000)
  - All in-place operations on single buffer (peak memory = 1 n×m matrix)
  - Performance: n=5000 3.8x, n=10000 3.5x, n=50000 13.4x faster than sklearn

- **Nystroem GPU optimization**:
  - K_mm eigendecomposition moved to CPU (avoids GPU kernel launch overhead for small matrices)
  - Landmark normalization stored on CPU, converted to GPU only when needed
  - Matches sklearn output with correlation = 1.000000

- **Data consistency**:
  - GPU input → GPU output (no automatic numpy conversion)
  - Float64 input small matrices → float64 output
  - Float64 input large matrices → float32 output (avoids OOM)

### Validation (2026-06-17)

- **Three-backend benchmark** (Tesla P100-16GB, n=5000-100000):
  - LedoitWolf: torch 44.8x faster than sklearn at n=100000
  - Nystroem: cupy 43.7x faster than sklearn at n=100000
  - RBF Kernel: cupy 797x, torch 929x faster than sklearn at n=10000
  - ANOVA: torch 2.1x faster than scipy at n=100000
- **Precision**: All modules match external frameworks within 1e-14 (float64)
- **112 tests**: 5 test files covering all P2 modules, all passing
- **Benchmark JSON**: `results/p2_benchmark_final.json` (with GPU warmup)

### Code Review Rounds 9-10 (2026-06-15)

**Bug fixes:**
- Newton solver convergence check was 10,000x too strict (`_norm2_dev` returns L2 norm, not squared)
- `_resolve_loss_name` imported from wrong module — CV pipeline would crash with `ImportError`
- ElasticNet Lipschitz returned 0 for the `"en"` alias
- Debiased inference cleared `_resid`/`_X_design`/`_y`, breaking `rsquared`/`aic`/`bic`
- `fista_lla_path` ignored `sample_weight` in XtX fast paths (both GPU and numpy)
- Missing `xp_ones` import in `_fit_gpu_backend` — NameError for large-feature GPU fits

**Performance:**
- Deleted `_solver_utils.py` (442-line duplicate of solvers/ modules)
- IRLS: hoisted `_to_backend(y)` outside closure (was 30x/iter), reused `eta_raw` matmul
- Fused dispatch dict promoted to module-level constant
- `xp.sum(sw*ps)` → `xp.dot(sw,ps)` — avoids O(n) temporary allocation

**Refactoring:**
- Unified `_fit_gpu`/`_fit_torch` into single `_fit_gpu_backend` method (-468 lines)
- Extracted `_nesterov_momentum`/`_nesterov_update` helpers (12 sites across 6 files)
- Extracted gradient clipping constants to `solvers/_constants.py`
- Added type hints to all public solver function signatures
- Added `_call_with_weight` helper replacing 8 `try/except TypeError` blocks
- Removed duplicate entries in top-level `__init__.py`
- Replaced `SelectivePenalty` thread-local singleton with fresh-per-call instance
- Cached `_family_for_loss()` result

### Refactored (2026-06-14)

- **Top-level module reorganization (Phases 0-6)**:
  - Extracted `statgpu/solvers/` as a generic top-level module with 6 solvers (FISTA, FISTA-BB, FISTA-LLA, Newton, L-BFGS, ADMM). Solvers are now loss-agnostic — they work with any loss implementing the `GLMLoss` interface.
  - Extracted `statgpu/cross_validation/` with `CVEstimatorBase`, `kfold_indices`, `hash_cv_data`, `batch_mse`, `run_cv`. Shared by `linear_model` and `survival`.
  - Split `PenalizedGeneralizedLinearModel` (3968 lines) into mixin architecture: `_base.py` + `_fit_mixin.py` (2185 lines) + `_inference_mixin.py` (1174 lines) + `_predict_mixin.py` (215 lines).
  - Reorganized `linear_model/` into `wrappers/` (13 models), `penalized/` (mixin + 9 subclasses + CV), `cv/` (4 CV wrappers), `legacy/` (6 files).
  - Moved GLM-specific fused functions to `glm_core/_fused.py`.
  - Added optimization hint attributes to `GLMLoss` base class (`_lipschitz_safety`, `_momentum_beta_cap`, `_has_constant_hessian`, etc.) — solvers read these instead of hardcoding loss names.
  - Cleaned up 4 duplicate files in `nonparametric/` (old `_kde.py`, `_kernel_regression.py`, `_bandwidth_selection.py`, `_kernel_common.py`).
  - 62 safety net tests + remote GPU verification (Tesla P100): 51/51 precision benchmarks PASS.

- **New wrappers**:
  - `AdaptiveLasso` — adaptive L1 penalty (Zou 2006)
  - `SCADRegression` — SCAD penalty (Fan & Li 2001)
  - `MCPRegression` — MCP penalty (Zhang 2010)

- **Bug fix: adaptive_l1/scad GPU backend compatibility**:
  - `_irls_ridge_init_cd` now uses backend-agnostic `xp` operations instead of numpy-only code. Previously failed on CuPy/Torch with `TypeError`.
  - No CPU↔GPU transfers — computation stays on the original device.

- **Documentation**:
  - Fixed math formula display delimiters in 28 model docs (`\[ \]` → `$$ $$`).
  - Updated AGENTS.md with new module structure.
  - Added changelog writing conventions to AGENTS.md.

### Added (2026-06-13 ~ 2026-06-14)

> PR #55~#58 were split from the original PR #36 (GLM+Penalty full module). PR #36 delivered the complete GLM + Penalty system achieving 1043/1043 ALL PASS (100%) in full-matrix benchmark.

- **PR #36 — GLM+Penalty full module (original, split into PR-A~D)**:
  - 7 GLM families: `squared_error`, `logistic`, `poisson`, `gamma`, `inverse_gaussian`, `negative_binomial`, `tweedie`
  - 10 penalties: `none`, `l1`, `l2`, `elasticnet`, `scad`, `mcp`, `adaptive_l1`, `group_lasso`, `group_mcp`, `group_scad`
  - 6 solvers: `exact`, `newton`, `lbfgs`, `irls`, `fista`, `fista_bb` — dispatched per family+penalty combination
  - 3 backends: CPU (NumPy), CuPy, PyTorch — with auto device selection
  - Key technical features:
    - LLA routing for non-convex penalties (SCAD, MCP, group variants)
    - Augmented intercept handling for log-link GLMs (Poisson, gamma, etc.)
    - Iterate-dependent Lipschitz computation
    - Async FISTA for GLM+non-smooth penalties (2-5.5x speedup at n=5000)
    - L-BFGS fused penalty gradient fix — correctly converges to `loss_grad + α·coef = 0`
    - GPU sync batching optimizations for CuPy/Torch backends
    - Kernel fusion for GLM loss+gradient computation
  - Benchmark Results (v23c):
    | Section | Description | Tests | Status |
    |---------|-------------|-------|--------|
    | A | Cross-backend timing+precision | 816 | ALL PASS |
    | B | vs sklearn | 13 | ALL PASS |
    | D | vs statsmodels | 68 | ALL PASS |
    | E | Cross-solver consistency | 146 | ALL PASS |
    | **Total** | | **1043** | **ALL PASS** |
  - GPU Speedup (Section A):
    | Scale | CPU avg | Torch avg | Speedup |
    |-------|---------|-----------|---------|
    | n=500, p=50 | 953ms | 954ms | 1.00x |
    | n=2000, p=200 | 3995ms | 9108ms | 0.44x |
    | n=5000, p=500 | 2875ms | 1313ms | **2.19x** |
  - n=5000 solver-level: fista-Torch 2.56x, newton-Torch 2.10x, irls-Torch 2.40x
  - Files:
    - Core solver & GLM: `statgpu/glm_core/_solver.py`, `_negative_binomial.py`, `_irls.py`, `_gamma.py`, `_inverse_gaussian.py`, `_tweedie.py`
    - Penalized models: `statgpu/linear_model/_penalized.py`, `_gamma_glm.py`, `_inverse_gaussian_glm.py`, `_negative_binomial_glm.py`, `_tweedie_glm.py`
    - Penalties: `statgpu/penalties/_adaptive_l1.py`, `_mcp.py`, `_scad.py`, `_group_lasso.py`, `_group_mcp.py`, `_group_scad.py`
    - Backends: `statgpu/backends/_array_ops.py`, `_cupy.py`
    - Docs: changelog (EN+CN), benchmarks (EN+CN), model docs (GLM, Logistic, Poisson, Ridge; EN+CN), `dev/tests/_bench_v23c_report.md`
  - Full report: `dev/tests/_bench_v23c_report.md`

- **PR #55 — Core GLM solver, backends, penalties, inference (PR-A, from PR #36)**:
  - 7 GLM families: squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie
  - 10 penalties: none, l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
  - 6 solvers: irls, fista, fista_bb, admm, lbfgs, newton — dispatched per family+penalty combination
  - 3 backends: NumPy (CPU), CuPy (CUDA), PyTorch (CUDA) with auto device selection
  - Unified inference: 15 distributions, p-value adjustment, bootstrap, permutation test
  - Key technical features: LLA routing for non-convex penalties (SCAD/MCP), augmented intercept for log-link GLMs, iterate-dependent Lipschitz computation, kernel fusion for loss+gradient
  - Stability fixes:
    - Fixed 3 Critical NameErrors in CuPy paths and circular import issues
    - Fixed torch device mismatch for HC2/HC3 leverage computation
    - Fixed power-iteration seed for reproducible Lipschitz computation
    - Fixed CuPy cumop dtype kernels for empty inputs
    - Fixed KDE logpdf NameError and binomial IRLS deviance calculation
    - Restored irls_solver main loop after accidental deletion
  - Backend improvements:
    - Added GPU sync batching for solver operations (H6 fix)
    - Split solver into modular components (H4 fix)
    - Converted relative imports to absolute `statgpu.xx` imports
    - Added backend-aware gradient computation
  - Penalty fixes:
    - Added missing group_mcp/group_scad to non_smooth validation set
    - Updated derived attributes after group auto-fill
    - Fixed CompositePenalty backend handling
  - Testing:
    - Added regression tests for all fixes
    - Marked LassoCV tests as xfail (PR-B feature)

- **PR #56 — Penalized models + CV framework (PR-B, from PR #36)**:
  - 7 Penalized estimators: PenalizedLinearRegression, PenalizedLogisticRegression, PenalizedPoissonRegression, PenalizedGammaRegression, PenalizedInverseGaussianRegression, PenalizedNegativeBinomialRegression, PenalizedTweedieRegression
  - PenalizedGLM_CV: full CV over families x penalties x solvers
  - Lasso, Ridge, ElasticNet with full inference
  - LogisticRegression, LinearRegression with GPU
  - Stability fixes (8 rounds of code review):
    - Fixed P0/P1 bugs: NameError + TypeError in solver runtime
    - Fixed GPU/CPU prediction tolerance (relaxed then tightened to max_iter=2000 + tol=1e-10)
    - Unified NB tolerance across device paths
    - Fixed get_params, sample_weight, backend-aware issues
    - Consolidated hardcoded penalty/loss sets into shared constants
  - Code quality:
    - Extracted ~500 lines of dead code to legacy files
    - Removed magic numbers, added named constants
    - Deduplicated score/summary methods across estimators
    - Fixed BOM encoding issues and __all__ exports
    - Cleaned up imports and removed self-imports
  - Performance:
    - Added batched GPU syncs for penalty operations
    - Optimized penalty category detection
  - Testing:
    - Relaxed then tightened GPU/CPU prediction tolerance
    - Removed xfail markers after fixes

- **PR #57 — New modules (PR-C, from PR #36)**:
  - ANOVA: `f_oneway` — GPU-accelerated one-way ANOVA, float32/float64 support
  - Covariance: `EmpiricalCovariance`, `LedoitWolf`, `OAS` — covariance estimation with shrinkage
  - Panel Data: `PanelOLS` (one/two-way fixed effects), `RandomEffects` (Swamy-Arora), `PanelSummary`, clustered covariance
  - Splines: `bspline_basis`, `natural_cubic_spline_basis`, penalized regression with GCV
  - Semiparametric: `GAM` (penalized B-splines + GCV smoothing parameter selection)
  - Kernel Methods: `KernelRidge`, `KernelRidgeCV`, 6 kernel functions (rbf, polynomial, linear, laplacian, sigmoid, cosine)
  - Python compatibility:
    - Fixed `__future__` import ordering for Python 3.9 compatibility
    - Moved `__all__` after `__future__` in 4 files
    - Fixed covariance module exports
  - Runtime fixes:
    - Fixed RandomEffects group means calculation
    - Added missing NumpyBackend methods for new modules
    - Fixed panel test fit() argument order (y, X → X, y)
  - Code review fixes:
    - Fixed 8 Critical + 2 High issues in round 1
    - Fixed import conventions across all new modules
    - Fixed H2/M5/M6/L2 issues in subsequent rounds

- **PR #58 — Infrastructure, exports, backward compatibility (PR-D, from PR #36)**:
  - Unified `statgpu/__init__.py` exports (~60 public names)
  - `BaseEstimator` with device management and sklearn-compatible `get_params`/`set_params`
  - `Device` enum (CPU/CUDA/TORCH/AUTO) with auto-detection
  - Backward-compat shims for `kernel_methods/` and `splines/` old import paths
  - sklearn compatibility:
    - Fixed `get_params` to only return own `__init__` params (not parent class)
    - Preserved string identity for `simultaneous_method` and `cov_type` (sklearn clone() requirement)
  - CoxPH fixes:
    - Defined `n` before null model path in `_compute_partial_likelihood`
    - Added penalty warning for null model risk set
  - Code review:
    - Fixed `__all__` exports and import fallbacks
    - Fixed 6 remaining comment issues

- **PR #48 — Module reorganization**:
  - Moved kernel_methods/ and splines/ under nonparametric/ subpackage
  - Created kernel_smoothing/ subpackage for KDE + kernel regression
  - Extracted GAM to semiparametric/ package for future extensibility
  - Backward-compat shims for old import paths
  - IRLS solver improvements:
    - Fixed log-link intercept initialization (was using wrong starting values)
    - Added per-iteration convergence check (was only checking at end)
    - Hoisted `_dev_val` computation out of IRLS loop (performance)
  - CuPy fixes:
    - Fixed cummin/cummax exception handling for empty inputs
    - Fixed cumop dtype kernels for non-contiguous arrays
    - Wrapped CuPy arrays with `_to_numpy` in covariance tests
  - Code quality:
    - Stripped BOM from `_irls.py` encoding
    - Added `from __future__ import annotations` to `_lasso.py`
    - Narrowed bare `except Exception` clauses to specific exceptions
    - Fixed splines `__all__` exports
  - Security:
    - Removed hardcoded SSH credentials from remote config
  - Testing:
    - Added 6-stage real-data benchmark suite for RTX 4090
    - Added regression tests for all PR #47 code review fixes
  - Python 3.8 compatibility fixes

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
  - Comprehensive CHANGELOG with all PRs from #1 to #64

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
  - Merged PLAN_UNIFIED.md gates + PR #49 coding conventions into TO_DO.md
  - Unified CV framework:
    - Created `_cv_base.py` with shared `kfold_indices`, `CVCache`, `batch_mse`
    - Created `_cv_engine.py` with generic CV loop engine
    - Implemented `PenalizedGLM_CV` with full family × penalty × solver matrix
    - Added warm-start across alpha values (reuse model instance)
    - Added batch eigendecomposition for RidgeCV (avoids per-alpha solve)
  - CuPy fused kernel issue:
    - Discovered numerical issue with SCAD/MCP CuPy fused kernel
    - Disabled fused kernel for SCAD/MCP LLA path
    - Added diagnostic scripts and documentation
  - Panel fixes:
    - Fixed unbalanced two-way fixed effects
    - Fixed PanelOLS documentation
  - Ridge fixes:
    - Fixed weighted intercept calculation
    - Fixed ElasticNetCV warm-start with `fit_intercept=False`
  - Code quality:
    - Replaced duplicated `_kfold_indices` with shared imports
    - Fixed Lasso defaults and cache keys
    - Added inference guard for PenalizedGLM_CV scoring

### Added (2026-06-07 ~ 2026-06-09)

- **PR #50 — Add val_sample_weight to GLM sparse CV path**:
  - Validation sample weight support for sparse GLM cross-validation
  - Enables weighted CV folds for imbalanced datasets
  - Removed stray CuPy line
  - Used loss_fn.value for numpy path
  - Passed unaugmented Xv to _evaluate_loss_numpy for weighted scoring

- **PR #53 — Fix weighted Ridge inference**:
  - Correct scale calculation for weighted Ridge regression
  - Preserve bse/pvalues/conf_int with sample weights

- **PR #54 — Refactor CV dispatch table**:
  - Created dispatch table for _compute_cv_scores
  - Extracted _cv_fold_general for cleaner separation
  - Added path failure warnings and LLA cleanup
  - Fixed Tweedie per-sample loss sign error
  - Removed incorrect fallback weights
  - Removed dead code and self-import
  - Added fallback warning
  - Optimized Ridge CV scoring
  - Extracted hardcoded constants to module-level named variables
  - Added warnings for silent fallbacks
  - Fixed non-Gaussian MSE fallback
  - Raised clear error for non-uniform weights with non-L2 penalties
  - Added loss formula comments and narrowed exception catches
  - Added cv_splits parameter to PenalizedGLM_CV for custom fold generators
  - Parameterized NB alpha and Tweedie power from loss object defaults
  - Created unified loss formula registry (replaced inline if/elif chains)
  - Fixed LassoCV cache_key variable name after cache refactor
  - Fixed _res_logistic returns gradient (sigmoid(eta)-y) not loss
  - Fixed Poisson residual returns gradient, NB denominator, InvGauss clipping
  - Fixed weighted Lipschitz uses sum(w), cv_splits normalizes generator

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

### Added (2026-05-24 ~ 2026-05-29)

- **PR #37 — GLM penalty correctness + auto GPU routing**:
  - Fixed penalized GLM predict() to return inverse-link mean-scale predictions
  - Auto GPU routing for penalized models based on problem size
  - Fixed predict backend fallback when GPU backend unavailable
  - Enforced explicit GPU prediction backend contract
  - Handled GPU sample_weight conversion

- **PR #38 — Gamma inverse-power FISTA**:
  - Link-aware Gamma FISTA support across CPU/CuPy/Torch
  - Fixed objective mismatch for inverse-power link function
  - Fixed inverse-power Gamma FISTA init and torch dtype alignment
  - Used backend-native inverse-power FISTA warm start
  - Fixed inverse-power gamma FISTA init and clipping consistency
  - Fixed torch FISTA dtype for non-Gaussian intercept path
  - Fixed integer design dtype promotion across GLM intercept paths
  - Fixed CuPy FISTA init dtype

- **PR #39~#42 — GLM solver refactoring**:
  - Fixed GLM GPU dtype and review regressions
  - Refactored GLM solver backend helpers
  - IRLS solve backend aliases and compatibility
  - Tested IRLS solve backend aliases

- **PR #43, #44 — Linear inference result fixes**:
  - Refactored Gaussian linear inference helpers
  - Fixed CuPy inference critical value dtype
  - Added shared inference result containers
  - Completed linear inference result wiring
  - Fixed weighted penalized inference state
  - Cleared stale linear inference results
  - Fixed inference edge case cleanup
  - Cleared stale t-statistics for z results
  - Cleared unavailable GPU inference precompute cache
  - Used ridge sandwich covariance for penalties

- **PR #47 — CuPy cummin/cummax fix**:
  - Fixed CuPy cummin/cummax CUDA kernels on non-contiguous arrays
  - adjust_pvalues BH/BY/Hochberg now returns correct results (was 0% agreement with statsmodels)
  - Root cause: CUDA kernel reads sequential memory, but flip() returns negative-stride view
  - Fixed IRLS log-link intercept initialization
  - Added per-iteration convergence check
  - Added 6-stage real-data benchmark suite for RTX 4090
  - Removed hardcoded SSH creds + used backend utils in IRLS
  - Narrowed bare except clauses
  - Added regression tests for all code review fixes

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


### Added (2026-05-03 ~ 2026-05-11)

- **PR #27~#29 — Unsupervised learning Phase 3/3B/3C**:
  - Added 12 estimators: PCA, KMeans, DBSCAN, GaussianMixture, NMF, AgglomerativeClustering, UMAP, TSNE, MiniBatchKMeans, MiniBatchNMF, IncrementalPCA, TruncatedSVD
  - GPU exact paths for agglomerative clustering (single/complete/average/ward linkage)
  - Documentation and validation benchmarks for all estimators

- **PR #30, #32 — Agglomerative GPU exact paths**:
  - GPU-accelerated exact linkage for all distance metrics
  - Supports single, complete, average, ward linkage

- **PR #33 — Nonparametric module review**:
  - GPU memory fixes for KDE
  - Bandwidth selection GPU化
  - Log-sum-exp stabilization for numerical stability

- **PR #34, #35 — Documentation**:
  - Clarified runtime device selection
  - Explicit Torch backend docs
  - README installation and requirements updates

## 2026-04

### Added (2026-04-26)

- **PR #24 — Precision fixes, hochberg/stouffer, package restructure**:
  - Phase 1: Ordered Model Cross-Backend Precision Fixes
  - GPU acceleration with torch.compile and Triton kernels
  - Unified cross-package imports to absolute form (PEP 8)
  - Resolved 8 Codex review comments (shared_mem, lazy pandas, fit_intercept)
  - Added missing transpose to CuPy/Numpy backends
  - Fixed cv_results_ key naming
  - Preserved formula intercept semantics during fit

- **PR #26 — README refresh**:
  - Reorganized features, added models, recommended editable install
  - Exported combine_pvalues
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

- **PR #19 — Cython Efron optimization**:
  - Cython-optimized Efron gradient and Hessian computation
  - Comprehensive CoxPH accuracy and runtime benchmarks
  - Updated documentation for RidgeCV, LogisticRegressionCV and CoxPHCV
  - Fixed logistic cv duplicate batch log-loss helper names
  - Fixed cox cv cache key typing and CUDA kernel launch error surfacing
  - Aligned CoxPHCV status across docs
  - Updated RidgeCV and LogisticRegressionCV status to full implementation

- **PR #21 — Distribution backends unification**:
  - Consolidated `_distributions_gpu.py`, `_distributions_torch.py` into single `_distributions_backend.py`
  - 15 distributions across 3 backends via `SpecialFunctions` protocol and factory pattern
  - Fixed distribution backend routing and torch device propagation
  - Fixed proxy resolve args for rvs and two-sided critical
  - Streamlined proxy backend auto resolution args
  - Updated distribution API docs for unified 3-backend architecture

- **PR #22 — Backend utility consolidation**:
  - Consolidated duplicated backend utility functions
  - Cleaner backend abstraction layer

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

- **PR #18 — Remote config + backend enhancements**:
  - Removed hardcoded SSH credentials (security fix)
  - Added remote config module with env var support
  - Added Torch GPU backend support for knockoff filter
  - Added Elastic Net with optimized GPU implementations
  - Added LassoCV cross-validated Lasso implementation
  - Fixed review-thread issues in remote config, lasso/elasticnet cv
  - Fixed benchmark config error message env var name

- **PR #20 — CoxPHCV CuPy optimization**:
  - Optimized CoxPHCV CuPy Hessian path and defaults
  - Hardened coxphcv env parsing defaults cache key
  - Added cv tests for CoxPHCV
  - Clarified coxcv defaults and env fallback assertions
  - Updated Cox GPU entry+efron path and documented safe rollout
  - Synced Cox model docs for entry+efron GPU status

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

- **PR #16 — Torch backend support**:
  - Enhanced Ridge and CoxPH models with Torch support
  - Added memory management improvements
  - Fixed torch backend/device issues from review
  - Fixed reproducibility concerns
  - Avoided loop sync in Cox torch path
  - Tightened tolerance for validation

- **PR #17 — Elastic Net implementation**:
  - Added Elastic Net with optimized GPU implementations
  - Integrated optimized code into core implementation
  - Added Elastic Net documentation and changelog updates
  - Added benchmarks and test scripts
  - Removed hardcoded SSH credentials from large-scale benchmark runner
  - Tightened SSH auth logic for env-based remote benchmark runner
  - Allowed passphrase usage with discovered default SSH keys

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

### Added (2026-04-11 ~ 2026-04-15)

- **PR #10 — HAC covariance support**:
  - HAC covariance for LinearRegression and LogisticRegression
  - Newey-West bandwidth selection
  - Fixed penalized bread for Ridge inference
  - Added NotImplementedError in CV scaffolding for unsupported features
  - Clarified implemented vs interface-only scope for CV classes

- **PR #11 — Documentation for new models**:
  - Knockoff feature selection documentation
  - New model documentation

- **PR #12 — Distribution compatibility layer**:
  - Added compatibility layer for legacy distribution functions
  - Refactored inference methods for unified backend access
  - Fixed Lasso GPU sync overhead (removed unnecessary transfers)
  - Fixed distribution proxy resolve args for rvs and two-sided critical
  - Precomputed Lasso exclusion indices for performance
  - Clarified t-ppf bisection bounds in documentation

- **PR #13 — F-test p-value handling**:
  - Perfect fit F-test p-value handling (returns near-zero p-value)
  - Optimized Lasso p-value calculation for edge cases

- **PR #14 — Kernel regression + Lasso GPU optimization**:
  - Added kernel regression implementation with NumPy/CuPy support
  - Optimized Lasso GPU computation logic
  - Fixed F-statistic p-value for perfect fit cases
  - Reduced GPU index memory usage in nonparametric API
  - Addressed PR review: fixed nonparametric API naming

- **PR #15 — Lasso inference GPU support**:
  - Added debiased Lasso simultaneous inference with GPU nodewise bottleneck
  - Refined CN/EN model documentation structure and references
  - Fixed API naming, full-design cache keys
  - Removed redundant array casts
  - Avoided unnecessary copies in debiased matrix hashing paths

### Added (2026-04-03 ~ 2026-04-07)

- **PR #1 — CoxPH cluster-robust covariance**:
  - Added `cov_type="cluster"` for grouped sandwich covariance estimation
  - Breslow tie handling improvements
  - New benchmarking scripts for CoxPH

- **PR #2 — Runtime comparison tables**:
  - Reproducible runtime comparison tables across CPU/GPU and external frameworks
  - Added multi-target linear regression shape handling
  - Added multi-target sklearn and R benchmark scripts
  - Fixed Ridge.score host conversion for CUDA predictions
  - Optimized diagnostics and stepwise selection
  - Improved Cox inference paths
  - Fixed cache/convergence handling across models

- **PR #3 — Benchmark structure refactor**:
  - Refactored benchmark structure and updated documentation

- **PR #4 — Pluggable backends abstraction**:
  - Created BackendBase ABC with NumPy/CuPy/Torch implementations
  - Removed redundant model implementations (two LinearRegression classes, three Ridge variants)
  - Clean path for multi-backend support
  - Normalized codebase with backend abstraction layer

- **PR #5 — Ridge inference support**:
  - Full inference parity with LinearRegression
  - `cov_type`: nonrobust/hc0/hc1 (CPU + GPU)
  - `summary()`, `rsquared_adj`, `fvalue`, `f_pvalue`, `llf`, `aic`, `bic`

- **PR #6 — Logistic Regression evaluation metrics**:
  - Comprehensive evaluation metrics: ROC, AUC, confusion matrix
  - `evaluate_binary_classification` function
  - Fixed CuPy safety in logistic eval methods
  - Added finiteness checks for y_score validation
  - Aligned CuPy/Torch precision fallback with NumPy
  - Eliminated metrics duplication via delegation
  - Cached training evaluation metrics for reuse

- **PR #7, #8 — Bug fixes and experiment results**:
  - Various bug fixes
  - Updated experiment results

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
