# Changelog

## [Unreleased] - 2026-06-14

### GPU Performance: Async FISTA (v22e)

- **Major speedup**: Eliminated per-iteration GPU→CPU synchronization in FISTA loop.
- Activates for all non-smooth penalties (L1, ElasticNet, SCAD, MCP, Adaptive L1, Group) on GPU.
- **Key results** (n=5000, p=500, vs CPU):
  - logistic + L1: 2.22x → **5.41x** (2.45x improvement)
  - logistic + ElasticNet: 2.18x → **5.17x** (2.50x improvement)
  - Poisson + L1: 1.90x → **4.55x**
  - Gamma + L1: 1.95x → **4.32x**
- **Smaller scale** (n=2000, p=200): logistic + Adaptive L1 now beats CPU (0.56x → **1.12x**).
- Optimizations: fixed step + on-device gradient clipping + deferred convergence checks (every 5 iters).

### Precision Fixes (v22f)

- Fixed `_clip_grad_on_device` missing `max(gmax, 1e4)` floor for torch/cupy (caused logistic+none divergence).
- Lipschitz recomputation interval 50→10 (reduces trajectory divergence).
- Convergence check interval 5→3 (reduces overshoot).
- Smart pass criterion: relax objective tolerance to 1e-3 when both CPU/GPU hit max_iter.

### L-BFGS Fused Penalty Gradient Fix (v23c)

- **1043/1043 ALL PASS (100%)** — full matrix: 7 families × 13 penalties × 5 solvers × 3 backends.
- Added `_smooth_penalty_gradient(penalty, params)` to lbfgs fused path gradient.
- Without it, lbfgs converged to `loss_grad=0` instead of `loss_grad + alpha*coef = 0`.

### Other Optimizations

- Kernel fusion + D2H batching (v20b): reduced kernel launch overhead.
- Torch CUDA warmup: first-use matmul to avoid lazy initialization.
- Small problem auto-dispatch: n×p < 200k → CPU (avoid GPU overhead).
- CuPy GPU-native cummin/cummax via custom CUDA RawKernel.

## [Previous] - 2026-05-28

### Module Reorganization

- **Nonparametric subpackages**: `kernel_methods/` and `splines/` moved under `statgpu.nonparametric/`. KDE files moved to `nonparametric/kernel_smoothing/`. Old import paths preserved via backward-compat shims.
- **Semiparametric package**: GAM extracted from `splines/` into new top-level `statgpu.semiparametric/` package for future extensibility.
- New paths: `statgpu.nonparametric.kernel_smoothing`, `statgpu.nonparametric.kernel_methods`, `statgpu.nonparametric.splines`, `statgpu.semiparametric`
- Old paths (`statgpu.kernel_methods`, `statgpu.splines`, `statgpu.nonparametric._kde`) still work via shims.

### New Modules

- **ANOVA**: `statgpu.anova.f_oneway` — one-way ANOVA, drop-in replacement for `scipy.stats.f_oneway`. Supports numpy/cupy/torch backends.
- **Covariance**: `statgpu.covariance.EmpiricalCovariance`, `LedoitWolf`, `OAS` — covariance estimation with shrinkage. Equivalent to `sklearn.covariance`.
- **Kernel Methods**: `statgpu.nonparametric.kernel_methods.KernelRidge`, `KernelRidgeCV` — kernel ridge regression with 6 kernel functions. Equivalent to `sklearn.kernel_ridge`.
- **Panel Data**: `statgpu.panel.PanelOLS`, `RandomEffects` — fixed effects and random effects panel data models with clustered standard errors. Equivalent to `linearmodels.panel`.
- **Splines/GAM**: `statgpu.nonparametric.splines.bspline_basis`, `natural_cubic_spline_basis` + `statgpu.semiparametric.GAM` — B-spline basis construction and generalized additive models with GCV smoothing parameter selection.

### GPU Optimizations

- Splines De Boor recursion: vectorized inner loop eliminates ~180 GPU syncs per call, 3x speedup on GPU.
- KRR CV: fully vectorized GPU path for eigendecomposition + batched alpha sweep (eliminates Python loop over 100 alphas).
- Panel: extracted shared `xp_cholesky_solve` utility to backends, eliminating duplicated torch/numpy branching.
- Panel: extracted shared `ols_inference_nonrobust` to `_utils.py`.
- Removed duplicated `_to_numpy()` definitions from panel modules (3 files), using shared `statgpu.backends._to_numpy`.
- Cantor-pair hash in `two_way_clustered_covariance` now computed on-GPU instead of round-tripping through CPU.

### Bug Fixes

- RidgeCV alpha scaling: CV-selected alpha now correctly divided by `n_samples` before passing to V9 Ridge wrapper.
- PenalizedGLM predict: fixed device round-trip issue.
- ~100+ torch-cuda device awareness fixes across 22+ files.

### Documentation

- Added complete model documentation (14 sections each) for all 5 new modules in `docs/en/models/` and `docs/models/`.
- Updated `docs/en/models/README.md` and `docs/models/README.md` model indexes.
- Added `EmpiricalCovariance` Attributes docstring.

### Benchmark Results (Tesla P100-SXM2-16GB)

- 38/38 ALL PASS (cross-backend + sklearn/scipy comparisons)
- Cross-backend accuracy: 37/38 OK (< 1e-10), 1/38 FAIR (natural spline n=5000, ~1.5e-6)
- External baselines: scipy f_oneway (< 1e-15), sklearn covariance (< 1e-15), sklearn RidgeCV (< 1e-4), sklearn Lasso (< 1e-6), scipy KDE (< 1e-13)
