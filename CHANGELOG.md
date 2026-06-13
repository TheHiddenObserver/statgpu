# Changelog

## [Unreleased] - 2026-05-28

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
