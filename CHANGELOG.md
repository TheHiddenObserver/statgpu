# Changelog

## 2026-06-14

### Documentation & Infrastructure

- **PR #63**: Add dev/ workspace guide, testing guide, benchmark results, architecture overview
- **PR #62**: Reorganize dev/ folder — archive 241 old scripts, update remote config
- **PR #61**: Compress README GLM section + remove redundancy
- **PR #60**: Clean up README Implemented Methods with tables
- **PR #59**: Documentation, changelog, and guides (PR-E)
- Root files reorganized: USAGE.md → docs/index.md, AGENTS.md → dev/, plans → dev/plans/

## 2026-06-13

### Major Release: Module Reorganization + New Modules

- **PR #58**: Infrastructure, exports, backward compatibility (PR-D)
- **PR #57**: New modules — ANOVA, Covariance, Panel, Splines, Semiparametric, Kernel Methods (PR-C)
- **PR #56**: Penalized models + CV framework (PR-B)
- **PR #55**: Core GLM solver, backends, penalties, inference (PR-A)

### New Modules

- **ANOVA**: `f_oneway` — GPU-accelerated one-way ANOVA, supports float32/float64
- **Covariance**: `EmpiricalCovariance`, `LedoitWolf`, `OAS` — covariance estimation with shrinkage
- **Panel Data**: `PanelOLS`, `RandomEffects` — fixed/random effects with clustered SE
- **Splines**: `bspline_basis`, `natural_cubic_spline_basis` — B-spline basis construction
- **Semiparametric**: `GAM` — generalized additive models with GCV smoothing
- **Kernel Methods**: `KernelRidge`, `KernelRidgeCV` — kernel ridge regression

### Penalized GLM

- 7 Penalized estimators: `PenalizedLinearRegression`, `PenalizedLogisticRegression`, `PenalizedPoissonRegression`, `PenalizedGammaRegression`, `PenalizedInverseGaussianRegression`, `PenalizedNegativeBinomialRegression`, `PenalizedTweedieRegression`
- `PenalizedGLM_CV`: full CV over families × penalties × solvers
- 10 penalties: L1, L2, ElasticNet, SCAD, MCP, Adaptive L1, Group Lasso, Adaptive Group Lasso, Group SCAD, Group MCP
- 6 solvers: IRLS, FISTA, FISTA-BB, ADMM, L-BFGS, Newton

### Bug Fixes (PR #49, #48)

- 110+ bug fixes across 16 files
- 428 test cases added (all passing)
- Cross-backend precision < 0.02%
- Unified `best_score_` to negative MSE (sklearn convention)

## 2026-06-07 ~ 2026-06-09

- **PR #54**: Refactor dispatch table for CV scores
- **PR #53**: Fix weighted Ridge inference — correct scale, preserve bse/pvalues/conf_int
- **PR #50**: Add val_sample_weight to GLM sparse CV path

## 2026-05-29

- **PR #47**: Fix CuPy cummin/cummax contiguous array bug + Poisson IRLS precision

## 2026-05-27

- **PR #44**: Merge linear inference result fixes into GPU feature branch
- **PR #43**: Refactor linear inference result containers

## 2026-05-25 ~ 2026-05-26

- **PR #42**: Test IRLS solve backend aliases
- **PR #41**: Restore IRLS solve helper compatibility
- **PR #40**: Refactor GLM solver backend helpers
- **PR #39**: Fix GLM GPU dtype and review regressions

## 2026-05-24

- **PR #38**: Implement gamma inverse-power FISTA
- **PR #37**: Fix GLM penalty correctness and auto GPU routing

## 2026-05-15

- **PR #35**: Clarify runtime device selection and add explicit Torch backend docs
- **PR #34**: Clarify README installation and requirements

## 2026-05-10 ~ 2026-05-11

- **PR #33**: Nonparametric module review — GPU memory, bandwidth GPU化, log-sum-exp
- **PR #32**: Add agglomerative GPU exact paths

## GPU Performance (v22e ~ v23c)

### Async FISTA (v22e)

- Eliminated per-iteration GPU→CPU synchronization in FISTA loop
- Key results (n=5000, p=500):
  - logistic + L1: 2.22x → **5.41x**
  - logistic + ElasticNet: 2.18x → **5.17x**
  - Poisson + L1: 1.90x → **4.55x**
- Smaller scale (n=2000, p=200): logistic + Adaptive L1 now beats CPU (0.56x → **1.12x**)

### Precision Fixes (v22f)

- Fixed gradient clipping, Lipschitz recomputation, convergence check intervals

### L-BFGS Fix (v23c)

- **1043/1043 ALL PASS** — full matrix: 7 families × 13 penalties × 5 solvers × 3 backends

### Other Optimizations

- Kernel fusion + D2H batching (v20b)
- Torch CUDA warmup
- Small problem auto-dispatch (n×p < 200k → CPU)
- CuPy GPU-native cummin/cummax

## Earlier History

See `dev/plans/archive/PLAN_UNIFIED.md` and git log for details before 2026-05.
