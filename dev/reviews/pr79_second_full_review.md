# PR #79 Second Full-Repository Review and Auto-Fix

Date: 2026-07-12  
Branch: `agent/code-review-fixes`  
Mode: `code-review` auto-fix

## Impact classification

Active axes: public API, backend/dtype/device, solver, CV, inference, feature
selection, nonparametric kernels, resampling, performance, tests, and docs.
Formula behavior was inspected but not changed. Objective normalization was preserved;
no statistical definition was changed merely to match an external library.

## Parallel review tracks

1. Core estimator/backends and clone/device contracts.
2. Linear/penalized regression, inference summaries, solvers, and Cox survival.
3. Cross-validation, resampling, diagnostics, ANOVA, and penalties.
4. Feature selection and knockoff wrappers.
5. Nonparametric kernels/KDE and dtype/performance paths.
6. Unsupervised, covariance, panel, splines/GAM, tests, CI, and documentation.

## Fixed findings

- [HIGH][BUG][fixed] `feature_selection/_stepwise.py` — backward search never entered;
  final prediction could use a different feature order from fitting; invalid direction,
  null models, hard feature caps, repeated fit, and noisy output were inconsistent.
- [HIGH][BACKEND][fixed] `anova/_welch.py` — advertised Torch/CuPy computation converted
  complete groups to NumPy; Welch fractional denominator df and degenerate variance
  contracts were hardened.
- [HIGH][BUG][fixed] `cross_validation/_engine.py` — an alpha that failed some folds
  could still win by averaging only successful folds. Candidates now require every fold.
- [HIGH][BACKEND][fixed] `nonparametric/kernel_methods/_kernels.py` — Torch RBF crashed on
  `torch.maximum(tensor, scalar)` and large float64 NumPy inputs were silently downcast.
- [HIGH][SOLVER][fixed] `solvers/_fista_lla.py` — weighted squared-error SCAD/MCP was
  routed to a generic proximal-Newton path that could fail its first line search;
  quadratic losses now use weighted-centred FISTA-LLA and weighted Lipschitz constants.
- [HIGH][INFER][fixed] Gaussian regression summaries mishandled perfect fits,
  intercept-only models, zero residual variance, survival-function tails, and a legacy
  `conf_int` method/attribute collision.
- [MEDIUM][INFER][fixed] regression diagnostics now distinguish internal/external
  studentization, support rank-deficient leverage, and match statsmodels influence.
- [MEDIUM][API][fixed] estimator clone support, knockoff selector transform/repeated-fit,
  model-X draw validation, CompositePenalty construction, effect sizes, resampling
  integers/finiteness, and root exports were hardened.
- [MEDIUM][PERF][fixed] Cox score inference computed the zero-coefficient gradient/Hessian
  twice and used a bare exception; it now performs one call with explicit failure types.
- [LOW][READ/PERF][fixed] KDE zero-density log-sum calculations no longer emit expected
  runtime warnings; stale skeleton and support documentation was removed.
- [LOW][MAINT][fixed] `PenalizedLinearRegression` referenced `Penalty` in a public type
  annotation without defining it. A `TYPE_CHECKING` import now keeps runtime imports
  acyclic while satisfying static analysis and type-hint resolution.
- [LOW][TEST-COMPAT][fixed] the Welch reference regression originally used SciPy's newer
  `f_oneway(equal_var=False)` API, unavailable in the SciPy builds selected for Python
  3.9 and 3.10. It now uses the stable statsmodels Welch ANOVA reference and compares
  statistic, p-value, numerator df, and fractional denominator df.

## Validation evidence

- Focused review suite: `dev/tests/test_second_full_review.py` (44 review regressions).
- Broad CPU suites cover losses/penalties/solvers, inference/distributions, covariance,
  panel, splines/GAM, nonparametric methods, unsupervised methods, backend contracts,
  and repository review regressions.
- Analytic/reference checks include statsmodels Welch ANOVA and influence diagnostics,
  Gaussian closed-form/inference invariants, weighted-centering identities, backend
  parity, and source/dtype/device contracts.
- The focused suite is included in the permanent Python 3.9–3.12 regression matrix and
  the full Python 3.11 CPU suite.
- Permanent static gates now cover every source path changed in this review, including
  CV engine, diagnostics, feature selection, Gaussian summaries, penalized linear
  wrappers, penalties, FISTA-LLA, and Cox.

## Capability decisions

- Backend: touched numeric paths implement NumPy and Torch locally; CuPy code paths are
  retained and optional tests require CUDA hardware.
- CV: generic CV and existing tunable wrappers remain supported; incomplete candidates
  are rejected rather than partially scored.
- Inference: Gaussian and Cox inference remain supported; diagnostics are explicitly
  reporting-side CPU utilities.
- Formula: no formula-facing behavior changed in this pass.
- Benchmark: local micro/performance regressions were checked; physical CUDA benchmark
  and transfer profiling remain remote-pending.

## `dev/AGENTS.md` compliance

- Public API, backend, dtype/device, solver, CV, inference, formula, and benchmark axes
  were classified before changes.
- A dedicated regression suite accompanies the fixes and is part of permanent CI.
- Constructor parameters are not mutated during fit; repeated-fit and sklearn clone
  contracts are explicitly tested for the affected estimators.
- Backend-native paths do not introduce silent complete-array host transfers or silent
  float64-to-float32 downcasts.
- Public capability changes are reflected in English and Chinese documentation and all
  maintained changelogs.

## Deferred items

- Cox Torch Hessian still materializes an `O(n*p*p)` intermediate; changing it requires
  physical-GPU memory/profiling evidence and direct numerical-equivalence tests.
- Physical CuPy CUDA and Torch CUDA parity, convergence, output device/type, transfer,
  peak-memory, runtime, and repeated-fit cleanup remain unavailable on hosted CPU runners.

## Hard exit status

`PARTIAL_REMOTE_PENDING`: no unresolved local CRITICAL/HIGH finding remains after the
fix-and-retest loop. Only physical-GPU/performance evidence remains pending.
