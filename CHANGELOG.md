# Changelog

All notable changes to statgpu are documented here, organized by date and PR.

## 2026-07-14

### PR #79 — Third review/fix cycle

- Fixed Torch vector Cholesky solves, Panel string-label/device paths, KernelPCA/RidgeCV/
  thin-plate Torch failures, and full-design CPU fallbacks in panel array workflows.
- Added shared finite-input validation for panel, covariance, unsupervised, KernelPCA,
  Nystroem, and thin-plate paths plus 21 focused regressions.

## 2026-07-12

### PR #79 — Second full-repository review and auto-fix

- Fixed Stepwise backward selection/order/state contracts, backend-native Welch ANOVA,
  incomplete-fold CV selection, regression diagnostics, summary-statistic edge cases,
  Torch RBF kernels, weighted quadratic SCAD/MCP routing, resampling validation, and
  Cox score-test duplication.
- Hardened estimator cloning, knockoff selectors/draw validation, composite penalties,
  effect sizes, backend factory semantics, KDE zero-density handling, and dtype/device
  preservation; added 40+ focused regression tests and synchronized public docs.

### PR #79 — Native three-backend execution follow-up

- Removed complete numeric-array NumPy fallbacks from `GraphicalLasso`,
  `GraphicalLassoCV`, `MinCovDet`, `SplineTransformer`, and `FamaMacBeth`.
- Kept Graphical Lasso block-coordinate descent/CV, FAST-MCD C-steps and
  reweighting, spline Cox–de Boor recurrence, and Fama–MacBeth regressions/HAC
  covariance on the selected NumPy, CuPy, or Torch backend.
- Kept Tukey/Bonferroni group reductions on-device; only scalar distribution
  CDF/quantile evaluations cross the CPU boundary.
- Added NumPy/Torch parity and backend-preservation tests plus optional CuPy CUDA
  checks. Physical CuPy/Torch CUDA memory, runtime, convergence, and repeated-fit
  validation remains `PARTIAL_REMOTE_PENDING`.
- Synchronized README, bilingual implemented-method lists, model pages, and all
  three changelogs with the corrected execution and validation boundaries.

### PR #79 — Public module statistical-contract follow-up

- Extended the repository review beyond Ridge to every top-level public module family,
  combining full-package high-signal static analysis with targeted numerical invariants,
  nested-model checks, and parity comparisons against established reference libraries.
- Corrected two-way ANOVA residual and balance semantics, Welch/post-hoc degenerate cases,
  chi-square kernels, KernelRidge/KernelRidgeCV scoring, KernelPCA embedding consistency,
  and Nystroem normalization for indefinite kernels.
- Corrected empirical precision estimation, Graphical Lasso block-coordinate updates,
  MinCovDet centered semantics, panel cluster/HAC contracts, Patsy side-array alignment,
  and rank-deficient panel regression fallbacks.
- Implemented real spline extrapolation modes; hardened B-spline, KDE, kernel regression,
  GAM, and binary-metric input contracts.
- Added three focused regression suites and expanded the permanent Python 3.9–3.12,
  full-CPU, static-contract, compilation, and complete-collection gates.
- Validation remains `PARTIAL_REMOTE_PENDING`: all hosted CPU/static gates pass, while
  physical CuPy/Torch CUDA numerical, memory, type/device, and performance validation is
  still required for affected GPU paths.

### PR #79 — Ridge objective and weighted-path consistency follow-up

- Confirmed that statgpu Ridge uses the package-wide average-loss objective rather
  than scikit-learn's unnormalized residual-sum-of-squares convention.
- Preserved the exact normal equations `X'X + n*alpha*I` for unweighted fits and
  `X'WX + sum(w)*alpha*I` for weighted fits; scikit-learn comparisons now use the
  explicit corresponding alpha mapping.
- Unified weighted Ridge behavior across the optimized wrapper, generic exact solver,
  FISTA, formula fitting, CPU/CuPy/Torch exact paths, Gaussian inference, RidgeCV,
  and `PenalizedGLM_CV(loss="squared_error", penalty="l2")`.
- Corrected weighted centering before square-root weighting, weighted intercept and
  residual construction for inference, and weighted default alpha-grid generation.
- `PenalizedGLM_CV` now generates weighted alpha grids from the normalized weighted
  null gradient and avoids building an unused host-side Gram cache for the default
  GPU Newton Ridge route.
- Formula evaluation now exposes retained row positions so sample weights remain
  aligned when Patsy drops rows containing missing values.
- GPU sample-weight validation and normalization use device-side reductions and
  synchronize only scalar results, avoiding full weight-vector host transfers.
- Added regression coverage for weighted closed forms, weight-rescaling invariance,
  exact/FISTA and wrapper/generic equality, formula missing rows, inference covariance,
  both Ridge CV implementations, and weighted scikit-learn alpha mapping.

## 2026-07-11

### PR #79 — Full repository review and hardening

- Completed an iterative repository-wide review covering correctness, backend routing,
  statistical/API contracts, readability, maintainability, extensibility, performance
  risks, test quality, and compliance with `dev/AGENTS.md`.
- Fixed backend/device validation, sklearn-style estimator parameters, Torch inference
  routing, UMAP fuzzy-union and random-state semantics, NNDescent correctness, adaptive
  L1 and knockoff runtime errors, CV input contracts, KMeans/UMAP edge cases, and Cox
  Efron observed-information orientation.
- Hardened tests so optional Torch/CuPy dependencies skip explicitly instead of failing
  collection or swallowing unexpected errors; moved the remote GPU runner outside the
  pytest test tree.
- Added focused review regression suites and permanent Python 3.9–3.12, full CPU,
  compilation, static-contract, and complete test-collection CI gates.
- Added `dev/reviews/pr79_full_repository_review.md` with accepted fixes, deferred
  architectural debt, and the physical-GPU validation plan.
- Validation status: `PARTIAL_REMOTE_PENDING`; CPU and contract gates pass, while
  physical CuPy/Torch CUDA numerical, memory, and performance validation remains required.

## 2026-07-08

### v0.2.1 — Packaging / PyPI release hygiene

- **Version bump** 0.2.0 → 0.2.1 (`pyproject.toml`, `statgpu/__init__.py`).
- **Pure-Python wheel policy**: the PyPI release workflow now sets `STATGPU_NO_EXT=1`,
  so the published wheel is tagged `py3-none-any` and installs on every OS / Python
  version. Previously `python -m build` compiled the optional Cython extensions during
  `bdist_wheel`, producing a platform-locked wheel (e.g. `cp311-linux_x86_64`) that
  served almost no one and forced everyone else onto the sdist.
- **setup.py**: added the `STATGPU_NO_EXT` switch — when set to `1`, `ext_modules` is
  empty (forces a pure-Python build). The Cython extensions remain optional CPU
  accelerators with pure-Python fallbacks; users who want the C speedups build them
  from the sdist, which still ships the `.pyx`/`.pxd` sources via `MANIFEST.in`.
- **publish.yml**: added `twine check dist/*` before upload.

### PR #74 — Ordered Newton-Raphson + Analytical Hessian Inference + Unified Sandwich Engine
- Ordered Logit/Probit: L-BFGS replaced with Newton-Raphson + trust-region (3-backend)
- Ordered inference: analytical Hessian, SE/z/p/CI, loglikelihood/aic/bic (CPU+GPU)
- Sandwich engine: m_estimation_inference, fisher_information, penalty curvature API
- Penalized inference: sandwich (L2/EN), oracle active-set (SCAD/MCP)
- QuantileRegression standalone class with kernel+bootstrap inference
- 28 bug fixes across 4 code review rounds; scipy→get_distribution; GPU guards
- Docs: ordered.md rewrite, v0.2.1 coverage matrix, solver-algorithms/quantile/robust
- Validated: R ordinal::clm, three-backend GPU (CuPy+Torch), 226 CI tests
