# Changelog

All notable changes to statgpu are documented here, organized by release and date.

## 0.2.4 — 2026-08-06

### Logistic regression and GLM correctness

- Corrected arbitrary-link Binomial IRLS Fisher weights, working responses, line-search objectives, backend-native warm starts, and quadratic-penalty validation.
- Hardened direct `LogisticRegression` response/control validation, transactional refits, convergence reporting, integer prediction dtype, single-column response handling, and finite decision thresholds.
- Unified fitted logistic likelihood diagnostics across NumPy, CuPy, and Torch with the registered numerically stable `LogisticLoss` objective; likelihood, AIC, BIC, pseudo-R², and convergence remain independent of covariance inference.
- Kept confusion-matrix and hard classification metrics available for one-class targets while preserving explicit class-support requirements for ROC-AUC and average precision.
- Kept CuPy/Torch analytic weights device-native and corrected weighted IRLS curvature, likelihood, dispersion, and sandwich-inference semantics.
- Standardized GLM analytic-weight behavior across ridge scaling, line search, pseudo-loglikelihood, information criteria, dispersion, and covariance; global weight rescaling leaves estimates and diagnostics unchanged.
- Added backend-native response-domain, finite-value, real-valued, shape, and length validation for scalar GLMs, including penalized and cross-validated entry points.
- Aligned formula sample weights only after Patsy missing-row filtering and corrected weighted Gaussian FISTA centering.

### Cross-validation, inference, and estimator contracts

- Made `RidgeCV`, `ElasticNetCV`, and `LogisticRegressionCV` fits failure-safe: stale state is cleared before fitting and selected parameters are published only after the final full-data refit succeeds.
- Preserved explicit Torch/CuPy requests and pinned `device="auto"` final refits to the backend selected during cross-validation.
- Updated Logistic and Elastic Net default regularization grids to incorporate analytic weights and satisfy integer-weight row-replication equivalence.
- Preserved declared validation losses and analytic weights in penalized CV; programming, shape, CUDA OOM, and device errors are no longer converted into candidate `NaN` values or unrelated MSE fallback.
- Completed the standalone `ElasticNet` and final-refit `ElasticNetCV` inference contract across NumPy, CuPy, and Torch.
- Corrected public ElasticNet/Ridge scaling documentation: under the shared average-loss convention, `ElasticNet(alpha, l1_ratio=0)` matches `Ridge(alpha)`.
- Made public estimator finite-input guards, cloning, sklearn tags, nested `set_params`, and fitted-state invalidation transactional, including legacy scikit-learn clone identity checks.

### Solver and backend safety

- Corrected the executable loss/penalty/solver matrix so Newton, L-BFGS, and L-BFGS-B reject unsupported non-smooth penalties rather than optimizing only the smooth component.
- Removed the incorrect Euclidean-prox Newton shortcut. Smooth L2/no-penalty objectives retain Newton updates; non-smooth proximal-Newton requests delegate visibly to backend-native FISTA until a Hessian-metric proximal solver exists.
- Narrowed Armijo, linear-solve, alpha-grid, and inference fallbacks to recognized numeric or rank failures; CUDA OOM, device, index, contract, and unrelated runtime failures propagate.
- Normalized warm starts for FISTA, Newton-family, L-BFGS-family, and ADMM solvers to the preprocessed design backend, device, and dtype.
- Completed ADMM's legitimate Cholesky fallback and hardened L-BFGS-B feasible directions, backend-native bounds, and NaN-bound validation.
- Added centralized, observable Torch compilation policy: eager remains the default for unset, `auto`, and `disable`; `default` and `reduce-overhead` are explicit opt-ins, and only the known CUDA Graph output-lifecycle failure becomes a permanent eager fallback.
- Removed the package-initialization cycle between `statgpu.glm_core` and the Cox loss export by lazily exposing `CoxPartialLikelihoodLoss`; fresh-interpreter imports no longer depend on importing `LogisticRegression` first.

### Documentation, testing, and release preparation

- Reconciled the English and Chinese LogisticRegression, ElasticNet, cross-validation, solver-algorithm, and solver/penalty documentation with the maintained implementation.
- Removed unsupported universal GPU speedup, backend-threshold, and coefficient-tolerance claims; performance guidance now requires workload-specific benchmarking.
- Documented ownership boundaries between maintained pytest coverage and manual physical-GPU diagnostics.
- Bumped package metadata to `0.2.4` and added the authoritative GitHub Release document at `.github/releases/v0.2.4.md`.

### Validation

- The final PR #87 implementation head passed the complete CPU suite with 2239 passed and 719 skipped, static and documentation contracts, Python 3.9–3.12 regression jobs, scikit-learn 1.2.2/1.3.2/latest compatibility, and release-package validation.
- Physical NVIDIA validation passed on the unchanged numerical implementation: RTX 4090 with PyTorch 2.8.0+cu128 passed the selected compile/CUDA Graph matrix 9/9 and runtime assertions; Tesla P100 with CuPy 13.6.0 passed the corresponding runtime assertions.
- The focused release PR changes version metadata and release-facing documentation only; all exact release-head hosted gates must pass before creating tag `v0.2.4`.

## 0.2.3 — 2026-08-04

### Added
- Completed CoxPH Phase 1 with Breslow, Efron, and Exact ties; delayed-entry and `(start, stop]` counting-process data; shared-coefficient stratification; subject identifiers; and `Surv(start, stop, event)` formula input.
- Added shared NumPy, CuPy, and Torch-CUDA risk-set primitives for Cox objectives, gradients, information matrices, and baseline estimation, including backend-native dynamic programming for Exact ties.
- Extended `CoxPHCV` held-out partial likelihood to all supported tie methods, delayed entry, start-stop rows, strata, and subject-grouped folds.

### Changed
- Hardened Cox inference, numerical stability, formula NA alignment, singular-information handling, CV cache identity, fold eligibility, selected-penalty refitting, and failed-fit state resets.
- Hardened L1, L2, Elastic Net, SCAD, and MCP penalized Cox estimation; removed the unidentified intercept; corrected Cox-specific warm starts; and made Torch Efron value, gradient, and Hessian paths native.
- Standardized public Group Lasso and Adaptive Group Lasso behavior through the generic loss-gradient and exact group-proximal path across supported backends.
- Made requested CoxPHCV two-stage and successive-halving controls execute one explicit exhaustive full-precision candidate pass, avoiding repeated complete-grid fitting while preserving deterministic selection semantics.
- Made one-shot `CoxPHCV.cv_splits` iterators reusable across repeated fit, scikit-learn clone, parameter reconstruction, and pickle.

### Validation
- Hosted workflow #960 passed on the final reviewed head `f05a44ad363b46612e956e137e2f00d040765acb`: documentation, static, full CPU, and Python 3.9–3.12 regression jobs all passed; the complete CPU suite reported 1881 passed and 662 skipped.
- The final exact-head physical-GPU promotion artifact is published at https://gist.github.com/TheHiddenObserver/afdcad86a243e68a918d852b92e984a4. It records schema 3, 134/134 passing checks, zero child and nested return codes, empty gate-failure arrays, clean source state before and after execution, and SHA-256 `bd4058450def691dd29e9d78853534016c6da70c33192a97dc312d95cbe5d76d`.
- Added release-package validation that checks version consistency, builds the pure-Python wheel and sdist, runs `twine check`, validates artifact contents, clean-installs the sdist on Ubuntu, and clean-installs the same wheel on Ubuntu, Windows, and macOS.

### Packaging and release publication
- Bumped the package version to `0.2.3` in `pyproject.toml` and `statgpu/__init__.py`.
- The official wheel remains a universal `py3-none-any` artifact built with `STATGPU_NO_EXT=1`; optional Cython sources remain available in the sdist.
- Added the authoritative GitHub Release document at `.github/releases/v0.2.3.md`, a release-note completeness gate, and tag automation that publishes that file as the GitHub Release body after the PyPI job succeeds.

## Earlier history

Entries through 2026-07-27 are retained in
[`CHANGELOG-history-through-2026-07-27.md`](CHANGELOG-history-through-2026-07-27.md).
