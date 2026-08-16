# Changelog

All notable changes to statgpu are documented here, organized by release and date.

## 2026-08-09

### PR #126 — Panel Tier-1 Stage C covariance
- Added HC0/HC2/HC3, robust RandomEffects inference, cluster group debiasing, and Driscoll-Kraay covariance across NumPy, CuPy, and Torch.
- Preserved the existing HC1 (`robust`), Pooled row-HAC, default clustered covariance, coefficient-estimation, and panel fit-statistic contracts while fixing ordered-categorical chronology for Pooled legacy HAC.
- Hardened fixed-effect convergence and level prediction, Panel formula/refit/intercept semantics, rank-deficient inference, FirstDifference time validation, covariance numerical stability, and Fama-MacBeth chronology/formula/rank/inference behavior; all panel fits now fail closed transactionally on refit errors.
- Standardized Fama-MacBeth inference aliases/results without changing backend-native public arrays, replaced the retained full training design with a zero-length device anchor, and optimized GPU period solving without changing the shared SVD rank policy: Torch uses exact-size batched SVDs, while CuPy keeps supported 2-D SVDs but groups rows once and defers all rank scalars to one host transfer.
- Added maintained Python/R external-definition checks plus exact-source Tesla P100 validation. On clean numerical source `5fd9bcf214b1ba2abc5282c3922b6807fb6162ee`, the broad Stage-C, focused optimized Fama-MacBeth, and Pooled-HAC chronology gates all pass on CuPy and Torch. The retained scaling matrix shows the expected workload crossover: the 8K-row and 131K-row fixtures remain slower than NumPy, while the 524K-row / 16-regressor fixture reaches about 1.32x CuPy and 1.82x Torch speedup over the serial NumPy reference with resident GPU inputs. No universal GPU-speedup claim is made.

## 2026-08-08

### PR #122 — Panel Tier-1 diagnostics Stage B

- Added structured Panel `fit_statistics_` with parameter-based within/between/overall R², adjusted R², and classical model F statistics while preserving Stage-A inference and legacy df/R² attributes.
- Added classical pooling F, one-way entity Breusch-Pagan LM (including Baltagi-Li unbalanced panels), and one-way classical FE-vs-RE Hausman with explicit applicability diagnostics.
- Added NumPy/CuPy/Torch coverage, formula-row alignment, maintained Torch 2.0 CPU tests, executable linearmodels 7.0 definition alignment, and an exact-head physical GPU acceptance runner that rechecks coefficient inference as well as Stage-B diagnostics.

### PR #121 — CuPy inverse-quantile LUT correctness

- Fixed CuPy `betaincinv` and `gammaincinv` LUT cache tuple ordering so inverse quantiles no longer collapse to boundary values and downstream confidence intervals retain their correct width.
- Added maintained regression coverage for public CuPy Student-t, Beta, F, Gamma, and chi-square PPF/ISF paths, LUT/native-fallback boundaries, legacy inverse-quantile aliases, and Panel inference consumers.
- Validated the unchanged two-line numerical fix on Tesla P100 with Python 3.9.16 and CuPy 13.6.0: the original `t_{0.975,45}` failure now agrees with SciPy within `4.04e-09`, all expanded inverse-distribution checks pass, and the formerly zero-width Panel intervals match the reference.

## 2026-08-07

### PR #119 — Panel Tier-1 shared framework Stage A

- Added an internal `BasePanelModel`, shared panel index/balance metadata, and structured diagnostic/fit-stat result substrate for the later Tier-1 diagnostics stages without adding new public diagnostics.
- Centralized the existing residual-based panel OLS covariance dispatch while preserving each estimator's current nonrobust, HC1, clustered, and HAC normalization/df conventions; Fama-MacBeth keeps its distinct beta-series covariance.
- Migrated `PanelOLS`, `RandomEffects`, `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, and `FamaMacBeth` to the shared lifecycle where statistically valid, while preserving formula behavior, prediction/summary contracts, fixed-effect recovery, Swamy-Arora GLS, and backend-specific output semantics.
- Added pre-refactor golden regression coverage plus maintained Torch 2.0 CPU coverage for shared panel metadata/covariance/inference paths. Stage B diagnostics and Stage C covariance expansion remain pending under Issue #93.

### PR #116 — Torch LogisticRegressionCV strict-CUDA repair

- Fixed the mixed-precision Torch strict-CUDA `LogisticRegressionCV` failure by allocating batched IRLS parameters and ridge diagonals in the active CV working dtype and keeping candidate path outputs backend-native through validation scoring.
- Added regression coverage for float32/float64 CV, weighted and unweighted fitting, intercept/no-intercept paths, and the full CV selector, plus a Python 3.9 + Torch 2.0 CPU CI gate so optional-Torch coverage cannot silently skip.
- Validated the unchanged numerical implementation head `e6e4846b06604ed53e65fc9afd9054bd5777098f` on Tesla P100 with PyTorch 2.0.0+cu117/CUDA 11.7 and CuPy 13.6.0: all 18 statgpu canonical CV backend runs succeeded without CPU fallback, including `LogisticRegressionCV` on NumPy, CuPy, and Torch.
- Retained the historical pre-fix P100 failure source unchanged and registered the exact-head post-fix source under `results/pr116_p100/`; focused physical validation evidence is retained separately from dashboard timing data.

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
- Normalized warm starts for FISTA, Newton-family, L-BFGS-family, L-BFGS-B, and ADMM solvers to the preprocessed design backend, device, and dtype.
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
- Physical NVIDIA validation passed on the unchanged numerical implementation: RTX 4090 with PyTorch 2.8.0+cu128 passed the selected compile/CUDA Graph matrix 9/9 and runtime assertions; Tesla P100 with CuPy 13.6.0 passed the corresponding runtime assertions.
- The focused release PR changes version metadata and release-facing documentation only; all exact release-head hosted gates must pass before creating tag `v0.2.4`.

## Earlier history

Entries through 2026-07-27 are retained in
[`CHANGELOG-history-through-2026-07-27.md`](CHANGELOG-history-through-2026-07-27.md).