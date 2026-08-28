# Changelog

All notable changes to statgpu are documented here, organized by release and date.

## Unreleased — 2026-08-28

### PR #129 / Issue #127 — Gaussian linear-model backend-native inference
- Migrated maintained Gaussian linear-model covariance, standard-error, test-statistic, p-value, and confidence-interval numerical work to the executed NumPy/CuPy/Torch backend while preserving the established final NumPy reporting snapshot.
- Routed normal/Student-t inference through the maintained shared reference-distribution layer, including stable df=1/df=2 extreme-tail handling, and made missing/invalid executed-backend provenance fail closed instead of silently falling back to NumPy.
- Preserved Ridge/L2 average-loss semantics and the `n_eff * alpha` inference mapping, including weighted fits and `RidgeCV` final-refit inference; added public `LinearRegression`, formula, robust/weighted, rank-deficient, multi-target, float32, external statsmodels, and no-host-transfer regression coverage.
- Added focused hosted CI and a maintained exact-SHA physical CUDA validator covering CuPy/Torch backend/device provenance, clean-tree checks, covariance/BSE/statistic/p-value/CI errors, weighted/rank/multi-target/small-df cases, and a `RidgeCV` final-refit case.
- Validation remains intentionally incomplete while PR #129 is draft: hosted PR gates and exact clean-head CuPy/Torch CUDA acceptance must pass before #127 can be marked `COMPLETE`. No GPU speedup claim is made.

## 0.2.5 — 2026-08-26

### PR #126, #128 — Panel Tier-1 Stage C covariance
- Released the Panel Tier-1 Stage C covariance and inference surface: HC0/HC2/HC3 and legacy HC1 where statistically defined, one-/two-way clustering with opt-in `group_debias`, Driscoll-Kraay with Bartlett/Parzen/QS kernels, RandomEffects robust/HC inference on quasi-demeaned GLS scores, and legacy row-order HAC with ordered-categorical chronology.
- Added transactional panel fits, row-preserving formula prediction, classical Hausman/pooling-F/Breusch-Pagan LM diagnostics, and overflow-safe diagnostics at extreme float64 scales.
- Fixed silent numerical corruption: CuPy `maximum.at`/`scatter_max` return `inf` for float64 magnitudes around 1e7..1e308 (group min/max scatter now uses a magnitude-gated host fallback), and Torch CUDA SVD now requires the exact `gesvd` driver (the default `gesvdj` leaks ~1e-16 into structurally-zero `U` entries).
- Strengthened the coefficient-resolution certificate to a deterministic error bound independent of LAPACK-version-specific SVD rounding; unresolved near-collinear designs fail closed instead of returning unreliable coefficients.
- Kept ordinary designs on vectorized GPU paths while reserving exact per-row accumulation for genuinely recoverable cancellation residuals; two-way clustered covariance at 10k rows dropped from ~1000 s to ~1.3 s on Tesla P100.
- Recorded exact-source Tesla P100 acceptance for all 12 physical runners on the validated numerical source `697de113`; artifacts under `results/pr126_release_697de113/`.

## 2026-08-09

### PR #126 — Panel Tier-1 Stage C covariance
- Fixed `FamaMacBeth` period eligibility so exactly identified full-rank cross sections (`n_t == k`, including the period intercept) are retained; rank-deficient square periods still fail closed.
- **Fama-MacBeth explicit-device backend authority**: explicit `cpu`/`cuda`/`torch` requests now override heterogeneous input container types; only `device="auto"` uses input-native backend dispatch, preventing silent execution on a backend different from the public request.
- **Fama-MacBeth / shared panel numerical stability**: NumPy/CuPy/Torch now share the conservative Gram-certificate dispatch; non-finite Gram batches are masked before eigenspectrum evaluation, and non-finite Gram/RHS/solutions fall through to the maintained SVD rank policy. Shared SVD least-squares applies inverse singular values to $U^T$ before the raw-response reduction, uses a uniform safe working scale for collectively subnormal full-rank designs, and preserves the existing relative rank cutoff. Fama-MacBeth coefficient averages, fixed-effect group means, and parameter-R² scalar/group means now share a magnitude-tiered float64 reduction path for dynamic-range/cancellation risk while ordinary panel columns retain the existing fast scatter reduction. Sum and mean reductions apply range scaling only inside the active magnitude tier; safe mean tiers are summed before division, preserving collectively representable subnormal contributions beside huge cancelling values. Two-way demeaning refreshes its packed stability classification after the first projection so projection-created dynamic range enters the stable path before alternating dimensions; the maintained Torch CPU and physical CuPy/Torch validators include a fixture whose risk classification is safe before the entity projection and risky immediately afterwards. Coefficient-series covariance continues to use per-coordinate scales with symmetric large-scale-first restoration. Positive inference variances are no longer replaced by an absolute floor, while exact-zero variance maps zero coefficients to statistic 0 and nonzero coefficients to signed infinity instead of using a fake tiny denominator.
- **Panel covariance extreme-scale arithmetic**: residual-based covariance now keeps tiny-design/projection restore factors outside observation-level cancellation. Working projection coordinates are scaled only when projection×residual would overflow; residual vectors are never globally magnitude-normalized. Cluster and Driscoll-Kraay grouping occurs before selective Gram scaling, preserving representable small groups/periods beside huge cancelling observations, while already-safe subnormal-design HC/cluster paths retain their previous precision. Two-way clustering combines grouped components before restoration and detects nested dimensions by partition equivalence rather than arbitrary code numbering. When a multi-tier nonnested case also fails the cross-group row-reduction safety certificate, grouped row outer products remain separate terms, float group-debias corrections are decomposed into exact power-of-two factors, and two compensated error streams preserve magnitude-tier/CGM/debias cancellation; Gram-safe multi-tier cases retain the vectorized component-matrix path. Covariance symmetrization and inclusion-exclusion remain range-aware, while HAC/Driscoll-Kraay protect both pre-Gram products and the complete lag-sequence accumulator from avoidable overflow. Two-way multi-tier covariance also fails closed if a nonzero grouped component survives the common score scaling but a mathematically nonzero self/cross product would underflow on that common Gram scale; this prevents a representable low-order covariance remainder from being silently published as zero when the required estimator-level cancellation exceeds the supported float64 common-scale range. Public clustered, two-way clustered, HAC, and Driscoll-Kraay helpers reject non-finite `X`/residual inputs before signed/group reductions, preventing NaN/Inf scores from being silently reinterpreted as zero contributions. The physical Stage-C validator exercises this fail-closed contract together with pre-Gram cancellation, tiny-design grouping, mixed-dynamic-range cluster/DK, nested-code permutation, and the earlier extreme-scale cases on CuPy and Torch CUDA. Delayed covariance, projection, and design scales use mantissa/exponent scale restoration so compensating large/small factors cannot create a transient overflow or underflow before the final representable float64 covariance is rounded.
- **Panel fixed-effect prediction extreme-scale correctness**: one-way entity/time effect maps recover stable group means from uncentered level residuals and are stored directly on the level-residual scale, avoiding both observation-level centering that can erase a recoverable tail and a centered compact effect whose subtraction from a finite grand mean can itself overflow. Two-way additive-effect recovery uses the same level-scale map representation, certifies fast-path convergence with the stable reducer when alternating projections create new dynamic range, and defers its common-shift normalization until convergence; the final observation-weighted shift uses the stable reducer directly rather than multiplying huge effects by group counts. `_grand_mean` remains available for the established diagnostic contract but is no longer required to reconstruct a stored fitted effect in `PanelOLS.predict()`. This keeps known-label level predictions consistent with the hardened within transformation across NumPy, CuPy, and Torch. Two-way additive-effect recovery conditionally uses a range-minimizing common-shift gauge near the float64 boundary so sparse incidence graphs cannot drive an otherwise representable entity/time decomposition out of range; ordinary-scale recovery preserves the historical weighted-zero gauge.
- **Panel diagnostics extreme-scale correctness**: classical model F, pooling F, Breusch-Pagan LM, adjusted/legacy fit-statistic reductions, and estimator-side RSS/TSS reporting now share overflow-safe centering plus subnormal-safe backend normalization. Scale-invariant statistics are computed before restoring squared units, so representable finite results are not converted to false exact fits or `Inf/Inf`/underflow artifacts; the physical Stage-C runner now exercises these branches on both CuPy and Torch CUDA. Constant-only restricted diagnostics and parameter/adjusted R-squared now use range-safe response centering only when the physical value-minus-mean difference would exceed float64 range, keeping scale-invariant statistics finite without perturbing ordinary-scale subtraction.
- Corrected PanelOLS public residual degrees of freedom to count the full identified fixed-effect nuisance rank consistently across nonrobust scale, HC1 correction, Student-t inference, diagnostics, and explicit-dummy references; rank-deficient HC2/HC3 fits now preserve fitted values and fail closed only at coefficient inference when unit leverage makes the coordinate covariance undefined.
- Added HC0/HC2/HC3, robust RandomEffects inference, cluster group debiasing, and Driscoll-Kraay covariance across NumPy, CuPy, and Torch; RandomEffects now also fails closed when the Swamy-Arora between auxiliary regression has no positive residual degrees of freedom.
- Preserved the existing HC1 (`robust`), Pooled row-HAC, default clustered covariance, coefficient-estimation, and panel fit-statistic contracts while fixing ordered-categorical chronology for Pooled legacy HAC.
- Hardened fixed-effect convergence and level prediction, Panel formula/refit/intercept/prediction semantics (including row-preserving prediction, fail-closed pipe metadata conflicts, and RandomEffects token rejection), rank-deficient inference and Hausman applicability, FirstDifference time validation, covariance numerical stability (including fail-closed single-cluster inference), and Fama-MacBeth chronology/formula/rank/inference behavior; all panel fits now fail closed transactionally on refit errors.
- Standardized Fama-MacBeth inference aliases/results without changing backend-native public arrays, moved clearly well-conditioned NumPy/CuPy/Torch period solves to a shared conservative Gram-certified exact-size batch with the original SVD rank policy as fail-closed fallback, removed duplicate direct-fit finite scanning, and retained backend-native distribution inference plus a single reporting snapshot.
- Historical exact-source Tesla P100 acceptance on numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` recorded that Stage-C passed 35 cases and 12 public primitives per GPU backend, HAC chronology/Student-t checks passed, and Fama-MacBeth resident-array scaling beat NumPy on all maintained workloads (CuPy/Torch ratios micro 0.549/0.343, medium 0.204/0.168, large 0.114/0.109) with one Gram-certified batch, one control synchronization, and zero SVD fallbacks in every measured GPU scale.
- The subsequent review-fix loops changed valid Fama-MacBeth and shared panel least-squares paths, so the `8c60db00...` P100 artifacts are historical-only for the current branch; fresh exact-head CuPy/Torch CUDA acceptance is required before merge readiness can be promoted.
- Added maintained Python/R external-definition checks and preserved the historical physical artifacts under `results/pr126_p100_fama_fix/`; legacy Gaussian linear-model inference backend migration is tracked separately in #127.
- **Two-way clustered covariance performance**: the exact per-row dyadic two-sum fallback (which ordinary balanced panels above ~6.5k rows always triggered, ~1000s per CuPy fit at 10k rows) is now gated by a residual-acceptance check that keeps ordinary designs on the vectorized Gram path and reserves the exact path for genuinely recoverable cancellation residuals; 10k-row CuPy fits drop from ~1018s to ~1.3s and the Stage-C performance benchmark completes instead of timing out.
- **Numerical hardening**: CuPy `maximum.at`/`scatter_max` return `inf` for float64 magnitudes around 1e7..1e308, so group min/max scatter now uses the sequential host scatter; Torch CUDA SVD requires the exact `gesvd` driver and fails closed with RuntimeError when it is unavailable (the default `gesvdj` driver leaks ~1e-16 into structurally-zero `U` entries that huge responses amplify); failed panel fits retain the executed backend provenance; Student-t(1) p-values use the well-conditioned `2 atan(1/x)/pi` form so extreme statistics keep their representable tail; formula side-array alignment fails closed on over-long inputs.
- **CuPy device affinity**: backend availability probes no longer switch the current CUDA device, and panel allocations (scatter targets, dummies, row weights) bind to the reference device; a physical device-affinity gate covers both CuPy and Torch CUDA.

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
- The final exact-head physical-GPU promotion artifact is published at https://gist.github.com/TheHiddenObserver/afdcad86a243e68a918d852b92e984a4. It records schema 3, 134/134 passing checks, zero child and nested return codes, empty gate-failure arrays, clean source state before and after execution, and SHA-256 `bd4058450def691dd29e9d78853534016c6da70c33192a97dc312d95cbe5d76d`.
- Added release-package validation that checks version consistency, builds the pure-Python wheel and sdist, runs `twine check`, validates artifact contents, clean-installs the sdist on Ubuntu, and clean-installs the same wheel on Ubuntu, Windows, and macOS.

### Packaging and release publication
- Bumped the package version to `0.2.3` in `pyproject.toml` and `statgpu/__init__.py`.
- The official wheel remains a universal `py3-none-any` artifact built with `STATGPU_NO_EXT=1`; optional Cython sources remain available in the sdist.
- Added the authoritative GitHub Release document at `.github/releases/v0.2.3.md`, a release-note completeness gate, and tag automation that publishes that file as the GitHub Release body after the PyPI job succeeds.

## Earlier history

Entries through 2026-07-27 are retained in
[`CHANGELOG-history-through-2026-07-27.md`](CHANGELOG-history-through-2026-07-27.md).
