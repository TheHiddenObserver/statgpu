# Changelog

> Language: English<br>
> Last updated: 2026-08-28<br>
> This page: Changelog<br>
> Switch: [Chinese](../cn/changelog.md)

## Unreleased — Gaussian backend-native inference (PR #129 / Issue #127)

### Changed

- Maintained Gaussian linear-model numerical covariance, standard errors, statistics, p-values, and confidence intervals now execute on the actual NumPy/CuPy/Torch fit backend; established reporting attributes/results may still take a final NumPy snapshot after numerical inference completes.
- Normal and Student-t inference routes through the maintained reference-distribution layer, including stable df=1/df=2 extreme-tail handling. Missing or invalid executed-backend provenance fails closed instead of silently choosing NumPy.
- Ridge/L2 inference preserves the existing average-loss convention and `n_eff * alpha` normal-equation mapping, including weighted fits and `RidgeCV` final-refit inference.

### Validation

- Added public `LinearRegression`, formula, weighted/robust, rank-deficient, multi-target, float32, statsmodels-alignment, no-host-transfer, non-L2 delegation, and Ridge/RidgeCV regression coverage plus a focused hosted CI workflow.
- Added a maintained exact-SHA physical CUDA validator for CuPy and Torch with clean-tree proof, requested/executed backend and concrete-device provenance, covariance/BSE/statistic/p-value/CI error reporting, weighted/rank/multi-target/small-df cases, and `RidgeCV` final-refit inference.
- PR #129 remains draft until hosted gates and exact clean-head CuPy/Torch CUDA acceptance pass. No GPU speedup claim is made, and #127 is not yet `COMPLETE`.

## 0.2.5 — 2026-08-26 (released)

### Added

- **Panel Tier-1 Stage C covariance and inference**: HC0/HC2/HC3 and legacy HC1 (`robust`) covariance for the transform-based panel estimators; one-way and two-way clustered covariance with opt-in `group_debias=True`; Driscoll-Kraay covariance with Bartlett, Parzen, and Quadratic-Spectral kernels; robust/HC inference for `RandomEffects` on quasi-demeaned GLS scores; legacy row-order HAC for `PooledOLS` with ordered-categorical chronology support.
- **Diagnostics**: classical Hausman FE-vs-RE, pooling F, Breusch-Pagan LM, within/between/overall/adjusted R-squared, and model F — overflow-safe at extreme float64 scales on NumPy/CuPy/Torch.
- **Transactional panel fits** with row-preserving formula prediction and fail-closed refit semantics.

### Fixed

- CuPy `maximum.at`/`scatter_max` return `inf` for float64 magnitudes around 1e7..1e308; group min/max scatter now uses a magnitude-gated host fallback (`<= 1e6` keeps the native GPU scatter) and is exact in both paths.
- Torch CUDA SVD now requires the exact `gesvd` driver and fails closed when it is unavailable; the default `gesvdj` driver leaks ~1e-16 into structurally-zero `U` entries that huge responses amplify into wrong coefficients.
- The panel coefficient-resolution certificate was made deterministic (independent of LAPACK-version-specific SVD rounding); unresolved near-collinear full-rank designs fail closed instead of returning unreliable slopes, while single-column fixed-effect-absorbed designs and rank-deficient designs report their actual rank.
- Formula side-array alignment fails closed unless a side array matches either the original formula-data row count or the retained row count.
- Failed panel fits retain the executed backend provenance while clearing fitted/inference state.

### Optimized

- Two-way clustered covariance at 10k rows dropped from ~1000 s to ~1.3 s per CuPy fit on Tesla P100 after an over-broad row-expansion fallback was replaced by a residual-acceptance check that keeps ordinary balanced panels on the vectorized Gram path.
- Fama-MacBeth resident-array scaling (P100, this release's artifact): CuPy/Torch GPU-over-NumPy median-time ratios **1.314/0.706** micro, **0.174/0.126** medium, **0.092/0.084** large — Torch faster than NumPy at every scale (1.4×/7.9×/11.9×), CuPy crossing over from the medium workload onward, every measured case in one `gram-certified` batch with zero SVD fallbacks.

### Validation

- Exact-source Tesla P100 acceptance on the validated numerical source `697de113`: all 12 physical runners passed (Stage-C 35 cases + 12 primitives per backend; Fama-MacBeth oracle + provenance; HAC chronology; extreme t(2) tail; device affinity; scaling; RHS cancellation ×2; rank precedence ×2; intercept cancellation ×2). Artifacts: `results/pr126_release_697de113/`. The release head itself is the PR #128 merge commit onto `master`; `697de113` is the immutable numerical source those artifacts validated.
- TestPyPI rehearsal: pure-Python wheel installed in fresh environments from `test.pypi.org` with import and CPU fit/predict smoke tests passing.

## 2026-08-09 — Panel Stage C covariance completion (PR #126)

Stage C completes the Panel Tier-1 covariance and inference surface while preserving estimator coefficients and standardized fit-statistic definitions. `robust` remains the existing HC1 contract; `hc0`, `hc2`, and `hc3` use each estimator's actual transformed fit space. `RandomEffects` supports robust/HC, clustered, and Driscoll-Kraay covariance on quasi-demeaned GLS scores, and fails closed if the Swamy-Arora between auxiliary regression has no positive residual degrees of freedom. One-/two-way clustering supports opt-in `group_debias=True`; clustered inference now also fails closed when any supplied clustering dimension has fewer than two distinct groups, instead of returning the degenerate one-cluster sandwich. `PooledOLS(cov_type="hac")` remains the legacy row-order Bartlett/Newey-West path.

The implementation also hardens two-way fixed-effect convergence and prediction. Pipe-named Panel formula metadata is now authoritative: conflicting explicit entity/time IDs fail closed, missing named pipe columns cannot be replaced by unrelated explicit IDs, RandomEffects accepts a second pipe time variable only when Driscoll-Kraay covariance actually consumes it, and fixed-effect magic tokens are rejected for RandomEffects rather than being reinterpreted as grouping metadata. Entity/time projection metadata is factorized once and reused on the selected backend; convergence checks residual means for both effect dimensions, uses a scale-aware roundoff floor for numerically absorbed directions, and exposes fail-closed `demean_max_iter`/`demean_tol` controls for weakly connected panels. Two-way fixed effects are recovered jointly for unbalanced prediction, while known entity/time labels from different disconnected incidence components are rejected as unidentified. Formula prediction now fails closed if Patsy would drop input rows or if a formula transformation creates non-finite design values. Prediction also no longer guesses that every one-column-short matrix omitted an intercept, and an explicitly fitted non-unit constant is restored by value and position only on the compatible path. Known fixed-effect labels now restore the centered level grand mean so `PanelOLS.predict()` returns the complete fixed-effect level projection; formula-enabled effects no longer leak across refits, formulas with more than two fixed-effect variables fail closed, and no-FE `PanelOLS` formulas preserve Patsy/R default-intercept semantics (with `0 +` / `-1` retaining the explicit no-intercept path); no-intercept `rsquared_within` now uses the standard uncentered total sum of squares. The same review also strengthens rank-deficient coefficient identifiability and makes classical Hausman fail closed whenever either fitted coefficient vector is non-unique, plus `FirstDifferenceOLS` duplicate/time semantics, HC2/HC3 leverage stability, metadata alignment, CuPy scatter-add, RandomEffects formula intercept/name behavior, and quadratic-spectral weights. External definitions are checked against pinned `statsmodels==0.14.6`, `linearmodels==7.0`, R `plm==2.6-7`, and R `sandwich==3.1-3` references.

The latest numerical hardening unifies NumPy/CuPy/Torch Fama-MacBeth period dispatch under the same conservative Gram certificate and maintained SVD fallback. The certificate rejects non-finite Gram/RHS/solutions before they can preempt fallback; shared SVD least-squares uses inverse-singular-value factor ordering plus a safe uniform working scale for collectively subnormal full-rank designs. Fama-MacBeth coefficient averages and shared parameter-R² means use reduction-length scaling only when overflow is possible, avoiding extra magnitude-normalization loss without claiming arbitrary compensated cancellation recovery; coefficient-series covariance uses per-coordinate scales with symmetric restoration. Shared panel inference no longer imposes an absolute variance floor: exact-zero variance maps a zero coefficient to statistic 0 and a nonzero coefficient to signed infinity. Classical model F, pooling F, and Breusch-Pagan LM use overflow-safe centering and subnormal-safe backend normalization. Residual-based covariance now delays tiny-design/projection scale restoration until after cancellation: projection coordinates are scaled only when projection×residual would overflow, cluster and DK scores are grouped before selective Gram scaling, and residual vectors are never globally magnitude-normalized. This preserves small representable components beside huge cancelling observations without degrading already-safe subnormal-design precision. Two-way clustering recognizes nested partitions independently of arbitrary code numbering and cancels identical marginal/intersection components before restoration. Range-aware symmetrization/inclusion-exclusion plus HAC/DK pre-Gram and full-lag accumulators then prevent avoidable intermediate overflow. The maintained physical Stage-C runner includes diagnostic-scale, zero-variance, pre-Gram, tiny-design, mixed-range, nested-partition, covariance extreme-scale, and lag-accumulation branches for both CuPy and Torch CUDA.


Physical validation is recorded as an exact-source evidence chain. Historical Stage-C and Fama-MacBeth artifacts remain valid only for their original numerical heads. The previously accepted P100 source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` is now historical for the current PR branch because later review-fix loops changed valid Fama-MacBeth and shared panel least-squares paths; fresh exact-head CuPy/Torch CUDA acceptance is required before merge readiness can be promoted. On Tesla P100, the broad Stage-C runner passed 35/35 estimator/covariance cases and 12/12 public primitives on each of CuPy and Torch; the dedicated HAC chronology runner passed ordered-categorical/numeric equivalence, the lexical negative control, and shared backend-native Student-t inference. Fama-MacBeth now uses a conservative Gram-spectrum certificate for exact-size GPU batches: only clearly well-conditioned periods may consume the batched Gram solve, while every uncertified period remains owned by the original SVD rank policy. The accepted scaling artifact reports CuPy/Torch GPU-over-NumPy median-time ratios of **0.549/0.343** on micro (64×128×4), **0.204/0.168** on medium (128×1,024×8), and **0.114/0.109** on large (128×4,096×16), corresponding to about 1.82×/2.92×, 4.91×/5.97×, and 8.75×/9.16× speedups. Every measured GPU scale used one `gram-certified` batch, one control synchronization, and zero SVD fallbacks; input transfer is excluded from this resident-array timing protocol, so these results are workload/hardware-specific evidence rather than a universal GPU guarantee. The focused Fama-MacBeth gate also verifies chronology/formula/rank/inference behavior, backend-native public arrays, backend-native distribution inference, and prediction/device provenance. All four final physical runners are preserved under `results/pr126_p100_fama_fix/` and point to the same numerical source.

### Validation (2026-08-22)

A fresh review-fix loop on the PR branch hardened the remaining numerical and device paths and re-ran the full physical matrix at exact head `5068da3f`:

- **Two-way clustered covariance performance**: the exact per-row dyadic two-sum fallback (always triggered by ordinary balanced panels above ~6.5k rows, ~1000s per CuPy fit at 10k rows) is now gated by a residual-acceptance check. Ordinary designs stay on the vectorized Gram path; only genuinely recoverable cancellation residuals fall back to the exact row products. On Tesla P100, `pooled_cluster_two_way` 10k-row CuPy fits drop from **~1018s to ~1.3s** (Torch ~0.2s; 100k rows ~0.4s), and `benchmark_panel_stage_c_covariance.py` completes its 60-row matrix in ~40s instead of timing out.
- **Numerical hardening**: CuPy `maximum.at`/`cupyx.scatter_max` return `inf` for float64 magnitudes around 1e7..1e308 (observed CuPy 13.6), so group min/max scatter now uses the sequential host scatter; Torch CUDA SVD uses the exact `gesvd` driver (the default `gesvdj` leaks ~1e-16 into structurally-zero `U` entries that huge responses amplify); failed panel fits retain the executed backend provenance; Student-t(1) p-values use the well-conditioned `2 atan(1/x)/pi` form so extreme statistics (e.g. |t|=1e154) keep their representable tail instead of the subtractive survival collapse; formula side-array alignment fails closed on over-long inputs.
- **CuPy device affinity**: backend availability probes no longer switch the current CUDA device, and panel allocations (scatter targets, dummy matrices, row weights, SVD identities) bind to the reference device; a physical device-affinity gate (`validate_panel_cupy_device_affinity_gpu.py`) covers both CuPy and Torch CUDA.
- All 12 physical runners pass at the exact head on Tesla P100 (CuPy 13.6.0 / Torch 2.0.0+cu117): Stage-C correctness (35 cases + 12 primitives per backend), focused Fama-MacBeth oracle + certified-Gram provenance, HAC chronology, extreme t(2) tail, device affinity, Fama-MacBeth scaling, RHS cancellation, rank precedence, and intercept cancellation. Artifacts: `results/pr126_perf_fix_528d967e/`, `results/pr126_review_fix_da3604ee/`.


### PR #122 — Panel Tier-1 diagnostics Stage B

- Added public structured `PanelTestResult` and `PanelFitStatistics` outputs plus standardized `fit_statistics_` on the maintained panel estimators. The new fit statistics use parameter-based within/between/overall R², an explicitly defined adjusted R², and a classical homoskedastic model F statistic where the estimator has a residual-OLS fit space.
- Kept Stage-A coefficient inference and legacy R²/df behavior unchanged. In particular, `PanelOLS` continues to expose its historical public residual df and BSE/t/p/CI, while Stage-B diagnostics use a separate standard fixed-effect nuisance-rank df; the classical Hausman calculation consumes only a diagnostic small covariance rescaled to that standard denominator.
- Added the classical pooling F test for fixed effects, the one-way entity error-components Breusch-Pagan LM test including the Baltagi-Li unbalanced-panel formula, and the classical one-way entity FE-vs-RE Hausman test. Inapplicable econometric cases return structured reasons; singular positive-semidefinite Hausman covariance differences use a documented generalized-inverse/rank extension, while materially indefinite differences are rejected.
- Added optional `entity_ids` to `PooledOLS.fit()` and `FamaMacBeth.fit()` solely for Stage-B within/between fit statistics and the panel BP-LM path. Pooled HAC sorting now carries entity diagnostic metadata through the same stable permutation as X/y. Formula missing-row filtering aligns observation-level side arrays before diagnostics are formed.
- Added analytic/fitted regressions, maintained Python 3.9 + Torch 2.0 CPU parity, and an executable `linearmodels==7.0` definition-alignment job. FirstDifference external comparison is restricted to panels where both implementations use the same transformed sample; Stage B does not silently redefine the Stage-A adjacent-observed-row differencing contract for internal time gaps.
- Added `dev/benchmarks/validate_panel_stage_b_gpu.py` as the exact-head physical correctness/provenance gate. The previously accepted P100 artifacts at numerical implementation `a57efcea29b0e87ecb89865c5a6902d5773812c6` remain immutable historical evidence: CuPy and Torch each passed all 17 estimator cases with requested-backend provenance and no fallback, while the focused disconnected two-way FE artifact validated the df=1 inference boundary to machine precision. The four Hausman parameterizations per backend in that run were all correctly structured `applicable=false` cases, so they validate applicability/reason parity but do not physically exercise an applicable Hausman statistic/p-value/df path.
- The reopened physical gate is now closed on exact clean measurement head `2701aa9feb3796c33c94e6480fcb78c80c6a809c`: Tesla P100 CuPy and Torch each passed all 17 estimator cases and all five Hausman diagnostics with requested/executed backend identity and no CPU fallback. The dedicated 48-observation nonzero-effect fixture is `applicable=true` on both backends with df=1; its Hausman statistic differs from NumPy by at most `1.10e-13` and p-value by at most `2.19e-14`. The promoted 44-row canonical validation source preserves statistic/pvalue/df for that branch, while the older 42-row a57efcea source remains historical audit evidence. No timing or speedup claim is made.

Related: Issue #93 and pull request #122.

### PR #121 — CuPy inverse-quantile LUT correctness

- Corrected the CuPy LUT cache tuple order used by `betaincinv()` and `gammaincinv()`. The LUT builders already stored `(x_grid, y_grid)`, but the cached values were unpacked in reverse, so inverse lookup searched the wrong axis and could collapse quantiles to clipped boundary values.
- The original failure was exposed by Panel Stage A physical validation: on Tesla P100, the CuPy `t.isf(0.025, 45)` path produced an almost-zero critical value and zero-width confidence intervals. The corrected two-line numerical implementation now returns `2.014103388876289`, within `4.04e-09` absolute error of the SciPy reference.
- Added maintained regression coverage for raw inverse-beta/inverse-gamma cache reuse; public CuPy Student-t, Beta, F, Gamma, and chi-square PPF/ISF plus round trips; Student-t LUT/native-fallback boundaries at df 1, 10, 45, 60, and 80; module-level distribution proxies; legacy inverse-quantile aliases; and representative Panel inference consumers.
- Expanded physical validation on exact numerical head `f768b312d05f47debdb8fa13ae4da09b27d00239` used Tesla P100, Python 3.9.16, and CuPy 13.6.0 with a clean working tree. Maximum PPF/ISF absolute errors were `4.04e-09` for Student-t, `5.49e-13`/`1.68e-13` for Beta, `4.51e-12` for F, `1.45e-08` for Gamma, and `2.90e-08` for chi-square, all well inside the maintained inverse-quantile accuracy contract.
- The reconstructed Panel CI and actual shared Panel inference consumer now match their references within `3.42e-10` and `1.96e-10`, respectively; the formerly zero-width intervals are non-degenerate and agree with the Torch/reference result.

Related: Issue #120 and pull request #121.

## 2026-08-07

### PR #119 — Panel Tier-1 shared framework Stage A

- Added an internal `BasePanelModel`, `PanelIndexInfo`, `PanelTestResult`, and `PanelFitStatistics` substrate for Issue #93. Stage A establishes shared lifecycle and panel-structure contracts only; Hausman/pooling-F/Breusch-Pagan LM tests and expanded fit statistics remain Stage B work.
- Centralized dispatch for the covariance definitions that already existed in the panel estimators. Nonrobust scaling, HC1 corrections, one-/two-way clustering, HAC behavior, rank/df conventions, and unsupported-name errors remain estimator-equivalent to the pre-refactor implementation. No HC0/HC2/HC3 or Driscoll-Kraay support is added in this stage.
- Migrated `PanelOLS`, `RandomEffects`, `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, and `FamaMacBeth` to statistically neutral shared lifecycle helpers where valid. Fixed-effect recovery/prediction, Swamy-Arora variance components and quasi-demeaning, and Fama-MacBeth beta-series covariance remain model-specific.
- Preserved the existing formula, missing-row alignment, intercept/effect-token behavior, prediction output contracts, summary schemas/printing behavior, balanced/unbalanced semantics, residual-df definitions, and strict explicit-device/no-fallback rules.
- Added a pre-refactor golden suite before any panel source migration and kept it active after the refactor. Dedicated Python 3.9 + Torch 2.0 CPU CI now also executes the shared panel metadata/covariance/inference regression tests so optional Torch coverage cannot silently skip.

Stage B diagnostics and Stage C covariance expansion remain pending under Issue #93; this Stage-A refactor does not advertise them as public capabilities.

### PR #116 — Torch LogisticRegressionCV strict-CUDA repair

- Fixed the maintained Torch strict-CUDA `LogisticRegressionCV` failure in the batched GPU IRLS path. Mixed-precision CV now allocates parameters and ridge diagonals in the active working dtype, and coefficient/intercept paths remain backend-native through validation scoring.
- Added regression coverage for float32 and float64 CV, weighted/unweighted execution, intercept/no-intercept paths, and the full selector consumer. A dedicated Python 3.9 + Torch 2.0 CPU CI job prevents the optional Torch regression suite from silently skipping.
- Physical validation ran on exact numerical implementation head `e6e4846b06604ed53e65fc9afd9054bd5777098f` using Tesla P100-SXM2-16GB, Python 3.9.16, PyTorch 2.0.0+cu117 / CUDA 11.7, and CuPy 13.6.0. All four focused Torch CUDA cases selected the same `C=0.2` as the CPU reference; the largest mean-loss difference was below `6.2e-8`, with the float64 path agreeing to machine precision.
- The canonical six-family rerun recorded all 18 statgpu NumPy/CuPy/Torch backend rows as successful, with zero failed candidates/folds and converged final refits. `LogisticRegressionCV` selected `C=0.1` on NumPy, CuPy, Torch, and sklearn; the Torch/NumPy validation-loss difference was below `4.7e-8`.
- The historical pre-fix P100 source remains immutable and registered. The post-fix exact-head source is registered separately from `results/pr116_p100/cv_benchmark_pr116_p100.json`; `focused_validation.json` remains validation-only evidence rather than dashboard timing data.

Related: Issue #112 and pull request #116.

## 0.2.4 — 2026-08-06

### Logistic regression and GLM correctness

- Corrected arbitrary-link Binomial IRLS Fisher weights, working responses, line-search objectives, backend-native warm starts, and quadratic-penalty validation.
- Hardened direct `LogisticRegression` validation, transactional refits, convergence reporting, integer hard predictions, single-column response handling, and finite decision thresholds.
- Unified fitted logistic likelihood diagnostics across NumPy, CuPy, and Torch with the registered stable `LogisticLoss` objective. Likelihood, AIC, BIC, pseudo-R², and convergence remain available independently of covariance inference.
- Kept confusion-matrix metrics available for one-class targets while retaining explicit class-support errors for ROC-AUC and average precision.
- Kept analytic weights device-native on CuPy/Torch fits and corrected weighted IRLS curvature, likelihood, dispersion, and sandwich-inference semantics.
- Standardized GLM analytic-weight behavior across fitting, line search, diagnostics, and covariance. Globally rescaling analytic weights does not change fitted parameters or reported diagnostics.
- Added backend-native response-domain, real-valued, finite, shape, and length validation for scalar GLMs, including penalized and CV entry points.
- Aligned formula sample weights after Patsy row filtering and corrected weighted Gaussian FISTA centering.

### Cross-validation, inference, and estimator contracts

- Made `RidgeCV`, `ElasticNetCV`, and `LogisticRegressionCV` failure-safe: stale state is cleared before fitting and selected parameters are published only after the final full-data refit succeeds.
- Preserved explicit Torch/CuPy requests and pinned `device="auto"` final refits to the backend selected during cross-validation.
- Updated Logistic and Elastic Net default regularization grids to incorporate analytic weights and satisfy integer-weight row-replication equivalence.
- Preserved declared validation losses and analytic weights in penalized CV; programming, shape, CUDA OOM, and device errors are no longer converted into candidate `NaN` values or unrelated MSE fallback.
- Completed standalone `ElasticNet` and final-refit `ElasticNetCV` inference across NumPy, CuPy, and Torch. Fold models remain estimation-only.
- Corrected ElasticNet/Ridge scaling documentation: under the shared average-loss convention, `ElasticNet(alpha, l1_ratio=0)` matches `Ridge(alpha)`.
- Made public finite-input guards, cloning, sklearn tags, nested `set_params`, and fitted-state invalidation transactional, including legacy scikit-learn clone identity checks.

### Solver and backend safety

- Corrected the solver matrix so Newton, L-BFGS, and L-BFGS-B reject unsupported non-smooth penalties rather than optimizing only the smooth component.
- Removed the incorrect Euclidean-prox Newton shortcut. Smooth L2/no-penalty objectives retain Newton; non-smooth proximal-Newton requests delegate visibly to backend-native FISTA until a Hessian-metric proximal solver exists.
- Narrowed Armijo, linear-solve, CV-grid, and inference fallbacks to recognized numeric or rank failures. CUDA OOM, device, index, contract, and unrelated runtime failures propagate.
- Normalized warm starts for FISTA, Newton-family, L-BFGS-family, and ADMM solvers to the preprocessed design backend, device, and dtype.
- Completed ADMM's legitimate Cholesky fallback and hardened L-BFGS-B directions, backend-native bounds, and NaN-bound validation.
- Added a centralized, observable Torch compile policy: eager remains the default for unset, `auto`, and `disable`; `default` and `reduce-overhead` are explicit opt-ins. Only the known CUDA Graph output-lifecycle failure becomes a permanent eager fallback.
- Removed the package-initialization cycle between `statgpu.glm_core` and the Cox loss export by lazily exposing `CoxPartialLikelihoodLoss`; fresh-interpreter imports no longer require a particular order.

### Documentation and release preparation

- Reconciled the English and Chinese LogisticRegression, ElasticNet, cross-validation, solver-algorithm, and solver/penalty documentation with the maintained implementation.
- Removed unsupported universal GPU speedup, backend-threshold, and coefficient-tolerance claims. Performance guidance now requires workload-specific benchmarking.
- Documented the ownership boundary between maintained pytest coverage and manual physical-GPU diagnostics.
- Bumped package metadata to `0.2.4` and added `.github/releases/v0.2.4.md` as the authoritative GitHub Release body.

### Validation

- The final PR #87 implementation head passed 2239 tests with 719 skipped, static and documentation contracts, Python 3.9–3.12 regression jobs, scikit-learn 1.2.2/1.3.2/latest compatibility, and release-package validation.
- Physical NVIDIA validation passed on the unchanged numerical implementation: RTX 4090 with PyTorch 2.8.0+cu128 passed the selected compile/CUDA Graph matrix 9/9 and runtime assertions; Tesla P100 with CuPy 13.6.0 passed the corresponding runtime assertions.
- The focused release PR changes version metadata and release-facing documentation only. Exact release-head hosted gates must pass before tag `v0.2.4` is created.

Related: Issue #45, Issue #81, Issue #82, Issue #83, and pull request #87.

## 0.2.3 — 2026-08-04

### Survival analysis

- Completed CoxPH Phase 1 with Breslow, Efron, and Exact ties; delayed-entry
  and `(start, stop]` counting-process data; shared-coefficient stratification;
  subject identifiers; and `Surv(start, stop, event)` formula input.
- Added shared NumPy, CuPy, and Torch-CUDA risk-set primitives for objectives,
  gradients, information matrices, and baseline estimation. Exact tied-event
  partitions use backend-native dynamic programming.
- Extended `CoxPHCV` held-out partial likelihood to all supported tie methods,
  delayed entry, start-stop rows, strata, and subject-grouped folds.
- Hardened Cox inference, centered risk-set numerics, log-domain baseline
  prediction, formula NA alignment, singular-information handling, CV cache
  identity, fold eligibility, selected-penalty refitting, and failed-fit state
  resets.
- Hardened L1, L2, Elastic Net, SCAD, and MCP penalized Cox estimation; removed
  the unidentified intercept; corrected Cox-specific warm starts; and made the
  Torch Efron value, gradient, and Hessian paths native.

### Cross-validation and grouped penalties

- Requested CoxPHCV two-stage and successive-halving controls now execute one
  explicit exhaustive full-precision candidate pass, preserving deterministic
  selection while avoiding repeated complete-grid fitting.
- One-shot `CoxPHCV.cv_splits` iterators are reusable across repeated fit,
  scikit-learn clone, parameter reconstruction, and pickle.
- Public Group Lasso and Adaptive Group Lasso use the generic loss-gradient and
  exact group-proximal path consistently across supported backends.

### Validation and packaging

- Hosted workflow #960 passed on final reviewed head
  `f05a44ad363b46612e956e137e2f00d040765acb`: documentation, static, full CPU,
  and Python 3.9–3.12 regression jobs all passed; the complete CPU suite reported
  1881 passed and 662 skipped.
- The final exact-head physical-GPU promotion artifact is published as
  [schema-3 evidence](https://gist.github.com/TheHiddenObserver/afdcad86a243e68a918d852b92e984a4).
  It records 134/134 passing checks, zero child and nested return codes, empty
  gate-failure arrays, clean source state before and after execution, and SHA-256
  `bd4058450def691dd29e9d78853534016c6da70c33192a97dc312d95cbe5d76d`.
- The package version is now `0.2.3`. Release-package validation checks version
  consistency, builds the pure-Python wheel and sdist, runs `twine check`,
  validates artifact contents, and smoke-installs both distributions in clean
  environments.

## Earlier history

Detailed entries through 2026-08-03 are retained in
[the archived changelog](changelog-history-through-2026-08-03.markdown).
