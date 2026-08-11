# Changelog

> Language: English<br>
> Last updated: 2026-08-11<br>
> This page: Changelog<br>
> Switch: [Chinese](../cn/changelog.md)

## 2026-08-09 — Panel Stage C covariance completion (PR #126)

Stage C extends the Panel Tier-1 inference layer without changing estimator coefficients or the Stage-B diagnostic definitions. `robust` remains the historical HC1 contract; new `hc0`, `hc2`, and `hc3` use each estimator's actual transformed fit space. `RandomEffects` supports robust/HC, clustered, and Driscoll-Kraay covariance on the quasi-demeaned GLS scores. One-/two-way clustering has opt-in `group_debias=True`; the default clustered result is unchanged. `PooledOLS(cov_type="hac")` remains the legacy row-order Bartlett/Newey-West path.

The repaired covariance implementation derives bread and influence rows from the design pseudoinverse, computes HC2/HC3 leverage from `diag(X X+)`, validates entity/time/cluster metadata consistently, keeps CuPy group scatter-add backend-native, publishes the shared inference result contract, preserves RandomEffects formula intercept and feature names, and stabilizes quadratic-spectral weights for very large bandwidths. External definitions are checked against pinned `statsmodels`, `linearmodels`, and R `sandwich`/`plm` references.

The `ec511f53...` Tesla P100 32/32-per-backend correctness and 58-row performance run is retained as immutable historical evidence. Because later review fixed the shared numerical-rank cutoff and FirstDifference ordered-categorical chronology, current physical acceptance is pending a fresh exact-head 26-estimator + 12-primitive (**38/38 per backend**) correctness run and synchronized performance rerun.

## 2026-08-08

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
