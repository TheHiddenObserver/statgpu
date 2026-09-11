# Changelog

> Language: English<br>
> Last updated: 2026-09-11<br>
> This page: Changelog<br>
> Switch: [Chinese](../cn/changelog.md)

## Unreleased — Penalized GLM inference contract repair (PR #142, targeted for 0.2.6)

### Changed

- Generic and typed penalized-GLM estimators now use `inference_method="auto"` as the public reconciliation boundary while specialized sparse-Gaussian wrappers keep their established explicit defaults. Fitted provenance distinguishes requested, resolved, and reported methods plus the inferential target and tuning/selection conditioning.
- Supported smooth non-Gaussian L2/no-penalty inference resolves to fixed-penalty `m_estimation` with nonrobust/HC0/HC1 covariance. Non-Gaussian L1/ElasticNet coefficient inference now fails closed rather than exposing the historical L2-curvature-only full-vector sandwich.
- Residual `bootstrap` is explicitly an unweighted CPU Gaussian residual bootstrap with `cov_type="nonrobust"`; refits preserve the actual penalty/tuning/intercept contract, at least two resamples are required, and an executed CuPy/Torch fit is never silently resampled on CPU.
- Non-Gaussian M-estimation follows the fit-recorded NumPy/CuPy/Torch backend and concrete device, including heterogeneous Torch↔CuPy input alignment. The maintained Newton solver now supports genuine non-uniform analytic weights with one normalized objective across value, gradient, Hessian, and Armijo trials, so weighted smooth L2/no-penalty public `solver="auto"` no longer needs PR #142's temporary FISTA override. Direct fits and `PenalizedGLM_CV` selection/final refit again follow the canonical dispatch, including backend-native Newton for applicable logistic/Poisson L2 rows, while the public request remains `auto` and historical floating uniform-weight `allclose` semantics are preserved.
- `PenalizedGLM_CV` now exposes inference controls and runs coefficient inference exactly once on the selected full-data final refit. Results condition on the CV-selected penalty and explicitly report `penalty_selection_adjusted_=False`.

### Validation

- Added targeted contract, formula, clone/compatibility, failure-transaction, no-penalty, weighted-CV, installer-idempotence, weighted-Newton objective/compatibility, and cross-backend alignment regressions plus bilingual model/CV/inference documentation.
- `dev/benchmarks/validate_penalized_glm_inference_gpu.py` schema v4 is the maintained physical CUDA gate. At exact clean numerical source `db448d718f523eacf97bcb3c419e376c9812362d`, Tesla P100 validation passed with CuPy 13.6.0, Torch 2.0.0+cu117, and NumPy 1.24.2: Logistic unweighted retained explicit FISTA coverage, while Logistic weighted and Poisson weighted/unweighted followed canonical `solver="auto"` → Newton; CuPy, Torch, Torch→CuPy, and CuPy→Torch routes all passed the unchanged coefficient/intercept (`2e-6`) and inference (`1e-5`) thresholds with concrete `cuda:0` provenance. The retained artifact is `results/pr142_penalized_glm_inference_gpu/pr142_penalized_glm_inference_gpu.json`. Subsequent changelog/evidence-only commits reuse this immutable numerical-source artifact by explicit approval; any numerical, validator, solver, backend, inference, or tolerance change reopens physical validation.

## Unreleased — Post-selection OLS inference API cleanup (PR #138 / Issue #137)

### Changed

- Added canonical hardware-neutral `inference_method="post_selection_ols"` for sparse Gaussian `Lasso`, `ElasticNet`, and the public generic `PenalizedGeneralizedLinearModel(loss="squared_error", penalty="l1" | "elasticnet")` surface. Legacy `cpu_ols` / `gpu_ols` are one-cycle `FutureWarning` aliases; `LassoCV` keeps its older `cpu_ols_inference` / `gpu_ols_inference` spellings at the CV compatibility boundary.
- Statistical-method identity is orthogonal to execution hardware. Explicit `device="cpu"`, `"cuda"`, or `"torch"` remains authoritative even for heterogeneous input containers; only genuine AUTO policy may preserve native CuPy/Torch-CUDA input. LassoCV now keeps CV and selected-alpha final refit on one resolved backend and aligns CuPy response/weight arrays to the design's concrete CUDA ordinal.
- `post_selection_ols` performs an unpenalized active-set OLS/WLS refit on the fit-recorded NumPy/CuPy/Torch backend while leaving penalized `coef_` unchanged for prediction. Nonrobust inference retains Student-t semantics and the historical inactive-coordinate placeholders. Rank-deficient active designs use effective rank for residual df plus a design-level Moore-Penrose/SVD refit and covariance bread; robust/HAC and empty-active no-intercept cases preserve their requested covariance/reference family.
- Kept active-refit diagnostic state separate from penalized-fit R-squared/F/log-likelihood/AIC/BIC ownership. `summary()` reports penalized-fit and post-selection residual DoF separately, formula fitting preserves categorical/missing-row/sample-weight alignment, and failed refits fail closed instead of retaining prior or partially updated fit/inference state.
- Unified sparse-Gaussian analytic-weight semantics across direct NumPy/CuPy/Torch fits and weighted LassoCV. Weighted centering is performed on original observations before the equivalent `sqrt(w * n / sum(w))` row transform; default CV alpha grids, fold objectives, weighted validation MSE, and final refits share the same convention. Positive constant weights are the exact unweighted CV problem. Weighted non-Gaussian sparse GLMs remain on their loss-specific sample-weight-aware objectives.
- Unified debiased inference around the same centered average-loss working problem on NumPy/CuPy/Torch, so omitted weights, all-one weights, and globally rescaled analytic weights are consistent. Intercept-inclusive simultaneous max-|Z| inference now includes the original-coordinate intercept influence in the bootstrap maximum itself, and successful refits clear stale simultaneous/precision state before publishing a new result.
- String and public `Penalty`-object forms participate in the same sparse-Gaussian migration and AUTO-routing contract. Clone/get-params/set-params, warning call sites, LassoCV final-refit ownership, backend/device provenance, formula routing, and failure transactions are covered by maintained regressions.

### Validation

- Hosted validation covers Python 3.9/3.12, Torch 2.0 CPU, the full CPU suite, scikit-learn 1.2.2/1.3.2/current maintenance compatibility, static/ruff, documentation, release packaging, and benchmark-frontend contracts.
- `dev/benchmarks/validate_post_selection_ols_gpu.py` is the final physical-CUDA gate and is now schema v7 with **22 cases**: the original four direct-Lasso cases plus 18 CuPy/Torch closure cases for ElasticNet/generic sparse Gaussian, weighted and unweighted debiased inference, real weighted multi-alpha LassoCV selection/final refit, rank-deficient SVD refits, Penalty-object AUTO routing, empty-active HC3, and intercept-inclusive simultaneous max-|Z| inference. Existing post-selection numerical tolerances are not relaxed.
- Earlier Tesla P100 artifacts remain immutable evidence only for the exact historical SHAs they validated. Because subsequent review/fix loops changed valid production numerical paths, current-source physical acceptance is still **pending an exact clean-head schema-v7 22/22 CuPy/Torch CUDA run**. Hosted checks do not substitute for that gate, and no GPU performance claim is made.

## Unreleased — Penalized solver API cleanup (PR #135)

### Changed

- Direct public penalized estimators use backend-neutral `solver` as the authoritative direct-fit solver selector. Legacy `cpu_solver` remains accepted for one compatibility cycle and emits `FutureWarning` for caller-owned explicit use, but it is not silently remapped into `solver`, so existing unified-engine numerical behavior is preserved.
- `LassoCV` now separates `solver` (final full-data refit) from `cv_solver` (CV folds/path). `cv_solver="auto"` resolves to coordinate descent on CPU and FISTA on CUDA/Torch; `cv_solver_` records the algorithm that actually executed.
- Deprecated `LassoCV(cpu_solver=...)` preserves its historical stage semantics: on CPU it remains a legacy CV-solver alias, while on CUDA/Torch it warns but stays non-authoritative so the maintained GPU FISTA path is unchanged.

### Compatibility

- Omitted direct `cpu_solver` values and framework reconstruction stay warning-free, including scikit-learn 1.2's `get_params() -> constructor` clone path and newer `__sklearn_clone__` reconstruction.
- Explicit `set_params(cpu_solver=...)` and legacy `LassoCV(..., cpu_solver=...).fit(...)` warnings point to the caller rather than statgpu reconstruction/validation frames.
- Migration guidance distinguishes behavior-preserving removal of an already non-authoritative direct `cpu_solver` from an intentional solver change that explicitly moves an old algorithm choice into `solver`.

### Validation

- Added focused solver/deprecation regression coverage for direct solver authority, omission versus explicit legacy values, sklearn clone/reconstruction, internal helper warning suppression, `set_params`, LassoCV CPU/GPU alias behavior, `cv_solver_`, and warning call sites.
- The maintenance compatibility workflow runs the focused suite under scikit-learn 1.2.2, 1.3.2, and current; exact-head hosted results are recorded in PR #135 after the final source head completes CI.

## Unreleased — Gaussian backend-native inference (PR #129 / Issue #127)

### Changed

- Maintained Gaussian linear-model numerical covariance, standard errors, statistics, p-values, and confidence intervals now execute on the actual NumPy/CuPy/Torch fit backend; established reporting attributes/results may still take a final NumPy snapshot after numerical inference completes.
- Normal and Student-t inference routes through the maintained reference-distribution layer, including stable df=1/df=2 extreme-tail handling. Missing or invalid executed-backend provenance fails closed instead of silently choosing NumPy.
- Ridge/L2 inference preserves the existing average-loss convention and `n_eff * alpha` normal-equation mapping, including weighted fits and `RidgeCV` final-refit inference.

### Validation

- Added public `LinearRegression`, formula, weighted/robust, rank-deficient, multi-target, float32, statsmodels-alignment, no-host-transfer, non-L2 delegation, and Ridge/RidgeCV regression coverage plus a focused hosted CI workflow.
- Added a maintained exact-SHA physical CUDA validator for CuPy and Torch with clean-tree proof, requested/executed backend and concrete-device provenance, covariance/BSE/statistic/p-value/CI error reporting, weighted/rank/multi-target/small-df cases, and `RidgeCV` final-refit inference.
- All hosted gates and the fresh complete-diff review are green on the reviewed implementation head. Final acceptance still requires exact clean-head CuPy/Torch CUDA validation on the final source SHA; PR #129 remains open/unmerged and #127 is not yet `COMPLETE`. No GPU speedup claim is made.

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

## 2026-08-09 — Panel Tier-1 Stage C covariance completion (PR #126)

Stage C completes the Panel Tier-1 covariance and inference surface while preserving estimator coefficients and standardized fit-statistic definitions. `robust` remains the existing HC1 contract; `hc0`, `hc2`, and `hc3` use each estimator's actual transformed fit space. `RandomEffects` supports robust/HC, clustered, and Driscoll-Kraay covariance on quasi-demeaned GLS scores, and fails closed if the Swamy-Arora between auxiliary regression has no positive residual degrees of freedom. One-/two-way clustering supports opt-in `group_debias=True`; clustered inference now also fails closed when any supplied clustering dimension has fewer than two distinct groups, instead of returning the degenerate one-cluster sandwich. `PooledOLS(cov_type="hac")` remains the legacy row-order Bartlett/Newey-West path.

The implementation also hardens two-way fixed-effect convergence and prediction. Pipe-named Panel formula metadata is now authoritative: conflicting explicit entity/time IDs fail closed, missing named pipe columns cannot be replaced by unrelated explicit IDs, RandomEffects accepts a second pipe time variable only when Driscoll-Kraay covariance actually consumes it, and fixed-effect magic tokens are rejected for RandomEffects rather than being reinterpreted as grouping metadata. Entity/time projection metadata is factorized once and reused on the selected backend; convergence checks residual means for both effect dimensions, uses a scale-aware roundoff floor for numerically absorbed directions, and exposes fail-closed `demean_max_iter`/`demean_tol` controls for weakly connected panels. Two-way fixed effects are recovered jointly for unbalanced prediction, while known entity/time labels from different disconnected incidence components are rejected as unidentified. Formula prediction now fails closed if Patsy would drop input rows or if a formula transformation creates non-finite design values. Prediction also no longer guesses that every one-column-short matrix omitted an intercept, and an explicitly fitted non-unit constant is restored by value and position only on the compatible path. Known fixed-effect labels now restore the centered level grand mean so `PanelOLS.predict()` returns the complete fixed-effect level projection; formula-enabled effects no longer leak across refits, formulas with more than two fixed-effect variables fail closed, and no-FE `PanelOLS` formulas preserve Patsy/R default-intercept semantics (with `0 +` / `-1` retaining the explicit no-intercept path); no-intercept `rsquared_within` now uses the standard uncentered total sum of squares. The same review also strengthens rank-deficient coefficient identifiability and makes classical Hausman fail closed whenever either fitted coefficient vector is non-unique, plus `FirstDifferenceOLS` duplicate/time semantics, HC2/HC3 leverage stability, metadata alignment, CuPy scatter-add, RandomEffects formula intercept/name behavior, and quadratic-spectral weights. External definitions are checked against pinned `statsmodels==0.14.6`, `linearmodels==7.0`, R `plm==2.6-7`, and R `sandwich==3.1-3` references.

The latest numerical hardening unifies NumPy/CuPy/Torch Fama-MacBeth period dispatch under the same conservative Gram certificate and maintained SVD fallback. The certificate rejects non-finite Gram/RHS/solutions before they can preempt fallback; shared SVD least-squares uses inverse-singular-value factor ordering plus a safe uniform working scale for collectively subnormal full-rank designs. Fama-MacBeth coefficient averages and shared parameter-R² means use reduction-length scaling only when overflow is possible, avoiding extra magnitude-normalization loss without claiming arbitrary compensated cancellation recovery; coefficient-series covariance uses per-coordinate scales with symmetric restoration. Shared panel inference no longer imposes an absolute variance floor: exact-zero variance maps a zero coefficient to statistic 0 and a nonzero coefficient to signed infinity. Classical model F, pooling F, and Breusch-Pagan LM use overflow-safe centering and subnormal-safe backend normalization. Residual-based covariance now delays tiny-design/projection scale restoration until after cancellation: projection coordinates are scaled only when projection×residual would overflow, cluster and DK scores are grouped before selective Gram scaling, and residual vectors are never globally magnitude-normalized. This preserves small representable components beside huge cancelling observations without degrading already-safe subnormal-design precision. Two-way clustering recognizes nested partitions independently of arbitrary code numbering and cancels identical marginal/intersection components before restoration. Range-aware symmetrization/inclusion-exclusion plus HAC/DK pre-Gram and full-lag accumulators then prevent avoidable intermediate overflow. The maintained physical Stage-C runner includes diagnostic-scale, zero-variance, pre-Gram, tiny-design, mixed-range, nested-partition, covariance extreme-scale, and lag-accumulation branches for both CuPy and Torch CUDA.

Physical validation is recorded as an exact-source evidence chain. Historical Stage-C and Fama-MacBeth artifacts remain valid only for their original numerical heads. The previously accepted P100 source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` is now historical for the current PR branch because later review-fix loops changed valid Fama-MacBeth and shared panel least-squares paths; fresh exact-head CuPy/Torch CUDA acceptance is required before merge readiness can be promoted. On Tesla P100, the broad Stage-C runner passed 35/35 estimator/covariance cases and 12/12 public primitives on each of CuPy and Torch; the dedicated HAC chronology runner passed ordered-categorical/numeric equivalence, the lexical negative control, and shared backend-native Student-t inference. Fama-MacBeth now uses a conservative Gram-spectrum certificate for exact-size GPU batches: only clearly well-conditioned periods may consume the batched Gram solve, while every uncertified period remains owned by the original SVD rank policy. The accepted scaling artifact reports CuPy/Torch GPU-over-NumPy median-time ratios of **0.549/0.343** on micro (64×128×4), **0.204/0.168** on medium (128×1,024×8), and **0.114/0.109** on large (128×4,096×16), corresponding to about 1.82×/2.92×, 4.91×/5.97×, and 8.75×/9.16× speedups. Every measured GPU scale used one `gram-certified` batch, one control synchronization, and zero SVD fallbacks; input transfer is excluded from this resident-array timing protocol, so these results are workload/hardware-specific evidence rather than a universal GPU guarantee. The focused Fama-MacBeth gate also verifies chronology/formula/rank/inference behavior, backend-native public arrays, backend-native distribution inference, and prediction/device provenance. All four final physical runners are preserved under `results/pr126_p100_fama_fix/` and point to the same numerical source.

### Validation (2026-08-22)

A fresh review-fix loop on the PR branch hardened the remaining numerical and device paths and re-ran the full physical matrix at exact head `5068da3f`:

- **Two-way clustered covariance performance**: the exact per-row dyadic two-sum fallback (always triggered by ordinary balanced panels above ~6.5k rows, ~1000s per CuPy fit at 10k rows) is now gated by a residual-acceptance check. Ordinary designs stay on the vectorized Gram path; only genuinely recoverable cancellation residuals fall back to the exact row products. On Tesla P100, `pooled_cluster_two_way` 10k-row CuPy fits drop from **~1018s to ~1.3s** (Torch ~0.2s; 100k rows ~0.4s), and `benchmark_panel_stage_c_covariance.py` completes its 60-row matrix in ~40s instead of timing out.
- **Numerical hardening**: CuPy `maximum.at`/`cupyx.scatter_max` return `inf` for float64 magnitudes around 1e7..1e308 (observed CuPy 13.6), so group min/max scatter now uses the sequential host scatter; Torch CUDA SVD uses the exact `gesvd` driver (the default `gesvdj` leaks ~1e-16 into structurally-zero `U` entries that huge responses amplify); failed panel fits retain the executed backend provenance; Student-t(1) p-values use the well-conditioned `2 atan(1/x)/pi` form so extreme statistics (e.g. |t|=1e154) keep their representable tail instead of the subtractive survival collapse; formula side-array alignment fails closed on over-long inputs.
- **CuPy device affinity**: backend availability probes no longer switch the current CUDA device, and panel allocations (scatter targets, dummies, row weights) bind to the reference device; a physical device-affinity gate (`validate_panel_cupy_device_affinity_gpu.py`) covers both CuPy and Torch CUDA.
- All 12 physical runners pass at the exact head on Tesla P100 (CuPy 13.6.0 / Torch 2.0.0+cu117): Stage-C correctness (35 cases + 12 primitives per backend), focused Fama-MacBeth oracle + certified-Gram provenance, HAC chronology, extreme t(2) tail, device affinity, Fama-MacBeth scaling, RHS cancellation, rank precedence, and intercept cancellation. Artifacts: `results/pr126_perf_fix_528d967e/`, `results/pr126_review_fix_da3604ee/`.

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
