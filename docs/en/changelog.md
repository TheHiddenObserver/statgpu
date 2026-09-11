# Changelog

> Language: English  
> Last updated: 2026-09-12  
> This page: Release history  
> Switch: [Chinese](../cn/changelog.md)

This page records user-visible changes for current and recent statgpu releases.

## Unreleased — Weighted explicit Newton/L-BFGS GLM fits (PR #151 / Issue #150, targeted for 0.2.6)

### Changed

- Ordinary `GeneralizedLinearModel` explicit `solver="newton"` and `solver="lbfgs"` now accept genuine non-uniform analytic `sample_weight` on supported GLM rows without silently substituting IRLS/FISTA or changing an explicit NumPy/CuPy/Torch execution request.
- Maintained GLM L-BFGS uses one normalized analytic-weight objective, `sum(w_i * loss_i) / sum(w_i)`, for the initial gradient, line-search objective/candidates, and accepted-point gradient. Global positive weight rescaling, uniform-weight identity, zero-weight-row equivalence, integer row replication, and representative statsmodels alignment are covered by regression tests. Generic Huber/Quantile/Cox direct non-uniform weighted L-BFGS remains fail-closed.
- Existing `solver="auto"`, IRLS/FISTA, explicit smooth-solver `C`, Ordered GLM, and standalone LogisticRegression behavior is preserved. Ordinary GLM fits now record the solver/backend/device that actually completed successfully. Weighted inverse-power Gamma explicit Newton/L-BFGS requires `fit_intercept=True`; the weighted no-intercept row fails closed precisely rather than relying on a data-dependent feasible initialization.
- Existing weighted smooth-GLM L-BFGS consumers in penalized fitting and `PenalizedGLM_CV` are covered, including the family-specific CV L2 rows that already dispatch Gamma, Inverse Gaussian, and Negative Binomial through L-BFGS. The final promotion gate remains exact-source physical CuPy/Torch CUDA validation with the frozen schema-v2 validator; hosted CI alone does not substitute for that evidence.

## Unreleased — Backend-native Gaussian residual bootstrap (PR #147 / Issue #145, targeted for 0.2.6)

### Changed

- The existing unweighted Gaussian `residual_bootstrap` row now executes on the backend and concrete device recorded by the successful fit: NumPy/CPU, CuPy CUDA, or Torch CUDA. The statistical procedure is unchanged: residuals are resampled around the fixed fitted values and every bootstrap draw refits the same penalized Gaussian estimator.
- For a fixed `bootstrap_random_state`, NumPy, CuPy, and Torch consume the same deterministic residual-index schedule. GPU fits keep response construction and child optimization on the selected GPU device; only the small index/control data, the already-established small parent parameter snapshot used to reconstruct fitted values, and the completed reporting snapshots cross the host/device boundary.
- Bootstrap child refits preserve the fitted penalty family/object, `alpha`, `l1_ratio`, penalty kwargs, intercept semantics, solver/stopping controls, `lipschitz_L`, applicable loss kwargs, and SCAD/MCP LLA controls. Child inference remains disabled so bootstrap never recursively invokes itself.
- `PenalizedGLM_CV` continues to run bootstrap only once, after selecting the penalty and refitting on the full dataset. Results therefore condition on the selected penalty; tuning-selection uncertainty is not corrected.
- Weighted residual bootstrap, robust/HC bootstrap, HAC/block bootstrap, non-Gaussian bootstrap, Cox bootstrap, and batched bootstrap refits remain outside this procedure and fail closed rather than switching to a different resampling method.

### Validation

- Hosted coverage includes deterministic NumPy preservation, L1/ElasticNet/SCAD/MCP refit ownership, exact-device child contexts, formula and typed-wrapper consumers, failure transactions, CV final-refit-only behavior, runtime-installer idempotence, and the unchanged fail-closed rows.
- `dev/benchmarks/validate_gaussian_residual_bootstrap_gpu.py` is the frozen schema-v1 physical CUDA gate for CuPy/Torch L1 and ElasticNet parity, deterministic same-draw identity, concrete device provenance, Torch↔CuPy heterogeneous-container routing, and no CPU numerical fallback. Physical exact-source CUDA evidence remains required before the PR can be promoted to Ready/merge.

## Unreleased — Penalized GLM inference contract (PR #142, targeted for 0.2.6)

### Changed

- Generic and typed penalized GLMs now recommend public `inference_method="auto"` while preserving specialized sparse-Gaussian wrapper defaults. Successful inference-enabled fits expose the requested method, resolved numerical method, reported result method, inferential target, and penalty/tuning conditioning instead of inferring those semantics from a single `inference_method_` string.
- Supported smooth non-Gaussian L2/no-penalty rows use fixed-penalty `m_estimation` with nonrobust/HC0/HC1 covariance. Positive L2 inference targets the penalized estimating equation; zero-penalty aliases target the unpenalized population parameter. Non-Gaussian L1/ElasticNet coefficient inference now fails closed instead of publishing the historical partial sandwich approximation.
- `inference_method="bootstrap"` is intentionally narrow: it resolves only to an unweighted Gaussian residual bootstrap with `cov_type="nonrobust"` on the current NumPy/CPU refit path. Refits preserve the fitted penalty family and tuning (`alpha`, `l1_ratio`, penalty kwargs, intercept, solver/stopping and LLA controls), require at least two draws, and never silently transfer CuPy/Torch fits to CPU. Backend-native Gaussian resampling is tracked separately in Issue #145.
- Supported non-Gaussian M-estimation now treats the fit-recorded numerical backend and concrete device as authoritative. CuPy and Torch inference stays on that backend/device, including heterogeneous input containers; there is no inference-time re-resolution that can silently switch a fit to another numerical engine.
- `PenalizedGLM_CV` exposes the same inference controls. CV folds/candidates remain estimation-only; inference runs once on the selected full-data final refit. Successful CV results report conditioning on the selected penalty with `penalty_selection_adjusted_=False`, so tuning-selection uncertainty is not implied to be corrected.
- The generic/typed GLM pages now distinguish these current methods from reserved-but-unimplemented robust/HAC/bootstrap combinations and keep EN/CN support claims aligned.
- Genuine non-uniform analytic `sample_weight` is now supported by the shared Newton solver for losses that expose weighted value/gradient/Hessian primitives. Weighted smooth non-Gaussian L2 fits no longer need an inference-only FISTA override: public `solver="auto"` keeps the canonical dispatch and applicable Logistic/Poisson L2 rows resolve to backend-native Newton across direct fit, CV, selected final refit, and M-estimation inference. The public request remains `auto`; the executed solver is recorded separately. Floating-point weight vectors covered by the historical Newton `allclose` uniformity rule still follow the established unweighted numerical path.

### Validation

- Hosted validation covers API/clone compatibility, formula/categorical/missing-row alignment, failure transactions, NumPy/CuPy/Torch provenance guards, deterministic residual-bootstrap ownership, final-refit-only CV inference, and targeted external references.
- Exact clean numerical source `db448d718f523eacf97bcb3c419e376c9812362d` passed the maintained Tesla P100 schema-v4 physical validator with CuPy 13.6.0, Torch 2.0.0+cu117, NumPy 1.24.2, and Python 3.9.16. The four maintained family/weight rows covered Logistic explicit-FISTA unweighted plus canonical `solver="auto"` → Newton for Logistic weighted and Poisson weighted/unweighted. Every CuPy, Torch, Torch→CuPy, and CuPy→Torch route passed the unchanged coefficient/intercept (`2e-6`) and inference (`1e-5`) tolerances with concrete `cuda:0` provenance; the retained artifact is `results/pr142_penalized_glm_inference_gpu/pr142_penalized_glm_inference_gpu.json` (15944 B).
- Final hosted CI and the fresh independent review are green on the merged review head. The P100 artifact remains anchored to the immutable numerical source above; subsequent changelog/evidence-only closure reuses it by explicit approval, while any numerical, validator, backend, solver, inference, or tolerance change would require a fresh physical run.

## Unreleased — Node-wise Lasso inference tuning contract (PR #139, targeted for 0.2.6)

### Changed

- Added public `nodewise_alpha=None` control for debiased sparse-Gaussian inference on `PenalizedGeneralizedLinearModel`, `PenalizedLinearRegression`, `Lasso`, `ElasticNet`, `LassoCV`, and `ElasticNetCV`. Explicit finite positive real values are authoritative; `PenalizedGLM_CV` keeps this control at its current unsupported/default boundary.
- Replaced the response-scale-dependent internal node-wise penalty with a response-independent design rule. Omitted `nodewise_alpha` now resolves from the standardized centered/weighted working design as `sqrt(2 * log(max(p, 2)) / n_nodewise)`, with a Kish-style effective sample size under non-uniform analytic weights.
- Node-wise precision is solved in standardized design coordinates, validated by a full KKT certificate, and back-transformed as `D^{-1} Theta_Z D^{-1}`. The analytic `p=1` case does not consume a node-wise penalty.
- Added fitted/provenance reporting for requested/resolved node-wise tuning and cache keys that distinguish numerical node-wise precision identity from the main-model tolerance. Explicit/CV/formula/GPU paths therefore cannot silently reuse stale precision from another node-wise request.
- Preserved the existing debiased inferential target, covariance/reference-distribution semantics, active prediction coefficients, and post-selection OLS behavior. Legacy response-scale-dependent node-wise tuning is intentionally not exposed as a new public mode.

### Validation

- Added deterministic fixed-design simulation evidence comparing the old response-dependent rule, the new automatic rule, and a representative explicit rule at `n=320`, `p=18`, 120 repetitions and two response-noise scales. The omitted/default rule keeps its node-wise penalty constant across noise scales while the legacy rule changes by exactly 8×; the new rule is non-inferior on coverage error and simultaneously improves average interval width in that migration design.
- Hosted tests cover low-level scalar/array node-wise tuning, KKT validation, cache isolation, public API/clone/refit behavior, CV selected-final-refit inference, formula routing, unavailable GPU/Torch-CUDA behavior, and the committed migration simulation artifact.
- `dev/benchmarks/validate_nodewise_alpha_inference_gpu.py` is the maintained physical CUDA gate for CuPy/Torch execution. It requires both frameworks on a real CUDA device and checks automatic/explicit node-wise tuning, NumPy parity, response-scale invariance of the automatic penalty, weighted rescaling, and cross-container provenance.

## Unreleased — Post-selection OLS inference API (PR #138 / Issue #137, targeted for 0.2.6)

### Changed

- Added canonical hardware-neutral `inference_method="post_selection_ols"` for sparse Gaussian `Lasso`, `ElasticNet`, and the public generic `PenalizedGeneralizedLinearModel(loss="squared_error", penalty="l1" | "elasticnet")` surface. The historical `cpu_ols` / `gpu_ols` strings remain one-cycle compatibility aliases and emit `FutureWarning`; `LassoCV` keeps `cpu_ols_inference` / `gpu_ols_inference` at its CV-specific compatibility boundary.
- Statistical method selection is now orthogonal to hardware. Explicit `device="cpu"`, `"cuda"`, and `"torch"` remain authoritative even for heterogeneous input containers, while genuine AUTO may preserve native CuPy/Torch-CUDA input. LassoCV now uses one resolved backend for folds and the selected-alpha final refit and preserves concrete CuPy device affinity.
- `post_selection_ols` refits an unpenalized OLS/WLS model on the selected active set using the fit-recorded NumPy/CuPy/Torch backend, but keeps the original penalized `coef_` / `intercept_` as the prediction model. Rank-deficient active designs use effective rank for residual degrees of freedom and the same Moore-Penrose/SVD design geometry for the refit and covariance bread.
- Post-selection diagnostics now belong to the refit rather than overwriting penalized-fit R-squared, F, log-likelihood, AIC, or BIC. Summary output labels effective residual DoF separately and preserves the penalized fit diagnostics on the estimator.
- Formula-based sparse fits retain categorical/interaction/missing-row semantics and align array-like sample weights to Patsy-retained rows. Rejected or post-fit-inference-failed refits clear stale result-bearing state and fail closed.
- Unified sparse-Gaussian analytic weights across direct NumPy/CuPy/Torch fits and weighted LassoCV: weighted centering precedes the equivalent `sqrt(w * n / sum(w))` transform; default alpha grids, fold objectives, validation MSE, and final refits use the same convention. Positive constant weights are the exact unweighted CV problem.
- Unified debiased inference around the same centered average-loss working problem. Intercept-inclusive simultaneous max-|Z| inference now includes the original-coordinate intercept influence, and successful refits clear stale simultaneous/precision state before publication.

### Validation

- Hosted coverage includes Python 3.9/3.12, Torch 2.0 CPU, the full CPU suite, maintenance compatibility, static/documentation checks, release packaging, and benchmark-frontend contracts.
- `dev/benchmarks/validate_post_selection_ols_gpu.py` is the final schema-v7 22-case CuPy/Torch physical gate. Historical P100 artifacts remain evidence only for their recorded source SHAs; a fresh exact-head physical run is still required before the PR can be considered merge-ready. No GPU speedup claim is made.

## Unreleased — Penalized solver API cleanup (PR #135, targeted for 0.2.6)

### Changed

- Made backend-neutral `solver` the authoritative direct-fit solver selector across the public penalized estimator family.
- Retained legacy `cpu_solver` for one compatibility cycle. Meaningful legacy use emits `FutureWarning`; omission remains quiet and preserves historical behavior.
- Added stage-specific `LassoCV.cv_solver` for fold/path execution and fitted `cv_solver_` for the algorithm that actually ran. `solver` continues to control the final full-data refit.
- Preserved legacy CUDA/Torch FISTA behavior during the migration window instead of silently remapping `cpu_solver` into a different direct numerical path.

### Validation

- Added constructor/signature/get-params/set-params/clone coverage across maintained sklearn versions and caller-facing warning-location checks.
- Added CV regressions proving `cv_solver` belongs to folds/path while `solver` belongs to the final refit.
- Updated English and Chinese migration, solver, CV, and model documentation to distinguish behavior-preserving API migration from an intentional solver change.

## Unreleased — 2026-08-28

### PR #129 / Issue #127 — Gaussian linear-model backend-native inference

- Migrated maintained Gaussian linear-model covariance, standard-error, statistic, p-value, and confidence-interval numerical work to the executed NumPy/CuPy/Torch backend while preserving the established final NumPy reporting snapshot.
- Routed normal/Student-t reference-distribution work through the maintained shared inference layer, including stable df=1/df=2 extreme-tail handling. Missing or invalid executed-backend provenance now fails closed instead of silently routing inference through NumPy.
- Preserved Ridge/L2 average-loss scaling, analytic weights, `RidgeCV` selected-final-refit inference, formula row alignment, rank-deficient designs, multi-target inference, and the public NumPy result/reporting contract.
- Added exact-backend provenance checks and no-host-transfer regressions for direct Gaussian and Ridge/CV consumers.

### Validation

- Hosted CI and the fresh final review are complete for the merged numerical source.
- Tesla P100 physical validation passed on the accepted exact numerical source with NumPy/CuPy/Torch covariance/BSE/statistic/p-value/CI parity, weighted/rank/multi-target/small-df cases, concrete backend/device provenance, and a `RidgeCV` final-refit case. Later docs-only closure reused that immutable numerical artifact under the explicitly approved documentation-only evidence exception.

## 0.2.5 — 2026-08-26

### Panel Tier-1 Stage C covariance and release closure

- Shipped HC0/HC2/HC3, legacy HC1 where defined, one-/two-way clustering with optional `group_debias`, Driscoll-Kraay with Bartlett/Parzen/QS kernels, RandomEffects robust/HC inference, and legacy row-order HAC with ordered-categorical chronology.
- Added transactional panel fits, row-preserving formula prediction, classical Hausman/pooling-F/Breusch-Pagan LM diagnostics, and overflow-safe fit statistics.
- Hardened fixed-effect nuisance-rank residual df, rank-deficient HC2/HC3 behavior, extreme-scale covariance arithmetic, level prediction, panel diagnostics, Fama-MacBeth rank handling, and concrete-device CuPy/Torch affinity.
- Kept ordinary panel workloads on vectorized GPU paths while reserving exact accumulation for genuine cancellation risk; maintained 10k-row CuPy two-way covariance dropped from roughly 1000 s to about 1.3 s on the recorded Tesla P100 benchmark workload.
- Final release acceptance on numerical source `697de113` passed all 12 maintained physical runners; artifacts are retained under `results/pr126_release_697de113/`.

### Distribution inference and packaging

- Fixed CuPy inverse Beta/Gamma LUT cache tuple ordering, restoring Student-t/Beta/F/Gamma/chi-square inverse quantiles and downstream confidence-interval widths.
- Released the corresponding panel/distribution regression coverage and universal `py3-none-any` package artifacts for `0.2.5`.

## Unreleased — 2026-08-09

### PR #126 — Panel Tier-1 Stage C covariance

- Finalized panel Tier-1 Stage C covariance and inference across `PanelOLS`, `RandomEffects`, `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, and `FamaMacBeth`.
- Added HC0/HC2/HC3, clustered, two-way clustered, Driscoll-Kraay, robust RandomEffects inference, stable fixed-effect level prediction, and hardened Hausman/pooling-F/Breusch-Pagan diagnostics.
- Hardened the shared panel numerical layer against extreme float64 dynamic range, recoverable cancellation, subnormal designs, singular/rank-deficient systems, unsafe CuPy scatter reductions, and Torch CUDA SVD driver instability.
- Preserved existing HC1/legacy HAC/public coefficient inference where applicable while making unsupported or numerically unresolved covariance cases fail closed.
- Completed exact-source physical CUDA validation and retained the numerical evidence under the corresponding `results/pr126_*` artifacts.

### PR #122 — Panel Tier-1 diagnostics Stage B

- Added structured `fit_statistics_` for within/between/overall and adjusted R², classical model F, pooling F, one-way entity Breusch-Pagan LM, and classical FE-vs-RE Hausman diagnostics.
- Preserved Stage-A coefficient inference and legacy df/R² attributes while adding NumPy/CuPy/Torch, formula-row-alignment, linearmodels-definition, and physical-GPU evidence.

### PR #121 — CuPy inverse-quantile LUT correctness

- Fixed CuPy `betaincinv` / `gammaincinv` lookup-table cache ordering and restored public CuPy Student-t, Beta, F, Gamma, and chi-square inverse quantiles.
- Validated the unchanged two-line numerical repair on Tesla P100 and added maintained public-distribution plus downstream Panel inference regressions.

### PR #119 — Panel Tier-1 shared framework Stage A

- Added internal `BasePanelModel`, shared panel index/balance metadata, and the shared residual-based covariance/inference substrate without expanding public diagnostics beyond Stage A.
- Migrated the maintained panel estimators to the shared lifecycle while preserving formula, prediction, summary, fixed-effect recovery, RandomEffects GLS, Fama-MacBeth chronology, backend output, and no-fallback behavior.
- Added pre-refactor golden regression coverage and Python 3.9 + Torch 2.0 CPU execution for shared panel metadata/covariance/inference tests.

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
- Documented ownership boundary between maintained pytest coverage and manual physical-GPU diagnostics.
- Bumped package metadata to `0.2.4` and added `.github/releases/v0.2.4.md` as the authoritative GitHub Release body.

### Validation

- The final PR #87 implementation head passed 2239 tests with 719 skipped, static and documentation contracts, Python 3.9–3.12 regression jobs, scikit-learn 1.2.2/1.3.2/latest compatibility, and release-package validation.
- Physical NVIDIA validation passed on the unchanged numerical implementation: RTX 4090 with PyTorch 2.8.0+cu128 passed the selected compile/CUDA Graph matrix 9/9 and runtime assertions; Tesla P100 with CuPy 13.6.0 passed the corresponding runtime assertions.
- The focused release PR changes version metadata and release-facing documentation only. Exact release-head hosted gates must pass before tag `v0.2.4` is created.

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