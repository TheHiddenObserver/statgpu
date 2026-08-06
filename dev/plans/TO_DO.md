# statgpu TO DO

> Compact execution queue and mandatory completion checklist.  
> Canonical roadmap: [`ROADMAP.md`](ROADMAP.md)  
> Issue index: [`ISSUES.md`](ISSUES.md)  
> Development guide: [`../AGENTS.md`](../AGENTS.md)  
> Hard automation protocol: [`.claude/workflows/new-module-dev.md`](../../.claude/workflows/new-module-dev.md)  
> Last synchronized: **2026-08-06**, release **0.2.4**, commit `0aeeb95b60e3e274053b8f1b6427ae50c8eec015`.

This file is intentionally shorter than `dev/AGENTS.md` and the `.claude` workflows, but it is not a weaker checklist. When wording conflicts, the applicable `.claude` workflow/skill takes precedence, followed by `dev/AGENTS.md`. `ROADMAP.md` controls priority; GitHub issues control executable scope. Roadmap and issue scope may narrow work but may not weaken the hard gates.

## 1. Required task classification

Before implementation, classify the touched impact axes and record which gates are active:

- public API;
- backend, dtype, device, memory ownership, or fallback;
- loss, penalty, solver, or loss × penalty capability;
- cross-validation;
- inference;
- formula/model-matrix semantics;
- benchmark or performance;
- documentation-only.

Choose the broader classification when uncertain. Documentation-only work does not activate runtime gates unless it changes a support or performance claim.

Every development report must end with exactly one workflow status:

- `COMPLETE` — all active local blocking gates pass and required docs/artifacts are current;
- `PARTIAL_REMOTE_PENDING` — local work is complete, but specified physical-GPU, R/external, or large-benchmark evidence is unavailable;
- `BLOCKED_NEEDS_USER_APPROVAL` — continuation requires an explicit decision such as a backend deferral, API break, performance caveat, commit, push, merge, release, or publication;
- `FAILED` — a blocking correctness, backend, formula, precision, convergence, fallback, review, or artifact gate remains unresolved.

Do not close work as “mostly complete” or treat `planned` as a completion status.

## 2. Non-negotiable development gates

### 2.1 Public contract

- [ ] Define inputs, outputs, shapes, dtype/device behavior, errors, fallback behavior, statistical parameterization, and explicit non-goals before final implementation.
- [ ] Preserve sklearn-style constructor identity, `get_params` / `set_params`, cloning, fitted-state invalidation, pipeline, and CV behavior where applicable.
- [ ] User-visible unsupported combinations fail early and precisely; they do not optimize an incomplete objective or change behavior silently.

### 2.2 Three backends and device locality

- [ ] Every new or materially changed statistical method implements NumPy, CuPy, and Torch; CPU-only work is incomplete.
- [ ] A backend deferral requires explicit user approval plus the reason, user-visible failure behavior, deterministic skip condition, and follow-up issue.
- [ ] Explicit `device="cuda"` and `device="torch"` never silently fall back to CPU or another backend; only `device="auto"` may select a backend automatically.
- [ ] Core fitting, prediction, scoring, inference, and validation remain on the selected backend; no hidden full-array GPU-to-CPU transfer is introduced.
- [ ] Fallback, approximate inference, dtype conversion, or device conversion is part of the public contract and is visible through an error, warning, result field, or report.
- [ ] GPU-buffer-owning estimators implement the documented `gpu_memory_cleanup` lifecycle, including cleanup methods and finalization behavior without discarding fit state prematurely.

### 2.3 Reuse and architecture

- [ ] Reuse `BaseEstimator`, `statgpu/backends/`, existing array helpers, solver/penalty registries, `statgpu/cross_validation/`, formula infrastructure, and `statgpu/inference/` before adding private parallel implementations.
- [ ] Model modules do not scatter direct CuPy imports or duplicate backend selection and conversion logic without a documented architectural reason.
- [ ] New inference distribution, p-value, or interval logic checks existing backend-aware inference utilities before adding another implementation.

### 2.4 Direct fit and CV closure

- [ ] Every public tunable loss × penalty capability supported by direct `fit()` also supports the CV layer: path/grid generation, deterministic folds, fold scoring, best-parameter selection, and final refit.
- [ ] CV preserves the declared loss, weighting, backend, device, dtype, formula alignment, and objective normalization.
- [ ] A capability may omit CV only when it is explicitly non-tunable or the user approves a deferral with failure behavior, tests, docs, and a follow-up issue.
- [ ] Do not advertise a partially completed penalized module: when a roadmap package declares a penalty matrix, direct fit and CV must close for the whole declared matrix before the package is marked complete.

### 2.5 Inference contract

- [ ] A model family that exposes `compute_inference`, `summary()`, covariance, SE, p-values, or confidence intervals implements inference or is explicitly documented and tested as estimation-only.
- [ ] Strict inference is the default path; strict failure raises by default. Approximate or downgraded inference requires explicit opt-in and visible status.
- [ ] Inference outputs remain consistent across supported backends, including applicable `coef`, `bse`, `t/z`, `p`, confidence intervals, `AIC`, `BIC`, and `LLF` fields.
- [ ] Current default external-alignment thresholds are recorded where applicable: coefficient error `<= 1e-6`, BSE error `<= 1e-3`, and p-value error `<= 5e-2`; a different tolerance requires a statistical or numerical justification.
- [ ] Direct-fit and final-CV-refit inference use the same declared estimator contract; fold models remain estimation-only only when that behavior is intentional and tested.

### 2.6 Formula contract

- [ ] Formula-facing methods test intercept handling, categorical reference levels, interactions, transforms, missing-data row alignment, feature names, and prediction column order.
- [ ] Array and formula paths agree after model-matrix alignment.
- [ ] R-style/Patsy semantics are externally checked where applicable; unsupported syntax has a precise error and documented boundary.

### 2.7 Objective, penalty, precision, and convergence

- [ ] State whether the objective uses a sum or average loss and whether the intercept is penalized.
- [ ] Map external regularization scales explicitly; for example, use `lambda_external = n * lambda` when comparing average-loss statgpu objectives with summed-loss references.
- [ ] Do not alter the statgpu objective merely to force agreement with an external package.
- [ ] Validate loss value, gradient, Hessian or Hessian-vector behavior, proximal/KKT conditions, line search, stopping rules, and convergence status for the active component matrix.
- [ ] Precision and convergence are blocking before performance optimization.
- [ ] Numeric recovery catches only recognized numerical-domain or rank failures; OOM, device, shape, index, contract, and programming errors remain fatal.

### 2.8 External and architecture-specific validation

- [ ] Use the strongest available baseline: analytic/derivative check, trusted statgpu path, Python reference, R reference, then documented numerical invariants.
- [ ] External comparisons align feature sets, weights, ties, solver, penalty, objective normalization, `alpha` / `C`, `max_iter`, and `tol`.
- [ ] Prefer statsmodels for statistical inference, sklearn for estimator/prediction behavior, and authoritative R packages for key statistical definitions.
- [ ] Activate the relevant architecture matrix: loss, penalty, solver, direct-fit/CV, inference, formula, backend helper, survival, or nonparametric/unsupervised tests.
- [ ] Broad cross-axis changes extend a maintained matrix test rather than relying only on isolated smoke tests.

### 2.9 Testing, review, and validation tier

- [ ] Run applicable lint, type, unit, regression, compatibility, formula, external-alignment, and import-order tests.
- [ ] Add deterministic NumPy/CuPy/Torch parity tests and explicit unavailable-backend errors/skips.
- [ ] Complete maintained physical CuPy and Torch validation for a `COMPLETE` claim when those paths are active; otherwise report `PARTIAL_REMOTE_PENDING` with exact commands and missing resources.
- [ ] Record the highest completed validation tier: `local-minimal`, `local-full`, or `remote-full`.
- [ ] Run code review and fix cycles until no unresolved CRITICAL or HIGH issue remains; remaining medium findings require a documented behavior boundary or follow-up issue.
- [ ] Tests must independently calculate expected statistical values where feasible rather than only comparing one statgpu path with another.

### 2.10 Performance and evidence artifacts

- [ ] Performance work starts only after correctness, precision, and convergence gates pass.
- [ ] GPU timing synchronizes the correct CuPy/Torch backend before and after each measured region.
- [ ] Record target scale, data shape, dtype, hardware, software environment, timing scope, transfer policy, repeats, seeds, and comparison identity.
- [ ] Store machine-readable benchmark evidence under `results/*.json` and a concise audit summary; do not support public claims with rounded prose alone.
- [ ] Do not claim universal GPU acceleration; report measured crossover and slower regimes.
- [ ] Benchmark and remote evidence must be provenance-bearing and reproducible, with source hashes or equivalent source identity where the workflow requires them.

### 2.11 Documentation and release surface

- [ ] Update exports, README/USAGE where applicable, model pages, compatibility matrices, and changelogs in the same feature change.
- [ ] Follow EN-first/CN-follow: update `docs/en/` and English entry points, then synchronize `docs/cn/` and Chinese entry points.
- [ ] Keep root `CHANGELOG.md`, `docs/en/changelog.md`, and `docs/cn/changelog.md` consistent with the actual capability and validation evidence.
- [ ] Model documentation includes applicable objective, estimating equation, covariance/inference, parameters, CPU/CuPy/Torch examples, strict/approx behavior, outputs, FAQ, external validation, and references.
- [ ] Remote or benchmark claims cite auditable artifact paths rather than only verbal conclusions.

### 2.12 Required completion report

- [ ] Report impact classification, workflow status, validation tier, files changed, backend matrix, CV status, inference status, formula status, objective/penalty mapping, precision/convergence evidence, tests, external baselines, physical-GPU evidence, benchmark artifacts, review outcome, documentation changes, and any pending remote commands.
- [ ] Commits, pushes, PR creation, merges, tags, releases, and package publication occur only after an explicit user request.
- [ ] Credentials are never read from tracked Markdown or settings files; remote execution uses the maintained untracked/environment configuration path.

## 3. Active execution queue

### P0 — planning and integration

- [ ] Merge roadmap reconciliation PR #89 and use `ROADMAP.md` as the only current priority source.
- [ ] #90 — synchronize benchmark dashboard PR #76 with current `master` without adding new benchmark families in the same change.
- [ ] Regenerate and validate dashboard data, inventory, parse report, and deployed assets during #90.

### P1 — benchmark evidence and dashboard readiness

- [ ] #91 — add a canonical CV benchmark source covering RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV, PenalizedGLM_CV, and CoxPHCV.
- [ ] Record folds, grid/path size, warm starts, CV time, final-refit time, selected parameter, score, failures, synchronization, and timing scope.
- [ ] #92 — complete production-path browser QA, cross-browser smoke, accessibility checks, and documentation navigation before proposing PR #76 for `master` integration.

### P1 — panel workflow completion

- [ ] #93 — refactor panel models onto a shared base and covariance registry while preserving numerical behavior.
- [ ] Add Hausman, pooling F, and Breusch-Pagan LM tests.
- [ ] Add within/between/overall/adjusted R-squared and model F-statistics.
- [ ] Add robust RandomEffects covariance, HC0/HC2/HC3 where defined, and Driscoll-Kraay covariance.
- [ ] Validate against `linearmodels`, R `plm`, and aligned sandwich covariance references.

### P2 — survival foundations

- [ ] #94 — implement Kaplan-Meier and Nelson-Aalen estimators with variance, confidence intervals, grouped output, and external alignment.
- [ ] #95 — implement Weibull, log-normal, and log-logistic AFT models with three backends, model-based inference, formula support, and prediction functions.

### P2 — multinomial and sparse foundations

- [ ] #96 — define and implement the unpenalized-only multinomial/softmax estimator, including identifiability, shapes, inference, formula semantics, and three-backend parity.
- [ ] #96 must expose no regularization parameter or penalized solver; it is non-tunable and therefore introduces no multinomial CV surface.
- [ ] #98 — after #96, implement the complete penalized multinomial suite as one capability package.
- [ ] #98 must cover at least L2, L1, ElasticNet, SCAD, and MCP across NumPy, CuPy, and Torch.
- [ ] #98 must close direct fit, path/grid, deterministic CV, selection, final refit, supported inference, external alignment, physical-GPU validation, and EN/CN docs for every declared penalty before completion.
- [ ] #97 — define a shared SciPy/CuPy/Torch sparse-input contract with no silent densification.

### P3 — feature-driven technical debt

- [ ] Split `_penalized_cv.py` by candidate generation, fold execution, selection, and final refit when #91 supplies regression coverage.
- [ ] Split long FISTA-family solver functions into bounded numerical components without changing objective or stopping contracts.
- [ ] Unify duplicated array-copy and scalar-extraction helpers.
- [ ] Reduce backend duplication only where device behavior remains explicit and fully tested.

## 4. Deferred

The following are not immediate priorities: Panel IV, high-dimensional fixed effects, DID/event study, dynamic-panel GMM, frailty, Fine-Gray, multi-state survival, mixed models, GEE, meta-analysis, changepoints, copulas, multiple imputation, nonlinear least squares, and broad new unsupervised families.

Promote a deferred item only through a scoped GitHub issue satisfying `ROADMAP.md`, this checklist, `dev/AGENTS.md`, and the applicable `.claude` workflow/skill.
