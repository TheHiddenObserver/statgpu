# statgpu TO DO

> Compact execution queue and mandatory completion checklist.  
> Canonical roadmap: [`ROADMAP.md`](ROADMAP.md)  
> Issue index: [`ISSUES.md`](ISSUES.md)  
> Development guide: [`../AGENTS.md`](../AGENTS.md)  
> Hard automation protocol: [`.claude/workflows/new-module-dev.md`](../../.claude/workflows/new-module-dev.md)  
> Last synchronized: **2026-08-28**, release **0.2.5**, commit `84f8bc7e17f66466b3a325cbb007b6cb41843821`.

This file is intentionally shorter than `dev/AGENTS.md` and the `.claude` workflows, but it is not a weaker checklist. When wording conflicts, the applicable `.claude` workflow/skill takes precedence, followed by `dev/AGENTS.md`. `ROADMAP.md` controls priority; GitHub issues control executable scope. Roadmap and issue scope may narrow work but may not weaken hard gates.

## 1. Required task classification

Before implementation, classify touched impact axes and record active gates:

- public API;
- backend, dtype, device, memory ownership, or fallback;
- loss, penalty, solver, or loss x penalty capability;
- cross-validation;
- inference;
- formula/model-matrix semantics;
- benchmark or performance;
- documentation-only.

Choose the broader classification when uncertain. Documentation-only work does not activate runtime gates unless it changes a support, evidence, or performance claim.

Every development report ends with exactly one workflow status:

- `COMPLETE` — all active blocking gates pass and required docs/artifacts are current;
- `PARTIAL_REMOTE_PENDING` — local work is complete, but specified physical-GPU, R/external, or large-benchmark evidence is unavailable;
- `BLOCKED_NEEDS_USER_APPROVAL` — continuation requires an explicit decision such as backend deferral, API break, performance caveat, commit/push/PR/merge/release/publication;
- `FAILED` — a blocking correctness, backend, formula, precision, convergence, fallback, review, or artifact gate remains unresolved.

Do not use “mostly complete” or `planned` as a completion status.

## 2. Non-negotiable development gates

### 2.1 Public contract

- [ ] Define inputs, outputs, shapes, dtype/device behavior, errors, fallback behavior, statistical parameterization, and explicit non-goals before final implementation.
- [ ] Preserve sklearn-style constructor identity, `get_params` / `set_params`, cloning, fitted-state invalidation, pipeline, and CV behavior where applicable.
- [ ] Unsupported user-visible combinations fail early and precisely; they do not optimize an incomplete objective or change behavior silently.

### 2.2 Three backends and device locality

- [ ] Every new or materially changed statistical method implements NumPy, CuPy, and Torch; CPU-only work is incomplete.
- [ ] A backend deferral requires explicit user approval plus reason, user-visible failure behavior, deterministic skip condition, and follow-up issue.
- [ ] Explicit `device="cuda"` and `device="torch"` never silently fall back to CPU/another backend; only `device="auto"` may select automatically.
- [ ] Core fitting, prediction, scoring, inference, and validation remain on the selected backend; no hidden full-array GPU-to-CPU transfer is introduced.
- [ ] Fallback, approximate inference, dtype conversion, or device conversion is part of the public contract and visible through an error, warning, result field, or report.
- [ ] GPU-buffer-owning estimators follow the documented `gpu_memory_cleanup` lifecycle without discarding needed fit state prematurely.

### 2.3 Reuse and architecture

- [ ] Reuse `BaseEstimator`, `statgpu/backends/`, existing array helpers, solver/penalty registries, `statgpu/cross_validation/`, formula infrastructure, and `statgpu/inference/` before adding private parallel implementations.
- [ ] Model modules do not scatter direct backend imports or duplicate backend selection/conversion without a documented reason.
- [ ] New reference-distribution, p-value, or interval logic checks existing backend-aware inference utilities first.

### 2.4 Direct fit and CV closure

- [ ] Every public tunable loss x penalty capability supported by direct `fit()` also supports path/grid generation, deterministic folds, fold scoring, best-parameter selection, and final refit.
- [ ] CV preserves loss, weighting, backend, device, dtype, formula alignment, and objective normalization.
- [ ] CV may be omitted only for an explicitly non-tunable capability or an approved deferral with failure behavior, tests, docs, and follow-up issue.
- [ ] A declared penalty matrix is not complete until direct fit and CV close for the entire declared matrix.

### 2.5 Inference contract

- [ ] A family exposing inference/summary/covariance/SE/p-values/CI implements inference or is explicitly documented/tested as estimation-only.
- [ ] Strict inference is the default; downgrade/approximation requires explicit opt-in and visible status.
- [ ] Inference outputs remain consistent across supported backends, including applicable coefficient, BSE, t/z, p, CI, AIC, BIC, and LLF fields.
- [ ] Default external-alignment thresholds remain explicit where applicable: coefficient error `<= 1e-6`, BSE error `<= 1e-3`, p-value error `<= 5e-2`; deviations require numerical/statistical justification.
- [ ] Direct-fit and final-CV-refit inference use the same declared estimator contract.

### 2.6 Formula contract

- [ ] Formula-facing methods test intercept handling, categorical reference levels, interactions/transforms, missing-data row alignment, feature names, and prediction column order.
- [ ] Array and formula paths agree after model-matrix alignment.
- [ ] R-style/Patsy semantics are externally checked where applicable; unsupported syntax has precise failure behavior.

### 2.7 Objective, penalty, precision, and convergence

- [ ] State sum/average objective normalization and intercept-penalty policy.
- [ ] Map external regularization scales explicitly instead of changing the statgpu objective to force agreement.
- [ ] Validate active loss/gradient/Hessian/prox/KKT/line-search/stopping/convergence behavior.
- [ ] Precision and convergence are blocking before performance optimization.
- [ ] Numerical recovery catches only recognized numerical-domain/rank failures; OOM, device, shape, index, contract, and programming errors remain fatal.

### 2.8 External and architecture-specific validation

- [ ] Use the strongest available baseline: analytic check, trusted statgpu path, Python reference, R reference, then documented numerical invariants.
- [ ] Align feature sets, weights, ties, solver, penalty, normalization, hyperparameters, and tolerances.
- [ ] Prefer statsmodels for statistical inference, sklearn for estimator/prediction behavior, and authoritative R packages for key statistical definitions.
- [ ] Activate and extend the relevant architecture matrix rather than relying only on isolated smoke tests.

### 2.9 Testing, review, and validation tier

- [ ] Run applicable lint/type/unit/regression/compatibility/formula/external-alignment/import-order tests.
- [ ] Add deterministic NumPy/CuPy/Torch parity and unavailable-backend error/skip tests.
- [ ] Complete maintained physical CuPy/Torch validation for a `COMPLETE` claim when those paths are active; otherwise use `PARTIAL_REMOTE_PENDING` with exact missing commands/resources.
- [ ] Record highest completed tier: `local-minimal`, `local-full`, or `remote-full`.
- [ ] Run review/fix cycles until no unresolved CRITICAL or HIGH remains; relevant actionable MEDIUM must be fixed or explicitly bounded as a non-blocking follow-up.
- [ ] Independently calculate expected statistical values where feasible rather than only comparing one statgpu path with another.

### 2.10 Performance and evidence artifacts

- [ ] Performance work starts only after correctness/precision/convergence gates pass.
- [ ] GPU timing synchronizes the correct CuPy/Torch backend around each measured region.
- [ ] Record target scale, shape, dtype, hardware/software, timing scope, transfer policy, repeats, seeds, and comparison identity.
- [ ] Store machine-readable evidence under `results/*.json`; public claims do not rely only on rounded prose.
- [ ] Do not claim universal GPU acceleration; report measured crossover/slower regimes.
- [ ] Remote/benchmark evidence is provenance-bearing and reproducible.
- [ ] Canonical physical evidence is tied to both numerical source and validator acceptance contract; changing validator acceptance logic after a run requires rerunning affected evidence.

### 2.11 Documentation and release surface

- [ ] Update exports/README/USAGE/model pages/compatibility matrices/changelogs where applicable.
- [ ] Follow EN-first/CN-follow.
- [ ] Keep root, EN, and CN changelog capability/evidence claims consistent.
- [ ] Model docs include applicable objective, estimating equation, covariance/inference, parameters, backend examples, strict/approx behavior, outputs, FAQ, external validation, and references.
- [ ] Remote/benchmark claims cite auditable artifact paths.

### 2.12 Required completion report

- [ ] Report impact classification, workflow status, validation tier, files changed, backend matrix, CV/inference/formula status, objective/penalty mapping, precision/convergence evidence, tests, external baselines, physical-GPU evidence, artifacts, review outcome, docs, and pending remote commands.
- [ ] Commits, pushes, PR creation, merges, tags, releases, and package publication occur only after explicit user request.
- [ ] Credentials are never read from tracked Markdown/settings; remote execution uses maintained untracked/environment configuration.

## 3. Active execution queue

### P0 — post-0.2.5 planning and issue hygiene

- [ ] Keep planning files synchronized to release 0.2.5 / `84f8bc7e17f66466b3a325cbb007b6cb41843821`.
- [ ] Audit #93 against Stage A/B/C, #126, and v0.2.5 evidence; do not reopen delivered Panel numerical scope solely because the issue remains open.
- [ ] If #93 has no demonstrated missing acceptance criterion, close/reclassify it through issue/evidence reconciliation rather than another Panel implementation PR.

### P1 — current correctness/inference implementation

- [ ] #127 — execute [`gaussian_inference_backend_native_plan.md`](gaussian_inference_backend_native_plan.md).
- [ ] Inventory the exact Gaussian inference consumer/data-lifecycle graph before production edits.
- [ ] Keep numerical covariance/distribution inference backend-native through completion; only the final reporting snapshot may convert full results to NumPy.
- [ ] Preserve nonrobust, HC0-HC3, HAC, Ridge/L2, weighting, rank, multi-target, formula, sklearn, and applicable CV final-refit behavior.
- [ ] Prove actual fit/inference backend and concrete device; input type alone is insufficient.
- [ ] Freeze/review the physical validator before canonical GPU evidence; rerun evidence if its acceptance contract changes afterward.

### P1 — evidence after #127

- [ ] #105 — systematic linear/GLM inference benchmark and validation coverage after #127 stabilizes backend-native inference.
- [ ] #108 — extend canonical Panel estimator/covariance evidence for the released 0.2.5 capability.

### P2 — survival foundations

- [ ] #94 — implement Kaplan-Meier and Nelson-Aalen with variance, CI, grouped output, external alignment, and three-backend parity.
- [ ] #95 — after #94 unless isolated resources justify parallel work, implement Weibull/log-normal/log-logistic AFT with inference, formula, prediction, and three backends.

### P2 — multinomial and sparse foundations

- [ ] #96 — implement the unpenalized-only multinomial/softmax contract; expose no penalty or regularization parameter and no multinomial CV surface.
- [ ] #98 — after #96, close L2/L1/ElasticNet/SCAD/MCP direct-fit + CV + final-refit + supported inference + physical-GPU + docs as one declared capability package.
- [ ] #97 — define the shared SciPy/CuPy/Torch sparse-input contract with no silent densification.

### P3 — benchmark breadth and bounded hardening

- [ ] #101-#104, #106, #107, #109 — expand canonical benchmark breadth without fabricating missing measurements or displacing higher-priority correctness work.
- [ ] #114 — dashboard bundle/DOM optimization only under its measurement-first contract.
- [ ] #117 — clarify input versus working dtype provenance for mixed-precision benchmark sources.
- [ ] #118 — measure/bound GPU CV path-buffer memory before changing the current backend-native buffering design.

### P3 — feature-driven technical debt

- [ ] Split `_penalized_cv.py` only with regression coverage preserving candidate/fold/selection/refit semantics.
- [ ] Split long FISTA-family solvers without changing objective/stopping contracts.
- [ ] Unify array-copy/scalar-extraction helpers only where device behavior remains explicit and tested.
- [ ] Do not create a repository-wide backend/solver/inference unification PR.

## 4. Deferred

Not immediate priorities: Panel IV/HDFE/DID/dynamic-panel GMM, frailty/Fine-Gray/multi-state survival, mixed models, GEE, meta-analysis, changepoints, copulas, multiple imputation, nonlinear least squares, and broad new unsupervised families.

Promote a deferred item only through a scoped issue satisfying `ROADMAP.md`, this checklist, `dev/AGENTS.md`, and the applicable `.claude` workflow/skill.
