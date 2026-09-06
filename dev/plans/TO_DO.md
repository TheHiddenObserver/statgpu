# statgpu TO DO

> Compact execution queue and mandatory completion checklist.  
> Canonical roadmap: [`ROADMAP.md`](ROADMAP.md)  
> Issue index: [`ISSUES.md`](ISSUES.md)  
> Development guide: [`../AGENTS.md`](../AGENTS.md)  
> Canonical automation skill: [`.claude/skills/new-module-dev/SKILL.md`](../../.claude/skills/new-module-dev/SKILL.md)  
> Canonical review skill: [`.claude/skills/code-review/SKILL.md`](../../.claude/skills/code-review/SKILL.md)  
> Last synchronized: **2026-08-28**, release **0.2.5**, commit `84f8bc7e17f66466b3a325cbb007b6cb41843821`.

This file is intentionally shorter than `dev/AGENTS.md` and the canonical `.claude/skills/*/SKILL.md` protocols, but it is not a weaker checklist. Impact classification determines which gates are active. When wording conflicts, the applicable canonical skill takes precedence, followed by `dev/AGENTS.md`. Legacy flat `.claude/skills/*.md` and `.claude/workflows/new-module-dev.md` files are compatibility pointers only. `ROADMAP.md` controls priority; GitHub issues control executable scope.

Impact-driven scope does not allow new capability to define away repository defaults: new shared numerical/statistical capability closes NumPy/CuPy/Torch by default, and new tunable loss x penalty capability closes direct fit + CV by default, unless a legitimate narrower contract is already part of the task or an explicit deferral is approved. The blocking fork contract of the canonical `code-review` skill requires Claude Code >= 2.1.218.

## 1. Required task classification

Before implementation, classify touched impact axes and record active gates:

- public API / backward compatibility / deprecation;
- backend, dtype, device, memory ownership, or fallback;
- loss, penalty, solver, or loss x penalty capability;
- cross-validation;
- inference;
- formula/model-matrix semantics;
- benchmark or performance;
- documentation/evidence-only.

Choose the broader **plausibly affected** classification when uncertain, but do not activate unrelated gates solely because the edited class already exposes them. Documentation-only work does not activate runtime gates unless it changes a support, evidence, statistical, or performance claim. Conversely, a repository-default gate for genuinely new capability is not inactive merely because a first draft omits it.

Every development report ends with exactly one workflow status:

- `COMPLETE` — all active/default local blocking gates pass and required docs/artifacts are current;
- `PARTIAL_REMOTE_PENDING` — local work is complete, but specified physical-GPU, R/external, or large-benchmark evidence is unavailable;
- `BLOCKED_NEEDS_USER_APPROVAL` — continuation requires an explicit decision such as backend/CV deferral, an unrequested API break, accepted performance caveat, merge/release/publication, or credentials;
- `FAILED` — an active/default blocking correctness, compatibility, backend, CV, formula, precision, convergence, fallback, review, or artifact gate remains unresolved.

Do not use “mostly complete” or `planned` as a completion status. Do not call a genuinely inactive gate incomplete, and do not call a repository-default capability complete by silently shrinking its scope.

## 2. Non-negotiable development gates

### 2.1 Public contract and migration

- [ ] Define inputs, outputs, shapes, dtype/device behavior, errors, fallback behavior, statistical parameterization, and explicit non-goals before final implementation.
- [ ] Preserve sklearn-style constructor identity, `get_params` / `set_params`, cloning, fitted-state invalidation, pipeline, and CV behavior where the estimator claims those contracts.
- [ ] Public API migrations/deprecations cover omitted versus explicitly supplied historical defaults, warning category/message/stacklevel, old/new conflict handling, internal clone/reconstruction warning noise, legacy behavior during the compatibility window, migration docs, and planned removal boundary.
- [ ] Unsupported user-visible combinations fail early and precisely; they do not optimize an incomplete objective or change behavior silently.

### 2.2 Backends and device locality

- [ ] Every **new or materially changed shared numerical/statistical capability** implements/tests **NumPy, CuPy, and Torch by default**. A narrower backend contract is valid only when it is already a legitimate task/issue scope or an explicit deferral is approved with reason, user-visible failure behavior, deterministic test/skip condition, and follow-up scope.
- [ ] CPU-only implementation may be an intermediate step, but it is not `COMPLETE` for a new shared capability whose repository-default backend contract remains three-backend.
- [ ] API-only/deprecation-only/docs-only/narrow refactor work does not require new backend implementation when numerical dispatch is unchanged; instead verify existing backend support and routing are preserved where the interface can affect them.
- [ ] Explicit `device="cuda"` and `device="torch"` never silently fall back to CPU/another backend; only `device="auto"` may select automatically.
- [ ] Active core fitting, prediction, scoring, inference, and validation remain on the selected backend according to the declared contract; no hidden full-array GPU-to-CPU transfer is introduced.
- [ ] Fallback, approximate inference, dtype conversion, or device conversion is part of the public contract and visible through an error, warning, result field, or report.
- [ ] GPU-buffer-owning estimators follow their documented `gpu_memory_cleanup` lifecycle without discarding needed fitted state prematurely.

### 2.3 Reuse and architecture

- [ ] Reuse `BaseEstimator`, `statgpu/backends/`, existing array helpers, solver/penalty registries, `statgpu/cross_validation/`, formula infrastructure, and `statgpu/inference/` before adding private parallel implementations.
- [ ] Model modules do not scatter direct backend imports or duplicate backend selection/conversion without a documented kernel/device-specific reason.
- [ ] New reference-distribution, p-value, or interval logic checks existing backend-aware inference utilities first.

### 2.4 Direct fit and CV closure

- [ ] Every **new tunable loss x penalty capability** closes **direct fit + CV path/grid/folds/scoring/selection/final refit by default**.
- [ ] CV may be omitted only when the new capability is genuinely non-tunable or an explicit CV deferral is approved with failure behavior, tests, docs, and follow-up scope; merely omitting a CV claim does not make the default gate inactive.
- [ ] CV preserves loss, weighting, backend, device, dtype, formula alignment, objective normalization, and stage-specific solver semantics.
- [ ] A direct-estimator API cleanup that does not alter CV semantics does not automatically create new CV scope; add targeted CV regression only when the API can affect CV dispatch/refit/clone behavior.
- [ ] A declared tunable loss x penalty matrix is not complete until the direct and CV capability promised by the repository contract close for the declared surface.

### 2.5 Inference contract

- [ ] Inference is an active blocking gate when the API/docs claim it or the change touches `compute_inference`, `summary()`, covariance, SE, p-values, CI, inference results, or inference backend behavior.
- [ ] A prediction/estimation estimator with no inference public contract may legitimately remain estimation-only; do not require inference solely because an external package offers it.
- [ ] When inference is supported, outputs remain semantically consistent across declared backends, including applicable coefficient, BSE, t/z, p, CI, AIC, BIC, and LLF fields.
- [ ] Strict/fallback behavior follows the estimator's explicit public contract; downgrade/approximation never occurs silently.
- [ ] Historical external-alignment thresholds are applied only where the maintained test/validator names them; deviations elsewhere use method-specific numerical/statistical justification rather than a universal threshold.
- [ ] Direct-fit and final-CV-refit inference use the declared stage contract when both are supported.

### 2.6 Formula contract

- [ ] Formula-facing changes test intercept handling, categorical reference levels, interactions/transforms, missing-data row alignment, feature names, and prediction column order.
- [ ] Array and formula paths agree after model-matrix alignment when both are supported.
- [ ] R-style/Patsy semantics are externally checked where applicable; unsupported syntax has precise failure behavior.

### 2.7 Objective, penalty, precision, and convergence

- [ ] State sum/average objective normalization and intercept-penalty policy for active numerical changes/comparisons.
- [ ] Map external regularization scales explicitly instead of changing the statgpu objective to force agreement.
- [ ] Validate active loss/gradient/Hessian/prox/KKT/line-search/stopping/convergence behavior.
- [ ] Precision and convergence are blocking before performance optimization.
- [ ] Numerical recovery catches only recognized numerical-domain/rank failures; OOM, device, shape, index, contract, and programming errors remain fatal.

### 2.8 External and architecture-specific validation

- [ ] Use the strongest available baseline for the active contract: analytic check, trusted statgpu path, Python reference, R reference, then documented numerical invariants.
- [ ] Align feature sets, weights, ties, solver, penalty, normalization, hyperparameters, and tolerances.
- [ ] Prefer statsmodels when inference alignment is active, sklearn when estimator/prediction compatibility is active, and authoritative R packages for key statistical definitions when needed.
- [ ] Extend the relevant architecture matrix rather than relying only on isolated smoke tests; do not run unrelated matrices for inactive axes.

### 2.9 Testing, review, and validation tier

- [ ] Run applicable lint/type/unit/regression/compatibility/formula/external-alignment/import-order tests proportional to the change's blast radius.
- [ ] Add NumPy/CuPy/Torch parity and unavailable-backend behavior for new shared numerical capability; for existing narrow changes, test only affected backends plus preservation regressions proportional to blast radius.
- [ ] Complete maintained physical CuPy/Torch validation for a `COMPLETE` claim when the active/default change requires physical-GPU evidence; otherwise do not convert skipped GPU tests into evidence.
- [ ] Record highest completed evidence tier where useful: `local-minimal`, `local-full`, or `remote-full`.
- [ ] Run a fresh canonical `code-review` pass until no unresolved CRITICAL or HIGH remains; relevant actionable MEDIUM must be fixed or explicitly bounded as non-blocking follow-up.
- [ ] Independently calculate expected statistical values where feasible rather than only comparing one statgpu path with another.

### 2.10 Performance and evidence artifacts

- [ ] Performance work starts only after correctness/precision/convergence gates pass.
- [ ] GPU timing synchronizes the correct CuPy/Torch backend around each measured region.
- [ ] Record exact source, clean/dirty state, environment identity, target scale, shape, dtype, hardware/software, timing scope, transfer policy, repeats, seeds, and comparison identity.
- [ ] Store machine-readable evidence under `results/*.json` when the result is meant to be retained/published; public claims do not rely only on rounded prose.
- [ ] Keep materially different hardware/software environment groups separate; do not aggregate their speedups as one homogeneous measurement.
- [ ] Do not claim universal GPU acceleration; report measured crossover/slower regimes.
- [ ] Canonical physical evidence is tied to both numerical source and validator acceptance contract; changing validator acceptance logic after a run requires rerunning affected evidence before claiming the new contract is proven.

### 2.11 Documentation and release surface

- [ ] Update exports/README/USAGE/model pages/compatibility matrices/changelogs only where applicable to the changed capability.
- [ ] Follow EN-first/CN-follow when both maintained language surfaces are affected, and keep capability claims conceptually aligned.
- [ ] Learner-facing model pages lead with problem/motivation/intuition/use guidance, include a self-contained example and interpretation, and keep advanced solver/backend details later.
- [ ] `Key parameters` teaching tables may be selective, but the public API reference/inventory must be complete or link to the canonical complete API reference.
- [ ] Reference/implementation pages may remain reference-first.
- [ ] Remote/benchmark claims cite auditable source/environment/artifact provenance.

### 2.12 Required completion report and repository actions

- [ ] Report impact classification, workflow status, files changed, repository-default gates and approved exceptions, active backend/CV/inference/formula status, API compatibility decision, objective/penalty mapping and precision/convergence evidence when active, tests, benchmarks/evidence, review outcome, docs, and pending remote commands.
- [ ] Task-scoped implementation authorizes relevant file edits and validation. Commit/push/PR actions occur only when explicitly requested or when the active task already includes an approved branch/PR workflow.
- [ ] Merge, tag, release, package publication, credential setup, or an unrequested breaking API decision requires explicit user direction.
- [ ] Credentials are never read from tracked Markdown/settings; remote execution uses maintained untracked/environment configuration.

## 3. Active execution queue

### P0 — post-0.2.5 planning and issue hygiene

- [ ] Keep planning files synchronized to release 0.2.5 / `84f8bc7e17f66466b3a325cbb007b6cb41843821`.
- [ ] Audit #93 against Stage A/B/C and #126 implementation evidence; do not reopen delivered Panel numerical scope solely because the issue remains open.
- [ ] During #93/#108 evidence work, keep the PR #128 release-validator provenance caveat explicit: `results/pr126_release_697de113/` predates the final maintained runner contract and is historical numerical-source evidence, not proof of that later validator contract.
- [ ] If #93 has no demonstrated missing production acceptance criterion, close/reclassify it through issue/evidence reconciliation rather than another Panel implementation PR.

### P1 — current correctness/inference implementation

- [ ] #127 — execute [`gaussian_inference_backend_native_plan.md`](gaussian_inference_backend_native_plan.md).
- [ ] Inventory the exact Gaussian inference consumer/data-lifecycle graph before production edits.
- [ ] Keep numerical covariance/distribution inference backend-native through completion; only the final reporting snapshot may convert full results to NumPy.
- [ ] Preserve nonrobust, HC0-HC3, HAC, Ridge/L2, weighting, rank, multi-target, formula, sklearn, and applicable CV final-refit behavior.
- [ ] Prove actual fit/inference backend and concrete device; input type alone is insufficient.
- [ ] Freeze/review the physical validator before canonical GPU evidence; rerun evidence if its acceptance contract changes afterward.

### P1 — evidence after #127

- [ ] #105 — systematic linear/GLM inference benchmark and validation coverage after #127 stabilizes backend-native inference.
- [ ] #108 — extend canonical Panel estimator/covariance evidence for the released 0.2.5 capability using fresh or contract-matched provenance rather than automatic reuse of the disputed final-release validator artifacts.

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

Promote a deferred item only through a scoped issue satisfying `ROADMAP.md`, this checklist, `dev/AGENTS.md`, and the applicable canonical `.claude/skills/<skill-name>/SKILL.md` protocol.
