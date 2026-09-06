---
name: new-module-dev
description: Develop or substantially change a statgpu statistical method, estimator, loss, penalty, solver, inference path, CV path, or backend-aware numerical API with impact-based gates, targeted tests, documentation, and an independent review pass. Use for real implementation work, not for small docs edits or narrow API cleanups that do not change numerical capability.
when_to_use: Trigger for adding a new method or materially changing statistical/numerical behavior. Do not trigger for ordinary wording edits, isolated deprecations, or narrow refactors unless they also change numerical capability.
argument-hint: "[module-or-task]"
---

# statgpu New Module Development

Develop `$ARGUMENTS` using the smallest set of gates that proves the changed public contract **without weakening statgpu's default capability closure for genuinely new numerical/statistical features**.

Read [workflow.md](workflow.md) before broad implementation, when more than one impact axis is active, or when a completion/blocking decision is needed.

## 1. Ground the task

Read the relevant subset of:

- `dev/AGENTS.md`;
- target source and exports;
- nearby tests and docs;
- existing CV/inference/backend abstractions that the feature will reuse;
- external reference/paper/package behavior when named by the task;
- existing benchmark scripts when performance is active.

Prefer existing project patterns over new abstractions.

## 2. Classify impact first

Record active axes:

- public API / compatibility;
- backend / dtype / device / fallback / memory ownership;
- loss / objective;
- penalty;
- solver / convergence / stopping;
- CV;
- inference;
- formula/model matrix;
- benchmark/performance;
- docs/evidence only.

**Impact classification controls the gates.** Do not activate unrelated backend/CV/inference/formula/performance work merely because an existing estimator class happens to expose those capabilities.

That scope rule does **not** let a newly introduced capability define away repository defaults. For a new or materially changed shared statistical/numerical capability, backend closure is active by default; for a new tunable loss x penalty capability, direct-fit + CV closure is active by default. Narrower capability requires an explicit non-applicable rationale or an approved deferral rather than simply omitting the claim.

Examples:

- New shared numerical estimator: API + correctness + NumPy/CuPy/Torch backend closure by default; CV/inference/formula are added when the model's contract requires them.
- New tunable loss x penalty capability: direct fit + CV selection/refit closure by default, plus affected backends and inference/formula when applicable.
- New solver path: solver + affected backends + convergence, plus CV when the solver participates in CV dispatch.
- Constructor deprecation only: API/compatibility + targeted regression; no new numerical backend matrix unless dispatch changes.
- Docs-only wording: docs gate only, unless the edit changes a support/evidence claim.

## 3. Define the public contract

Before final implementation, define relevant inputs, outputs, shapes, dtype/device ownership, statistical parameterization, errors, unsupported combinations, fallback/warning behavior, formula semantics, and backward compatibility.

Changing a public API incompatibly requires an explicit breaking decision unless the user already requested that break. A staged deprecation is preferred when practical.

For API migrations, include omission versus explicit historical default, `get_params`, `set_params`, sklearn clone compatibility where claimed, warning behavior, and old/new conflict semantics.

Do not narrow an otherwise shared new statistical capability to CPU-only or direct-fit-only merely to avoid repository completion gates. If a backend or CV layer is intentionally deferred, record the reason, user-visible behavior, tests, and follow-up scope and obtain the required approval.

## 4. Implement numerical capability only where active

For a new or materially changed shared statistical/numerical capability, statgpu's default backend contract is **NumPy + CuPy + Torch**. Implement and test all three unless the task/issue already defines a legitimate narrower capability or the user explicitly approves a backend deferral. A deferral must be visible in API behavior and must not masquerade as complete three-backend support.

For a new tunable public loss x penalty capability, the default completion contract also includes **direct fit + CV path/grid/folds/scoring/selection/final refit**. CV may be omitted only when the capability is genuinely non-tunable or an explicit deferral is approved.

For API-only/refactor-only changes that do not alter numerical capability or dispatch, preserve existing numerical paths instead of reopening or rewriting them without evidence. These changes may keep backend/CV/inference axes inactive except for targeted regressions needed to prove compatibility.

Explicit CUDA/Torch requests must not silently fall back to CPU. Only automatic device selection may choose another available backend under the documented contract.

## 5. Tests

Add the smallest deterministic tests that prove the active contract, then widen for shared infrastructure.

When applicable test:

- numerical baseline or analytic identity;
- NumPy/CuPy/Torch parity for new shared numerical capability, or affected-backend parity for narrower existing changes;
- dtype/device and unavailable-backend behavior;
- objective/penalty scaling;
- convergence/KKT/gradient/Hessian/prox semantics;
- public constructor and error behavior;
- CV grid/folds/selection/refit, including the default closure for new tunable loss x penalty capability;
- inference fields/summary/covariance;
- formula alignment;
- API migration/clone/set_params/deprecation behavior.

Do not require unrelated matrices for genuinely inactive axes, but do not label a default repository capability inactive merely because the implementation has not been completed yet.

## 6. External alignment and numerical gates

Use the strongest available comparison in this order:

1. analytic closed form/identity;
2. trusted existing statgpu path;
3. Python reference such as sklearn/statsmodels/scipy/lifelines;
4. authoritative R reference;
5. numerical invariant such as finite differences, KKT, monotonic objective, simulation coverage, or backend parity.

Align objective normalization and regularization scale before treating a difference as a bug.

Precision/convergence failures block performance work.

## 7. Performance

Invoke the `benchmark` skill when performance is requested, a new performance-sensitive kernel/path is added, or the change makes a performance claim.

Do not benchmark merely to satisfy ceremony for an API/docs-only change.

## 8. Independent review

Before completion of implementation work, invoke `code-review` for a fresh independent pass. Use `auto-fix` when the parent task authorizes implementation fixes; otherwise use `audit` and address blocking findings in the main task.

Because `code-review` runs in a forked context, pass enough scope in its arguments for it to identify the relevant branch/diff/files without relying on the current conversation.

## 9. Documentation

Update only the user/developer surfaces changed by the task.

For learner-facing model pages, prefer a learner-first explanation while preserving a complete public API inventory or an explicit link to the canonical API reference. Key-parameter teaching tables may be selective; the reference inventory may not silently omit public parameters. Keep EN/CN capability claims conceptually aligned when both pages are maintained.

Reference/implementation pages may remain reference-first.

Document objective/penalty mapping when external comparisons depend on it. Performance/evidence claims must identify auditable source artifacts.

## 10. Repository actions

Implementation requests authorize task-scoped file edits and validation. Commit, push, or PR creation may be performed when the user explicitly requested them or the active task already includes an approved branch/PR workflow.

Merge, tag, release, package publication, credential setup, or an unrequested breaking API decision always requires explicit user direction.

Never read or write credentials from tracked Markdown or `.claude/settings.json`; use maintained untracked/environment configuration for remote work.

## Completion status

End implementation workflows with one of:

- `COMPLETE`: all active local blocking gates pass; a new shared statistical/numerical capability has closed its default backend contract, and a new tunable loss x penalty capability has closed its default CV contract unless an approved exception applies; no unresolved CRITICAL/HIGH review finding remains.
- `PARTIAL_REMOTE_PENDING`: local work is complete and only explicitly identified remote GPU/R/external/large-scale evidence remains.
- `BLOCKED_NEEDS_USER_APPROVAL`: progress requires a user decision such as a backend/CV deferral, breaking API choice, merge/release/publication, credentials, or an accepted performance caveat.
- `FAILED`: an active local correctness/compatibility/backend/CV/convergence/fallback/review gate remains unresolved.

Report active axes, changed files, tests/benchmarks, compatibility decisions, approved exceptions, unrun evidence, and review outcome. Do not call genuinely inactive gates incomplete.