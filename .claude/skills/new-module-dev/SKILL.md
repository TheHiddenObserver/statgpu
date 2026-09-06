---
name: new-module-dev
description: Develop or substantially change a statgpu statistical method, estimator, loss, penalty, solver, inference path, CV path, or backend-aware numerical API with impact-based gates, targeted tests, documentation, and an independent review pass. Use for real implementation work, not for small docs edits or narrow API cleanups that do not change numerical capability.
when_to_use: Trigger for adding a new method or materially changing statistical/numerical behavior. Do not trigger for ordinary wording edits, isolated deprecations, or narrow refactors unless they also change numerical capability.
argument-hint: "[module-or-task]"
---

# statgpu New Module Development

Develop `$ARGUMENTS` using the smallest set of gates that proves the changed public contract.

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

**Impact classification controls the gates.** Do not activate unrelated backend/CV/inference/formula/performance work merely because an estimator class happens to expose those capabilities.

Examples:

- New numerical estimator: backend + API + tests, plus CV/inference/formula only if claimed by that estimator.
- New solver path: solver + affected backends + convergence, plus CV only if the solver participates in CV dispatch.
- Constructor deprecation only: API/compatibility + targeted regression; no new numerical backend matrix unless dispatch changes.
- Docs-only wording: docs gate only, unless the edit changes a support/evidence claim.

## 3. Define the public contract

Before final implementation, define relevant inputs, outputs, shapes, dtype/device ownership, statistical parameterization, errors, unsupported combinations, fallback/warning behavior, formula semantics, and backward compatibility.

Changing a public API incompatibly requires an explicit breaking decision unless the user already requested that break. A staged deprecation is preferred when practical.

For API migrations, include omission versus explicit historical default, `get_params`, `set_params`, sklearn clone compatibility where claimed, warning behavior, and old/new conflict semantics.

## 4. Implement numerical capability only where active

For new or materially changed numerical/statistical capability, implement every backend that capability declares. statgpu normally targets NumPy, CuPy, and Torch for shared statistical methods; an intentional backend deferral must be explicit in the task contract and fail visibly.

For API-only/refactor-only changes, preserve existing numerical paths instead of reopening or rewriting them without evidence.

Explicit CUDA/Torch requests must not silently fall back to CPU. Only automatic device selection may choose another available backend under the documented contract.

## 5. Tests

Add the smallest deterministic tests that prove the active contract, then widen for shared infrastructure.

When applicable test:

- numerical baseline or analytic identity;
- affected backend parity and unavailable-backend behavior;
- objective/penalty scaling;
- convergence/KKT/gradient/Hessian/prox semantics;
- public constructor and error behavior;
- CV grid/folds/selection/refit;
- inference fields/summary/covariance;
- formula alignment;
- API migration/clone/set_params/deprecation behavior.

Do not require unrelated matrices for inactive axes.

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

- `COMPLETE`: all active local blocking gates pass; no unresolved CRITICAL/HIGH review finding remains.
- `PARTIAL_REMOTE_PENDING`: local work is complete and only explicitly identified remote GPU/R/external/large-scale evidence remains.
- `BLOCKED_NEEDS_USER_APPROVAL`: progress requires a user decision such as a breaking API choice, backend deferral, merge/release/publication, credentials, or an accepted performance caveat.
- `FAILED`: an active local correctness/compatibility/backend/convergence/fallback/review gate remains unresolved.

Report active axes, changed files, tests/benchmarks, compatibility decisions, unrun evidence, and review outcome. Do not call inactive gates incomplete.
