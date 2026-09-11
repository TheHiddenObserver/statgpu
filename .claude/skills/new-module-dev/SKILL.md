---
name: new-module-dev
description: Develop, reconcile, or substantially change a statgpu statistical method, estimator, loss, penalty, solver, inference path, CV path, or backend-aware numerical API with contract reconnaissance, impact-based gates, targeted tests, documentation, exact-head evidence closure, and an independent review pass. Use for real implementation work and cross-cutting public statistical-contract repair, not for small docs edits or narrow API cleanups that do not change capability or dispatch.
when_to_use: Trigger for adding a new method, materially changing statistical/numerical behavior, or reconciling an existing capability whose shared implementation, public wrappers, runtime dispatch, tests, or docs disagree. Do not trigger for ordinary wording edits, isolated deprecations, or narrow refactors unless they also change capability or dispatch.
argument-hint: "[module-or-task]"
---

# statgpu New Module Development

Develop `$ARGUMENTS` using the smallest set of gates that proves the changed public contract **without weakening statgpu's default capability closure for genuinely new numerical/statistical features**.

Read [workflow.md](workflow.md) before broad implementation, when more than one impact axis is active, when shared base/mixin/dispatch code is touched, or when a completion/blocking decision is needed.

## 1. Ground and resolve the task

Read the relevant subset of:

- `dev/AGENTS.md`;
- target source and exports;
- nearby tests and docs;
- existing CV/inference/backend abstractions that the feature will reuse;
- runtime compatibility installers/wrappers that can alter the public surface after import;
- external reference/paper/package behavior when named by the task;
- existing benchmark scripts when performance is active.

Resolve the exact writable branch/head or working-tree scope before mutation. Prefer existing project patterns over new abstractions.

## 2. Contract reconnaissance before implementation

For any multi-axis change, shared base/mixin/dispatch change, or existing-capability reconciliation, inventory the current contract **before editing production code**.

At minimum record the relevant rows of a capability matrix covering:

- public entry point / subclass / meta-estimator;
- loss/family and penalty;
- requested public method/control and actual resolved runtime path;
- fitted/reporting method identity and statistical target where inference is involved;
- backend/device support and fallback behavior;
- CV/final-refit and formula consumers where applicable;
- current tests, docs/changelog claims, and evidence;
- status: implemented, tested, documented, unsupported, inconsistent, legacy, or superseded.

When shared infrastructure changes, enumerate the **consumer graph**: public subclasses, CV/meta-estimators, formula paths, runtime installers, and maintained docs that consume the changed contract. Mark each consumer affected, preserved, or intentionally unsupported.

Do not treat the existence of code as proof of a supported public capability. Conversely, do not reimplement a capability before checking whether the numerical path already exists and only its public contract is inconsistent.

## 3. Classify change type and impact

First classify the task as one or more of:

- new numerical/statistical capability;
- existing capability reconciliation / public contract repair;
- API migration/deprecation;
- numerical refactor with preserved public behavior;
- docs/evidence-only change.

Then record active axes:

- public API / compatibility;
- backend / dtype / device / fallback / memory ownership;
- loss / objective;
- penalty;
- solver / convergence / stopping;
- CV;
- inference / resampling;
- formula/model matrix;
- benchmark/performance;
- docs/evidence only.

**Impact classification controls the gates.** Do not activate unrelated backend/CV/inference/formula/performance work merely because an existing estimator class happens to expose those capabilities.

That scope rule does **not** let a newly introduced capability define away repository defaults. For a new or materially changed shared statistical/numerical capability, backend closure is active by default; for a new tunable loss x penalty capability, direct-fit + CV closure is active by default. Narrower capability requires an explicit non-applicable rationale or an approved deferral rather than simply omitting the claim.

Examples:

- New shared numerical estimator: API + correctness + NumPy/CuPy/Torch backend closure by default; CV/inference/formula are added when the model's contract requires them.
- New tunable loss x penalty capability: direct fit + CV selection/refit closure by default, plus affected backends and inference/formula when applicable.
- Existing capability reconciliation: preserve already-correct numerical paths, but close disagreements across public API, shared/typed consumers, resolved method identity, tests, docs, and affected backend/CV/formula surfaces.
- New solver path: solver + affected backends + convergence, plus CV when the solver participates in CV dispatch.
- Constructor deprecation only: API/compatibility + targeted regression; no new numerical backend matrix unless dispatch changes.
- Docs-only wording: docs gate only, unless the edit changes a support/evidence claim.

## 4. Freeze the desired public/statistical contract

Before implementation, define relevant inputs, outputs, shapes, dtype/device ownership, statistical parameterization, errors, unsupported combinations, fallback/warning behavior, formula semantics, backward compatibility, and release boundary.

For public inference behavior, explicitly define:

- the inferential target/estimand (for example penalized coefficient, unpenalized coefficient, debiased coefficient, or active-set refit coefficient);
- `requested method -> resolved method -> reported result method` identity;
- covariance/bread/meat/reference-distribution conventions;
- whether inference is conditional on fixed tuning, CV-selected tuning, or a selected active set;
- whether shrinkage, selection, and tuning uncertainty are corrected, conditioned on, or intentionally uncorrected;
- marginal versus simultaneous coverage when relevant;
- unsupported combinations and fail-closed behavior.

For bootstrap/permutation/resampling inference, also define the resampling unit and data-generating mechanism, residual/response construction, fixed/random design assumption, weights, estimator/refit identity, whether selection/CV is repeated or conditioned on, backend, randomness/reproducibility, and reported statistical target. A helper labeled generic must not silently implement only one family/model's resampling scheme.

For API migrations, include omission versus explicit historical default, `get_params`, `set_params`, sklearn clone compatibility where claimed, warning behavior, and old/new conflict semantics. If runtime installers alter the public signature, include import-order/idempotence/introspection behavior and the source-static versus runtime-public documentation boundary.

For unreleased behavior, distinguish the current published version from current `master` and any target release. Do not describe a target release as already shipped or bump package metadata early unless the task is the release/version-bump itself.

Changing a public API incompatibly requires an explicit breaking decision unless the user already requested that break. A staged deprecation is preferred when practical.

For cross-cutting public statistical changes (normally three or more active axes, a shared base/mixin/dispatch change, or an inference-method contract change), write/update an implementation plan and perform a pre-implementation design audit before production numerical edits.

## 5. Implement numerical capability only where active

For a new or materially changed shared statistical/numerical capability, statgpu's default backend contract is **NumPy + CuPy + Torch**. Implement and test all three unless the task/issue already defines a legitimate narrower capability or the user explicitly approves a backend deferral. A deferral must be visible in API behavior and must not masquerade as complete three-backend support.

For a new tunable public loss x penalty capability, the default completion contract also includes **direct fit + CV path/grid/folds/scoring/selection/final refit**. CV may be omitted only when the capability is genuinely non-tunable or an explicit deferral is approved.

For reconciliation/API/refactor work, fix the canonical source of truth and preserve already-correct numerical paths instead of layering additional compatibility wrappers without need. If a compatibility installer is necessary, keep it bounded and test idempotence/introspection/reconstruction behavior.

Explicit CUDA/Torch requests must not silently fall back to CPU. Only automatic device selection may choose another available backend under the documented contract.

## 6. Tests and characterization

Before changing ambiguous legacy behavior, add characterization tests when they help distinguish intentional compatibility from a bug.

Add the smallest deterministic tests that prove the active contract, then widen for shared infrastructure. When applicable test:

- numerical baseline or analytic identity;
- NumPy/CuPy/Torch parity for new shared numerical capability, or affected-backend parity for narrower existing changes;
- dtype/device and unavailable-backend behavior;
- objective/penalty scaling;
- convergence/KKT/gradient/Hessian/prox semantics;
- public constructor and error behavior;
- consumer-graph preservation for shared base/mixin/dispatch changes;
- CV grid/folds/selection/refit, including the default closure for new tunable loss x penalty capability;
- inference estimand/method identity, fields/summary/covariance, tuning/selection conditioning, and unsupported combinations;
- resampling semantics where bootstrap/permutation is claimed;
- formula alignment;
- API migration/clone/set_params/deprecation behavior.

Do not require unrelated matrices for genuinely inactive axes, but do not label a default repository capability inactive merely because the implementation has not been completed yet.

## 7. External alignment, simulation, and performance

Use the strongest available comparison in this order:

1. analytic closed form/identity;
2. trusted existing statgpu path;
3. Python reference such as sklearn/statsmodels/scipy/lifelines;
4. authoritative R reference;
5. numerical invariant such as finite differences, KKT, monotonic objective, simulation coverage, or backend parity.

Align objective normalization and regularization scale before treating a difference as a bug. For inferential procedures whose main claim is coverage/error control rather than pointwise algebra, use targeted simulation evidence when appropriate.

Precision/convergence failures block performance work.

Invoke the `benchmark` skill when performance is requested, a new performance-sensitive kernel/path is added, or the change makes a performance claim. Do not benchmark merely to satisfy ceremony for an API/docs-only change.

## 8. Documentation and release boundary

Update the user/developer surfaces changed by the task **before the final independent review**, so the review covers the final public claims.

For learner-facing model pages, prefer a learner-first explanation while preserving a complete public API inventory or an explicit link to the canonical API reference. Key-parameter teaching tables may be selective; the reference inventory may not silently omit public parameters. Keep EN/CN capability claims conceptually aligned when both pages are maintained.

Reference/implementation pages may remain reference-first. Document objective/penalty mapping when external comparisons depend on it. Performance/evidence claims must identify auditable source artifacts.

For unreleased public behavior, state the current published version separately from the target release (for example “implemented on master; targeted for 0.2.6”). Release-time wording/version metadata should be updated by the release task, not preemptively by an unrelated feature task.

## 9. Independent review and fix loop

After production code, tests, and public docs for the intended state are present, invoke `code-review` for a fresh independent pass. Use `auto-fix` when the parent task authorizes implementation fixes; otherwise use `audit` and address blocking findings in the main task.

Because `code-review` runs in a forked context, pass enough scope in its arguments for it to identify the relevant branch/diff/files without relying on the current conversation.

If review fixes change runtime behavior, API, tests, or docs, rerun the affected validation and re-review the resulting state from scratch. Do not treat an earlier clean verdict as transferable to a later head.

## 10. Exact-head evidence closure

Before completion, re-resolve the effective base/head and close the evidence graph for the exact final state.

- Hosted CI, code-review verdicts, physical GPU/R runs, and benchmark artifacts prove only the source identity/fingerprint they actually record.
- After HEAD moves, earlier evidence is historical unless the artifact separately records an unchanged relevant `source_sha`/content fingerprint and its validator contract explicitly allows that reuse.
- Do not use a green workflow from an earlier head to certify a later head.
- Re-run only the evidence invalidated by the final changes; docs-only changes need not force unrelated numerical reruns when the evidence contract separately fingerprints the unchanged numerical source, but that reuse must be explicit rather than assumed.
- Final completion requires a freshness recheck after all source/docs/review-fix commits that belong to the task.

## 11. Repository actions

Implementation requests authorize task-scoped file edits and validation. Commit, push, or PR creation may be performed when the user explicitly requested them or the active task already includes an approved branch/PR workflow.

Merge, tag, release, package publication, credential setup, or an unrequested breaking API decision always requires explicit user direction.

Never read or write credentials from tracked Markdown or `.claude/settings.json`; use maintained untracked/environment configuration for remote work.

## Completion status

End implementation workflows with one of:

- `COMPLETE`: all active local blocking gates pass; a new shared statistical/numerical capability has closed its default backend contract, and a new tunable loss x penalty capability has closed its default CV contract unless an approved exception applies; final docs/claims are reviewed; exact-head evidence is fresh; no unresolved CRITICAL/HIGH review finding remains.
- `PARTIAL_REMOTE_PENDING`: local work is complete and only explicitly identified remote GPU/R/external/large-scale evidence remains.
- `BLOCKED_NEEDS_USER_APPROVAL`: progress requires a user decision such as a backend/CV deferral, breaking API choice, merge/release/publication, credentials, or an accepted performance caveat.
- `FAILED`: an active local correctness/compatibility/backend/CV/convergence/fallback/inference/resampling/review/evidence-freshness gate remains unresolved.

Report change type, active axes, consumer/capability inventory summary, changed files, tests/benchmarks, compatibility/statistical-contract decisions, approved exceptions, docs/release-boundary status, exact-head evidence, unrun evidence, and review outcome. Do not call genuinely inactive gates incomplete.
