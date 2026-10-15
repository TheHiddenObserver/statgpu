# statgpu Development Workflow Reference

This is the detailed gate matrix for `new-module-dev/SKILL.md`. The skill entrypoint is authoritative for when this file should be loaded; this file expands the mechanics.

## Hard-exit contract

A development run ends with exactly one status:

- `COMPLETE`: every **active** local blocking gate passes, repository-default capability closure is satisfied or explicitly approved as an exception, final public docs/claims are included in the independent review, exact-head evidence is fresh, no unresolved CRITICAL/HIGH review finding remains, and any claimed project-skill trigger/output behavior has actual runtime/grading evidence.
- `PARTIAL_REMOTE_PENDING`: local/static work is complete; only identified remote GPU, R/external package, large-scale, or project-skill runtime behavioral evidence remains.
- `BLOCKED_NEEDS_USER_APPROVAL`: continuing requires a user decision such as a breaking API choice, backend/CV deferral, accepted performance caveat, merge/release/publication, or credentials.
- `FAILED`: an active local correctness, compatibility, backend, CV, convergence, fallback, formula, inference/resampling, review, or evidence-freshness gate remains unresolved.

Never mark a genuinely inactive gate as missing work. Conversely, do not make a repository-default gate inactive merely by narrowing the initial capability declaration.

## Phase 0 — target and contract reconnaissance

Before production edits for multi-axis work, shared base/mixin/dispatch work, or capability reconciliation:

1. Resolve the exact writable branch/head or working-tree scope and record the baseline.
2. Inventory public entrypoints, shared bases/mixins, typed wrappers, CV/meta-estimators, formula paths, runtime compatibility installers, and maintained docs that consume the capability.
3. Trace actual runtime dispatch rather than relying only on constructor names/docstrings.
4. Build the relevant capability matrix: public request, resolved runtime path, reported method/result, statistical target, backend/device, CV/final-refit stage, formula support, tests, docs, and evidence.
5. Mark each row implemented/tested/documented/unsupported/inconsistent/legacy/superseded.
6. Freeze characterization tests for ambiguous legacy behavior when useful before changing it.

This phase prevents two common mistakes: reimplementing a numerical capability that already exists behind an inconsistent public contract, and documenting a nominally generic path whose actual helper only implements one estimator/family.

### Consumer-graph rule

A shared base/mixin/dispatch/compatibility-installer change must enumerate maintained consumers. At minimum consider:

- public subclasses/wrappers;
- CV/meta-estimators and final-refit reconstruction;
- formula/model-matrix entrypoints;
- runtime installers that alter signatures/dispatch after import;
- docs/changelog/support matrices;
- validators/benchmarks that encode the old contract.

For each consumer record affected, preserved, or intentionally unsupported. Shared infrastructure may not be declared safe merely because one representative wrapper passes.

## Impact classification

### Change type

Classify the work as one or more of:

- new numerical/statistical capability;
- existing capability reconciliation / public contract repair;
- API migration/deprecation;
- numerical refactor with preserved public behavior;
- docs/evidence-only change.

`existing capability reconciliation` applies when implementation already exists in some form but public wrappers, requested/resolved method identity, inheritance consumers, CV/formula behavior, tests, or docs disagree. It is not treated as a brand-new numerical algorithm merely because the public contract needs repair.

### Existing capability reconciliation / contract repair

Use this path when a statistical/numerical implementation already exists but the public contract is inconsistent across shared bases, typed wrappers, runtime method resolution, fitted/reporting metadata, CV/formula consumers, tests, or documentation. Start from the Phase 0 capability matrix and consumer graph, preserve already-correct numerical paths, repair the canonical public/statistical contract, and make unsupported rows explicit. Do not reopen unrelated numerical capability merely because the shared class exposes it.

### Impact axes

| Axis | Activate when |
| --- | --- |
| Public API | imports, constructor/method args, defaults, returns, errors, warnings, public attributes, deprecations |
| Backend/device | kernels, dtype/device selection, memory ownership, transfers, fallback; **also by default for new shared statistical/numerical capability** |
| Loss/objective | formula, normalization, gradient/Hessian/Lipschitz, family behavior |
| Penalty | value/gradient/prox/LLA, group/adaptive semantics, category/dispatch |
| Solver | algorithm, dispatch, stopping, convergence, KKT/line search |
| CV | grid/path, folds, scoring, candidate failure, selection, refit; **also by default for new tunable loss x penalty capability** |
| Inference/resampling | `compute_inference`, estimand, method resolution, covariance, reference distribution, p/CI/summary/results, bootstrap/permutation semantics |
| Formula | parser/model matrix, row alignment, feature names/order, prediction matrix |
| Performance | fast path/kernel/scaling claim/timing comparison |
| Docs/evidence | prose/changelog/release-boundary/evidence provenance without runtime change |

If uncertain, activate the smallest broader set that can plausibly be affected and record why. This impact discipline is intended to stop API-only/refactor-only work from reopening unrelated numerical surfaces; it does not weaken default completion rules for genuinely new public numerical capabilities.

### Repository-default closure

Unless the task/issue already defines a legitimate narrower scope or the user explicitly approves a deferral:

- a new or materially changed **shared statistical/numerical capability** closes NumPy, CuPy, and Torch implementation/validation;
- a new **tunable loss x penalty capability** closes direct fit plus CV path/grid/folds/scoring/selection/final refit;
- an explicit backend or CV deferral records reason, user-visible behavior, deterministic tests/skips, and follow-up scope.

A CPU-only or direct-fit-only first draft is a valid intermediate implementation step, but it is not `COMPLETE` for a capability whose repository default remains broader.

### Scope examples

**New shared estimator**: API + correctness + NumPy/CuPy/Torch by default; add CV/inference/formula according to the model contract. A narrower backend set requires explicit approved scope rather than silent omission.

**New tunable loss x penalty capability**: direct fit + CV closure by default, affected backends, and inference/formula where applicable.

**Existing capability reconciliation**: API + affected statistical/dispatch axes + consumer graph + docs/evidence. Preserve already-correct numerical implementations; do not demand unrelated new backends/CV merely because the shared base exposes them, but do verify every affected declared consumer.

**New solver option used by direct fit and CV**: solver + affected backends + CV + convergence; inference is active only if solver choice can change or invalidate inference behavior.

**Rename/deprecate constructor control without numerical dispatch change**: API/compatibility only, plus regression tests that prove existing direct/CV/backend behavior is preserved. Do not reopen unrelated numerical implementation.

**Docs wording**: docs/evidence only unless a support/performance/statistical claim changes.

## Phase 1 — desired contract freeze

Before implementation, write down the intended public contract and unsupported matrix. For cross-cutting public statistical changes (normally three or more active axes, a shared base/mixin/dispatch change, or an inference-method contract change), create/update an implementation plan and perform a pre-implementation design audit before production numerical edits.

### Public API and compatibility

For any public API change:

1. Inventory the current signature, runtime `inspect.signature`, public import path, defaults, fitted attributes, and docs.
2. Decide whether the change is additive, reconciliation, deprecation, behavior change, or removal.
3. Preserve omitted-argument behavior unless the task explicitly changes the default.
4. For deprecations, distinguish omission from explicit legacy use when warnings depend on caller intent.
5. Validate `get_params` / `set_params` / clone behavior where estimator compatibility is claimed.
6. Avoid warning spam from internal constructor reconstruction.
7. Define old/new conflict behavior.
8. Preserve legacy numerical behavior during the compatibility window unless the migration explicitly changes it.
9. Add migration docs and removal horizon.
10. Add regression tests for the compatibility surface.

When runtime compatibility installers alter public signatures or dispatch, additionally define:

- installer idempotence and import-order behavior;
- `__wrapped__`/introspection expectations;
- clone/meta-estimator reconstruction semantics;
- source-static versus runtime-public documentation boundaries;
- whether the contract can now be moved safely into normal source rather than adding another installer layer.

An API-only/deprecation-only/refactor-only task may keep backend/CV/inference gates inactive when it does not change those capabilities or dispatch. Prove preservation with targeted regression rather than reopening unrelated implementation.

### Inference contract

For public inference, freeze all applicable items before implementation:

1. **Estimand/target**: penalized coefficient, unpenalized population coefficient, debiased coefficient, active-set refit coefficient, prediction functional, or another explicitly named target.
2. **Method identity**: public requested control -> resolved numerical method -> `_inference_result.method`/summary wording. These may differ only under a documented `auto`/alias/migration contract, never accidentally.
3. **Covariance/reference distribution**: bread/meat, penalty curvature, dispersion, t/z/reference family, and intercept treatment.
4. **Conditioning**: fixed tuning, CV-selected tuning, selected active set, or other data-dependent choices.
5. **Uncertainty not corrected**: shrinkage, selection, tuning, model/family selection, etc.; state whether corrected, conditioned on, or explicitly uncorrected.
6. **Coverage scope**: marginal versus simultaneous; target family for joint procedures.
7. **Backend/provenance**: numerical backend/device, reporting boundary, and fallback behavior.
8. **Unsupported matrix**: unsupported loss x penalty x method combinations fail before returning plausible-looking partial results.

A default constructor spelling that names one inferential procedure while silently executing/reporting another is a public-contract defect unless the default is explicitly an `auto` resolver and the resolved method is user-visible.

### Resampling contract

For bootstrap/permutation/resampling inference define:

- resampling unit;
- response/residual/data-generating mechanism for each claimed family/model;
- fixed/random-design assumption where relevant;
- analytic/sample weights handling;
- estimator/refit identity and penalty preservation;
- whether selection/CV/tuning is repeated inside each resample or conditioned on;
- backend/device and host-transfer behavior;
- RNG/seed reproducibility;
- reported estimand/interval/test;
- unsupported families/combinations.

A “generic” bootstrap helper must not silently construct a different model/family during refit. If only Gaussian residual bootstrap is implemented, expose that scope explicitly rather than advertising a universal fallback.

### Release-boundary contract

For unreleased public behavior record separately:

- current published version;
- behavior currently present on `master`/branch;
- target release, if known.

Documentation should say “implemented on master; targeted for X.Y.Z” until the release/version-bump task actually ships it. Do not preemptively bump package metadata in an unrelated capability task.

## Gate matrix

| Gate | Blocking when active? | Evidence |
| --- | --- | --- |
| Contract reconnaissance | Yes for multi-axis/shared/reconciliation work | consumer graph + current capability matrix + runtime-path trace |
| Public API contract | Yes | documented inputs/outputs/errors/defaults/compatibility + runtime introspection when installers are involved |
| Numerical backend parity | Yes for new/materially changed shared numerical capability by default; otherwise when affected | NumPy/CuPy/Torch tests or an approved narrower scope; explicit unavailable behavior |
| Objective/penalty correctness | Yes | analytic/external/invariant evidence and scale mapping |
| Solver/convergence | Yes | status, KKT/gradient/monotonicity/line-search evidence as relevant |
| CV | Yes for new tunable loss x penalty capability by default; otherwise when public tuning behavior is changed/affected | grid/folds/scoring/selection/refit/no-leakage tests or approved non-tunable/deferral contract |
| Inference | Yes when public inference behavior is changed/claimed | estimand + requested/resolved/reported method + covariance/reference + conditioning/uncorrected uncertainty + result fields/backend/strict-fallback evidence |
| Resampling | Yes when bootstrap/permutation/resampling is claimed | family-valid DGP/resampling unit/refit identity/tuning-selection semantics/RNG/backend tests |
| Formula | Yes when formula-facing behavior is changed/claimed | model-matrix alignment and failure-mode tests |
| Silent fallback | Always if path is touched | explicit device/error/warning/result contract |
| Performance | Conditional | benchmark JSON with provenance and correctness |
| Docs/artifacts | Yes for changed public behavior/claims | updated relevant surfaces and auditable evidence references, including release-boundary wording |
| Review | Yes | fresh `code-review` pass over final code/tests/docs without unresolved CRITICAL/HIGH |
| Exact-head evidence freshness | Yes at completion | re-resolved base/head and evidence identity for final state |
| Project-skill behavioral runtime | Yes only when trigger/output behavior is claimed empirically closed | executed skill-runtime/grading artifact; eval definitions/static contract tests are definition evidence only |
| Remote GPU/R/large benchmark | No for local completion | `PARTIAL_REMOTE_PENDING` with exact missing evidence |

## Phase 2 — implementation

For new or materially changed shared numerical capability:

1. Implement the clearest correctness baseline.
2. Close NumPy, CuPy, and Torch by default using existing backend abstractions where possible; if a narrower capability is intended, document and obtain the required approval before calling the work complete.
3. Keep explicit GPU requests fail-closed; no hidden CPU substitution.
4. Preserve dtype/device ownership and output contract.
5. Validate objective/gradient/Hessian/prox/KKT/stopping semantics relevant to the component.
6. If the capability is a tunable loss x penalty surface, close the direct + CV contract unless it is genuinely non-tunable or an approved CV deferral exists.
7. Compare with the strongest available reference after objective/penalty alignment.
8. Only then optimize.

For capability reconciliation, fix the canonical source of truth rather than stacking wrappers around inconsistent behavior. Reuse already-correct numerical paths and focus changes on public resolution, validation, reporting identity, consumer preservation, and unsupported boundaries.

Shared backend abstractions are preferred when they reduce duplication without hiding device-specific behavior. Backend-specific kernels may remain explicit.

## Phase 3 — tests and validation

### Characterization before behavior change

When historical behavior is ambiguous, add characterization tests before changing it. These tests should identify what must be preserved versus what is intentionally repaired; they are not a reason to preserve a known bug indefinitely.

### Loss

When a loss changes, cover relevant registry/export, value/gradient/fused behavior, finite differences, Hessian/Lipschitz claims, NumPy/CuPy/Torch for new shared capability (or affected backends for a narrower existing change), representative penalties/solver dispatch, and CV/inference only if those public paths consume the changed loss. A newly introduced tunable loss x penalty surface must also satisfy the default CV closure above.

### Penalty

Cover value/gradient/prox/LLA semantics, parameter validation, group/adaptive initialization when relevant, solver compatibility, NumPy/CuPy/Torch for new shared capability (or affected backends for a narrower existing change), and only the inference paths that use the changed penalty. New tunable loss x penalty combinations close direct + CV by default.

### Solver

Cover objective classes the solver claims, explicit unsupported combinations, convergence/KKT or monotonicity evidence, affected backends, and dispatch sites (including CV when shared).

### CV

Cover deterministic/custom folds, sample weights, path/grid scale, candidate failure classification, finite evidence, tie handling, selected parameter, final refit, refit solver/device, inference stage, and leakage between CV-only and final-estimator controls.

### Inference/resampling

Cover enabled/disabled behavior, fitted-state checks, requested/resolved/reported method identity, estimand/parameter ownership, result container/reporting fields, covariance/reference distribution, intercept/formula names, sample weights, tuning/selection conditioning, backend provenance, strict/fallback behavior, unsupported combinations, and external/statistical baseline where appropriate.

For bootstrap/permutation/resampling paths, test that the resampled/refit estimator preserves the claimed family/loss/penalty and that selection/CV repetition versus conditioning matches the contract.

### Formula

Cover intercept, categorical reference levels, interactions/transforms used by project docs, missing-row alignment, column names/order, array-path equivalence, and prediction reconstruction.

### Backend helper/kernel

Cover dtype/device preservation, unavailable device behavior, no hidden full host transfer, synchronization-sensitive behavior, memory cleanup/ownership, and all consumers whose dispatch contract changes.

## External-baseline hierarchy

Use the strongest available reference:

1. analytic closed form/identity;
2. trusted existing statgpu implementation;
3. Python reference such as sklearn/statsmodels/scipy/lifelines/patsy;
4. authoritative R package/reference;
5. numerical invariants.

Do not block indefinitely on a weaker/missing external dependency when a stronger analytic invariant is available.

When loss normalization differs, record the mapping. For example, if statgpu uses an average loss and a reference uses summed residual loss, map the regularization parameter instead of mutating statgpu's definition.

For procedures whose main claim is coverage/error control, targeted simulation may be the strongest practical invariant; record the design, repetitions, target, and interpretation rather than treating one simulation as a universal theorem.

## Validation tiers

Use these as evidence labels, not as excuses to activate unrelated gates:

- `local-minimal`: import, targeted CPU/numerical/API/error checks for the active change.
- `local-full`: all locally available evidence required by the active/default capability contract, including affected backends and compatibility/inference/formula/matrix checks.
- `remote-full`: local-full plus required physical GPU/R/external/large-scale evidence.

For project-skill/eval changes, skill-runtime/grading evidence is a separate behavioral tier: committed eval definitions and static contract tests do not execute the skill. `COMPLETE` requires that tier only when the task claims actual trigger/output quality is empirically closed; otherwise record it as explicitly pending.

`COMPLETE` requires the local evidence needed by active and repository-default gates. If the only missing proof genuinely requires remote hardware/software or an unavailable skill runtime, use `PARTIAL_REMOTE_PENDING` rather than shrinking the capability/evidence claim.

## Performance workflow

When performance is active:

1. Define target scale and timing scope.
2. Prove correctness/precision/convergence first.
3. Run one profiling pass.
4. Make at most two focused optimization attempts before reassessing scope.
5. Re-benchmark after each attempt.
6. Record environment and exact source provenance.

Do not claim a universal speedup from one machine/shape. Do not aggregate measurements from materially different hardware/software environments into one speedup statistic.

## Phase 4 — documentation and release boundary

Update the surfaces that actually expose the changed capability **before the final independent review**. Keep support claims synchronized across maintained English/Chinese docs and public entrypoints when relevant.

For learner-facing model pages:

1. explain the problem and motivation;
2. give intuition and method-selection guidance;
3. bridge to the objective/model with only necessary math;
4. include a self-contained runnable example and interpretation;
5. provide decision-oriented key-parameter guidance;
6. keep a complete API inventory or link to the canonical complete reference;
7. state backend/inference/CV limitations and scientific pitfalls;
8. move solver/backend implementation detail later in the page.

A selective teaching table is allowed; a reference inventory may not silently omit public API.

For unreleased behavior, distinguish the current published release from current branch/master and target release. Do not write “available since X.Y.Z” before X.Y.Z is actually released.

## Phase 5 — independent review/fix

Run the canonical `code-review` skill over the intended final code, tests, docs, and evidence claims. Use `auto-fix` when authorized. If fixes change the target, rerun affected validation and re-review the new state from scratch.

A review that ran before final public docs were added does not certify those docs. An earlier clean verdict is historical after the reviewed target changes.

## Phase 6 — exact-head evidence closure

Before final status:

1. Re-resolve effective base/head and worktree fingerprint if relevant.
2. Verify each hosted CI result belongs to the final head.
3. Verify review verdict identity matches the final reviewed state.
4. Verify physical GPU/R/external/benchmark artifacts record the source identity/fingerprint and validator contract they actually tested.
5. For project-skill/eval changes, distinguish committed eval definitions/static contract tests from an actually executed skill-runtime/grading run; never present the former as behavioral execution evidence.
6. After any later source/docs/review-fix commit, mark earlier evidence historical unless an artifact separately fingerprints the unchanged relevant source and explicitly permits reuse.
7. Re-run only evidence invalidated by the final changes; do not assume reuse silently.
8. Perform one final freshness check after all task commits.

This is an evidence-DAG rule, not a demand to rerun every numerical benchmark after wording-only changes. Reuse is valid only when the evidence contract proves the relevant source/input stayed unchanged.

## Repository-action boundary

Task-scoped file edits and validation are part of implementation work. Commit/push/PR actions are allowed only when the user explicitly requested them or the current task already operates under an approved branch/PR workflow.

Always require explicit user direction for merge, tag, release, package publication, credential setup, or an unrequested breaking API decision.

## Completion report

Report:

- status (`COMPLETE`, `PARTIAL_REMOTE_PENDING`, `BLOCKED_NEEDS_USER_APPROVAL`, or `FAILED`);
- change type and active impact axes;
- contract-reconnaissance/consumer-graph summary when required;
- repository-default capability gates and any approved exceptions;
- changed files;
- API/backward-compatibility and statistical-target/method-identity decisions;
- implemented/tested backends for the active numerical capability;
- CV/inference/resampling/formula status if active or required by default closure;
- objective/penalty mapping and precision/convergence evidence if active;
- docs and current-release/target-release wording;
- tests/benchmarks and validation tier;
- independent review result;
- exact final base/head and evidence freshness;
- remote/runtime/unrun evidence and exact follow-up commands when needed.
