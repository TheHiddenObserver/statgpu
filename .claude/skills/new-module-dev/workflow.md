# statgpu Development Workflow Reference

This is the detailed gate matrix for `new-module-dev/SKILL.md`. The skill entrypoint is authoritative for when this file should be loaded; this file expands the mechanics.

## Hard-exit contract

A development run ends with exactly one status:

- `COMPLETE`: every **active** local blocking gate passes and no unresolved CRITICAL/HIGH review finding remains.
- `PARTIAL_REMOTE_PENDING`: local work is complete; only identified remote GPU, R/external package, or large-scale evidence remains.
- `BLOCKED_NEEDS_USER_APPROVAL`: continuing requires a user decision such as a breaking API choice, backend deferral, accepted performance caveat, merge/release/publication, or credentials.
- `FAILED`: an active local correctness, compatibility, backend, convergence, fallback, formula, inference, or review gate remains unresolved.

Never mark an inactive gate as missing work.

## Impact classification

| Axis | Activate when |
| --- | --- |
| Public API | imports, constructor/method args, defaults, returns, errors, warnings, public attributes, deprecations |
| Backend/device | kernels, dtype/device selection, memory ownership, transfers, fallback |
| Loss/objective | formula, normalization, gradient/Hessian/Lipschitz, family behavior |
| Penalty | value/gradient/prox/LLA, group/adaptive semantics, category/dispatch |
| Solver | algorithm, dispatch, stopping, convergence, KKT/line search |
| CV | grid/path, folds, scoring, candidate failure, selection, refit |
| Inference | `compute_inference`, covariance, reference distribution, p/CI/summary/results |
| Formula | parser/model matrix, row alignment, feature names/order, prediction matrix |
| Performance | fast path/kernel/scaling claim/timing comparison |
| Docs/evidence | prose/changelog/evidence provenance without runtime change |

If uncertain, activate the smallest broader set that can plausibly be affected and record why.

### Scope examples

**New estimator with NumPy/CuPy/Torch claim**: API + backend + statistical correctness, plus CV/inference/formula only if part of its public contract.

**New solver option used by direct fit and CV**: solver + affected backends + CV + convergence; inference is active only if solver choice can change or invalidate inference behavior.

**Rename/deprecate constructor control without numerical dispatch change**: API/compatibility only, plus regression tests that prove existing direct/CV/backend behavior is preserved. Do not reopen unrelated numerical implementation.

**Docs wording**: docs/evidence only unless a support/performance/statistical claim changes.

## Gate matrix

| Gate | Blocking when active? | Evidence |
| --- | --- | --- |
| Public API contract | Yes | documented inputs/outputs/errors/defaults/compatibility |
| Numerical backend parity | Yes for declared new/changed capability | affected-backend tests, explicit unavailable behavior |
| Objective/penalty correctness | Yes | analytic/external/invariant evidence and scale mapping |
| Solver/convergence | Yes | status, KKT/gradient/monotonicity/line-search evidence as relevant |
| CV | Yes when public tuning behavior is changed/claimed | grid/folds/scoring/selection/refit/no-leakage tests |
| Inference | Yes when public inference behavior is changed/claimed | result fields/summary/covariance/backend/strict-fallback evidence |
| Formula | Yes when formula-facing behavior is changed/claimed | model-matrix alignment and failure-mode tests |
| Silent fallback | Always if path is touched | explicit device/error/warning/result contract |
| Performance | Conditional | benchmark JSON with provenance and correctness |
| Review | Yes | fresh `code-review` pass without unresolved CRITICAL/HIGH |
| Docs/artifacts | Yes for changed public behavior/claims | updated relevant surfaces and auditable evidence references |
| Remote GPU/R/large benchmark | No for local completion | `PARTIAL_REMOTE_PENDING` with exact missing evidence |

## API and compatibility workflow

For any public API change:

1. Inventory the current signature, public import path, defaults, and fitted attributes.
2. Decide whether the change is additive, deprecation, behavior change, or removal.
3. Preserve omitted-argument behavior unless the task explicitly changes the default.
4. For deprecations, distinguish omission from explicit legacy use when warnings depend on caller intent.
5. Validate `get_params` / `set_params` / clone behavior where estimator compatibility is claimed.
6. Avoid warning spam from internal constructor reconstruction.
7. Define old/new conflict behavior.
8. Preserve legacy numerical behavior during the compatibility window unless the migration explicitly changes it.
9. Add migration docs and removal horizon.
10. Add regression tests for the compatibility surface.

## Numerical implementation workflow

For new or materially changed numerical capability:

1. Implement the clearest correctness baseline.
2. Close each declared backend using existing backend abstractions where possible.
3. Keep explicit GPU requests fail-closed; no hidden CPU substitution.
4. Preserve dtype/device ownership and output contract.
5. Validate objective/gradient/Hessian/prox/KKT/stopping semantics relevant to the component.
6. Compare with the strongest available reference after objective/penalty alignment.
7. Only then optimize.

Shared backend abstractions are preferred when they reduce duplication without hiding device-specific behavior. Backend-specific kernels may remain explicit.

## Architecture-specific tests

### Loss

When a loss changes, cover relevant registry/export, value/gradient/fused behavior, finite differences, Hessian/Lipschitz claims, affected backends, representative penalties/solver dispatch, and CV/inference only if those public paths consume the changed loss.

### Penalty

Cover value/gradient/prox/LLA semantics, parameter validation, group/adaptive initialization when relevant, solver compatibility, affected backends, and only the CV/inference paths that use the changed penalty.

### Solver

Cover objective classes the solver claims, explicit unsupported combinations, convergence/KKT or monotonicity evidence, affected backends, and dispatch sites (including CV only when shared).

### CV

Cover deterministic/custom folds, sample weights, path/grid scale, candidate failure classification, finite evidence, tie handling, selected parameter, final refit, refit solver/device, and leakage between CV-only and final-estimator controls.

### Inference

Cover enabled/disabled behavior, fitted-state checks, result container/reporting fields, covariance/reference distribution, intercept/formula names, sample weights, backend provenance, strict/fallback behavior, and external/statistical baseline where appropriate.

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

## Validation tiers

Use these as evidence labels, not as excuses to activate unrelated gates:

- `local-minimal`: import, targeted CPU/numerical/API/error checks for the active change.
- `local-full`: all locally available active backends and compatibility/inference/formula/matrix checks required by the change.
- `remote-full`: local-full plus required physical GPU/R/external/large-scale evidence.

`COMPLETE` requires the local evidence needed by the active gates. If the only missing proof genuinely requires remote hardware/software, use `PARTIAL_REMOTE_PENDING`.

## Performance workflow

When performance is active:

1. Define target scale and timing scope.
2. Prove correctness/precision/convergence first.
3. Run one profiling pass.
4. Make at most two focused optimization attempts before reassessing scope.
5. Re-benchmark after each attempt.
6. Record environment and exact source provenance.

Do not claim a universal speedup from one machine/shape. Do not aggregate measurements from materially different hardware/software environments into one speedup statistic.

## Documentation workflow

Update the surfaces that actually expose the changed capability. Keep support claims synchronized across maintained English/Chinese docs and public entrypoints when relevant.

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

## Repository-action boundary

Task-scoped file edits and validation are part of implementation work. Commit/push/PR actions are allowed only when the user explicitly requested them or the current task already operates under an approved branch/PR workflow.

Always require explicit user direction for merge, tag, release, package publication, credential setup, or an unrequested breaking API decision.

## Completion report

Report:

- status (`COMPLETE`, `PARTIAL_REMOTE_PENDING`, `BLOCKED_NEEDS_USER_APPROVAL`, or `FAILED`);
- active impact axes and why;
- changed files;
- API/backward-compatibility decision;
- implemented/tested backends for the active numerical capability;
- CV/inference/formula status if active;
- objective/penalty mapping and precision/convergence evidence if active;
- tests/benchmarks and validation tier;
- independent review result;
- remote/unrun evidence and exact follow-up commands when needed.
