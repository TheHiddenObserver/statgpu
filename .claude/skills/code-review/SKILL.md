---
name: code-review
description: Review statgpu changes for correctness, public API compatibility, statistical and numerical behavior, backend dispatch, CV/inference/formula contracts, tests, docs, and performance. Use when the user asks for a review or audit, or when a development workflow explicitly requests a final independent review pass. Do not use as a substitute for ordinary implementation work.
when_to_use: Trigger for PR review, audit, fresh review, or an explicit review/fix pass. Do not trigger merely because code is being edited.
argument-hint: "[audit|auto-fix] [scope]"
compatibility: "Claude Code >= 2.1.218 is required for the blocking fork contract provided by background: false."
context: fork
background: false
---

# statgpu Code Review

Review `$ARGUMENTS` as an independent pass. Start from the current repository state rather than inheriting an earlier clean verdict.

The intended blocking fork behavior requires Claude Code >= 2.1.218. On an older client, do not assume this skill's synchronous review contract is enforced by `background: false`; upgrade before relying on it as a completion gate.

## Modes

- `audit`: findings only; do not edit files.
- `auto-fix`: fix in-scope CRITICAL/HIGH issues and relevant MEDIUM issues, run targeted validation, then re-review.
- If no mode is supplied, infer `audit` for review-only requests and `auto-fix` only when the caller explicitly asks for fixes or a development workflow requests a review/fix pass.

## 1. Classify impact before opening gates

Record the active axes:

- public API / compatibility / deprecation
- backend, dtype, device, memory ownership, fallback
- loss / objective
- penalty
- solver / convergence / stopping
- CV / tuning / refit
- inference
- formula/model matrix
- benchmark/performance
- docs/evidence only

Only activate gates that the change can affect. A public API or refactor-only change does **not** require new numerical backend, CV, or inference capability merely because it touches an existing estimator. It must instead prove that the existing declared capabilities and dispatch behavior remain unchanged where relevant.

That scope discipline does not permit a genuinely new public capability to define away repository defaults. New shared statistical/numerical capability defaults to NumPy/CuPy/Torch closure; new tunable loss x penalty capability defaults to direct-fit + CV closure unless the capability is genuinely non-tunable or an explicit approved deferral applies.

A docs-only change becomes behavioral review scope if it changes a support, API, inference, backend, performance, or evidence claim.

## 2. Core review order

1. Correctness and public contract.
2. API migration/backward compatibility when active.
3. Repository-default capability closure for genuinely new numerical features.
4. Numerical/backend/CV/inference/formula gates that are active.
5. Tests and failure behavior.
6. Performance only after correctness passes.
7. Docs, evidence provenance, and maintainability.

For the detailed component matrix and severity rules, read [review-matrix.md](review-matrix.md) when the change touches more than one axis, changes public statistical behavior, or needs a blocking verdict.

## 3. Correctness

Check formulas, objective normalization, penalty scaling, gradients/Hessians/prox/KKT, prediction semantics, convergence status, sample weights, intercept handling, inference fields, and edge cases that are relevant to the active axes.

When comparing with an external package, align the objective before changing production code. If statgpu uses an average loss and a reference uses a summed loss, compare at the equivalent regularization scale rather than changing statgpu's statistical definition just to match the reference.

Precision or convergence failure is a correctness issue, not a benchmark-only issue.

## 4. Public API migration and deprecation

When constructor parameters, method signatures, defaults, warnings, aliases, fitted attributes, or public imports change, explicitly review:

- omitted argument versus explicitly supplied historical default;
- constructor signature and introspection;
- `get_params()` and `set_params()`;
- sklearn `clone()` and meta-estimator reconstruction where applicable;
- warning category, message, stacklevel, and duplicate-warning noise;
- old/new parameter conflict handling;
- legacy numerical behavior during a deprecation window;
- fitted-state invalidation after parameter changes;
- public aliases and import paths;
- migration docs/examples/changelog and planned removal boundary.

Do not silently reinterpret a deprecated parameter if that changes historical numerical behavior unless the migration explicitly authorizes that behavior change.

## 5. Backend and fallback review

For a **new or materially changed shared statistical/numerical capability**, the repository-default backend gate is NumPy + CuPy + Torch. Missing one is blocking unless the task/issue already defines a legitimate narrower capability or an explicit backend deferral has been approved with user-visible failure behavior and follow-up scope.

For changes to an existing capability that can affect numerical execution or dispatch, verify the affected declared backends proportional to blast radius. API-only/refactor-only work should verify that existing backend claims and device routing were not changed accidentally; it does not require a new backend implementation merely because an estimator already exposes multiple devices.

Explicit `device="cuda"` or `device="torch"` must not silently substitute CPU. Approximation, fallback, dtype conversion, or device conversion must be visible through the public contract.

## 6. CV, inference, and formula

Activate these gates from both the repository defaults for new capabilities and the existing public contract:

- A new **tunable loss x penalty capability** includes direct-fit + CV path/grid/folds/scoring/selection/final-refit closure by default. It may omit CV only when genuinely non-tunable or under an explicit approved deferral.
- For existing/API-only work, CV is blocking when the change can affect CV API, path/grid generation, fold scoring, selection, clone/reconstruction, or final refit; do not expand CV scope when none of those can change.
- Inference is blocking when the current API/docs claim it or the change alters inference-facing behavior. Estimation-only can be a legitimate design when it is explicit and tested.
- Formula is blocking when the changed path is formula-facing or can alter model-matrix semantics.

Do not expand scope solely because an external package offers an additional capability that statgpu does not claim.

## 7. Tests and evidence

Prefer the smallest test set that proves the active/default contracts, then widen when shared infrastructure is touched. Include negative tests for unsupported combinations and compatibility tests for public migrations.

For a new shared numerical capability, lack of NumPy/CuPy/Torch closure is not made acceptable by simply documenting a narrower first draft; record an approved deferral or keep the work incomplete. The same principle applies to CV for new tunable loss x penalty capability.

For performance claims, require synchronized GPU timing, explicit timing scope, target scale, environment provenance, and numerical correctness before comparing speed.

Historical benchmark or GPU evidence proves only the source, validator contract, hardware, and environment it actually records.

## 8. Auto-fix loop

In `auto-fix` mode:

1. Record findings with severity and active dimension.
2. Fix all in-scope CRITICAL/HIGH issues that can be fixed without an unrequested breaking/deferral decision.
3. Fix MEDIUM issues that affect the requested feature or a completion gate.
4. Run targeted tests/static checks/benchmarks as applicable.
5. Re-review the affected surface from the new state.
6. Stop only when no unresolved CRITICAL/HIGH issue remains, or report the blocking decision/evidence gap.

Do not broaden an API cleanup into unrelated numerical refactoring.

## 9. Severity

- `CRITICAL`: wrong results/objective/inference, silent backend substitution, security/credential leak, or a public path can return materially incorrect results without detection.
- `HIGH`: broken public compatibility, missing repository-default backend/CV closure for a new capability without approved deferral, required declared backend failure, convergence failure, missing active CV/inference/formula contract, serious regression, or missing tests that can hide a changed public behavior.
- `MEDIUM`: important maintainability, docs, evidence, coverage, warning quality, or bounded performance issue that does not invalidate core behavior.
- `LOW`: small cleanup, wording, style, or optional coverage improvement.

Useful dimensions: `BUG`, `API`, `BACKEND`, `CV`, `INFER`, `FORMULA`, `SOLVER`, `PERF`, `FALLBACK`, `TEST`, `DOC`, `ARTIFACT`, `MAINT`.

## 10. Report

Findings lead the response. Use:

```text
[SEVERITY][DIMENSION][status] path:line - issue
Impact: ...
Fix: ...
Evidence: ...
```

For `auto-fix`, also report changed files, tests/benchmarks run, approved exceptions/deferred items, and the remaining exit status. If no finding remains, say so and state residual unrun evidence rather than inventing certainty.
