---
name: code-review
description: Review statgpu changes for correctness, public API compatibility, statistical and numerical behavior, backend dispatch, CV/inference/formula contracts, tests, docs, and performance. Use when the user asks for a review or audit, or when a development workflow explicitly requests a final independent review pass. Do not use as a substitute for ordinary implementation work.
when_to_use: Trigger for PR review, audit, fresh review, or an explicit review/fix pass. Do not trigger merely because code is being edited.
argument-hint: "[audit|auto-fix] [scope]"
context: fork
background: false
---

# statgpu Code Review

Review `$ARGUMENTS` as an independent pass. Start from the current repository state rather than inheriting an earlier clean verdict.

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

Only activate gates that the change can affect. A public API or refactor-only change does **not** require new numerical backend, CV, or inference capability merely because it touches an estimator. It must instead prove that the existing declared capabilities and dispatch behavior remain unchanged where relevant.

A docs-only change becomes behavioral review scope if it changes a support, API, inference, backend, performance, or evidence claim.

## 2. Core review order

1. Correctness and public contract.
2. API migration/backward compatibility when active.
3. Numerical/backend/CV/inference/formula gates that are active.
4. Tests and failure behavior.
5. Performance only after correctness passes.
6. Docs, evidence provenance, and maintainability.

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

For changes that can affect numerical execution or dispatch, verify every **declared** backend affected by the change. New numerical/statistical capability should normally close NumPy/CuPy/Torch parity; API-only/refactor-only work should verify that existing backend claims and device routing were not changed accidentally.

Explicit `device="cuda"` or `device="torch"` must not silently substitute CPU. Approximation, fallback, dtype conversion, or device conversion must be visible through the public contract.

Do not require a new backend implementation for an inactive backend axis.

## 6. CV, inference, and formula

Activate these gates from the public contract, not from model-family stereotypes:

- CV is blocking when the change alters a tunable public capability, CV API, path/grid generation, fold scoring, selection, or final refit.
- Inference is blocking when the current API/docs claim it or the change alters inference-facing behavior. Estimation-only can be a legitimate design when it is explicit and tested.
- Formula is blocking when the changed path is formula-facing or can alter model-matrix semantics.

Do not expand scope solely because an external package offers an additional capability that statgpu does not claim.

## 7. Tests and evidence

Prefer the smallest test set that proves the active contracts, then widen when shared infrastructure is touched. Include negative tests for unsupported combinations and compatibility tests for public migrations.

For performance claims, require synchronized GPU timing, explicit timing scope, target scale, environment provenance, and numerical correctness before comparing speed.

Historical benchmark or GPU evidence proves only the source, validator contract, hardware, and environment it actually records.

## 8. Auto-fix loop

In `auto-fix` mode:

1. Record findings with severity and active dimension.
2. Fix all in-scope CRITICAL/HIGH issues that can be fixed without an unrequested breaking decision.
3. Fix MEDIUM issues that affect the requested feature or a completion gate.
4. Run targeted tests/static checks/benchmarks as applicable.
5. Re-review the affected surface from the new state.
6. Stop only when no unresolved CRITICAL/HIGH issue remains, or report the blocking decision/evidence gap.

Do not broaden an API cleanup into unrelated numerical refactoring.

## 9. Severity

- `CRITICAL`: wrong results/objective/inference, silent backend substitution, security/credential leak, or a public path can return materially incorrect results without detection.
- `HIGH`: broken public compatibility, required declared backend failure, convergence failure, missing active CV/inference/formula contract, serious regression, or missing tests that can hide a changed public behavior.
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

For `auto-fix`, also report changed files, tests/benchmarks run, deferred items, and the remaining exit status. If no finding remains, say so and state residual unrun evidence rather than inventing certainty.
