---
name: code-review
description: Review statgpu changes for correctness, public API compatibility, statistical and numerical behavior, backend dispatch, CV/inference/resampling/formula contracts, tests, docs, evidence freshness, and performance. Use when the user asks for a review or audit, or when a development workflow explicitly requests a fresh independent review pass.
when_to_use: Trigger for PR review, audit, fresh review, an explicit review/fix pass, or a pre-implementation audit of a cross-cutting public statistical contract. Do not trigger merely because code is being edited.
argument-hint: "[audit|auto-fix|--fix] [PR|branch|range|path] [--comment]"
compatibility: "Claude Code >= 2.1.218 supports background: false explicitly; earlier versions block forked skills by default."
context: fork
background: false
---

# statgpu Code Review

Review `$ARGUMENTS` as an independent pass. Do not inherit an earlier clean verdict.

Before classifying findings:

1. read `dev/AGENTS.md` for project-specific contracts;
2. resolve the exact review target using [target-resolution.md](target-resolution.md);
3. record the resolved target identity before inspecting findings.

A review that cannot identify its target precisely must not return a clean/blocking verdict.

`background: false` explicitly keeps this fork blocking on Claude Code >= 2.1.218. Before v2.1.218, forked skills already blocked the invoking turn by default; no upgrade is required merely to preserve blocking behavior.

## Modes and bundled-override compatibility

- `audit`: findings only; do not edit files. This mode may also review an implementation plan/design contract before production edits.
- `auto-fix`: fix in-scope CRITICAL/HIGH issues and relevant MEDIUM issues, run targeted validation, then re-review the resulting state.
- `--fix` is accepted as an alias for `auto-fix` so the common bundled-skill spelling does not silently lose effect after the project skill overrides `/code-review`.
- `--comment` explicitly authorizes posting a review summary only when the resolved target is a PR. The published verdict must correspond to the PR's current **remote** exact head; local unpushed fixes may be mentioned only as draft work and must not be presented as if the PR itself were already fixed/clean.
- If no mode is supplied, infer `audit` for review-only requests and `auto-fix` only when the caller explicitly asks for fixes or a development workflow requests a review/fix pass.
- Bundled effort-tier flags are not part of statgpu's project review contract; if supplied, do not let them change target resolution or hard gates. Report unsupported/ignored review-control flags rather than silently assigning them project semantics.

## 1. Target freshness is a hard gate

The review report must identify the target using the fields defined in `target-resolution.md`, including `target_kind`, `base_sha`, `head_sha`, and `worktree_fingerprint` when dirty working-tree content is part of scope.

Before the final verdict, re-resolve the effective comparison **base and head** plus the audited working-tree content fingerprint where relevant. If either side of the diff or the audited content changed unexpectedly, discard the stale verdict and review the new state or report that the target moved.

In `auto-fix`, report the **post-fix** state separately from the original reviewed state. Do not claim that pre-fix CI or an earlier review proves a later head/fingerprint.

Evidence freshness is source-specific: hosted CI, GPU/R runs, benchmark artifacts, and prior reviews prove only the source identity/fingerprint they actually record. After the target moves, treat them as historical unless the artifact separately fingerprints the unchanged relevant source and its validator contract explicitly permits reuse.

## 2. Classify change type and impact before opening gates

Classify the change as one or more of:

- new numerical/statistical capability;
- existing capability reconciliation / public contract repair;
- API migration/deprecation;
- numerical refactor with preserved public behavior;
- docs/evidence-only change.

Then record the active axes:

- public API / compatibility / deprecation;
- backend, dtype, device, memory ownership, fallback;
- loss / objective;
- penalty;
- solver / convergence / stopping;
- CV / tuning / refit;
- inference / resampling;
- formula/model matrix;
- benchmark/performance;
- docs/evidence only.

Only activate gates that the change can affect. A public API or refactor-only change does **not** require new numerical backend, CV, or inference capability merely because it touches an existing estimator. It must instead prove that the existing affected capabilities and dispatch behavior remain unchanged where relevant.

That scope discipline does not permit a genuinely new public capability to define away repository defaults. New shared statistical/numerical capability defaults to NumPy/CuPy/Torch closure; new tunable loss x penalty capability defaults to direct-fit + CV closure unless the capability is genuinely non-tunable or an explicit approved deferral applies.

An existing capability reconciliation is not automatically a new numerical feature: preserve already-correct implementations, but require closure across every affected declared consumer and public claim.

A docs-only change becomes behavioral review scope if it changes a support, API, inference, backend, performance, release-boundary, or evidence claim.

## 3. Core review order

1. Target identity and freshness.
2. Current-versus-intended contract and consumer graph for shared/reconciliation work.
3. Correctness and public contract.
4. API migration/backward compatibility when active.
5. Repository-default capability closure for genuinely new numerical features.
6. Numerical/backend/CV/inference/resampling/formula gates that are active.
7. Tests and failure behavior.
8. Performance only after correctness passes.
9. Docs, release-boundary wording, evidence provenance, and maintainability.
10. Final exact-head freshness/evidence check.

Read [review-matrix.md](review-matrix.md) when the change spans multiple axes, changes public statistical behavior, reconciles an existing cross-cutting capability, or needs a blocking verdict.

## 4. Correctness

Check formulas, objective normalization, penalty scaling, gradients/Hessians/prox/KKT, prediction semantics, convergence status, sample weights, intercept handling, inference fields/targets, resampling semantics, and relevant edge cases.

When comparing with an external package, align the objective before changing production code. If statgpu uses an average loss and a reference uses a summed loss, compare at the equivalent regularization scale rather than changing statgpu's statistical definition just to match the reference.

Precision or convergence failure is a correctness issue, not a benchmark-only issue.

## 5. Public API migration, runtime installers, and reconciliation

When constructor parameters, method signatures, defaults, warnings, aliases, fitted attributes, public imports, shared dispatch, or runtime compatibility installers change, explicitly review:

- omitted argument versus explicitly supplied historical default;
- source and runtime constructor signature/introspection;
- `get_params()` and `set_params()`;
- sklearn `clone()` and meta-estimator reconstruction where applicable;
- warning category, message, stacklevel, and duplicate-warning noise;
- old/new parameter conflict handling;
- legacy numerical behavior during a deprecation window;
- fitted-state invalidation after parameter changes;
- public aliases and import paths;
- installer idempotence/import-order behavior and `__wrapped__`/introspection where applicable;
- source-static versus runtime-public documentation boundaries;
- migration docs/examples/changelog and planned removal boundary.

For shared base/mixin/dispatch changes, require a consumer inventory covering maintained public subclasses, CV/meta-estimators, formula paths, runtime installers, and docs/support matrices. A representative wrapper passing is not enough to prove the shared contract.

Do not silently reinterpret a deprecated parameter if that changes historical numerical behavior unless the migration explicitly authorizes that behavior change.

## 6. Backend and fallback review

For a **new or materially changed shared statistical/numerical capability**, the repository-default backend gate is NumPy + CuPy + Torch. Missing one is blocking unless the task/issue already defines a legitimate narrower capability or an explicit backend deferral has been approved with user-visible failure behavior and follow-up scope.

For existing-capability changes that can affect numerical execution or dispatch, verify the affected declared backends proportional to blast radius. API-only/refactor-only work verifies preservation; it does not require a new backend implementation merely because an estimator already exposes multiple devices.

Explicit `device="cuda"` or `device="torch"` must not silently substitute CPU. Approximation, fallback, dtype conversion, or device conversion must be visible through the public contract.

## 7. CV, inference, resampling, and formula

- A new **tunable loss x penalty capability** includes direct-fit + CV path/grid/folds/scoring/selection/final-refit closure by default. It may omit CV only when genuinely non-tunable or under an explicit approved deferral.
- For existing/API-only work, CV is blocking when the change can affect CV API, path/grid generation, fold scoring, selection, clone/reconstruction, final refit, or the stage at which inference runs.
- Inference is blocking when the current API/docs claim it or the change alters inference-facing behavior. Estimation-only can be legitimate when explicit and tested.
- Resampling is blocking when bootstrap/permutation/resampling is claimed or changed. Verify family/model-valid data generation, resampling unit, weights, refit estimator/penalty, whether tuning/selection is repeated or conditioned on, RNG semantics, backend, and target.
- Formula is blocking when the changed path is formula-facing or can alter model-matrix semantics.

For inference, explicitly verify:

- inferential target/estimand;
- public requested method -> resolved numerical method -> reported `_inference_result.method`/summary identity;
- covariance/bread/meat/reference distribution and penalty curvature where applicable;
- conditioning on fixed tuning, CV-selected tuning, active-set selection, or other data-dependent choices;
- whether shrinkage/selection/tuning uncertainty is corrected, conditioned on, or left uncorrected and documented;
- marginal versus simultaneous coverage;
- unsupported combinations fail closed before returning plausible-looking partial results.

A constructor default named after one inference procedure that silently executes/reports another is blocking unless the default is explicitly an `auto` resolver and the resolved method is visible to the user.

Do not expand scope solely because an external package offers an additional capability that statgpu does not claim.

## 8. Tests and evidence

Prefer the smallest test set that proves the active/default contracts, then widen when shared infrastructure is touched. Include negative tests for unsupported combinations, consumer-graph regressions for shared dispatch changes, and compatibility tests for public migrations.

For performance claims, require synchronized GPU timing, explicit timing scope, target scale, environment provenance, and numerical correctness before comparing speed.

Historical benchmark or GPU evidence proves only the source, validator contract, hardware, and environment it actually records.

## 9. Auto-fix loop

In `auto-fix` mode:

1. record the original exact target and working-tree fingerprint when applicable;
2. if the target is an explicit committed PR/branch target, require a clean writable worktree at its resolved head; if the target is the current no-scope working tree, capture its existing dirty content/fingerprint as `reviewed_before` instead of rejecting it;
3. record findings with severity and active dimension;
4. fix all in-scope CRITICAL/HIGH issues that can be fixed without an unrequested breaking/deferral decision;
5. fix MEDIUM issues that affect the requested feature or a completion gate;
6. run targeted tests/static checks/benchmarks as applicable;
7. update affected docs/support claims before the clean verdict when fixes changed public behavior;
8. resolve and record the new head/fingerprint;
9. re-review that new state from scratch.

Do not broaden an API cleanup into unrelated numerical refactoring. Do not commit, push, merge, or retarget branches unless the caller separately authorizes those repository actions.

## 10. Severity and report

- `CRITICAL`: wrong results/objective/inference, statistically invalid generic resampling accepted as valid, silent backend substitution, credential leak, or materially incorrect public results without detection.
- `HIGH`: wrong/stale review target; broken public compatibility; missing repository-default backend/CV closure without approved deferral; convergence failure; requested/resolved/reported inference-method mismatch; wrong/undocumented estimand; invalid family/model resampling; missing affected consumer closure for shared dispatch; missing active CV/inference/formula contract; or missing regression protection for changed public behavior.
- `MEDIUM`: important maintainability, docs/release-boundary, evidence, coverage, warning quality, or bounded performance issue that does not invalidate core behavior.
- `LOW`: small cleanup, wording, style, or optional coverage improvement.

Useful dimensions: `BUG`, `API`, `BACKEND`, `CV`, `INFER`, `RESAMPLE`, `FORMULA`, `SOLVER`, `PERF`, `FALLBACK`, `TEST`, `DOC`, `ARTIFACT`, `MAINT`, `TARGET`.

Findings lead the response. Include target identity, change type/active axes, consumer/contract observations when relevant, findings, validation, deferred evidence, docs/release-boundary status, and final exact-head freshness/evidence check. If no finding remains, say so only for the exact recorded target and state residual unrun evidence.
