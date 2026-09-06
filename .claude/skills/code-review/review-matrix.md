# statgpu Review Matrix

Load this file from `SKILL.md` when a review spans multiple axes or needs a blocking verdict.

## Impact-axis matrix

| Active axis | Required review evidence |
| --- | --- |
| Public API / deprecation | Signature/introspection, defaults, old/new conflict behavior, warnings, `get_params`, `set_params`, clone/meta-estimator behavior, fitted-state invalidation, migration docs |
| Backend / device | For new shared numerical capability: NumPy/CuPy/Torch closure by default or approved deferral; otherwise affected declared routing, dtype/device ownership, no hidden host transfer, explicit unavailable-backend behavior, output ownership |
| Loss / objective | Formula, normalization, intercept policy, gradient/Hessian or finite-difference checks, external/analytic baseline |
| Penalty | Value/gradient/prox/LLA semantics, parameter validation, solver compatibility, objective scaling |
| Solver | Dispatch, supported objective classes, convergence/KKT/monotonicity, stopping semantics, backend compatibility, CV dispatch if shared |
| CV | For new tunable loss x penalty capability: direct + CV closure by default or approved non-tunable/deferral contract; otherwise grid/path generation, folds, sample weights, scoring, selection, determinism, no leakage, final refit, backend/device contract |
| Inference | Result container/fields, covariance, reference distribution, SE/t-or-z/p/CI, summary, sample weights, backend provenance, strict/fallback behavior |
| Formula | Intercept, categorical reference levels, interactions/transforms, missing-row alignment, feature names/order, prediction matrix behavior |
| Performance | Correctness first, synchronized timing, environment provenance, transfer scope, target scale, external comparison identity |
| Docs/evidence | Current support claim, exact-source provenance, EN/CN conceptual parity when both pages exist, no unsupported performance/statistical claim |

## Change-type rules

### New numerical/statistical capability

For a new or materially changed **shared statistical/numerical capability**, statgpu's default completion contract is NumPy + CuPy + Torch. Missing one is blocking unless the task/issue already defines a legitimate narrower capability or an explicit backend deferral has been approved with reason, user-visible failure behavior, deterministic tests/skips, and follow-up scope.

For a new **tunable loss x penalty capability**, direct fit + CV path/grid/folds/scoring/selection/final refit are part of the default completion contract. CV may be absent only when the capability is genuinely non-tunable or an explicit deferral is approved.

Do not let a first implementation silently redefine a repository-default capability as CPU-only or direct-fit-only merely to avoid these gates. Intermediate partial work is allowed; calling it `COMPLETE` is not.

Do not use a new capability as an excuse to silently fallback from an explicit GPU device to CPU.

### API-only/refactor/deprecation change

Do **not** demand new backend, CV, inference, formula, or benchmark capability merely because the edited class already has those features.

Instead verify:

- dispatch and numerical paths are unchanged unless intentionally modified;
- existing public support claims remain true;
- relevant targeted regressions cover the touched interfaces;
- shared infrastructure changes receive broader regression coverage proportional to blast radius.

This exemption applies to genuinely narrow existing-capability work; it does not weaken the default closure rules above for newly introduced public capability.

### Docs-only change

No runtime gate is activated by prose alone unless the edit changes a public capability/support/evidence claim. If it does, verify the implementation/evidence that supports the new claim.

## Public API migration checklist

Review every applicable item:

- [ ] Old code that omits the changed parameter behaves as before unless the change intentionally changes the default.
- [ ] Explicit historical default is distinguished from omission when deprecation semantics require that distinction.
- [ ] `inspect.signature` and public docs agree.
- [ ] `get_params(deep=False)` exposes the intended public constructor contract.
- [ ] `set_params` accepts/rejects old/new names consistently and invalidates stale fitted state where required.
- [ ] sklearn `clone` works on maintained sklearn versions when estimator compatibility is claimed.
- [ ] Framework-internal reconstruction does not spam user-facing deprecation warnings.
- [ ] Warning category and stacklevel point users to their call site.
- [ ] Deprecated aliases preserve documented legacy behavior during the compatibility window.
- [ ] Conflicting old/new arguments fail clearly instead of choosing one silently, unless preserving a documented historical override is part of the migration.
- [ ] Final fitted attributes do not expose stale semantics from a deprecated control.
- [ ] Migration docs show the replacement and removal horizon.
- [ ] Tests cover omission, explicit old default, non-default old value, new value, conflict, and clone/set_params where applicable.

## Statistical correctness checks

When active, check the strongest applicable evidence:

1. Analytic closed form or identity.
2. Trusted existing statgpu path.
3. External Python reference such as sklearn/statsmodels/scipy/lifelines.
4. Authoritative R package/reference.
5. Numerical invariants such as finite differences, KKT, monotonic objective, coverage, or backend parity.

External agreement is meaningful only after aligning objective normalization, penalty scaling, feature set, weighting, ties, solver, and convergence settings.

## Backend checks

When backend behavior is active:

- for new shared numerical capability, NumPy/CuPy/Torch are all covered unless an approved narrower scope exists;
- explicit CPU/CUDA/Torch requests have deterministic routing;
- `device="auto"` is the only path that may select among available backends automatically;
- unavailable explicit backends fail visibly;
- arrays are not copied to host merely to reuse a CPU implementation unless that transfer is an explicit public path;
- random seeds and dtype conversions remain comparable across backends;
- cleanup does not destroy state required by later predict/score/inference calls.

## CV checks

When CV is active:

- for a new tunable loss x penalty capability, direct + CV closure is default unless a non-tunable/approved-deferral contract applies;
- fold construction and custom splits are validated;
- sample weights align with the selected objective;
- alpha/lambda/C grids use the correct public scale;
- candidate failure handling does not hide infrastructure/programming errors;
- selection uses finite evidence and deterministic tie rules where defined;
- final refit uses the selected hyperparameter and intended final-refit solver/device;
- CV-only controls do not leak into final-estimator semantics unless documented;
- inference, when supported after CV, runs only at the contractually defined stage.

## Inference checks

When inference is active:

- `compute_inference` and `summary()` have clear behavior before/after fit;
- coefficient indexing includes intercept/formula names correctly;
- covariance and reference-distribution choices are explicit;
- `_inference_result` and established public/reporting fields agree;
- unavailable or unsupported inference fails explicitly rather than returning plausible-looking partial data;
- strict/fallback modes are distinguishable;
- backend-native claims record the actual numerical backend/device, not merely the input type.

## Formula checks

When formula behavior is active:

- formula and array routes agree after design-matrix construction;
- intercept semantics are stable;
- categorical reference levels and column ordering are deterministic;
- interactions/transforms used by project docs are supported;
- missing-data row filtering is aligned across X/y/weights;
- prediction reconstructs columns in training order.

## Performance checks

Performance findings require measurement, not intuition. Record:

- exact commit and clean/dirty tree status;
- Python/statgpu/NumPy/CuPy/Torch versions;
- CUDA/driver, CPU/GPU model, dtype, data shape;
- warmup, repeats, seeds;
- timing boundaries and synchronization;
- transfer policy;
- numerical error versus the comparison path;
- target scale and whether the comparison uses the same algorithm/objective.

Do not aggregate speedups from different hardware/software environments as if they were one homogeneous measurement.

## Severity examples

### CRITICAL

- wrong coefficient/objective/inference result accepted as valid;
- explicit GPU request silently executes on CPU;
- candidate/error fallback hides OOM/device/programming failure;
- secrets or credentials enter tracked artifacts.

### HIGH

- a new shared numerical capability is declared complete without NumPy/CuPy/Torch or an approved deferral;
- a new tunable loss x penalty capability is declared complete without CV closure or an approved non-tunable/deferral contract;
- deprecation breaks clone/set_params or legacy numerical behavior without an approved break;
- a declared backend path no longer works;
- CV selects/refits the wrong hyperparameter/solver;
- inference fields are stale, mis-indexed, or unavailable despite the public contract;
- public behavior changes without regression tests.

### MEDIUM

- warning is noisy or points to internal code;
- active docs/API inventory is incomplete;
- evidence provenance is ambiguous but not used to justify a wrong result;
- maintainability problem materially raises future regression risk.

### LOW

- wording/style cleanup;
- optional test simplification;
- non-blocking refactor suggestion outside the requested scope.
