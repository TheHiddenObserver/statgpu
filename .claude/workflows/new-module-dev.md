# New Module Development Workflow

> Reusable workflow for adding or changing statistical methods in statgpu.
> Invoke with: `/new-module-dev <module_description>`.

## Purpose

This workflow is the hard protocol for automatic development. It defines what
must be true before work can be called complete, when work may continue
automatically, and when the agent must stop with a clear status.

Implementation details live in these skills:

- `.claude/skills/new-module-dev.md`
- `.claude/skills/code-review.md`
- `.claude/skills/benchmark.md`

## Hard Exit Contract

Every run must end with exactly one status:

- `COMPLETE`: All active blocking gates pass. Active statistical methods support
  `numpy`, `cupy`, and `torch`; precision and convergence pass when active;
  review has no unresolved CRITICAL or HIGH issues; docs and artifacts are
  updated.
- `PARTIAL_REMOTE_PENDING`: Local work is complete, but remote GPU, R/external
  framework, or large benchmark validation is unavailable. Report exact skipped
  commands, missing resources, and the remaining validation plan. Do not claim
  full GPU validation.
- `BLOCKED_NEEDS_USER_APPROVAL`: Continuing requires a user decision, such as
  backend deferral, public API break, performance caveat, credential setup,
  commit, push, PR, merge, release, or package upload. State the recommended
  option and the tradeoff.
- `FAILED`: A blocking gate remains unresolved: missing backend, incorrect
  formula, incorrect loss/gradient/Hessian, precision or convergence failure,
  silent fallback, unresolved CRITICAL/HIGH review issue, or unsafe artifact.

Do not end with informal statuses such as "mostly done" or "needs more work".

## Non-Negotiable Rules

- All statistical methods touched by this workflow, including new or changed
  methods, MUST implement all three backends: `numpy`, `cupy`, and `torch`.
- CPU-only work is incomplete. Correctness may come before performance, but the
  workflow cannot stop after only the numpy path.
- Backend deferral requires explicit user approval and must record the reason,
  user-visible failure behavior, test skip condition, and follow-up task.
- Silent fallback is forbidden. Backend fallback, approximate inference, dtype
  changes, or CPU fallback from GPU must be explicit in the API contract and
  visible in outputs or reports.
- Precision and convergence are blocking before performance optimization.
- Public loss x penalty model support includes the CV layer. If a loss/penalty
  combination is supported by `fit()`, its CV path must also be implemented and
  tested unless the feature is explicitly non-tunable or the user approves
  deferral.
- Public statistical estimation support includes inference when the estimator
  exposes `compute_inference`, `summary()`, covariance, standard errors,
  p-values, confidence intervals, or when inference is standard for the model
  family. Otherwise, the feature must be explicitly documented and tested as
  estimation-only.
- Commits, pushes, PRs, merges, tags, releases, and package publication require
  an explicit user request.
- Credentials must not be read from memory, Markdown, or `.claude/settings.json`.
  Remote work must use `dev/scripts/remote_config.py` or an untracked local
  config/environment variable.

## Required Grounding

Before editing code, read the relevant subset of:

- `dev/AGENTS.md`
- target package code
- existing tests
- existing docs and changelog conventions
- existing benchmarks or comparable scripts
- external baseline docs/code when the request names one

If the repo contradicts this workflow, follow this workflow and report the
contradiction.

## Impact Classification

Before implementing, classify the change and activate only the relevant gates.
Always report the classification in the completion report.

| Impact axis | Activate when touched |
| --- | --- |
| Public API | public imports, constructor args, return types, errors, docs examples |
| Backend | dtype/device behavior, kernels, array helpers, memory ownership, fallback |
| Loss | loss formula, gradient, Hessian, Lipschitz, registry, GLM family |
| Penalty | value, gradient, proximal, LLA, group/adaptive setup, categories |
| Solver | optimizer logic, convergence, dispatch, stopping rules |
| Loss x penalty model | public penalized fit capability or wrapper behavior |
| CV | alpha/lambda/C paths, folds, scoring, refit, CV dispatch |
| Inference | `compute_inference`, covariance, p-values, CI, summary/result objects |
| Formula | formula parsing, model matrix, feature names, intercept semantics |
| Benchmark/performance | kernels, fast paths, scaling claims, external timing comparisons |
| Docs-only | documentation or changelog without behavior changes |

If classification is uncertain, choose the broader relevant set and state why.
Docs-only changes do not activate code gates unless they change support claims.

## Phase Order

1. Classify the impact axes and choose active gates.
2. Define the public API, shape, dtype, device, error, formula, and fallback
   contract.
3. Implement the numpy baseline.
4. Implement the CuPy path.
5. Implement the Torch path.
6. Add or update three-backend consistency tests.
7. Add or update architecture-specific compatibility tests for the touched
   component type.
8. Validate against external baselines, including objective scaling and penalty
   scale mapping.
9. Run precision and convergence gates.
10. Run performance benchmark and the bounded optimization loop when the
    benchmark/performance gate is active.
11. Run `code-review` in `auto-fix` mode and re-test.
12. Update docs, changelog, exports, and benchmark artifacts.
13. Report one Hard Exit Contract status.

## Blocking Gates

| Gate | Blocking? | Deferrable? | Required Evidence |
| --- | --- | --- | --- |
| Impact classification | Yes | No | active axes and inactive gates with reasons |
| Three backends | Yes | User approval only | numpy/cupy/torch tests or explicit approved deferral |
| Public API contract | Yes | No | documented inputs, outputs, errors, dtype/device behavior |
| Architecture-specific matrix | Yes | User approval only | required compatibility tests for the touched component type |
| CV layer parity | Yes for tunable model capabilities | User approval only | CV fit/refit/alpha-or-C selection across supported backends |
| Inference layer parity | Yes for inferential model capabilities | User approval only | inference fields, summary/result container, external baseline, backend parity |
| R formula compatibility | Yes when formula-facing | User approval only | formula/model-matrix tests or documented unsupported syntax |
| Objective scaling | Yes | No | loss normalization and equivalent penalty mapping, e.g. `lambda_external = n * lambda` when needed |
| Precision | Yes | No | external or analytic baseline, tolerance source, failure diagnosis |
| Convergence | Yes | No | solver status, monotonicity/KKT/gradient checks where applicable |
| Silent fallback | Yes | No | explicit warnings/results fields for fallback or no fallback used |
| Performance | Conditional | Yes with caveat | benchmark JSON, baseline comparison, optimization notes |
| Review | Yes | No for CRITICAL/HIGH | `code-review` auto-fix report |
| Docs/artifacts | Yes | No | docs/changelog/export/benchmark files or explicit not-applicable reason |
| Remote GPU/large benchmark | No for local completion | Yes | `PARTIAL_REMOTE_PENDING` with exact follow-up command |

## Capability Table

Maintain an explicit capability decision for each public model family or
wrapper touched by the change.

| Capability | Allowed values | Meaning |
| --- | --- | --- |
| backend | `three-backend`, `approved-deferral` | whether numpy/cupy/torch are implemented |
| CV | `supported`, `non-tunable`, `planned`, `approved-deferral` | whether CV must be implemented now |
| inference | `supported`, `estimation-only`, `planned`, `approved-deferral` | whether inference must be implemented now |
| formula | `supported`, `not-formula-facing`, `planned`, `approved-deferral` | whether formula tests are active |
| benchmark | `required`, `not-performance-sensitive`, `remote-pending` | whether benchmark evidence is needed |

`planned` is not a completion status by itself. It requires a user-visible
failure mode, tests, docs, and a follow-up task. User approval is required when
`approved-deferral` changes expected public behavior.

## Validation Tiers

Use the highest tier available and report it.

- `local-minimal`: local imports, targeted numpy tests, unavailable-backend
  errors/skips, active API/error tests, and no unsafe artifacts.
- `local-full`: `local-minimal` plus every locally available backend, active
  matrix tests, local external Python baselines, and benchmark JSON key checks.
- `remote-full`: `local-full` plus remote GPU, R/external packages, or large
  benchmarks through `dev/scripts/remote_config.py`.

`COMPLETE` requires `local-full` for active gates. If only remote-only evidence
is missing, end as `PARTIAL_REMOTE_PENDING`. If local required evidence is
missing, end as `FAILED` or `BLOCKED_NEEDS_USER_APPROVAL`.

## External Baseline Hierarchy

Use the strongest available baseline and record the choice:

1. Analytic closed form or derivative check.
2. Existing trusted statgpu implementation.
3. Python reference: sklearn, statsmodels, scipy, lifelines, patsy.
4. R reference package.
5. Numerical invariants: finite differences, KKT, monotonic objective,
   simulation coverage, backend parity.

Do not block indefinitely on an unavailable external package. Use the next
baseline tier, record the missing package/command, and mark remote or external
work pending when appropriate.

## Architecture-Specific Test Matrix

The package is organized around pluggable axes: losses, penalties, solvers,
backends, formula/model-matrix parsing, inference, CV, wrappers, survival, and
nonparametric/unsupervised estimators. A change must test the axes it touches.

| Touched component | Required extra tests |
| --- | --- |
| New or changed loss | registry/export, per-sample value/gradient, fused value+gradient parity, finite-difference gradient, Hessian/Lipschitz when advertised, three-backend parity, supported penalties through `solver='auto'`, explicit solver constraints, external/analytic baseline |
| New or changed penalty | registry/export, value/gradient/prox/LLA semantics, category constants, three-backend parity, compatibility with representative losses, solver auto-dispatch, explicit solver rejection for unsupported combinations, group/adaptive init when applicable |
| New or changed solver | smooth/non-smooth/nonconvex/group penalty matrix, monotonic objective or KKT evidence, convergence status, explicit incompatibility errors, three-backend parity, auto-dispatch/CV dispatch if used |
| Public loss x penalty capability | direct fit plus CV layer, alpha/lambda/C path generation, fold scoring, best parameter selection, refit behavior, inference after direct fit and CV refit when supported, backend parity, external/sklearn/R CV baseline when available |
| New estimator or wrapper | sklearn-style fit/predict/score/get_params behavior, public exports, formula path if exposed, docs examples, external baseline, three-backend parity, inference/CV tests when those flags are supported |
| CV path | deterministic folds, sample weights, alpha grid/path, best score/refit behavior, no leakage, backend selection, GPU/CPU parity, external baseline when available |
| Inference path | strict/default behavior, estimation-only errors when unsupported, fallback visibility, `_inference_result` container, summary output, coef/bse/t-or-z/p/CI/AIC/BIC/LLF fields where applicable, backend parity, external baseline |
| Formula path | intercept, categorical reference levels, interactions/transforms, missing data, column names/order, R or patsy baseline |
| Backend helper or kernel | dtype/device preservation, no hidden host transfer, memory cleanup when owning GPU buffers, synchronization-sensitive benchmark, numpy/cupy/torch parity |
| Survival path | censoring, risk sets, Breslow/Efron ties, gradient/Hessian checks, external lifelines/R baseline, backend parity |
| Nonparametric or unsupervised estimator | sklearn-style API, transform/predict semantics, random_state determinism, external/sklearn baseline, large-scale/memory behavior, backend parity |

For broad cross-axis changes, update or extend a matrix test such as
`dev/tests/test_loss_penalty_solver_matrix.py` and a precision/benchmark script
covering the affected loss x penalty x solver x backend combinations.

## Optimization Budget

Performance work is mandatory when CPU is slower than an external framework, GPU
is slower than CPU at target scales, or benchmark data contradicts expected
acceleration.

Target scale must be explicit. Prefer an existing benchmark scale for the
module; otherwise report `target_scale_source="temporary"` with the data shape,
dtype, backend, and rationale. Do not make performance claims without a target
scale and timing scope.

Use this bounded loop:

1. One profiling pass to identify bottlenecks.
2. Up to two algorithmic or kernel optimization attempts.
3. One re-benchmark per attempt.
4. If still failing, report `BLOCKED_NEEDS_USER_APPROVAL` or
   `PARTIAL_REMOTE_PENDING` with `optimization_notes`, `cpu_vs_external`,
   `gpu_vs_cpu`, and `crossover_n`.

Do not keep optimizing indefinitely without user input.

## Required Completion Report

The final report must include:

- Hard Exit Contract status.
- Impact classification and validation tier.
- Files changed.
- Backends implemented and tested.
- Architecture-specific compatibility tests added or updated.
- CV layer status for tunable loss x penalty capabilities.
- Inference layer status, or explicit estimation-only contract.
- Precision/convergence result and threshold source.
- Objective scaling and penalty mapping decisions.
- Formula compatibility result when applicable.
- Performance result, benchmark artifact path, and optimization notes.
- Review-and-fix result.
- Tests or benchmarks run, including skipped remote work and why.
