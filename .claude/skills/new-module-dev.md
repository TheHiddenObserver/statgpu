---
name: new-module-dev
description: Automatically develop statgpu methods with mandatory numpy/cupy/torch support, tests, docs, benchmark, and review-fix loop
---

# New Module Development Skill

Use this skill when the user asks to add or substantially change a statistical
method, estimator, loss, kernel, inference path, or backend-aware API.

This skill implements `.claude/workflows/new-module-dev.md`. That workflow is
authoritative for blocking gates and exit status.

## Operating Mode

This is an automatic development skill. It may edit source, tests, docs,
exports, and benchmark scripts when invoked for implementation work.

It must not commit, push, open PRs, merge, tag, release, or publish/upload
packages unless the user explicitly asks for that action.

## Hard Rules

- Every statistical method touched by this workflow MUST implement `numpy`,
  `cupy`, and `torch`.
- Backend deferral requires explicit user approval and must record reason,
  failure behavior, test skip condition, and follow-up task.
- Silent fallback is forbidden. Fallback or approximate inference must be part
  of the public contract and visible in results.
- Precision and convergence failures are correctness failures. Fix these before
  performance work.
- Public loss x penalty support includes CV. If the direct estimator supports a
  tunable loss/penalty combination, the corresponding CV path must be
  implemented and tested in the same development flow unless the combination is
  explicitly non-tunable or the user approves deferral.
- Public statistical estimators must implement inference when they expose
  `compute_inference`, `summary()`, covariance, standard errors, p-values,
  confidence intervals, or when inference is standard for the model family. If
  inference is not implemented, add an explicit estimation-only contract,
  user-visible error behavior, tests, docs, and follow-up task.
- Credentials must not come from memory, Markdown, or `.claude/settings.json`.
  Remote execution must use `dev/scripts/remote_config.py` or untracked local
  config/environment variables.

## Required Grounding

Read the relevant subset before editing:

- `dev/AGENTS.md`
- target module source and exports
- existing tests for nearby modules
- existing docs, changelog, and examples
- existing benchmarks in `dev/benchmarks/` or comparable scripts
- external reference implementation, paper, package docs, or R behavior when
  named by the task

Prefer existing project patterns over new abstractions.

## Execution Order

### 0. Impact Classification

Before editing, classify the touched axes:

- public API
- backend
- loss
- penalty
- solver
- loss x penalty model
- CV
- inference
- formula
- benchmark/performance
- docs-only

Activate only the gates for touched axes. If unsure, choose the broader
classification and explain why. Docs-only changes do not activate code gates
unless they change support claims.

Record a capability decision for touched public model families:

- backend: `three-backend` or `approved-deferral`
- CV: `supported`, `non-tunable`, `planned`, or `approved-deferral`
- inference: `supported`, `estimation-only`, `planned`, or `approved-deferral`
- formula: `supported`, `not-formula-facing`, `planned`, or `approved-deferral`
- benchmark: `required`, `not-performance-sensitive`, or `remote-pending`

`planned`, `estimation-only`, and `non-tunable` require visible API behavior,
tests, docs, and a follow-up task when they affect a public capability.

### 1. API Contract

Define and document:

- function/class name, public import path, and backward compatibility
- input shape, dtype, device, missing values, sparse/dense support
- output type, backend ownership, and whether outputs stay on CPU/GPU
- error behavior for unavailable backends or unsupported options
- formula/model-matrix compatibility when the method is formula-facing
- fallback behavior, warnings, and result fields

Changing a public API incompatibly requires user approval unless the user
explicitly requested the breaking change.

### 2. Three-Backend Implementation

Implement in this order:

1. Numpy correctness baseline.
2. CuPy path with no hidden CPU fallback.
3. Torch path with no hidden CPU fallback.

Use shared helpers only when they reduce real duplication and preserve dtype,
device, and backend semantics. Keep backend-specific code explicit when kernels,
autograd, memory layout, or synchronization differ.

### 3. Tests

Add or update targeted tests for:

- numpy/cupy/torch parity
- dtype and device behavior
- invalid input and missing dependency behavior
- fallback visibility, if fallback is supported
- formula/model-matrix behavior, if applicable
- objective scaling and penalty mapping against external baselines
- precision, convergence, gradients, Hessians, KKT, or monotonic loss where
  relevant
- inference fields and summary behavior when inference is supported
- explicit estimation-only errors when inference is not supported

Skip GPU tests only when the backend is unavailable, and make the skip reason
specific.

Use validation tiers:

- `local-minimal`: imports, targeted numpy tests, unavailable-backend behavior,
  active API/error tests, and no unsafe artifacts.
- `local-full`: `local-minimal` plus locally available backends, active matrix
  tests, local external Python baselines, and benchmark JSON key checks.
- `remote-full`: `local-full` plus remote GPU, R/external packages, or large
  benchmarks through `dev/scripts/remote_config.py`.

Completion requires `local-full` for active gates unless only remote-only
evidence is missing.

### 3a. Architecture-Specific Compatibility Tests

Use the workflow architecture matrix for the active axes. Minimum expectations:

- loss changes cover registry/export, value/gradient/fused parity, finite
  differences, representative penalties, CV if tunable, and inference if
  inferential
- penalty changes cover registry/export, category constants, value/gradient/prox
  or LLA, representative losses, CV if tunable, and inference if inferential
- solver changes cover smooth, non-smooth, nonconvex, group/adaptive penalties,
  convergence/KKT evidence, and auto-dispatch/CV dispatch when selectable
- public loss x penalty capabilities cover direct fit, CV refit, backend parity,
  equivalent penalty scaling, and inference when `compute_inference=True`
- estimator/wrapper, formula, backend/kernel, survival, nonparametric, and
  unsupervised changes follow their corresponding workflow matrix row

For loss x penalty x solver changes, extend matrix coverage such as
`dev/tests/test_loss_penalty_solver_matrix.py`. For CV-facing combinations,
extend CV tests near existing `*_cv*` tests or add a targeted CV matrix test.
For inference-facing combinations, extend inference tests near existing
`test_*inference*` files or add a targeted inference matrix test.
For precision/performance, extend a benchmark or validation script and write
results to `results/*.json` when it is run.

### 3b. Inference Layer Tests

When inference is supported or expected, test direct estimators, CV final refit,
`_inference_result`, `summary()`, parameter fields, covariance options, sample
weights, intercept/formula feature names, backend parity, external baselines,
and strict/fallback behavior.

For unsupported inference, test that `compute_inference=True` or `summary()`
raises a clear error or warning-backed result. Do not silently return missing
fields.

### 4. Formula Compatibility Gate

For formula-facing methods, validate compatibility with R-style expectations:

- intercept handling
- categorical encoding and reference levels
- interactions and transforms supported by the project
- missing-data behavior
- model matrix column names and ordering

If syntax is unsupported, document the exact unsupported feature and failure
mode. Unsupported formula behavior is blocking unless approved by the user.

### 5. Precision and Convergence Gate

Before changing algorithms for speed, diagnose numerical correctness:

- compare loss/objective, coefficients, predictions, gradients, Hessians, and
  inference outputs against analytic or external baselines
- distinguish loss formula bugs from optimizer or convergence bugs
- check optimizer status, iteration counts, tolerances, and stopping criteria
- compare backend-to-backend precision
- record the tolerance source

Use this external baseline hierarchy:

1. Analytic closed form or derivative check.
2. Existing trusted statgpu implementation.
3. Python reference: sklearn, statsmodels, scipy, lifelines, patsy.
4. R reference package.
5. Numerical invariants: finite differences, KKT, monotonic objective,
   simulation coverage, backend parity.

If the strongest baseline is unavailable, use the next tier and record the
missing package, command, or remote requirement.

When statgpu optimizes `n^{-1} sum_i loss_i + lambda * penalty` but an external
framework optimizes `sum_i loss_i + lambda * penalty`, align tests with an
equivalent penalty scale such as `lambda_external = n * lambda`. Do not modify
statgpu loss definitions just to mimic a differently normalized baseline.

### 6. Performance and Algorithm Optimization Gate

Run the benchmark skill when performance matters, when a new kernel/path is
added, or when baseline comparison is part of the request.

Performance action is required when:

- CPU is materially slower than an established external framework at comparable
  settings
- GPU is slower than CPU at target problem sizes
- memory transfers dominate expected GPU speedups
- asymptotic complexity is worse than the known/reference algorithm

Define the target scale before optimizing. Prefer an existing benchmark scale
for the module. Otherwise record `target_scale_source="temporary"` with data
shape, dtype, backend, and rationale.

Use the workflow optimization budget: one profiling pass, up to two algorithmic
or kernel optimization attempts, and one re-benchmark per attempt. If still
failing, stop with `BLOCKED_NEEDS_USER_APPROVAL` or `PARTIAL_REMOTE_PENDING`.

### 7. Review-And-Fix Loop

Before completion, invoke `code-review` in `auto-fix` mode:

1. Review correctness, backend parity, API, formula, precision, convergence,
   performance, tests, docs, and artifacts.
2. Fix CRITICAL and HIGH issues.
3. Fix relevant MEDIUM issues when they affect the requested feature.
4. Re-run targeted tests and benchmark checks.
5. Re-review until no unresolved CRITICAL/HIGH issues remain or a hard exit
   status is required.

### 8. Docs And Artifacts

Update only relevant files:

- package exports and `__init__.py`
- user docs or examples
- changelog or development notes if the repo uses them
- benchmark script in `dev/benchmarks/`
- benchmark JSON under `results/` when benchmark is run

Documentation format:

- English first; add Chinese notes only when the surrounding docs already use
  bilingual style or the user asks for it.
- Start with the statistical model and objective, then API, backend behavior,
  formula behavior, examples, numerical notes, and limitations.
- Mention objective scaling and penalty mapping when external comparisons are
  documented.
- Do not include credentials, host-specific secrets, or private remote paths.

## Completion Report

End with exactly one workflow status:

- `COMPLETE`
- `PARTIAL_REMOTE_PENDING`
- `BLOCKED_NEEDS_USER_APPROVAL`
- `FAILED`

Report:

- changed files
- impact classification, capability decisions, and validation tier
- implemented backends
- tests and benchmarks run
- architecture-specific compatibility tests added or updated
- CV implementation/test status for tunable loss x penalty capabilities
- inference implementation/test status, or explicit estimation-only contract
- precision/convergence status and threshold source
- formula compatibility status when applicable
- objective scaling and penalty mapping
- performance result and benchmark artifact path
- review-and-fix result
- skipped validation and exact follow-up commands
