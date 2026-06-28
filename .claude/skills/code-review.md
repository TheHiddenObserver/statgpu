---
name: code-review
description: Review or auto-fix statgpu code across correctness, three-backend behavior, inference, maintainability, and performance
---

# Code Review Skill

Use this skill for statgpu source, tests, benchmarks, docs, and workflow
artifacts.

## Modes

- `audit`: Use when the user explicitly asks only for review, audit, inspection,
  or findings. Do not edit files. Findings lead the response.
- `auto-fix`: Use when the user asks to develop, implement, fix, continue
  automatic development, or when called by `new-module-dev`. Fix CRITICAL and
  HIGH issues, fix relevant MEDIUM issues, run targeted checks, and re-review.

If a request includes implementation work, choose `auto-fix`.

## Hard Review Gates

### Scope And Gate Selection

Review the impact classification before reviewing findings. The report should
state active axes such as backend, loss, penalty, solver, CV, inference,
formula, benchmark/performance, or docs-only.

Findings should map to active gates. If a gate is inactive, verify the reason is
credible. If the change alters public support claims, docs-only classification
is invalid.

Review capability decisions for touched public model families:

- backend: `three-backend` or approved deferral
- CV: supported, non-tunable, planned, or approved deferral
- inference: supported, estimation-only, planned, or approved deferral
- formula: supported, not-formula-facing, planned, or approved deferral
- benchmark: required, not-performance-sensitive, or remote-pending

Missing impact classification or missing capability decision is MEDIUM. If it
hides a missing backend, CV, inference, formula, or precision gate, escalate to
HIGH or CRITICAL according to impact.

### Correctness

Check formulas, loss normalization, penalties, gradients, Hessians, likelihoods,
degrees of freedom, KKT checks, prediction semantics, inference outputs, and
edge cases. Precision or convergence failure is a correctness issue, not a
benchmark-only issue.

When comparing to external frameworks, verify objective scaling before changing
code. If statgpu uses `n^{-1} sum_i loss_i + lambda * penalty` and the external
framework uses `sum_i loss_i + lambda * penalty`, tests should compare with an
equivalent penalty scale such as `lambda_external = n * lambda`.

For statistical estimators, review whether inference is part of the public
capability. If the estimator exposes `compute_inference`, `summary()`,
covariance, standard errors, p-values, confidence intervals, or a standard
external analogue provides inference, estimation-only implementation is
incomplete unless explicitly documented and tested as estimation-only.

### Three-Backend Behavior

Every touched statistical method must support `numpy`, `cupy`, and `torch`.
Review dtype, device, memory ownership, random seeds, error behavior, and output
contracts for each backend. CPU-only additions are HIGH or CRITICAL depending on
whether the public feature is unusable on GPU.

Silent fallback is forbidden. Any fallback or approximate path must be explicit
in API behavior and visible in results or reports.

### Architecture-Specific Coverage

Review tests against the component type touched by the change:

- Loss changes must cover registry/export, value/gradient/fused parity,
  finite-difference gradients, advertised Hessian/Lipschitz behavior,
  three-backend parity, representative penalties, solver auto-dispatch, and an
  external or analytic baseline. If the loss is part of a tunable penalized
  model, it must also have CV coverage for representative penalties. If it is
  part of an inferential estimator, it must have inference coverage or explicit
  unsupported-inference behavior.
- Penalty changes must cover registry/export, category constants,
  value/gradient/prox/LLA semantics, three-backend parity, representative
  losses, solver auto-dispatch, and explicit unsupported solver errors. If the
  penalty is part of tunable models, it must also have CV coverage for
  representative losses. If it is part of inferential models, it must also have
  inference coverage or explicit unsupported-inference behavior.
- Solver changes must cover smooth, non-smooth, nonconvex, group, and adaptive
  penalty cases, convergence/KKT or monotonic objective evidence, three-backend
  parity, and CV/auto-dispatch if applicable.
- Public loss x penalty support must cover both direct fit and CV unless the
  feature is explicitly non-tunable or the user approved deferral. Review CV
  grid/path generation, fold scoring, best parameter selection, refit behavior,
  sample weights, deterministic seeds, backend parity, inference after final
  refit when `compute_inference=True`, and external CV baseline or penalty-scale
  mapping when available.
- Inference-capable models must test `_inference_result`, `summary()`,
  `_params`, `_bse`, `_tvalues` or `_zvalues`, `_pvalues`, `_conf_int`,
  covariance options, sample weights, intercept/formula feature names, backend
  parity, external baseline, and explicit estimation-only errors when
  unsupported.
- Estimator, wrapper, CV, inference, formula, backend, survival,
  nonparametric, and unsupervised changes must satisfy the corresponding
  workflow matrix row.

Missing matrix coverage for a changed public statistical component is HIGH.
Missing CV coverage for a public tunable loss x penalty capability is HIGH.
Missing inference coverage for a public inferential estimator is HIGH.
Missing coverage that allows wrong coefficients, wrong inference, wrong selected
regularization parameter, silent fallback, or missing required backend is
CRITICAL.

### Formula Compatibility

For formula-facing methods, check R-style expectations:

- intercept handling
- categorical encoding and reference levels
- interactions and transforms supported by the project
- missing-data behavior
- model matrix column names and ordering

Unsupported formula behavior must be documented with a clear failure mode.

### Performance And Algorithm Choice

Review whether the algorithmic complexity matches the reference approach and
whether backend-specific implementation choices cause unnecessary transfers,
synchronization, scalar loops, or dense materialization.

Performance is actionable when:

- CPU is materially slower than an external framework
- GPU is slower than CPU at target sizes
- GPU only wins at scales that are undocumented
- memory transfers dominate GPU runtime

Performance findings may be deferred only with benchmark evidence,
`optimization_notes`, and a user-visible caveat.

### Tests, Docs, Artifacts

Check that tests cover three-backend parity, precision, convergence, formula
compatibility, fallback visibility, and external baseline scaling. Docs must
explain model/objective, API, backend behavior, formula support, numerical
notes, and limitations. Benchmark results must be machine-readable and free of
credentials.

Validation evidence should state one tier: `local-minimal`, `local-full`, or
`remote-full`. `COMPLETE` requires `local-full` for active local gates. Remote
GPU, R, or large-scale gaps may justify `PARTIAL_REMOTE_PENDING`; missing local
required evidence does not.

## Severity

- `CRITICAL`: Incorrect results, wrong objective, missing required backend for a
  public method, silent fallback, security/credential leak, or release/publish
  risk.
- `HIGH`: Backend parity break, convergence failure, unsupported formula path
  without clear failure, missing inference for a public inferential estimator,
  serious performance regression, or missing tests for a changed public
  behavior.
- `MEDIUM`: Important maintainability, coverage, docs, artifact, or performance
  issue that does not invalidate core behavior, including missing impact
  classification when no blocking gate is hidden.
- `LOW`: Small cleanup, wording, style, or optional coverage improvement.

Use dimensions:

- `BUG`
- `BACKEND`
- `INFER`
- `PERF`
- `API`
- `FALLBACK`
- `FORMULA`
- `MATRIX`
- `TEST`
- `DOC`
- `ARTIFACT`
- `MAINT`
- `READ`

Use statuses:

- `open`
- `fixed`
- `deferred`
- `needs remote GPU`
- `needs precision fix`
- `needs optimization`
- `needs user approval`

## Auto-Fix Loop

In `auto-fix` mode:

1. Inspect changed and adjacent files.
2. Record findings with severity, dimension, file:line, and status.
3. Fix all CRITICAL and HIGH issues that can be fixed locally.
4. Fix MEDIUM issues that affect the requested feature or completion gates.
5. Run targeted tests, static checks, and benchmark checks when relevant.
6. Re-review the affected files.
7. Repeat until no unresolved CRITICAL/HIGH issues remain, or stop with the
   correct workflow status.

Do not auto-fix by changing statistical definitions merely to match an external
framework with a different objective normalization. Adjust comparison tests or
document the equivalent penalty mapping instead.

## Exit Rules

- `audit`: Return findings only. If none are found, say so and mention residual
  test or benchmark risk.
- `auto-fix COMPLETE`: No unresolved CRITICAL/HIGH issues remain and targeted
  checks pass.
- `auto-fix PARTIAL_REMOTE_PENDING`: Only remote GPU, R/external framework, or
  large-scale benchmark evidence is missing.
- `auto-fix BLOCKED_NEEDS_USER_APPROVAL`: Fix requires user approval for API
  break, backend deferral, performance caveat, remote credentials, commit,
  push, PR, merge, release, or package upload.
- `auto-fix FAILED`: A local blocking correctness, precision, convergence,
  backend, fallback, or review issue remains unresolved.

## Report Format

For each finding, use:

```text
[SEVERITY][DIMENSION][status] path:line - issue
Impact: ...
Fix: ...
Evidence: ...
```

For `auto-fix`, include:

- changed files
- fixed findings
- deferred findings and why
- tests or benchmarks run
- remaining hard exit status

Findings must lead audit-mode responses. Summaries are secondary.
