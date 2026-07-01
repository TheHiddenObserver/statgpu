---
name: benchmark
description: Create and run synchronized three-backend statgpu benchmarks with external precision checks and remote_config-based execution
---

# Benchmark Skill

Use this skill when benchmarking statgpu methods, validating GPU performance, or
comparing precision/convergence against external frameworks.

## Core Rules

- Benchmark all three backends: `numpy`, `cupy`, and `torch`.
- Prefer reusable scripts in `dev/benchmarks/`.
- Use `dev/tests/` only for pytest-style tests and assertions.
- Save machine-readable output to `results/*.json`.
- Use `dev/scripts/remote_config.py` for remote execution.
- Do not read credentials from memory, Markdown, or `.claude/settings.json`.
- Do not hardcode passwords, SSH ports, private remote paths, or host-specific
  secrets in benchmark scripts or results.
- Do not commit, push, publish, or upload packages unless explicitly requested.

## Required Output Schema

Each benchmark JSON should include these fields when applicable:

```json
{
  "method": "...",
  "backend_times": {
    "numpy": null,
    "cupy": null,
    "torch": null
  },
  "external_baseline": {
    "name": null,
    "time": null,
    "version": null
  },
  "precision_vs_external": {},
  "convergence_status": {},
  "backend_precision": {},
  "compatibility_matrix": {},
  "cv_matrix": {},
  "inference_matrix": {},
  "threshold_source": {},
  "objective_scaling": null,
  "penalty_scale_mapping": null,
  "cpu_vs_external": null,
  "gpu_vs_cpu": null,
  "crossover_n": null,
  "target_scale_source": null,
  "optimization_notes": [],
  "validation_tier": "local-minimal",
  "schema_status": "unchecked",
  "timing_scope": {},
  "reproducibility": {},
  "uncovered_reasons": []
}
```

Use `null` only when the value is not applicable or could not be collected.
Explain missing values in `uncovered_reasons`.

After writing JSON, validate required keys. Set `schema_status` to `ok` only
when all required keys are present. Missing keys must be added or explained in
`uncovered_reasons`; do not leave the schema unchecked.

## Timing Rules

GPU timings must synchronize before and after the measured region:

```python
def sync_backend(backend):
    if backend == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif backend == "torch":
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

Separate setup, data transfer, fitting, prediction, inference, and result
conversion where possible. Report `timing_scope` so GPU speedups are not hidden
or exaggerated by mixed scopes.

Use repeated runs, warmup, fixed random seeds, and dtype/device metadata. If a
method is stochastic, report seed and solver settings.

## Validation Tiers

- `local-minimal`: imports, targeted numpy checks, unavailable-backend behavior,
  active API/error checks, and no unsafe artifacts.
- `local-full`: `local-minimal` plus every locally available backend, active
  matrix checks, local external Python baselines, and JSON schema checks.
- `remote-full`: `local-full` plus remote GPU, R/external packages, or large
  benchmarks through `dev/scripts/remote_config.py`.

If a tier cannot run, record the exact command that should be run later and end
the parent workflow with `PARTIAL_REMOTE_PENDING` when local work is otherwise
complete.

## Baseline Hierarchy

Use the strongest available baseline and record it in `threshold_source` or
`external_baseline`:

1. Analytic closed form or derivative check.
2. Existing trusted statgpu implementation.
3. Python reference: sklearn, statsmodels, scipy, lifelines, patsy.
4. R reference package.
5. Numerical invariants: finite differences, KKT, monotonic objective,
   simulation coverage, backend parity.

If a stronger baseline is unavailable, record the missing package or command and
use the next tier.

## Precision And Convergence Evidence

Benchmarks must report numerical correctness, not only speed:

- objective/loss values
- coefficients or parameters
- predictions or risk scores
- gradients, Hessians, KKT, or monotonic objective checks where relevant
- standard errors, confidence intervals, p-values, or inference outputs where
  relevant
- `_inference_result`, summary fields, covariance type, and fallback status when
  inference is supported
- convergence status, tolerance, iteration count, and stop reason
- backend-to-backend differences

When statgpu and an external framework use different objective normalization,
do not force statgpu to match by changing its loss. Instead record the mapping,
for example:

```text
statgpu: n^{-1} sum_i loss_i + lambda * penalty
external: sum_i loss_i + lambda * penalty
equivalent comparison: lambda_external = n * lambda_statgpu
```

Precision failure triggers the parent workflow's precision/convergence gate
before any performance optimization.

## Cross-Axis Matrix Evidence

When a change touches losses, penalties, solvers, CV dispatch, or backend
kernels, benchmark or validation output should include a compact compatibility
matrix for the affected axes:

- loss x penalty x solver x backend status
- loss x penalty x CV x backend status when the combination is tunable
- coefficient/objective difference against a reference path
- selected alpha/lambda/C, fold score, and refit coefficient difference for CV
- inference field status and bse/p-value/CI differences when inference is
  supported
- convergence status and iteration count
- skipped combinations with explicit incompatibility or missing-backend reason
- timing for supported combinations when performance is relevant

This can be sampled for very large matrices, but every newly added or changed
axis member must appear in the matrix with representative partners.

## Inference Evidence

When inference is supported or expected, benchmark or validation output should
include:

- direct-estimator inference status
- CV final-refit inference status when CV supports `compute_inference=True`
- `_inference_result` result type and method
- `coef`, `bse`, `t` or `z`, `p`, CI, covariance type, and summary availability
- AIC/BIC/LLF/R-squared/F-test fields where applicable
- backend-to-backend differences for numpy/cupy/torch
- external baseline differences against statsmodels, R, lifelines, sklearn, or
  an analytic reference when available
- explicit estimation-only reason when inference is unsupported

Inference failures trigger the parent workflow's inference gate before
performance optimization.

## Formula Compatibility Evidence

For formula-facing methods, benchmark or validation scripts should record:

- formula string
- generated model matrix shape
- intercept handling
- categorical levels and reference level
- interaction/transform support
- missing-data behavior
- external R or formula-library comparison when available

Unsupported formula behavior belongs in `uncovered_reasons` with a clear failure
mode.

## Performance And Optimization Evidence

Report:

- `cpu_vs_external`: numpy/statgpu CPU time compared with the external baseline
- `gpu_vs_cpu`: CuPy/Torch time compared with numpy
- `crossover_n`: smallest problem size where GPU is faster, if found
- `target_scale_source`: existing benchmark, user request, or temporary scale
- memory transfer costs when measurable
- algorithmic complexity notes
- `optimization_notes`: profiling result, attempted optimizations, and residual
  bottlenecks

Trigger algorithm optimization when:

- CPU is materially slower than an established external framework
- GPU is slower than CPU at target scales
- GPU speedup depends on undocumented scale thresholds
- memory transfers dominate runtime
- asymptotic behavior is worse than the known/reference method

Target scale must be explicit. Prefer existing module benchmark scales. If none
exist, use a temporary small/medium/large scale ladder and record the rationale;
do not make speedup claims without target scale and timing scope.

Follow the workflow optimization budget: one profiling pass, up to two
optimization attempts, and one re-benchmark per attempt. If performance still
misses the target, report a caveat and let the parent workflow choose
`BLOCKED_NEEDS_USER_APPROVAL` or `PARTIAL_REMOTE_PENDING`.

## Script Placement

- Put reusable benchmarks in `dev/benchmarks/bench_<method>.py`.
- Put pytest checks in `dev/tests/` only when they should be run as tests.
- Put result JSON under `results/` or the repo's established results directory.
- Keep benchmark scripts parameterized by backend, dtype, seed, scale, and
  output path.

## Completion Report

Report:

- benchmark script path
- result JSON path
- command used
- validation tier
- schema status
- three-backend timings
- external baseline result
- precision and convergence result
- objective scaling and penalty mapping
- formula compatibility result when applicable
- optimization notes
- skipped tiers and exact follow-up commands
