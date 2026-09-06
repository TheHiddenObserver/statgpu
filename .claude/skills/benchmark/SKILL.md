---
name: benchmark
description: Design, run, or review statgpu correctness/performance benchmarks with synchronized GPU timing, exact-source provenance, external or analytic precision checks, and conditional evidence for backend/CV/inference/formula behavior. Use when benchmarking, validating GPU performance, or comparing numerical precision/convergence against a reference.
when_to_use: Trigger for benchmark, performance, scaling, crossover, physical GPU, or numerical comparison work. Do not trigger merely because a code change has a GPU implementation.
argument-hint: "[method-or-benchmark-scope]"
---

# statgpu Benchmark

Benchmark `$ARGUMENTS` only after defining what is being measured and what correctness evidence makes the timing meaningful.

Read [schema.md](schema.md) when writing a machine-readable result, comparing more than one backend/environment, or publishing a performance/evidence claim.

## Core rules

- Benchmark only the backends and axes relevant to the question; when the task is a cross-backend statgpu performance claim, normally include NumPy, CuPy, and Torch or explicitly record why one is unavailable/not applicable.
- Reuse parameterized scripts in `dev/benchmarks/`.
- Keep pytest assertions in `dev/tests/`.
- Never report speed without numerical correctness/convergence evidence appropriate to the method.
- Do not read credentials from tracked Markdown, memory, or `.claude/settings.json`; remote execution uses `dev/scripts/remote_config.py` or maintained untracked/environment configuration.
- Do not commit/push/publish/upload artifacts unless the active task authorizes repository mutation/publication.

## Timing

GPU timing must synchronize around the measured region:

```python
def sync_backend(name):
    if name == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif name == "torch":
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

Separate where relevant:

- setup/data generation;
- host-to-device transfer;
- fitting/optimization;
- prediction/scoring;
- inference;
- device-to-host/report conversion.

Record the timing scope rather than hiding transfer/setup differences inside one number.

Use warmup, repeated runs, fixed seeds, and explicit dtype/device settings. Report the statistic used for timing (for example median of N post-warmup runs).

## Correctness before performance

Use the strongest available baseline:

1. analytic closed form or derivative/identity check;
2. trusted existing statgpu implementation;
3. Python reference (sklearn, statsmodels, scipy, lifelines, patsy, etc.);
4. authoritative R reference;
5. numerical invariants (finite differences, KKT, monotonic objective, coverage, backend parity).

Check the relevant outputs: objective/loss, coefficients/parameters, predictions/risk scores, gradients/Hessians/KKT, selected CV parameter, inference fields, and convergence status.

When objective normalization differs across libraries, map the public regularization scale instead of modifying statgpu to force a same-number comparison.

A precision/convergence failure blocks a speedup claim.

## Provenance

Every result used as evidence should identify at least:

- exact git commit and clean/dirty working-tree state;
- benchmark script and arguments;
- Python/statgpu/NumPy/CuPy/Torch versions as applicable;
- CUDA/runtime/driver and CPU/GPU identity when available;
- dtype, data shape, seed/data identity;
- timing scope, warmup, repeats, transfer policy;
- reference implementation/version and objective-scale mapping;
- validation tier and uncovered reason(s).

Do not aggregate speedups from materially different hardware/software environments into one homogeneous metric. Keep environment groups separate.

## Conditional evidence

Only emit evidence sections for active axes:

- `backend`: backend parity/device ownership;
- `cv`: selected grid/folds/scores/refit and selected parameter;
- `inference`: coef/BSE/t-or-z/p/CI/covariance/summary and direct versus final-CV-refit behavior;
- `formula`: formula/model-matrix/intercept/categorical/missing-data behavior;
- `performance`: timings, crossover, profiling, transfer cost.

Do not fill a result with meaningless `null` sections solely to satisfy a monolithic schema.

## Optimization budget

When optimization is part of the task:

1. establish the correct baseline and target scale;
2. profile once;
3. make up to two focused algorithm/kernel attempts before reassessing;
4. re-benchmark after each attempt;
5. retain correctness evidence after optimization.

If GPU is slower at the requested scale, report that result. Do not move the goalposts to a larger shape solely to manufacture a speedup.

## Result location

Prefer `results/*.json` (or the established module results directory) for machine-readable evidence and keep reusable benchmark code in `dev/benchmarks/`.

Machine-readable results should follow the common envelope and conditional sections in [schema.md](schema.md).

## Completion report

Report:

- script/result paths and exact command;
- exact source/environment;
- timing scope and run statistic;
- correctness/precision/convergence result;
- backend timings relevant to the claim;
- external/analytic reference and scale mapping;
- active conditional evidence (CV/inference/formula) when applicable;
- target scale/crossover or slower regime;
- skipped environment/tier and exact reason;
- optimization notes if performance work was requested.
