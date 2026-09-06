# Benchmark Result Schema

Use a compact common envelope plus conditional sections. Omit inactive sections instead of filling them with `null` placeholders.

## Common envelope

Recommended JSON shape:

```json
{
  "schema_version": "statgpu-benchmark-v2",
  "benchmark": {
    "name": "...",
    "method": "...",
    "script": "dev/benchmarks/...py",
    "command": "..."
  },
  "source": {
    "git_commit": "...",
    "working_tree": "clean",
    "statgpu_version": "..."
  },
  "environment": {
    "environment_id": "...",
    "python": "...",
    "numpy": "...",
    "cupy": null,
    "torch": null,
    "cuda_runtime": null,
    "driver": null,
    "cpu": "...",
    "gpu": null,
    "os": "..."
  },
  "problem": {
    "shape": {},
    "dtype": "float64",
    "seed": 0,
    "data_id": "...",
    "parameters": {}
  },
  "timing": {
    "scope": "fit-only",
    "warmup": 2,
    "repeats": 5,
    "statistic": "median",
    "transfer_policy": "excluded",
    "backend_seconds": {}
  },
  "correctness": {
    "reference": {},
    "objective_scaling": null,
    "penalty_scale_mapping": null,
    "metrics": {},
    "convergence": {},
    "status": "pass"
  },
  "validation": {
    "tier": "local-full",
    "status": "complete",
    "uncovered_reasons": []
  }
}
```

Use `null` inside the common envelope only when the field is genuinely part of the recorded environment but unavailable/not applicable. Do not create inactive top-level sections merely to populate nulls.

## Source contract

`source` should be strong enough to answer "what exact code produced this number?" Record:

- exact commit SHA;
- `clean`/`dirty` tree state;
- statgpu version;
- optional validator commit/version when a separate acceptance runner determines pass/fail.

If the tree is dirty and the result is used beyond local exploration, record a diff identifier or do not present it as canonical evidence.

## Environment grouping

`environment_id` identifies measurements that are safe to compare/aggregate. Include or derive it from materially relevant fields such as:

- CPU model;
- GPU model;
- CUDA/runtime/driver;
- Python;
- NumPy/CuPy/Torch versions;
- OS/container image when relevant.

Do not average or combine speedups across different `environment_id` values as if they came from one machine/software stack. Present cross-environment results as separate groups.

## Problem identity

Record enough information to reproduce the workload:

- n/p/targets/classes/folds/etc.;
- dtype;
- random seed;
- distribution/generator parameters;
- data hash or stable generated-data identifier when practical;
- estimator/solver/tolerance/regularization parameters.

For stochastic algorithms record all seeds that affect data, initialization, CV, bootstrap, or randomized solvers.

## Timing contract

`timing.scope` must name what is measured, for example:

- `fit-only`;
- `predict-only`;
- `inference-only`;
- `fit+inference`;
- `end-to-end-with-transfer`.

If both fit-only and end-to-end matter, store separate timing records rather than one ambiguous number.

GPU regions must be synchronized before and after timing. Record warmup/repeats and whether the reported value is mean/median/min/etc.

## Correctness contract

A performance result is valid only with an appropriate correctness comparison. `correctness.reference` should record:

```json
{
  "name": "sklearn.linear_model.Lasso",
  "version": "...",
  "settings": {},
  "reference_type": "external-python"
}
```

Possible `reference_type` values include:

- `analytic`;
- `trusted-statgpu`;
- `external-python`;
- `external-r`;
- `numerical-invariant`.

Record objective scaling and regularization mapping whenever same-number hyperparameters do not represent the same objective across implementations.

`correctness.metrics` may contain only relevant quantities, for example:

```json
{
  "coef_max_abs_diff": 2.1e-8,
  "prediction_max_abs_diff": 6.0e-9,
  "objective_abs_diff": 1.0e-10
}
```

Do not invent a universal tolerance. Record the threshold source in the benchmark metadata or result when the acceptance threshold is non-obvious.

## Backend section (conditional)

Use when the benchmark makes a backend parity/device claim:

```json
"backend": {
  "numpy": {"status": "pass", "device": "cpu"},
  "cupy": {"status": "pass", "device": "cuda:0"},
  "torch": {"status": "skipped", "reason": "not installed"},
  "pairwise_metrics": {}
}
```

Include explicit skipped/unavailable reasons.

## CV section (conditional)

Use when CV/tuning is active:

```json
"cv": {
  "folds": 5,
  "grid": {},
  "selected": {},
  "scores": {},
  "refit": {
    "solver": "...",
    "device": "...",
    "coef_diff_vs_direct_reference": null
  },
  "status": "pass"
}
```

Record CV-only solver/method controls separately from final-refit controls when the API distinguishes those stages.

## Inference section (conditional)

Use when inference is active:

```json
"inference": {
  "method": "...",
  "cov_type": "...",
  "direct": {},
  "cv_final_refit": {},
  "reference_metrics": {},
  "backend_provenance": {},
  "status": "pass"
}
```

Relevant metrics may include coefficient, BSE, t/z, p-value, CI, covariance, AIC/BIC/LLF, summary availability, and simultaneous-interval properties.

If inference is intentionally unsupported, record that in capability/validation evidence only when the benchmark is actually evaluating that boundary; do not add an empty inference section to unrelated performance results.

## Formula section (conditional)

Use for formula-facing validation:

```json
"formula": {
  "formula": "y ~ x1 + C(group)",
  "matrix_shape": [100, 4],
  "intercept": true,
  "feature_names": [],
  "missing_rows": [],
  "reference": "patsy",
  "status": "pass"
}
```

## Performance section (conditional)

Use when the result supports a performance claim:

```json
"performance": {
  "cpu_vs_external": {},
  "gpu_vs_cpu": {},
  "crossover": {},
  "memory_or_transfer": {},
  "profiling": {},
  "optimization_notes": []
}
```

A speedup must identify numerator/denominator timing scopes and environment group. `crossover` should record measured workload points; do not interpolate a precise threshold from sparse data without saying it is an estimate.

## Validation status

Suggested values:

- `local-minimal`;
- `local-full`;
- `remote-full`.

`validation.uncovered_reasons` lists only evidence that matters to the active claim. Do not call a benchmark incomplete because an unrelated section was not run.

## Canonical evidence rules

A result used for release/public documentation should be tied to:

- exact numerical source;
- exact validator/acceptance contract if separate;
- environment identity;
- reproducible command;
- machine-readable raw result.

If validator acceptance logic changes after evidence was collected, the older artifact remains historical evidence for its recorded source but does not automatically prove the newer acceptance contract.
