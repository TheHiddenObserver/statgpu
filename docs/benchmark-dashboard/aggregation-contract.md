# Benchmark Frontend Aggregation Contract

## Scope

The benchmark frontend stores one normalized run per canonical method/case identity. Some source files already contain aggregate statistics; other sources contain seed- or replicate-level observations that a parser aggregates before producing the run.

The current implementation does **not** apply a generic cross-source metric pooling layer. Aggregation is parser-owned, while the generator centrally handles model-registry merging, identity validation, speedup-reference resolution, and bundle construction.

This document defines the semantics that parser-owned aggregation must preserve.

## Aggregation unit

Observations may be combined only when they share the same canonical run identity except for fields explicitly designated as replicate dimensions.

The canonical identity is:

```text
source_id
case_id
method_config_id
env_id
model_id
variant
implementation
loss
penalty
solver
framework
backend
scale_key
```

Typical replicate dimensions include:

- Random seed.
- Repeated timing trial.
- Independent benchmark repeat.

A field that changes the model, benchmark design, method configuration, framework, backend, implementation, or scale must not be averaged away.

## Replicate metadata

When a parser aggregates independent repetitions, include:

```json
{
  "replicate": {
    "n_runs": 10,
    "seed_count": 10,
    "n_failed": 0
  }
}
```

Semantics:

- `n_runs`: number of independent replicate records represented by the normalized run.
- `seed_count`: number of distinct seeds when seed is the replication mechanism.
- `n_failed`: number of attempted replicates that failed, when known.

`n_runs` is not the same as `timing.sample_count`. A benchmark replicate may itself contain multiple raw timing measurements.

## Timing aggregation

### Mean

For replicate-level timing values `t_1, ..., t_n`, parsers may emit the arithmetic mean:

```text
fit_time_ms = (1/n) Σ t_i
```

### Standard deviation

The parser must declare the convention:

- `std_ddof: 0` for population standard deviation.
- `std_ddof: 1` for sample standard deviation.

The parser must also declare the scope:

- `std_scope: "raw_measurements"`
- `std_scope: "replicates"`

And the number of observations represented by the dispersion:

- `sample_count`

Example for ten seed-level timings aggregated with population standard deviation:

```json
{
  "fit_time_ms": 12.4,
  "std_ms": 0.8,
  "sample_count": 10,
  "std_ddof": 0,
  "std_scope": "replicates",
  "quality": "measured",
  "source_file": "..."
}
```

### Missing dispersion

If the source does not provide replicate-level values or a documented standard deviation, omit `std_ms`. Do not replace unknown dispersion with zero.

### Min and max

`min_ms` and `max_ms` may be emitted only when they refer to the same observation scope as the timing summary. The invariant `min_ms <= max_ms` is enforced.

## Scalar metric aggregation

For seed- or replicate-level scalar metrics, the default implemented pattern is:

- Arithmetic mean for the central value.
- Population standard deviation when multiple replicate values exist.
- Zero standard deviation may be emitted only when there is exactly one observed value and the parser deliberately represents the dispersion of that one-value population.

Examples include:

- Iteration count.
- Selected alpha.
- Train/test MSE.
- Coefficient relative error.
- Precision, recall, F1, and Jaccard similarity.

A parser should omit a metric when no observations are available. It should not insert zero as a missing-value placeholder.

## Quality and provenance during aggregation

Quality labels must reflect the origin of the aggregate:

- `measured`: aggregate of directly observed benchmark measurements.
- `reported`: upstream file already reports the aggregate and raw observations are not recomputed.
- `computed`: parser derives the metric from source fields, including deterministic formulas or support-set comparisons.
- `partial`: source information is incomplete or not fully comparable.

Examples:

- Mean timing over measured seed runs: `measured`.
- Iteration mean copied from an upstream summary: `reported`.
- F1 computed from source precision/recall or truth sets: `computed`.

## Selection metrics

When selection metrics are computed per replicate, aggregate each metric over the same eligible replicate set whenever possible.

Supported metrics include:

- Precision.
- Recall.
- False discovery proportion.
- F1.
- Jaccard similarity to the true support.
- Estimated FDR.
- Number selected.

`target_fdr` is normally a method/configuration parameter rather than a replicate average. It should be emitted as an identical value only when all aggregated replicates share the same target.

Do not average runs with different target FDR values into one canonical run unless target FDR is explicitly part of `case_id`, `method_config_id`, or `parameters` and the identities remain distinct.

## Validation status aggregation

Validation statuses have the severity order:

```text
pass < warn < fail
```

When an upstream source contains multiple validation checks, the overall status should be the strictest status represented by the checks unless the source defines an authoritative overall status.

Individual checks should be preserved rather than reduced to only the overall status. A check can include:

- Metric name.
- Operator.
- Observed value.
- Tolerance.
- Reference.
- Status.

## Convergence aggregation

For replicate-level convergence information:

- `n_iter_mean`: arithmetic mean of available iteration counts.
- `n_iter_std`: declared replicate dispersion.
- `converged_rate`: converged replicate count divided by eligible replicate count.

Failed or missing replicates must not silently enter the denominator unless the source contract explicitly treats them as non-converged. Record `replicate.n_failed` when this distinction is available.

## Prediction and accuracy aggregation

Prediction and accuracy metrics may have paired standard-deviation fields. Examples:

- `test_mse` and `test_mse_std`.
- `c_index` and `c_index_std`.
- `coef_l2_rel_error` and `coef_l2_rel_error_std`.

The central value and standard deviation must be computed from the same replicate subset. If only some replicates provide a metric, the parser should either:

1. Aggregate the available subset and make the reduced count recoverable through source-specific parameters or documentation; or
2. Omit the aggregate and emit a warning when partial aggregation would be misleading.

## Speedup aggregation

Speedup is not pooled across unrelated references.

### Computed speedup

Computed speedup is tied to one resolved reference run:

```text
speedup = reference fit_time_ms / current fit_time_ms
```

The current and reference runs must agree on comparison identity, environment, model/case/method configuration, variant, loss, penalty, solver, and scale. Framework/backend/implementation identify the compared series.

The generator resolves `reference_run_id` only when the compatible reference is unique.

### Reported speedup

A runner-reported speedup is preserved with `reported_semantics: "reported_by_runner"`. It must not be pooled with computed speedups or reinterpreted as a ratio of frontend-visible timing rows.

## Model metadata merge

Model registry entries are the only centrally merged metadata objects.

Rules:

- Union and sort `category_ids`.
- Logical OR for `supports_penalty`.
- Logical OR for `supports_inference`.
- Use the central primary-category registry for known models.

This merge is order-independent.

## Cross-source behavior

Runs from different source IDs are not automatically pooled. The shared `comparison_id` permits chart-level comparison, not statistical aggregation.

This distinction is intentional:

- `source_id` remains part of canonical run identity.
- `comparison_id` is part of chart grouping identity.
- Separate sources can appear in the same comparison without losing provenance.

Any future cross-source pooling must define:

- Eligibility and compatibility conditions.
- Weighting rule.
- Variance rule.
- Provenance merge rule.
- Failure/missingness handling.
- A new identity or source representation for the pooled result.

It must not be added implicitly inside a chart.

## Determinism

Aggregation must be deterministic:

- Sort keys before hashing identity mappings.
- Do not depend on dictionary or file iteration order.
- Use stable numeric formulas.
- Round only at the final emitted field, not between aggregation steps.
- Never emit NaN or infinity.

## Current implemented example: LassoCV

The LassoCV parser aggregates seed-level runs as follows:

- Mean timing and population timing standard deviation.
- Mean and population standard deviation for iterations and alpha.
- Mean and population standard deviation for prediction and coefficient-error metrics.
- Mean and optional population standard deviation for support metrics.
- `replicate.n_runs` and `replicate.seed_count` equal the number of seed runs.
- Timing uses `std_scope: "replicates"` and `std_ddof: 0`.

This example is an implementation of the contract, not a requirement that all source families use population standard deviation. Other parsers may use a different documented convention when dictated by source semantics.

## Tests

Aggregation tests should verify:

- Correct grouping boundaries.
- Correct mean and dispersion convention.
- Missing-field behavior.
- Stable case/method identities.
- Replicate counts.
- Provenance labels.
- No cross-source pooling.
- Deterministic output under reordered inputs.

Run:

```bash
pytest dev/tests/test_benchmark_frontend_data.py \
       dev/tests/test_frontend_contracts.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources
```
