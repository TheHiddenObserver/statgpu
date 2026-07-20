# Benchmark Frontend Schema v1.1

## Status

Schema version `1.1.0` is the current benchmark dashboard contract.

Sources of truth:

- JSON Schema: `dev/benchmarks/benchmark_frontend_schema.json`
- TypeScript representation: `frontend/src/schema.ts`
- Runtime consumer check: `frontend/src/data.ts`
- Python structural and semantic validation: `dev/benchmarks/frontend_data/cli.py`

The JSON Schema pins `schema_version` to the constant `1.1.0`. The frontend rejects any other version rather than attempting a best-effort interpretation.

## Bundle

One generator execution produces three files:

| File | Version field | Purpose |
|---|---|---|
| `benchmark_data.json` | `schema_version: "1.1.0"` | Registries and normalized benchmark runs |
| `parse_report.json` | `report_version: "2.0"` | Parse coverage and structured issues |
| `source_inventory.json` | `inventory_version: "1.0"` | Catalog and registered-source coverage |

All three files contain the same 64-character lowercase SHA256 `generation_id`.

The ID is computed over the complete logical three-file bundle after removing the `generation_id` fields. Therefore, changing any generated run, registry entry, issue, or inventory count changes the shared bundle ID.

## Top-level benchmark data

`benchmark_data.json` contains:

```text
schema_version
generated
meta
environments
categories
models
frameworks
comparisons
runs
```

### `meta`

- `generator`: repository path of the generator.
- `git_sha`: source revision or `deterministic` in deterministic mode.
- `generation_id`: bundle hash.

### Registries

The top-level arrays provide referential registries used by runs:

- `environments[].env_id`
- `categories[].category_id`
- `models[].model_id`
- `frameworks[].framework_id`
- `comparisons[].comparison_id`

Every run reference must resolve to the corresponding registry.

## Run contract

A normalized run requires:

- `run_id`
- `env_id`
- `category_ids`
- `model_id`
- `comparison_id`
- `case_id`
- `method_config_id`
- `framework`
- `backend`
- `scale`
- `source`
- `metrics`

Optional identity or presentation fields include:

- `benchmark_session_id`
- `loss`
- `penalty`
- `solver`
- `solver_display`
- `solver_kind`
- `variant`
- `implementation`
- `parameters`
- `replicate`
- `quality`

### Backend policy

Each framework declares one backend policy:

- `required`: every run must provide a backend.
- `forbidden`: every run must have `backend: null`.
- `optional`: either form is allowed.

The current statgpu framework uses `required`. External packages use `forbidden` and are identified by framework rather than a statgpu backend.

Valid statgpu backends are:

- `numpy`
- `cupy`
- `torch`

## Canonical identity

The canonical `RunIdentity` is the ordered tuple:

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

The generator serializes this tuple as compact canonical JSON and sets:

```text
run_id = first 16 hexadecimal characters of SHA256(identity_json)
```

Semantic validation recomputes the hash and rejects duplicate identities or mismatched run IDs.

### Chart identity

Chart grouping intentionally excludes `source_id`, but includes comparison and method identity:

```text
comparison_id
env_id
model_id
case_id
method_config_id
variant
loss
penalty
solver
scale_key
```

Series identity is separate:

```text
framework
backend
implementation
```

This permits equivalent methods from different sources to be compared while preventing distinct variants or implementations from collapsing into one bar.

## Scale

Each run has:

- `scale_key`: canonical stable key, commonly `n{n_samples}_p{n_features}`.
- `n_samples`: nonnegative integer.
- `n_features`: nonnegative integer.
- `label`: human-readable form such as `100K×50`.

Filtering and chart identity use `scale_key`, not the label.

## Source provenance

Each run contains a source object with:

- `source_id`: canonical manifest identifier.
- `file`: source filename visible to the parser.
- `original_path`: original benchmark path when applicable.
- `sha256`: normalized UTF-8 source hash in canonical mode.
- `date`: benchmark date.
- `parser`: parser provenance name.
- `parser_version`: parser contract version.

Canonical mode rejects transitional source IDs.

## Metric groups

A run may contain any subset of the following groups.

### Timing

Required when present:

- `fit_time_ms`
- `quality`
- `source_file`

Optional dispersion fields:

- `std_ms`
- `min_ms`
- `max_ms`
- `sample_count`
- `std_ddof`
- `std_scope`

When a standard deviation is emitted for canonical aggregated data, the record should identify the sample count, degrees-of-freedom convention, and scope.

### Speedup

Required when present:

- `value`
- `reference_backend`
- `reference_framework`
- `reported_semantics`
- `quality`
- `source_file`

`reported_semantics` is either:

- `computed`
- `reported_by_runner`

Computed speedups require a valid `reference_run_id` and must agree with the timing ratio within semantic-validation tolerance.

### Accuracy

Supported fields include coefficient differences, relative errors, and standard-error differences, with optional standard deviations and reference labels.

### Inference

Supported fields include standard error, Wald statistic, p-value, and an optional Boolean validity indicator.

### Convergence

Supported fields include mean iterations, iteration standard deviation, and converged rate.

### Selection

Supported fields include precision, recall, FDP, F1, Jaccard similarity, estimated FDR, target FDR, and selected-set size, each with optional dispersion fields where applicable.

### Prediction

Supported fields include train/test MSE, noiseless test MSE, C-index, and selected alpha summaries.

### Validation

A validation group contains an overall status and optional checks. Status values are `pass`, `warn`, or `fail`. A check may include metric, operator, value, tolerance, and reference.

## Metric quality

Metric provenance uses one of:

- `measured`
- `reported`
- `computed`
- `partial`

This value describes how the number was obtained. It is not a performance grade.

## Validation layers

The generator applies three validation layers before writing files:

1. **Basic output validation**
   - Required fields.
   - Unique run IDs.
   - Backend consistency.
   - Nonnegative timing and speedup values.
   - No NaN or infinity.

2. **JSON Schema validation**
   - Draft 2020-12 schema.
   - Type, enum, required-field, and additional-property constraints.

3. **Semantic validation**
   - Registry referential integrity.
   - Framework backend policy.
   - Model/category consistency.
   - Canonical identity hashing and uniqueness.
   - Speedup reference existence and compatibility.
   - Computed speedup equality.
   - Canonical source/case identifiers.

Invalid output is never written.

## Versioning policy

A schema change requires a version change when an existing conforming consumer could no longer interpret the bundle safely.

Examples requiring a new version:

- Removing or renaming a required field.
- Changing identity semantics.
- Changing the meaning of a metric field.
- Changing enum semantics.

Backward-compatible optional fields may remain within v1.1 only when:

- Existing fields retain their meaning.
- The JSON Schema and TypeScript types are updated together.
- Contract tests and generated fixtures are updated.
- The frontend remains safe when the optional field is absent.

## Verification

```bash
pytest dev/tests/test_benchmark_frontend_data.py \
       dev/tests/test_frontend_contracts.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources

cd frontend
npm run typecheck
```
