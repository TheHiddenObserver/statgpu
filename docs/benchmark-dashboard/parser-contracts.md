# Benchmark Frontend Parser Contracts

## Purpose

Benchmark source files use different shapes and naming conventions. A parser converts one source family into normalized schema-v1.1 runs and model metadata.

The current implementation uses a manifest-driven source registry and a parser function registry:

- Manifest: `dev/benchmarks/frontend_sources.json`
- Parser lookup: `dev/benchmarks/frontend_data/registry.py`
- Parser implementations: `dev/benchmarks/frontend_data/parsers/`

## Parser function interface

Every registered parser has the callable interface:

```python
def parse_source(filepath: Path, env_id: str) -> tuple[
    list[dict],  # normalized runs
    list[dict],  # model registry entries
    list[str],   # warnings/informational messages
]:
    ...
```

A parser must be deterministic for a fixed source file and environment.

### Return value: runs

Each run should provide all source-specific normalized fields that the parser can determine, including:

- Environment, categories, model, framework, and backend.
- Scale information.
- Loss, penalty, solver, variant, and implementation when applicable.
- Metric groups.
- Source parser provenance.
- Canonical `case_id` and `method_config_id` when the source contains multiple cases or method configurations.

The generator is responsible for injecting or finalizing:

- Canonical `source_id` from the manifest.
- `comparison_id` from the manifest.
- Source date override when declared by the manifest.
- Default `case_id` or `method_config_id` when omitted.
- Canonical `run_id` from `RunIdentity`.
- Resolved `reference_run_id` for computed speedups.

A parser must not rely on input traversal order to produce stable identities.

### Return value: model entries

A model entry has the form:

```python
{
    "model_id": "LassoCV",
    "primary_category_id": "linear_models",
    "category_ids": ["linear_models", "penalized_glm"],
    "supports_penalty": True,
    "supports_inference": False,
}
```

Model entries from multiple sources are merged centrally:

- `category_ids` are unioned and sorted.
- `supports_penalty` and `supports_inference` use logical OR.
- The central model registry controls the primary category for known models.

### Return value: warnings

Warnings are returned as human-readable strings. The generator maps them into structured parse-report issues:

- Messages containing “unavailable” or “not available” become `METHOD_UNAVAILABLE` with `info` severity.
- Other parser messages become `PARSE_WARNING` with `warning` severity.
- Exceptions become `PARSE_ERROR` with `error` severity.

In strict canonical mode, warning/error issue codes must be explicitly allowed for the source in the manifest. Informational `METHOD_UNAVAILABLE` entries can be permitted for benchmarks where an optional external package is absent.

## Manifest source contract

Each entry in `frontend_sources.json` contains:

```json
{
  "source_id": "stable-canonical-id",
  "comparison_id": "comparison-group-id",
  "path": "results/benchmark_frontend_sources/source.json",
  "original_path": "results/original/source.json",
  "sha256": "normalized-source-sha256",
  "parser": "registered_parser_name",
  "parser_version": "1.0",
  "env_id": "remote-p100",
  "required": true,
  "allowed_issue_codes": []
}
```

Optional fields include `source_date` when the canonical source does not contain an authoritative date.

### Required-source behavior

With `--strict-sources`:

- A missing required file fails generation.
- A required source without a manifest SHA256 fails generation.
- A SHA256 mismatch fails generation.
- An unapproved warning or error issue code fails generation.
- A parser exception fails generation.

## Source hashing

Canonical source SHA256 is computed after strict text normalization:

1. Input must be valid UTF-8.
2. UTF-8 BOM is rejected.
3. Bare carriage returns are rejected.
4. CRLF is normalized to LF.
5. SHA256 is computed over the normalized bytes.

This contract makes hashes stable across normal Windows/Unix line-ending differences while rejecting ambiguous text encodings.

## Registered parser implementations

The current parser registry contains eight implementations:

| Parser name | Source family |
|---|---|
| `penalized_glm_bench_perf` | Penalized GLM timing and accuracy |
| `glm_solver_benchmark` | GLM solver timing and reported speedup |
| `elasticnet_benchmark_full` | statgpu/glmnet ElasticNet sources |
| `coxph_efron_bench` | CoxPH Efron variants and backends |
| `comprehensive_validation` | Multi-family GLM validation |
| `coxph_package_comparison` | CoxPH cross-package comparison |
| `lassocv_combined` | Seed-aggregated LassoCV benchmark |
| `knockoff_benchmark` | Knockoff and feature-selection baselines |

Nine canonical source files are registered because the shared ElasticNet parser handles separate statgpu and glmnet sources.

## Framework and backend normalization

Parsers must emit a canonical framework identifier registered by the manifest.

For statgpu runs:

- `framework` is `statgpu`.
- `backend` is one of `numpy`, `cupy`, or `torch`.

For external packages:

- `framework` is the canonical external framework ID.
- `backend` is `null`.

Current external framework IDs include:

- `sklearn`
- `glmnet`
- `statsmodels`
- `lifelines`
- `scikit_survival`
- `knockpy`

A parser should normalize upstream aliases rather than passing display strings into canonical fields.

## Case and method identity

Use `case_id` for benchmark-design conditions that should define the comparison cell, for example:

- Data-generating parameters.
- Sample and feature dimensions beyond the scale key.
- Tie regime or benchmark variant.
- Noise, correlation, or signal configuration.

Use `method_config_id` for method-specific settings that distinguish comparable algorithm configurations, for example:

- Hyperparameter grid policy.
- Internal optimization configuration.
- Method-specific tuning options.

When an identity is derived from a mapping:

1. Select only the intended identity fields.
2. Serialize with sorted keys and compact separators.
3. Hash the canonical JSON.
4. Prefix the hash with a readable identity type, such as `case-`.

Replicate identifiers such as random seeds should not create separate canonical runs when the parser is intentionally aggregating them.

## Metric provenance

Every emitted metric group should contain:

- `quality`
- `source_file`

Use:

- `measured` for direct source observations.
- `reported` for source-provided aggregate/statistic values.
- `computed` for deterministic parser derivations.
- `partial` when only an incomplete comparison is available.

Do not label parser-computed values as measured.

## Timing contract

Timing is stored in milliseconds.

When aggregating repeated observations, include:

- `fit_time_ms`
- `std_ms`
- `sample_count`
- `std_ddof`
- `std_scope`

`std_scope` is:

- `raw_measurements` when dispersion is over repeated timings within a run.
- `replicates` when dispersion is over independently repeated benchmark runs or seeds.

Do not synthesize `std_ms: 0` merely because dispersion is absent from the source. Missing dispersion and observed zero dispersion have different meanings.

## Speedup contract

### Computed speedup

A parser may emit:

```json
{
  "value": 3.2,
  "reference_backend": "numpy",
  "reference_framework": "statgpu",
  "reported_semantics": "computed",
  "quality": "computed",
  "source_file": "..."
}
```

The generator resolves the unique compatible timing reference and injects `reference_run_id`. The reference must match the chart cell identity and requested framework/backend/implementation.

### Reported speedup

When the upstream runner supplies speedup directly, emit:

```json
{
  "value": 3.2,
  "reference_backend": "numpy",
  "reference_framework": "statgpu",
  "reported_semantics": "reported_by_runner",
  "quality": "reported",
  "source_file": "..."
}
```

Reported values are not required to have `reference_run_id` because the original runner may not expose the exact reference timing record.

## Numeric safety

Parsers must not emit:

- NaN.
- Positive or negative infinity.
- Negative timing.
- Negative standard deviation.
- A `min_ms` larger than `max_ms`.

When an upstream value is unavailable, omit the optional field instead of using a misleading numeric sentinel.

## Adding a parser or source

1. Add or update a parser under `frontend_data/parsers/`.
2. Export it from `frontend_data/parsers/__init__.py`.
3. Add the parser name to `PARSER_FUNCTIONS` in `registry.py`.
4. Add the canonical source file under `results/benchmark_frontend_sources/`.
5. Add a manifest entry with a normalized SHA256.
6. Add parser-focused tests and contract assertions.
7. Generate with `--strict-sources`.
8. Confirm schema, semantic, identity, and speedup validation.
9. Regenerate committed frontend data and deployed assets.

## Verification

```bash
pytest dev/tests/test_benchmark_frontend_data.py \
       dev/tests/test_frontend_contracts.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources
```
