# statgpu Benchmark Dashboard

The statgpu benchmark dashboard provides a common view of timing, speedup, numerical quality, convergence, prediction, validation, and feature-selection results produced by several benchmark families.

The dashboard is a presentation layer over a generated benchmark bundle. It does not parse raw benchmark files in the browser and does not recompute statistical metrics except for sorting, filtering, and chart layout.

## Current coverage

As of PR #78, the canonical manifest registers nine benchmark sources and eight parser implementations. The generated bundle contains 1,491 normalized runs across 13 models.

Covered benchmark families include:

- Penalized GLM performance.
- GLM solver comparison.
- ElasticNet cross-framework comparison.
- CoxPH Efron variants.
- Comprehensive GLM validation.
- CoxPH package comparison.
- LassoCV backend comparison.
- Knockoff and feature-selection baselines.

The source registry is `dev/benchmarks/frontend_sources.json`. Generated files are committed in both `frontend/public/data/` and `docs/assets/benchmarks/data/`.

## Reading the page

### Environment and category

The sidebar selects the benchmark environment and one or more model categories. The initial state is data-driven:

1. Prefer `remote-p100` when it exists and has runs.
2. Otherwise select the first environment that has runs.
3. Prefer `penalized_glm` when it is available in the selected environment.
4. Otherwise select the first available category.

This avoids loading a valid environment/category combination with no rows.

### Progressive filters

Filters are applied in the following order:

```text
Environment and category
  → Model
    → Variant
      → Penalty
        → Solver
          → Scale
            → Backend
              → External framework
```

Changing an upstream filter clears incompatible downstream selections. For example, changing the model clears variant, penalty, solver, and scale selections.

Filter behavior:

- **Model**: all models present in the selected environment and categories.
- **Variant**: shown only when the selected model has variants.
- **Penalty**: shown after a model is selected.
- **Solver**: shown after a penalty is selected.
- **Scale**: multi-select chips based on `scale_key`.
- **Backend**: applies only to statgpu runs; available values are NumPy, CuPy, and Torch.
- **External framework**: hidden by default and shown only when the current filter context contains applicable external runs.

External frameworks currently registered by the manifest are scikit-learn, glmnet, statsmodels, lifelines, scikit-survival, and knockpy. The browser only renders entries actually referenced by generated runs.

## Charts

### Timing chart

The timing chart displays `metrics.timing.fit_time_ms`. A bar represents a fully identified run series within a benchmark comparison and scale.

The grouping identity includes:

- Comparison.
- Environment.
- Model.
- Case and method configuration.
- Variant.
- Loss, penalty, and solver.
- Scale.

The series identity includes:

- Framework.
- Backend.
- Implementation.

This distinction prevents results from different CoxPH variants, solver configurations, implementations, or external packages from being merged into the same visual cell.

### Speedup chart

A speedup greater than one means the displayed run is faster than its reference. A value below one is a slowdown.

There are two speedup semantics:

#### Computed speedup

```text
speedup = reference fit time / current fit time
```

The generated record contains `reference_run_id`, `reference_framework`, and `reference_backend`. Semantic validation checks that:

- The reference run exists.
- Both timings are positive.
- Comparison and scale are compatible.
- The stored value agrees with the timing ratio within tolerance.

#### Reported speedup

Some benchmark runners directly emit a speedup value. These records use `reported_semantics: "reported_by_runner"`. They are labeled as reported values and should not be interpreted as a frontend recomputation.

Speedup labels include enough identity information to distinguish variant, penalty, solver, framework/backend, implementation, and scale when present.

## Overview table

The overview table provides run-level inspection and supports:

- Stable keyed-column sorting.
- Ascending/descending toggling.
- A default 200-row limit.
- “Show all” for the filtered result set.
- Framework-aware display for external rows whose backend is `null`.

Sort ties are resolved by `run_id`, so repeated renders are deterministic.

## Domain panels

Panels are rendered only when the filtered runs contain the corresponding metric group.

### Validation

Displays overall `pass`, `warn`, or `fail` status and individual validation checks. A check can include an operator, value, tolerance, and reference.

### Accuracy

Displays coefficient and standard-error differences, including absolute, relative, and seed-aggregated forms when available.

Typical fields include:

- Coefficient L2 difference.
- Maximum absolute coefficient difference.
- Relative coefficient L2 error.
- Maximum absolute standard-error difference.

### Prediction

Displays predictive metrics such as train/test MSE, noiseless test MSE, selected regularization level, and survival C-index.

### Convergence

Displays iteration summaries and convergence rates. Iteration summaries may include a mean and standard deviation over benchmark replicates.

### Selection

Displays feature-selection metrics such as precision, recall, false discovery proportion, F1, Jaccard similarity to truth, estimated FDR, target FDR, and selected-set size.

## Metric quality

Metric groups carry a quality label:

- `measured`: directly observed by the benchmark.
- `reported`: copied from an upstream aggregate or benchmark report.
- `computed`: derived deterministically by the parser from source values.
- `partial`: incomplete or only partially comparable information.

Quality describes provenance, not whether a method performed well.

## Generated metadata

The frontend loads three files:

### `benchmark_data.json`

Contains the schema version, environment/category/model/framework/comparison registries, and normalized runs.

### `parse_report.json`

Contains source counts, generated run count, and structured parse issues. The frontend accepts report version `2.0`.

### `source_inventory.json`

Contains catalog, eligibility, registration, availability, and parsed-source counts. The frontend accepts inventory version `1.0`.

All three files must have the same `generation_id`. A mismatch indicates that files from different generator executions were mixed.

## Reproducing the dashboard

From the repository root:

```bash
python -m pip install -U pytest jsonschema

python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

cd frontend
npm ci
npm run typecheck
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

Preview the production files:

```bash
cd docs/assets/benchmarks
python -m http.server 8000
```

## Adding a benchmark source

A maintainer should:

1. Add a canonical source JSON file under `results/benchmark_frontend_sources/`.
2. Add the source, SHA256, environment, comparison, parser, and allowed issue codes to `dev/benchmarks/frontend_sources.json`.
3. Implement or reuse a parser registered in `dev/benchmarks/frontend_data/registry.py`.
4. Return normalized runs and model metadata according to the parser contract.
5. Add parser and contract tests.
6. Regenerate all three data files and rebuild the deployed frontend assets.
7. Run Python validation, TypeScript checks, the production build, and Playwright tests.

See the following technical references:

- `docs/benchmark-dashboard/schema-v1.1.md`
- `docs/benchmark-dashboard/parser-contracts.md`
- `docs/benchmark-dashboard/aggregation-contract.md`
