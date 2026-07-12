# statgpu Benchmark Frontend

Interactive benchmark dashboard for statgpu, built with Vite, TypeScript, and ECharts. The dashboard consumes a generated, versioned benchmark bundle and is published from `docs/assets/benchmarks/`.

## Current coverage

The canonical bundle currently contains 14 registered sources, 1,623 normalized runs, and 36 models. All published dashboard categories now have data, including:

- penalized GLM, GLM, and linear models;
- robust and quantile regression;
- survival analysis;
- unsupervised learning;
- ordered models;
- nonparametric methods;
- panel models;
- covariance estimation;
- feature selection.

The linear-model category includes the June 2026 squared-error rows from `penalized_glm_perf_20260622.json` and `glm_solver_20260623.json`, in addition to the older ElasticNet and LassoCV comparisons.

## What the dashboard shows

- Environment and category navigation.
- Progressive filters for model, variant, penalty, solver, scale, backend, and external framework.
- Timing and speedup charts.
- A sortable and paginated overview table.
- Validation, accuracy, inference, prediction, convergence, and selection panels.
- Parse-report and source-inventory metadata.

Speedups have two distinct meanings:

- **Computed**: reference timing divided by current-run timing. The generated record contains `reference_run_id`.
- **Reported by runner**: copied from a benchmark runner that already computed the speedup. These rows are labeled separately and do not imply frontend recomputation.

## Requirements

- Node.js 20 or later.
- Python 3.9 or 3.11.
- Python test packages: `pytest` and `jsonschema`.

## Development

Run commands from the repository root unless a command explicitly changes directory.

```bash
cd frontend
npm ci
cd ..

python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

cd frontend
npm run dev
# Open http://localhost:5173
```

The generator automatically uses `dev/benchmarks/frontend_sources.json`. Required canonical sources are SHA256-verified; unapproved warnings or errors fail under `--strict-sources`.

## Validation and tests

```bash
python -m pip install -U pytest jsonschema
pytest \
  dev/tests/test_benchmark_frontend_data.py \
  dev/tests/test_frontend_contracts.py \
  dev/tests/test_frontend_domain_coverage.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources

cd frontend
npm run typecheck
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

The domain-coverage suite verifies that robust/quantile, unsupervised, ordered, nonparametric, panel, and covariance categories all produce runs; it also guards the June 2026 linear-model sources and quantile GPU inference rows.

## Production build and staleness

```bash
cd frontend
npm ci
npm run build

cd ../docs/assets/benchmarks
python -m http.server 8000
```

The Vite build writes to `docs/assets/benchmarks/`. CI regenerates the deterministic bundle, rebuilds the frontend, and fails if either `frontend/public/data/` or `docs/assets/benchmarks/` differs from the committed output.

## Project structure

```text
frontend/
├── public/data/                 # Generated benchmark bundle
├── e2e/
│   ├── dashboard.spec.ts
│   ├── domain-coverage.spec.ts
│   └── state.spec.ts
└── src/
    ├── main.ts                  # Application orchestration
    ├── schema.ts                # Schema v1.1.0 TypeScript types
    ├── data.ts                  # Loading and filtering
    ├── state.ts                 # Defaults and cascade resets
    ├── identity.ts              # Chart/group identities
    ├── charts/
    │   ├── TimingChart.ts
    │   └── SpeedupChart.ts
    ├── components/
    │   ├── Header.ts
    │   ├── Sidebar.ts
    │   ├── FilterBar.ts
    │   ├── OverviewTable.ts
    │   └── panels/
    │       ├── PanelTable.ts
    │       ├── ValidationPanel.ts
    │       ├── AccuracyPanel.ts
    │       ├── InferencePanel.ts
    │       ├── PredictionPanel.ts
    │       ├── ConvergencePanel.ts
    │       └── SelectionPanel.ts
    └── utils/
```

## Data flow

```text
results/benchmark_frontend_sources/*.json
  + dev/benchmarks/frontend_sources.json
        → dev/benchmarks/generate_benchmark_data.py
        → frontend/public/data/{benchmark_data,parse_report,source_inventory}.json
        → Vite build
        → docs/assets/benchmarks/
```

All three generated JSON files share one `generation_id`, computed from the complete bundle after removing the `generation_id` fields themselves.

## Documentation

- Dashboard guide: `docs/en/guides/statgpu_benchmark_dashboard.md`
- Schema v1.1: `docs/benchmark-dashboard/schema-v1.1.md`
- Parser contract: `docs/benchmark-dashboard/parser-contracts.md`
- Aggregation contract: `docs/benchmark-dashboard/aggregation-contract.md`
- Rollout record: `docs/benchmark-dashboard/rollout-plan.md`
