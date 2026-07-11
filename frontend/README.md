# statgpu Benchmark Frontend

Interactive benchmark dashboard for statgpu, built with Vite, TypeScript, and ECharts. The dashboard consumes a generated, versioned benchmark bundle and is published from `docs/assets/benchmarks/`.

## What the dashboard shows

The current frontend provides:

- Environment and category navigation.
- Progressive filters for model, variant, penalty, solver, scale, backend, and external framework.
- Timing and speedup charts.
- A sortable and paginated overview table.
- Validation, accuracy, prediction, convergence, and selection panels.
- Parse-report and source-inventory metadata.

Speedups have two distinct meanings:

- **Computed**: reference timing divided by the current run timing. The generated record contains `reference_run_id`.
- **Reported by runner**: copied from a benchmark runner that already computed the speedup. These rows are labeled separately and do not imply a frontend recomputation.

## Requirements

- Node.js 20 or later.
- Python 3.9 or 3.11.
- Python packages used by CI: `pytest` and `jsonschema`.

## Development

Run commands from the repository root unless a command explicitly changes directory.

```bash
# Install frontend dependencies
cd frontend
npm ci
cd ..

# Generate the canonical three-file data bundle
python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

# Start the development server
cd frontend
npm run dev
# Open http://localhost:5173
```

The generator automatically uses `dev/benchmarks/frontend_sources.json` when the manifest is present. In canonical mode, required sources are SHA256-verified and unapproved warnings or errors fail under `--strict-sources`.

## Validation and tests

```bash
# From the repository root: parser, schema, inventory, and contract tests
python -m pip install -U pytest jsonschema
pytest dev/tests/test_benchmark_frontend_data.py \
       dev/tests/test_frontend_contracts.py -v

# Validate generated output without writing files
python dev/benchmarks/generate_benchmark_data.py --check --strict-sources

# From frontend/: source, E2E, and schema-contract type checks
cd frontend
npm run typecheck

# Production build
npm run build

# Install the Playwright browser once, then run E2E tests
npx playwright install --with-deps chromium
npm run test:e2e
```

`npm run typecheck` covers the application, Playwright tests, and the JSON-Schema/TypeScript parity fixture.

## Production build and local preview

```bash
cd frontend
npm ci
npm run build

cd ../docs/assets/benchmarks
python -m http.server 8000
# Open http://localhost:8000
```

The Vite build writes directly to `docs/assets/benchmarks/`. Both generated data under `frontend/public/data/` and the deployed assets under `docs/assets/benchmarks/` are committed.

## Generated-asset staleness check

CI regenerates the deterministic bundle, rebuilds the frontend, and fails when either location has a diff:

```bash
python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

cd frontend
npm ci
npm run build
cd ..

git status --porcelain -- frontend/public/data docs/assets/benchmarks
```

A non-empty status means the committed generated assets are stale and must be refreshed in the same change.

## Project structure

```text
frontend/
├── index.html
├── package.json
├── public/
│   └── data/
│       ├── benchmark_data.json
│       ├── parse_report.json
│       └── source_inventory.json
├── e2e/
│   └── state.spec.ts
└── src/
    ├── main.ts                 # Application orchestration and render loop
    ├── schema.ts               # TypeScript representation of schema v1.1.0
    ├── data.ts                 # Data loading and run filtering
    ├── state.ts                # UI state, defaults, and cascade resets
    ├── identity.ts             # Frontend chart/group identity helpers
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
    │       ├── PredictionPanel.ts
    │       ├── ConvergencePanel.ts
    │       └── SelectionPanel.ts
    └── utils/
```

## Data flow

```text
results/benchmark_frontend_sources/*.json
  + dev/benchmarks/frontend_sources.json
        │
        ▼
dev/benchmarks/generate_benchmark_data.py
        │
        ├── benchmark_data.json
        ├── parse_report.json
        └── source_inventory.json
        │
        ▼
frontend/public/data/
        │
        ▼
Vite build
        │
        ▼
docs/assets/benchmarks/
```

All three generated JSON files share one `generation_id`, computed from the complete bundle after removing the `generation_id` fields themselves. This prevents metadata from different generator runs from being silently mixed.

## Documentation

- User and maintainer guide: `docs/en/guides/statgpu_benchmark_dashboard.md`
- Schema v1.1 contract: `docs/benchmark-dashboard/schema-v1.1.md`
- Parser contract: `docs/benchmark-dashboard/parser-contracts.md`
- Aggregation contract: `docs/benchmark-dashboard/aggregation-contract.md`
- Rollout record: `docs/benchmark-dashboard/rollout-plan.md`
