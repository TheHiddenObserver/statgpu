# statgpu Benchmark Frontend

Interactive benchmark dashboard for statgpu, built with Vite, TypeScript, and ECharts. The dashboard consumes a generated, versioned benchmark bundle and is published from `docs/assets/benchmarks/`.

## Current coverage

The canonical dashboard is restricted to benchmark sources dated **2026-06-01 or later**. The manifest currently registers **eight sources**, producing **1,625 normalized runs across 36 models**:

- `p2_benchmark_20260617.json`;
- `penalized_glm_perf_20260622.json`;
- `coxph_efron_20260622.json`;
- `glm_solver_20260623.json`;
- `loss_functions_20260623.json`;
- `new_modules_full_20260624.json`;
- `unsupervised_20260627.json`;
- `ordered_inference_pr74.json`.

Covered categories include penalized GLM and GLM, recent linear models, robust and quantile regression, survival analysis, unsupervised learning, ordered models, nonparametric methods, panel models, covariance estimation, and ANOVA.

ANOVA coverage includes one-way ANOVA, two-way ANOVA, Welch ANOVA, Tukey HSD, and Bonferroni correction on NumPy, CuPy, and Torch. One-way ANOVA also includes aligned SciPy timing and F-statistic validation rows.

The linear-model category uses the June 2026 squared-error rows from `penalized_glm_perf_20260622.json` and `glm_solver_20260623.json`. April 2026 ElasticNet, LassoCV, comprehensive-validation, Cox package-comparison, and knockoff results are intentionally not registered.

Current June-or-later sources provide external comparisons through scikit-learn, SciPy, statsmodels, linearmodels, and pyGAM. The feature-selection category remains part of Schema v1.1, but it is intentionally empty until a June 2026-or-later benchmark is available.

## What the dashboard shows

- Environment and category navigation.
- Progressive filters for model, variant, penalty, solver, scale, backend, and external framework.
- Timing and speedup charts.
- A sortable and paginated overview table.
- Validation, accuracy, inference, prediction, convergence, and selection panels.
- Parse-report and source-inventory metadata.

Speedups have two distinct meanings:

- **Computed**: reference timing divided by current-run timing. The generated record contains `reference_run_id`.
- **Reported by runner**: copied from a benchmark runner that already computed the speedup. These rows carry an `Ⓡ` marker and do not imply frontend recomputation.

The speedup chart uses a thin solid gray 1× parity line without an overlapping `1.0x` label. Runner-reported bars use a subtle border instead of a patterned fill. The summary card displays the computed and reported maxima separately when both are present, because they may use different references.

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

The generator automatically uses `dev/benchmarks/frontend_sources.json`. Required canonical sources are SHA256-verified. Manifest loading rejects a missing `source_date` or any source earlier than the configured `minimum_source_date`, currently `2026-06-01`. Unapproved warnings or errors fail under `--strict-sources`.

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

The domain-coverage suite verifies that robust/quantile, unsupervised, ordered, nonparametric, panel, covariance, and ANOVA categories produce runs. It also guards the June 2026 linear-model sources, quantile GPU inference rows, ANOVA backend/SciPy coverage, speedup-summary semantics, and the ban on pre-June dashboard sources.

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
