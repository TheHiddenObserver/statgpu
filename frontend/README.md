# statgpu Benchmark Frontend

Interactive benchmark dashboard for statgpu, built with Vite, TypeScript, and
ECharts. The dashboard consumes a generated, versioned benchmark bundle and is
assembled under `/dashboard/` in the public website artifact.

## Current coverage

The canonical dashboard is manifest-driven and restricted by the maintained
source policy in `dev/benchmarks/frontend_sources.json`. Run counts, model
counts, registered-source counts, and unresolved evidence gaps change as new
validated benchmark packages are added, so they are intentionally not frozen in
this README.

Use the generated deployment metadata as the live source of truth:

- `source_inventory.json` — audited source catalog, eligibility/registration
  counts, and method-coverage status;
- `parse_report.json` — files parsed, runs generated, and structured parser
  issues;
- `benchmark_data.json` — normalized environments, models, comparisons, and
  benchmark runs.

Current canonical evidence spans linear and generalized linear models,
regularized/CV workflows, robust and quantile regression, survival analysis,
unsupervised learning, ordered models, nonparametric methods, panel models,
covariance estimation, ANOVA, inference, and physical-validation records where
structured sources are available. Historical evidence remains preserved when it
is needed to document a before/after repair rather than being rewritten as a
current measurement.

The UI may group several benchmark sessions into one hardware selector entry via
`physical_env_id`. That field is a maintainer-assigned **hardware-class grouping
key**, not proof that every session used the same physical host. Session IDs and
source provenance remain visible in panels where historical/current evidence can
coexist.

## What the dashboard shows

- Hardware-group and category navigation.
- Progressive filters for metric scope, model, variant, penalty, solver, scale,
  backend, and external framework.
- Explicit **Focused** and **Full matrix** chart views.
- Timing and speedup charts when those metrics exist for the selected runs.
- A sortable and paginated overview table.
- Validation, accuracy, inference, prediction, convergence, selection, and
  cross-validation panels.
- Benchmark-session/source provenance for grouped CV and validation evidence.
- Parse-report and source-inventory metadata.

Focused is the default chart view. It keeps charts readable by selecting a
representative scale and Auto/best solver groups when possible. This is a
chart-only presentation rule: it does not change the overview table or filter
state. Full matrix restores all filtered chart groups.

Speedups have two distinct meanings:

- **Computed**: reference timing divided by current-run timing. The generated
  record contains `reference_run_id`.
- **Reported by runner**: copied from a benchmark runner that already computed
  the speedup. These rows carry an `Ⓡ` marker and do not imply frontend
  recomputation.

Validation-only runs do not participate in timing/speedup charts unless their
canonical source actually contains timing evidence.

## Data provider boundary

The loading boundary is `BenchmarkDataProvider` in
`src/providers/benchmark.ts`. Phase 1 uses `StaticJsonBenchmarkProvider`, which
loads the three generated files relative to the dashboard base URL, validates
schema/version and `generation_id` consistency, and caches only a successful
bundle. A transient required-data failure is not permanently cached.

A future API provider must return the same normalized bundle. Transport,
pagination, authentication, retries, and service-specific DTOs stay behind the
provider boundary rather than changing chart/filter code. See
`docs/benchmark-dashboard/provider-contract.md`.

## Requirements

- Node.js 24.13.0 and npm 11.6.2 (pinned by the repository).
- Python 3.9 or 3.11 for maintained benchmark-data validation jobs.
- Python test packages: `pytest` and `jsonschema`.

## Development

Run commands from the repository root unless a command explicitly changes
directory.

```bash
npm ci
npm ci --prefix frontend

python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

npm run dev --prefix frontend
# Open http://localhost:5173
```

Generated dashboard JSON under `frontend/public/data/` is ignored. Canonical
source artifacts remain tracked through the manifest and source catalog.

## Validation and tests

Focused benchmark/parser validation:

```bash
python -m pip install -U pytest 'jsonschema[format]'
pytest \
  dev/tests/test_benchmark_frontend_data.py \
  dev/tests/test_benchmark_catalog.py \
  dev/tests/test_benchmark_inventory_v2.py \
  dev/tests/test_benchmark_cv_source.py \
  dev/tests/test_frontend_contracts.py \
  dev/tests/test_frontend_cv_determinism.py \
  dev/tests/test_frontend_domain_coverage.py \
  dev/tests/test_panel_stage_b_frontend_source.py \
  dev/tests/test_panel_stage_b_applicable_hausman_parser.py \
  dev/tests/test_panel_stage_c_frontend_source.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources
```

Dashboard regression:

```bash
npm ci --prefix frontend
npm run typecheck --prefix frontend
npm exec --prefix frontend -- playwright install chromium
npm run test:e2e --prefix frontend
```

Complete assembled-site validation:

```bash
npm ci
npm ci --prefix frontend
npm run site:build
npm exec --prefix frontend -- playwright install --with-deps chromium firefox webkit
npm run test:e2e:production --prefix frontend
```

The `Website and Benchmark Frontend` workflow additionally builds both the
GitHub Pages project base (`/statgpu/`) and custom-domain root base (`/`),
compares complete artifact hashes for reproducibility, requires a clean
repository tree, and runs production QA in Chromium, Firefox, and WebKit.
Pull-request runs never deploy; publication is restricted to a push to
`master`.

## Project structure

```text
frontend/
├── public/data/                 # CI-generated, ignored benchmark bundle
├── e2e/                         # Fast Chromium dashboard regressions
├── e2e-production/              # Assembled-site Chromium/Firefox/WebKit QA
└── src/
    ├── main.ts                  # Application orchestration
    ├── schema.ts                # Schema v1.1.0 TypeScript types
    ├── data.ts                  # Provider delegation and filtering
    ├── providers/
    │   └── benchmark.ts         # Static provider + future provider contract
    ├── state.ts                 # Defaults, env groups, cascade resets
    ├── identity.ts              # Chart/group identities
    ├── charts/
    └── components/
        ├── Header.ts
        ├── Sidebar.ts
        ├── FilterBar.ts
        ├── OverviewTable.ts
        └── panels/
```

## Data flow

```text
results/benchmark_frontend_sources/*.json
  + dev/benchmarks/frontend_sources.json
        → dev/benchmarks/generate_benchmark_data.py
        → frontend/public/data/{benchmark_data,parse_report,source_inventory}.json
        → StaticJsonBenchmarkProvider
        → dashboard filters/charts/panels
        → Vite build
        → frontend/dist/
        → .site-dist/dashboard/
```

All three generated JSON files share one `generation_id`, computed from the
complete logical bundle after removing the `generation_id` fields themselves.

## Documentation

- Dashboard guide: `docs/en/guides/statgpu_benchmark_dashboard.md`
- Schema v1.1: `docs/benchmark-dashboard/schema-v1.1.md`
- Provider contract: `docs/benchmark-dashboard/provider-contract.md`
- Website deployment: `docs/website-deployment.md`
- Production QA: `frontend/QA.md`
- Source/catalog/coverage policy: generated `source_inventory.json` plus the
  maintained files under `dev/benchmarks/` and `docs/benchmark-dashboard/`.
