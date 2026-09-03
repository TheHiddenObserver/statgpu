# statgpu Benchmark Frontend

Interactive benchmark dashboard for statgpu, built with Vite, TypeScript, and ECharts. The dashboard consumes a generated, versioned benchmark bundle and is assembled under `/dashboard/` in the GitHub Pages artifact.

## Current coverage

The canonical dashboard is restricted to benchmark sources dated **2026-06-01 or later**. The manifest currently registers **11 sources**, producing **1,852 normalized runs across 46 models**:

- `p2_benchmark_20260617.json`;
- `penalized_glm_perf_20260622.json`;
- `coxph_efron_20260622.json`;
- `glm_solver_20260623.json`;
- `loss_functions_20260623.json`;
- `new_modules_full_20260624.json`;
- `unsupervised_20260627.json`;
- `ordered_inference_pr74.json`;
- `cv_benchmark_20260807.json`;
- `results/pr116_p100/cv_benchmark_pr116_p100.json`;
- `panel_stage_b_pr122_p100_20260808.json`.

Covered categories include penalized GLM and GLM, recent linear models, robust and quantile regression, survival analysis, unsupervised learning, ordered models, nonparametric methods, panel models, covariance estimation, ANOVA, and current cross-validation families.

Survival coverage combines the dedicated Efron benchmark with the aligned Breslow rows embedded in `loss_functions_20260623.json`. Breslow contributes five scales, NumPy/CuPy/Torch and statsmodels timings, runner-reported speedups against statsmodels, and CPU/CuPy precision validation. The richer Efron source retains its light-ties and heavy-ties variants.

GAM coverage exposes two distinct fixed-lambda pyGAM comparison variants at `1K×3`, `10K×5`, and `100K×10`: the ordinary source comparison and the uniform-knot precision-aligned comparison. Each variant includes NumPy, CuPy, Torch, and pyGAM timing, reported speedup, and prediction-difference validation. Other nonparametric and covariance families remain limited by available source artifacts rather than hidden frontend rows.

Panel coverage has two complementary evidence classes. The June 24 timing source exposes aligned `10K×10` and `100K×20` PanelOLS and RandomEffects comparisons with NumPy, CuPy, Torch, and linearmodels timing, runner-reported speedup, and coefficient-relative-error metrics. PR #122 additionally registers `results/benchmark_frontend_sources/panel_stage_b_pr122_p100_20260808.json` (SHA256 `882892c6e3077fe3b9f6084212647311da795fd05d1ed9f12ec53da1e05d0d4d`) as **validation-only** P100 evidence for PooledOLS, BetweenOLS, FirstDifferenceOLS, PanelOLS, RandomEffects, and FamaMacBeth. It contributes 34 CuPy/Torch validation rows covering Stage-B fit statistics/specification diagnostics, backend provenance, and Stage-A coefficient-inference regression. No timing was collected by that physical validator, so these rows deliberately expose neither timing nor speedup metrics.

Unsupervised coverage now retains all 131 rows present in the June 27 source rather than selecting one scale per estimator. This includes complete small/medium/large matrices for PCA, KMeans, GaussianMixture, NMF, TruncatedSVD, IncrementalPCA, MiniBatchKMeans, and MiniBatchNMF; both DBSCAN dimensional variants; and every feasible AgglomerativeClustering, UMAP, and t-SNE row. Large input dimensions are labelled from the arrays actually passed to fit, so capped estimators correctly show `100K×50` rather than the uncapped `100K×100` template.

The PR #74 source now contributes all of its inference methods: Ordered Logit/Probit, Quantile kernel/bootstrap inference, penalized-logistic HC0 sandwich and oracle inference, and penalized-linear bootstrap inference. The latter three are explicitly labelled as fit-plus-inference timing configurations.

ANOVA coverage includes one-way ANOVA, two-way ANOVA, Welch ANOVA, Tukey HSD, and Bonferroni correction on NumPy, CuPy, and Torch. One-way ANOVA also includes aligned SciPy timing and F-statistic validation rows.

The linear-model category uses the June 2026 squared-error rows from `penalized_glm_perf_20260622.json` and `glm_solver_benchmark_20260623.json`. April 2026 ElasticNet, LassoCV, comprehensive-validation, Cox package-comparison, and knockoff results are intentionally not registered.

Current June-or-later sources provide external comparisons through scikit-learn, SciPy, statsmodels, linearmodels, and pyGAM. The feature-selection category remains part of Schema v1.1, but it is intentionally empty until a June 2026-or-later structured benchmark is available. A June distribution report also exists, but it remains outside the dashboard until its rounded Markdown tables are converted or rerun as a structured source with full timing and precision provenance.

## What the dashboard shows

- Environment and category navigation.
- Progressive filters for model, variant, penalty, solver, scale, backend, and external framework.
- Explicit **Focused** and **Full matrix** chart views.
- Timing and speedup charts when those metrics exist for the selected runs.
- A sortable and paginated overview table.
- Validation, accuracy, inference, prediction, convergence, and selection panels.
- Parse-report and source-inventory metadata.

Focused is the default chart view. It keeps the timing chart readable by selecting the largest workload in the current unscaled context and retaining Auto/best solver groups when available. This is a chart-only presentation rule: it does not change the table or filter state. Full matrix restores all filtered chart groups.

The visual theme uses a low-saturation blue-gray background, white cards, soft borders, muted backend colors, and responsive chart/card layouts; it changes presentation only, not benchmark semantics.

Speedups have two distinct meanings:

- **Computed**: reference timing divided by current-run timing. The generated record contains `reference_run_id`.
- **Reported by runner**: copied from a benchmark runner that already computed the speedup. These rows carry an `Ⓡ` marker and do not imply frontend recomputation.

Validation-only runs such as PR #122 do not participate in either speedup class because their source contains no timing measurements.

The speedup chart uses a dashed gray 1× parity line with a compact in-chart `1×` badge and `×` axis labels. Runner-reported bars use a subtle border instead of a patterned fill. The global headline card displays only the fastest runner-reported GPU speedup; computed ratios remain available in the chart and raw data for auditing.

## Requirements

- Node.js 24.13.0 and npm 11.6.2 (pinned by the repository).
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
  dev/tests/test_benchmark_catalog.py \
  dev/tests/test_benchmark_inventory_v2.py \
  dev/tests/test_frontend_contracts.py \
  dev/tests/test_frontend_domain_coverage.py \
  dev/tests/test_panel_stage_b_frontend_source.py -v

python dev/benchmarks/generate_benchmark_data.py --check --strict-sources

cd frontend
npm run typecheck
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

The domain-coverage suite verifies robust/quantile, survival, unsupervised, ordered, nonparametric, panel, covariance, and ANOVA runs. It specifically guards CoxPH Breslow timing/speedup/validation, both complete GAM comparison variants, both aligned Panel timing scales, the PR #122 34-row validation-only Panel source, all 131 Unsupervised rows and corrected scale labels, all PR #74 inference methods, Focused/Full matrix switching, the dashed 1× parity contract, June 2026 linear-model sources, ANOVA backend/SciPy coverage, speedup-summary semantics, and removal of pre-June framework controls.

## Production build and deployment artifact

```bash
python dev/benchmarks/generate_benchmark_data.py \
  --out frontend/public/data/benchmark_data.json \
  --report frontend/public/data/parse_report.json \
  --inventory-out frontend/public/data/source_inventory.json \
  --deterministic --strict-sources

npm ci
npm ci --prefix frontend
npm run site:build
npm run site:preview
```

The dashboard Vite build writes to ignored `frontend/dist/`. The site assembler copies it to ignored `.site-dist/dashboard/` beside the VitePress output. CI verifies internal links, the project Pages base path, bundle versions, size budgets, build reproducibility, and a clean tracked tree before uploading the Pages artifact. Generated deployment files are not committed.

## Project structure

```text
frontend/
├── public/data/                 # CI-generated, ignored benchmark bundle
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
        → frontend/dist/
        → .site-dist/dashboard/
```

All three generated JSON files share one `generation_id`, computed from the complete bundle after removing the `generation_id` fields themselves.

The loading boundary is `BenchmarkDataProvider` in
`src/providers/benchmark.ts`. Phase 1 uses the static JSON implementation.
Future API transport, pagination, authentication, and retries stay inside an
alternative provider rather than changing chart and filter code.

## Documentation

- Dashboard guide: `docs/en/guides/statgpu_benchmark_dashboard.md`
- Schema v1.1: `docs/benchmark-dashboard/schema-v1.1.md`
- Provider contract: `docs/benchmark-dashboard/provider-contract.md`
- Website deployment: `docs/website-deployment.md`
- Parser contract: `docs/benchmark-dashboard/parser-contracts.md`
- Aggregation contract: `docs/benchmark-dashboard/aggregation-contract.md`
- Domain coverage audit and benchmark plan: `docs/benchmark-dashboard/domain-coverage-audit-plan.md`
- Method-level coverage audit: `docs/benchmark-dashboard/method-coverage-audit.md`
- Remaining-module audit: `docs/benchmark-dashboard/remaining-module-audit.md`
- Robust-loss comparison plan: `docs/benchmark-dashboard/robust-loss-comparison-plan.md`
- Penalized robust/quantile plan: `docs/benchmark-dashboard/penalized-robust-quantile-plan.md`
- Rollout record: `docs/benchmark-dashboard/rollout-plan.md`
