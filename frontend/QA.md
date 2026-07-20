# Frontend Dashboard QA

## Scope

- **Pull request**: #76
- **Branch**: `feature/benchmark-frontend-dashboard`
- **Last updated**: 2026-07-20
- **QA type**: automated contract, build, and browser tests
- **Manual production smoke test**: still required before final merge

This file describes the current dashboard rather than the earlier 1,416-run prototype.

## Data summary

```text
8 registered benchmark sources
8 parsed sources
0 skipped sources
1,774 normalized runs
36 models
```

The canonical source policy requires `source_date >= 2026-06-01`. April ElasticNet, LassoCV, Cox package-comparison, comprehensive-validation, and knockoff artifacts remain excluded from the deployed bundle.

## Automated validation

The current CI matrix verifies:

- project tests on Python 3.9, 3.10, 3.11, and 3.12;
- benchmark parser and schema tests on Python 3.9 and 3.11;
- strict manifest/source-date/SHA validation;
- TypeScript type checking;
- Vite production build;
- deterministic data and deployment-asset staleness;
- Playwright Chromium interaction tests.

## Dashboard checks

### Page loading and deployment

- [x] The three generated JSON files are committed:
  - `benchmark_data.json`;
  - `parse_report.json`;
  - `source_inventory.json`.
- [x] Vite builds to `docs/assets/benchmarks/`.
- [x] Nested-base asset paths are covered by the production configuration.
- [ ] Perform a final manual load from `docs/assets/benchmarks/index.html` before merge.
- [ ] Confirm no browser-console error in the manually served production build.

### Navigation and filter state

- [x] Default environment is selected only when it has runs.
- [x] Default category avoids a valid-but-empty initial view.
- [x] Category search is wired to English and Chinese metadata.
- [x] Upstream changes clear incompatible downstream filters.
- [x] Scale chips remain multi-selectable because options are derived without applying the active scale filter.
- [x] Backend filtering applies to statgpu rows only.
- [x] External frameworks are hidden by default and are context-aware.

### Metric scope

- [x] Scope control supports All, Fit, CV, Inference, Prediction, and Selection.
- [x] Existing inference rows are directly selectable.
- [x] CV remains visible as disabled `CV (0)` until a current structured CV source is registered.
- [x] Overview rows show an explicit Scope column.
- [x] Metric panels appear above the potentially long overview table.

### Timing chart

- [x] Comparison groups define the x-axis once.
- [x] Every framework/backend/implementation series supplies a value or `null` at each group index.
- [x] Missing backend values do not shift bars under unrelated labels.
- [x] Focused mode applies representative-scale and Auto/best rules only to charts.
- [x] Full matrix restores the broader filtered chart matrix.
- [x] Tooltip values use the same normalized timing records as the table.
- [x] Existing ECharts instances are disposed before re-render.

### Speedup chart

- [x] Computed and runner-reported speedups use distinct semantics.
- [x] Computed speedups carry a matched `reference_run_id`.
- [x] Runner-reported rows use an `Ⓡ` marker and are not silently recomputed.
- [x] A dashed 1× parity marker is present.
- [x] The global speedup headline uses runner-reported GPU rows only.

### Overview table and metric panels

- [x] Sorting supports null-last ordering and deterministic run-id tie breaks.
- [x] Show all / Show first 200 uses `Infinity` / `200` state and renders the requested count.
- [x] Validation, Accuracy, Inference, Prediction, Convergence, and Selection panels render only when relevant rows exist.
- [x] The Inference panel displays method, penalty, backend, scale, timing scope, BSE, Wald statistic, p-value, status, and source.

## Source-matrix regression coverage

Automated tests guard:

- CoxPH Breslow plus Efron variants;
- both complete GAM comparison variants and all three GAM scales;
- both aligned Panel scales;
- all 131 Unsupervised source rows and corrected capped-feature labels;
- all PR #74 inference configurations;
- ANOVA functions and SciPy reference rows;
- June 2026 linear-model sources;
- removal of pre-June framework controls;
- Inference scope and CV frontend readiness.

## Known coverage gaps

These are benchmark-data gaps rather than hidden frontend rows:

- Bisquare/Fair and full robust GPU comparisons;
- current CV benchmark sources;
- large-scale Ordered crossover;
- synchronization-safe ANOVA crossover;
- complete Covariance, Nonparametric, Feature Selection, Penalized Survival, extended Panel, Distribution, and Multiple Testing sources.

Detailed plans are under `docs/benchmark-dashboard/`.

## Merge gate

Before merging PR #76:

1. all required CI checks must pass on the final functional head;
2. unresolved review threads must be resolved or explicitly dispositioned;
3. the PR description and benchmark indexes must match the generated bundle;
4. a final manual production smoke test must pass;
5. generated data and deployment assets must remain current.
