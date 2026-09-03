# Frontend Dashboard QA

## Scope

- **Website issue**: #130
- **Last updated**: 2026-09-03
- **QA type**: data contract, reproducible build, link, accessibility, and browser tests

This file describes the dashboard as deployed inside the assembled public site.

## Data summary

The deployment bundle is generated from the canonical manifest during CI and is
not committed. Treat the generated `parse_report.json` and
`source_inventory.json` as the source of truth for run, model, and source
counts.

## Automated validation

The current CI matrix verifies:

- benchmark parser and schema tests on Python 3.9 and 3.11;
- strict manifest/source-date/SHA validation;
- TypeScript type checking;
- VitePress plus Vite production assembly at `/statgpu/` and `/`;
- deterministic final-artifact hashes and tracked-tree cleanliness;
- Playwright interaction and cross-browser production tests.

Run `npm run site:build` from the repository root before
`npm run test:e2e --prefix frontend`; the browser suite previews the built
`frontend/dist/` resources.

## Dashboard checks

### Page loading and deployment

- [x] The three generated JSON files are created in ignored staging storage:
  - `benchmark_data.json`;
  - `parse_report.json`;
  - `source_inventory.json`.
- [x] Vite builds to ignored `frontend/dist/` and the site assembler copies it to `.site-dist/dashboard/`.
- [x] Nested-base asset paths are covered by the production configuration.
- [ ] Perform a final manual load from `/statgpu/dashboard/` using `npm run site:preview` before merge.
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
