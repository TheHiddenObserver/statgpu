# Frontend Dashboard QA

## Scope

- **Website issue**: #130
- **Last updated**: 2026-09-05
- **QA type**: data contract, reproducible build, link, accessibility, and browser tests

This file describes the dashboard as deployed inside the assembled public site.

## Data summary

The deployment bundle is generated from the canonical manifest during CI and is
not committed. Treat the generated `parse_report.json` and
`source_inventory.json` as the source of truth for run, model, and source
counts. Do not duplicate mutable counts in this QA checklist.

## Automated validation

The `Website and Benchmark Frontend` workflow verifies:

- benchmark parser and schema tests on Python 3.9 and 3.11;
- strict manifest/source-date/SHA validation;
- TypeScript source, E2E, and provider-contract type checking;
- VitePress plus Vite production assembly at `/statgpu/` and `/`;
- byte-for-byte deterministic final-artifact hashes;
- a clean repository tree including non-ignored untracked files;
- dashboard Chromium regression tests;
- assembled-site production QA in Chromium, Firefox, and WebKit.

`npm run site:build` from the repository root builds and verifies the complete
site. The faster dashboard regression suite uses the Vite development server;
the production suite previews the assembled `.site-dist/` artifact.

## Dashboard checks

### Page loading and deployment

- [x] The three generated JSON files are created in ignored staging storage:
  - `benchmark_data.json`;
  - `parse_report.json`;
  - `source_inventory.json`.
- [x] Vite builds to ignored `frontend/dist/` and the site assembler copies it to `.site-dist/dashboard/`.
- [x] Project-path and custom-domain root-base builds are both verified.
- [x] Internal deployment links are checked case-sensitively against the assembled artifact.
- [x] Production console/page errors are covered by browser QA.
- [x] Pull requests run validation without publishing; only a push to `master` may publish the Pages artifact.

### Navigation and filter state

- [x] Default environment is selected only when it has runs.
- [x] Session-level environments may be grouped by `physical_env_id` for the hardware selector without discarding member `env_id` values.
- [x] Default category avoids a valid-but-empty initial view.
- [x] Category search is wired to English and Chinese metadata.
- [x] Upstream changes clear incompatible downstream filters.
- [x] Scale chips remain multi-selectable because options are derived without applying the active scale filter.
- [x] Backend filtering applies to statgpu rows only.
- [x] External frameworks are hidden by default and are context-aware.

### Metric scope

- [x] Scope control supports All, Fit, CV, Inference, Prediction, and Selection.
- [x] Scope buttons are enabled only when matching current rows exist.
- [x] Current inference rows are directly selectable.
- [x] Current CV rows are directly selectable, including preserved historical failure evidence and repaired post-fix evidence.
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
- [x] Validation, Accuracy, Inference, Prediction, Convergence, Selection, and Cross-validation panels render only when relevant rows exist.
- [x] Inference rows retain their source identity.
- [x] CV and Validation panels retain benchmark-session and source identity when multiple sessions are grouped into one hardware selector entry.

## Source-matrix regression coverage

Automated tests exercise the canonical bundle rather than relying on a frozen
source count. Current high-signal coverage includes:

- CoxPH Breslow plus Efron variants;
- GAM comparison variants;
- Panel timing, diagnostics, covariance, and physical validation evidence;
- the maintained unsupervised matrix and corrected capped-feature labels;
- PR #74 inference configurations;
- current canonical CV evidence, including the preserved pre-fix Torch failure and the PR #116 repaired row;
- ANOVA functions and external reference rows;
- current linear/GLM and Gaussian-inference evidence;
- metric-scope, environment-grouping, source-provenance, and provider-failure contracts.

For exact live coverage and unresolved evidence gaps, use the generated
`source_inventory.json` and the maintained method coverage matrix rather than
this prose file.

## Merge gate

Before merging the current website PR:

1. all required hosted checks must pass on the exact final head;
2. the assembled project-path and root-base site builds must both pass verification;
3. complete artifact hashes must reproduce exactly;
4. dashboard regression and Chromium/Firefox/WebKit production QA must pass;
5. unresolved review findings must be fixed or explicitly dispositioned;
6. the PR description must reflect the final validation state and user-visible runtime/reporting changes;
7. generated deployment artifacts must remain ignored and the repository tree must remain clean.
