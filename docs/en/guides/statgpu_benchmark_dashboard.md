# statgpu Benchmark Dashboard

The statgpu benchmark dashboard provides a common view of timing, speedup, numerical quality, inference, convergence, prediction, validation, and feature-selection metrics.

The browser is a presentation layer over a generated benchmark bundle. Raw result files are parsed and validated in Python before the frontend is built.

## Current coverage

The canonical manifest registers **eight benchmark sources**, all dated **2026-06-01 or later**. The generated bundle contains **1,774 normalized runs across 36 models**:

| Source | Frontend coverage |
|---|---|
| `p2_benchmark_20260617.json` | Empirical covariance, Nystroem, RBF kernel, and spline benchmarks |
| `penalized_glm_perf_20260622.json` | Penalized GLM and recent squared-error NumPy/CuPy/Torch timings |
| `coxph_efron_20260622.json` | CoxPH Efron variants and cross-backend timing |
| `glm_solver_20260623.json` | GLM and squared-error solver speedups |
| `loss_functions_20260623.json` | Robust/quantile timing, validation, sklearn comparison, and CoxPH Breslow rows |
| `new_modules_full_20260624.json` | Panel, two complete fixed-lambda GAM comparison variants, and ANOVA benchmarks |
| `unsupervised_20260627.json` | Complete source matrix for PCA, clustering, decomposition, mini-batch methods, UMAP, and t-SNE |
| `ordered_inference_pr74.json` | Ordered, Quantile, sandwich, oracle, and bootstrap inference configurations |

These sources populate penalized GLM and GLM, recent linear models, robust/quantile regression, survival analysis, unsupervised learning, ordered models, nonparametric methods, panel models, covariance estimation, and ANOVA.

GAM coverage includes `1K×3`, `10K×5`, and `100K×10` for two distinct variants: the ordinary pyGAM comparison and the uniform-knot precision-aligned comparison. Each variant contains statgpu NumPy/CuPy/Torch rows and a pyGAM reference, together with runner-reported speedup and prediction-difference validation. The solver is represented as fixed `lambda=1.0`, matching the source runner rather than incorrectly labelling the work as GCV.

Aligned Panel coverage includes `10K×10` and `100K×20` for both PanelOLS and RandomEffects. Each model/scale contains statgpu NumPy/CuPy/Torch rows and a linearmodels reference, together with runner-reported speedup and coefficient-relative-error metrics.

Unsupervised coverage retains all 131 source rows. PCA, KMeans, GaussianMixture, NMF, TruncatedSVD, IncrementalPCA, MiniBatchKMeans, and MiniBatchNMF expose every small/medium/large configuration. DBSCAN exposes both 10-dimensional and 50-dimensional variants at all three scales; AgglomerativeClustering, UMAP, and t-SNE expose every scale actually run. Large labels follow the arrays passed to fit, so estimators capped at 50 input features are correctly shown as `100K×50` instead of the uncapped `100K×100` runner template.

The PR #74 source now exposes all of its methods. In addition to Ordered Logit/Probit and Quantile kernel/bootstrap inference, it includes penalized-logistic HC0 sandwich inference, penalized-logistic SCAD oracle inference, and penalized-linear bootstrap inference. These additional configurations are explicitly marked as fit-plus-inference timings.

ANOVA coverage includes one-way ANOVA, two-way ANOVA, Welch ANOVA, Tukey HSD, and Bonferroni correction at three scales on NumPy, CuPy, and Torch. One-way ANOVA also contains aligned SciPy timing and F-statistic validation rows.

The current bundle should not be interpreted as complete coverage of every implementation in the repository. In particular, the robust source contains CPU Huber and Quantile fit comparisons but no Bisquare, Fair, or robust-loss GPU fit matrix. Ordered scales remain too small to locate a GPU crossover; covariance currently contains only EmpiricalCovariance; Feature Selection has no eligible structured source; and ANOVA has too few synchronization-safe scale points for a precise crossover interval.

A June distribution benchmark also exists and reports 139/139 SciPy precision checks plus NumPy/CuPy/Torch timings for 15 distributions. It is not yet registered because only a rounded Markdown report is committed, without raw repeats, per-check errors, or a structured category/source contract. It is recorded as a P1 structured-conversion or rerun task rather than being ignored or represented with invented metadata.

Coverage gaps and source-quality findings are recorded in:

- `docs/benchmark-dashboard/domain-coverage-audit-plan.md`;
- `docs/benchmark-dashboard/method-coverage-audit.md`;
- `docs/benchmark-dashboard/remaining-module-audit.md`;
- `docs/benchmark-dashboard/robust-loss-comparison-plan.md`;
- `docs/benchmark-dashboard/penalized-robust-quantile-plan.md`.

April 2026 ElasticNet, LassoCV, comprehensive-validation, Cox package-comparison, and knockoff sources are not registered. The feature-selection category remains part of Schema v1.1 but is intentionally empty until a June 2026-or-later structured benchmark is available.

The source registry is `dev/benchmarks/frontend_sources.json`. It sets `minimum_source_date` to `2026-06-01`, and every registered source must provide an explicit `source_date` on or after that date. Generated files are committed in `frontend/public/data/` and `docs/assets/benchmarks/data/`.

## Filters

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

Changing an upstream value clears incompatible downstream selections. External frameworks are hidden by default and offered only when relevant to the current context.

External frameworks currently used by registered June-or-later sources are scikit-learn, SciPy, statsmodels, linearmodels, and pyGAM.

## Chart view modes

The chart toolbar has two explicit modes:

- **Focused** is the default. When no scale chip is selected, the timing chart chooses the largest workload in the current filter context. If canonical Auto/best dispatch groups are available, it keeps those groups while preserving aligned external reference rows. Focused timing is capped at 14 groups and focused speedup at 18 rows.
- **Full matrix** shows all filtered chart groups up to the configured larger chart limits.

This distinction is presentation-only. Switching chart modes does not change the table, selected categories, model filters, scale chips, backend selection, or external-framework state. The chart subtitle states which representative scale and solver rule were applied.

This approach is preferred over silently setting scale and solver filters because it keeps the table/filter contract exact while providing a readable first view. It is also preferred over globally truncating data because users can restore the complete matrix with one visible control.

## Charts

### Timing

The timing chart uses `metrics.timing.fit_time_ms`. Group identity includes comparison, environment, model, case, method configuration, variant, loss, penalty, solver, and scale. Series identity includes framework, backend, and implementation.

Focused labels omit repeated scale and Auto/best text. Full-matrix labels use two lines. Both modes use bounded label width and full tooltip text, avoiding a dense diagonal label wall.

### Speedup

A value above one means faster than the reference; a value below one is a slowdown. A dashed gray line marks 1× parity, with a compact `1×` badge above the bar area. Horizontal tick labels include the `×` unit.

- **Computed** speedups use `reference time / current time` and carry `reference_run_id`.
- **Runner-reported** speedups are copied from an upstream benchmark and use `reported_semantics: "reported_by_runner"`. They are marked with `Ⓡ` and a subtle border rather than a patterned bar fill.

Semantic validation checks computed references, positive timings, compatible identities, and numerical agreement with the timing ratio.

The global summary card shows only the fastest runner-reported GPU speedup. Computed timing ratios remain available in the chart and raw data for auditing; the two reference semantics are not mixed into one headline.

## Visual theme

The page uses a low-saturation blue-gray application background, white cards, soft borders and shallow shadows. Backend and framework colors are deliberately muted so that long benchmark sessions remain comfortable to scan. Selected categories use a light primary tint rather than a high-contrast block, and chart grid lines are lighter than the 1× parity marker.

The responsive layout keeps paired charts on large screens and stacks them below 1080 px. Summary cards collapse from six to three columns below 1450 px.

## Overview and metric panels

The overview table supports stable keyed sorting, a default 200-row limit, “Show all,” source provenance, and framework-aware display.

Panels appear only when filtered rows contain the corresponding metric group:

- **Validation**: pass/warn/fail checks and tolerances.
- **Accuracy**: coefficient and standard-error differences.
- **Inference**: BSE, Wald statistic, p-value, backend, scale, and status.
- **Prediction**: train/test MSE, noiseless MSE, selected alpha, and C-index.
- **Convergence**: iteration summaries and convergence rates.
- **Selection**: precision, recall, FDP, F1, Jaccard, FDR, and selected-set size when a current source exists.

The Inference panel covers Ordered Logit/Probit, Quantile kernel/bootstrap inference, and the restored sandwich/oracle/bootstrap configurations. ANOVA one-way rows expose SciPy-relative F-statistic validation in the Validation panel.

## Metric provenance

- `measured`: directly observed;
- `reported`: copied from an upstream report;
- `computed`: deterministically derived by a parser;
- `partial`: incomplete or partially comparable.

Quality records provenance, not whether a method performed well.

## Generated bundle

The frontend loads:

- `benchmark_data.json`: registries and normalized runs;
- `parse_report.json`: source/run counts and structured issues;
- `source_inventory.json`: catalog, registration, availability, and parsed counts.

All three files share one `generation_id`. In canonical mode, `eligible_total`, `registered_sources`, `available_sources`, and `parsed_sources` refer only to the eight manifest-registered June-or-later sources.

## Reproduce and test

```bash
python -m pip install -U pytest jsonschema
pytest \
  dev/tests/test_benchmark_frontend_data.py \
  dev/tests/test_frontend_contracts.py \
  dev/tests/test_frontend_domain_coverage.py -v

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

## Adding a source

1. Copy canonical JSON under `results/benchmark_frontend_sources/`.
2. Register SHA256, environment, comparison, parser, allowed issue codes, and `source_date` in `frontend_sources.json`.
3. Ensure `source_date` is on or after the manifest's `minimum_source_date`.
4. Implement or reuse a parser and register it in `registry.py`.
5. Return schema-compliant runs with canonical case/method identities.
6. Add parser, date-policy, domain-coverage, and interaction tests.
7. Regenerate the bundle and rebuild deployed assets.

Technical references:

- `docs/benchmark-dashboard/schema-v1.1.md`
- `docs/benchmark-dashboard/parser-contracts.md`
- `docs/benchmark-dashboard/aggregation-contract.md`
- `docs/benchmark-dashboard/domain-coverage-audit-plan.md`
- `docs/benchmark-dashboard/method-coverage-audit.md`
- `docs/benchmark-dashboard/remaining-module-audit.md`
- `docs/benchmark-dashboard/robust-loss-comparison-plan.md`
- `docs/benchmark-dashboard/penalized-robust-quantile-plan.md`
