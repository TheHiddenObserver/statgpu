# statgpu Benchmark Dashboard

The statgpu benchmark dashboard provides a common view of timing, speedup, numerical quality, inference, convergence, prediction, validation, and feature-selection results.

The browser is a presentation layer over a generated benchmark bundle. Raw result files are parsed and validated in Python before the frontend is built.

## Current coverage

As of PR #78, the canonical manifest registers **14 benchmark sources** handled by **13 parser implementations**. The generated bundle contains **1,623 normalized runs across 36 models**.

All published categories now contain runs:

- Penalized GLM and GLM.
- Linear models, including June 2026 squared-error performance and solver results.
- Robust and quantile regression.
- Survival analysis.
- Unsupervised learning.
- Ordered models.
- Nonparametric methods.
- Panel models.
- Covariance estimation.
- Feature selection.

The newly connected sources include:

| Source | Frontend coverage |
|---|---|
| `loss_functions_20260623.json` | Robust/quantile timing, validation, sklearn comparison |
| `ordered_inference_pr74.json` | Ordered logit/probit inference and quantile kernel/bootstrap GPU inference |
| `unsupervised_20260627.json` | PCA, clustering, decomposition, UMAP and t-SNE timings |
| `new_modules_full_20260624.json` | Panel and aligned GAM comparisons |
| `p2_benchmark_20260617.json` | Covariance, Nystroem, RBF kernel and spline benchmarks |
| `penalized_glm_perf_20260622.json` | Recent squared-error NumPy/CuPy/Torch timings under `linear_models` |
| `glm_solver_20260623.json` | Recent squared-error solver speedups under `linear_models` |

The source registry is `dev/benchmarks/frontend_sources.json`. Generated files are committed in `frontend/public/data/` and `docs/assets/benchmarks/data/`.

## Filters

Filters follow a dependency chain:

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

Changing an upstream value clears incompatible downstream selections. External frameworks are hidden by default and are offered only when relevant to the current filter context.

Registered external frameworks include scikit-learn, glmnet, statsmodels, lifelines, scikit-survival, knockpy, linearmodels, and pyGAM.

## Charts

### Timing

The timing chart uses `metrics.timing.fit_time_ms`. Group identity includes comparison, environment, model, case, method configuration, variant, loss, penalty, solver, and scale. Series identity includes framework, backend, and implementation.

This prevents different CoxPH variants, solver configurations, unsupervised algorithms, ordered inference methods, and external packages from overwriting one another.

### Speedup

A value above one means faster than the reference; a value below one is a slowdown.

- **Computed** speedups use `reference time / current time` and carry `reference_run_id`.
- **Runner-reported** speedups are copied from an upstream benchmark and use `reported_semantics: "reported_by_runner"`.

Semantic validation checks computed references, positive timings, compatible comparison/scale identity, and numerical agreement with the timing ratio.

## Overview and metric panels

The overview table supports stable keyed sorting, a default 200-row limit, “Show all,” source provenance, and framework-aware display.

Panels appear only when the filtered rows contain the corresponding metric group:

- **Validation**: pass/warn/fail checks and tolerances.
- **Accuracy**: coefficient and standard-error differences.
- **Inference**: BSE, Wald statistic, p-value, backend, scale, and status.
- **Prediction**: train/test MSE, noiseless MSE, selected alpha, and C-index.
- **Convergence**: iteration summaries and convergence rates.
- **Selection**: precision, recall, FDP, F1, Jaccard, FDR, and selected-set size.

The Inference panel is particularly relevant to ordered logit/probit and quantile kernel/bootstrap results.

## Metric provenance

Metric quality labels are:

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

All three files share one `generation_id`. A mismatch means files from different generator executions were mixed.

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

1. Copy the canonical JSON under `results/benchmark_frontend_sources/`.
2. Register its SHA256, environment, comparison, parser, and allowed issue codes in `frontend_sources.json`.
3. Implement or reuse a parser in `frontend_data/parsers/` and register it in `registry.py`.
4. Return schema-compliant runs and model metadata with canonical case/method identities.
5. Add parser, coverage, and interaction tests.
6. Regenerate the three-file bundle and rebuild deployed assets.

Technical references:

- `docs/benchmark-dashboard/schema-v1.1.md`
- `docs/benchmark-dashboard/parser-contracts.md`
- `docs/benchmark-dashboard/aggregation-contract.md`
