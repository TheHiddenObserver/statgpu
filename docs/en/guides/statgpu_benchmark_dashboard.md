# statgpu Benchmark Dashboard

[Open the dashboard](../../assets/benchmarks/index.html) to explore timing, speedup, numerical quality, inference, convergence, cross-validation, prediction, validation, and feature-selection metrics when those measurements are available.

## Coverage

The dashboard displays a recorded benchmark bundle, not live measurements of the installed library. It uses sources dated **2026-06-01 or later**. Use the [source inventory](../../assets/benchmarks/data/source_inventory.json) for registered, available, and parsed source counts; the [parse report](../../assets/benchmarks/data/parse_report.json) for generated-run counts and issues; and the [normalized results](../../assets/benchmarks/data/benchmark_data.json) for model entries and individual runs. These quantities need not be equal, and filters further narrow what is visible.

Coverage includes GLM and penalized GLM, linear models, robust and quantile regression, survival, unsupervised learning, ordered models, nonparametric methods, panel models, covariance estimation, ANOVA, and CV families. It is not a complete matrix of every method, backend, scale, or metric supported by statgpu:

- Some robust/quantile records describe CPU fits; they do not establish a robust-loss GPU performance comparison for every loss.
- Ordered-model scales are too limited to establish a general GPU crossover. Covariance-estimator rows cover EmpiricalCovariance rather than every covariance estimator.
- The Feature Selection category has no eligible structured source in this bundle. A category or metric panel can exist even when no matching measurement is available.
- April 2026 ElasticNet, LassoCV, comprehensive-validation, Cox package-comparison, and knockoff sources are outside the dashboard's date range. Rounded distribution summaries without raw repeats and precision records are also excluded.

Do not merge distinct comparisons just because their model names match. For example, GAM has ordinary and uniform-knot precision-aligned pyGAM comparisons at fixed `lambda=1.0`; these are not GCV timings. Panel records include both timed comparisons and separate correctness-only measurements. Unsupervised scales describe the arrays actually fitted, which can be smaller than an experiment's nominal input template. Historical source status describes that experiment, not necessarily the current release.

## Filters

Choose an environment and one or more categories, then narrow the model, variant, penalty, solver, scale, and backend. Changing an upstream selection clears incompatible downstream selections. Scale chips allow multiple scales. Backend filters apply to statgpu rows; external frameworks are offered separately when relevant to the selected comparison and are hidden by default.

Use **Metric scope** to distinguish Fit, CV, Inference, Prediction, and Selection. A row's scope describes the work measured, not simply the model's capabilities. Read the overview table's Scope column before comparing timings.

External reference implementations available in the recorded bundle include scikit-learn, SciPy, statsmodels, linearmodels, and pyGAM. A missing reference in a filter context means no matching comparison is available there.

## Chart view modes

- **Focused** is the default compact view. Without an explicit scale selection, it selects a representative largest workload from the current context and prefers Auto/best solver groups when available. It can emphasize penalized rows in the Penalized GLM category and the default NumPy implementation in Survival. The chart subtitle explains the selection used.
- **Full matrix** restores the broader filtered chart view, subject to chart display limits. Use the overview table and exact-value chart tables to inspect individual records.

These modes affect chart presentation only. They do not change the table's filtered rows, selected categories, model, scales, backend, or external-framework selections. A bar missing from Focused mode is not evidence that a run failed or a backend is unsupported.

## Timing and speedup

The timing chart displays the recorded duration in milliseconds. Comparable groups keep the environment, comparison, model, variant, method configuration, loss, penalty, solver, and scale distinct. Framework, backend, and implementation identify the series. A duration can cover fit plus inference or another labelled scope; do not assume every bar is fit-only.

Correctness-only records without timing do not create timing bars. Missing times and missing uncertainty estimates are not zero.

A speedup is a ratio against a specified reference. Above one means faster, below one slower, and the dashed **1×** line marks equal time.

- **Computed**: reference time divided by current-run time, with a reference run recorded in the data.
- **Runner-reported**: copied from the upstream benchmark and marked **Ⓡ**. Its reference and timing policy must be read in the source; it is not a new measurement performed by the dashboard.

The global summary card uses the fastest runner-reported GPU speedup, rather than mixing reported and computed ratios. That maximum is a selected historical observation, not an expected speedup for every model or your own workload. Correctness-only sources do not participate in speedup summaries.

Match objectives, precision, convergence settings, hardware, and transfer/timing boundaries before comparing ratios. See [benchmark interpretation and reproduction](../benchmarks.md) for a measurement checklist.

## Overview and metric panels

The overview table supports sorting, a default 200-row view, **Show all**, source provenance, and framework-aware labels. Charts also provide filter-synchronized exact-value tables with full labels. Filters, scale chips, sorting, and panel controls support keyboard navigation with visible focus; responsive layouts stack charts on narrower screens.

A panel appears only when the filtered records contain the corresponding metric group:

- **Validation**: pass/warn/fail checks and their tolerances.
- **Cross-validation**: run status, CV evaluation, final-refit and total timing, selected parameters, scores and scoring direction, candidate/fold failures, refit convergence, and explicit non-success reasons.
- **Accuracy**: coefficient and standard-error differences against a reference.
- **Inference**: standard errors, Wald statistics, p-values, backend, scale, and status when recorded.
- **Prediction**: training/test MSE, noiseless MSE, selected alpha, or C-index when recorded.
- **Convergence**: iteration summaries and convergence rates.
- **Selection**: precision, recall, FDP, F1, Jaccard, FDR, or selected-set size when recorded.

A missing panel means there are no matching metric records in the current view, not that a measured value is zero. An unavailable or unsuccessful CV row should be read together with its reason; do not treat it as a successful timing observation. Likewise, a statistical check reported as not applicable provides no test statistic. A reported inference-pass status alone is not a full set of coefficient-level inference results.

## Provenance and limitations

Metric quality labels describe where values came from:

- `measured`: directly observed;
- `reported`: copied from an upstream report;
- `computed`: deterministically derived from source data;
- `partial`: incomplete or only partly comparable.

These labels do not grade how well a method performed. Check numerical errors, tolerances, and convergence separately. Inspect a row's source and environment before using it as evidence for your workload; a historical benchmark does not automatically validate a newer release or a different device.

The three linked JSON files share a `generation_id`, identifying the same generated bundle. They let you inspect the source inventory and individual records behind the charts without relying on counts copied into prose.

For benchmark authors, source registration, data generation, tests, browser QA, and deployment-asset updates are documented in the [contributor maintenance guide](../../../frontend/docs/benchmark-dashboard-maintenance.md).
