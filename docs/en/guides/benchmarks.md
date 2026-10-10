# Benchmark dashboard and results

> Language: English<br>
> Last updated: 2026-10-09<br>
> Switch: [Chinese](../../cn/guides/benchmarks.md)

## Explore the results

- [Open the interactive dashboard](../../assets/benchmarks/index.html)
- [Understand filters, charts, metrics, and provenance](statgpu_benchmark_dashboard.md)
- [Compare workloads and reproduce measurements](../benchmarks.md)

Start by selecting an environment and model family, then narrow the method, solver, scale, and backend. Use **Metric scope** to distinguish Fit, CV, Inference, Prediction, and Selection. Only metric groups present in the selected records appear in the detail panels; a missing panel is not a zero result or a claim that the library lacks that capability.

The dashboard includes cross-validation records. The **Cross-validation** panel shows the available CV evaluation, final-refit and total timings, selected parameters, scores, convergence information, and reasons for unsuccessful or unavailable runs. Read the status and scoring direction before comparing values.

## Coverage and data sources

The dashboard uses benchmark sources dated **2026-06-01 or later**. For current coverage and counts, use the deployed [source inventory](../../assets/benchmarks/data/source_inventory.json), [parse report](../../assets/benchmarks/data/parse_report.json), and [normalized results](../../assets/benchmarks/data/benchmark_data.json). Source counts, run counts, and model-registry entries are different quantities; filtered views can be smaller than the whole bundle.

Available records span GLM and penalized GLM, linear models, robust and quantile regression, survival, unsupervised learning, ordered models, nonparametric methods, panel models, covariance estimation, and ANOVA. Coverage varies by method, metric, scale, and backend. The Feature Selection category has no eligible structured source in this bundle. April 2026 ElasticNet, LassoCV, comprehensive-validation, Cox package-comparison, and knockoff results are outside the dashboard's date policy. A rounded distribution-report summary without raw timing and precision records is also excluded.

## Read comparisons carefully

- Match the environment, workload, objective, solver, dtype, and timing scope. Fit-plus-inference and complete CV timings are not fit-only timings.
- A speedup above one is faster than its stated reference; below one is slower. Runner-reported and computed ratios have different provenance.
- Correctness-only records do not provide timing or speedup. Missing values are not zero.
- A historical measurement describes its recorded source and environment; it does not automatically describe the current release or your hardware.

For example, the [post-selection OLS inference benchmark](../../../dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py) measures the complete fit-plus-inference operation on NumPy CPU and CuPy CUDA, including coefficient and inference comparisons. It is not an inference-only speedup measurement; `device` selects the hardware, while `inference_method="post_selection_ols"` selects the statistical procedure.

Contributor workflows are in the [dashboard maintenance guide](../../../frontend/docs/benchmark-dashboard-maintenance.md).
