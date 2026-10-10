# Model validation reference for contributors

This reference catalogs developer tests and comparison tools moved out of
learner-facing model guides. The presence of a script is not execution evidence.
A result applies only to its recorded source, environment, parameters and test
selection; GPU skips do not establish physical GPU execution, and benchmark
results are not universal performance or statistical guarantees.

The model guides retain the settings that users need when comparing results:
objective normalization, feature scaling, intercepts, weights, covariance and
reference-distribution choices. Run the relevant checks after changes, and record
fresh results in a review or evidence artifact rather than in a learner tutorial.

## Linear regression

[`test_external_consistency.py`](../tests/test_external_consistency.py) includes
OLS estimation, inference and robust covariance comparisons with statsmodels:

- `test_linear_estimation_and_inference_match_statsmodels`
- `test_linear_robust_covariance_matches_statsmodels`
- `test_linear_robust_covariance_gpu_matches_statsmodels`
- `test_linear_hac_covariance_matches_statsmodels`

Align the observation rows, design/intercept, covariance convention, lag count,
small-sample correction and reference distribution. Use the corresponding WLS
reference and identical weights for weighted comparisons.

## Ridge

[`test_ridge_weighted_consistency.py`](../tests/test_ridge_weighted_consistency.py)
covers weighted exact/FISTA fits, formula-row alignment, inference and RidgeCV
weight-rescaling invariance. Internal comparisons use the average-loss closed
form and generic penalized-linear estimator. sklearn's summed-loss Ridge requires
`sklearn_alpha = n * statgpu_alpha` without weights and
`sklearn_alpha = sum(sample_weight) * statgpu_alpha` with weights.

Covariance and interval checks must align the penalty, weights, degrees of
freedom, covariance choice and reference distribution.

## Lasso and Elastic Net

Regression and migration checks:

- [`test_lasso_debiased_inference.py`](../tests/test_lasso_debiased_inference.py)
- [`test_nodewise_alpha_inference_contract.py`](../tests/test_nodewise_alpha_inference_contract.py)
- [`test_post_selection_ols_inference_api.py`](../tests/test_post_selection_ols_inference_api.py)
- [`test_penalized_solver_api_cleanup.py`](../tests/test_penalized_solver_api_cleanup.py)

Focused comparisons and physical-backend runners:

- [`validate_post_selection_ols_gpu.py`](../benchmarks/validate_post_selection_ols_gpu.py)
- [`validate_gaussian_residual_bootstrap_gpu.py`](../benchmarks/validate_gaussian_residual_bootstrap_gpu.py)
- [`benchmark_lasso_inference_gpu_vs_cpu.py`](../benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py): post-selection OLS CPU/CuPy fit-plus-inference parity and timing.
- [`benchmark_lasso_cpu_gpu_tol.py`](../benchmarks/benchmark_lasso_cpu_gpu_tol.py)
- [`compare_lasso_kkt_stopping.py`](../comparisons/compare_lasso_kkt_stopping.py)

Keep solver/API migration, objective accuracy and statistical inference targets
separate. Comparing two implementations does not establish selective coverage,
calibration after CV, or correctness on unexecuted backends.

## Knockoff

Comparison tools:

- [`benchmark_knockoff_fixedx.py`](../benchmarks/benchmark_knockoff_fixedx.py)
- [`benchmark_knockoff_vs_baselines.py`](../benchmarks/benchmark_knockoff_vs_baselines.py)
- [`benchmark_knockoff_same_xk_parity.py`](../benchmarks/benchmark_knockoff_same_xk_parity.py)

Historical benchmark artifacts use `results/benchmark_knockoff_*.json`.
Distinguish same-X/Xk statistic parity from validity of the knockoff construction,
threshold implementation and empirical FDR calibration. A single tutorial run
can demonstrate inputs and outputs; it cannot establish FDR control.

## Nonparametric models

- [`benchmark_kde_vs_scipy.py`](../benchmarks/benchmark_kde_vs_scipy.py)
- [`benchmark_kernel_regression_vs_statsmodels.py`](../benchmarks/benchmark_kernel_regression_vs_statsmodels.py)
- [`benchmark_nonparametric_vs_r.py`](../benchmarks/benchmark_nonparametric_vs_r.py)
- [`benchmark_nonparametric_comparison_suite.py`](../benchmarks/benchmark_nonparametric_comparison_suite.py)

Align KDE data orientation, weights and covariance bandwidth factor. For kernel
regression, align kernel, regression mode, metric and absolute per-feature
bandwidths. The public scalar bandwidth factor is not interchangeable with a
vector of absolute widths. Record which selectors and host/device paths ran.

## Torch backend

Torch test and benchmark evidence should record:

- exact commit SHA and, for uncommitted changes, a content fingerprint;
- Python, Torch, CUDA, and driver versions;
- GPU model and the concrete device used;
- synchronized timing methodology, including warmup, repeats, and transfer scope;
- accuracy or statistical parity metrics with aligned objectives and parameters;
- passed, failed, and skipped tests, with physical CUDA runs distinguished from
  CPU runs and skips.

Current and historical benchmark artifacts live under `results/` and
`dev/benchmarks/`. The retained
[Torch backend report](../docs/torch_backend_final_report.md) is a dated evidence
snapshot, not a current support matrix. Its results do not establish support or
performance for a later source revision or a different environment.
