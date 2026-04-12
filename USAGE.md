# statgpu Documentation Portal

> Language: English  
> Last updated: 2026-04-12  
> This page: Primary documentation entrypoint  
> Switch: [Chinese](USAGE_CN.md)

Language switch:
- Chinese: [USAGE_CN.md](USAGE_CN.md)
- Legacy alias: [USAGE_EN.md](USAGE_EN.md)

`USAGE.md` is the primary (English) documentation entrypoint. Detailed docs are organized in `docs/en/`.

## 1) Getting Started

- [Quickstart](docs/en/getting-started/quickstart.md)
- [Device and GPU Memory](docs/en/guides/device-and-memory.md)
- [Inference Modes (Lasso)](docs/en/guides/inference-modes.md)
- [Distribution API (GPU Native + Explicit Fallback)](docs/en/guides/distribution-api.md)
- [Global P-value Combination (Fisher/Cauchy/ACAT)](docs/en/guides/multiple-testing-combine-pvalues.md)
- [Changelog](docs/en/changelog.md)

Install note:
- Choose CuPy wheel by CUDA major version:
  - CUDA 11.x -> `cupy-cuda11x`
  - CUDA 12.x -> `cupy-cuda12x`

## 2) Model Docs

- [Models Overview](docs/en/models/README.md)
- [Knockoff Feature Selection](docs/en/models/knockoff.md)

Implemented estimators:
- `LinearRegression`
- `Ridge`
- `Lasso`
- `LassoCV`
- `LogisticRegression`
- `CoxPH`

Exported CV classes currently in skeleton state:
- `RidgeCV`
- `LogisticRegressionCV`
- `CoxPHCV`
- Current behavior: `fit()` raises `NotImplementedError`.

Implemented feature selection:
- `knockoff_filter`
- `fixed_x_knockoff_filter`
- `model_x_knockoff_filter`
- `KnockoffSelector`
- `FixedXKnockoffSelector`

Inference highlights:
- `LinearRegression`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU)
- `Ridge`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU)
- `Lasso`: `cpu_ols_inference/gpu_ols_inference/bootstrap`
- `LogisticRegression`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU)
- Multiple-testing utilities: `statgpu.adjust_pvalues` / `statgpu.multipletests` (`bh/by/holm/bonferroni`)
- Global p-value combination: `statgpu.combine_pvalues` (`fisher/cauchy/acat`)
- Unified resampling engine: `statgpu.bootstrap_statistic` / `statgpu.permutation_test`

## 3) Benchmarks and Validation

- [Benchmark Index](docs/en/benchmarks.md)

Primary scripts:
- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
- `dev/benchmarks/benchmark_all_methods_large_scale.py`

Latest tri-backend covariance artifact:
- `results/remote_covariance_full_compare_2026-04-10.json` (`statsmodels` / `statgpu CPU` / `statgpu GPU`, `hc2/hc3/hac`)

Recommended large-scale command:

```bash
python dev/benchmarks/benchmark_all_methods_large_scale.py \
  --devices cpu,cuda \
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out results/bench_all_large_results.json
```

## 4) Collaboration Notes

- For performance reports, include: device info, data shape, `repeats/warmup`, and whether inference is timed.
- If you add new features, also update:
  - `docs/en/models/*.md`
  - `docs/en/benchmarks.md`
  - `docs/en/changelog.md`
