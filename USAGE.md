# statgpu Documentation Portal

> Language: English  
> Last updated: 2026-04-02  
> This page: Primary documentation entrypoint  
> Switch: [中文](USAGE_CN.md)

Language switch:
- Chinese: [USAGE_CN.md](USAGE_CN.md)
- Legacy alias: [USAGE_EN.md](USAGE_EN.md)

`USAGE.md` is the primary (English) documentation entrypoint. Detailed docs are organized in `docs/en/`.

## 1) Getting Started

- [Quickstart](docs/en/getting-started/quickstart.md)
- [Device and GPU Memory](docs/en/guides/device-and-memory.md)
- [Inference Modes (Lasso)](docs/en/guides/inference-modes.md)
- [Changelog](docs/en/changelog.md)

Install note:
- Choose CuPy wheel by CUDA major version:
  - CUDA 11.x -> `cupy-cuda11x`
  - CUDA 12.x -> `cupy-cuda12x`

## 2) Model Docs

- [Models Overview](docs/en/models/README.md)

Implemented estimators:
- `LinearRegression`
- `Ridge`
- `Lasso`
- `LogisticRegression`
- `CoxPH`

Inference highlights:
- `LinearRegression`: `cov_type=nonrobust/hc0/hc1` (CPU+GPU)
- `Lasso`: `cpu_ols_inference/gpu_ols_inference/bootstrap`
- `LogisticRegression`: `cov_type=nonrobust/hc0/hc1` (CPU+GPU)
- Multiple-testing utilities: `statgpu.adjust_pvalues` / `statgpu.multipletests` (`bh/by/holm/bonferroni`)
- Unified resampling engine: `statgpu.bootstrap_statistic` / `statgpu.permutation_test`

## 3) Benchmarks and Validation

- [Benchmark Index](docs/en/benchmarks.md)

Primary scripts:
- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
- `dev/benchmarks/benchmark_all_methods_large_scale.py`

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
