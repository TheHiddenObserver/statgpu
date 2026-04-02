# Benchmark Index

> Language: English  
> Last updated: 2026-04-02  
> This page: Benchmark index  
> Switch: [中文](../benchmarks.md)

Language switch: [中文](../benchmarks.md)

## Inference

- `examples/benchmark_lasso_inference_gpu_vs_cpu.py`
  - Compares `cpu_ols_inference` vs `gpu_ols_inference`

## GPU Memory

- `examples/benchmark_gpu_memory_cleanup.py`
  - Compares `gpu_memory_cleanup=False/True`

## Large-scale All-method Runtime

- `examples/benchmark_all_methods_large_scale.py`
  - Covers `LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`
  - Separates data construction from fit timing
  - Supports CPU/GPU, warmup, repeats, and JSON output

Recommended command:

```bash
python examples/benchmark_all_methods_large_scale.py \
  --devices cpu,cuda \
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out examples/bench_all_large_results.json
```

## External Framework Comparison (accuracy + runtime)

- `examples/benchmark_external_frameworks.py`
  - Primary comparison: `statsmodels`, `sklearn`
  - Optional comparison: `R` (if `Rscript` and required packages are available)
  - Outputs: `fit_ms` + coefficient/inference differences (+ JSON option)

Recommended command (statsmodels + sklearn):

```bash
python examples/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow \
  --skip-r
```

Recommended command (including R):

```bash
python examples/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow
```

Comparison gate recommendations:
- Explicitly use the same feature set across frameworks (avoid accidental `y ~ .` leakage)
- Explicitly fix Cox tie method (`breslow` or `efron`)
- Explicitly log regularization and convergence settings (`alpha/C/max_iter/tol`)
