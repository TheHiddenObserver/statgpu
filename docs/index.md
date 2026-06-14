# statgpu Documentation Portal

> Language: English  
> Last updated: 2026-04-26  
> This page: Primary documentation entrypoint  
> Switch: [Chinese](USAGE_CN.md)

Language switch:
- Chinese: [cn/usage.md](cn/usage.md)
- English: [en/usage.md](en/usage.md)

Detailed docs are organized in `en/` and `cn/`.

## 1) Getting Started

- [Quickstart](en/getting-started/quickstart.md)
- [Device and GPU Memory](en/guides/device-and-memory.md)
- [PyTorch Backend](en/guides/pytorch-backend.md)
- [Inference Modes (Lasso)](en/guides/inference-modes.md)
- [Distribution API (GPU Native + Explicit Fallback)](en/guides/distribution-api.md)
- [Multiple Testing: Adjust & Combine P-values (BH/BY/Holm/Bonferroni/Hochberg + Fisher/Cauchy/Stouffer)](en/guides/multiple-testing-combine-pvalues.md)
- [GLM + Penalty Module](en/models/generalized-linear-model.md) — 7 families × 10 penalties × 3 backends
- [Solver-Penalty Matrix](en/guides/solver-penalty-matrix.md) — solver dispatch and penalty routing
- [Cross-Validation Guide](en/guides/cross-validation.md) — PenalizedGLM_CV, LassoCV, RidgeCV
- [Changelog](en/changelog.md)

Install note:
- Choose CuPy wheel by CUDA major version:
  - CUDA 11.x -> `cupy-cuda11x`
  - CUDA 12.x -> `cupy-cuda12x`
- PyTorch backend (alternative GPU option):
  - PyTorch 2.0+ -> `pip install statgpu[torch]`

## 2) Model Docs

- [Models Overview](en/models/README.md)
- [GeneralizedLinearModel and Penalized GLM](en/models/generalized-linear-model.md)
- [PoissonRegression](en/models/poisson-regression.md)
- [Knockoff Feature Selection](en/models/knockoff.md)
- [Ordered Generalized Linear Models (Logit/Probit)](en/models/ordered.md)
- [Nonparametric Methods](en/models/nonparametric.md)

Implemented estimators:
- `LinearRegression`
- `GeneralizedLinearModel`
- `PoissonRegression`
- `PenalizedLinearRegression`
- `PenalizedLogisticRegression`
- `PenalizedPoissonRegression`
- `Ridge` ✅ (Torch backend)
- `Lasso` ✅ (Torch backend)
- `ElasticNet`
- `LassoCV`
- `LogisticRegression` ✅ (Torch backend)
- `CoxPH` ✅ (Torch backend)
  - `cov_type=nonrobust/hc0/hc1/cluster` (cluster is CPU path)
  - `ties=breslow/efron` (Efron with numerical stability clipping)
  - C-index, baseline hazard, AIC/BIC
  - **Performance**: Torch GPU 15.44x speedup on n=5000, p=20 (vs statsmodels)
  - See `results/coxph_benchmark_report_2026-04-20.md` for comprehensive benchmark
- `OrderedLogitRegression` / `OrderedProbitRegression` ✅ (3 backends)
  - Ordered response models with cumulative logit/probit link
  - Cross-backend precision fix (2026-04-26): coef diff < 1e-2 across backends

Exported CV classes:
- `RidgeCV` ✅ (Full implementation with GPU acceleration)
- `LogisticRegressionCV` ✅ (Full implementation with GPU acceleration)
- `CoxPHCV` (Skeleton, pending full CV training/search implementation)

Implemented feature selection:
- `knockoff_filter`
- `fixed_x_knockoff_filter`
- `model_x_knockoff_filter`
- `KnockoffSelector`
- `FixedXKnockoffSelector`

Inference highlights:
- `LinearRegression`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU)
- `Ridge`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU) ✅ (Torch backend)
- `Lasso`: `cpu_ols_inference/gpu_ols_inference/bootstrap` ✅ (Torch backend)
- `LogisticRegression`: `cov_type=nonrobust/hc0/hc1/hc2/hc3/hac` (CPU+GPU) ✅ (Torch backend)
- Multiple-testing utilities: `statgpu.adjust_pvalues` / `statgpu.multipletests` (`bh/by/holm/bonferroni/hochberg`)
- Global p-value combination: `statgpu.combine_pvalues` (`fisher/cauchy/stouffer`)
- Ordered response models: `OrderedLogitRegression` / `OrderedProbitRegression` (CPU/CuPy/Torch)
- Unified resampling engine: `statgpu.bootstrap_statistic` / `statgpu.permutation_test`

## 3) Benchmarks and Validation

- [Benchmark Index](en/guides/benchmarks.md)

Primary scripts:
- `dev/benchmarks/_bench_inference_timing.py` (multiple-testing, p=100-10k)
- `dev/benchmarks/_bench_inference_timing_large.py` (multiple-testing, p=50k-1M)
- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
- `dev/benchmarks/benchmark_all_methods_large_scale.py`
- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`

Latest nonparametric artifacts:
- Fair-kernel parity run `20260415_103036` (statsmodels parity in diagonal metric mode)
- Local-linear optimization run `20260415_120903` (~4.8-5.4x CPU and ~115-116x GPU speedups in multidim local-linear)

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
  - `docs/en/guides/benchmarks.md`
  - `docs/en/changelog.md`
