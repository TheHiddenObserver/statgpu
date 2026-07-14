# statgpu Documentation Portal

> Language: English  
> Last updated: 2026-07-12
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
- [Cox Proportional Hazards](en/models/coxph.md) — CoxPH, CoxPHCV, and penalized Cox
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
- [Cox Proportional Hazards](en/models/coxph.md)

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
  - `ties=breslow/efron/exact`; Exact inference currently supports `cov_type=nonrobust` only
  - Right-censored and delayed-entry data, `(start, stop]` counting-process rows, shared coefficients across strata, and `Surv(start, stop, event)` formulas
  - `cov_type=nonrobust/hc0/hc1/cluster`, C-index, stratum-specific baseline hazard/survival, AIC/BIC
  - NumPy, CuPy, and Torch-CUDA implementations; performance is workload-dependent rather than guaranteed
- `PenalizedCoxPHModel`
  - Validated for L1, L2, ElasticNet, SCAD, and MCP with Breslow/Efron ties
  - No intercept; estimation-only (`compute_inference=True` raises `NotImplementedError`)
- `OrderedLogitRegression` / `OrderedProbitRegression` ✅ (3 backends)
  - Ordered response models with cumulative logit/probit link
  - Cross-backend precision fix (2026-04-26): coef diff < 1e-2 across backends

Exported CV classes:
- `RidgeCV` ✅ (Full implementation with GPU acceleration)
- `LogisticRegressionCV` ✅ (Full implementation with GPU acceleration)
- `CoxPHCV` ✅ (L2 penalty selection with Breslow/Efron/Exact held-out partial likelihood)
  - Propagates delayed entry/start-stop, strata, and subject IDs; repeated rows from one subject stay in the same fold
  - Reports per-candidate convergence/failure diagnostics and refits the selected model
  - Quick/full remote validation selected the same penalty across NumPy, CuPy, and Torch-CUDA

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
- `CoxPH`: model-based and robust inference; Exact currently uses model-based covariance only
- `PenalizedCoxPHModel`: estimation-only; use `CoxPH(penalty=...)` when L2 Cox inference is required
- Unified resampling engine: `statgpu.bootstrap_statistic` / `statgpu.permutation_test`

## 3) Benchmarks and Validation

- [Benchmark Index](en/guides/benchmarks.md)

Primary scripts:
- `dev/benchmarks/benchmark_survival_completion.py` (Cox Phase-1 precision, convergence, and synchronized NumPy/CuPy/Torch timing)
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

Latest survival artifacts:
- `results/survival_completion_2026-07-12.json` (quick)
- `results/survival_completion_full_2026-07-12.json` (full; includes workload-specific GPU/CPU ratios and limitations)

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
