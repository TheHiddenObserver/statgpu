# statgpu Documentation Portal (English)

> Language: English
>
> Last updated: 2026-07-12
>
> Switch: [Chinese](../cn/usage.md)

Primary English entrypoint. See also: [Documentation Index](../index.md)

## 1) Getting Started

- [Quickstart](getting-started/quickstart.md)
- [Device and GPU Memory](guides/device-and-memory.md)
- [Inference Modes (Lasso)](guides/inference-modes.md)
- [Distribution API (GPU Native + Explicit Fallback)](guides/distribution-api.md)
- [Global P-value Combination (Fisher/Cauchy/ACAT)](guides/multiple-testing-combine-pvalues.md)
- [Changelog](changelog.md)

Install note:
- Choose CuPy wheel by CUDA major version:
  - CUDA 11.x -> `cupy-cuda11x`
  - CUDA 12.x -> `cupy-cuda12x`

## 2) Model Docs

- [Models Overview](models/README.md)
- [Knockoff Feature Selection](models/knockoff.md)
- [Nonparametric Methods](models/nonparametric.md)

Implemented estimators:
- `LinearRegression`
- `Ridge`
- `Lasso`
- `LassoCV`
- `LogisticRegression`
- `CoxPH`
- `CoxPHCV`
- `PenalizedCoxPHModel`

Implemented cross-validation estimators include:
- `RidgeCV`
- `LogisticRegressionCV`
- `CoxPHCV`

`CoxPHCV` searches a scalar L2 grid and refits the selected model. Its Cox
interface supports `start`, `strata`, subject-preserving folds through
`subject_id`, and Breslow/Efron/Exact ties on NumPy, CuPy, and Torch.

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
- `CoxPH`: `cov_type=nonrobust/hc0/hc1/cluster` on NumPy/CuPy/Torch;
  Exact ties support `nonrobust` only
- `PenalizedCoxPHModel`: estimation-only; `compute_inference=True` raises
  `NotImplementedError`
- Multiple-testing utilities: `statgpu.adjust_pvalues` / `statgpu.multipletests` (`bh/by/holm/bonferroni`)
- Global p-value combination: `statgpu.combine_pvalues` (`fisher/cauchy/acat`)
- Unified resampling engine: `statgpu.bootstrap_statistic` / `statgpu.permutation_test`

## 3) Benchmarks and Validation

- [Benchmark Index](guides/benchmarks.md)

Primary scripts:
- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_gpu_memory_cleanup.py`
- `dev/benchmarks/benchmark_all_methods_large_scale.py`
- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`

Latest nonparametric artifacts:
- Fair-kernel parity run `20260415_103036` (statsmodels parity in diagonal metric mode)
- Local-linear optimization run `20260415_120903` (~4.8-5.4x CPU and ~115-116x GPU speedups in multidim local-linear)

Latest tri-backend covariance artifact:
- `results/remote_covariance_full_compare_2026-04-10.json` (`statsmodels` / `statgpu CPU` / `statgpu GPU`, `hc2/hc3/hac`)

Latest survival artifacts:

- `results/survival_completion_2026-07-12.json`
- `results/survival_completion_full_2026-07-12.json`

They cover Breslow/Efron/Exact, delayed-entry and stratified start-stop fits,
inference, baselines, predictions, and CoxPHCV across NumPy, CuPy, and Torch.
On the recorded RTX 5880 Ada float64 runs, the quick delayed-entry case measured
0.647x CuPy and 0.959x Torch relative to NumPy; the full delayed-entry case
measured 1.044x and 1.374x. The full stratified start-stop case measured only
0.241x and 0.411x, and the Exact/heavy-tie target cases were also slower on GPU.
These artifacts do not establish a general crossover threshold.

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
  - `models/*.md`
  - `guides/benchmarks.md`
  - `changelog.md`
