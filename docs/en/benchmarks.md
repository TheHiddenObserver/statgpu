# Benchmarks: interpretation and reproduction

> Language: English<br>
> Last updated: 2026-10-09<br>
> Switch: [Chinese](../cn/benchmarks.md)

## Start with a comparable workload

Use the [interactive dashboard](../assets/benchmarks/index.html) to find recorded runs and the [dashboard guide](guides/statgpu_benchmark_dashboard.md) to interpret their filters and metrics. A benchmark describes a particular workload and environment, not a universal ranking of models or backends.

Before comparing timings, match:

- The statistical task, input rows and columns, preprocessing, sample weights, and objective normalization. Align penalty scales across packages; equal parameter names need not mean equal objectives.
- The method variant, solver, convergence tolerance, dtype, and numerical accuracy. For Cox models, also match the tie method. For CV, match splits, candidate grids, scoring, and whether final refit is included.
- The measured operation: fit only, fit plus inference, full CV, prediction, or validation. A correctness-only record has no implied timing or speedup.
- The hardware and software environment, warmup, repeats, synchronization, and data-transfer policy. Keep results from different environments separate.

A speedup is `reference time / measured time`: values above one are faster, below one slower. Check the reference and the accuracy result before interpreting a ratio. Small numerical error on one workload does not establish accuracy for all settings.

## Measure the workflow you will use

Measure representative sample **and feature** counts rather than choosing a backend from sample count alone. Include the dtype, sparsity, solver settings, CV grid, and inference options your application needs. Small problems may be dominated by startup or data movement; different algorithms can have different crossover points on the same hardware.

Warm up the selected path, repeat measurements, and report dispersion as well as a central time. Synchronize the concrete GPU device that executed the work before reading elapsed time. State whether conversion and host/device transfer are included; fit-only and end-to-end measurements answer different questions. Record package versions, CPU/GPU identity, source revision, shapes, seeds, and numerical differences so the result is interpretable later. See [device and memory guidance](guides/device-and-memory.md) for selecting an explicit backend.

## Advanced appendix: benchmark runners

The runners below are for readers reproducing or designing measurements from a source checkout. Optional Python/R packages and GPU hardware may be required. Inspect each runner's options and timing boundaries before using its output; the index does not certify every historical script as a current benchmark protocol. Dashboard source registration, builds, tests, and publication are covered in the [contributor maintenance guide](../../frontend/docs/benchmark-dashboard-maintenance.md).

Older remote-phase output lists are preserved in the [historical artifact index](../../dev/guides/historical-benchmark-artifacts.md).

### Inference

- [dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py](../../dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py)
  - Benchmarks canonical `inference_method="post_selection_ols"` end to end on NumPy CPU and CuPy CUDA.
  - Reports complete fit+inference timing plus CPU/CuPy parity for penalized coefficients, active-refit parameters, SE/statistic/p-value/CI, active-set identity, and inference backend/device provenance.
  - It is not an inference-only speedup benchmark; hardware is selected with `device`, not with `inference_method`.

### Nonparametric

- [dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py](../../dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py)
  - Compares `statgpu` vs `statsmodels.nonparametric.kernel_regression.KernelReg`
  - Supports `regression=nw/local_linear` and multidimensional settings
  - Supports fair parity mode via `--kernel-metric diagonal`
  - Reports `statgpu CPU/GPU` and `statsmodels` accuracy/runtime comparisons
  - Outputs precision and runtime JSON under `results/`

- [dev/benchmarks/benchmark_kde_vs_scipy.py](../../dev/benchmarks/benchmark_kde_vs_scipy.py)
  - Compares `statgpu` vs `scipy.stats.gaussian_kde`
  - Reports `statgpu CPU/GPU` and SciPy accuracy/runtime comparisons

- [dev/benchmarks/benchmark_nonparametric_vs_r.py](../../dev/benchmarks/benchmark_nonparametric_vs_r.py)
  - Compares `statgpu` with R `density()` / `ksmooth()` / `KernSmooth::locpoly()`
  - Supports `--statgpu-backend numpy/cupy`
  - Supports `--ci-method normal/bootstrap`
  - Reports `statgpu CPU/GPU`, R, and KDE CI vs SciPy comparisons

### Unsupervised Learning

- Detailed model docs: [docs/en/unsupervised/](unsupervised/README.md)
- [dev/benchmarks/benchmark_unsupervised.py](../../dev/benchmarks/benchmark_unsupervised.py)
  - Compares `PCA` and `KMeans` across `statgpu` CPU/CuPy/Torch and sklearn when available.
  - Outputs timing and numerical differences to JSON.
- [dev/benchmarks/benchmark_unsupervised_phase2.py](../../dev/benchmarks/benchmark_unsupervised_phase2.py)
  - Compares `DBSCAN`, `GaussianMixture`, `NMF`, and `AgglomerativeClustering` against available sklearn/SciPy/R baselines.
  - Records optional `umap-learn` and `openTSNE` smoke/runtime comparison baselines.
- [dev/benchmarks/benchmark_unsupervised_dbscan_cython.py](../../dev/benchmarks/benchmark_unsupervised_dbscan_cython.py)
  - Validates the optional statgpu-owned DBSCAN Cython CPU fast path against the exact fallback, sklearn CPU, CuPy, and Torch.
- [dev/benchmarks/benchmark_unsupervised_phase3.py](../../dev/benchmarks/benchmark_unsupervised_phase3.py)
  - Compares `TruncatedSVD`, `MiniBatchKMeans`, `UMAP`, and `TSNE` across statgpu CPU/CuPy/Torch and available external baselines.
  - Records warmup/repeat timings, precision or quality metrics, and skipped optional frameworks.
  - GPU measurements use backend-resident input arrays for the main timing path.
- [dev/benchmarks/benchmark_unsupervised_phase3b.py](../../dev/benchmarks/benchmark_unsupervised_phase3b.py)
  - Compares `GaussianMixture` covariance variants and `AgglomerativeClustering` linkage variants.
  - Covers statgpu CPU/CuPy/Torch where supported plus sklearn, SciPy, and R `cluster::agnes` where available.
  - Outputs JSON and Markdown summaries under `results/`.
- [dev/benchmarks/benchmark_unsupervised_phase3c.py](../../dev/benchmarks/benchmark_unsupervised_phase3c.py)
  - Compares `IncrementalPCA` and `MiniBatchNMF` across statgpu CPU/CuPy/Torch and sklearn when available.
  - Records reconstruction quality, explained variance metrics, warmup/repeat timings, and skipped optional frameworks.

DBSCAN CPU Cython note:
- The optional `_dbscan_cpu` extension is a statgpu-owned implementation, not a sklearn wrapper.
- Compact dense CPU cases use the extension when it is built and selected; fallback remains available for variable-density, sparse/all-noise, or no-compiler environments.

### Multiple-testing and Global P-value Combination

- [dev/benchmarks/benchmark_inference_backends.py](../../dev/benchmarks/benchmark_inference_backends.py)
  - Includes `combine_pvalues` benchmarks for `fisher/cauchy/acat`
  - Includes consistency checks:
    - Fisher vs `scipy.stats.combine_pvalues`
    - Cauchy vs independent NumPy reference
    - statgpu NumPy vs CuPy
  - Outputs structured JSON under `results/`

### GPU Memory

- [dev/benchmarks/benchmark_gpu_memory_cleanup.py](../../dev/benchmarks/benchmark_gpu_memory_cleanup.py)
  - Compares `gpu_memory_cleanup=False/True`

### Large-scale All-method Runtime

- [dev/benchmarks/benchmark_all_methods_large_scale.py](../../dev/benchmarks/benchmark_all_methods_large_scale.py)
  - Covers `LinearRegression / Ridge / Lasso / LogisticRegression / CoxPH`
  - Separates data construction from fit timing
  - Supports CPU/GPU, warmup, repeats, and JSON output

Recommended command:

```bash
python dev/benchmarks/benchmark_all_methods_large_scale.py \
  --devices cpu,cuda \
  --include-external \
  --repeats 3 \
  --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 \
  --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out results/bench_all_large_results.json
```

To include inference-statistics computation time in measurements, add:

```bash
--compute-inference
```

### External Framework Comparison (accuracy + runtime)

- [dev/benchmarks/benchmark_external_frameworks.py](../../dev/benchmarks/benchmark_external_frameworks.py)
  - Primary comparison: `statsmodels`, `sklearn`
  - Optional comparison: `R` (if `Rscript` and required packages are available)
  - Outputs: `fit_ms` + coefficient/inference differences (+ JSON option)

Recommended command (statsmodels + sklearn):

```bash
python dev/benchmarks/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow \
  --skip-r
```

Recommended command (including R):

```bash
python dev/benchmarks/benchmark_external_frameworks.py \
  --n 1200 --p 10 \
  --cox-ties breslow
```

For a comparable measurement:
- Explicitly use the same feature set across frameworks (avoid accidental `y ~ .` leakage)
- Explicitly fix Cox tie method (`breslow` or `efron`)
- Explicitly log regularization and convergence settings (`alpha/C/max_iter/tol`)

### Cox Covariance Benchmark

- [dev/benchmarks/benchmark_cox_cluster.py](../../dev/benchmarks/benchmark_cox_cluster.py)
  - Compares `CoxPH cov_type=nonrobust/hc1/cluster` on runtime and numerical differences
  - Covers `statgpu CPU/GPU` and `statsmodels.PHReg` when available

### Elastic Net Benchmarks

These April 18, 2026 experiments are historical examples, not expected speedups for current releases or a rule for selecting a backend.

- [Small-workload runner](../../dev/benchmarks/benchmark_elasticnet_sklearn.py) and [recorded JSON](../../results/benchmark_elasticnet_sklearn_2026-04-18.json): six synthetic datasets with different sample/feature counts, sparsity, and noise. The JSON contains coefficient vectors and per-row differences from sklearn. Read those errors for the particular case; they do not guarantee accuracy for another objective, tolerance, or dataset.
- [Large-workload runner](../../dev/benchmarks/benchmark_large_scale.py) and [recorded JSON](../../results/large_scale/benchmark_elasticnet_large_scale_2026-04-18.json): six dense synthetic Gaussian-design workloads, `n=10,000–100,000`, `p=100 or 500`, ten nonzero generating coefficients, seed 42, and noise standard deviation 0.5. The associated runner generates NumPy float64 inputs and uses `alpha=1.0`, `l1_ratio=0.5`, `max_iter=5000`, and `tol=1e-8` for sklearn and statgpu. These are runner settings, not a complete record of the executed environment.
- [R glmnet comparison](../../results/benchmark_full/benchmark_glmnet_all.json): interpret alongside the [paired statgpu output](../../results/benchmark_full/benchmark_statgpu_all.json). Unaligned objective normalization, penalty scale, standardization, or stopping rules can change both coefficients and work performed. A coefficient-norm difference does not establish equivalent optimization problems.

For example, the large-workload JSON reports `615.59 ms / 141.05 ms ≈ 4.36×` versus sklearn at `n=100,000, p=500` for a row labelled `statgpu_gpu_torch`. At `n=10,000, p=100`, that same label has a ratio below one. **The label is not verified Torch provenance:** the associated runner constructs that row with `device='cuda'`, and its timing block has no explicit GPU synchronization. The file does not record CPU/GPU model, package versions, source commit, or repeated-timing dispersion. It stores coefficient norms, not full vectors for a componentwise accuracy check. These limitations prevent attributing the ratio to a known Torch/hardware configuration or treating it as a current performance guarantee.

Choose a backend by measuring your own model, feature count, dtype, convergence settings, and transfer policy as described above. There is no sample-count-only crossover threshold in these results. See the [historical artifact index](../../dev/guides/historical-benchmark-artifacts.md) for the older experiment records.

---

### Knockoff Feature Selection

- [dev/benchmarks/benchmark_knockoff_fixedx.py](../../dev/benchmarks/benchmark_knockoff_fixedx.py)
  - Runs fixed-X knockoff at multiple `q` values and reports selected-set diagnostics.

- [dev/benchmarks/benchmark_knockoff_vs_baselines.py](../../dev/benchmarks/benchmark_knockoff_vs_baselines.py)
  - Compares fixed-X/model-X knockoff with baseline selectors:
    - marginal-correlation top-k
    - statgpu lasso top-k
    - sklearn `LassoCV` (if installed)
    - `knockpy` Gaussian knockoff + lasso statistic (if installed)
  - Supports configurable knockoff statistic via `config.knockoff_method` (current default: `ols_coef_diff`).
  - Model-X path uses covariance-shrinkage plus multi-draw W aggregation (draw count depends on statistic).
  - Run output includes model-X calibration metadata (`modelx_n_draws`, `modelx_covariance_shrinkage`).
  - Environment and method blocks include optional availability flags and pairwise deltas for `knockpy` when present.
  - Reports precision/recall/FDP/F1/Jaccard and timing in one JSON file.
  - Additional environment controls:
    - `STATGPU_KNOCKOFF_COMPAT_MODE`: `statgpu` or `knockpy`
    - `STATGPU_KNOCKOFF_LASSO_CV_IMPL`: `auto` / `statgpu` / `sklearn`

- [dev/benchmarks/benchmark_knockoff_same_xk_parity.py](../../dev/benchmarks/benchmark_knockoff_same_xk_parity.py)
  - Compares `statgpu` and `knockpy` using the exact same `Xk` generated once by knockpy.
  - Key outputs: `W` correlation, `W` error, threshold difference, and selected-set Jaccard.
  - Useful for correctness diagnostics when sampler randomness must be held constant.
