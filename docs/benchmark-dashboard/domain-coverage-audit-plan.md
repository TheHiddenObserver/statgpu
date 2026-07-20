# Benchmark Dashboard Domain Coverage Audit and Expansion Plan

## 1. Purpose and audit rule

This document records a repository-wide audit of five dashboard areas that remain incomplete or insufficiently informative:

- Ordered models;
- Nonparametric methods;
- Covariance estimation;
- Feature selection;
- ANOVA.

The audit distinguishes two materially different situations:

1. **Frontend/parser omission:** an eligible source dated 2026-06-01 or later already contains the required benchmark rows, but the canonical parser or frontend does not expose them. These issues should be fixed immediately without rerunning the benchmark.
2. **Benchmark-data gap:** an implementation or runner exists, but no eligible, committed, reproducible result artifact contains the required measurements. These issues must be recorded as new benchmark work; the frontend must not synthesize or relabel data.

Historical pre-June artifacts remain excluded by the dashboard source-date policy.

The audit covers:

- the PR #78 canonical manifest and source copies;
- current parser implementations and generated-data contracts;
- maintained benchmark runners in the repository;
- result artifacts discoverable in the repository;
- the implementation-hardening work in PR #79 where it affects the validity of a future GPU rerun.

It does not treat uncommitted local files or remote-server-only output as available benchmark data.

## 2. Executive status

| Domain | Current dashboard state | Repo audit result | Required action |
|---|---|---|---|
| Ordered | Ordered Logit/Probit at only `500×5` and `2K×10` | Parser accepts arbitrary scales, but the source and runner contain only two small configurations | New large-scale benchmark required |
| Nonparametric | Nystroem, RBF kernel, B-spline, and GAM are visible, but coverage appears partial | Existing GAM source contains small/medium/large aligned rows; parser exposed only large. Additional KDE/kernel-regression runners exist but have no committed eligible result | Fix GAM parser now; run and register a complete nonparametric source later |
| Covariance | Only EmpiricalCovariance is shown | Current P2 source contains only EmpiricalCovariance. A broader runner covers LedoitWolf and MinCovDet, but its output is not committed | New covariance benchmark required |
| Feature selection | Category has no rows | Knockoff/baseline runner exists, but no eligible June-or-later canonical artifact is committed | New feature-selection benchmark required |
| ANOVA | Five methods and three scales are shown | Parser already exposes the complete June 24 source. Existing points show some large-scale GPU wins, but the grid is too coarse to identify the crossover interval and the runner lacks explicit synchronization | New crossover benchmark required; no current parser repair |

## 3. Immediate frontend repair from existing data

### 3.1 Complete aligned GAM scale coverage

The June 24 source contains aligned pyGAM comparisons at all three scales:

```text
small   = 1K×3
medium  = 10K×5
large   = 100K×10
```

For each scale it contains:

- statgpu NumPy timing;
- statgpu CuPy timing;
- statgpu Torch timing;
- pyGAM timing;
- runner-reported speedup against pyGAM;
- prediction relative difference.

The previous parser selected only:

```text
gam_fixed_large_numpy
gam_fixed_large_cupy
gam_fixed_large_torch
```

This was a genuine parser omission. The canonical parser must replace the old large-only rows with the complete aligned small/medium/large matrix.

Expected result:

```text
3 scales × (3 statgpu backends + 1 pyGAM reference) = 12 GAM rows
```

The old bundle contained four aligned GAM rows, so the repair adds eight normalized runs without changing the source manifest count.

Acceptance checks:

- GAM has scale labels `1K×3`, `10K×5`, and `100K×10`;
- each scale has NumPy, CuPy, Torch, and pyGAM timing;
- each statgpu row has the source-reported pyGAM speedup and validation metric;
- no duplicate large-scale rows remain;
- Focused mode may select the representative large scale, while Full matrix and the scale selector expose all three scales.

### 3.2 No other eligible parser omission found

The audit found no comparable hidden rows for the other four domains:

- the Ordered parser already accepts any matching `n`/`p`, but the source contains only two scales;
- the P2 parser imports every available EmpiricalCovariance, Nystroem, RBF-kernel, and spline row in its source;
- the ANOVA parser imports all five methods, all three scales, and all available NumPy/CuPy/Torch/SciPy rows;
- no feature-selection source is registered.

The remaining work must therefore produce new result artifacts rather than change labels or invent rows.

## 4. Ordered-model benchmark plan

### 4.1 Current limitation

The current source contains only:

```text
500×5
2K×10
```

At these scales the ordered likelihood and inference work is too small to amortize GPU launch and synchronization overhead. The current timings consequently show NumPy faster than CuPy and Torch. This is a valid small-scale result, not evidence that GPU acceleration is never useful.

The existing parser is scale-generic and will ingest larger `ordered_logit_n{n}_p{p}` and `ordered_probit_n{n}_p{p}` keys without a schema change.

### 4.2 Runner work

Extend or replace:

```text
dev/benchmarks/bench_pr74_results.py
```

The current runner hardcodes:

```text
CFG = [(500, 5), (2000, 10)]
```

Recommended crossover matrix:

| Regime | n | p | Ordered categories K | Purpose |
|---|---:|---:|---:|---|
| Existing baseline | 2,000 | 10 | 3 | Preserve current comparison |
| Medium | 10,000 | 20 | 3 and 5 | Initial GPU-overhead crossover |
| Large-n | 50,000 | 20 | 5 | Throughput regime |
| Large-n/high-p | 100,000 | 50 | 5 | Strong GPU candidate |
| Higher-p | 20,000 | 100 | 5 | Hessian/linear-algebra stress |
| More categories | 50,000 | 50 | 10 | Threshold-parameter stress |

Run Ordered Logit and Ordered Probit separately. Record both:

- fit-only timing;
- fit plus inference timing.

Inference can materially change the crossover because Hessian and covariance calculations may dominate.

### 4.3 Timing and correctness contract

- 2 warmups;
- 5 measured fits;
- 3 deterministic data seeds;
- float64;
- backend arrays prepared before fit-only timing;
- explicit CuPy/Torch synchronization before and after each measured fit;
- mean/std/min/max and seed aggregation metadata;
- convergence rate and iteration count;
- log-likelihood agreement;
- coefficient and threshold agreement;
- BSE/Wald agreement when inference is enabled;
- peak GPU memory or an explicit memory-failure record for large cases.

Recommended artifact:

```text
ordered_models_crossover_202607xx.json
```

Exit gate: at least two scales below and two scales above the observed GPU/NumPy crossover, or an explicit conclusion that no crossover occurs within the feasible P100 memory range.

## 5. Nonparametric benchmark plan

### 5.1 What is already available

The current canonical sources provide:

- aligned GAM at three scales after the parser repair;
- Nystroem at three scales, with NumPy/CuPy and sklearn;
- RBF kernel at three scales, with NumPy/CuPy and sklearn;
- B-spline basis at three scales, with NumPy and sklearn.

The current P2 source does not contain a complete backend matrix for every method. This is a source limitation, not a frontend filter bug.

### 5.2 Existing runners without committed eligible output

The repository already contains:

```text
dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py
dev/benchmarks/benchmark_kde_vs_scipy.py
dev/benchmarks/benchmark_nonparametric_vs_r.py
dev/benchmarks/benchmark_nonparametric_comparison_suite.py
dev/tests/benchmark_p2_expanded.py
```

These runners cover or partially cover:

- KDE versus SciPy;
- Nadaraya-Watson and local-linear kernel regression versus statsmodels;
- R comparisons;
- SplineTransformer on three backends;
- RBF and chi-square kernels on three backends;
- Nystroem on three backends;
- KernelPCA on three backends.

No corresponding committed June-or-later complete result artifact was found. The runner code alone is not dashboard data.

### 5.3 Recommended source structure

Create one canonical source with method-specific scale grids rather than forcing one unsafe grid on all algorithms:

| Model family | Suggested scales | External reference |
|---|---|---|
| KDE | `10K×1`, `50K×1`, `100K×1`, with fixed evaluation counts | SciPy |
| Kernel regression NW/local-linear | vary both training n and evaluation m; include 1D and 5D | statsmodels |
| GAM | preserve `1K×3`, `10K×5`, `100K×10` | pyGAM |
| SplineTransformer | `5K×10`, `20K×10`, `50K×10`, `100K×20` | sklearn |
| Nystroem | `10K×20`, `50K×20`, `100K×20`, plus component-count variants | sklearn |
| KernelPCA | memory-safe scales only, with explicit skipped/OOM rows | sklearn where aligned |
| Raw RBF/chi-square kernel | use rectangular `X` versus `Y` or capped n; do not create infeasible full `n×n` matrices | sklearn/reference implementation |

Required metrics:

- fit/transform/predict scope made explicit;
- NumPy/CuPy/Torch timing where supported;
- external timing;
- prediction or transform relative error;
- RMSE against known truth for KDE/kernel regression;
- output shape and finite-value checks;
- peak memory for quadratic methods;
- transfer-inclusive timing recorded separately from fit-only timing.

Recommended artifact:

```text
nonparametric_full_202607xx.json
```

## 6. Covariance benchmark plan

### 6.1 Current limitation

The dashboard currently shows only `EmpiricalCovariance`, because that is the only covariance estimator in the eligible P2 result source. All three P2 scales and available backends are already parsed.

### 6.2 Existing runner without committed output

The repository contains:

```text
dev/tests/benchmark_p2_expanded.py
```

It includes:

- LedoitWolf versus sklearn on NumPy/CuPy/Torch;
- MinCovDet precision versus sklearn;
- multiple scales.

Its expected output:

```text
results/p2_benchmark_expanded.json
```

is not committed. In addition, PR #79 contains correctness and backend-boundary changes for shrinkage covariance, GraphicalLasso, and MinCovDet. A physical GPU benchmark should run against the corrected implementation rather than the older code snapshot.

### 6.3 Recommended estimator matrix

Phase 1:

```text
EmpiricalCovariance
ShrunkCovariance
LedoitWolf
OAS
```

Phase 2, with method-specific smaller grids:

```text
GraphicalLasso
GraphicalLassoCV
MinCovDet
```

Recommended general scales:

```text
10K×50
50K×50
100K×50
20K×200
5K×500
```

GraphicalLasso and MinCovDet require smaller method-specific grids because their complexity is driven strongly by p and repeated optimization/subsampling.

Required metrics:

- NumPy/CuPy/Torch timing where the estimator supports the backend;
- sklearn reference timing;
- covariance and precision-matrix relative error;
- shrinkage coefficient agreement;
- log-likelihood/score agreement where available;
- GraphicalLasso support Jaccard and dual-gap/convergence information;
- MinCovDet location, covariance, support overlap, and support size;
- explicit OOM/unsupported records rather than silently missing rows.

Recommended artifact:

```text
covariance_estimators_202607xx.json
```

## 7. Feature-selection benchmark plan

### 7.1 Current limitation

The category is intentionally empty because no eligible June-or-later source is registered. The old April result must not be reconnected.

### 7.2 Existing runner

The repository contains:

```text
dev/benchmarks/benchmark_knockoff_fixedx.py
dev/benchmarks/benchmark_knockoff_vs_baselines.py
dev/benchmarks/benchmark_knockoff_same_xk_parity.py
```

The baseline-comparison runner already records:

- fixed-X and model-X knockoff;
- marginal-correlation top-k;
- statgpu Lasso top-k;
- sklearn LassoCV when installed;
- knockpy when installed;
- NumPy and optional CuPy;
- precision, recall, FDP, F1, Jaccard, selected count, estimated FDR, and timing.

Its current default configuration is only `n=400`, `p=80`, three seeds. This is useful for correctness but insufficient as a GPU-performance source.

### 7.3 Recommended benchmark matrix

Use separate grids for fixed-X and model-X because covariance construction can dominate model-X:

| Regime | n | p | signal count | Purpose |
|---|---:|---:|---:|---|
| Correctness | 1,000 | 100 | 15 | external parity and FDR calibration |
| Medium | 5,000 | 250 | 25 | practical selector comparison |
| Large-n | 20,000 | 500 | 50 | GPU throughput |
| High-p | 5,000 | 1,000 | 60 | covariance/optimization stress |

Use:

```text
q ∈ {0.05, 0.10, 0.20}
rho ∈ {0.0, 0.3, 0.7}
5 or more data seeds
```

Required frontend metrics:

- timing;
- precision/recall/FDP/F1/Jaccard;
- target FDR and estimated FDR;
- selected-set size;
- threshold non-null rate;
- NumPy/CuPy agreement;
- knockpy comparison only for an aligned model-X configuration;
- model-X draw count and covariance-shrinkage metadata.

Recommended artifact:

```text
feature_selection_knockoff_202607xx.json
```

Exit gate: the Selection panel must contain meaningful non-degenerate rows across at least two q values and three scales; the source must not be added merely to make the sidebar non-empty.

## 8. ANOVA crossover benchmark plan

### 8.1 What the current data actually shows

The June 24 source is fully parsed and should not be described as having no GPU advantage. At its largest scale:

- one-way ANOVA shows a CuPy advantage over NumPy;
- two-way ANOVA shows a CuPy advantage over NumPy;
- Welch is near parity for CuPy;
- Tukey HSD is approximately backend-neutral;
- Bonferroni remains CPU-favored.

The problem is that only three scale points exist and different methods have different computational drivers. The dashboard cannot identify a reliable crossover interval from those points.

### 8.2 Timing-protocol concern

The current `benchmark_new_modules.py` timing loop does not explicitly synchronize CuPy or Torch around measured calls. GPU timings should therefore be rerun with a synchronization-safe protocol before being used as a precise crossover claim.

### 8.3 Recommended crossover grids

Do not combine all ANOVA methods under one scale definition.

For one-way and Welch, vary both total observations and group count:

```text
total n ∈ {1K, 5K, 10K, 50K, 100K, 500K, 1M, 2M, 5M}
groups ∈ {5, 20, 100}
```

For two-way ANOVA, vary:

```text
cell count ∈ {3×4, 10×10, 20×20}
observations per cell ∈ {100, 1K, 10K, 100K}
```

For Tukey HSD and Bonferroni, group count and number of pairwise comparisons are more important than raw n. Record them as separate post-hoc variants rather than mixing them into the one-way crossover statement.

Required protocol:

- arrays prepared before fit-only timing;
- explicit synchronization before and after every GPU measurement;
- 2 warmups;
- at least 10 measured calls for sub-millisecond methods, using batched repetitions inside one timed block to reduce timer noise;
- 5 measured calls for expensive post-hoc methods;
- 3 data seeds;
- statistic/p-value agreement with SciPy or an independent NumPy reference;
- separate reporting of host-side distribution-CDF time if it forces a CPU boundary.

Recommended artifact:

```text
anova_gpu_crossover_202607xx.json
```

Exit gate: report the smallest tested scale where each GPU backend remains faster than NumPy across all seeds and repeats, or state that no stable crossover was observed.

## 9. Delivery priority

### P0 — immediate existing-data repair

1. Expose all three aligned GAM scales.
2. Add parser tests and frontend navigation checks.
3. Regenerate canonical and deployed assets.

### P1 — empty or misleading headline coverage

1. Feature-selection canonical benchmark, because the category is empty.
2. Ordered large-scale crossover benchmark, because current data only demonstrates GPU overhead.
3. ANOVA synchronization-safe crossover benchmark, because current scale coverage is too coarse for a crossover claim.

### P2 — broaden partial method-family coverage

1. Full nonparametric comparison source using existing runners.
2. Full covariance-estimator source after PR #79 corrections are present on the benchmarked code.

## 10. Common source and parser contract

Every new artifact must:

- be dated on or after the dashboard cutoff;
- identify the exact git commit benchmarked;
- record P100 hardware and software versions;
- define timing scope and transfer policy;
- record warmups, repeats, seeds, synchronization, and aggregation scope;
- include failed/unsupported/OOM rows explicitly;
- preserve method-specific parameters in `method_config_id` inputs;
- include matched NumPy references for GPU speedups;
- use external references only for aligned objectives/outputs;
- be copied under `results/benchmark_frontend_sources/`;
- be SHA256-registered in `frontend_sources.json`;
- receive a dedicated parser or documented parser-version update;
- pass schema, semantic, deterministic-generation, TypeScript, build, staleness, and Playwright checks.

## 11. Completion definition

The audit is complete when:

- the existing GAM omission is fixed and deployed;
- Ordered has enough scales to identify or rule out a GPU crossover;
- Nonparametric includes complete aligned backend matrices for the selected method families;
- Covariance includes shrinkage and robust/sparse estimators, not only EmpiricalCovariance;
- Feature Selection has a real current source and meaningful Selection-panel metrics;
- ANOVA reports method-specific GPU crossover intervals from synchronized timing;
- no pre-June source is reintroduced;
- documentation distinguishes available data from planned benchmark work.
