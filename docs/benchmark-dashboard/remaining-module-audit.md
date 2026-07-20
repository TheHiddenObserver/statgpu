# Benchmark Dashboard Remaining-Module Audit

## 1. Scope and decision rule

This audit extends the domain- and method-level plans after a second repository-wide pass over:

- every source registered by `dev/benchmarks/frontend_sources.json`;
- the complete key space inside each registered source;
- maintained benchmark runners;
- June 2026-or-later result artifacts outside the canonical source directory;
- implemented model, inference, distribution, and utility surfaces;
- canonical parser output and frontend filter identities.

The classification rule is strict:

1. **Eligible source omission** — a registered, sufficiently specified source already contains the row. Repair the parser/frontend immediately.
2. **Result exists but is not canonical-ready** — a June-or-later result exists, but it is rounded Markdown, lacks dimensions/provenance, has known alignment problems, or is not machine-readable. Preserve the evidence and create a structured conversion/rerun task; do not fabricate canonical fields.
3. **Benchmark-data gap** — implementation or runner exists but no current reproducible result artifact exists. Record a new benchmark.
4. **Intentional deduplication** — a source contains a less informative duplicate of a richer registered comparison. Do not emit duplicate dashboard rows; document the choice.

## 2. New eligible-source omissions repaired

### 2.1 Complete Unsupervised matrix

`unsupervised_20260627.json` contains substantially more data than the previous parser exposed. The old parser selected one representative scale for most estimators, even though the source contains:

- PCA, KMeans, GaussianMixture, NMF, TruncatedSVD, and IncrementalPCA at small/medium/large scales;
- AgglomerativeClustering at small/medium scales;
- DBSCAN in 10-dimensional and 50-dimensional variants at all three scales;
- UMAP at small/medium scales;
- t-SNE at its feasible small scale;
- MiniBatchKMeans and MiniBatchNMF at all three scales.

Parser v2.0 now emits the complete source matrix:

```text
131 rows
```

The repair also corrects a source-runner interpretation error. The runner template names the large regime as `(100K, 100)`, but several estimators receive `min(p, 50)` and therefore actually fit `100K×50`. The canonical scale now describes the array passed to the estimator rather than the uncapped template.

Expected model-level row counts are:

| Model | Rows |
|---|---:|
| PCA | 12 |
| KMeans | 12 |
| GaussianMixture | 12 |
| NMF | 12 |
| TruncatedSVD | 12 |
| IncrementalPCA | 12 |
| AgglomerativeClustering | 8 |
| DBSCAN | 24 |
| UMAP | 6 |
| TSNE | 3 |
| MiniBatchKMeans | 9 |
| MiniBatchNMF | 9 |

This adds 83 normalized runs relative to the representative-scale parser.

### 2.2 Complete PR #74 inference matrix

`ordered_inference_pr74.json` was previously parsed only for Ordered Logit/Probit and Quantile kernel/bootstrap inference. The source also contains two scales and three backends for:

- penalized logistic L2 with HC0 sandwich inference;
- penalized logistic SCAD with oracle inference;
- Lasso/penalized linear regression with bootstrap inference.

Parser v2.0 now emits:

```text
3 methods × 2 scales × 3 backends = 18 rows
```

The rows retain:

- the correct model family, loss, penalty, and alpha;
- inference method and covariance type;
- `timing_scope=fit_plus_inference`;
- BSE output;
- matched GPU-versus-NumPy computed speedup;
- source date and parser provenance.

These rows must not be described as pure fit timings because the benchmark runner times model fitting together with inference.

### 2.3 Both GAM comparison configurations

The June 24 GAM source contains two distinct pyGAM comparison blocks:

1. the ordinary/source-default comparison;
2. a precision-aligned comparison using uniform knots, `gamma=1.4`, and fixed `lambda=1.0`.

They are not duplicate measurements and have materially different timings and prediction differences. The complete parser now emits both variants:

```text
pygam-comparison
aligned-pygam
```

Each variant contains:

```text
3 scales × (NumPy + CuPy + Torch + pyGAM) = 12 rows
```

for 24 GAM rows in total. The solver label is also corrected from `gcv` to `fixed_lam`, because the source runner fits with `lam=1.0` rather than performing GCV.

### 2.4 Previously repaired source omissions

The following earlier repairs remain valid:

- CoxPH Breslow rows embedded in the June 23 loss-function source;
- all five ANOVA functions in the June 24 source;
- complete aligned PanelOLS/RandomEffects medium and large scales;
- complete aligned GAM scales.

After the new repairs, the expected canonical bundle is:

```text
8 registered sources
1,774 normalized runs
36 models
```

## 3. June-or-later result artifacts that are not yet canonical-ready

### 3.1 Distribution benchmark

`results/distribution_bench_2026-06-21.md` is a real P100 result artifact and should no longer be treated as if no distribution benchmark exists. It reports:

- 139/139 precision checks passing against SciPy;
- 15 distributions;
- PDF/PMF, CDF, SF, PPF, and ISF methods where applicable;
- NumPy timings at 10K, 100K, and 1M points;
- CuPy and Torch timings/speedups at the same scale grid.

It is not directly canonical-ready because:

- only a rounded Markdown report is committed; no structured raw JSON was found;
- the report does not retain per-repeat timings, standard deviations, seeds, or raw precision errors;
- method parameters are embedded in display strings;
- the current dashboard taxonomy has no distribution/function category;
- source SHA/provenance and parser contracts should not depend on fragile table formatting.

Required action:

1. regenerate or convert the result into a structured JSON source;
2. retain distribution name, parameters, operation, vector length, backend, repeats, synchronization, and precision metrics;
3. add a dedicated `distributions` category rather than placing these function benchmarks under GLM or inference;
4. mark any values reconstructed only from the rounded report as `reported`, not `measured`;
5. register the source only after deterministic parser and browser tests exist.

Recommended artifact:

```text
results/benchmark_frontend_sources/distributions_202607xx.json
```

Priority: **P1**, because the existing result already demonstrates large method-dependent GPU crossover behavior and represents a substantial implemented module currently absent from the dashboard.

### 3.2 P2 Panel rows with incomplete scale identity

`p2_benchmark_20260617.json` contains PooledOLS and Fama–MacBeth timing fields, but its scale keys are malformed:

```text
n=5000p=
n=20000p=
n=50000p=
```

The number of regressors and panel entity/time dimensions cannot be recovered from the source. The frontend must not invent `p`, entity count, or time count. These rows therefore require a structured rerun or corrected source artifact, despite timing values being present.

Recommended action:

- rerun with `n_entities`, `n_times`, `n_features`, covariance type, and effect structure explicitly recorded;
- include all supported backends and aligned references;
- use the existing panel expansion plan rather than attaching guessed scales to the P2 rows.

### 3.3 Older June 24 external Panel comparison

An older external-comparison artifact contains small Panel rows and PooledOLS rows, but it is not suitable for direct canonical registration:

- the backend matrix is incomplete in the first version;
- some PooledOLS coefficient relative differences are extremely large;
- the corrected v2 artifact intentionally retains only the better-aligned PanelOLS and RandomEffects medium/large matrix.

The dashboard should not recover small/PooledOLS rows from the older artifact merely to increase coverage. A corrected rerun is required.

## 4. Remaining benchmark-data gaps

### 4.1 Robust and Quantile

- Bisquare/Tukey-biweight and Fair loss are implemented but have no canonical comparison source.
- Current Huber and Quantile fit performance is CPU-only.
- Penalized Quantile/Robust matrices remain unrun.
- Quantile levels beyond the median, especially `q=0.1` and `q=0.9`, are missing.

Use:

```text
docs/benchmark-dashboard/robust-loss-comparison-plan.md
docs/benchmark-dashboard/penalized-robust-quantile-plan.md
```

### 4.2 Linear/GLM inference breadth

The restored PR #74 rows cover only three inference configurations. Still missing are systematic comparisons of:

- HC0, HC1, HC2, HC3;
- HAC bandwidth/kernel choices;
- debiased Lasso inference;
- bootstrap replicate-count scaling;
- oracle inference support recovery;
- fit-only versus fit-plus-inference timing.

A maintained inference-backend runner exists, but no current structured model-inference result source covers this matrix.

Recommended artifact:

```text
linear_inference_202607xx.json
```

### 4.3 Cross-validation

No current canonical source covers:

- RidgeCV;
- LassoCV;
- ElasticNetCV;
- LogisticRegressionCV;
- PenalizedGLM_CV;
- CoxPHCV.

Benchmark fold count, grid size, path construction, warm starts, selected hyperparameters, refit time, and backend scaling.

### 4.4 Penalized survival

Current survival coverage is unpenalized CoxPH only. Missing:

- PenalizedCoxPHModel;
- SCAD/MCP/L2 paths;
- Breslow versus Efron under penalty;
- CoxPHCV;
- support, convergence, C-index, and path/refit timing.

### 4.5 Nonparametric breadth

Still missing or incomplete:

- KDE;
- Nadaraya–Watson and local-linear kernel regression;
- KernelRidge and KernelRidgeCV;
- KernelPCA;
- natural cubic splines;
- pairwise kernels beyond RBF;
- complete NumPy/CuPy/Torch/external matrices for each operation scope.

### 4.6 Covariance breadth

Only EmpiricalCovariance is canonical. Missing current sources for:

- ShrunkCovariance;
- LedoitWolf;
- OAS;
- GraphicalLasso/GraphicalLassoCV;
- MinCovDet.

### 4.7 Feature Selection

The category remains empty because no current eligible artifact exists. Run the multi-scale Knockoff/baseline matrix already described in the domain plan.

### 4.8 Panel estimator breadth

Beyond aligned PanelOLS/RandomEffects, structured current results are missing for:

- PooledOLS and covariance variants;
- two-way PanelOLS;
- BetweenOLS;
- FirstDifferenceOLS;
- Fama–MacBeth;
- balanced versus unbalanced panels.

### 4.9 Multiple testing and resampling utilities

A maintained runner exists for:

- BH/BY/Holm/Bonferroni adjustments;
- Fisher/Cauchy/ACAT p-value combination;
- bootstrap statistics;
- permutation tests.

No current result JSON is committed. Run it on the canonical environment and add a dedicated function-level category rather than mixing it into ANOVA.

### 4.10 Standalone NNDescent and auxiliary unsupervised operations

NNDescent is used by UMAP and is implemented as a standalone method, but the current source does not benchmark it independently. Also consider separating fit and transform/predict timings for estimators where those operations have different scaling.

### 4.11 Ordered and ANOVA crossover

The previously recorded issues remain:

- Ordered models have only `500×5` and `2K×10`;
- ANOVA has too few scale points and lacks a fully explicit synchronization contract.

## 5. Timing-protocol issues discovered during the audit

### 5.1 Unsupervised synchronization

The complete source matrix is now exposed, but the existing runner should be rerun before making precise GPU-crossover claims:

- some Torch paths are invoked with a generic CUDA device token whose synchronization branch is CuPy-specific;
- direct KMeans/MiniBatchKMeans blocks do not consistently synchronize after the measured operation;
- transfer-inclusive and fit-only scopes are not separated.

The current rows remain valid as source-reported measurements, but future benchmark conclusions should use a corrected runner.

### 5.2 New Modules and ANOVA

The existing Panel/GAM/ANOVA runner similarly needs explicit backend synchronization around each timed operation. Existing aligned correctness rows remain useful, but exact crossover claims require reruns.

## 6. Registered-source audit closure

After parser versions:

```text
ordered_inference_benchmark 2.0
unsupervised_benchmark 2.0
new_modules_with_anova_benchmark 1.4
```

no additional sufficiently specified row was found hidden inside the eight registered sources.

Intentional exclusions are:

- Efron rows in `loss_functions_20260623.json`, because the dedicated Efron source has richer light/heavy-ties coverage;
- raw GAM performance rows that duplicate the same fixed-lambda configuration without the external/validation metadata available in the selected comparison blocks;
- P2 ANOVA rows, because the June 24 ANOVA source is richer;
- malformed P2 Panel rows, because their scale identity is incomplete.

## 7. Updated priority

### P0 — completed source repairs

1. Complete Unsupervised matrix and correct actual dimensions.
2. Complete PR #74 inference methods.
3. Preserve both GAM comparison configurations and correct fixed-lambda semantics.
4. Preserve complete Panel and GAM scale matrices.

### P1 — existing evidence or misleading core gaps

1. Structured distribution source/category from the June benchmark.
2. Huber/Bisquare/Fair all-backend benchmark.
3. Feature Selection.
4. Ordered crossover.
5. ANOVA synchronized crossover.
6. Current CV source.
7. Systematic linear/GLM inference source.

### P2 — method-family breadth

1. Full covariance estimators.
2. Full nonparametric matrix.
3. Penalized CoxPH/CoxPHCV.
4. Additional panel estimators.
5. Multiple-testing/resampling utilities.
6. Standalone NNDescent and operation-specific unsupervised timing.

## 8. Completion rule

A module is considered dashboard-covered only when:

- a current structured result artifact exists;
- all material method and case parameters are retained;
- timing scope and backend synchronization are explicit;
- correctness/statistical-quality metrics accompany timing;
- failed, unsupported, and OOM cases are explicit;
- parser output retains all eligible source rows;
- intentional deduplication is documented;
- frontend filters and panels represent the method without relabeling;
- deterministic generation, Python 3.9/3.11, TypeScript, build, and Playwright checks pass.
