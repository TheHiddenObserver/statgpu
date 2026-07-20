# Benchmark Dashboard Method-Level Coverage Audit

## 1. Scope

This audit complements the domain-level plan in `domain-coverage-audit-plan.md`. It compares:

- implemented methods and loss families;
- maintained benchmark runners;
- committed result artifacts dated 2026-06-01 or later;
- canonical parser output;
- current dashboard model and category coverage.

The same decision rule applies:

- if an eligible source already contains a row, repair the parser/frontend;
- if only implementation or runner code exists, record a benchmark-data gap;
- do not reconnect pre-June artifacts or synthesize missing rows.

## 2. Executive matrix

| Area | Implemented or runnable | Current canonical coverage | Classification | Action |
|---|---|---|---|---|
| Robust losses | Huber, Bisquare, Fair | CPU Huber only for fit performance | benchmark-data gap | run full robust-loss comparison |
| Quantile variants | general quantile loss and penalized quantile | primarily median/one benchmark setup | benchmark-data gap | add multiple quantiles after core source |
| Panel scale coverage | medium and large aligned PanelOLS/RandomEffects rows exist | large only before parser v1.3 | frontend/parser omission | expose both scales now |
| Panel estimator breadth | PooledOLS, FamaMacBeth, BetweenOLS, FirstDifferenceOLS and others have runners | only PanelOLS and RandomEffects aligned rows | benchmark-data gap | generate structured multi-estimator source |
| Cross-validation | RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV, PenalizedGLM_CV, CoxPHCV | no eligible current CV source | benchmark-data gap | run current CV benchmark matrix |
| Penalized survival | PenalizedCoxPHModel and CoxPHCV | unpenalized CoxPH only | benchmark-data gap | add penalized/CV survival source |
| Nonparametric breadth | KernelRidge/CV, multiple pairwise kernels, natural cubic splines, KDE/kernel regression runners | GAM, Nystroem, RBF and B-spline subset | benchmark-data gap | use full nonparametric plan |
| Covariance breadth | shrinkage, sparse and robust estimators | EmpiricalCovariance only | benchmark-data gap | use covariance-estimator plan |
| Feature selection | fixed-X/model-X knockoff runners | no eligible source | benchmark-data gap | run and register selection source |
| Multiple testing | p-value adjustment/combination/permutation implementations and runner | no dashboard source/category rows | benchmark-data gap | add function-level benchmark source |
| Ordered scale | arbitrary-scale parser and model implementation | two small scales | benchmark-data gap | run ordered crossover source |
| ANOVA crossover | five functions and existing data | complete source, coarse scale grid | benchmark-design gap | synchronized crossover rerun |

## 3. Existing-data repairs

### 3.1 GAM

The aligned small, medium and large pyGAM rows already existed and are now exposed by parser v1.2.

### 3.2 Panel

The June 24 source contains aligned linearmodels comparisons at:

```text
medium = 10K×10
large  = 100K×20
```

for both:

```text
PanelOLS
RandomEffects
```

Each model/scale combination contains statgpu NumPy, CuPy, Torch and a linearmodels reference. The previous parser selected only `panel_large_*` rows.

Parser v1.3 replaces the large-only subset with:

```text
2 scales × 2 models × (3 statgpu backends + 1 external reference)
= 16 aligned panel rows
```

This adds eight normalized runs without changing the source manifest count.

Acceptance checks:

- both scale labels appear;
- each model/scale has four aligned rows;
- statgpu rows preserve runner-reported speedup and coefficient relative error;
- no duplicate large rows remain;
- all rows use parser version 1.3.

## 4. Robust and quantile gaps

The current eligible loss-function source benchmarks only:

```text
Quantile: statgpu CPU versus scikit-learn
Huber:    statgpu CPU versus scikit-learn
```

It does not provide fit-performance rows for CuPy or Torch, and it contains no Bisquare or Fair results.

The implementation supports:

```text
Huber
Bisquare / Tukey biweight
Fair
```

The repository has P100 smoke scripts for Huber/Bisquare solver paths, but they print one-off timing output and do not produce canonical JSON. Fair has no comparable benchmark artifact.

Required action is defined in:

```text
docs/benchmark-dashboard/robust-loss-comparison-plan.md
```

Important additional dimensions are:

- clean versus contaminated data;
- MAD versus Proposal-2 scale estimation;
- Bisquare initialization policy;
- all three statgpu backends;
- matched external or independent references;
- unpenalized loss comparison separated from the penalized matrix.

Multiple quantile levels such as `q=0.1`, `0.5` and `0.9` should be added only after the median-regression source is stable. Different quantiles must remain distinct variants.

## 5. Panel estimator breadth

The maintained module runner contains additional panel estimators, including:

```text
PooledOLS
PooledOLS with HAC
FamaMacBeth
BetweenOLS
FirstDifferenceOLS
PanelOLS two-way effects
```

The current canonical source does not contain structured aligned rows for these methods. Its panel performance block is empty and its external comparison block is limited to PanelOLS and RandomEffects.

A future panel source should vary:

- estimator;
- entity/time dimensions;
- number of regressors;
- covariance type;
- fixed-effect structure;
- balanced versus unbalanced panels where supported.

Required metrics include timing, coefficient agreement, standard-error agreement, convergence/failure status and memory usage for absorbed fixed effects.

Recommended artifact:

```text
panel_estimators_202607xx.json
```

## 6. Cross-validation coverage

The implementation surface includes:

```text
RidgeCV
LassoCV
ElasticNetCV
LogisticRegressionCV
PenalizedGLM_CV
CoxPHCV
```

Historical LassoCV parsing code and older artifacts exist, but no eligible June-or-later CV source is registered. The dashboard therefore cannot currently compare:

- path construction;
- fold parallelism;
- warm starts;
- selected alpha/l1-ratio;
- refit time;
- backend scaling as folds and grid size grow.

Recommended source design:

```text
models: RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV
folds: 3, 5, 10
grid sizes: 20, 50, 100
scales: method-specific medium and large grids
backends: NumPy, CuPy, Torch where supported
references: aligned scikit-learn implementations
```

Keep `PenalizedGLM_CV` and `CoxPHCV` in later method-specific sources if a single combined matrix becomes too dense.

## 7. Penalized survival coverage

Current survival sources benchmark unpenalized CoxPH with Breslow and Efron ties. They do not cover:

```text
PenalizedCoxPHModel
CoxPHCV
SCAD/MCP/L2 Cox paths
```

This gap is important because the solver, convergence and GPU crossover behavior differs materially from unpenalized CoxPH.

A future source should separate:

- Breslow and Efron ties;
- low and high tie density;
- penalty type;
- requested and resolved solver;
- path fitting versus final refit;
- concordance and coefficient-support metrics.

Recommended artifact:

```text
penalized_coxph_202607xx.json
```

## 8. Nonparametric method gaps

Beyond the currently planned KDE/kernel-regression/P2 expansion, the implemented-method inventory includes methods not represented in canonical sources:

```text
KernelRidge
KernelRidgeCV
natural_cubic_spline_basis
pairwise kernels beyond RBF
```

Do not force these into the same scale grid:

- KernelRidge is constrained by kernel-matrix memory;
- KernelRidgeCV adds grid/fold dimensions;
- pairwise kernels need rectangular or capped matrices;
- basis-generation methods scale differently from model fitting.

These should be incorporated into the full nonparametric source or a follow-up source with explicit operation scope (`fit`, `transform`, `predict`, or kernel construction).

## 9. Multiple-testing and inference utilities

The repository implements:

```text
adjust_pvalues
combine_pvalues
permutation_test
```

and has a benchmark runner covering at least Fisher/Cauchy-style p-value combination and backend consistency. Existing referenced result artifacts are historical and are not current dashboard sources.

A function-level source should cover:

- number of hypotheses from `1K` through multi-million scale;
- Fisher, Cauchy/ACAT and Stouffer combinations;
- BH, BY, Holm, Hochberg and Bonferroni adjustment;
- NumPy/CuPy/Torch where implemented;
- SciPy or independent formula parity;
- extreme-tail numerical stability;
- permutation count and batch size for permutation tests.

Because these are functions rather than fitted models, the frontend may need a clearly named `multiple_testing` category or a documented function-benchmark representation. Do not place them under ANOVA merely because Bonferroni appears there as a post-hoc function.

Recommended artifact:

```text
multiple_testing_202607xx.json
```

## 10. Priority

### P0 — existing eligible data

1. Complete aligned Panel medium/large coverage.
2. Keep complete GAM scale coverage.
3. Regenerate and deploy canonical assets.

### P1 — misleading or absent core coverage

1. Huber/Bisquare/Fair comparison on all backends and contamination regimes.
2. Feature-selection source.
3. Ordered crossover source.
4. ANOVA synchronized crossover source.
5. Current cross-validation source.

### P2 — method-family breadth

1. Full covariance estimators.
2. Full nonparametric methods including KernelRidge.
3. Penalized CoxPH and CoxPHCV.
4. Additional panel estimators.
5. Multiple-testing and permutation utilities.

## 11. Completion rule

A method is considered covered only when:

- a current reproducible result artifact exists;
- timing scope and backend synchronization are explicit;
- method/case identities retain all material parameters;
- correctness or statistical-quality metrics accompany timing;
- failed, unsupported and OOM cases are explicit;
- the parser exposes all eligible rows;
- frontend filters and panels can represent the results without relabeling another method;
- deterministic generation and browser tests pass.
