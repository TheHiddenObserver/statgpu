# StatGPU Unified Plan (Merged 2026-04-05)

This file is the consolidated planning entry in the workspace root.
It now includes both the actionable short queue and the detailed long-range blueprint.
Merged from:
- `plan.md` (priority queue after bootstrap phase)
- `TO_DO.md` (engineering gate + status board)
- session plan snapshot (long-range architecture and release strategy)

## 1. Hard Gates (Must Follow)

- Every new feature must provide both CPU and GPU implementations.
- CPU and GPU paths must be independently verifiable.
- Every new statistical feature (inference, stopping rule, memory behavior affecting numerics) must include external baseline checks:
  - `statsmodels` for inference/statistics first
  - `sklearn` for estimator/prediction consistency first
  - `R` for key-method supplement
- External comparisons must use explicit, aligned settings:
  - same feature set
  - same ties/solver configuration
  - same regularization and convergence settings (`alpha/C/max_iter/tol`)

## 2. Priority Queue (Current)

1. P0 (completed): Multiple-testing expansion batch 1
- Added global p-value combination methods: Fisher and Cauchy (ACAT alias).
- Added unified API in `statgpu.inference` and thin wrapper on `BaseEstimator`.
- Added NumPy/CuPy consistency and axis-behavior tests.

2. P0/P1 (completed in this cycle): Fisher/Cauchy precision + timing benchmark
- Added benchmark section for `combine_pvalues` Fisher/Cauchy.
- Includes precision comparison:
  - Fisher vs SciPy `scipy.stats.combine_pvalues`.
  - Cauchy vs independent NumPy reference implementation.
  - NumPy vs CuPy consistency for both methods.
- Includes runtime comparison:
  - statgpu NumPy vs statgpu CuPy for Fisher/Cauchy.
  - SciPy Fisher vs statgpu NumPy Fisher.

3. P1 (high): Multiple-testing expansion batch 2
- Add knockoff roadmap (fixed-X first, model-X next) for linear-family feature selection.
- Standardize outputs: selected set, W statistics, q/FDR trajectory, random-state trace.
- Add R/Python baseline comparison when optional packages are available.

4. P2 (medium): Multiple-testing expansion batch 3
- Add correlation-aware/global methods as needed:
  - Brown/Kost style Fisher correction
  - Harmonic mean p-value
  - weighted global combination extensions

5. P3 (lower): Time-series methods
- Add time-series inference/diagnostic methods after multiple-testing v1 is stable.

6. P3 (lower): Spatial econometrics methods
- Add SAR/SEM roadmap after the same v1 checkpoint.

## 3. In-Progress P0 Track

- Improve inference rigor:
  - extend robust covariance to `cluster-robust` and `HAC`
  - improve cross-device alignment for `SE/t/z/p/CI` and `AIC/BIC/LLF`
- Lasso inference enhancements:
  - move toward de-biased/post-selection inference
  - continue bootstrap GPU optimization and large-scale benchmarks
- CoxPH inference/evaluation:
  - robust/cluster sandwich covariance
  - strict pairwise vs approximate C-index switchable path

## 4. Completed Capability Snapshot (2026-04)

- Lasso inference semantic rename with compatibility aliases.
- GPU-side inference enhancement for `gpu_ols_inference` path.
- Added `gpu_memory_cleanup` across core models.
- Fixed `LogisticRegression.fit()` CUDA input conversion path.
- Added `cov_type=nonrobust/hc0/hc1` for Linear/Logistic/Cox (with CPU/GPU availability by model path).
- Added Cox `cov_type=cluster` (CPU path).
- Added statsmodels comparison tests for Linear/Logistic HC0/HC1.
- Added benchmark scripts for lasso inference, gpu memory cleanup, large-scale methods, external frameworks.
- Ridge full inference support added:
  - `cov_type=nonrobust/hc0/hc1` CPU+GPU
  - inference switch
  - key statistics fields and `summary()`
  - dedicated ridge inference tests

## 5. P1-P3 Model/Feature Backlog

### P1 API parity and feature completion
- Lasso: `ElasticNet(l1_ratio)`, `positive`, `warm_start`, alpha path
- Ridge: `warm_start`, path
- LogisticRegression: multinomial/softmax, L1/elastic-net, richer diagnostics
- CoxPH: strata, frailty, time-varying covariates, penalized Cox
- sparse input support (CSR/CSC)

### P2 model selection and preprocessing
- unified `path/cv/grid-search/warm_start`
- preprocessing switches (`center/standardize/normalize`)

### P3 benchmark framework standardization
- unified split timing (`data build / fit / inference`)
- unified KKT-equivalent stopping calibration
- unified numeric-difference templates (`L_inf`, `L2_rel`, `bse/t/p/CI`)
- unified `gpu_memory_cleanup` report template

## 6. Long-Range Program (Execution and Release)

- Keep strict inference as default behavior.
- Build unified inference abstraction across Linear/Ridge/Logistic/Cox (Lasso with specialized extension).
- Use high-strength engineering guardrails:
  - lint/type/test baseline
  - nightly + monthly dual-track release
- Monthly stable remains blocking on:
  - correctness gates
  - external consistency matrices
  - benchmark non-regression
  - documentation parity (EN/CN + changelog consistency)

## 7. Verification Matrix (Unified)

1. Strict inference gate
- Ridge/Lasso strict mode must pass external alignment thresholds.

2. Device consistency gate
- strict mode output aligns on CPU/GPU for key inference fields.

3. New feature admission gate
- each new method must include:
  - implementation
  - tests
  - external comparison or benchmark
  - docs update

4. Engineering gate
- nightly: lint/type/test
- monthly: external matrix + benchmark non-regression + docs sync

## 8. Thresholds and Conventions

- strict alignment threshold (current baseline):
  - coef: `1e-6`
  - bse: `1e-3`
  - p-value: `5e-2`
- strict failure policy:
  - default: raise error
  - fallback only when explicitly enabled

## 9. Baseline Mapping Notes

- Linear/Ridge align with lm/lm.ridge style baselines.
- Lasso/ElasticNet align with glmnet/selective-inference style baselines where available.
- Logistic aligns with glm/glmnet style baselines.
- Cox aligns with survival::coxph style baselines (advanced paths extend later).

## 10. Next Action Queue

1. Start knockoff batch 2 skeleton (fixed-X first).
2. Add structured benchmark/comparison script for knockoff once API skeleton lands.
3. Add docs page for `combine_pvalues` (Fisher/Cauchy/ACAT) with benchmark interpretation notes.

## 11. Consolidation Note

- This file is the workspace-level single entry point for planning.
- Existing files are retained temporarily for backward compatibility and traceability.

## 12. User Blueprint (Detailed Merge)

This section integrates the previously provided detailed blueprint into the unified file.

### 12.1 Strategic Spine

- Keep default strict inference.
- Enforce high-strength engineering gates.
- Use dual-track releases: nightly and monthly stable.
- Sequence: close current inference correctness gaps first (Ridge/Lasso), then scale methods and tooling through a unified inference abstraction and baseline matrix.

### 12.2 Phase Roadmap (Detailed)

1. Phase 0 (P0 blocking): freeze global statistical and interface contracts.
- Unify cov_type semantics, strict defaults, df/distribution rules, CPU/GPU consistency policy, and R/statsmodels/sklearn mapping.

2. Phase 0 (P0 parallel): establish engineering baseline.
- lint/type/test entrypoints, CI layering, nightly/monthly responsibilities.

3. Phase 1 (P0 correctness): Ridge penalty-aware covariance fix.
- CPU+GPU alignment, intercept non-penalized, alpha-bucket consistency and external baseline checks.

4. Phase 1 (P0 correctness): Lasso inference redesign.
- strict defaults to de-biased lasso, approximate route remains non-default with explicit fallback switch and strong warnings.

5. Phase 1 (P1 parallel): unify inference abstraction.
- Consolidate validation, covariance, p-value/CI, summary fields across Linear/Ridge/Logistic/Cox; Lasso keeps specialized extension interface.

6. Phase 2 (P1 first): linear-family extension batch.
- ElasticNet, Ridge/Lasso path, warm_start with strict inference and GPU alignment from first release.

7. Phase 2 (P1 second): survival extension batch.
- Cox strata/frailty/time-varying in staged delivery: estimation, inference, then baseline comparability.

8. Phase 2 (P1 next): classification extension batch.
- Logistic multinomial + L1/ElasticNet, estimation/probability first, strict inference next.

9. Phase 2 (P1 next): sparse input support.
- CSR/CSC first in linear family, then Logistic/Cox, with sparse-path benchmarks.

10. Phase 3 (P1/P2): unified model-selection tools.
- path/cv/grid-search/warm-start API convergence.

11. Phase 3 (P1/P2 parallel): complete external consistency matrix.
- Five-model unified report for coef/bse/p/CI/AIC/BIC with strict thresholds.

12. Phase 4 (P2): release process finalization.
- nightly for experimental capability with rolling regressions, monthly stable only after all gates pass and docs/changelogs are synchronized.

### 12.3 Detailed Verification Rules

1. strict inference gate
- Ridge and Lasso strict paths must pass statsmodels/R comparison within thresholds.

2. device consistency gate
- strict outputs for key inference fields must align across CPU/GPU.

3. feature admission gate
- every new method must include implementation + tests + external baseline + benchmark + docs.

4. engineering gate
- nightly must run lint/type/test, monthly additionally enforces baseline matrix and benchmark non-regression.

5. release gate
- README, USAGE, EN/CN docs and changelog capability statements must remain consistent.

6. remote experiment gate
- performance-sensitive inference methods must include remote CUDA rerun artifacts (JSON + short MD summary) under `results/` for auditability.

### 12.4 Detailed Decisions (Locked)

- Expansion scope includes ElasticNet/path/warm_start, Logistic multiclass/penalties, advanced Cox features, model-selection tools, and sparse input.
- Priority order keeps linear-family pathing first and advanced Cox second.
- strict default remains the main inference strategy.
- Lasso strict mainline uses de-biased inference; approximate route is explicit fallback only.
- strict threshold baseline: coef 1e-6, bse 1e-3, p-value 5e-2.
- strict failures raise by default; downgrade only when explicitly enabled.
- CUDA usage layering rule stays active: model layer should not scatter direct cupy imports.

### 12.5 Two-Week Execution Cadence (Blueprint)

Week 1 focus:
- statistical contract freeze
- Ridge strict covariance corrections
- first CUDA layering cleanup
- minimum CI gate closure
- inference documentation alignment

Week 2 focus:
- Lasso strict mainline (de-biased)
- linear-family first extension
- Cox second-priority extension
- full external matrix expansion
- dual-track release dry run

### 12.6 Documentation Blueprint (EN-first, CN-follow)

- Freeze model-page template and citation policy.
- Build central references index and enforce bidirectional linking.
- Rebuild model pages in two batches (Linear/Ridge/Lasso then Logistic/Cox).
- Upgrade guides and quickstart/benchmarks with strict/approx and device-path guidance.
- Monthly stable blocks on documentation quality and EN/CN parity.

Model-page hard requirements:
- Overview, Path, Objective Function, Estimating Equation, Covariance/Inference, Parameters, CPU+GPU Examples, strict/approx difference, Outputs, FAQ, External Validation, References.

### 12.7 Optimizer Program (Detailed Merge)

Core decisions:
- Unified `optimizer` field + model-specific extensions.
- CPU+GPU support target for LARS (dense first).
- Consistency judged by objective/KKT first, not coefficient identity.
- Performance-first defaults: CPU FISTA, GPU adaptive ADMM/FISTA.
- strict inference outputs must satisfy thresholds across optimizers.

Implementation steps:
1. define unified optimizer contract and support matrix.
2. migrate Lasso legacy solver fields into unified interface.
3. add optimizer consistency tests (objective/KKT/prediction/inference).
4. enforce strict inference gate across optimizers.
5. implement GPU optimizer selection rules v1.
6. deliver LARS CPU+GPU dense MVP.
7. upgrade benchmarks with structured optimizer metrics.
8. update EN/CN docs with optimizer selection guidance.
9. enforce optimizer gates in nightly and monthly pipelines.

### 12.8 New Model Onboarding Blueprint

Layering:
- base class contract
- estimator API contract
- CPU/GPU execution-path contract
- strict/approx inference contract (if supported)
- export contract

Mandatory onboarding flow:
1. interface contract proposal.
2. CPU implementation, then GPU path + fallback.
3. strict inference first when applicable.
4. submodule and top-level exports.
5. unit tests + external consistency + optimizer tests as applicable.
6. structured benchmark output.
7. EN-first/CN-follow docs.
8. release only after nightly/monthly gates.

### 12.9 Additional Baseline Extension (R/statsmodels alignment backlog)

P0 recommendations:
- GLM family framework starting with Poisson.
- unified `predict_interval` (confidence vs prediction semantics).
- covariance expansion to HC2/HC3/HAC.
- Cox diagnostic baseline (Schoenfeld/martingale/deviance and PH tests).

P1 recommendations:
- structured result objects (`to_dict`/`to_dataframe`).
- unified `wald_test/t_test/f_test`.
- standardized `sample_weight/offset` semantics.
- logistic marginal effects.
- standardized missing-data policy.

P2 recommendations:
- formula/patsy-style layer.
- summary HTML/LaTeX export.
- Cox exact ties and richer survival intervals.

### 12.10 HC0-HC3 Cross-Framework Alignment Blueprint

- Explicit formula and mapping matrix for HC0/HC1/HC2/HC3 across statgpu, statsmodels, and R sandwich.
- User-facing API remains lower-case (`nonrobust/hc0/hc1/hc2/hc3/hac/cluster`).
- Add leverage-sensitive dedicated tests.
- Keep implementation default `nonrobust`, while docs recommend robust defaults by scenario.

### 12.11 Classification Evaluation Blueprint

- Promote classification-evaluation capability to P1.
- Build generic metrics modules for confusion matrix/table, thresholded metrics, ROC/AUC, PR/AP.
- Add plotting layer as optional dependency.
- Keep Logistic model wrappers thin and reusable.
- Add consistency tests against sklearn and doc updates in EN/CN.

### 12.12 Post-Bootstrap Inference Delta (Merged)

1. P0 complete:
- Fisher and Cauchy/ACAT combine-pvalue support with wrappers and CPU/GPU consistency tests.

1.1 P0 complete (remote supplement):
- remote Fisher/Cauchy benchmark rerun is attached with precision + timing evidence:
  - JSON: `results/remote_fisher_cauchy_benchmark_2026-04-05.json`
  - summary: `results/remote_fisher_cauchy_benchmark_2026-04-05.md`
  - key metrics: Fisher SciPy vs statgpu NumPy `88.152x`; Fisher NumPy vs CuPy `4.879x`; Cauchy NumPy vs CuPy `4.634x`; Fisher/Cauchy consistency diffs remained at floating-point noise level.

2. P1 high:
- knockoff roadmap with fixed-X first and model-X next, plus standardized outputs and baseline comparisons.

3. P2 medium:
- correlation-aware global methods (Brown/Kost/HMP/weighted variants).

4. P3 lower:
- time-series methods after inference v1 stabilizes.

5. P3 lower:
- spatial econometrics methods after the same checkpoint.
