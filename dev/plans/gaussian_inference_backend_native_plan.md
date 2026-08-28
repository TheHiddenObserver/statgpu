# Gaussian Linear-Model Backend-Native Inference Plan

Issue: #127  
Baseline release: `0.2.5`  
Baseline `master`: `84f8bc7e17f66466b3a325cbb007b6cb41843821`  
Status: planning review/fix in progress; production work not started

## 1. Goal

Migrate the maintained Gaussian linear-model inference path so covariance, standard errors, test statistics, p-values, and confidence intervals execute on the selected NumPy/CuPy/Torch numerical backend without silent full-design/residual fallback to NumPy/SciPy.

The change is an execution-locality and provenance repair. It does not redefine established Gaussian/Ridge statistics, fitting objectives, penalties, CV selection, formula semantics, or public reporting results.

A NumPy reporting snapshot may remain after numerical inference where required by the established result/reporting API. The implementation must therefore separate **backend-native numerical inference state** from **legacy reporting state** instead of forcing one representation to serve both purposes.

This work is intentionally narrower than a repository-wide inference rewrite.

## 2. Impact classification and capability decisions

Active axes:

- backend/device locality and failure behavior;
- inference;
- numerical precision;
- public reporting compatibility;
- formula regression compatibility for formula-facing consumers;
- Ridge/L2 objective-scale preservation;
- CV final-refit inference where the repaired path is consumed;
- physical-GPU validation and provenance;
- documentation/evidence artifacts.

Inactive axes:

- new loss definitions;
- new penalties;
- solver redesign;
- new model families;
- new CV selection/path algorithms;
- sparse-input support;
- repository-wide removal of SciPy;
- new formula support;
- user-facing GPU speedup claims.

Eliminating unintended full-design/residual GPU-to-CPU inference transfers is a correctness/backend-contract gate, not merely a performance optimization.

### 2.1 Minimum known capability table

Phase 1 must extend this table if another public Gaussian consumer is discovered before production edits continue.

| Public capability | backend | CV | inference | formula | benchmark |
| --- | --- | --- | --- | --- | --- |
| `LinearRegression` | `three-backend` | `non-tunable` | `supported` | `supported` | `not-performance-sensitive` |
| `Ridge` | `three-backend` | `supported` via maintained Ridge CV surface | `supported` | `supported` | `not-performance-sensitive` |
| `RidgeCV` | `three-backend` | `supported` | `supported` on final refit when enabled | `not-formula-facing` | `not-performance-sensitive` |
| `PenalizedLinearRegression`, squared-error + L2 path | `three-backend` | `supported` through maintained penalized CV surface | `supported` when enabled | `supported` | `not-performance-sensitive` |

A newly discovered public consumer must receive one of the repository's explicit capability decisions before it enters implementation scope. `planned` is not a valid completion state for an already maintained backend or inference capability.

### 2.2 Adjacent shared-mixin boundary

`statgpu/linear_model/penalized/_inference_mixin.py` also dispatches non-L2 inference paths. #127 must not accidentally reroute or redefine:

- L1/debiased inference;
- ElasticNet/debiased inference;
- SCAD/MCP oracle or bootstrap inference;
- other already documented estimation-only/failure behavior.

These penalty families are **not** added to #127's implementation scope, but their branch dispatch and established outputs are regression surfaces whenever the shared mixin is edited.

## 3. Mandatory consumer and data-lifecycle inventory

Before implementation, record the exact consumer graph and array lifecycle.

Known shared/consumer locations include:

1. `statgpu/linear_model/_gaussian_inference.py`;
2. `statgpu/linear_model/wrappers/_linear.py`;
3. `statgpu/linear_model/penalized/_inference_mixin.py`;
4. `statgpu/linear_model/wrappers/_ridge.py`;
5. `statgpu/linear_model/cv/_ridge_cv.py` for final-refit ownership.

For every public consumer record:

- requested device/backend;
- executed fit backend and concrete device;
- where design, residuals, parameters, scale, and weights are created;
- whether a backend-native fit buffer already exists;
- every `_to_numpy()`, `np.asarray()`, or equivalent host boundary before numerical inference completion;
- covariance implementation reached;
- distribution implementation reached;
- final reporting conversion;
- formula behavior;
- sample-weight behavior;
- scalar versus multi-target behavior;
- CV final-refit ownership where applicable.

The scope is this resolved Gaussian consumer graph, not every `scipy.stats` import under `statgpu/linear_model`.

If inventory finds another public Gaussian consumer, update the capability table and plan before editing that consumer.

## 4. Statistical and optimization non-change contract

Freeze established ordinary-regime behavior for:

- nonrobust inference;
- HC0 / HC1 / HC2 / HC3;
- Bartlett HAC and lag semantics;
- residual degrees of freedom;
- Ridge/L2 covariance and intercept-penalty treatment;
- weighted inference;
- rank-deficient/pseudoinverse behavior;
- scalar and multi-target output shape/order;
- formula feature names/order;
- sklearn clone/fitted-state/transactional-refit behavior;
- RidgeCV folds, candidate alphas, scoring, selected alpha, final-refit coefficients, and failure transactionality.

Historical output is not frozen when it is a demonstrated precision defect; small-df/extreme-tail behavior follows maintained distribution precision requirements.

### 4.1 Ridge/L2 objective and penalty mapping

#127 must not change the fit objective or regularization convention.

Current Ridge direct/CV paths use the repository's average-loss convention and equivalent `n_eff * alpha` contribution in unnormalized normal equations, with `n_eff` equal to the declared sample/weight normalization. Inference bread/penalty construction must preserve that same mapping.

External Ridge comparisons must state equivalent objective and penalty scaling explicitly. Do not change statgpu's objective merely to match an external framework.

`RidgeCV` already selects alpha and performs a final `Ridge(..., compute_inference=...)` refit. #127 may change the locality/provenance of that final inference only; it must not change fold generation, candidate grid, scoring, selection, tie behavior, or final fit semantics.

## 5. Two-state architecture: numerical state and reporting state

This separation is a hard design requirement.

### 5.1 Backend-native numerical inference state

Keep the numerical working state on the selected backend through:

- post-formula design transfer;
- weight application;
- residual construction;
- Gram/bread construction;
- inverse/pseudoinverse;
- covariance meat/bread calculations;
- BSE and statistic calculation;
- normal/Student-t probability and quantile evaluation;
- confidence-interval endpoints.

A backend-neutral numerical-state object may replace or supplement `GaussianFitState`. It must not be typed or implemented as NumPy-only storage.

Requirements:

- create intercept columns on the selected backend;
- use backend-native weight operations;
- preserve dtype and concrete Torch/CUDA device;
- use fit metadata for intercept penalty policy instead of scanning a GPU design on CPU when possible;
- catch only recognized rank/linear-algebra failures for pseudoinverse recovery;
- OOM, device, dtype, shape, programming, and contract errors remain fatal.

### 5.2 Established reporting state

Do **not** solve #127 by permanently changing legacy reporting attributes to CuPy/Torch arrays.

After numerical inference completes, populate the established representation expected by current result/reporting code. This may include NumPy snapshots for attributes such as:

- `_X_design` where retained for reporting/diagnostics;
- `_y`;
- `_resid`;
- `_params`;
- `_bse`;
- `_tvalues` / `_zvalues`;
- `_pvalues`;
- `_conf_int`;
- `_inference_result`.

The implementation may instead refactor reporting consumers to accept backend-neutral data, but that would enlarge public/adjacent scope and requires separate review justification. The default #127 design is therefore: **native numerical state first, established reporting snapshot second**.

Existing downstream reporting/fit-statistic behavior must remain valid, including where exposed:

- `summary()`;
- `rsquared` / adjusted R-squared;
- `fvalue` / F p-value;
- `llf`;
- `aic`;
- `bic`.

A pre-existing small public coefficient/intercept snapshot is not by itself evidence of numerical fallback. The prohibited behavior is using host-side full design/residual/covariance/statistic state to perform numerical GPU inference that the public contract says is running on CuPy/Torch.

### 5.3 Formula boundary

Patsy/dataframe formula parsing may remain CPU-side. After the resulting model matrix and aligned side arrays are transferred to the selected backend, numerical fit/inference must not silently return to NumPy before the reporting boundary.

Do not add formula support to `RidgeCV` or any other currently non-formula-facing consumer as part of #127.

## 6. Backend-native covariance and linear algebra

Maintain the existing covariance surface:

- `nonrobust`;
- `hc0`;
- `hc1`;
- `hc2`;
- `hc3`;
- `hac`.

For all maintained numerical backends:

- compute Gram/bread and Ridge penalty additions natively;
- preserve existing rank/pseudoinverse semantics;
- keep leverage, scores, HAC lag accumulation, and covariance working arrays native;
- do not introduce an `n x n` hat matrix;
- preserve concrete Torch device placement;
- do not broaden exception recovery.

Before deleting/bypassing any existing `LinearRegression` covariance or HAC helper, prove whether it is active and whether it owns behavior such as mixed-precision HAC selection. Code unification is not a goal; statistical and device behavior are.

## 7. Backend-native distribution inference

Estimator-level direct `scipy.stats.t` / `scipy.stats.norm` numerical inference must be replaced by the maintained inference-distribution infrastructure.

Requirements:

- inference backend matches executed numerical backend;
- concrete Torch device is preserved;
- normal/Student-t p-values and critical values use shared maintained distribution logic;
- no estimator-specific SciPy fallback for explicit CuPy/Torch inference.

Existing CPU LUT/cache construction internal to the shared distribution subsystem is not automatically estimator fallback. It is acceptable only if user statistic arrays are not transferred to CPU for the actual evaluation and the numerical result remains on the requested backend/device until reporting conversion.

If a required shared distribution primitive itself moves the actual statistic vector to CPU, treat that as a blocking inference-layer defect.

## 8. Multi-target and synchronization policy

Correctness/locality precede vectorization.

A per-target loop is acceptable when:

- each target remains on the selected backend;
- no target-by-target full-array GPU-to-CPU-to-GPU round trip occurs;
- output shape/order remains unchanged;
- scalar synchronization inside the loop is audited and bounded.

Vectorize only when numerical equivalence and memory behavior are demonstrated.

## 9. Provenance contract

Tests and physical artifacts must distinguish:

- requested backend;
- executed fit backend;
- executed numerical-inference backend;
- concrete execution device;
- reporting representation.

Prefer internal/additive metadata over changing public result shapes. Input type alone is not proof of executed backend.

## 10. Required hosted regression matrix

### 10.1 Golden statistical regression

Freeze ordinary-regime results for:

- nonrobust;
- HC0-HC3;
- HAC;
- weighted/unweighted;
- intercept/no-intercept;
- Ridge/L2;
- scalar/multi-target;
- full-rank/rank-deficient.

### 10.2 Inference result/reporting contract

For maintained consumers verify applicable:

- `_inference_result`;
- `_params`;
- `_bse`;
- `_tvalues` or `_zvalues`;
- `_pvalues`;
- `_conf_int`;
- `summary()`;
- covariance type and df metadata;
- `rsquared`, adjusted R-squared, F statistic/F p-value, LLF, AIC, and BIC where currently exposed.

The migration must not make these fail merely because the numerical path became backend-native.

### 10.3 Three-backend execution matrix

For NumPy, CuPy, and Torch verify:

- covariance;
- BSE;
- statistic;
- p-value;
- CI;
- df;
- actual fit/inference backend and concrete device;
- no silent fallback.

Use Torch CPU hosted coverage where useful; final CUDA evidence remains physical.

### 10.4 Shared-mixin non-L2 regression guards

When `_inference_mixin.py` changes, add focused guards showing existing L1/ElasticNet/SCAD/MCP dispatch is not unintentionally redirected through the L2 Gaussian helper and that documented inference/estimation-only behavior remains unchanged.

This is regression protection, not scope expansion.

### 10.5 Precision matrix

Cover:

- small residual df;
- Student-t df=1 and df=2;
- central probabilities;
- extreme but float64-representable tails;
- normal-tail cases;
- CI critical values.

Use the strongest available analytic/reference implementation. Precision failure is blocking.

### 10.6 External alignment

Use:

- analytic closed forms where available;
- pre-migration NumPy/SciPy behavior as an ordinary-regime migration oracle;
- statsmodels OLS/WLS for matched nonrobust/HC/HAC definitions;
- analytic/matched Ridge covariance identities.

Record intercept, weighting, covariance type, HAC lag definition, df convention, Ridge penalty mapping, and tolerance.

### 10.7 Formula regression

For formula-facing consumers verify intercept, categorical/reference behavior, interactions/transforms already supported, missing-row alignment, feature names/order, and array/formula numerical agreement.

### 10.8 RidgeCV final-refit regression

`RidgeCV` is a confirmed consumer because it selects alpha and constructs a final `Ridge(..., compute_inference=...)` estimator.

Prove:

- fold/candidate semantics unchanged;
- scores and selected alpha unchanged;
- refit device unchanged;
- final coefficients/intercept unchanged within tolerance;
- final inference executes on the intended backend;
- failed final refit remains transactional.

If inventory finds another Gaussian/L2 CV final-refit consumer, add it to the capability table and choose representative coverage.

### 10.9 No-host-transfer and failure tests

For explicit GPU execution prove:

- no full design/residual/covariance/statistic host conversion before numerical inference completes;
- estimator-level SciPy inference is not invoked;
- unsupported operations fail closed;
- rank recovery does not swallow unrelated errors;
- Torch concrete device is preserved;
- failed inference does not leave misleading partial reporting state.

Behavioral/instrumented tests are primary evidence; source-text guards alone are insufficient.

## 11. Physical GPU validator

Add a dedicated maintained validator, for example:

`dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py`

Finalize and review the validator contract **before** canonical physical evidence is collected.

Required artifact fields:

- schema version;
- exact `git_sha`;
- clean-worktree proof;
- explicit `--expected-sha`;
- explicit `--validation-tier`;
- requested backend;
- executed fit backend;
- executed inference backend;
- concrete CUDA device;
- Python/CuPy/Torch versions;
- GPU model;
- case/covariance identity;
- maximum covariance/BSE/statistic/p-value/CI errors;
- rank/full-rank disposition;
- weighted/multi-target disposition;
- no-silent-fallback result;
- host-transfer/provenance result;
- overall status.

Physical matrix includes CuPy CUDA and Torch CUDA with at least:

- nonrobust;
- representative HC;
- HAC;
- Ridge/L2;
- weighted inference;
- rank-deficient inference;
- multi-target inference;
- small-df/tail inference;
- one `RidgeCV` final-refit inference case.

### 11.1 Validator freeze rule

Canonical evidence is tied to both numerical implementation and validator acceptance contract.

If validator code changes only in non-semantic reporting prose, document that fact. If its case matrix, acceptance thresholds, provenance checks, backend/device checks, or pass/fail logic changes after a run, rerun affected canonical evidence on an exact clean candidate head.

Historical artifacts generated under an older validator contract are historical evidence only; they do not prove the newer contract.

## 12. Performance boundary

#127 makes no GPU speedup claim.

Measure enough to establish:

- no full-design/residual device-to-host inference transfer;
- no pathological synchronization loop;
- no new dense `n x n` materialization;
- no obvious regression caused solely by the migration.

A user-facing speed/crossover claim activates the full synchronized benchmark gate and machine-readable performance artifacts.

## 13. Documentation

Update EN first, CN second where applicable:

- linear-model inference/device docs;
- Ridge/penalized Gaussian docs when execution behavior changes;
- device-and-memory docs if current wording implies CPU inference;
- root/EN/CN changelogs.

Document the boundary precisely:

> Numerical covariance and reference-distribution inference are backend-native; established reporting attributes/results may take a final NumPy snapshot after numerical inference completes.

Do not claim every repository inference implementation is backend-native.

## 14. Implementation and review sequence

### Phase 1 — inventory and golden freeze

- resolve exact consumer matrix and finalize capability table;
- freeze public/reporting outputs and ordinary statistical behavior;
- identify active/dead duplicate helpers;
- record direct/CV Ridge scaling;
- identify adjacent non-L2 mixin branches requiring regression guards.

### Phase 2 — native numerical state

- introduce/reuse backend-native inference working state;
- keep design/residual/weights/linear algebra native;
- preserve explicit device provenance.

### Phase 3 — covariance and distribution

- migrate bread/meat/BSE/statistics;
- route normal/t inference through shared distribution backend;
- preserve rank, HAC, Ridge, and precision behavior.

### Phase 4 — reporting boundary

- convert only after numerical inference;
- repopulate established reporting attributes/result container;
- verify summary and downstream fit-statistic properties.

### Phase 5 — consumers and CV

- `LinearRegression`;
- bounded Gaussian/L2 penalized path and `Ridge`;
- `RidgeCV` final refit;
- any additional consumer explicitly added during Phase 1;
- non-L2 shared-mixin regression guards.

### Phase 6 — hosted validation

- three-backend matrix;
- analytic/statsmodels alignment;
- objective-scale regression;
- formula/reporting/CV transaction tests;
- no-host-transfer/failure guards;
- full maintained hosted CI.

### Phase 7 — implementation review/fix

Run `.claude/skills/code-review.md` in auto-fix mode until:

- CRITICAL = 0;
- HIGH = 0;
- relevant actionable MEDIUM = 0, or a remaining MEDIUM is explicitly bounded as a non-blocking follow-up and does not conceal a completion gate.

### Phase 8 — validator freeze and physical acceptance

- review validator/negative controls first;
- run exact clean-head CuPy + Torch CUDA acceptance;
- persist canonical evidence.

### Phase 9 — post-physical re-review

If production numerical code, validator case/threshold/provenance logic, or pass/fail semantics changed after physical validation, rerun affected evidence. Evidence/docs-only changes must prove that none of those acceptance inputs changed.

### Phase 10 — docs and final gates

Synchronize docs/changelogs, rerun hosted gates, and produce the hard-exit report.

## 15. Completion criteria

#127 is `COMPLETE` only when:

- exact consumer graph and capability table are final;
- numerical Gaussian inference stays on the selected NumPy/CuPy/Torch backend;
- explicit GPU inference has no silent numerical NumPy/SciPy fallback;
- established reporting attributes/results still work after final conversion;
- nonrobust/HC0-HC3/HAC behavior passes;
- Ridge/intercept/weight/objective-scale behavior passes;
- scalar/multi-target/rank-deficient behavior passes;
- small-df/extreme-tail precision passes;
- formula-facing behavior remains compatible;
- non-L2 shared-mixin branches remain unchanged;
- RidgeCV and any additional confirmed CV final-refit consumers remain statistically/transactionally unchanged except for backend-local inference execution;
- actual fit/inference backend/device is auditable;
- hosted/external gates pass;
- exact clean-head CuPy and Torch CUDA acceptance passes;
- canonical evidence matches the validator contract that defines acceptance;
- EN/CN documentation is synchronized;
- final review has no unresolved CRITICAL/HIGH/relevant actionable MEDIUM findings.

If local work is complete and only physical GPU or remote external evidence remains, use `PARTIAL_REMOTE_PENDING`. Missing local correctness, backend, inference, reporting, formula, precision, CV, or provenance evidence is not eligible for that status.

## 16. Explicit non-goals

- no repository-wide inference rewrite;
- no removal of every SciPy import from `linear_model`;
- no redesign of all reporting/result containers;
- no new loss/penalty/solver behavior;
- no new CV selection/path behavior;
- no new formula surface;
- no sparse-input work;
- no penalized-multinomial work;
- no mandatory multi-target vectorization;
- no GPU speedup claim without separate synchronized evidence.

## 17. Successor sequencing

After #127:

1. #105 — systematic linear/GLM inference benchmark and canonical dashboard evidence;
2. #108 — Panel canonical estimator/covariance evidence using fresh or source/validator-contract-matched provenance rather than automatic reuse of the disputed 0.2.5 final-release artifact chain;
3. feature lanes may proceed independently when they do not compete for the same backend/inference surface:
   - #94 -> #95 for survival;
   - #96 -> #98 for multinomial;
   - #97 as the shared sparse-array foundation;
4. #114, #117, and #118 remain bounded performance/provenance follow-ups unless new evidence raises their severity.
