# Gaussian Linear-Model Backend-Native Inference Plan

Issue: #127  
Baseline release: `0.2.5`  
Baseline `master`: `84f8bc7e17f66466b3a325cbb007b6cb41843821`  
Planning review status: **REVIEW_CLEAN**  
Production implementation status: **implemented; final acceptance pending**

## 1. Goal

Migrate maintained Gaussian linear-model inference so covariance, standard errors, test statistics, p-values, and confidence intervals execute on the selected NumPy/CuPy/Torch numerical backend without silent full-design/residual fallback to NumPy/SciPy.

This is an execution-locality and provenance repair. It does not redefine established Gaussian/Ridge statistics, fitting objectives, penalties, CV selection, formula semantics, or public reporting results.

A NumPy reporting snapshot may remain after numerical inference where required by the established reporting API. The implementation must therefore separate **backend-native numerical inference state** from **established reporting state**.

## 2. Impact classification and capability decisions

Active axes:

- backend/device locality and failure behavior;
- inference and numerical precision;
- public reporting compatibility;
- formula regression compatibility for formula-facing consumers;
- Ridge/L2 objective-scale preservation;
- CV final-refit inference where the repaired path is consumed;
- physical-GPU validation and provenance;
- documentation/evidence artifacts.

Inactive axes:

- new losses or penalties;
- solver redesign;
- new model families;
- new CV selection/path algorithms;
- sparse-input support;
- repository-wide SciPy removal;
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
| `PenalizedLinearRegression`, squared-error + L2 | `three-backend` | `supported` through maintained penalized CV surface | `supported` when enabled | `supported` | `not-performance-sensitive` |

A newly discovered public consumer must receive an explicit capability decision before implementation. `planned` is not a valid completion state for an already maintained backend or inference capability.

### 2.2 Adjacent shared-mixin boundary

`statgpu/linear_model/penalized/_inference_mixin.py` also dispatches non-L2 inference. #127 must not accidentally reroute or redefine:

- L1/debiased inference;
- ElasticNet/debiased inference;
- SCAD/MCP oracle or bootstrap inference;
- existing documented estimation-only/failure behavior.

These paths are regression surfaces when the shared mixin is edited; they are not new #127 feature scope.

## 3. Mandatory consumer and data-lifecycle inventory

Before production edits, record the exact consumer graph and array lifecycle.

Known locations include:

1. `statgpu/linear_model/_gaussian_inference.py`;
2. `statgpu/linear_model/wrappers/_linear.py`;
3. `statgpu/linear_model/penalized/_inference_mixin.py`;
4. `statgpu/linear_model/wrappers/_ridge.py`;
5. `statgpu/linear_model/cv/_ridge_cv.py` for final-refit ownership.

For each public consumer record:

- requested device/backend;
- executed fit backend and concrete device;
- where design, residuals, parameters, scale, and weights are created;
- whether backend-native fit buffers already exist;
- every `_to_numpy()`, `np.asarray()`, or equivalent host boundary before numerical inference completion;
- covariance implementation reached;
- distribution implementation reached;
- final reporting conversion;
- formula and sample-weight behavior;
- scalar versus multi-target behavior;
- CV final-refit ownership where applicable.

The scope is the resolved Gaussian consumer graph, not every `scipy.stats` import under `statgpu/linear_model`.

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
- RidgeCV folds, candidate alphas, scoring, selected alpha, final-refit coefficients, and transactionality.

Historical output is not frozen when it is a demonstrated precision defect. Small-df and extreme-tail behavior follows maintained distribution precision requirements.

### 4.1 Ridge/L2 objective and penalty mapping

#127 must not change the fit objective or regularization convention.

Current Ridge direct/CV paths use the repository average-loss convention and the equivalent `n_eff * alpha` contribution in unnormalized normal equations, with `n_eff` equal to the declared sample/weight normalization. Inference bread/penalty construction must preserve the same mapping.

External Ridge comparisons must state equivalent objective and penalty scaling explicitly. Do not alter statgpu's objective merely to match an external framework.

`RidgeCV` already selects alpha and performs a final `Ridge(..., compute_inference=...)` refit. #127 may change only the locality/provenance of that final inference; it must not change fold generation, candidate grid, scoring, selection, tie behavior, or final fit semantics.

## 5. Two-state architecture

### 5.1 Backend-native numerical inference state

Keep numerical working state on the selected backend through:

- post-formula design transfer;
- weight application and residual construction;
- Gram/bread construction and inverse/pseudoinverse;
- covariance meat/bread calculations;
- BSE and statistic calculation;
- normal/Student-t probability and quantile evaluation;
- confidence-interval endpoints.

Requirements:

- numerical state is not NumPy-only;
- intercept columns and weighted designs are backend-native;
- dtype and concrete Torch/CUDA device are preserved;
- fit metadata is preferred to CPU scanning for intercept-penalty decisions;
- only recognized rank/linear-algebra failures trigger pseudoinverse recovery;
- OOM, device, dtype, shape, programming, and contract errors remain fatal.

### 5.2 Established reporting state

Do **not** solve #127 by permanently changing legacy reporting attributes to CuPy/Torch arrays.

After numerical inference completes, populate the established representation expected by current reporting code. Final snapshots may include, where currently retained:

- `_X_design`;
- `_y`;
- `_resid`;
- `_params`;
- `_bse`;
- `_tvalues` / `_zvalues`;
- `_pvalues`;
- `_conf_int`;
- `_inference_result`.

Existing reporting and fit-statistic behavior must remain valid, including where exposed:

- `summary()`;
- `rsquared` / adjusted R-squared;
- `fvalue` / F p-value;
- `llf`;
- `aic`;
- `bic`.

A small coefficient/intercept reporting snapshot is not by itself numerical fallback. The forbidden path is host-side full design/residual/covariance/statistic computation before GPU numerical inference completes.

### 5.3 Formula boundary

Patsy/dataframe parsing may remain CPU-side. After aligned design/side arrays move to the selected backend, numerical fit/inference must not silently return to NumPy before the reporting boundary.

Do not add formula support to `RidgeCV` or any currently non-formula-facing consumer in #127.

## 6. Covariance and linear algebra contract

Maintain:

- `nonrobust`;
- `hc0`;
- `hc1`;
- `hc2`;
- `hc3`;
- `hac`.

For maintained numerical backends:

- compute Gram/bread and Ridge penalty additions natively;
- preserve existing rank/pseudoinverse semantics;
- keep leverage, scores, HAC accumulation, and covariance working arrays native;
- never construct an `n x n` hat matrix;
- preserve concrete Torch device placement;
- do not broaden exception recovery.

Before deleting or bypassing `LinearRegression` covariance/HAC helpers, prove whether they are active and whether they own semantics such as mixed-precision HAC selection. Code unification is not a goal.

## 7. Distribution inference contract

Estimator-level direct `scipy.stats.t` / `scipy.stats.norm` numerical inference must route through maintained inference-distribution infrastructure.

Requirements:

- inference backend matches executed numerical backend;
- concrete Torch device is preserved;
- normal/Student-t p-values and critical values use shared maintained logic;
- no estimator-specific SciPy fallback for explicit CuPy/Torch inference.

CPU LUT/cache construction internal to the shared distribution subsystem is not automatically estimator fallback. It is acceptable only if actual user statistic arrays are not moved to CPU for evaluation and numerical results remain on the requested backend/device until reporting conversion.

If a required shared primitive moves the actual statistic vector to CPU, treat that as a blocking inference-layer defect.

## 8. Multi-target and synchronization policy

A per-target loop is acceptable when:

- each target remains on the selected backend;
- no target-by-target full-array GPU-to-CPU-to-GPU round trip occurs;
- output shape/order remains unchanged;
- scalar synchronization is audited and bounded.

Vectorize only after numerical equivalence and memory behavior are demonstrated.

## 9. Provenance contract

Tests and physical artifacts must distinguish:

- requested backend;
- executed fit backend;
- executed numerical-inference backend;
- concrete execution device;
- reporting representation.

Prefer additive/internal metadata over public result-shape changes. Input type alone is not proof of executed backend.

## 10. Required hosted regression matrix

### 10.1 Statistical and reporting regression

Cover ordinary-regime:

- nonrobust, HC0-HC3, HAC;
- weighted/unweighted;
- intercept/no-intercept;
- Ridge/L2;
- scalar/multi-target;
- full-rank/rank-deficient.

Verify applicable reporting fields and methods:

- `_inference_result`, `_params`, `_bse`, `_tvalues`/`_zvalues`, `_pvalues`, `_conf_int`;
- covariance/df metadata;
- `summary()`;
- R-squared/adjusted R-squared;
- F statistic/F p-value;
- LLF/AIC/BIC.

### 10.2 Three-backend execution

For NumPy, CuPy, and Torch verify:

- covariance, BSE, statistic, p-value, CI, df;
- actual fit/inference backend and concrete device;
- no silent fallback.

Use Torch CPU hosted coverage where useful; final CUDA evidence remains physical.

### 10.3 Shared-mixin non-L2 guards

When `_inference_mixin.py` changes, prove existing L1/ElasticNet/SCAD/MCP dispatch is not unintentionally redirected through the L2 Gaussian helper and documented behavior remains unchanged.

### 10.4 Precision

Cover:

- small residual df;
- Student-t df=1 and df=2;
- central probabilities;
- extreme but float64-representable tails;
- normal-tail cases;
- CI critical values.

Precision failure is blocking.

### 10.5 External alignment

Use analytic identities where available, pre-migration NumPy/SciPy as an ordinary-regime migration oracle, statsmodels OLS/WLS for matched nonrobust/HC/HAC definitions, and matched/analytic Ridge covariance identities.

Record intercept, weighting, covariance type, HAC lag definition, df convention, Ridge penalty mapping, and tolerances.

### 10.6 Formula regression

For formula-facing consumers verify intercept handling, categorical/reference behavior, supported interactions/transforms, missing-row alignment, feature names/order, and array/formula agreement.

### 10.7 RidgeCV final-refit regression

`RidgeCV` is a confirmed consumer because it constructs a final `Ridge(..., compute_inference=...)` estimator.

Prove:

- fold/candidate/scoring semantics unchanged;
- selected alpha unchanged;
- refit device unchanged;
- final coefficients/intercept unchanged within tolerance;
- final inference executes on intended backend;
- failed final refit remains transactional.

If Phase 1 finds another Gaussian/L2 CV final-refit consumer, add it to the capability table and representative matrix.

### 10.8 No-host-transfer and failure tests

For explicit GPU execution prove:

- no full design/residual/covariance/statistic host conversion before numerical inference completes;
- estimator-level SciPy numerical inference is not invoked;
- unsupported operations fail closed;
- rank recovery does not swallow unrelated errors;
- Torch concrete device is preserved;
- failed inference does not leave misleading partial reporting state.

Behavioral/instrumented tests are primary evidence; source-text guards alone are insufficient.

## 11. Physical GPU validator

Add a maintained validator such as:

`dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py`

Finalize and review the validator contract **before** canonical physical evidence is collected.

Required artifact fields:

- schema version;
- exact `git_sha` and clean-worktree proof;
- explicit `--expected-sha` and `--validation-tier`;
- requested backend, executed fit backend, executed inference backend;
- concrete CUDA device;
- Python/CuPy/Torch versions and GPU model;
- case/covariance identity;
- maximum covariance/BSE/statistic/p-value/CI errors;
- rank/full-rank and weighted/multi-target disposition;
- no-silent-fallback and host-transfer/provenance results;
- overall status.

Physical matrix includes CuPy CUDA and Torch CUDA with at least nonrobust, representative HC, HAC, Ridge/L2, weighted, rank-deficient, multi-target, small-df/tail, and one `RidgeCV` final-refit inference case.

### 11.1 Validator freeze rule

Canonical evidence is tied to both numerical implementation and validator acceptance contract.

If validator changes are purely non-semantic reporting prose, document that. If case matrix, thresholds, provenance checks, backend/device checks, or pass/fail logic changes after a run, rerun affected canonical evidence on an exact clean candidate head.

Historical artifacts generated under an older validator contract remain historical evidence only.

## 12. Performance boundary

#127 makes no GPU speedup claim.

Measure enough to establish:

- no full-design/residual device-to-host inference transfer;
- no pathological synchronization loop;
- no new dense `n x n` materialization;
- no obvious regression caused solely by migration.

Any user-facing speed/crossover claim activates the full synchronized benchmark gate and machine-readable performance artifacts.

## 13. Documentation

Update EN first, CN second where applicable:

- linear-model inference/device docs;
- Ridge/penalized Gaussian docs when execution behavior changes;
- device-and-memory docs if current wording implies CPU inference;
- root/EN/CN changelogs.

Document precisely:

> Numerical covariance and reference-distribution inference are backend-native; established reporting attributes/results may take a final NumPy snapshot after numerical inference completes.

Do not claim every repository inference implementation is backend-native.

## 14. Implementation sequence

### Phase 1 — inventory and golden freeze

- finalize consumer graph/capability table;
- freeze statistical/reporting behavior;
- identify active/dead duplicate helpers;
- record direct/CV Ridge scaling;
- identify non-L2 shared-mixin regression guards.

### Phase 2 — native numerical state

- introduce/reuse backend-native inference working state;
- keep design/residual/weights/linear algebra native;
- preserve device provenance.

### Phase 3 — covariance and distribution

- migrate bread/meat/BSE/statistics;
- route normal/t inference through shared distribution backend;
- preserve rank/HAC/Ridge/precision behavior.

### Phase 4 — reporting boundary

- convert only after numerical inference;
- populate established reporting attributes/result container;
- verify summary and fit-statistic properties.

### Phase 5 — consumers and CV

- `LinearRegression`;
- bounded Gaussian/L2 penalized path and `Ridge`;
- `RidgeCV` final refit;
- any additional Phase-1 consumer;
- non-L2 shared-mixin regression guards.

### Phase 6 — hosted validation

- three-backend matrix;
- analytic/statsmodels alignment;
- objective-scale regression;
- formula/reporting/CV transaction tests;
- no-host-transfer/failure guards;
- maintained hosted CI.

### Phase 7 — implementation review/fix

Run `.claude/skills/code-review.md` in auto-fix mode until:

- CRITICAL = 0;
- HIGH = 0;
- relevant actionable MEDIUM = 0, or any remaining MEDIUM is explicitly bounded as a non-blocking follow-up and does not conceal a completion gate.

### Phase 8 — validator freeze and physical acceptance

- review validator and negative controls before remote canonical execution;
- run exact clean-head CuPy + Torch CUDA acceptance;
- persist canonical evidence.

### Phase 9 — post-physical re-review

If production numerical code, validator case/threshold/provenance logic, or pass/fail semantics change after physical validation, rerun affected evidence. Evidence/docs-only changes must prove those acceptance inputs did not change.

### Phase 10 — docs and final gates

Synchronize docs/changelogs, rerun hosted gates, and produce the hard-exit report.

## 15. Completion criteria

#127 is `COMPLETE` only when:

- consumer graph and capability table are final;
- numerical Gaussian inference stays on selected NumPy/CuPy/Torch backend;
- explicit GPU inference has no silent numerical NumPy/SciPy fallback;
- established reporting attributes/results still work after final conversion;
- nonrobust/HC0-HC3/HAC pass;
- Ridge/intercept/weight/objective-scale behavior passes;
- scalar/multi-target/rank-deficient behavior passes;
- small-df/extreme-tail precision passes;
- formula-facing behavior remains compatible;
- non-L2 shared-mixin branches remain unchanged;
- RidgeCV and additional confirmed CV final-refit consumers remain statistically/transactionally unchanged except for backend-local inference execution;
- actual fit/inference backend/device is auditable;
- hosted/external gates pass;
- exact clean-head CuPy and Torch CUDA acceptance passes;
- canonical evidence matches the validator contract defining acceptance;
- EN/CN docs are synchronized;
- final review has no unresolved CRITICAL/HIGH/relevant actionable MEDIUM findings.

If local work is complete and only physical GPU or remote external evidence remains, use `PARTIAL_REMOTE_PENDING`. Missing local correctness, backend, inference, reporting, formula, precision, CV, or provenance evidence is not eligible for that status.

## 16. Explicit non-goals

- no repository-wide inference rewrite;
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

## 18. Planning review closure

Planning review used `.claude/skills/code-review.md` in auto-fix style against the actual branch diff and adjacent implementation contracts.

Fixed during the plan review/fix loop:

- release-evidence wording that initially overclaimed the 0.2.5 final-validator chain;
- stale/broken Stage-C plan filename;
- cross-document provenance-caveat inconsistency;
- missing capability-decision matrix and insufficient Ridge/RidgeCV objective-scale/final-refit specification;
- an unsafe design ambiguity that could have made legacy reporting state permanently GPU-native;
- missing regression protection for non-L2 branches sharing `_inference_mixin.py`.

Final planning review result:

- CRITICAL: **0 open**;
- HIGH: **0 open**;
- relevant actionable MEDIUM: **0 open**;
- changed scope: planning/docs only; no production numerical source changed at planning closure;
- branch CI: no push-triggered workflow runs were present during the planning review, so no CI result was claimed;
- next executable step at planning closure was #127 Phase 1 consumer inventory and golden freeze.

## 19. Production implementation status — 2026-08-28

The production implementation is now present on `agent/post-v0.2.5-next-phase-plan` and is being validated through draft PR #129.

Implemented and review-fixed before final acceptance:

- backend-native Gaussian working state and covariance/inference routing for NumPy/CuPy/Torch;
- final reporting conversion only after numerical inference;
- Gaussian/L2 public consumer routing, including `LinearRegression`, `Ridge`, bounded penalized squared-error L2 paths, and `RidgeCV` final refit;
- fail-closed executed-backend provenance instead of an implicit NumPy fallback;
- maintained shared normal/Student-t reference-distribution routing, including stable df=1/df=2 extreme-tail formulas;
- Ridge/L2 `n_eff * alpha` inference mapping for ordinary and weighted fits;
- representative hosted coverage for all covariance families, weighted/robust, rank-deficient, multi-target, formula, statsmodels alignment, Torch float32, non-L2 delegation, and no-host-transfer behavior;
- a focused PR CI workflow;
- an exact-SHA physical validator covering CuPy/Torch backend/device provenance, clean-tree state, representative covariance/Ridge/weighted/rank/multi-target/small-df cases, numerical error fields, and a `RidgeCV` final-refit case;
- synchronized root/English/Chinese unreleased changelog entries and user-facing inference/device documentation.

Review-fix findings already closed include the df=2 extreme-tail overflow in the generic t path, invalid backend-provenance fallback, a false SciPy extreme-tail test oracle, insufficient physical-validator matrix/provenance fields, and pseudo-coverage that monkeypatched the public router rather than its shared delegate.

Remaining acceptance gates are deliberately not claimed as passed:

- draft PR #129 hosted workflows must complete successfully on the final source head;
- the final complete diff must receive a fresh review with no unresolved CRITICAL/HIGH/relevant actionable MEDIUM findings;
- the physical validator contract must remain frozen after review;
- exact clean-head CuPy and Torch CUDA acceptance must run and produce canonical evidence matching that validator contract.

Until the hosted gates are green and the final fresh review is clean, this branch is **not** eligible for `PARTIAL_REMOTE_PENDING`. Once those local/hosted gates are closed, if only exact physical CUDA evidence remains, the correct hard-exit status is `PARTIAL_REMOTE_PENDING`, not `COMPLETE`.
