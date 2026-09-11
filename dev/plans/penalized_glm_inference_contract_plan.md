# Penalized GLM inference contract repair plan

Status: DESIGN AUDIT — PASS 2
Target release: 0.2.6 (current published release: 0.2.5)

## 1. Topology and purpose

This is stacked work on top of the unmerged `.claude/skills` workflow change.

- workflow parent: `chore/optimize-claude-skills-workflow`
- workflow parent SHA: `2d38f2c5108bfe53f52d389271572240d319496a`
- implementation branch: `fix/penalized-glm-inference-contract`
- production baseline before this plan: exactly the workflow-parent SHA above
- upstream `master` at plan start: `e364717bd62ff84d260eab4e0b7020e26a98caf0`

The purpose is to repair and document the already-partial penalized-GLM inference capability without inventing an unrelated new statistical algorithm. In particular, this plan must not silently turn the task into implementation of debiased inference for non-Gaussian L1/ElasticNet models.

The task also serves as a real integration test of the strengthened `new-module-dev` and `code-review` skills. Any workflow omission or over-broad gate discovered while carrying out this plan should be recorded and, if material, fixed on the workflow parent before relying on it.

## 2. Change classification and active axes

Change type:

- existing capability reconciliation / public contract repair;
- API behavior repair for inference method selection and fitted provenance;
- CV final-refit inference capability for already-tunable `PenalizedGLM_CV`;
- docs/release-boundary update.

Active axes:

- public API / compatibility;
- inference;
- resampling;
- backend/device/fallback;
- CV final refit;
- formula/model-matrix preservation;
- docs/evidence.

Inactive by default:

- new loss/penalty/solver algorithms;
- performance benchmark claims;
- new non-Gaussian debiasing theory;
- family-aware/parametric GLM bootstrap.

## 3. Phase-0 consumer graph

Canonical shared estimator:

- `PenalizedGeneralizedLinearModel`.

Typed scalar-response GLM consumers:

- `PenalizedLinearRegression`;
- `PenalizedLogisticRegression`;
- `PenalizedPoissonRegression`;
- `PenalizedGammaRegression`;
- `PenalizedInverseGaussianRegression`;
- `PenalizedNegativeBinomialRegression`;
- `PenalizedTweedieRegression`.

Meta-estimator / tuning consumer:

- `PenalizedGLM_CV`.

Other subclasses that must not regress accidentally:

- `PenalizedRobustRegression`;
- `PenalizedQuantileRegression`;
- `PenalizedCoxPHModel` (already explicitly estimation-only for this class);
- group-penalty models, whose installed contract already makes group inference estimation-only.

Specialized wrappers built on `PenalizedLinearRegression`, including `Ridge`, `Lasso`, and `ElasticNet`, are preservation consumers. Their established Gaussian inference contracts must remain intact.

Runtime compatibility consumers:

- `statgpu.linear_model._penalized_inference_api_contract` for sparse-Gaussian inference aliases and fit-device behavior;
- existing post-selection/nodewise compatibility layers installed from `statgpu.linear_model`.

Formula consumer:

- `PenalizedGeneralizedLinearModel.fit(formula=..., data=...)`.

Maintained documentation consumers:

- EN/CN generalized-linear-model pages;
- EN/CN inference-modes guide;
- EN/CN cross-validation guide where `PenalizedGLM_CV` inference is claimed;
- implemented-method/support matrices if their claims are affected;
- root/EN/CN changelog surfaces for the 0.2.6-targeted migration.

## 4. Current capability matrix

| Loss / family | Penalty | Current public request/default | Actual runtime inference | Current status |
| --- | --- | --- | --- | --- |
| squared_error | L2 | generic/PLR default says `debiased`; Ridge hides method | Gaussian classical/robust Ridge inference | method identity inconsistent on generic/PLR surface; numerical path maintained |
| squared_error | L1 | `debiased`, post-selection aliases, `bootstrap` | debiased / post-selection / residual bootstrap | maintained; preserve |
| squared_error | ElasticNet | `debiased`, post-selection aliases, `bootstrap` | debiased/post-selection; bootstrap helper incorrectly refits L1 | bootstrap refit identity inconsistent |
| squared_error | SCAD/MCP | `oracle` or `bootstrap` | oracle active-set or residual bootstrap, but bootstrap refits L1 | bootstrap refit identity inconsistent; oracle has CPU-refit device caveat |
| non-Gaussian Hessian GLM | L2 / no penalty | typed wrappers default `debiased` | unconditional penalized sandwich reported as `m_estimation` | requested/resolved/reported method mismatch |
| non-Gaussian Hessian GLM | ElasticNet | default `debiased` | full-vector sandwich using only L2 curvature | statistically under-specified for nonsmooth zero coordinates; do not productize as valid full-vector inference |
| non-Gaussian GLM | L1 | `bootstrap` is accepted | Gaussian Lasso residual bootstrap/refit | invalid family/refit; fail closed |
| non-Gaussian GLM | SCAD/MCP | `oracle` or `bootstrap` | oracle active-set CPU refit; bootstrap is Gaussian L1 | oracle exists but device fallback must be explicit; bootstrap invalid |
| group penalties | group variants | generic surface exists | installed contract raises | estimation-only; preserve |
| penalized Cox class | supported Cox penalties | constructor inherits inference spelling | class override raises for `compute_inference=True` | estimation-only; preserve |
| `PenalizedGLM_CV` | scalar-response penalties | no inference controls at all | inference only happens incidentally for one squared-error L2 fallback; optimized final-refit paths disable it | public CV inference contract missing |

Additional current defects:

- the base validator advertises `Any loss + bootstrap: universal fallback` although the helper is explicitly a Gaussian Lasso residual bootstrap;
- that bootstrap helper ignores analytic weights and `cov_type`, so neither weighted residual bootstrap nor robust-bootstrap inference has a defined current contract;
- non-Gaussian sandwich backend is inferred from the array passed to post-fit inference, while the main fit currently passes original `X/y`; an explicit GPU fit started from NumPy input can therefore perform inference on the wrong backend or encounter heterogeneous arrays;
- public fitted provenance does not consistently expose requested/resolved/reported inference method, inferential target, or tuning/selection conditioning.

## 5. Desired public inference-method contract

### 5.1 Canonical request vocabulary

Add/standardize these public request values:

- `auto` — recommended generic default and the only request that may resolve to another named statistical procedure without being an alias;
- `debiased` — sparse Gaussian L1/ElasticNet only;
- `post_selection_ols` — existing sparse-Gaussian active-set diagnostic/refit contract only;
- `m_estimation` — fixed-penalty inference for supported non-Gaussian smooth L2/no-penalty models;
- `oracle` — SCAD/MCP active-set refit inference when explicitly requested and supported;
- `bootstrap` — unweighted Gaussian residual-bootstrap request only; the reported method remains `residual_bootstrap`.

Historical `cpu_ols` / `gpu_ols` aliases keep their existing one-cycle sparse-Gaussian migration contract.

### 5.2 Default and compatibility rule

Change the generic `PenalizedGeneralizedLinearModel`, `PenalizedLinearRegression`, and typed non-Gaussian penalized wrappers from the misleading default `inference_method="debiased"` to `inference_method="auto"`.

Specialized sparse-Gaussian `Lasso`/`ElasticNet` wrappers may retain their established explicit `debiased` default because it names the actual maintained method for their fixed penalty family.

For L2/no-penalty models, an explicitly supplied historical `inference_method="debiased"` may be accepted for one migration cycle only as a deprecated alias for `auto`, with a user-facing `FutureWarning`. The warning must state that L2 inference was never debiased-Lasso inference and that `auto` (or explicit `m_estimation` for supported non-Gaussian L2) is the canonical replacement. Omitted arguments under the new source signature do not warn.

Do not alias `debiased` to a nonsmooth non-Gaussian ElasticNet path; that row becomes unsupported rather than blessing an invalid approximation.

### 5.3 `auto` resolution

`auto` resolves before fitting, after loss and penalty objects are available:

- squared_error + L2/no penalty:
  - `classical` when `cov_type="nonrobust"`;
  - `sandwich` for supported robust Gaussian covariance;
- squared_error + L1/ElasticNet: `debiased`;
- non-squared Hessian-equipped scalar loss + L2/no penalty: `m_estimation`;
- SCAD/MCP: do **not** silently choose the assumption-heavy oracle procedure; require explicit `oracle` (or a supported Gaussian `bootstrap` request);
- unsupported/nonsmooth rows fail before numerical inference begins.

The resolved method must be public after fit.

### 5.4 Covariance support

For the non-Gaussian `m_estimation` path in this repair:

- `cov_type="nonrobust"` is the model-based penalized-information covariance;
- `cov_type="hc0"` and `"hc1"` are supported sandwich covariance choices;
- HC2/HC3/HAC remain unsupported for this path and fail visibly, matching the actual shared sandwich engine;
- `hac_maxlags` does not imply HAC support for penalized GLMs.

Do not document HC2/HC3/HAC as available merely because those names exist elsewhere in statgpu.

## 6. Inferential targets and fitted provenance

A successful inference-enabled fit publishes:

- `inference_requested_method_` — normalized request (`auto`, `debiased`, etc.);
- `inference_resolved_method_` — concrete method expected from dispatch;
- `inference_method_` — method actually reported by `_inference_result.method`;
- `inference_target_`;
- `penalty_conditioning_`;
- `penalty_selection_adjusted_`.

The `_inference_result.metadata` must carry the same provenance plus numerical backend/device where applicable.

Target vocabulary:

- L2/no-penalty fixed-penalty sandwich/Gaussian inference: `penalized_estimating_equation` for positive penalty and the ordinary unpenalized parameter target when penalty strength is zero;
- debiased sparse Gaussian: `unpenalized_population_coefficient`;
- post-selection OLS/oracle: `active_set_refit_coefficient`;
- residual bootstrap: `penalized_coefficient_distribution` conditional on the declared fixed penalty/refit procedure.

Direct-fit inference uses `penalty_conditioning_="fixed_penalty"`. `penalty_selection_adjusted_` is `None` when no data-driven penalty selection occurred.

## 7. Supported matrix after repair

### 7.1 Supported and productized

- Gaussian L2/no penalty: existing Gaussian inference, NumPy/CuPy/Torch preserved.
- Gaussian L1/ElasticNet: existing debiased and post-selection paths preserved.
- Gaussian residual bootstrap: L1/ElasticNet/SCAD/MCP may be supported only after the helper preserves the estimator's actual Gaussian penalty, l1_ratio, penalty kwargs, intercept, and fixed alpha during every refit; this repair supports only unweighted CPU-executed residual bootstrap with `cov_type="nonrobust"`.
- Non-Gaussian Hessian scalar loss + L2/no penalty: `m_estimation`, with penalty curvature in the bread, native fit backend/device, and public fixed-penalty target/conditioning metadata; supported covariance types are nonrobust/HC0/HC1.
- SCAD/MCP: explicit `oracle` only. If the maintained implementation cannot preserve an explicit GPU backend without silently moving to CPU, GPU oracle requests fail visibly until a backend-native oracle refit exists.

### 7.2 Explicitly unsupported/fail-closed

- non-Gaussian L1 inference;
- non-Gaussian ElasticNet full-vector sandwich/debiased inference;
- non-Gaussian bootstrap through the Gaussian residual-bootstrap helper;
- weighted Gaussian residual bootstrap in this repair;
- Gaussian residual bootstrap with robust/HAC `cov_type` in this repair;
- non-Gaussian M-estimation with HC2/HC3/HAC;
- group-penalty inference;
- penalized-Cox inference in `PenalizedCoxPHModel`;
- any method/loss/penalty combination not in the support matrix.

Error messages must list the actually supported alternatives for the current row rather than recommending `bootstrap` universally.

## 8. Resampling contract

The maintained `bootstrap` request in this repair is **unweighted Gaussian residual bootstrap only**:

- fixed design;
- no `sample_weight`;
- `cov_type="nonrobust"`; robust/HAC covariance requests are rejected rather than silently ignored;
- residual resampling from the fitted Gaussian model;
- same fixed alpha;
- same penalty family and ElasticNet l1_ratio / penalty kwargs;
- same intercept semantics;
- refit selection is repeated inside each bootstrap sample because the penalized estimator itself is refit;
- tuning/CV is not rerun inside bootstrap;
- output is heuristic penalized-estimator bootstrap inference, not selective-inference coverage.

It is not a generic GLM bootstrap. Non-Gaussian calls fail before resampling. Weighted residual bootstrap is also not inferred from analytic weights: a future implementation must choose and document a statistically valid weighted resampling scheme explicitly.

This repair does not add a parametric/family-aware GLM bootstrap, wild bootstrap, or weighted bootstrap.

## 9. Backend/device contract

For supported non-Gaussian L2/no-penalty `m_estimation`:

- numerical inference must execute on the backend selected by the fit (`_selected_backend_name`) and concrete selected device (`_selected_backend_device`), not merely infer a backend from the original input container;
- explicit `device="cuda"` and `device="torch"` must not silently execute inference on NumPy;
- CuPy allocations must occur under the concrete selected device context;
- reporting may snapshot final small arrays to NumPy only after numerical covariance/statistic/p-value/CI work;
- result metadata records `numerical_backend`, `numerical_device`, `reporting_backend="numpy"`, and reporting boundary.

For Gaussian residual bootstrap, the maintained implementation is CPU-executed in this repair. It is supported only when the fit's selected backend is NumPy. Explicit GPU/Torch bootstrap requests fail visibly; `device="auto"` is accepted only when the fit itself selected NumPy.

For oracle inference, the existing CPU refit must not masquerade as backend-native. Until rewritten, explicit GPU/Torch oracle requests fail with a clear unsupported-backend error; an `auto` request may use oracle only when the fit actually selected NumPy, and `auto` does not select oracle by default anyway.

## 10. CV final-refit contract

Add to `PenalizedGLM_CV`:

- `compute_inference=False`;
- `inference_method="auto"`;
- `cov_type="nonrobust"`;
- `hac_maxlags=None` where applicable to the underlying estimator contract.

Rules:

- CV folds/grid/path scoring never compute inference;
- after alpha selection, inference is run exactly once on the full-data final refit if requested;
- the final refit receives the selected alpha and public inference controls;
- if an optimized final-refit path populates coefficients manually, inference-enabled final refit must route through a normal estimator fit (or an equivalently proven inference-safe path) rather than fabricating partial fitted state;
- successful CV inference delegates `_inference_result` and public inference fields from `estimator_`;
- CV fit publishes `penalty_conditioning_="cv_selected_penalty"` and `penalty_selection_adjusted_=False`;
- docs explicitly state that reported standard errors/p-values/CIs condition on the selected alpha and do not adjust for CV tuning uncertainty;
- unsupported direct-fit inference rows remain unsupported after CV selection;
- `loss="cox_ph"` continues to inherit the estimation-only `PenalizedCoxPHModel` inference boundary.

This is final-refit closure, not inference inside the CV selection loop.

## 11. Formula contract

For a representative supported non-Gaussian L2 model, array and formula routes must agree after model-matrix construction for:

- coefficient/intercept ordering;
- bse/statistic/p-value/CI;
- feature names;
- fitted inference provenance.

Formula changes are not otherwise in scope.

## 12. Validation matrix

### Public API / compatibility

- signatures/defaults for generic base, PLR, typed non-Gaussian wrappers, and `PenalizedGLM_CV`;
- `get_params`, `set_params`, clone where claimed;
- explicit old `debiased` + L2 migration warning; omission with new `auto` default does not warn;
- refit invalidates stale fitted inference provenance.

### Statistical/runtime contract

At minimum cover logistic and Poisson L2, plus one additional positive-response family:

- `auto` -> `m_estimation` -> reported `m_estimation`;
- explicit `m_estimation` equivalence to auto;
- explicit incompatible `debiased`/bootstrap fail or migrate exactly as specified;
- `cov_type` nonrobust/HC0/HC1 metadata and target/conditioning fields;
- HC2/HC3/HAC fail visibly for non-Gaussian M-estimation;
- sample weights for supported M-estimation;
- unsupported non-Gaussian L1/ElasticNet fail closed before helper execution;
- Gaussian L1/ElasticNet bootstrap preservation with penalty-preserving refits on unweighted NumPy fits;
- weighted/robust Gaussian bootstrap fails before resampling;
- SCAD/MCP oracle CPU behavior and explicit GPU fail-closed boundary.

### Backend

- hosted NumPy tests for all logic;
- deterministic fake/dispatch tests proving supported `m_estimation` consumes fit-selected backend/device and does not infer from original NumPy input;
- existing CuPy/Torch inference paths preserved;
- physical GPU validator for representative non-Gaussian L2 NumPy-vs-CuPy-vs-Torch coefficient/inference parity and concrete device provenance before remote-full completion.

### CV

- final-refit-only inference for non-Gaussian L2;
- no inference in folds (spy/monkeypatch contract);
- selected alpha reused;
- delegated result/provenance;
- `cv_selected_penalty` + selection-adjustment false;
- unsupported rows fail on final refit without returning stale CV inference state;
- Cox branch remains estimation-only.

### Formula

- representative logistic or Poisson L2 array/formula parity.

### External/statistical baseline

For supported non-Gaussian L2/no-penalty M-estimation, compare a CPU case to an independent NumPy/statsmodels-style Hessian/score calculation after aligning statgpu's average-loss penalty scaling. The invariant is the stated penalized estimating-equation covariance, not agreement with an unpenalized package fit at a mismatched target.

## 13. Documentation

EN first, CN follow. Update only affected surfaces.

Required learner-facing content:

- what parameter the penalized GLM inference targets;
- why smooth L2 allows fixed-penalty M-estimation inference;
- the bread/meat formula and penalty-curvature role;
- model-based (`nonrobust`) versus HC0/HC1 covariance;
- `auto` method resolution table;
- unsupported non-Gaussian L1/ElasticNet inference and why it is not silently approximated;
- Gaussian residual-bootstrap limits (unweighted, NumPy-selected fit, nonrobust only);
- CV final-refit conditioning and unadjusted tuning uncertainty;
- backend behavior and reporting boundary;
- explicit support matrix;
- examples for a non-Gaussian penalized GLM direct fit and CV final-refit inference.

Release wording: current published version is 0.2.5. The repaired behavior is branch/master-targeted for 0.2.6 until an actual release task ships it.

## 14. Pre-implementation audit acceptance

Before production edits, a fresh design audit must confirm:

- the support matrix does not bless nonsmooth ElasticNet full-vector sandwich inference;
- the invalid generic bootstrap is removed/narrowed rather than merely renamed;
- weighted and robust-bootstrap semantics are not guessed from the current helper;
- explicit GPU inference has no silent CPU fallback;
- CV inference is final-refit-only and its tuning uncertainty limitation is public;
- sparse-Gaussian and Ridge specialized wrappers retain their established contracts;
- Cox/group estimation-only boundaries remain intact;
- docs/version wording is consistent with v0.2.5 current and 0.2.6 targeted.

Any blocking design finding updates this plan and triggers a fresh audit before production implementation.

## 15. Skill-validation observations from Phase 0 / design audit

The strengthened workflow has already changed the shape of this task in useful ways:

1. requested/resolved/reported identity exposed the misleading `debiased` defaults immediately;
2. resampling reconnaissance found the nominally generic bootstrap actually rebuilds Gaussian L1;
3. the resampling contract then exposed a second-order design gap: the helper ignores analytic weights and `cov_type`, so the repair must not silently claim weighted or robust bootstrap;
4. the consumer graph found that `PenalizedGLM_CV` has no public inference controls and disables inference on non-Gaussian final refits;
5. backend provenance review found that non-Gaussian sandwich inference currently resolves from the post-fit input container rather than the executed fit backend;
6. consumer enumeration prevented accidental changes to robust/quantile/Cox/group and sparse-Gaussian compatibility contracts;
7. the release-boundary gate verified v0.2.5 is current and 0.2.6 is only a target.

No workflow over-gating has been identified: backend, CV, resampling, formula, and docs are all activated by real consumers or current public claims in this task.
