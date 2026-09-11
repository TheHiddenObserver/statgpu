# Penalized GLM inference contract repair plan

Status: DESIGN AUDIT
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
- new non-Gaussian debiasing theory.

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
- non-Gaussian sandwich backend is inferred from the array passed to post-fit inference, while the main fit currently passes original `X/y`; an explicit GPU fit started from NumPy input can therefore perform inference on the wrong backend or encounter heterogeneous arrays;
- public fitted provenance does not consistently expose requested/resolved/reported inference method, inferential target, or tuning/selection conditioning.

## 5. Desired public inference-method contract

### 5.1 Canonical request vocabulary

Add/standardize these public request values:

- `auto` — recommended generic default and the only request that may resolve to another named statistical procedure without being an alias;
- `debiased` — sparse Gaussian L1/ElasticNet only;
- `post_selection_ols` — existing sparse-Gaussian active-set diagnostic/refit contract only;
- `m_estimation` — fixed-penalty sandwich inference for supported non-Gaussian smooth L2/no-penalty models;
- `oracle` — SCAD/MCP active-set refit inference when explicitly requested and supported;
- `bootstrap` — Gaussian residual-bootstrap request only; the reported method remains `residual_bootstrap`.

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
  - `sandwich` for robust supported Gaussian covariance;
- squared_error + L1/ElasticNet: `debiased`;
- non-squared Hessian-equipped scalar loss + L2/no penalty: `m_estimation`;
- SCAD/MCP: do **not** silently choose the assumption-heavy oracle procedure; require explicit `oracle` (or a supported Gaussian `bootstrap` request);
- unsupported/nonsmooth rows fail before numerical inference begins.

The resolved method must be public after fit.

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
- Gaussian residual bootstrap: L1/ElasticNet/SCAD/MCP may be supported only after the helper preserves the estimator's actual Gaussian penalty, l1_ratio, penalty kwargs, intercept, and fixed alpha during every refit.
- Non-Gaussian Hessian scalar loss + L2/no penalty: `m_estimation`, with penalty curvature in the bread, native fit backend/device, and public fixed-penalty target/conditioning metadata.
- SCAD/MCP: explicit `oracle` only. If the maintained implementation cannot preserve an explicit GPU backend without silently moving to CPU, GPU oracle requests fail visibly until a backend-native oracle refit exists.

### 7.2 Explicitly unsupported/fail-closed

- non-Gaussian L1 inference;
- non-Gaussian ElasticNet full-vector sandwich/debiased inference;
- non-Gaussian bootstrap through the Gaussian residual-bootstrap helper;
- group-penalty inference;
- penalized-Cox inference in `PenalizedCoxPHModel`;
- any method/loss/penalty combination not in the support matrix.

Error messages must list the actually supported alternatives for the current row rather than recommending `bootstrap` universally.

## 8. Resampling contract

The maintained `bootstrap` request in this repair is **Gaussian residual bootstrap only**:

- fixed design;
- residual resampling from the fitted Gaussian model;
- same fixed alpha;
- same penalty family and ElasticNet l1_ratio / penalty kwargs;
- same intercept semantics;
- refit selection is repeated inside each bootstrap sample because the penalized estimator itself is refit;
- tuning/CV is not rerun inside bootstrap;
- output is heuristic penalized-estimator bootstrap inference, not selective-inference coverage.

It is not a generic GLM bootstrap. Non-Gaussian calls fail before resampling.

This repair does not add a parametric/family-aware GLM bootstrap.

## 9. Backend/device contract

For supported non-Gaussian L2/no-penalty `m_estimation`:

- numerical inference must execute on the backend selected by the fit (`_selected_backend_name`) and concrete selected device (`_selected_backend_device`), not merely infer a backend from the original input container;
- explicit `device="cuda"` and `device="torch"` must not silently execute inference on NumPy;
- CuPy allocations must occur under the concrete selected device context;
- reporting may snapshot final small arrays to NumPy only after numerical covariance/statistic/p-value/CI work;
- result metadata records `numerical_backend`, `numerical_device`, `reporting_backend="numpy"`, and reporting boundary.

For Gaussian residual bootstrap, if the maintained implementation is CPU-only, explicit GPU bootstrap requests must fail visibly rather than silently refit on CPU. `device="auto"` may use CPU only if the fit itself selected CPU; no post-fit backend substitution is permitted.

For oracle inference, the existing CPU refit must not masquerade as backend-native. Until rewritten, explicit GPU oracle requests fail with a clear unsupported-backend error.

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
- unsupported direct-fit inference rows remain unsupported after CV selection.

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
- `cov_type` metadata and target/conditioning fields;
- sample weights;
- unsupported non-Gaussian L1/ElasticNet fail closed before helper execution;
- Gaussian L1/ElasticNet/bootstrap preservation and penalty-preserving refits;
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
- unsupported rows fail on final refit without returning stale CV inference state.

### Formula

- representative logistic or Poisson L2 array/formula parity.

### External/statistical baseline

For supported non-Gaussian L2/no-penalty M-estimation, compare a CPU case to an independent NumPy/statsmodels-style Hessian/score sandwich calculation after aligning statgpu's average-loss penalty scaling. The invariant is the stated penalized estimating-equation covariance, not agreement with an unpenalized package fit at a mismatched target.

## 13. Documentation

EN first, CN follow. Update only affected surfaces.

Required learner-facing content:

- what parameter the penalized GLM inference targets;
- why L2 can use fixed-penalty M-estimation sandwich;
- the bread/meat formula and penalty-curvature role;
- `auto` method resolution table;
- unsupported L1/ElasticNet non-Gaussian inference and why it is not silently approximated;
- CV final-refit conditioning and unadjusted tuning uncertainty;
- backend behavior and reporting boundary;
- explicit support matrix;
- examples for a non-Gaussian penalized GLM direct fit and CV final-refit inference.

Release wording: current published version is 0.2.5. The repaired behavior is branch/master-targeted for 0.2.6 until an actual release task ships it.

## 14. Pre-implementation audit acceptance

Before production edits, a fresh design audit must confirm:

- the support matrix does not bless nonsmooth ElasticNet full-vector sandwich inference;
- the invalid generic bootstrap is removed/narrowed rather than merely renamed;
- explicit GPU inference has no silent CPU fallback;
- CV inference is final-refit-only and its tuning uncertainty limitation is public;
- sparse-Gaussian and Ridge specialized wrappers retain their established contracts;
- Cox/group estimation-only boundaries remain intact;
- docs/version wording is consistent with v0.2.5 current and 0.2.6 targeted.

Any blocking design finding updates this plan and triggers a fresh audit before production implementation.

## 15. Skill-validation observations from Phase 0

The strengthened workflow has already changed the shape of this task in useful ways:

1. requested/resolved/reported identity exposed the misleading `debiased` defaults immediately;
2. resampling reconnaissance found the nominally generic bootstrap actually rebuilds Gaussian L1;
3. the consumer graph found that `PenalizedGLM_CV` has no public inference controls and disables inference on non-Gaussian final refits;
4. backend provenance review found that non-Gaussian sandwich inference currently resolves from the post-fit input container rather than the executed fit backend;
5. consumer enumeration prevented accidental changes to robust/quantile/Cox/group and sparse-Gaussian compatibility contracts;
6. the release-boundary gate verified v0.2.5 is current and 0.2.6 is only a target.

No workflow over-gating has been identified yet: backend, CV, resampling, formula, and docs are all activated by real consumers or current public claims in this task.
