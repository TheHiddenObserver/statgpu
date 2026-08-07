# Panel P1 Stage A — shared framework and covariance registry plan

Issue: #93  
Baseline: `master` at `070176fedbc9cae6e55dd8bbb9e389e08744bb30`  
Branch: `agent/panel-p1-stage-a-framework`  
Scope: PR-A only — behavior-preserving shared panel framework refactor

## 1. Stage boundary

Issue #93 explicitly decomposes Tier-1 panel work into at least three PRs. This plan implements **Stage A only**:

1. freeze current numerical/public behavior with golden regression tests;
2. add an internal shared panel base abstraction;
3. centralize covariance dispatch for the existing covariance contracts;
4. migrate existing estimators without changing their statistical definitions or public constructor signatures.

Stage B (Hausman/pooling-F/Breusch-Pagan LM and fit statistics) and Stage C (HC0/HC2/HC3, RandomEffects robust covariance, Driscoll-Kraay, expanded cluster contracts) are intentionally not implemented here.

The purpose of Stage A is to create a safe internal substrate for those later public features without mixing structural refactoring with new econometric definitions.

## 2. Current-source observations

The existing panel package already has substantial shared infrastructure:

- `_formula.py` centralizes formula parsing, missing-row alignment, fixed-effect tokens, and prediction design reconstruction;
- `_utils.py` already contains `PanelSummary`, panel label factorization, numeric/alpha validation, group operations, demeaning, and a shared OLS inference helper used by `BetweenOLS` and `FirstDifferenceOLS`;
- `_covariance.py` already provides one-way cluster, two-way cluster, and HAC covariance functions with backend-native score accumulation after metadata factorization;
- `PanelOLS`, `PooledOLS`, `RandomEffects`, and `FamaMacBeth` still contain estimator-local inference/summary/lifecycle logic;
- all six estimators duplicate some combination of formula-state setup, fitted-state validation, summary construction, feature-name reconstruction, backend conversion, or covariance dispatch.

The refactor must build on these existing helpers rather than replace them with a parallel framework.

## 3. Impact classification

Active axes:

- **Public API compatibility**: constructor signatures, fitted attributes, output array types, summary fields, prediction return types, and exception behavior must remain unchanged;
- **Backend**: shared helpers must preserve NumPy, CuPy, and Torch device/dtype ownership and strict explicit-device behavior;
- **Inference**: covariance, standard errors, test statistics, p-values, and confidence intervals are structurally refactored but numerically frozen;
- **Formula**: shared formula-state/input helpers must preserve intercept, effects-token, missing-row, categorical/interactions, and prediction design behavior;
- **Architecture/maintenance**: duplication removal is the principal goal;
- **Tests**: golden before/after regression coverage is a blocking gate;
- **Docs/changelog**: architecture change and behavior-preservation evidence must be recorded.

Inactive axes:

- CV: panel estimators in this stage are not CV/tuning wrappers;
- loss/penalty/solver: no optimization objective or penalty changes;
- new diagnostics: deferred to Stage B;
- new covariance formulas: deferred to Stage C;
- performance claims: no speedup claim is made. Any incidental transfer reduction is not a release performance claim.

## 4. Capability decisions

| Family | backend | inference | formula | benchmark |
| --- | --- | --- | --- | --- |
| PanelOLS | three-backend | supported, behavior-preserving | supported | no performance claim; physical GPU regression required |
| RandomEffects | three-backend | supported, nonrobust unchanged | supported | no performance claim; physical GPU regression required |
| PooledOLS | three-backend | supported, existing covariance set unchanged | supported | no performance claim; physical GPU regression required |
| BetweenOLS | three-backend | supported, existing covariance set unchanged | supported | no performance claim; physical GPU regression required |
| FirstDifferenceOLS | three-backend | supported, existing covariance set unchanged | supported | no performance claim; physical GPU regression required |
| FamaMacBeth | three-backend | supported, estimator-specific covariance unchanged | supported | no performance claim; physical GPU regression required |

`FamaMacBeth` must **not** be routed through the OLS residual sandwich registry: its covariance is computed from the time series of cross-sectional coefficient estimates and is statistically distinct. It may reuse lifecycle/summary/formula helpers only.

## 5. Golden behavior freeze — before structural migration

Add a deterministic `dev/tests/test_panel_stage_a_golden.py` based on the current master behavior. Use fixed seeds and explicit balanced/unbalanced datasets to record invariants for all six estimators.

For every relevant model freeze:

- `coef_`;
- fitted/predicted values on deterministic evaluation rows;
- residual-dependent public statistics where currently exposed;
- `bse_`, `tvalues_`, `pvalues_`, `conf_int_`;
- covariance matrix if publicly stored;
- `nobs`, `df_resid`, rank/effect metadata where currently exposed;
- existing R-squared field(s);
- summary `to_dict()` content/field names;
- prediction return type (NumPy versus backend-native) as currently documented/implemented.

Coverage matrix:

- `PanelOLS`: no effects, entity FE, time FE, two-way FE; nonrobust/robust/one-way clustered/two-way clustered where currently supported; string labels; balanced and unbalanced panels;
- `RandomEffects`: balanced and unbalanced entity sizes, formula/array parity, variance components and theta;
- `PooledOLS`: nonrobust/robust/clustered/HAC;
- `BetweenOLS`: nonrobust/robust;
- `FirstDifferenceOLS`: nonrobust/robust with and without explicit time sorting;
- `FamaMacBeth`: nonrobust/newey-west and existing backend-native prediction contract.

Use the unchanged current implementation as the trusted baseline; do not hard-code opaque constants when a same-test pre-refactor calculation or analytic OLS definition is clearer. Where exact snapshots are appropriate, tolerances must be strict enough to catch scaling/df changes.

## 6. Shared base design

Add internal `statgpu/panel/_base.py` with `BasePanelModel(BaseEstimator)` and small structured helper types. It is internal in Stage A and is **not** exported as a public API.

The base should provide only statistically neutral lifecycle primitives:

### A. Formula/model-matrix state

A helper wrapping the existing `_prepare_formula_fit()` / `_align_formula_side_array()` behavior should:

- preserve `model_has_intercept` and `support_pipe` as explicit per-model arguments;
- store `_design_info`, `_feature_names`, `_formula_has_intercept` exactly as today;
- align named side arrays (entity/time/cluster/time-index) only through existing formula helpers;
- return fixed-effect metadata without mutating constructor options unless the subclass explicitly applies current behavior (notably `PanelOLS`).

Do not reimplement patsy/fixest/token parsing in `_base.py`.

### B. Backend numeric preparation

Provide a helper that:

- resolves the estimator backend through the existing `BaseEstimator` device contract;
- converts numerical X/y to backend float64 while preserving selected device;
- reshapes a 1-D design to `(n, 1)`;
- calls existing `validate_panel_alpha()` and `validate_panel_numeric_data()`;
- does not convert full numerical GPU arrays to NumPy.

Metadata labels may continue to be factorized/aligned on CPU as already documented.

### C. Shared linear prediction helper

Provide a helper for ordinary linear prediction that accepts explicit flags:

- add intercept or not;
- formula-aware reconstruction or raw arrays;
- return NumPy or backend-native result.

This is required because existing return contracts differ: most panel estimators return NumPy predictions, while `FamaMacBeth` currently preserves a backend-native result. Stage A must preserve this difference rather than normalize it silently.

### D. Shared summary construction

Provide a helper that creates the existing `PanelSummary` from fitted public attributes while accepting model-specific metadata (`entity_effects`, `time_effects`, variance components, theta, within R², extra fields).

`PanelSummary` remains the public structured summary container in Stage A. New Stage-B test/fit-stat result types will be added later rather than prematurely expanding the Stage-A public contract.

## 7. Covariance registry design

Extend `statgpu/panel/_covariance.py` with an internal centralized registry/dispatcher for the **existing** covariance names only.

The registry must distinguish statistical contexts rather than assume every estimator uses the same sandwich object:

- OLS/transformed-OLS residual covariance: nonrobust, HC1 (`robust`), clustered, HAC when currently supported;
- Fama-MacBeth coefficient-series covariance: remains estimator-specific and outside the residual-sandwich dispatcher.

The OLS dispatcher should accept explicit context rather than infer hidden corrections:

- `X`, `resid`, `scale`;
- `df_resid` and effective rank/parameter count as applicable;
- cluster metadata;
- bandwidth/kernel;
- backend/xp;
- allowed covariance names for the calling estimator.

Behavior-preserving formulas:

- nonrobust: preserve current scale and bread normalization exactly;
- `robust`: preserve the current HC1 correction used by each migrated path; where the existing correction is `n/df_resid`, pass that explicitly rather than replacing it by `n/(n-k)`;
- clustered: call the existing `clustered_covariance()` / `two_way_clustered_covariance()` functions without adding a new small-sample correction;
- HAC: call the existing `hac_covariance()` implementation with unchanged bandwidth/kernel semantics.

The dispatcher must reject unsupported covariance names explicitly. Stage A must not add HC0/HC2/HC3/Driscoll-Kraay aliases; those belong to Stage C.

## 8. Estimator migration order

Migrate in risk order and rerun golden tests after each group.

### Group 1 — BetweenOLS and FirstDifferenceOLS

These already share `compute_panel_inference`; move them to `BasePanelModel`, the shared input/summary/prediction lifecycle, and the covariance registry with the smallest numerical surface change.

### Group 2 — PooledOLS

Replace estimator-local formula/backend/summary/prediction duplication and covariance `if/elif` dispatch with shared primitives. Preserve:

- automatic intercept;
- rank-aware df;
- robust HC1 scaling;
- cluster requirement and factorization;
- HAC temporal stable sort behavior;
- NumPy prediction return type.

### Group 3 — PanelOLS

Adopt the shared base and covariance registry while leaving fixed-effect transformation and effect-map computation model-specific. Preserve:

- constructor flags and formula token/pipe mutation behavior;
- FE df counting exactly as current behavior;
- one-way/two-way clustered contracts;
- within R²;
- existing prediction/effect-map semantics.

Do not make effects lazy in this PR unless golden tests prove the externally observable maps and predictions remain identical. The proposal's lazy-effects idea is subordinate to behavior preservation.

### Group 4 — RandomEffects

Adopt shared formula/backend/summary/prediction lifecycle and shared nonrobust inference primitives, but leave Swamy-Arora variance-component estimation and quasi-demeaning model-specific. Do not add `cov_type` here in Stage A; robust RE belongs to Stage C.

### Group 5 — FamaMacBeth

Inherit from the shared base only for formula state, parameter validation helpers, and summary construction where this can be done without changing backend-native output contracts. Its beta-series covariance and backend-native `predict()` result remain specialized.

If migration of a specialized Fama-MacBeth path creates more duplication/branches than it removes, Stage A may leave its numerical path unchanged while still sharing the neutral summary/formula lifecycle. This must be documented in the PR review rather than forcing an inappropriate abstraction.

## 9. Rank/df and validation contract

Stage A must not silently harmonize currently different df conventions.

Before migration, tests must freeze:

- PooledOLS effective-rank residual df;
- PanelOLS absorbed-effect df logic;
- BetweenOLS group-level residual df;
- FirstDifferenceOLS differenced-sample residual df;
- RandomEffects within/between/GLS df definitions;
- FamaMacBeth `T-1` inference df.

A later Stage B may expose/document unified fit-statistics df definitions, but this PR must not change the fitted inference numbers to achieve superficial architectural consistency.

Rank-deficient behavior should remain explicit and should not become silently more permissive or restrictive through a shared helper.

## 10. Backend and host-transfer contract

Blocking rules:

- NumPy, CuPy, and Torch remain supported for all touched public estimators;
- explicit `device='cuda'` / `device='torch'` never falls back to CPU;
- full transformed designs, residuals, scores, and covariance accumulation remain backend-native;
- CPU transfer is allowed for metadata factorization and final small result vectors/scalars exactly where existing contracts permit it;
- do not introduce a new `_to_numpy(X)` / `_to_numpy(resid)` in numerical core paths.

Add Torch-CPU backend tests for deterministic hosted coverage and preserve/extend static host-transfer assertions. Physical CuPy/Torch validation remains a required remote gate because the refactor touches shared backend paths.

## 11. Test plan

Add or strengthen:

1. `test_panel_stage_a_golden.py` — behavior freeze for all six estimators;
2. `test_panel_stage_a_framework.py` — base helper lifecycle and covariance registry contracts;
3. formula parity tests covering intercept, categorical, interaction, missing-row alignment, FE tokens/pipe syntax;
4. balanced/unbalanced panel validation and string label ordering;
5. rank-deficiency/df failure tests;
6. covariance registry direct tests for existing nonrobust/HC1/cluster/HAC formulas versus the pre-refactor analytic/current implementations;
7. NumPy/Torch-CPU parity across all touched estimators;
8. CuPy/Torch CUDA physical smoke matrix after hosted CI is clean;
9. static tests that forbid new full-data host transfers in transformed design/residual/covariance paths.

Existing `test_panel_p2.py`, `test_panel_formula.py`, and panel coverage in `test_third_full_review.py` remain regression gates and should be reused rather than duplicated blindly.

## 12. External alignment

Stage A is behavior-preserving, so its primary correctness baseline is the exact pre-refactor statgpu implementation plus analytic OLS/sandwich identities.

Where `linearmodels` is available locally/CI, retain or add representative coefficient/covariance comparison. Do not change Stage-A formulas merely to match a different external small-sample/df convention.

The full linearmodels/plm/Stata definition matrix is a Stage B/C acceptance item when new diagnostics/covariance APIs are introduced.

## 13. Documentation

Update:

- root changelog with PR-A architecture summary after validation;
- EN/CN changelog consistently;
- panel developer/design documentation to mark Stage A implemented and Stage B/C still pending.

Because Stage A is intended to preserve user behavior, do not advertise new diagnostics or covariance types.

## 14. Review/fix gates

Plan review must specifically challenge:

- whether the base abstraction changes any output/device/formula contract;
- whether covariance normalization/HC1 df corrections remain estimator-equivalent;
- whether FamaMacBeth is being forced through an invalid OLS abstraction;
- whether golden tests are sufficiently broad before refactoring;
- whether effect maps/predictions are accidentally changed by proposed lazy handling.

Implementation review must inspect:

- formulas/scaling/df;
- three-backend device ownership and host transfers;
- formula/missing-row/index alignment;
- rank-deficiency behavior;
- summary/prediction output types;
- covariance dispatch and unsupported-name errors;
- code reuse and whether the new base actually removes duplication.

Fix all CRITICAL/HIGH and relevant MEDIUM findings, rerun targeted/full gates, then fresh-review again.

## 15. Exit criteria for PR-A

`COMPLETE` requires:

- golden behavior tests pass after migration;
- all existing panel tests and full hosted CPU suite pass;
- Python compatibility/static/docs gates pass;
- NumPy/Torch hosted parity passes;
- physical CuPy and Torch CUDA regression passes on the final numerical implementation head;
- no unresolved CRITICAL/HIGH review findings;
- docs/changelog updated.

If only physical GPU validation is unavailable after all local/hosted gates pass, stop at `PARTIAL_REMOTE_PENDING` with an exact final-head remote command. Do not merge PR-A until the user explicitly requests merge.
