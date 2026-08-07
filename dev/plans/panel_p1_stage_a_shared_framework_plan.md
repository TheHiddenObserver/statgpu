# Panel P1 Stage A — shared framework and covariance registry plan

Issue: #93  
Baseline: `master` at `070176fedbc9cae6e55dd8bbb9e389e08744bb30`  
Branch: `agent/panel-p1-stage-a-framework`  
Scope: PR-A only — behavior-preserving shared panel framework refactor

## 1. Stage boundary

Issue #93 decomposes Tier-1 panel work into at least three PRs. This plan implements **Stage A only**:

1. freeze current numerical/public behavior with golden regression tests;
2. add an internal shared panel base abstraction;
3. centralize covariance dispatch for the existing covariance contracts;
4. add shared panel index/structure validation and structured result substrate required by #93;
5. migrate existing estimators without changing statistical definitions or public constructor signatures.

Stage B (Hausman/pooling-F/Breusch-Pagan LM and populated fit statistics) and Stage C (HC0/HC2/HC3, RandomEffects robust covariance, Driscoll-Kraay, expanded cluster corrections/contracts) remain separate PRs.

Stage A may define internal result dataclasses that Stage B will populate, but it must **not** advertise new diagnostic methods or new fitted statistics before their formulas and applicability rules are implemented and externally aligned.

## 2. Current-source observations

The existing panel package already has substantial shared infrastructure:

- `_formula.py` centralizes formula parsing, missing-row alignment, fixed-effect tokens, and prediction design reconstruction;
- `_utils.py` already contains `PanelSummary`, label factorization, numeric/alpha validation, group operations, demeaning, and a shared OLS inference helper used by `BetweenOLS` and `FirstDifferenceOLS`;
- `_covariance.py` already provides one-way cluster, two-way cluster, and HAC covariance functions with backend-native score accumulation after metadata factorization;
- `PanelOLS`, `PooledOLS`, `RandomEffects`, and `FamaMacBeth` still contain estimator-local inference/summary/lifecycle logic;
- all six estimators duplicate some combination of formula-state setup, fitted-state validation, summary construction, feature-name reconstruction, backend conversion, or covariance dispatch.

The refactor must extend these helpers rather than create a parallel panel stack.

## 3. Impact classification

Active axes:

- **Public API compatibility**: constructor signatures, fitted attributes, output array types, summary fields, prediction return types, and current exception behavior are frozen;
- **Backend**: NumPy/CuPy/Torch device and dtype ownership, plus strict explicit-device behavior, are active gates;
- **Inference**: covariance, standard errors, t statistics, p-values, and confidence intervals are structurally refactored but numerically frozen;
- **Formula**: intercept/effect token/missing-row/categorical/interaction/prediction-design behavior is active;
- **Panel structure**: entity/time metadata, balanced/unbalanced detection, ordering, and validation become shared primitives;
- **Architecture/maintenance**: duplication removal is the principal goal;
- **Tests/docs**: golden regression tests and synchronized changelog/design docs are blocking.

Inactive axes:

- CV: panel estimators here are not tunable CV wrappers;
- loss/penalty/solver: no objective or optimizer changes;
- new diagnostics: Stage B;
- new covariance formulas: Stage C;
- performance claims: none. Incidental transfer reductions are not marketed as speedups.

## 4. Capability decisions

| Family | backend | inference | formula | benchmark |
| --- | --- | --- | --- | --- |
| PanelOLS | three-backend | supported, behavior-preserving | supported | physical GPU regression required; no speed claim |
| RandomEffects | three-backend | supported, nonrobust unchanged | supported | physical GPU regression required; no speed claim |
| PooledOLS | three-backend | supported, existing covariance set unchanged | supported | physical GPU regression required; no speed claim |
| BetweenOLS | three-backend | supported, existing covariance set unchanged | supported | physical GPU regression required; no speed claim |
| FirstDifferenceOLS | three-backend | supported, existing covariance set unchanged | supported | physical GPU regression required; no speed claim |
| FamaMacBeth | three-backend | supported, estimator-specific covariance unchanged | supported | physical GPU regression required; no speed claim |

`FamaMacBeth` must **not** use the OLS residual-sandwich dispatcher. Its inference covariance is based on the time series of cross-sectional coefficient estimates and is a different statistical object. It may reuse statistically neutral lifecycle/formula/summary helpers only.

## 5. Golden behavior freeze before refactoring

Create `dev/tests/test_panel_stage_a_golden.py` in a **pre-refactor commit** and require that commit to pass before changing panel source files. The same file remains active after migration.

For every relevant model freeze:

- `coef_`;
- deterministic predictions/fitted behavior;
- `bse_`, `tvalues_`, `pvalues_`, `conf_int_`;
- covariance matrix if currently public/stored;
- `nobs`, `df_resid`, rank/effect metadata currently exposed;
- current R-squared field(s);
- `PanelSummary.to_dict()` keys/values;
- prediction return type (NumPy versus backend-native);
- current failure behavior for rank/df/required identifiers and unsupported covariance names.

Coverage:

- `PanelOLS`: no effects, entity, time and two-way FE; nonrobust/robust/one-way/two-way cluster; balanced/unbalanced; string labels;
- `RandomEffects`: balanced/unbalanced group sizes, variance components, theta, formula/array parity;
- `PooledOLS`: nonrobust/robust/clustered/HAC, including time-index sorting;
- `BetweenOLS`: nonrobust/robust;
- `FirstDifferenceOLS`: nonrobust/robust with and without explicit time sorting;
- `FamaMacBeth`: nonrobust/newey-west plus backend-native prediction contract.

Prefer analytic/current-formula reference calculations in tests over unexplained opaque constants. For behavior that is easiest to snapshot, use deterministic constants generated from the pre-refactor commit and strict tolerances that would catch covariance normalization or df drift.

## 6. Shared result and panel-structure substrate

Add internal `statgpu/panel/_results.py` (not publicly exported in Stage A) with stable dataclasses for later Stage B use:

### `PanelTestResult`

Fields should be capable of representing the #93 Stage-B contract without yet creating diagnostic methods:

- `statistic`;
- `pvalue`;
- `distribution`;
- degrees of freedom (scalar or structured tuple as appropriate);
- `null`;
- `alternative`;
- `applicable`;
- `reason` / applicability diagnostic;
- small `metadata` mapping.

### `PanelFitStatistics`

Optional fields prepared for Stage B:

- within/between/overall R-squared;
- adjusted R-squared;
- model F statistic / p-value / df;
- any definition metadata needed to state the df convention.

All Stage-A fields may remain `None` where the statistic is not currently implemented. Stage A must not attach misleading new public values to fitted estimators.

### `PanelIndexInfo`

Add a shared internal structure/helper (in `_base.py`, `_utils.py`, or `_results.py`) for observation metadata:

- `nobs`;
- entity/time factor codes and original labels where provided;
- entity/time counts;
- `n_entities`, `n_times`;
- balanced/unbalanced status when both dimensions are available;
- optional duplicate `(entity, time)` detection status;
- observation-order metadata needed to preserve row alignment.

Building this structure may factorize metadata on CPU, but it must not transfer full numerical X/y/residual arrays to CPU. It must not reorder observations implicitly.

This closes the Stage-A requirement for common entity/time/balance validation and prevents Stage B diagnostics from reimplementing panel structure logic independently.

## 7. Shared base design

Add internal `statgpu/panel/_base.py` with `BasePanelModel(BaseEstimator)`. It is **not** exported publicly in Stage A.

The base provides statistically neutral primitives only.

### A. Formula/model-matrix state

Wrap existing `_prepare_formula_fit()` / `_align_formula_side_array()` rather than reimplementing parser logic. The helper must:

- accept explicit `model_has_intercept` and `support_pipe` flags;
- store `_design_info`, `_feature_names`, `_formula_has_intercept` exactly as today;
- align named side arrays through the existing formula helper;
- return FE metadata without silently mutating estimator options. `PanelOLS` explicitly applies its current formula-induced effect mutation.

### B. Backend numeric preparation

A helper should:

- resolve through the existing `BaseEstimator` device contract;
- convert numerical X/y to backend float64 on the selected device;
- reshape 1-D X to `(n, 1)`;
- reuse `validate_panel_alpha()` and `validate_panel_numeric_data()`;
- optionally build `PanelIndexInfo` from side metadata;
- never convert full numerical GPU arrays to NumPy.

### C. Shared linear prediction

Provide a helper with explicit switches for:

- formula-aware design reconstruction;
- intercept addition;
- expected feature count;
- return NumPy versus backend-native.

The explicit return switch is required because most panel models return NumPy predictions, while `FamaMacBeth` currently keeps a backend-native result.

### D. Shared summary construction

Construct the existing `PanelSummary` from fitted fields while accepting model-specific metadata (effect flags, within R², variance components, theta, `extra`). Preserve the current `PanelSummary.to_dict()` contract exactly in Stage A.

## 8. Covariance registry design

Extend `statgpu/panel/_covariance.py` with an internal registry/dispatcher for the **existing** OLS/transformed-OLS covariance names only.

Inputs must make statistical corrections explicit:

- `X`, `resid`, `scale`;
- `df_resid` / effective rank information as required;
- backend/xp;
- cluster metadata;
- bandwidth/kernel;
- allowed names for the calling estimator.

Behavior-preserving formulas:

- nonrobust: preserve existing bread/scale normalization exactly;
- `robust`: preserve current HC1 correction. If current behavior is `n/df_resid`, pass that correction explicitly rather than replacing it with a generic hidden `n/(n-k)`;
- clustered: delegate to existing one-/two-way cluster functions with no new small-sample correction;
- HAC: delegate to existing `hac_covariance()` with unchanged bandwidth/kernel semantics.

Unsupported names must fail explicitly. Do **not** add HC0/HC2/HC3/Driscoll-Kraay aliases in Stage A.

The registry should return a backend-native covariance matrix. A shared inference helper may then compute bse/t-or-z/p/CI and transfer only final small vectors.

## 9. Estimator migration order

Migrate and rerun golden tests in risk order.

### Group 1 — BetweenOLS / FirstDifferenceOLS

They already share `compute_panel_inference`; move to `BasePanelModel`, shared formula/backend/summary/prediction helpers and registry with minimal numerical change.

### Group 2 — PooledOLS

Replace local lifecycle/summary/prediction and covariance dispatch. Preserve:

- automatic intercept;
- effective-rank residual df;
- robust HC1 scaling;
- cluster requirement/factorization;
- HAC stable sort by `time_index`;
- NumPy prediction output.

### Group 3 — PanelOLS

Share base/registry while keeping within transformation and effect-map computation specialized. Preserve:

- constructor and formula effect behavior;
- absorbed-effect df logic;
- one-/two-way cluster behavior;
- within R²;
- existing effect maps and prediction semantics.

Do not make effects lazy in Stage A unless the same public maps/predictions are demonstrably identical. Behavior preservation outranks the older proposal's lazy-effects preference.

### Group 4 — RandomEffects

Share neutral formula/backend/summary/prediction and nonrobust inference primitives. Keep Swamy-Arora variance-component estimation and quasi-demeaning specialized. Do not add a `cov_type` constructor argument in Stage A; robust RE is Stage C.

### Group 5 — FamaMacBeth

Reuse only neutral formula state, common parameter validation where equivalent, `PanelIndexInfo`, and summary construction. Keep backend detection, beta-series covariance and backend-native prediction specialized if sharing them would change current behavior or create more branching than duplication removed.

## 10. Rank/df and validation contract

Stage A must not harmonize currently different df conventions.

Freeze and preserve:

- PooledOLS effective-rank residual df;
- PanelOLS absorbed-effect df;
- BetweenOLS group-level residual df;
- FirstDifferenceOLS differenced-sample residual df;
- RandomEffects within/between/GLS df definitions;
- FamaMacBeth `T-1` inference df.

Shared `PanelIndexInfo` validation must explicitly cover:

- one-dimensional entity/time labels of correct length;
- balanced and unbalanced panels;
- deterministic original-label/code mapping;
- observation-order preservation;
- duplicate `(entity, time)` detection as metadata/error according to current model needs, without introducing a blanket new rejection that would break existing valid inputs.

Rank-deficient paths must not become silently more or less permissive because of shared helpers.

## 11. Backend and host-transfer contract

Blocking rules:

- NumPy, CuPy and Torch remain supported for every touched public model;
- explicit CUDA/Torch requests never silently fall back;
- transformed design, residual, score and covariance accumulation remain on selected backend;
- CPU transfer is allowed for metadata factorization and final small result/scalar conversion only, consistent with current contracts;
- no new `_to_numpy(X)`, `_to_numpy(y)`, `_to_numpy(resid)` in numerical core paths.

Hosted Torch-CPU tests provide deterministic backend coverage; physical CuPy/Torch CUDA remains a remote completion gate.

## 12. Test plan

Add/strengthen:

1. `test_panel_stage_a_golden.py` — committed and green before source refactor, then retained;
2. `test_panel_stage_a_framework.py` — base/result/index/registry contracts;
3. direct covariance formula tests for nonrobust/HC1/cluster/HAC against analytic pre-refactor definitions;
4. formula parity: intercept, categorical, interactions, missing-row alignment, FE tokens/pipe syntax;
5. balanced/unbalanced/index-order/string-label/duplicate-pair metadata tests;
6. rank-deficiency and residual-df failure tests;
7. NumPy/Torch-CPU parity across touched estimators;
8. static host-transfer checks in transformed-design/residual/covariance paths;
9. physical CuPy/Torch CUDA smoke matrix after hosted CI is clean.

Reuse existing `test_panel_p2.py`, `test_panel_formula.py`, and panel checks in `test_third_full_review.py` rather than duplicating them blindly.

## 13. External alignment

Stage A's primary baseline is exact pre-refactor statgpu behavior plus analytic OLS/sandwich identities. If `linearmodels` is available, retain/add representative comparisons, but do not change formulas merely to match another package's small-sample or df convention.

The complete linearmodels/plm/Stata definition matrix becomes blocking in Stage B/C when new diagnostics/covariances are public.

## 14. Documentation

After validation update:

- root changelog;
- EN/CN changelogs;
- panel design/proposal status showing Stage A implemented and Stage B/C pending.

Do not advertise diagnostics/covariance types that Stage A does not implement.

## 15. Review/fix gates

Plan review must challenge:

- output/device/formula compatibility of the base abstraction;
- covariance normalization and HC1 df correction;
- whether FamaMacBeth is forced into an invalid OLS abstraction;
- whether golden coverage is genuinely pre-refactor;
- whether effect maps are accidentally changed;
- whether Stage-A structured result/index requirements from #93 are actually represented without prematurely publishing Stage-B statistics.

Implementation review must inspect formulas/scaling/df, three-backend ownership/transfers, formula/missing-row/index alignment, rank behavior, result/summary/prediction types, covariance unsupported-name errors, and whether duplication is actually removed.

Fix every CRITICAL/HIGH and relevant MEDIUM finding, rerun affected gates, and perform a fresh review again.

## 16. Exit criteria

`COMPLETE` requires:

- pre-refactor golden tests were green and remain green after migration;
- existing panel tests plus full hosted CPU suite pass;
- compatibility/static/docs gates pass;
- NumPy/Torch hosted parity passes;
- final numerical implementation head passes physical CuPy and Torch CUDA regression;
- no unresolved CRITICAL/HIGH review findings;
- docs/changelog are synchronized.

If only physical GPU evidence is unavailable after all hosted/local gates pass, finish Stage A as `PARTIAL_REMOTE_PENDING` with an exact final-head validation command. PR-A must not be merged without an explicit user merge request.
