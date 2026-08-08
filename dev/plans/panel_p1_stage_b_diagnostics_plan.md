# Panel Tier-1 Stage B — diagnostics and fit-statistics plan

Issue: #93  
Stage-A base: PR #119 / merge commit `e9e0ec43b8b2cfcd7600013a60cb02100c72f4f5`  
Branch: `agent/panel-p1-stage-b-diagnostics`

## 1. Scope and impact classification

Stage B adds the user-visible specification tests and fit statistics promised by Issue #93 while preserving the Stage-A estimator transformations, coefficient estimates, covariance definitions, prediction contracts, and strict-device rules.

Active impact axes:

- **Public API** — new structured diagnostic functions/results and new fitted-statistics attributes.
- **Inference** — Hausman, pooling F, Breusch–Pagan LM, model F statistics, p-values, and degrees of freedom.
- **Backend** — all sufficient-statistic accumulation must work on NumPy, CuPy, and Torch without explicit-device fallback.
- **Formula** — diagnostic metadata and sample identity must remain aligned after Patsy missing-row filtering.
- **Docs/artifacts** — EN/CN model docs, changelogs, external-definition matrix, and physical-GPU evidence.

Inactive gates:

- **Loss / penalty / solver / CV** — Stage B does not modify an optimization objective, regularization, solver, or tuning path.
- **Performance benchmark** — no speedup claim is planned. A physical runner is a correctness/provenance gate, not a timing benchmark. Any material memory/performance regression discovered during review reactivates this gate.

Validation target: `remote-full` before Stage B is called COMPLETE.

## 2. Capability decisions

| Model / capability | backend | inference | formula | Stage-B diagnostics / fit stats |
| --- | --- | --- | --- | --- |
| `PanelOLS` | three-backend | supported | supported | pooling F; standard within/between/overall R²; adjusted R²; classical model F; Hausman input |
| `RandomEffects` | three-backend | supported | supported | within/between/overall R²; adjusted R²; classical model F; Hausman input |
| `PooledOLS` | three-backend | supported | supported | overall R² always; within/between R² and BP-LM when `entity_ids` supplied; adjusted R²; classical model F |
| `BetweenOLS` | three-backend | supported | supported | within/between/overall R²; adjusted R²; classical model F |
| `FirstDifferenceOLS` | three-backend | supported | supported | within/between/overall R²; adjusted R² on differenced fit basis; classical model F |
| `FamaMacBeth` | three-backend | supported | supported | overall R² always; within/between R² only if optional `entity_ids` supplied; no residual-OLS model F is synthesized from the beta-series covariance |

`FamaMacBeth` remains a special covariance/inference family. Stage B must not route it through residual OLS sandwich or claim that a pooled-OLS F statistic is its beta-series joint significance test.

## 3. External definition matrix

Stage B will record and test against the following definitions. External packages are references, not authorities that override statgpu's existing estimator contracts.

### 3.1 Parameter-based R²

Primary alignment: `linearmodels` 7.0.

References:

- https://bashtage.github.io/linearmodels/panel/faq.html
- https://bashtage.github.io/linearmodels/panel/mathematical-formula.html
- https://bashtage.github.io/linearmodels/_modules/linearmodels/panel/model.html

`linearmodels` explicitly distinguishes parameter-based R² from correlation-based measures used by some other software. Stage B adopts the parameter-based family because it evaluates the actual estimated coefficient vector.

For unweighted data and a fitted coefficient vector `beta`:

- **Overall**: residual `e_o = y - X beta`; center `y` only when the corresponding level equation has an identified explicit/implicit constant; `R²_o = 1 - SSE_o / TSS_o`.
- **Between**: form entity means `y_bar_i`, `X_bar_i`; residual `e_b = y_bar - X_bar beta`; center the between response when the corresponding level equation has an identified explicit/implicit constant; `R²_b = 1 - SSE_b / TSS_b`.
- **Within**: entity-demean `y` and `X`; `e_w = y_within - X_within beta`; `R²_w = 1 - SSE_w / TSS_w`.

For the new standardized `fit_statistics_` fields, a zero total sum of squares follows the `linearmodels` convention and reports `0.0`, with `metadata['degenerate_total_ss'][<measure>] = True`. Stage B does **not** rewrite legacy estimator attributes that already have another degenerate-TSS behavior; those remain frozen for compatibility.

Important compatibility rule: Stage A froze the existing public `PanelOLS.rsquared_within`. For two-way FE that legacy attribute is computed on the full entity+time transformed fit, whereas the standard `linearmodels` `rsquared_within` is entity-within. **Stage B must not silently change the legacy attribute.** The new `fit_statistics_.rsquared_within` is the explicitly documented standard entity-within measure; metadata records `legacy_rsquared_within` when the existing attribute differs (notably two-way FE). For one-way entity FE the two coincide up to numerical tolerance.

### 3.2 Classical model F statistic

Primary alignment: `linearmodels.PanelResults.f_statistic`.

Reference:

- https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.results.PanelResults.f_statistic.html

For a model fit in its estimator-specific estimation space:

`F = ((RSS_R - RSS_U) / q) / (RSS_U / df_resid)`

where the restriction sets all estimable non-constant slope coefficients to zero, `q` is the **effective restriction rank** (`rank_unrestricted - rank_restricted`, not blindly the raw column count), and `df_resid` is the model's established residual degrees of freedom.

Stage B's `PanelFitStatistics.f_statistic` is the **classical homoskedastic model F**. It does not silently turn into a robust Wald statistic when `cov_type='robust'` or `'clustered'`. Robust Wald/model tests are a separate contract and are not added in Stage B.

Estimator fit spaces:

- `PooledOLS`: pooled level design including its intercept.
- `PanelOLS`: the existing effect-transformed design and current absorbed-effect residual df.
- `RandomEffects`: the existing quasi-demeaned GLS design.
- `BetweenOLS`: entity-mean regression.
- `FirstDifferenceOLS`: first-difference regression.
- `FamaMacBeth`: do not manufacture a residual-OLS F statistic; beta-series joint Wald inference may be a later explicitly named capability.

If there are no estimable non-constant restrictions, the F field is unavailable with an explicit metadata reason rather than dividing by zero.

### 3.3 Pooling F / fixed-effect significance test

Primary alignment: `linearmodels.PanelEffectsResults.f_pooled` and `plm::pFtest`.

References:

- https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.results.PanelEffectsResults.f_pooled.html
- https://bashtage.github.io/linearmodels/_modules/linearmodels/panel/model.html
- https://rdrr.io/cran/plm/man/pFtest.html
- https://rdrr.io/cran/plm/src/R/test_general.R

For the same aligned estimation sample and regressors:

`F_pool = ((RSS_pool - RSS_FE) / df_num) / (RSS_FE / df_resid_FE)`.

The restricted pooled model must be constructed with the **same constant convention as the FE model's nested null**, including the no-explicit-constant correction used by `linearmodels`:

- when the level design contains an identified explicit constant, fit the pooled regression with that level design;
- when the FE specification has effects but no explicit constant column, project both pooled `y` and pooled `X` off the common constant before the pooled slope fit, and reduce the effect-test numerator df by one. This prevents the common mean from being incorrectly counted as a tested fixed effect.

The primary numerator-df calculation is the nested-model rank/df difference, equivalent to `df_resid_pool - df_resid_FE` after the constant correction. The implementation records both effective ranks and the final df in metadata; it never hard-codes `N-1`, `T-1`, or `N+T-2`.

Contract:

- only `PanelOLS` fits with at least one included effect are applicable;
- pooling comparison is recomputed internally on the same post-formula/post-missing-data sample during `PanelOLS.fit()`; users do not have to fit a second `PooledOLS` object;
- the test is classical/homoskedastic, matching the external definitions; covariance type does not change the RSS-based statistic;
- null: all included fixed effects are jointly zero;
- alternative: at least one included effect is nonzero;
- distribution: `F(df_num, df_resid_FE)`.

Numerical nesting checks:

- if `RSS_pool - RSS_FE >= 0`, use it directly;
- if the difference is negative only within a scale-aware floating-point tolerance, normalize it to zero and record `metadata['roundoff_normalized']=True`;
- a materially negative difference indicates a violated nesting/rank contract and returns `applicable=False` with an explicit reason; it is never silently clipped into a valid-looking positive statistic.

### 3.4 Breusch–Pagan LM for entity random effects

Primary alignment: `plm::plmtest(type='bp', effect='individual')`, including the Baltagi–Li unbalanced-panel version.

References:

- https://rdrr.io/cran/plm/man/plmtest.html
- https://rdrr.io/cran/plm/src/R/test_general.R

The test uses **pooled-OLS residuals**. Let `e_it` be pooled residuals, `n` the number of observations, and `T_i` the observation count of entity `i`:

`CP = sum_it e_it^2`

`A1 = sum_i (sum_t e_it)^2 / CP - 1`

`M11 = sum_i T_i^2`

`LM1 = n * sqrt(1 / (2 * (M11 - n))) * A1`

`LM_BP = LM1^2 ~ chi2(1)`.

This formula applies to balanced and unbalanced panels and matches the current `plm` implementation attributed to Baltagi and Li (1990) for incomplete panels.

Stage-B scope is **one-way entity BP-LM** because statgpu's current `RandomEffects` is one-way entity RE. A two-way BP statistic would test a broader error-components model that statgpu does not currently estimate and is not exposed under the RE-vs-pooled diagnostic name in this PR.

Contract:

- add optional `entity_ids=None` to `PooledOLS.fit()`; existing calls remain unchanged;
- when entity IDs are supplied, formula row filtering aligns them through the existing side-array machinery;
- pooled residual group sums and counts are accumulated on the selected backend during fit; full numerical residual arrays are not copied to CPU for the test;
- if `cov_type='hac'` and `time_index` causes a stable numerical row reorder, the aligned `entity_ids` diagnostic codes are reordered by the **same** permutation before any residual grouping, R² accumulation, or sample fingerprinting; diagnostic metadata may never remain in pre-sort order while X/y are post-sort;
- null: entity random-effect variance is zero (pooled OLS sufficient);
- alternative: a nonzero entity random-effect component is present;
- distribution: `chi2(1)`;
- require at least two entities, positive pooled residual sum of squares, and `M11 > n`; otherwise return an inapplicable structured result with the exact reason.

The name/documentation must distinguish this panel error-components BP-LM from the cross-sectional heteroskedasticity Breusch–Pagan test.

### 3.5 Classical Hausman FE-vs-RE

Primary alignment: Hausman (1978), `plm::phtest` original quadratic-form method, and Stata `hausman`.

References:

- https://rdrr.io/cran/plm/man/phtest.html
- https://rdrr.io/cran/plm/src/R/test_general.R
- https://www.stata.com/manuals/rhausman.pdf

For common non-intercept coefficients:

`d = beta_FE - beta_RE`

`D = V_FE - V_RE`

`H = d' D^{-1} d ~ chi2(q)` for full-rank `D`.

Stage-B applicability is deliberately stricter than a blind matrix solve:

- `fe_model` must be a fitted `PanelOLS` with `entity_effects=True` and `time_effects=False`;
- `re_model` must be a fitted `RandomEffects`;
- both models must represent the same aligned estimation sample and common slope design;
- the original quadratic-form test is available only for the current **classical/nonrobust** covariance pair. A robust/clustered FE covariance is not relabeled as a robust Hausman test. `plm` documents robustification through an auxiliary-regression Hausman variant, which is outside Stage B;
- intercepts are excluded, following `plm` and Stata; common slope names/order are matched explicitly;
- no common estimable slope => inapplicable.

Covariance-difference handling:

1. symmetrize `D` numerically as `(D + D.T)/2` on the small final matrix;
2. compute an eigenvalue/rank tolerance scaled by matrix norm and machine epsilon;
3. if an eigenvalue is materially negative, return `applicable=False` with reason `covariance difference is not positive semidefinite`; do **not** force a statistic by absolute values or eigenvalue clipping;
4. if `D` is positive semidefinite but rank-deficient, statgpu may use a Moore–Penrose inverse on the identified range and set chi-square df to `rank(D)`, but only if `d` lies in the column space within tolerance; record `metadata['used_pinv']=True`, numerical rank, tolerance, and `metadata['definition_extension']='singular PSD generalized-inverse Hausman'`;
5. this singular-PSD generalized-inverse case is a documented statgpu extension to the ordinary full-rank `plm`/Stata path, not claimed as byte-for-byte external behavior;
6. if `d` has a material component in the null space, return inapplicable rather than pretending the unidentified direction contributes zero;
7. a computed statistic slightly below zero only from roundoff may be normalized to zero with metadata; a materially negative statistic is inapplicable.

This makes singular/indefinite behavior explicit as required by Issue #93 and avoids generic linear-algebra exceptions.

## 4. Public API proposal

### 4.1 Result objects

Keep the Stage-A frozen dataclasses and make them public through `statgpu.panel` and top-level `statgpu`:

- `PanelTestResult`
- `PanelFitStatistics`

`PanelTestResult` fields remain:

- `statistic`
- `pvalue`
- `distribution`
- `df`
- `null`
- `alternative`
- `applicable`
- `reason`
- `metadata`

No exception is used for an econometrically inapplicable but otherwise well-formed diagnostic. Programming errors (wrong object type, unfitted object when a fitted model is required, malformed metadata length) remain exceptions.

### 4.2 Diagnostic functions

Add public functions in `statgpu.panel._diagnostics` and export them from `statgpu.panel` and top-level `statgpu`:

```python
hausman_test(fe_model: PanelOLS, re_model: RandomEffects) -> PanelTestResult
pooling_f_test(fe_model: PanelOLS) -> PanelTestResult
breusch_pagan_lm_test(pooled_model: PooledOLS) -> PanelTestResult
```

These functions consume fitted model state/sufficient statistics. They do not accept arbitrary covariance matrices as a pseudo-public escape hatch in Stage B.

Convenience estimator methods may delegate exactly to these functions:

```python
fe.pooling_f_test()
fe.hausman_test(re)
pooled.breusch_pagan_lm_test()
```

If methods are added, there must be one implementation source of truth in `_diagnostics.py`; methods are thin delegates only.

### 4.3 Fit statistics

After every supported fit, expose:

```python
model.fit_statistics_: PanelFitStatistics
```

with standard fields:

- `rsquared_within`
- `rsquared_between`
- `rsquared_overall`
- `rsquared_adj`
- `f_statistic`
- `f_pvalue`
- `f_df`
- `metadata`

`metadata` must identify the R² convention and estimator-specific adjusted-R² basis, and give explicit reasons for unavailable fields.

Do not rename/remove the existing `PanelOLS.rsquared_within` or estimator-specific `rsquared` attributes in Stage B.

## 5. Internal architecture

### 5.1 Shared diagnostics helpers

Create `statgpu/panel/_diagnostics.py` for:

- construction of applicable/inapplicable `PanelTestResult` objects;
- parameter-based within/between/overall R² sufficient-statistic helpers;
- adjusted-R² helper;
- classical model-F helper;
- pooling-F helper;
- BP-LM helper;
- Hausman small-matrix comparison and applicability logic;
- compact backend-native sample/design fingerprint construction.

Core observation-scale operations take `xp` and backend arrays. Only final scalars, O(k) numerical fingerprint components, small `k x k` covariance matrices, feature names, and index metadata may be converted to NumPy.

### 5.2 Covariance persistence

`BasePanelModel._panel_store_ols_inference()` already returns `cov_params`. Stage B stores the final small covariance matrix needed by diagnostics as an **internal** CPU ndarray (for example `_panel_cov_params`) on FE/RE and other relevant OLS-style models while preserving all existing `bse_/tvalues_/pvalues_/conf_int_` values.

Do not create a new universal public `cov_params_` contract merely to implement Hausman. `FamaMacBeth.cov_params_` is an existing estimator-specific attribute and remains unchanged.

The conversion is limited to `k x k`; no full design or residual matrix is copied to host merely for Hausman.

### 5.3 Sample/design identity for Hausman

FE/RE fits store compact immutable diagnostic metadata sufficient to reject mismatched samples/designs without retaining a full second host copy of X/y.

Identity components:

- `nobs`;
- aligned entity label/code sequence signature and entity counts;
- aligned retained-row signature for formula fits when available;
- feature-name sequence after formula/model-matrix construction;
- numeric design width and intercept-presence metadata;
- effect specification;
- a **backend-native numerical fingerprint** of aligned `X`, `y`, and row order, reduced on the selected backend to O(k) scalars before host conversion.

The numerical fingerprint must include multiple independent deterministic moments, e.g. per-column/y sum, sum of squares, and an index-weighted first moment (using a deterministic row weight sequence), all accumulated in float64. It is an integrity check rather than a cryptographic hash. Comparison uses a scale-aware floating-point tolerance so the same float64 data on NumPy/CuPy/Torch are accepted while materially different samples/designs are rejected.

For raw array fits without formula names, deterministic positional slope names (`x1`, `x2`, ...) are used for coefficient matching **in addition to** the numerical fingerprint. Same `nobs`, entity counts, or shape alone is never treated as proof that the samples match.

If identity metadata are missing or disagree materially, `hausman_test` returns `applicable=False` with the precise mismatch reason rather than guessing.

### 5.4 R² accumulation

Compute R² variants during `fit()` while backend numerical arrays are available, then store only scalar results. Do not retain a second full copy of `X`/`y` for later diagnostics.

For estimators where the model coefficient includes an explicit intercept, use the existing design convention to determine centering. For FE, standard R² variants use the level slope vector and the level/entity-demeaned data as defined above; fixed effects themselves are not inserted into overall/between predictions.

For `PooledOLS` and `FamaMacBeth`, add optional `entity_ids=None` to `fit()` only to unlock panel decomposition metrics; coefficient estimates are unchanged. Absence of IDs leaves within/between fields as `None` with reasons in metadata.

Any estimator-specific row reorder (currently notably PooledOLS HAC time sorting and FirstDifference sorting/differencing) must carry diagnostic metadata through the exact same permutation/transform before sufficient statistics are accumulated.

### 5.5 Adjusted R²

Use an explicit residual-variance / total-variance definition on each estimator's **primary fit space**:

`R²_adj = 1 - (RSS / df_resid) / (TSS / df_total)`.

`df_total` is the effective total-sum-of-squares df in that fit space:

- centered fit with an identified constant / implicit absorbed mean: `n_fit - 1`;
- uncentered fit with no constant: `n_fit`.

For FE, `n_fit` is the number of retained observations while `df_resid` remains the established Stage-A residual df including absorbed effects. For BetweenOLS, `n_fit` is the number of entity means; for FirstDifferenceOLS, the number of retained first differences; for RE, the quasi-demeaned sample size; for PooledOLS, pooled sample size.

`metadata['rsquared_adj_basis']` records `fit_space`, `df_total`, and `df_resid` so the convention is auditable.

FamaMacBeth is not assigned this residual-OLS adjusted R²; its average period adjusted R² is a distinct statistic in some software and is not silently put into `PanelFitStatistics.rsquared_adj` without a separate explicit contract.

## 6. Edge-case and failure contracts

Tests must cover at least:

- unfitted models passed to diagnostic functions;
- wrong estimator types;
- FE model with no effects passed to pooling F;
- time-only/two-way FE passed to one-way FE-vs-RE Hausman;
- FE robust/clustered covariance passed to classical Hausman;
- same shapes/entity counts but materially mismatched X/y samples in Hausman;
- same data across NumPy/CuPy/Torch accepted by fingerprint tolerance;
- no common slope coefficients;
- singular PSD Hausman covariance difference with identified `d`;
- singular PSD difference with `d` outside the identified range;
- materially indefinite covariance difference;
- roundoff-level negative Hausman/pooling quantities versus materially negative violations;
- BP-LM with one entity, singleton-only entities, zero pooled RSS, and unbalanced panels;
- PooledOLS HAC sorting with unsorted `time_index` and entity IDs, proving X/y/entity diagnostic alignment after sorting;
- constant outcome / zero TSS standardized R² behavior (`0.0` plus degenerate metadata) while legacy attributes remain unchanged;
- rank-deficient pooled design using effective restriction rank;
- formula missing-row alignment for IDs;
- explicit CUDA/Torch request with unavailable backend must fail rather than fall back.

## 7. Test plan

### 7.1 Analytic unit tests

Add `dev/tests/test_panel_stage_b_diagnostics.py` with deterministic small panels and hand-computed sufficient statistics for:

- pooling F formula, no-explicit-constant correction, and df;
- balanced and unbalanced entity BP-LM;
- full-rank Hausman quadratic form;
- singular/indefinite Hausman applicability behavior;
- parameter-based within/between/overall R²;
- adjusted R² basis;
- classical model F.

### 7.2 Existing-estimator regression

Extend the Stage-A golden suite or add a Stage-B compatibility file to assert that Stage B does not change existing:

- coefficients;
- bse/t/p/CI;
- predictions;
- existing `rsquared` / legacy `PanelOLS.rsquared_within` values;
- RE variance components/theta;
- FMB beta-series covariance.

### 7.3 Three-backend parity

For every new observation-scale statistic, compare NumPy/CuPy/Torch results on the same deterministic balanced and unbalanced panels.

Hosted optional-backend tests may use existing CPU-compatible Torch coverage, but physical CuPy/Torch CUDA remains the final backend acceptance gate.

Target numeric parity unless an external implementation uses a different estimator definition:

- deterministic analytic/backend parity: `rtol <= 5e-8`, `atol <= 5e-9` by default;
- external model-statistics comparisons: tighter where definitions are identical, with any relaxed tolerance justified per field.

### 7.4 Formula tests

Cover:

- explicit/implicit intercept behavior;
- categorical terms and interactions already supported by panel formula parsing;
- missing-row alignment of `entity_ids`/`time_ids`;
- effect tokens for PanelOLS;
- diagnostics computed from exactly the retained estimation sample.

### 7.5 External alignment

Add a reproducible external-comparison script/artifact that records definitions rather than just numbers.

Python `linearmodels` comparisons:

- PanelOLS one-way and two-way: coefficients, standard R² variants, classical model F, pooled F;
- PooledOLS, BetweenOLS, FirstDifferenceOLS, RandomEffects: R² variants and model F where definitions/parameterization match;
- note any Swamy-Arora parameterization difference before comparing RE quantities.

R `plm` comparisons:

- `pFtest(within, pooling)`;
- `plmtest(pooling, type='bp', effect='individual')` on balanced and unbalanced panels;
- `phtest(within, random)` for a well-conditioned classical Hausman example.

For the singular-PSD generalized-inverse Hausman extension, use an analytic matrix fixture rather than claiming `plm`/Stata parity.

Stata is documentation/reference-only unless a licensed callable environment is available. Record its Hausman formula/interpretation, not unverifiable claimed numeric parity.

## 8. Physical GPU acceptance

Add `dev/benchmarks/validate_panel_stage_b_gpu.py` as a correctness/provenance runner.

Requirements:

- exact expected SHA;
- clean working tree;
- requested backend must actually execute (`cupy` / CUDA Torch);
- balanced and unbalanced data;
- compare new fit statistics and diagnostics against NumPy references;
- include at least PanelOLS pooling F, PooledOLS BP-LM, FE/RE Hausman prerequisites, R² variants, adjusted R², and model F;
- include an unsorted-HAC PooledOLS case with entity IDs to guard metadata permutation;
- record environment/package/GPU provenance and max absolute differences in JSON.

No performance claim is made from this runner.

## 9. Documentation plan

Update, in EN-first / CN-follow order:

- panel model documentation covering `PanelOLS`, `RandomEffects`, `PooledOLS`, and shared diagnostics;
- model/index capability table if present;
- root `CHANGELOG.md`;
- `docs/en/changelog.md`;
- `docs/cn/changelog.md`.

Documentation must explicitly state:

- parameter-based versus correlation-based R²;
- standardized zero-TSS behavior and preservation of legacy attributes;
- the preserved legacy `PanelOLS.rsquared_within` compatibility distinction for two-way FE;
- classical/homoskedastic nature of model F and pooling F;
- pooling-F implicit-constant correction;
- BP-LM is the panel error-components test, not the heteroskedasticity BP test;
- classical Hausman restrictions, data-identity checks, and explicit singular/indefinite behavior;
- the generalized-inverse singular-PSD Hausman case is a statgpu extension with rank df;
- which statistics are unavailable without entity IDs;
- three-backend/no-silent-fallback behavior.

## 10. Implementation sequence and gates

1. **Plan review gate** — audit this document against Issue #93, Stage-A contracts, external definitions, and repo workflow. Fix all HIGH and relevant MEDIUM findings before source edits.
2. **Result/API substrate** — export Stage-A result dataclasses; add `_diagnostics.py`; define applicability helpers and public exports.
3. **Sufficient-statistic helpers** — R²/model-F/pooling-F/BP helpers with NumPy tests first, written backend-generically from the start.
4. **Estimator integration** — persist small internal covariance matrices; populate `fit_statistics_`; compute pooling/BP contexts during fit; add optional entity metadata where required without changing coefficients; propagate every numerical row transform to diagnostics metadata.
5. **Hausman integration** — numerical sample/design fingerprints, common-coefficient matching, PSD/rank logic.
6. **Three-backend targeted tests** — NumPy/CuPy/Torch parity and strict-device failure behavior.
7. **External Python/R alignment** — run strongest available local baselines; if R unavailable locally, retain exact script/command and mark only that external gate remote-pending.
8. **Full hosted CI** — complete test suite, Python matrix, static/docs, maintenance, release-package/front-end gates as applicable.
9. **Physical GPU** — exact clean-head CuPy/Torch runner.
10. **Auto-fix review loop** — fresh code review after evidence; no unresolved CRITICAL/HIGH/in-scope MEDIUM findings before promotion.
11. **Docs/changelog sync** — no user-visible capability advertised before tests and definitions are final.

## 11. Explicit non-goals for Stage B

- robust/auxiliary-regression Hausman test;
- two-way random effects or two-way BP-LM as an RE-vs-pooled selector;
- RandomEffects robust covariance;
- HC0/HC2/HC3 expansion;
- Driscoll–Kraay;
- cluster small-sample expansion;
- FamaMacBeth residual-OLS covariance or residual-OLS model F;
- changing Stage-A estimator coefficients, covariance normalizations, prediction behavior, or legacy R² attributes.

These remain Stage C or later work unless a blocking correctness dependency is discovered.

## 12. Plan-review findings closed before implementation

- **[HIGH][INFER] fixed** — pooling F now specifies the implicit-common-constant correction and corresponding numerator-df decrement when the FE design lacks an explicit constant, matching the nested linearmodels definition.
- **[HIGH][API/INFER] fixed** — Hausman sample compatibility no longer relies on `nobs`/entity order alone; the plan requires a backend-native O(k) numerical X/y/order fingerprint.
- **[HIGH][BACKEND] fixed** — PooledOLS HAC sorting must apply the identical permutation to entity diagnostic metadata before BP/R²/fingerprint accumulation.
- **[MEDIUM][INFER] fixed** — standardized fit-stat R² adopts linearmodels' zero-TSS `0.0` convention with explicit degenerate metadata while preserving existing legacy attributes.
- **[MEDIUM][API] fixed** — Hausman only needs an internal small covariance matrix; Stage B will not create a new universal public `cov_params_` contract.
- **[MEDIUM][INFER] fixed** — singular-PSD generalized-inverse Hausman behavior is labeled explicitly as a statgpu extension and validated analytically rather than presented as direct plm/Stata parity.

## 13. Completion criteria

Stage B can be called COMPLETE only when:

- the three public diagnostic tests return structured `PanelTestResult` objects with documented applicability behavior;
- `fit_statistics_` is populated consistently where defined;
- existing numerical estimator behavior remains frozen by regression tests;
- all new observation-scale operations have NumPy/CuPy/Torch parity with no silent fallback;
- linearmodels/plm definition alignment is recorded and tested where callable;
- formula/missing-row alignment is covered;
- hosted CI passes on the final exact head;
- physical CuPy/Torch evidence passes on the final exact clean head;
- fresh review has no unresolved CRITICAL, HIGH, or in-scope MEDIUM findings;
- EN/CN docs and all three changelogs are synchronized.
