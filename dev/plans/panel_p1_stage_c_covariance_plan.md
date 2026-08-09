# Panel Tier-1 Stage C — covariance completion plan

Issue: #93  
Base: PR #122 / merge commit `d5dd79956a17807b11b3c8ffdbd0b8686c34cc9e`  
Branch: `agent/panel-p1-stage-c-covariance`

## 1. Scope and impact classification

Stage C completes the covariance/inference work promised by Issue #93 while preserving all Stage-A/Stage-B coefficient estimates, fitted values, predictions, default covariance behavior, diagnostic definitions, and strict-device semantics.

Active impact axes:

- **Public API** — covariance names/options expand, `RandomEffects` gains covariance configuration, and covariance metadata becomes more explicit.
- **Inference** — HC0/HC2/HC3, robust RandomEffects inference, Driscoll–Kraay, cluster corrections, standard errors, test statistics, p-values, confidence intervals, and covariance matrices.
- **Backend** — covariance bread/meat, leverage, grouped score accumulation, and kernel lag accumulation must remain NumPy/CuPy/Torch native.
- **Formula/data alignment** — cluster/time/entity metadata must follow Patsy missing-row filtering and any estimator-specific ordering/transformation.
- **Benchmark/performance** — HC2/HC3 and Driscoll–Kraay add new backend kernels/reductions; correctness precedes performance, but representative physical timing/memory evidence is required before claiming Stage C complete.
- **Docs/artifacts** — EN/CN panel docs, changelogs, physical validation, benchmark evidence, and benchmark coverage metadata where applicable.

Inactive axes:

- **Loss / penalty / solver / CV** — covariance estimation does not alter an optimization objective, regularization path, solver, or tuning layer.

Validation target: `remote-full` before Stage C is called COMPLETE. If the only missing gate is physical GPU/R evidence, the correct hard exit is `PARTIAL_REMOTE_PENDING`.

## 2. Capability decisions

| Model | backend | inference | formula | covariance work in Stage C |
| --- | --- | --- | --- | --- |
| `PanelOLS` | three-backend | supported | supported | preserve nonrobust/HC1/clustered; add HC0/HC2/HC3 and Driscoll–Kraay; explicit one-/two-way cluster correction contract |
| `RandomEffects` | three-backend | supported | supported | preserve nonrobust default; add HC0/HC1/HC2/HC3, one-/two-way clustered, and Driscoll–Kraay on quasi-demeaned GLS fit space |
| `PooledOLS` | three-backend | supported | supported | preserve nonrobust/HC1/clustered/row-HAC; add HC0/HC2/HC3 and Driscoll–Kraay; explicit cluster correction contract |
| `BetweenOLS` | three-backend | supported | supported | preserve nonrobust/HC1; add HC0/HC2/HC3 on entity-mean regression; no Driscoll–Kraay because the fit space has one row per entity rather than a time-indexed panel score process |
| `FirstDifferenceOLS` | three-backend | supported | supported | preserve nonrobust/HC1; add HC0/HC2/HC3 on the retained first-difference regression; no Stage-C Driscoll–Kraay until a gap-aware differenced-time score contract is separately specified |
| `FamaMacBeth` | three-backend | supported | supported | unchanged; its beta-series covariance remains model-specific (`nonrobust` / `newey-west`) and is not routed through residual-OLS covariance |

Capability notes:

- `CV`: non-tunable/not applicable for all touched panel covariance capabilities.
- No supported existing covariance path is removed or renamed.
- A public covariance option is not declared supported until its NumPy/CuPy/Torch path and inference outputs are tested.

## 3. Backward-compatibility contract

Stage C is additive.

1. Existing `cov_type='nonrobust'` behavior remains unchanged.
2. Existing `cov_type='robust'` remains the historical **HC1** contract. It is not silently redefined.
3. `hc1` may be accepted as an explicit alias for `robust`; the canonical stored/displayed name may remain `robust` for backward compatibility.
4. Existing `PooledOLS(cov_type='hac')` remains the current row-order Newey–West/Bartlett estimator. It is **not** renamed or reinterpreted as Driscoll–Kraay.
5. Existing one-way/two-way `clustered` covariance remains un-debiased by default. Any finite-group correction is opt-in so old numerical output does not change.
6. Stage-B classical Hausman remains restricted to the matched nonrobust FE/RE covariance pair. Adding robust covariance to `RandomEffects` does not relabel robust/clustered/DK pairs as a valid classical Hausman test.
7. Existing `PanelOLS` legacy public residual-df/t-inference convention remains frozen where Stage B froze it. New covariance modes document their own correction basis without rewriting old default outputs.
8. Coefficients, fitted values, variance-component estimates, theta, R² definitions, specification-test definitions, formula parsing, and prediction semantics do not change solely because Stage C exists.

A regression test must compare pre-Stage-C default covariance/inference outputs against frozen Stage-B expectations.

## 4. Public API contract

### 4.1 Covariance names

The common residual-OLS covariance dispatcher will recognize, where the estimator capability matrix permits:

- `nonrobust`
- `robust` (historical HC1)
- `hc0`
- `hc1` (alias of `robust`)
- `hc2`
- `hc3`
- `clustered`
- `hac` (legacy row-order Newey–West only where already supported)
- `driscoll-kraay` with aliases `dk` and `kernel`

Estimator constructors validate against their own allowed subset. Unsupported names fail before inference with a precise `ValueError`; there is no fallback to another covariance.

### 4.2 New covariance options

Use shared names where meaningful:

- `bandwidth: Optional[int] = None`
- `kernel: str = 'bartlett'`
- `group_debias: bool = False`

`group_debias` affects only clustered covariance. Supplying it for another covariance is permitted as inert constructor state only if required for exact sklearn-style parameter identity; otherwise the estimator should reject semantically incompatible fit-time use rather than silently changing another covariance.

For `RandomEffects`, constructor support expands from

```python
RandomEffects(alpha=0.05, device='auto', n_jobs=None)
```

to a backward-compatible form containing at least:

```python
RandomEffects(
    cov_type='nonrobust',
    alpha=0.05,
    bandwidth=None,
    kernel='bartlett',
    group_debias=False,
    device='auto',
    n_jobs=None,
)
```

The old positional meaning of existing parameters must not be broken. New parameters should therefore be inserted only in a way compatible with the repository's constructor/clone policy; if positional compatibility cannot be preserved, make new covariance options keyword-only rather than shifting existing positional slots.

### 4.3 Fit metadata

Cluster/time side arrays remain fit-time data arguments. API rules:

- one-way clustering accepts an aligned 1-D label vector (and the historical `(n,1)` form where already accepted);
- two-way clustering accepts exactly two aligned cluster dimensions;
- Driscoll–Kraay requires aligned time identifiers after formula missing-row filtering;
- `RandomEffects` may begin validating/storing `time_ids` only when the selected covariance needs time ordering/grouping; old nonrobust calls with unused `time_ids` must not acquire a new failure mode unless the metadata is malformed in a path that actually consumes it;
- full numerical X/y/residual/score arrays are never copied to CPU merely to compute covariance; label factorization/order metadata may be represented on CPU.

## 5. HC0 / HC1 / HC2 / HC3 definitions

All HC estimators are defined on the estimator's **actual numerical fit-space regression**, not on a silently reconstructed full dummy-variable regression.

For fit-space design `Z` (`n x k`), residual vector `e`, and bread `B = (Z'Z)^+`, define leverage

```text
h_i = z_i' B z_i.
```

The meat is

```text
HC0: sum_i z_i z_i' e_i^2
HC1: HC0 * correction_existing
HC2: sum_i z_i z_i' e_i^2 / (1 - h_i)
HC3: sum_i z_i z_i' e_i^2 / (1 - h_i)^2
```

and covariance is `B * meat * B` using the repository's existing unnormalized-matrix convention.

Fit spaces:

- `PooledOLS`: pooled level design including its fitted intercept.
- `PanelOLS`: the existing effect-transformed slope design used for coefficient estimation.
- `RandomEffects`: Swamy–Arora quasi-demeaned `X_star` and GLS residuals.
- `BetweenOLS`: entity-mean regression design including its intercept.
- `FirstDifferenceOLS`: retained first-difference regression design.

This choice is explicit because HC2/HC3 leverage from an absorbed/transformed regression need not equal leverage from a literal full dummy expansion. Documentation and external comparisons must call it **transformed-fit-space HC2/HC3**.

Implementation requirements:

- compute leverage rowwise without materializing an `n x n` hat matrix, e.g. `sum((Z @ B) * Z, axis=1)`;
- use pseudoinverse semantics consistently when the fit-space Gram matrix is rank deficient;
- preserve backend dtype/device;
- reject materially invalid `1-h_i <= 0` for HC2/HC3 with a precise error rather than clipping leverage into a valid-looking answer;
- normalize only tiny dimensionless roundoff excursions around the `[0,1]` leverage boundary using a scale-independent machine-epsilon tolerance;
- tests cover high leverage, rank deficiency, rescaling, and exact/near-unit leverage boundaries.

Existing `robust` continues to use its historical HC1 correction (`n / df_resid` or the estimator's pre-existing explicit correction). New HC0/HC2/HC3 do not modify that path.

## 6. RandomEffects robust covariance

`RandomEffects` continues to estimate Swamy–Arora variance components and coefficients exactly as in Stage B. Covariance changes only the inference calculation after `X_star`, `y_star`, `beta_gls`, and `resid_gls` exist.

Stage-C robust/cluster/DK covariance uses the quasi-demeaned GLS score

```text
s_it = x*_it e*_it.
```

Contracts:

- `hc0/hc1/hc2/hc3` use `X_star` / `resid_gls`;
- clustered covariance groups `s_it` using aligned level-observation cluster labels; the quasi-demeaning transformation does not reorder observations;
- Driscoll–Kraay groups these transformed scores by aligned time IDs;
- coefficient estimates and variance components are independent of `cov_type`;
- `fit_statistics_` remains Stage-B behavior;
- classical Hausman diagnostic covariance remains based on the nonrobust contract and robust RE fits remain inapplicable to the classical Stage-B Hausman path.

## 7. One-way and two-way clustered covariance

The uncorrected cluster sandwich remains

```text
S_g = sum_{i in g} z_i e_i
M_g = sum_g S_g S_g'
V_g = B M_g B.
```

Two-way covariance uses inclusion-exclusion:

```text
V = V(cluster_1) + V(cluster_2) - V(intersection).
```

### 7.1 Group debias correction

When `group_debias=True`, each one-way covariance component is multiplied by

```text
c(G,n) = [G / (G - 1)] * [(n - 1) / n].
```

For two-way clustering, apply the correction separately to cluster 1, cluster 2, and the intersection component using each component's own number of groups before inclusion-exclusion. This follows the maintained `linearmodels` contract.

Requirements:

- `group_debias=False` is default and reproduces existing statgpu cluster output;
- requesting group debias with fewer than two groups in a required component raises a precise `ValueError`;
- cluster labels may be strings/objects on CPU input; factorization produces integer metadata without moving numerical score arrays off device;
- malformed cluster dimensions and post-formula length mismatches fail explicitly;
- two-way cluster identity uses collision-free paired-label factorization, not an arithmetic encoding that can alias distinct pairs;
- tests cover one-way, two-way, intersection, one-group invalidity, unbalanced panels, and formula row filtering.

## 8. Driscoll–Kraay covariance

Primary definition/alignment: `linearmodels==7.0` `DriscollKraay`.

For fit-space scores `s_it = z_it e_it`, aggregate by time:

```text
g_t = sum_i s_it.
```

Let `T` be the number of distinct observed time periods after alignment. Form the kernel HAC covariance of the ordered `g_t` series:

```text
Gamma_0 = sum_t g_t g_t'
Gamma_l = sum_{t=l+1}^T g_t g_{t-l}'
M_DK = Gamma_0 + sum_{l=1}^L w_l (Gamma_l + Gamma_l').
```

The final normalization/scaling must be algebraically matched to `linearmodels 7.0` for the aligned unweighted case rather than guessed from the existing row-HAC implementation. Tests pin the exact scale.

### 8.1 Time contract

- DK requires at least two distinct observed time periods.
- Scores are grouped by time label; input row order within a time period is irrelevant.
- Distinct time groups are ordered by the canonical/factorized time ordering used by the model metadata. Numeric/datetime naturally ordered values should follow sorted temporal order. For non-orderable arbitrary labels, require an explicit deterministic order contract rather than silently using hash/set order.
- Unbalanced panels are supported: each time aggregate uses only observed rows.
- Missing formula rows remove the corresponding time metadata before grouping.

### 8.2 Bandwidth

- explicit bandwidth must be a non-negative integer;
- cap effective bandwidth at `T-1`;
- `bandwidth=None` follows the pinned `linearmodels` default `floor(4 * (T / 100)^(2/9))`, where `T` is the number of distinct time periods, not number of observations;
- effective bandwidth is retained in covariance metadata or an auditable fitted inference field/test fixture.

### 8.3 Kernels and aliases

Stage C supports linearmodels-compatible aliases for:

- Bartlett / Newey–West;
- Parzen / Gallant;
- Quadratic Spectral (`qs`) / Andrews.

Kernel weights are implemented in a small shared helper with analytic unit tests. Invalid kernel names fail explicitly. The legacy row-HAC path may remain Bartlett-only to preserve its historical API; adding DK kernel aliases does not silently broaden/change legacy HAC semantics unless explicitly tested and documented.

## 9. Small-sample and inference-distribution conventions

The covariance correction and the reference distribution are separate contracts.

- existing `nonrobust` inference keeps the existing Student-t behavior;
- existing `robust` keeps its existing HC1 scale and normal-reference inference;
- new `hc0/hc1/hc2/hc3`, clustered, and DK use the existing sandwich-covariance normal-reference inference unless a separately named finite-sample t/F contract is introduced (not planned in Stage C);
- `group_debias` changes the clustered covariance magnitude only; it does not silently switch p-values to a `t_{G-1}` reference;
- DK finite-model df correction must match the pinned linearmodels definition for the aligned fit-space rank/effect count and be documented per estimator;
- for `PanelOLS`, DK/HC additions do not rewrite the historical Stage-A `robust` output. Where a new correction needs effect-rank accounting, use the Stage-B standard diagnostic/effect rank as explicit metadata rather than silently borrowing legacy df.

External tests must assert covariance/SE and record the correction/reference-distribution convention; a p-value mismatch caused solely by different documented reference distributions is not fixed by corrupting covariance scaling.

## 10. Backend-native algorithm contract

### 10.1 Numerical arrays

The following remain on the selected backend throughout covariance accumulation:

- fit-space design;
- residuals;
- row scores;
- leverage vector;
- grouped score matrices;
- kernel lag products;
- bread/meat/covariance matrices until the existing final inference conversion.

No full `X`, residual, score, or leverage array may be copied from CuPy/Torch to CPU.

### 10.2 Metadata

The following may be represented/factorized on CPU:

- cluster labels;
- entity/time labels;
- integer group-code metadata;
- small final scalar counts/configuration.

Codes are transferred back to the active device for scatter/reduce. GPU implementation should reuse the project’s existing scatter-add/group helpers rather than Python loops over observations.

### 10.3 Grouped score primitive

Add/reuse one backend-neutral primitive that computes grouped sums of an `n x k` score matrix from integer group codes. It must support NumPy, CuPy, and Torch without fallback. Cluster and DK code should reuse this primitive instead of implementing separate grouping semantics.

## 11. Estimator integration details

### `PanelOLS`

- Extend allowed covariance names without changing current defaults.
- HC2/HC3 use the transformed estimation design.
- Cluster side-array semantics retain one-/two-way support and gain `group_debias`.
- DK requires aligned `time_ids`; entity-only and two-way FE are both allowed if the transformed score process and effect-rank correction are well-defined.
- Robust/cluster/DK fits remain inapplicable to classical Stage-B Hausman.

### `RandomEffects`

- Add covariance constructor configuration without breaking old positional calls.
- Pass `X_star` / `resid_gls` to shared covariance inference.
- Accept aligned cluster metadata and consume `time_ids` for DK.
- Add formula/array tests for explicit constants, balanced/unbalanced panels, robust/DK inference, and no coefficient drift.

### `PooledOLS`

- Extend HC types and DK while preserving legacy `hac`.
- DK groups by time and therefore must not reuse the legacy HAC pre-sort/reorder branch as its statistical definition.
- Existing BP-LM/entity diagnostics remain aligned regardless of covariance path.

### `BetweenOLS` / `FirstDifferenceOLS`

- Add HC0/HC2/HC3 only.
- Preserve coefficient, transform, and legacy robust semantics.
- Explicitly reject DK/clustered unless a separately reviewed estimator-specific contract is added.

### `FamaMacBeth`

- No Stage-C residual-sandwich integration.
- Regression tests verify its existing covariance and inference are unchanged.

## 12. External alignment matrix

Pin external packages/definitions rather than relying on version-floating behavior.

### 12.1 `linearmodels==7.0`

Executable CI comparisons for aligned unweighted data:

- `PooledOLS`: robust/clustered/DK where definitions align;
- `PanelOLS`: HC1/cluster/DK and correction/rank structure; HC0/HC2/HC3 only where linearmodels exposes an equivalent definition;
- `RandomEffects`: robust/clustered/DK covariance and SE, including explicit constant and unbalanced cases;
- DK bandwidth/kernel aliases and group-debias scaling.

Every test states `debiased`, `group_debias`, effect count, cluster dimensions, kernel, bandwidth, and intercept/effect specification.

### 12.2 HC2/HC3 analytic/Python baseline

Because linearmodels does not expose HC2/HC3 as the primary Panel API, compare transformed-fit-space HC2/HC3 against:

1. a direct analytic sandwich computation; and
2. `statsmodels` OLS robust covariance on the **same transformed design/residual fit space** where executable.

Do not compare an absorbed-FE HC2/HC3 result to a full dummy-regression HC2/HC3 and then change statgpu merely to force equality.

### 12.3 R / documentation references

Where available, record R `plm`/`sandwich` and Stata/fixest definitions as definition references. R absence is remote/external pending rather than a reason to weaken Python/analytic gates.

## 13. Test matrix

### 13.1 Covariance primitives

Add focused tests for:

- HC0/HC1 identity and scale;
- HC2/HC3 leverage formulas;
- high-leverage/near-unit leverage rejection;
- rank-deficient bread/pseudoinverse behavior;
- one-way cluster grouped meat;
- two-way inclusion-exclusion;
- group-debias factors including intersection;
- collision-free paired cluster labels;
- DK time-score aggregation;
- Bartlett/Parzen/QS kernel weights;
- bandwidth default/cap/validation;
- unsorted labels, repeated time periods, and unbalanced panels.

### 13.2 Estimator matrix

For supported estimator/covariance combinations verify:

- coefficients unchanged across covariance choices;
- `_panel_cov_params_raw`/public BSE/t-or-z/p/CI consistency;
- NumPy reference precision;
- Torch CPU maintained regression where the project uses it as a compatibility gate;
- CuPy/Torch CUDA physical parity;
- explicit-device provenance/no fallback;
- formula and array API parity;
- intercept/categorical/interaction/missing-row behavior where existing formula syntax supports it.

### 13.3 Compatibility/golden tests

- freeze Stage-B default covariance outputs before changing dispatch;
- verify `robust` equals prior HC1 behavior;
- verify `PooledOLS.hac` remains prior row-HAC behavior;
- verify FamaMacBeth covariance is untouched;
- verify Stage-B Hausman/pooling/BP/fit-statistics results do not change for equivalent fits solely because covariance options were added.

### 13.4 Invalid contracts

Test precise failures for:

- unsupported covariance per estimator;
- DK without time IDs;
- fewer than two time periods;
- invalid bandwidth/kernel;
- malformed one-/two-way clusters;
- group debias with too few groups;
- HC2/HC3 undefined leverage;
- explicit `device='cuda'/'torch'` without the requested backend.

## 14. Physical GPU validation

Create `dev/benchmarks/validate_panel_stage_c_gpu.py` as a correctness/provenance gate, not a timing benchmark.

At minimum, per CuPy and Torch CUDA cover:

- Pooled HC0/HC2/HC3;
- PanelOLS one-way and two-way FE HC2/HC3;
- RandomEffects HC1/HC2/HC3, including explicit constant and unbalanced data;
- one-way cluster and two-way cluster, uncorrected and `group_debias=True`;
- DK Bartlett default bandwidth plus explicit non-Bartlett kernel case;
- Pooled legacy HAC regression;
- formula-aligned or equivalent side-array ordering contract where feasible.

For every case record:

- requested backend and executed backend;
- exact git SHA and clean tree;
- covariance/SE/t-or-z/p/CI differences vs NumPy;
- coefficients and Stage-B fit-statistics invariance;
- covariance configuration, effective bandwidth/group counts where applicable;
- environment/package/GPU metadata.

Any runner change after physical acceptance invalidates final acceptance of the old artifact for the changed runner contract, per `RELEASING.md`.

## 15. Performance gate

Create a bounded Stage-C covariance benchmark with representative small/medium/large panel shapes and synchronized timing.

Primary questions:

- does HC2/HC3 leverage avoid an `O(n^2)` hat matrix and scale approximately `O(n k^2)`/backend-equivalent;
- does grouped cluster/DK accumulation avoid observation-wise Python loops and full GPU-to-CPU numerical transfer;
- at target GPU-appropriate sizes, does the GPU path avoid an obvious regression caused by metadata conversion/synchronization.

No speedup promise is part of Stage C. Performance is blocking only if review/benchmark shows a material regression, pathological complexity, or transfer-dominated implementation. If optimization is needed, follow the workflow budget: one profiling pass, at most two algorithmic/kernel attempts, and one re-benchmark per attempt.

Machine-readable timing output belongs under `results/` and must remain separate from correctness-only validation evidence.

## 16. Documentation and artifacts

Update:

- `docs/en/models/panel.md` first, then `docs/cn/models/panel.md`;
- root `CHANGELOG.md`, detailed EN/CN changelogs;
- public constructor/API docs generated from docstrings as applicable;
- benchmark/coverage catalog entries if Stage-C evidence is promoted to the frontend;
- physical validation review record after remote acceptance.

Docs must state:

- `robust == historical HC1`;
- transformed-fit-space HC2/HC3 definition and limitation;
- legacy row-HAC vs Driscoll–Kraay distinction;
- DK time/bandwidth/kernel semantics;
- cluster one-/two-way and `group_debias` semantics;
- covariance reference distribution and small-sample correction choices;
- RandomEffects covariance uses quasi-demeaned GLS scores;
- three-backend/no-fallback behavior;
- unsupported covariance combinations and failure behavior.

## 17. Implementation sequence

1. Freeze Stage-B default covariance/inference regression fixtures.
2. Add shared covariance config normalization, grouped-score primitive, kernel weights, HC leverage/meat helpers.
3. Implement HC0/HC2/HC3 in shared dispatcher while preserving `robust` HC1.
4. Harden cluster 1-/2-way metadata and optional group-debias correction.
5. Implement Driscoll–Kraay and its time/kernel/bandwidth contract.
6. Integrate PooledOLS and PanelOLS.
7. Integrate RandomEffects on quasi-demeaned fit space.
8. Integrate HC extensions for BetweenOLS and FirstDifferenceOLS; explicitly preserve FamaMacBeth.
9. Add analytic/statsmodels and pinned linearmodels 7.0 external tests.
10. Add maintained Torch regression and physical CuPy/Torch runner contract.
11. Add performance benchmark and machine-readable schema checks.
12. Run targeted/full hosted gates.
13. Run `.claude/skills/code-review.md` in auto-fix mode; fix and repeat until no CRITICAL/HIGH/relevant MEDIUM finding remains locally.
14. Run exact-clean-head physical GPU validation and required remote/external gates.
15. Promote evidence/artifacts only after physical acceptance, rerun exact-final-head hosted gates, and perform one fresh review that does not inherit the prior READY conclusion.

## 18. Acceptance checklist

Stage C is complete only when all applicable items are true:

- [ ] Existing Stage-B default covariance/inference numerical behavior is preserved.
- [ ] HC0/HC1/HC2/HC3 definitions and transformed-fit-space leverage semantics are explicit and tested.
- [ ] RandomEffects robust/HC/cluster/DK covariance works without changing coefficient/variance-component estimates.
- [ ] Driscoll–Kraay matches the pinned linearmodels 7.0 scale/bandwidth/kernel contract on aligned cases.
- [ ] Legacy Pooled row-HAC remains distinct and unchanged.
- [ ] One-way/two-way clustering has explicit, backward-compatible `group_debias` semantics.
- [ ] Every supported new covariance path works on NumPy/CuPy/Torch without silent fallback.
- [ ] No full GPU numerical design/residual/score/leverage copy is required for covariance accumulation.
- [ ] Formula missing-row and panel metadata alignment is tested.
- [ ] HC2/HC3 analytic/statsmodels and linearmodels covariance alignments pass where definitionally comparable.
- [ ] Stage-B diagnostics/fit statistics remain regression-clean.
- [ ] EN/CN docs and changelogs are synchronized.
- [ ] Performance/complexity gate has no unresolved material regression.
- [ ] Exact-head hosted tests/compatibility/release/frontend gates pass.
- [ ] Exact-clean-head physical CuPy/Torch Stage-C validation passes and immutable evidence is recorded.
- [ ] Fresh final code review has zero unresolved CRITICAL/HIGH/relevant MEDIUM findings.

## 19. Non-goals

Stage C does not add:

- robust/auxiliary-regression Hausman variants;
- cluster-robust finite-group t/F reference distributions;
- weighted panel estimation;
- Panel IV/2SLS/GMM;
- high-dimensional fixed-effect absorption;
- DID/event study;
- dynamic-panel GMM;
- FamaMacBeth covariance redesign;
- PCSE/Beck–Katz;
- Conley/spatial HAC.

Any of these requires a separately reviewed statistical/API contract.