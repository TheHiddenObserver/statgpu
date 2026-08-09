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
- **Formula/data alignment** — cluster/time/entity metadata must follow Patsy missing-row filtering and estimator-specific ordering/transformation.
- **Benchmark/performance** — HC2/HC3 and Driscoll–Kraay add backend kernels/reductions; correctness precedes performance, but representative physical timing/memory evidence is required before Stage C is called complete.
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
| `BetweenOLS` | three-backend | supported | supported | preserve nonrobust/HC1; add HC0/HC2/HC3 on entity-mean regression; no DK because the fit space has one row per entity rather than a time-indexed panel score process |
| `FirstDifferenceOLS` | three-backend | supported | supported | preserve nonrobust/HC1; add HC0/HC2/HC3 on the retained first-difference regression; no Stage-C DK until a gap-aware differenced-time score contract is separately specified |
| `FamaMacBeth` | three-backend | supported | supported | unchanged; beta-series covariance remains model-specific (`nonrobust` / `newey-west`) and is not routed through residual-OLS covariance |

`CV` is non-tunable/not applicable for all touched capabilities. A covariance option is not declared supported until its NumPy/CuPy/Torch inference path is tested.

## 3. Backward-compatibility contract

Stage C is additive.

1. `cov_type='nonrobust'` remains unchanged.
2. `cov_type='robust'` remains the historical **HC1** contract; it is not redefined.
3. `hc1` is an explicit alias normalized internally to canonical `robust`.
4. `PooledOLS(cov_type='hac')` remains the historical row-order Newey–West/Bartlett estimator; it is not renamed/reinterpreted as Driscoll–Kraay.
5. Existing clustered covariance remains un-debiased by default. `group_debias=True` is opt-in, so old numerical output does not change.
6. Stage-B classical Hausman remains restricted to matched nonrobust FE/RE covariance. Robust/HC/cluster/DK RE fits remain inapplicable to this classical Hausman path.
7. Existing `PanelOLS` legacy public residual-df/t-inference behavior remains frozen where Stage B froze it. New covariance modes have their own documented correction basis.
8. Coefficients, fitted values, variance components, theta, R², specification tests, formula parsing, and prediction semantics do not change solely because Stage C exists.

Golden regression tests must freeze pre-Stage-C default covariance/inference outputs before dispatch changes.

## 4. Public API contract

### 4.1 Covariance names

Where permitted by the estimator matrix, normalize:

- `nonrobust`
- `robust`
- `hc0`
- `hc1 -> robust`
- `hc2`
- `hc3`
- `clustered`
- `hac` (legacy row-HAC only where already supported)
- `driscoll-kraay`, aliases `dk` and `kernel`

Unsupported names fail precisely; there is no covariance fallback.

### 4.2 Covariance options

Shared names where meaningful:

- `bandwidth: Optional[int] = None`
- `kernel: str = 'bartlett'`
- `group_debias: bool = False`

`group_debias=True` with a non-cluster covariance raises a precise fit-time `ValueError`. Existing default-valued bandwidth/kernel constructor state remains inert where it historically was inert; old calls do not begin failing merely because default metadata exists.

### 4.3 Positional compatibility

Current `RandomEffects(alpha=0.05, device='auto', n_jobs=None)` positional meaning is frozen. New covariance arguments are keyword-only after existing parameters:

```python
RandomEffects(
    alpha=0.05,
    device='auto',
    n_jobs=None,
    *,
    cov_type='nonrobust',
    bandwidth=None,
    kernel='bartlett',
    group_debias=False,
)
```

Other estimator constructors likewise append/make keyword-only new options rather than shifting old positional parameters. `get_params`, `set_params`, clone, and exact constructor identity are blocking tests.

### 4.4 Fit-time metadata

- one-way cluster: aligned 1-D labels (and historical `(n,1)` where accepted);
- two-way cluster: exactly two aligned cluster dimensions;
- `PanelOLS` DK uses existing `time_ids`;
- `PooledOLS` DK uses existing `time_index`;
- `RandomEffects.fit` adds `cluster=None`; existing `time_ids` is consumed only by DK;
- formula missing-row filtering must align cluster/time arrays through the shared side-array machinery;
- old RE nonrobust/HC calls with unused `time_ids` do not acquire a new rejection solely because Stage C exists;
- full numerical X/y/residual/score arrays are never copied to CPU for covariance; label metadata may be factorized on CPU.

## 5. HC0 / HC1 / HC2 / HC3

All HC estimators are defined on the estimator's **actual numerical fit-space regression**, not on a reconstructed full dummy-variable regression.

For fit-space design `Z`, residuals `e`, and `B=(Z'Z)^+`:

```text
h_i = z_i' B z_i
HC0 meat = sum_i z_i z_i' e_i^2
HC1 meat = HC0 meat * historical correction
HC2 meat = sum_i z_i z_i' e_i^2/(1-h_i)
HC3 meat = sum_i z_i z_i' e_i^2/(1-h_i)^2
V = B * meat * B.
```

Fit spaces:

- Pooled: level design including fitted intercept;
- PanelOLS: effect-transformed slope design;
- RandomEffects: Swamy–Arora quasi-demeaned `X_star`;
- Between: entity-mean design including intercept;
- FirstDifference: retained differenced design.

This is documented as **transformed-fit-space HC2/HC3**. It is not claimed equal to full dummy-regression HC2/HC3.

Implementation requirements:

- compute `h_i` rowwise as `sum((Z @ B) * Z, axis=1)`, never an `n x n` hat matrix;
- use pseudoinverse consistently for supported rank-deficient fit spaces;
- keep leverage/score arrays backend-native;
- reject materially invalid leverage and any numerically unit leverage that makes HC2/HC3 undefined; tiny dimensionless roundoff outside `[0,1]` may be normalized with a machine-epsilon guard only;
- existing `robust` continues its historical estimator-specific HC1 correction unchanged.

## 6. RandomEffects covariance

Swamy–Arora estimation is untouched. All Stage-C covariance uses the already-computed quasi-demeaned regression:

```text
score_it = x*_it e*_it.
```

- HC0/1/2/3 use `X_star`/`resid_gls`;
- clustered covariance groups transformed scores using aligned level-observation cluster labels;
- DK groups transformed scores by aligned time IDs;
- covariance choice never changes coefficients/variance components/theta;
- Stage-B fit statistics remain unchanged;
- robust RE fits are explicitly rejected by classical Stage-B Hausman.

## 7. One-way/two-way cluster covariance

Uncorrected:

```text
S_g = sum_{i in g} z_i e_i
M_g = sum_g S_g S_g'
V_g = B M_g B
V_two_way = V_g1 + V_g2 - V_intersection.
```

With `group_debias=True`, multiply each one-way **meat component before inclusion-exclusion** by

```text
c(G,n) = [G/(G-1)] * [(n-1)/n].
```

For two-way clustering, cluster 1, cluster 2, and intersection each use their own group count. This matches linearmodels 7.0 group-debias semantics.

Contracts:

- default `False` reproduces existing statgpu cluster covariance exactly;
- no extra linearmodels `extra_df` scale is inserted into the legacy default cluster path;
- `group_debias=True` with fewer than two groups in any required component raises;
- paired intersection groups use exact paired-label factorization (`np.unique(..., axis=0)`/equivalent) or another proven collision-free representation; no overflow-prone ad-hoc encoding;
- numerical grouped scores stay on device; only labels/codes may use CPU metadata.

## 8. Driscoll–Kraay

Primary alignment: official `linearmodels==7.0` `DriscollKraay` source/docs.

Aggregate fit-space scores by ordered observed time:

```text
g_t = sum_i z_it e_it.
```

For a kernel weight vector `w_j`, define

```text
M_DK = sum_t g_t g_t'
     + sum_{j=1}^{T-1} w_j * sum_{t=j+1}^T
         (g_t g_{t-j}' + g_{t-j} g_t'),
```

where kernels may set some `w_j=0`; the support is kernel-specific (Section 8.4).

Let `B=(Z'Z)^+`, `n` be fit-space observations, `r=rank(Z)`, and `extra_df` be nuisance parameters outside columns of `Z`.

### 8.1 Full-rank external contract

For `rank(Z)=k_columns`, Stage C uses the debiased linearmodels-compatible scale

```text
scale_DK = n/(n-extra_df-k_columns)
V_DK = scale_DK * B M_DK B.
```

This is algebraically equivalent to linearmodels 7.0's `xpxi=inv(Z'Z/n)`, grouped `cov_kernel`, `T/n`, and `_scale=n/(n-extra_df-k_columns)` with `debiased=True`. Full-rank tests compare the final covariance exactly within numerical tolerance.

### 8.2 Rank-deficient statgpu extension

linearmodels DK assumes invertible fit-space Gram. For a statgpu fit that validly reaches covariance with `r<k_columns`:

```text
scale_DK = n/(n-extra_df-r),  B=(Z'Z)^+.
```

This is a documented statgpu extension, not claimed external equality. If the denominator is nonpositive, DK raises explicitly.

`extra_df`:

- Pooled: 0;
- RandomEffects: 0;
- PanelOLS: Stage-B standard fixed-effect nuisance rank (`N`, `T`, or `N+T-C`); transformed slope rank is `r`.

### 8.3 Time ordering

- require at least two distinct observed periods;
- aggregate before lagging; row order within a period is irrelevant;
- order groups by sorted unique time label, matching grouped-then-sort external behavior;
- numeric/datetime/homogeneous strings are deterministic; mixed non-orderable labels are rejected rather than using hash/set order;
- unbalanced panels are supported;
- formula-removed rows remove their time metadata before grouping.

### 8.4 Kernel-specific bandwidth/support

`bandwidth` for the public Stage-C DK API is a non-negative integer (bool rejected). This is intentionally narrower than accepting arbitrary floating values; external comparisons use the same integer values.

Default:

```text
bw = floor(4 * (T/100)^(2/9)).
```

Kernel aliases/formulas follow linearmodels:

**Bartlett / Newey–West**

```text
w_0=1,
w_j = 1 - j/(bw+1), j=1,...,min(bw,T-1),
w_j = 0 beyond bw.
```

**Parzen / Gallant**

For `z_j=j/(bw+1)` and `j<=min(bw,T-1)`:

```text
w_j = 1 - 6 z_j^2 + 6 z_j^3,  z_j <= 1/2
w_j = 2(1-z_j)^3,             z_j > 1/2
```

and zero beyond the cutoff.

**Quadratic Spectral / QS / Andrews**

QS is **not truncated at `bw`**. For all observed lags `j=1,...,T-1` when `bw>0`:

```text
x_j = 6*pi*j/(5*bw)
w_j = 3/x_j^2 * (sin(x_j)/x_j - cos(x_j)),
w_0 = 1.
```

When `bw=0`, define `w_0=1` and `w_j=0` for `j>0` (no autocorrelation lags). Thus QS bandwidth is a smoothing scale while its lag support is all observed lags. The implementation and tests must not reuse a generic `range(1,bw+1)` loop for QS.

Effective bandwidth/support and canonical kernel name are retained in auditable covariance metadata/test output. Invalid kernel names fail precisely. Legacy Pooled row-HAC remains Bartlett-only and does not inherit the new DK aliases.

## 9. Small-sample and reference-distribution conventions

- nonrobust keeps Student-t;
- historical robust/explicit hc1 keeps HC1 scaling and normal-reference inference;
- HC0/HC2/HC3, cluster, and DK use existing sandwich normal-reference inference;
- `group_debias` changes covariance magnitude only, not the reference distribution;
- DK always uses Section-8 df correction;
- legacy cluster gets no new model-df factor by default;
- Panel DK uses Stage-B standard effect rank while historical Panel robust remains frozen.

External tests assert covariance/SE plus documented correction settings; covariance scaling is not changed merely to force p-values from a different reference distribution.

## 10. Backend-native algorithm contract

Numerical arrays kept on active NumPy/CuPy/Torch backend:

- fit-space design/residuals;
- row scores/leverage;
- grouped score matrices;
- lag products;
- bread/meat/covariance until existing final inference conversion.

Allowed CPU metadata:

- cluster/entity/time labels;
- integer group codes/order;
- small scalar configuration and kernel-weight vectors.

Codes/weights are transferred to device for numerical accumulation. No full numerical X/residual/score/leverage transfer is allowed.

A single backend-neutral grouped-score primitive must serve cluster and DK paths and use NumPy add-at/equivalent, CuPy add-at/equivalent, and Torch scatter-add without fallback or Python observation loops.

## 11. Estimator integration

### PanelOLS

- preserve existing defaults;
- HC2/3 use transformed design;
- cluster retains one-/two-way semantics plus `group_debias`;
- DK uses aligned `time_ids` and Stage-B standard effect rank;
- entity-only, time-only, and two-way FE are supported if df is valid;
- all non-nonrobust covariance remains inapplicable to classical Hausman.

### RandomEffects

- keyword-only covariance constructor config;
- add `cluster=None` to fit side-array alignment;
- shared inference receives `X_star`/GLS residuals;
- DK consumes time IDs only when selected;
- explicit-constant and unbalanced covariance paths are required tests.

### PooledOLS

- add HC0/2/3 and DK;
- DK consumes `time_index` by group, not legacy HAC row-order formula;
- preserve BP-LM/entity diagnostic alignment.

### BetweenOLS / FirstDifferenceOLS

- add HC0/2/3 only;
- preserve transform/coefficient/robust semantics;
- reject DK/cluster absent a separate reviewed contract.

### FamaMacBeth

- no residual-sandwich changes; regression-freeze existing covariance.

## 12. External alignment

### linearmodels==7.0

Executable aligned cases:

- Pooled robust/cluster/DK;
- Panel cluster group-debias primitive on same transformed design and DK with standard effect rank;
- RandomEffects robust/cluster/DK including explicit constant/unbalanced;
- DK full-rank scale, default/explicit bandwidth, Bartlett/Parzen/QS weights/support;
- cluster group-debias coefficient and two-way inclusion-exclusion.

Tests state linearmodels `debiased`, `extra_df`, `group_debias`, clusters, kernel, bandwidth, effects, and intercept. Legacy statgpu corrections that intentionally differ are compared through the same transformed primitive or documented ratio, not “fixed” by breaking compatibility.

### HC2/HC3

Primary baselines:

1. direct analytic sandwich;
2. statsmodels OLS robust covariance on exactly the same transformed fit-space design/residuals where executable.

Do not compare absorbed FE HC2/3 to a literal full-dummy HC2/3 and force equality.

R `plm`/`sandwich` and Stata/fixest definitions are documentation/remote references where useful; unavailable R is remote pending rather than permission to weaken Python/analytic gates.

## 13. Tests

### Primitive tests

- HC0/HC1 scale;
- HC2/3 leverage and unit-leverage failure;
- rank-deficient pseudoinverse;
- one-/two-way cluster and intersection;
- group-debias factors and one-group errors;
- exact pair factorization;
- DK time aggregation/full-rank scale/rank-deficient extension;
- Bartlett/Parzen/QS exact weights and QS all-lag support;
- bandwidth default/zero/cap/validation;
- unsorted/repeated/unbalanced time labels.

### Estimator tests

For every supported estimator/covariance family:

- coefficients invariant to covariance selection;
- covariance/BSE/t-or-z/p/CI consistent;
- NumPy analytic/external precision;
- maintained Torch CPU where used by project gates;
- CuPy/Torch physical CUDA parity;
- explicit backend provenance/no fallback;
- formula vs array parity and missing-row metadata alignment;
- intercept/categorical/interaction cases where existing formula API supports them.

### Compatibility tests

- Stage-B default inference frozen;
- `robust == hc1 == historical HC1`;
- Pooled legacy `hac` frozen;
- FamaMacBeth covariance frozen;
- Stage-B Hausman/pooling/BP/fit statistics unchanged for equivalent fits.

### Invalid tests

- unsupported covariance;
- DK missing time / <2 periods / non-orderable labels;
- invalid bandwidth/kernel;
- malformed cluster dimensions;
- group debias too few groups/non-cluster use;
- HC2/3 undefined leverage;
- requested GPU backend unavailable.

## 14. Physical GPU validation

Add `dev/benchmarks/validate_panel_stage_c_gpu.py` as correctness/provenance only.

Every new public covariance integration reaches both CuPy and Torch CUDA at least once.

Minimum per backend:

- Pooled: HC0/2/3, one-way cluster, DK Bartlett, legacy HAC regression;
- Panel: entity FE HC0/2/3, two-way FE HC2 or HC3, two-way group-debiased cluster, DK with effect-rank scaling;
- RandomEffects: HC0/1/2/3 including explicit constant/unbalanced, cluster, DK;
- Between: HC0/2/3;
- FirstDifference: HC0/2/3;
- at least one DK Parzen or QS explicit-bandwidth case, with QS all-lag execution physically covered;
- deliberately permuted/formula-equivalent metadata-order fixture where feasible.

Record exact SHA/clean tree, requested/executed backend, covariance/SE/t/p/CI vs NumPy, coefficient and Stage-B-stat invariance, covariance config/effective bandwidth/support/group counts/rank extension, and environment/GPU metadata.

A validator change after an accepted artifact invalidates acceptance for the changed validator contract per `RELEASING.md`.

## 15. Performance gate

Add a bounded synchronized Stage-C covariance benchmark.

Questions:

- HC2/3 avoids `O(n^2)` hat matrix and scales as row leverage + `k x k` operations;
- cluster/DK uses grouped backend reductions, not observation Python loops/full host numerical transfer;
- QS all-lag cost is reported at representative T and does not accidentally become `O(n^2)` in observations;
- GPU metadata conversion/synchronization is not transfer-dominated at target sizes.

No speedup claim is planned. Performance becomes blocking only for material regression/pathological complexity/transfer dominance. Optimization budget: one profile, at most two algorithmic/kernel attempts, one rebenchmark each. Timing JSON remains separate from correctness evidence.

## 16. Docs/artifacts

Update EN first then CN:

- `docs/en/models/panel.md`, `docs/cn/models/panel.md`;
- root + detailed EN/CN changelogs;
- docstrings/API examples;
- coverage/catalog/frontend only if physical Stage-C evidence is promoted;
- physical validation review record after remote acceptance.

Docs explicitly state HC fit-space semantics, robust==HC1, legacy HAC vs DK, DK exact scaling/time/kernel/QS support, cluster group-debias, normal vs t references, RE quasi-demeaned scores, no fallback, and unsupported combinations.

## 17. Implementation sequence

1. Freeze Stage-B inference golden outputs.
2. Add covariance-name normalization, grouped-score primitive, HC leverage/meat, kernel weights.
3. Implement HC0/2/3 preserving robust HC1.
4. Harden cluster metadata and group-debias.
5. Implement DK exact scaling and kernel-specific support.
6. Integrate Pooled/Panel.
7. Integrate RE quasi-demeaned covariance.
8. Integrate Between/FD HC; freeze FMB.
9. Add analytic/statsmodels + pinned linearmodels 7.0 tests.
10. Add maintained Torch and complete physical CUDA runner contract.
11. Add performance benchmark/schema tests.
12. Run targeted/full hosted gates.
13. Run `.claude/skills/code-review.md` auto-fix; repeat until no local CRITICAL/HIGH/relevant MEDIUM.
14. Run exact-clean-head physical GPU and remote/external gates.
15. Promote evidence only after physical acceptance; rerun exact-final hosted gates and perform a fresh non-inherited final review.

## 18. Acceptance checklist

- [ ] Stage-B default covariance/inference preserved.
- [ ] HC0/1/2/3 transformed-fit-space semantics tested.
- [ ] RE HC/cluster/DK works without coefficient/variance-component drift.
- [ ] DK full-rank covariance matches linearmodels 7.0; rank-deficient behavior is explicitly a statgpu extension.
- [ ] Bartlett/Parzen/QS kernel weights and QS all-observed-lag support match the pinned definition.
- [ ] Pooled legacy row-HAC distinct/unchanged.
- [ ] Cluster one-/two-way and group-debias backward-compatible.
- [ ] Every supported new public covariance works NumPy/CuPy/Torch without fallback.
- [ ] Physical CUDA includes Pooled/Panel/RE/Between/FD new HC integrations plus cluster and DK on both GPU backends.
- [ ] No full GPU numerical host copy for covariance accumulation.
- [ ] Formula/missing-row metadata alignment tested.
- [ ] Analytic/statsmodels/linearmodels comparisons pass where definitionally comparable.
- [ ] Stage-B diagnostics/fit stats regression-clean.
- [ ] EN/CN docs/changelogs synchronized.
- [ ] Performance gate has no material unresolved regression.
- [ ] Exact-head hosted tests/compatibility/release/frontend gates pass.
- [ ] Exact-clean-head physical Stage-C CuPy/Torch validation passes and immutable evidence is recorded.
- [ ] Fresh final review has zero unresolved CRITICAL/HIGH/relevant MEDIUM.

## 19. Non-goals

No robust auxiliary Hausman, cluster t/F finite-group reference distribution, generic second covariance-debias API, weights, Panel IV/GMM, HDFE absorb, DID/event study, dynamic panel, FamaMacBeth redesign, PCSE, or Conley/spatial HAC. Any such addition requires a separate reviewed statistical/API contract.