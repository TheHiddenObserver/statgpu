# Node-wise Lasso tuning contract — implementation plan

Status: **REVISED AFTER PLAN REVIEW ROUND 1**

Target baseline:

- repository: `TheHiddenObserver/statgpu`
- implementation branch: `fix/nodewise-alpha-inference-contract`
- base `master`: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
- affected capability: squared-error L1 / Elastic Net debiased inference and the node-wise Lasso approximate-precision construction used by marginal and simultaneous inference

This work is intentionally independent of documentation PR #134. Runtime/API behavior is fixed and reviewed here first; the still-Draft #132/#134 documentation stack is synchronized only after the runtime contract is stable.

## 1. Problem to fix

Current debiased inference has two coupled defects.

1. The node-wise Lasso penalty is an important statistical tuning parameter but is internal-only. Users cannot inspect or override it.
2. The current automatic rule uses the main response-model residual scale,

   $$
   \lambda_{\mathrm{nw}}
   =
   \hat\sigma_y\sqrt{\frac{2\log(\max(p,2))}{n}},
   $$

   even though the node-wise regressions estimate a design-side object, approximately `Sigma_X^{-1}`. Rescaling only `y` can therefore alter the estimated precision matrix `M` while `X` is unchanged.

The source also contains wording attributing the response-scale multiplier to van de Geer et al. (2014). The literature supports node-wise tuning of order `sqrt(log(p)/n)` but does not make the main-response residual scale the unique node-wise multiplier.

## 2. Impact classification

Active axes:

- public API / sklearn compatibility;
- inference and fitted-result provenance;
- NumPy / CuPy / Torch numerical parity;
- intercept and analytic-weight semantics;
- CV final-refit propagation for LassoCV and ElasticNetCV;
- formula/model-matrix parity;
- failure-state cleanup;
- docs / changelog / exact-source evidence.

Not changed:

- the main Lasso / Elastic Net objective;
- main-model `alpha` grids, CV folds, scoring, or selection;
- unrelated GLM/survival/nonparametric/feature-selection methods;
- solver availability or fallback policy;
- performance claims.

GPU batching is touched, so runtime/memory sanity is checked, but no speedup claim is part of acceptance.

## 3. Public API contract

### 3.1 Parameter

Add

```python
nodewise_alpha: Optional[float] = None
```

to maintained direct/CV surfaces that can consume node-wise debiased inference:

- `PenalizedGeneralizedLinearModel`;
- `PenalizedLinearRegression`;
- `Lasso`;
- `ElasticNet`;
- `LassoCV`;
- `ElasticNetCV`.

On the generic GLM surface the parameter is only consumed when the executed inference path is squared-error + L1/ElasticNet + `debiased`; other loss/penalty/inference combinations do not silently reinterpret it as another tuning control.

### 3.2 Raw validation and sklearn compatibility

Constructor validation must inspect the raw user value without replacing the stored constructor object:

- `None` is accepted;
- otherwise require a real scalar, finite, strictly positive value;
- reject bool, arrays/non-scalars, NaN, infinity, zero, and negatives with `ValueError`;
- do not eagerly cast the stored public constructor value to `float`, so sklearn clone/reconstruction identity behavior remains consistent with existing statgpu conventions.

The resolved numerical float is produced only when node-wise inference actually executes.

The same validation contract must be exercised after `set_params(...)` reconstruction. Internal clone/meta-estimator reconstruction must not emit warning noise.

### 3.3 Meaning

`alpha` and `nodewise_alpha` are independent:

- `alpha`: main penalized prediction/selection estimator;
- `nodewise_alpha`: node-wise regressions used only to construct the approximate precision matrix for `inference_method="debiased"`.

`nodewise_alpha` must not affect:

- LassoCV/ElasticNetCV candidate grids;
- fold scores;
- selected main `alpha` / `l1_ratio`;
- penalized prediction `coef_` / `intercept_`.

The first public contract is scalar-only. A per-coordinate vector is intentionally deferred because the maintained CuPy/Torch batched Gram solvers currently share one node-wise alpha.

### 3.4 Fitted state and provenance

When node-wise debiased inference succeeds, publish:

```python
model.nodewise_alpha_
```

as the resolved scalar actually passed to the node-wise solver.

Record in `_inference_result.metadata`:

```text
nodewise_alpha
nodewise_alpha_source = "user" | "auto"
nodewise_alpha_rule = "explicit" | "standardized_universal_v1"
nodewise_design_standardized = true
nodewise_n_samples
```

For LassoCV/ElasticNetCV, a successful final refit also publishes the same resolved `nodewise_alpha_` on the outer CV estimator.

When node-wise inference is not executed, `nodewise_alpha_` remains `None`.

## 4. Ownership of the canonical debiased working problem

### 4.1 Do not duplicate PR #138 transforms

The #138 runtime contracts are authoritative for centering and analytic-weight normalization. In particular, the installed sparse-Gaussian debiased wrappers already map the fit to one centered average-loss problem and use the row scaling

$$
\sqrt{w_i n / \sum_k w_k}
$$

when analytic weights are present.

**The new node-wise helper must not reconstruct centering or weights from raw `X`, `y`, or `sample_weight`.** It receives the already-canonical design that the existing CPU/CuPy/Torch debiased routine currently uses for `Sigma_hat` and node-wise regression.

This prevents double centering/double weighting and preserves the existing #138 identities:

- omitted weights = all-one weights;
- `w` and `c w` are equivalent;
- intercept-enabled inference uses the centered working problem;
- explicit GPU inference remains on the fit-resolved concrete device.

For documentation, the canonical working design may be described algebraically as:

$$
X_w = X-\bar X
$$

for unweighted intercept fits, `X_w=X` without an intercept, and

$$
X_{w,i}=\sqrt{\frac{n w_i}{W}}(x_i-\bar x_w),\qquad W=\sum_iw_i,
$$

for weighted intercept fits (without the centering term when no intercept is fitted).

The implementation, however, standardizes the design **after** the existing contract has produced this working problem.

## 5. Statistical node-wise contract

### 5.1 Standardized design

Let `X_w` denote the already-canonical working design supplied to the node-wise precision routine. Define

$$
d_j=\sqrt{\frac1n\sum_{i=1}^nX_{w,ij}^2},
\qquad D=\operatorname{diag}(d_1,\ldots,d_p),
\qquad Z=X_wD^{-1}.
$$

This makes the public node-wise penalty scale independent of feature measurement units.

### 5.2 Scale validation

All node-wise computations are float64 at the maintained inference boundary. Define one shared scale-validity rule across NumPy/CuPy/Torch:

1. every `d_j` and `d_max=max_j d_j` must be finite;
2. require `d_max > 0`;
3. define `d_tol = 64 * eps_float64 * d_max`;
4. require every `d_j > d_tol`.

A failing scale aborts debiased inference before publishing any inference result. Estimation with `compute_inference=False` is unaffected.

Do not introduce backend-specific `1e-30`-style scale magic.

### 5.3 Automatic alpha

If `nodewise_alpha is None`, use

$$
\boxed{\lambda_{\mathrm{nw}}^{\mathrm{auto}}
=\sqrt{\frac{2\log(\max(p,2))}{n}}}
$$

on standardized `Z`.

`n` is the row count of the existing canonical average-loss working problem. Analytic weights have already been normalized by the #138 transform; this change does not introduce a separate Kish/effective-sample-size convention.

The rule is a concrete statgpu default motivated by the standard theoretical order `sqrt(log(p)/n)`, not a claim that the constant `sqrt(2)` is uniquely theorem-mandated.

If the user supplies `nodewise_alpha=x`, use the same standardized design and pass exactly numerical `x` to every node-wise Lasso solve.

Thus an automatic run can be reproduced by a later explicit run using its reported `nodewise_alpha_`.

### 5.4 Node-wise objective

For each coordinate `j`, solve

$$
\hat\gamma_j
=\arg\min_\gamma\left\{
\frac{1}{2n}\lVert Z_j-Z_{-j}\gamma\rVert_2^2
+\lambda_{\mathrm{nw}}\lVert\gamma\rVert_1
\right\}.
$$

Let

$$
r_j=Z_j-Z_{-j}\hat\gamma_j,
\qquad
C_j^{(Z)}=\frac1nZ_j^\top r_j.
$$

Construct

$$
\hat\Theta_{Z,j}^\top
=\frac{(e_j-\tilde\gamma_j)^\top}{C_j^{(Z)}}.
$$

Then transform back to the original working-design coordinates:

$$
\boxed{M_X=D^{-1}\hat\Theta_ZD^{-1}.}
$$

The existing debiasing correction and variance formulas continue to use the canonical `X_w`, its covariance, and `M_X`.

### 5.5 Node-wise normalizer validity

On standardized data a valid node-wise solve should give a finite positive `C_j^{(Z)}`. Define one shared float64 threshold

$$
C_{\min}=64\,\epsilon_{64}.
$$

Require

```text
isfinite(C_j) and C_j > C_min
```

for every row. Negative, zero, non-finite, or numerically tiny values are inference failures.

Remove the current behavior that silently substitutes an identity row when `abs(C_j)` is tiny. A plausible-looking but invalid precision row is worse than an explicit inference failure.

### 5.6 p=1 scope

This repair does **not** redesign the historical single-feature debiased-inference support boundary. Existing maintained CPU/GPU behavior for `p=1` must not change accidentally while the node-wise setup is refactored. Add a targeted regression that records/preserves the current behavior on each declared backend; a deliberate p=1 unification is a separate change.

## 6. Implementation architecture

### 6.1 Shared helper layer

Add a focused maintained helper under `statgpu/linear_model/penalized/` for:

- raw `nodewise_alpha` validation;
- design-column scale validation;
- automatic/explicit alpha resolution;
- standardization/back-transformation utilities;
- shared `C_j` validation;
- metadata construction.

Do not add another install-time monkeypatch module merely for the new parameter.

The helper operates on the already-resolved node-wise working design; it does not accept `y` as an input to alpha resolution.

### 6.2 NumPy

Inside the maintained original debiased routine called by the #138 wrappers:

1. receive canonical `X_w`;
2. compute `D` and `Z`;
3. resolve alpha from `self.nodewise_alpha`;
4. run the existing per-coordinate L1 solver on `Z`;
5. validate positive `C_j`;
6. construct `Theta_Z`;
7. back-transform to `M_X`;
8. continue existing debiasing/reporting with `M_X` and canonical `X_w`.

### 6.3 CuPy/Torch

Stay on the already selected concrete device.

Prefer Gram-space scaling to a second full standardized matrix allocation:

$$
\Sigma_Z=D^{-1}\Sigma_{X_w}D^{-1}.
$$

Use `Sigma_Z` to form the existing batched Gram/cross-product inputs, solve every node-wise regression with the same resolved scalar alpha, construct `Theta_Z`, validate `C_j` on device, and back-transform to `M_X` on device.

No new host fallback is permitted. Existing reporting snapshots remain at the already-defined post-numerical boundary.

### 6.4 Current installed wrappers/finalizers

The implementation must explicitly preserve the installed #138 transaction layers:

- `_post_selection_ols_fifth_review_contract` remains authoritative for centered/weighted debiased working data;
- `_post_selection_ols_review_fix_contract._finalize_weighted_debiased_result` reconstructs weighted/centered result objects and therefore must preserve the new node-wise metadata copied from the base result;
- simultaneous-inference wrappers must reuse the same `M_X` and resolved alpha rather than invoking a second hidden resolver.

Tests must cover unweighted CPU, centered CPU, weighted CPU, CuPy, Torch, and simultaneous publication paths.

## 7. State reset and failure safety

`nodewise_alpha_` is result-bearing inference state. It must be cleared by every relevant reset path, including:

- the canonical `PenalizedGeneralizedLinearModel._clear_inference_state` chain;
- sparse-inference failed-refit invalidation in `_post_selection_ols_fifth_review_contract`;
- generic no-inference/failed-fit cleanup that delegates to `_clear_inference_state`;
- `LassoCV._reset_cv_fit_state`;
- `ElasticNetCV._reset_cv_fit_state` after that outer fitted attribute is introduced.

A fit that fails during scale validation, node-wise solve, `C_j` validation, marginal reporting, or simultaneous reporting must not leave a stale `nodewise_alpha_`, `M`, p-value, or CI from a previous successful fit.

Add success→failure→inspection and success→`set_params`→refit regressions.

## 8. Cache contract

Cache identity must include enough information to prevent reuse across distinct precision problems:

- existing design identity/provenance;
- resolved numerical `nodewise_alpha`;
- solver tolerance;
- a standardization-contract version token if the existing design identity alone does not distinguish old/new cache semantics.

Changing explicit `nodewise_alpha` forces a miss.

`None` and an explicit number equal to the automatically resolved value may share a cache entry because the statistical problem is then identical; source provenance (`auto` vs `user`) is still reported from the current request, not inferred from the cache.

Existing cache collision/hash policy is not redesigned in this task.

## 9. CV propagation

### 9.1 LassoCV

Add `nodewise_alpha=None` as **final-refit inference configuration**.

- do not add it to main-alpha selection cache keys;
- pass it unchanged to final `Lasso(...)`;
- changing only `nodewise_alpha` must not change selected `alpha_`, `mse_path_`, or final penalized `coef_`;
- after successful debiased final refit, outer `nodewise_alpha_ == estimator_.nodewise_alpha_`;
- reset it before every CV fit.

### 9.2 ElasticNetCV

Add and propagate `nodewise_alpha=None` analogously.

The existing `ElasticNetCV` hard-coded final `inference_method="debiased"` is an adjacent API inconsistency but is not required to implement node-wise tuning itself. Do not broaden this repair into a full ElasticNetCV inference-selector redesign unless a later implementation review proves the new contract cannot be correct without it. Record that inconsistency as follow-up scope if it remains.

### 9.3 PenalizedGLM_CV

At the reviewed baseline, generic `PenalizedGLM_CV` does not expose the affected final-refit inference toggle. It is therefore outside this parameter-propagation surface unless implementation inspection disproves that assumption.

## 10. Validation plan

### 10.1 API/compatibility

Cover all public surfaces that gain the parameter:

- omitted `None`;
- explicit positive Python/NumPy real scalar;
- bool/non-scalar/NaN/inf/zero/negative rejection;
- signature/introspection;
- `get_params(deep=False)`;
- transactional `set_params`;
- sklearn clone;
- no warning noise from internal reconstruction;
- non-debiased inference leaves `nodewise_alpha_ is None` and does not alter estimation.

### 10.2 Statistical invariants

1. **Response-scale independence of precision construction**: fixed canonical design gives identical automatic alpha and `M_X` when only `y` changes scale. Prefer a helper-level test so main-model alpha scaling is not a confounder.
2. **Feature-scale equivariance**: for positive diagonal feature rescaling `A`, require `M_{XA} ≈ A^{-1} M_X A^{-1}`.
3. **Auto/explicit equivalence**: capture `nodewise_alpha_` from auto, rerun explicitly with the same number, and require equivalent `M`, debiased params, SE, z, p, and CI.
4. **Global weight-scale invariance**: `w` and `c w` agree.
5. **All-one weight identity**: weighted all-one and unweighted agree.
6. **Positive-normalizer fail closed**: negative/non-finite/tiny `C_j` cannot publish inference.
7. **Degenerate-column fail closed**: invalid `d_j` cannot publish inference.
8. **Simultaneous reuse**: marginal and max-|Z| paths use the same resolved alpha and `M`.

### 10.3 Backend/device parity

For NumPy/CuPy/Torch:

- same auto alpha;
- explicit value reaches the actual node-wise solver;
- precision/debiased outputs agree within maintained numerical tolerances;
- concrete device provenance is correct;
- explicit CUDA/Torch never falls back to CPU;
- standardization and back-transform execute on the intended backend;
- p=1 historical behavior is preserved.

### 10.4 CV/formula

- LassoCV and ElasticNetCV propagate only to final inference;
- main CV selection outputs are invariant to changing only `nodewise_alpha`;
- final refit outer/inner `nodewise_alpha_` agrees;
- formula and array routes agree after model-matrix construction, including intercept semantics;
- categorical/formula column scaling does not reorder or rename inference parameters.

### 10.5 Independent numeric reference

Add a small independent NumPy reference for fixed explicit alpha:

- construct canonical `X_w` using existing test helpers;
- standardize to `Z`;
- solve node-wise Lasso with an independent/simple reference path at aligned objective scale;
- build `Theta_Z`, back-transform, and compare `M_X`;
- verify `M_X Sigma_X` approximation / node-wise KKT identities.

An optional R `hdi` comparison can align a manually chosen node-wise lambda and is supplementary. `hdi`'s own CV tuning is not expected to numerically match statgpu's fixed universal default.

### 10.6 Simulation evidence

Because the automatic default changes, add a deterministic-seed validation script comparing old vs new defaults on representative sparse Gaussian designs. Report at least:

- empirical marginal coverage;
- interval length;
- finite/failure count;
- `||M Sigma - I||_max` or equivalent precision residual;
- old/new resolved alpha rule.

This is evidence, not a brittle unit-test claim of exact 95% finite-sample coverage.

### 10.7 Physical GPU acceptance

Add a dedicated validator or focused extension covering both CuPy and Torch on a physical CUDA device:

- auto alpha;
- explicit override;
- auto/explicit equivalence;
- y-scale-independent precision construction;
- feature-scale equivariance;
- weighted identity/global-scale invariance;
- simultaneous reuse;
- metadata/backend/device provenance;
- failure artifact with exact SHA/environment if a case aborts.

Remote-only evidence may leave the implementation at `PARTIAL_REMOTE_PENDING`; it may not be silently omitted from the report.

## 11. Documentation/migration

Update runtime-branch source comments/docs/changelog to state:

- `nodewise_alpha` is separate from main `alpha`;
- `None` selects the standardized design-side automatic rule;
- explicit values are interpreted on standardized node-wise design scale;
- the default is motivated by `sqrt(log p/n)` theory, not uniquely mandated;
- the automatic precision construction is independent of `y` units;
- scalar-only support is intentional;
- LassoCV/ElasticNetCV use it only for final-refit inference.

Remove the incorrect source wording that attributes multiplication by main-response `sigma_hat` to van de Geer et al. (2014).

Because #134 contains newer learner-first docs, do not make #134 depend on an unmerged runtime branch. After this runtime contract is stable and eventually merged, sync master into #132/#134 and update those learner pages against the shipped API.

## 12. Implementation order

1. Add failing tests for missing public override, y-scale dependence, and stale-state leakage.
2. Add shared raw validation/resolution/scale/normalizer helpers.
3. Plumb constructor state through generic penalized linear, Lasso, ElasticNet.
4. Add `nodewise_alpha_` to canonical cleanup/reset transactions.
5. Implement NumPy standardize → node-wise solve → positive-normalizer check → back-transform.
6. Implement CuPy/Torch Gram-space equivalent on concrete devices.
7. Preserve metadata through weighted/centered finalizers and simultaneous inference.
8. Propagate through LassoCV/ElasticNetCV final refits and outer fitted state.
9. Close cache, weight, formula, p=1-preservation, and failure-state tests.
10. Add independent numeric reference and simulation validator.
11. Update runtime docs/changelog and #134 synchronization notes.
12. Run local-minimal then local-full relevant suites.
13. Run physical GPU acceptance when available.
14. Fresh independent code review on the exact implementation head; fix CRITICAL/HIGH and relevant MEDIUM findings; rerun affected validation; repeat until clean or user approval is required.

## 13. Acceptance criteria

- [ ] `nodewise_alpha=None|positive scalar` is public on every maintained direct/CV surface in scope.
- [ ] explicit value strictly overrides auto resolution and reaches actual node-wise solves.
- [ ] auto alpha depends only on canonical design-side state, not `y` scale.
- [ ] node-wise design is standardized and `M` is correctly back-transformed.
- [ ] feature-scale equivariance is tested.
- [ ] weighted all-one and global-scale identities remain intact.
- [ ] NumPy/CuPy/Torch use one statistical contract without explicit-device fallback.
- [ ] positive finite scale and `C_j` checks fail closed; no identity-row fallback remains.
- [ ] marginal and simultaneous inference reuse one resolved alpha/precision construction.
- [ ] cache semantics include the resolved tuning contract.
- [ ] `nodewise_alpha_` and metadata survive result finalization and are cleared on every reset/failure path.
- [ ] LassoCV/ElasticNetCV propagate only to final-refit inference and do not change hyperparameter selection.
- [ ] signature/get_params/set_params/clone/refit behavior is tested.
- [ ] formula parity and existing p=1 behavior are regression-protected.
- [ ] source/docs no longer misattribute the old `sigma_hat` multiplier.
- [ ] local blocking suites pass on the exact head.
- [ ] physical GPU evidence is recorded, or the result is explicitly `PARTIAL_REMOTE_PENDING` if that is the sole remaining gate.
- [ ] fresh independent implementation review has no unresolved CRITICAL/HIGH finding.

## 14. Explicit non-goals

- vector-valued per-coordinate node-wise alpha in v1;
- node-wise CV or square-root Lasso as the automatic selector in this repair;
- main-model alpha/CV redesign;
- full ElasticNetCV inference-selector redesign;
- p=1 capability redesign;
- cache hash-policy redesign unrelated to the new tuning key;
- cosmetic edits to private legacy implementations when no maintained public route uses them;
- performance claims.

## 15. Plan review log

### Round 1

Reviewed plan commit: `3b83e3a1a84d18769ba0267ba74fd4aa281a08ec`
Review artifact: `dev/reviews/nodewise-alpha-plan-review-round-1.md`
Verdict: **PLAN CHANGES REQUIRED**

Findings fixed in this revision:

- HIGH: made existing #138 centered/weighted working-data wrappers authoritative; the new helper no longer reconstructs raw working data.
- HIGH: added `nodewise_alpha_` to canonical failure/reset/CV cleanup requirements.
- HIGH: changed node-wise normalizer contract from `abs(C_j)` fallback semantics to finite positive `C_j > C_min`, with no identity-row publication.
- MEDIUM: specified raw constructor validation without lossy coercion.
- MEDIUM: specified metadata survival through weighted/centered finalizers and simultaneous inference.
- MEDIUM: explicitly preserved existing p=1 behavior rather than changing it accidentally.
- MEDIUM: centralized float64 degeneracy tolerance rules for design scales and standardized normalizers.

Next step: fresh plan review against this revised exact commit before implementation.