# Node-wise Lasso tuning contract — implementation plan

Status: **REVISED AFTER PLAN REVIEW ROUND 2**

Target baseline:

- repository: `TheHiddenObserver/statgpu`
- implementation branch: `fix/nodewise-alpha-inference-contract`
- base `master`: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
- affected capability: squared-error L1 / Elastic Net debiased inference and the node-wise Lasso approximate-precision construction used by marginal and simultaneous inference

This work is independent of documentation PR #134. Runtime/API behavior is fixed and reviewed here first; the still-Draft #132/#134 documentation stack is synchronized only after the runtime contract is stable.

## 1. Defects being fixed

1. The node-wise Lasso penalty is an inference-critical tuning parameter but is currently internal-only.
2. The automatic rule currently uses the main response residual scale,

   $$
   \lambda_{\mathrm{nw}}
   =\hat\sigma_y\sqrt{\frac{2\log(\max(p,2))}{n}},
   $$

   even though node-wise regressions estimate a design-side precision object. Rescaling only `y` can therefore change `M` while `X` is unchanged.
3. Source wording incorrectly suggests that multiplying by the main-response `sigma_hat` is the van de Geer et al. (2014) construction.
4. The internal node-wise numerical contract is underspecified: hidden solver settings are not reported, cache keys currently use the parent estimator tolerance rather than the actual node-wise tolerance, and an unconverged solve can still produce plausible finite output.

## 2. Impact classification

Active axes:

- public API / sklearn compatibility;
- inference / result provenance / failure safety;
- NumPy, CuPy, Torch backend closure;
- intercept and analytic-weight identities;
- LassoCV / ElasticNetCV final-refit propagation;
- formula/model-matrix parity;
- node-wise convergence/KKT correctness;
- docs/changelog/evidence.

Not changed:

- main Lasso/ElasticNet objectives;
- main-model `alpha`/`l1_ratio` CV selection;
- unrelated model families;
- public solver support;
- performance claims.

## 3. Public API

### 3.1 New parameter

Add to maintained surfaces that can consume squared-error node-wise debiased inference:

```python
nodewise_alpha: Optional[float] = None
```

Surfaces:

- `PenalizedGeneralizedLinearModel`;
- `PenalizedLinearRegression`;
- `Lasso`;
- `ElasticNet`;
- `LassoCV`;
- `ElasticNetCV`.

On the generic GLM surface it is consumed only by squared-error + L1/ElasticNet + `debiased` inference.

### 3.2 Validation / clone contract

Store the requested constructor object unchanged; validate without lossy constructor coercion.

Accepted:

- `None`;
- finite real scalar `> 0`.

Rejected with `ValueError`:

- bool;
- non-scalar/array;
- NaN/inf;
- zero/negative.

The same contract must survive `get_params`, transactional `set_params`, sklearn clone, and internal reconstruction without warning noise.

### 3.3 Meaning

- `alpha`: main penalized estimator.
- `nodewise_alpha`: only the node-wise Lasso precision construction used by `debiased` inference.

Changing only `nodewise_alpha` must not change direct-fit or CV penalized prediction coefficients, main CV candidate scores, selected `alpha_`, or selected `l1_ratio_`.

Version 1 is scalar-only. Per-coordinate node-wise penalties remain future scope.

### 3.4 Fitted state / metadata

After successful node-wise debiased inference publish:

```python
nodewise_alpha_
```

and include in `_inference_result.metadata`:

```text
nodewise_alpha
nodewise_alpha_source = "user" | "auto"
nodewise_alpha_rule = "explicit" | "standardized_universal_v1"
nodewise_design_standardized = true
nodewise_n_samples
nodewise_solver = "fista"
nodewise_tol
nodewise_max_iter
nodewise_max_kkt_residual
```

LassoCV/ElasticNetCV also copy the final estimator's resolved `nodewise_alpha_` to the outer fitted estimator.

No node-wise inference -> `nodewise_alpha_ is None`.

## 4. Canonical working-data ownership

PR #138's installed sparse-Gaussian wrappers remain authoritative for centering and analytic-weight normalization. They already produce the centered average-loss working problem, including row scaling

$$
\sqrt{w_i n / \sum_k w_k}.
$$

The new node-wise helper **must not accept raw `y` to select alpha and must not reconstruct centering/weights**. It receives only the already-canonical working design currently used by the debiased routine.

Therefore existing identities remain part of the contract:

- omitted weights = all-one weights;
- `w` = `c w` globally;
- intercept fits use the centered working design;
- explicit CuPy/Torch inference stays on the selected concrete device.

## 5. Statistical node-wise contract

### 5.1 Standardization

For canonical working design `X_w`, define

$$
d_j=\sqrt{n^{-1}\sum_iX_{w,ij}^2},\qquad
D=\operatorname{diag}(d_j),\qquad
Z=X_wD^{-1}.
$$

The public `nodewise_alpha` is always interpreted on `Z`, making its scale independent of feature units.

### 5.2 Design-scale validity

All maintained debiased inference is float64 at this boundary. Shared NumPy/CuPy/Torch rule:

1. every `d_j` and `d_max=max_j d_j` finite;
2. `d_max > 0`;
3. `d_tol = 64 * eps64 * d_max`;
4. every `d_j > d_tol`.

Failure aborts inference without publishing partial/stale state. No backend-specific scale magic.

### 5.3 Automatic alpha

For `nodewise_alpha=None`:

$$
\boxed{\lambda_{\mathrm{nw}}^{\mathrm{auto}}
=\sqrt{\frac{2\log(\max(p,2))}{n}}}
$$

on standardized `Z`.

`n` is the row count of the canonical average-loss problem. The existing analytic-weight transform has already normalized the weighted objective; this repair does not introduce a second effective-sample-size convention.

The order `sqrt(log p/n)` is literature-motivated; the exact `sqrt(2)` constant is documented as statgpu's default, not a uniquely theorem-mandated value.

Explicit `nodewise_alpha=x` passes numerical `x` unchanged to every node-wise solve on the same standardized scale.

### 5.4 Objective / precision transform

For coordinate `j`:

$$
\hat\gamma_j
=\arg\min_\gamma
\left\{\frac{1}{2n}\|Z_j-Z_{-j}\gamma\|_2^2
+\lambda_{\mathrm{nw}}\|\gamma\|_1\right\}.
$$

Define

$$
r_j=Z_j-Z_{-j}\hat\gamma_j,\qquad
C_j^{(Z)}=n^{-1}Z_j^\top r_j,
$$

$$
\hat\Theta_{Z,j}^\top
=\frac{(e_j-\tilde\gamma_j)^\top}{C_j^{(Z)}}.
$$

Back-transform:

$$
\boxed{M_X=D^{-1}\hat\Theta_ZD^{-1}.}
$$

Existing debiasing/variance formulas continue with canonical `X_w`, `Sigma_{X_w}`, and `M_X`.

### 5.5 Positive normalizer gate

On standardized data require for every node-wise row

$$
C_j^{(Z)} > C_{\min},\qquad C_{\min}=64\epsilon_{64},
$$

and finite.

Negative/zero/non-finite/tiny values fail closed. Remove the current `abs(C_j)<1e-30 -> identity row` publication behavior.

## 6. Node-wise numerical solver contract

The new public alpha must not expose an otherwise unchecked internal solve.

### 6.1 Internal v1 settings

Keep solver controls internal in this repair, but centralize them as named constants used by all backends and cache/provenance:

```text
NODEWISE_SOLVER = "fista"
NODEWISE_TOL = 1e-5
NODEWISE_MAX_ITER = 500
NODEWISE_KKT_TOL = 1e-5
```

If validation shows the existing iteration cap cannot satisfy the declared KKT gate on representative supported designs, increase iteration work or improve the internal solve; do not silently loosen the acceptance gate without recorded evidence.

No public `nodewise_tol`/`nodewise_max_iter` is added in v1.

### 6.2 Independent full KKT check

Do not trust a solver's convergence flag or coefficient-delta stopping alone. After every node-wise solve, recompute the exact standardized Lasso KKT residual.

For

$$
g=\Sigma_{-j,-j}^{(Z)}\hat\gamma_j-\Sigma_{-j,j}^{(Z)},
$$

define coordinate residual

$$
r_k^{\mathrm{KKT}}=
\begin{cases}
|g_k+\lambda_{\mathrm{nw}}\operatorname{sign}(\hat\gamma_{jk})|,
&\hat\gamma_{jk}\ne0,\\
\max(|g_k|-\lambda_{\mathrm{nw}},0),
&\hat\gamma_{jk}=0.
\end{cases}
$$

and require

$$
\max_k r_k^{\mathrm{KKT}}\le\texttt{NODEWISE_KKT_TOL}.
$$

This production check is authoritative even if the underlying FISTA helper reports convergence. The proximal solver naturally produces exact zeros, so no separate arbitrary active-set threshold is needed for this KKT calculation.

A row failing KKT does not publish `M` or inference results.

### 6.3 p=1

Do not redesign historical single-feature support in this task. Preserve the current maintained behavior on each backend with regression tests; p=1 unification is separate scope.

## 7. Implementation architecture

### 7.1 Shared helper

Add a focused maintained module under `statgpu/linear_model/penalized/` for:

- raw `nodewise_alpha` validation;
- design scaling/validation;
- auto/explicit resolution;
- `C_j` validation;
- full KKT residual computation;
- back-transform/provenance helpers.

It consumes canonical design state only and never `y` for alpha resolution.

Do not add another install-time monkeypatch layer solely for this feature.

### 7.2 NumPy

Inside the maintained original debiased function invoked by #138 wrappers:

1. receive canonical `X_w`;
2. standardize to `Z`;
3. resolve alpha;
4. solve each node-wise problem with the centralized internal settings;
5. recompute and validate full KKT;
6. validate positive `C_j`;
7. construct `Theta_Z`, then `M_X`;
8. continue existing marginal inference.

### 7.3 CuPy / Torch

Stay on concrete selected GPU device.

Use standardized Gram algebra rather than an unnecessary second full matrix when practical:

$$
\Sigma_Z=D^{-1}\Sigma_{X_w}D^{-1}.
$$

All batched Gram/cross-product inputs come from `Sigma_Z`.

**Recompute the FISTA Lipschitz bound from `Sigma_Z`, not the old unstandardized `Sigma_hat`.** The step size must correspond to the objective actually being solved.

Run full KKT and `C_j` checks on device; back-transform `M_X` on device. No CPU numerical fallback.

### 7.4 Existing #138 wrappers/finalizers

Preserve current installed contracts:

- `_post_selection_ols_fifth_review_contract` owns centered/weighted working-data formation;
- `_post_selection_ols_review_fix_contract._finalize_weighted_debiased_result` reconstructs weighted/centered `DebiasedInferenceResult` and must preserve node-wise metadata from the base result;
- simultaneous inference reuses the same validated `M_X` and resolved alpha.

No second hidden resolver may run during simultaneous calibration.

## 8. Atomic publication / reset safety

Build candidate alpha, scales, `Theta_Z`, `M_X`, KKT metrics, and reporting arrays in local/native candidate state first.

Publish `nodewise_alpha_`, `_debiased_M_cpu`/native capture, and result metadata only after the node-wise matrix has passed scale/KKT/normalizer gates and the relevant marginal/finalizer transaction is ready to commit. Existing cleanup is a safety net, not the primary way partial state is hidden.

Clear `nodewise_alpha_` through all relevant reset paths:

- canonical `_clear_inference_state` chain;
- `_post_selection_ols_fifth_review_contract` failed sparse-inference invalidation;
- no-inference/failed-fit cleanup delegating to inference clear;
- `LassoCV._reset_cv_fit_state`;
- `ElasticNetCV._reset_cv_fit_state` once outer fitted state is added.

Test success→failure→inspection and success→`set_params`→refit.

## 9. Cache contract

Cache identity must use the **actual node-wise numerical contract**, not parent-model `tol`:

- canonical design identity;
- resolved numerical `nodewise_alpha`;
- `NODEWISE_SOLVER`;
- `NODEWISE_TOL`;
- `NODEWISE_MAX_ITER`;
- standardization/precision-contract version token.

Changing only main-model `tol` must not invalidate a node-wise cache entry unless it actually changes node-wise numerical settings. Changing node-wise alpha must invalidate it.

Auto and explicit requests with equal resolved alpha may share the numerical cache; metadata source remains request-specific.

Existing hash/collision policy is otherwise unchanged.

## 10. CV propagation

### LassoCV

Add `nodewise_alpha=None` as final-refit inference configuration only.

- not part of CV alpha grid/selection cache;
- propagate unchanged to final `Lasso`;
- changing it cannot change selected `alpha_`, `mse_path_`, penalized `coef_`/`intercept_`;
- outer resolved `nodewise_alpha_` equals final estimator's value after successful debiased inference;
- reset before every CV fit.

### ElasticNetCV

Propagate analogously.

The existing hard-coded final `inference_method="debiased"` is an adjacent API inconsistency but not required to implement node-wise alpha control. Do not broaden this repair into a full ElasticNetCV inference-selector redesign unless later implementation review proves it is necessary.

### PenalizedGLM_CV

Current public generic CV does not expose the affected final-refit inference toggle, so no propagation is planned unless implementation inspection disproves that baseline assumption.

## 11. Tests and evidence

### 11.1 API / state

For every public surface gaining the parameter:

- omission / explicit positive Python and NumPy scalars;
- invalid bool/non-scalar/NaN/inf/zero/negative;
- inspect signature;
- get_params/set_params/clone;
- fitted-state invalidation;
- no internal reconstruction warning noise;
- non-debiased path leaves `nodewise_alpha_ is None`;
- direct `Lasso`/`ElasticNet` penalized `coef_` and `intercept_` are unchanged when only nodewise alpha changes.

### 11.2 Statistical invariants

1. fixed canonical design: auto alpha and `M_X` independent of `y` scale;
2. positive diagonal feature scaling `A`: `M_{XA} ≈ A^{-1}M_XA^{-1}`;
3. auto run then explicit same alpha: identical precision/debiased reports within tolerance;
4. `w` vs `c w` identity;
5. all-one weight identity;
6. degenerate `d_j` fail closed;
7. non-positive/non-finite `C_j` fail closed;
8. forced unconverged/bad node-wise coefficient fails full KKT gate;
9. simultaneous inference reuses same alpha/M;
10. p=1 current behavior preserved.

### 11.3 Backend parity

NumPy/CuPy/Torch:

- same auto alpha;
- explicit alpha reaches actual solver;
- same standardized statistical contract;
- KKT residuals pass declared gate;
- `M`, debiased params, SE/z/p/marginal CI and simultaneous CI agree within maintained tolerances;
- concrete device provenance correct;
- no explicit-device fallback;
- GPU Lipschitz derives from `Sigma_Z`.

### 11.4 CV / formula

- nodewise alpha affects final inference only;
- LassoCV/ElasticNetCV selection outputs invariant to nodewise alpha;
- outer/inner resolved alpha agrees;
- array/formula inference agrees after design construction, including intercept and feature ordering.

### 11.5 Independent reference

Implement a small independent NumPy reference at fixed explicit alpha:

- canonical working design from existing helpers;
- standardized `Z`;
- independently solved node-wise problems at aligned objective scale;
- `Theta_Z`/back-transform;
- compare `M_X` and full KKT residuals;
- verify `M_X Sigma_X` approximation.

Optional R `hdi` comparison may align a manually chosen lambda; its own CV tuning is not expected to match the fixed statgpu auto default.

### 11.6 Simulation validator

Because default behavior changes, create deterministic-seed evidence comparing old/new defaults on representative sparse Gaussian designs. Record:

- marginal coverage;
- interval length;
- finite/failure counts;
- precision residual such as `||M Sigma-I||_max`;
- resolved alpha/source/rule;
- KKT residual distribution.

Simulation is evidence, not a brittle unit-test assertion of exact nominal coverage.

### 11.7 Physical GPU acceptance

Dedicated/focused validator for CuPy + Torch physical CUDA:

- auto and explicit alpha;
- auto/explicit equivalence;
- response-scale independence;
- feature-scale equivariance;
- weight identities;
- KKT/normalizer gates;
- simultaneous reuse;
- metadata/backend/device provenance;
- exact SHA/environment and partial failure artifact.

If this is the only unavailable evidence, report `PARTIAL_REMOTE_PENDING` rather than shrinking the capability.

## 12. Documentation / integration with #134

Runtime source/docs/changelog must state:

- `nodewise_alpha` is separate from main `alpha`;
- `None` = standardized design-side auto rule;
- explicit value is on standardized node-wise scale;
- auto rule is motivated by `sqrt(log p/n)` order, not uniquely mandated;
- auto precision construction is independent of `y` units;
- scalar-only v1;
- internal node-wise solver settings are recorded in inference metadata;
- CV parameter affects final-refit inference only.

Remove the old source attribution tying main-response `sigma_hat` scaling to van de Geer et al. (2014).

Do not make #134 depend on an unmerged runtime branch. After runtime behavior is stable and eventually merged, sync master into #132/#134 and update learner-first pages against the shipped contract.

## 13. Implementation order

1. Add failing tests for missing override, response-scale dependence, stale state, and unchecked node-wise KKT.
2. Add shared validation/standardization/alpha/KKT/normalizer helpers and internal node-wise constants.
3. Plumb public constructor state through generic penalized linear, Lasso, ElasticNet.
4. Add result state to canonical cleanup/failure transactions.
5. Implement NumPy standardized node-wise solve, full KKT gate, positive normalizer, back-transform.
6. Implement CuPy/Torch `Sigma_Z` Gram-space version and recomputed Lipschitz bound.
7. Make publication atomic and preserve metadata through weighted/centered finalizers and simultaneous inference.
8. Propagate through LassoCV/ElasticNetCV and outer fitted state.
9. Close cache/formula/weight/p=1/direct-fit invariance tests.
10. Add independent reference + simulation validator.
11. Update runtime docs/changelog.
12. Run local-minimal then local-full relevant suites.
13. Run physical GPU acceptance when available.
14. Fresh independent implementation review; fix CRITICAL/HIGH and relevant MEDIUM; rerun affected gates; repeat until clean or a user decision is required.

## 14. Acceptance criteria

- [ ] public `nodewise_alpha=None|positive scalar` on all in-scope maintained surfaces;
- [ ] explicit value reaches every actual node-wise solve;
- [ ] auto rule independent of `y` scale;
- [ ] standardized design + correct `D^{-1}Theta_ZD^{-1}` back-transform;
- [ ] feature-scale equivariance;
- [ ] full node-wise KKT gate passes before publication;
- [ ] positive finite `C_j` gate; no identity-row fallback;
- [ ] cache keys actual node-wise settings, not parent-model tol;
- [ ] GPU Lipschitz bound uses standardized Gram;
- [ ] direct penalized coefficients and CV selection unaffected by nodewise alpha;
- [ ] weight identities preserved;
- [ ] NumPy/CuPy/Torch parity with no explicit-device fallback;
- [ ] marginal/simultaneous inference share one alpha/M;
- [ ] nodewise provenance survives finalizers and is atomically/reset safely published;
- [ ] get_params/set_params/clone/formula/CV/p=1 regressions pass;
- [ ] source/docs no longer misattribute old response-scale rule;
- [ ] exact-head local blocking tests pass;
- [ ] physical GPU evidence recorded or status explicitly `PARTIAL_REMOTE_PENDING`;
- [ ] fresh independent implementation review has no unresolved CRITICAL/HIGH.

## 15. Non-goals

- vector per-coordinate node-wise alpha in v1;
- node-wise CV / square-root Lasso as auto selector;
- public nodewise tolerance/max-iter controls in this repair;
- main-model CV redesign;
- full ElasticNetCV inference-selector redesign;
- p=1 behavior redesign;
- unrelated cache hash-policy redesign;
- private legacy cosmetic parity;
- performance claims.

## 16. Plan review log

### Round 1

Plan commit: `3b83e3a1a84d18769ba0267ba74fd4aa281a08ec`  
Artifact: `dev/reviews/nodewise-alpha-plan-review-round-1.md`  
Verdict: **PLAN CHANGES REQUIRED**

Closed: #138 working-data ownership, fitted-state cleanup, positive `C_j`, raw validation, finalizer metadata, p=1 scope, centralized degeneracy tolerance.

### Round 2

Plan commit: `cf71a6915b307763391663be5e0de255c9f8a474`  
Artifact: `dev/reviews/nodewise-alpha-plan-review-round-2.md`  
Verdict: **PLAN CHANGES REQUIRED**

Closed in this revision:

- HIGH: production full-KKT acceptance gate added;
- HIGH: cache now keys actual node-wise numerical settings rather than parent estimator tol;
- MEDIUM: GPU Lipschitz recomputed from standardized Gram;
- MEDIUM: atomic publication specified;
- MEDIUM: direct-fit coefficient invariance explicitly tested;
- MEDIUM: node-wise solver/tol/max-iter/KKT residual added to provenance.

Next step: fresh round-3 plan review on this exact revision. Implementation must not start before a clean plan verdict.