# Node-wise Lasso tuning contract — implementation plan

Status: **REVISED AFTER PLAN REVIEW ROUND 3**

Target baseline:

- repository: `TheHiddenObserver/statgpu`
- implementation branch: `fix/nodewise-alpha-inference-contract`
- base `master`: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
- affected capability: squared-error L1 / Elastic Net debiased inference and the approximate-precision construction used by marginal and simultaneous inference

This work is independent of documentation PR #134. Runtime/API behavior is fixed and reviewed here first; the still-Draft #132/#134 documentation stack is synchronized only after the runtime contract is stable.

## 1. Defects being fixed

1. The node-wise Lasso penalty is inference-critical but is internal-only.
2. The current automatic penalty multiplies by the main response residual scale, so changing only the units of `y` can change the design-side precision matrix `M`.
3. Source wording incorrectly presents that response-scale multiplier as the van de Geer et al. (2014) construction.
4. The internal node-wise numerical contract is underspecified: hidden convergence settings are not reported, cache identity uses the parent-model tolerance rather than the actual node-wise tolerance, and finite output can be published without an independent node-wise KKT gate.
5. Current single-feature (`p=1`) debiased behavior is not one clean cross-backend contract even though no nuisance node-wise regression is needed mathematically.

## 2. Impact classification

Active:

- public API / sklearn compatibility;
- inference, result provenance, failure safety;
- NumPy / CuPy / Torch closure;
- intercept and analytic-weight semantics;
- LassoCV / ElasticNetCV final-refit propagation;
- formula/model-matrix parity;
- node-wise convergence/KKT correctness;
- docs/changelog/evidence.

Not changed:

- main Lasso/ElasticNet objective;
- main-model `alpha` / `l1_ratio` selection;
- unrelated model families;
- public solver inventory;
- performance claims.

## 3. Public API

### 3.1 Parameter

Add

```python
nodewise_alpha: Optional[float] = None
```

to:

- `PenalizedGeneralizedLinearModel`;
- `PenalizedLinearRegression`;
- `Lasso`;
- `ElasticNet`;
- `LassoCV`;
- `ElasticNetCV`.

The generic GLM surface consumes it only for squared-error + L1/ElasticNet + `debiased` inference.

### 3.2 Validation / clone semantics

Store the requested constructor object unchanged. Validate the raw value without replacing it by a coerced float.

Accepted:

- `None`;
- finite real scalar `> 0` (including compatible NumPy real scalars).

Rejected with `ValueError`:

- bool;
- complex scalar;
- non-scalar/array;
- NaN/inf;
- zero/negative.

The same contract must hold through signature introspection, `get_params`, transactional `set_params`, sklearn clone, and internal reconstruction without warning noise.

### 3.3 Meaning

- `alpha`: main penalized prediction/selection estimator.
- `nodewise_alpha`: only the standardized node-wise regressions used to construct the approximate precision matrix for `debiased` inference when `p>=2`.

Changing only `nodewise_alpha` must not change direct or CV penalized prediction coefficients, fold scores, selected `alpha_`, or selected `l1_ratio_`.

Version 1 is scalar-only. Per-coordinate node-wise penalties are deferred.

### 3.4 Fitted state / provenance

For `p>=2`, after successful node-wise debiased inference publish:

```python
nodewise_alpha_
```

and metadata:

```text
precision_method = "nodewise_lasso"
nodewise_alpha
nodewise_alpha_source = "user" | "auto"
nodewise_alpha_rule = "explicit" | "standardized_universal_v1"
nodewise_design_standardized = true
nodewise_effective_n
nodewise_weighted
nodewise_solver = "fista"
nodewise_tol
nodewise_max_iter
nodewise_max_kkt_residual
```

For the analytic `p=1` path:

```text
precision_method = "analytic_univariate"
nodewise_alpha = null
nodewise_alpha_source = "not_applicable"
nodewise_alpha_rule = "not_applicable"
```

and `nodewise_alpha_ is None` because no node-wise Lasso solve occurs. A requested constructor `nodewise_alpha` remains visible in `get_params()` but is numerically unused for `p=1`; docs must say so.

When the executed inference path does not use node-wise precision, `nodewise_alpha_` remains `None`.

## 4. Canonical working-data ownership

PR #138's installed sparse-Gaussian contracts remain authoritative for centering and analytic-weight normalization. They already create the centered average-loss working design, including row scaling

$$
\sqrt{w_i n / \sum_k w_k}.
$$

The new precision helper must **not** reconstruct centering or weighted rows and must never use `y` to choose node-wise alpha. It receives the already-canonical working design currently used by the debiased routine.

The existing wrapper that owns analytic weights additionally supplies a response-independent node-wise effective sample-size context (Section 5) to the precision resolver. This context is transient execution state, not a second weighting transform, and must be installed/restored transactionally so it cannot leak between fits.

Existing identities remain blocking:

- omitted weights = all-one weights;
- global weight scaling `w -> c w` changes neither the canonical design nor node-wise auto tuning;
- adding/removing zero-weight rows changes neither the weighted precision problem nor node-wise auto tuning;
- explicit CuPy/Torch inference stays on the selected concrete device.

## 5. Weight-aware node-wise effective sample size

The default alpha is a statistical tuning heuristic, not merely an objective-normalization constant. For unweighted data use

$$
n_{\mathrm{nw}}=n.
$$

For analytic weights `w_i >= 0` with positive finite total `W`, use the Kish-style response-independent effective sample size

$$
\boxed{
n_{\mathrm{nw}}
=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
}
$$

Validate it as finite and in `(0,n]` up to floating-point tolerance; clamp only tiny roundoff above `n`, not materially invalid values.

This convention is a statgpu v1 default heuristic. It is chosen because it:

- is independent of `y`;
- is invariant to global multiplication of all weights;
- equals `n` for equal positive weights;
- equals the number of equally weighted nonzero observations when arbitrary zero-weight rows are present;
- becomes smaller when a few observations dominate the analytic weights.

It is not documented as a uniquely theorem-mandated effective sample size. Users who want a different tuning choice can supply `nodewise_alpha` explicitly.

Metadata records the resolved `nodewise_effective_n`.

## 6. Statistical precision contract

### 6.1 Standardized working design

For canonical working design `X_w`, define

$$
d_j=\sqrt{\frac1n\sum_iX_{w,ij}^2},\qquad
D=\operatorname{diag}(d_j),\qquad
Z=X_wD^{-1}.
$$

The objective still uses the canonical average-loss row normalization `1/n`; `n_nw` from Section 5 only selects the automatic penalty magnitude.

### 6.2 Design-scale validity

All maintained debiased inference is float64 at this boundary. Shared rule:

1. all `d_j` and `d_max=max_j d_j` finite;
2. `d_max>0`;
3. `d_tol = 64 * eps64 * d_max`;
4. all `d_j > d_tol`.

Failure aborts inference with no partial publication.

### 6.3 Automatic alpha

For `p>=2` and `nodewise_alpha=None`:

$$
\boxed{
\lambda_{\mathrm{nw}}^{\mathrm{auto}}
=
\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}
}
$$

on standardized `Z`.

The `sqrt(log p / n)` order is literature-motivated; the exact `sqrt(2)` and weighted `n_nw` convention are documented as statgpu defaults, not unique theorem prescriptions.

Explicit `nodewise_alpha=x` passes numerical `x` unchanged to every node-wise solve on the same standardized scale.

### 6.4 Node-wise objective (`p>=2`)

For coordinate `j`:

$$
\hat\gamma_j
=\arg\min_\gamma
\left\{
\frac{1}{2n}\|Z_j-Z_{-j}\gamma\|_2^2
+\lambda_{\mathrm{nw}}\|\gamma\|_1
\right\}.
$$

Let

$$
r_j=Z_j-Z_{-j}\hat\gamma_j.
$$

Use the paper-style node-wise normalizer

$$
\boxed{
\hat\tau_j^2
=
\frac{\|r_j\|_2^2}{n}
+
\lambda_{\mathrm{nw}}\|\hat\gamma_j\|_1.
}
$$

Construct

$$
\hat\Theta_{Z,j}^\top
=
\frac{(e_j-\tilde\gamma_j)^\top}{\hat\tau_j^2}.
$$

Back-transform:

$$
\boxed{M_X=D^{-1}\hat\Theta_ZD^{-1}.}
$$

Existing debiasing and variance formulas continue with canonical `X_w`, `Sigma_{X_w}`, and `M_X`.

### 6.5 Normalizer and KKT consistency

Require every `tau_j^2` finite and

$$
\hat\tau_j^2 > 64\epsilon_{64}.
$$

Also compute

$$
C_j^{\mathrm{cross}}=\frac1n Z_j^\top r_j.
$$

Under exact KKT, `C_j_cross = tau_j^2`. After the independent KKT check in Section 7, require their discrepancy to obey

$$
|C_j^{\mathrm{cross}}-\hat\tau_j^2|
\le
\|\hat\gamma_j\|_1\,R_j^{\mathrm{KKT}}
+64\epsilon_{64}\max(1,\hat\tau_j^2),
$$

up to a small centralized implementation rounding factor if backend validation demonstrates it is necessary. Any such factor must be shared across backends and recorded in the helper constant/tests.

The old `abs(C_j)<1e-30 -> identity row` behavior is removed.

### 6.6 Analytic univariate path (`p=1`)

For one valid standardized feature, no nuisance regression exists. Since

$$
\Sigma_Z=[1],
$$

set

$$
\Theta_Z=[1],\qquad
M_X=[1/d_1^2].
$$

Use this analytic precision on NumPy, CuPy, and Torch. No FISTA call, node-wise alpha resolution, KKT solve, or node-wise cache entry is needed.

This intentionally closes the previous backend inconsistency rather than preserving it.

## 7. Node-wise numerical solver contract (`p>=2`)

### 7.1 Internal v1 settings

Centralize:

```text
NODEWISE_SOLVER = "fista"
NODEWISE_TOL = 1e-5
NODEWISE_MAX_ITER = 500
NODEWISE_KKT_TOL = 1e-5
```

The CPU internal solve must pass `solver="fista"` explicitly; deprecated/shared `cpu_solver` must not be relied on to select the algorithm.

If representative supported designs cannot meet the KKT gate with the current iteration cap, improve/increase the internal solve rather than silently weakening correctness.

No public `nodewise_tol`/`nodewise_max_iter` in v1.

### 7.2 Independent full KKT gate

With standardized Gram blocks,

$$
g_j=\Sigma_{-j,-j}^{(Z)}\hat\gamma_j-\Sigma_{-j,j}^{(Z)}.
$$

For each coordinate `k`, define

$$
r_{jk}^{\mathrm{KKT}}=
\begin{cases}
|g_{jk}+\lambda_{\mathrm{nw}}\operatorname{sign}(\hat\gamma_{jk})|,&\hat\gamma_{jk}\ne0,\\
\max(|g_{jk}|-\lambda_{\mathrm{nw}},0),&\hat\gamma_{jk}=0.
\end{cases}
$$

and

$$
R_j^{\mathrm{KKT}}=\max_k r_{jk}^{\mathrm{KKT}}.
$$

Require

$$
R_j^{\mathrm{KKT}}\le\texttt{NODEWISE_KKT_TOL}.
$$

This independent gate is authoritative even if the underlying solver reports convergence.

## 8. Implementation architecture

### 8.1 Shared helper module

Add one maintained helper under `statgpu/linear_model/penalized/` for:

- raw public-parameter validation;
- design scales/validation;
- weighted effective-n validation/resolution;
- auto/explicit alpha resolution;
- KKT residual;
- `tau_j^2` / cross-product consistency validation;
- back-transform;
- provenance metadata.

It never uses `y` to resolve precision tuning.

Do not create another install-time monkeypatch solely for this feature.

### 8.2 NumPy

Inside the maintained original debiased routine invoked by #138 wrappers:

1. receive canonical `X_w` plus transient node-wise effective-n context;
2. validate/standardize to `Z`;
3. branch analytically for `p=1`;
4. otherwise resolve alpha;
5. solve every node-wise problem with explicit FISTA/internal settings;
6. recompute full KKT;
7. compute/validate `tau_j^2` and cross-product consistency;
8. build `Theta_Z` and back-transform `M_X`;
9. continue existing marginal inference.

### 8.3 CuPy/Torch

Remain on selected concrete GPU device.

Use

$$
\Sigma_Z=D^{-1}\Sigma_{X_w}D^{-1}
$$

to form batched Gram/cross-product inputs. Recompute the FISTA Lipschitz bound from `Sigma_Z`, solve with the single resolved alpha, run KKT/tau consistency checks on device, and back-transform `M_X` on device.

The `p=1` analytic path also remains device-native until the normal reporting boundary.

No CPU numerical fallback.

### 8.4 Existing #138 wrappers/finalizers

Preserve:

- `_post_selection_ols_fifth_review_contract` as owner of centered/weighted working-data formation;
- transactional installation/restoration of the additional node-wise effective-n context around calls into the original debiased routine;
- `_post_selection_ols_review_fix_contract._finalize_weighted_debiased_result` preserving node-wise metadata from the base result;
- simultaneous inference reusing the same validated `M_X` / alpha (or analytic univariate precision).

No second hidden resolver is allowed.

## 9. Atomic publication / state reset

Keep resolved alpha, effective-n, scales, precision matrix, KKT metrics, and reports in candidate/native local state until the inference transaction can commit.

Publish `nodewise_alpha_`, `_debiased_M_cpu`/native capture, and metadata only after precision validation and the required marginal/finalizer transaction succeeds.

Clear all node-wise fitted/transient state through:

- canonical `_clear_inference_state`;
- sparse-inference failed-refit invalidation;
- generic no-inference/failed-fit cleanup;
- LassoCV reset;
- ElasticNetCV reset;
- `finally` restoration of transient weighted effective-n context.

Test success→failure→inspection and success→`set_params`→refit.

## 10. Cache contract

For `p>=2`, key the actual node-wise problem:

- canonical design identity;
- resolved `nodewise_alpha`;
- `NODEWISE_SOLVER`;
- `NODEWISE_TOL`;
- `NODEWISE_MAX_ITER`;
- precision/standardization contract version.

Do not key parent-model `tol` unless it becomes part of the node-wise numerical contract.

Auto and explicit requests resolving to the same numeric alpha may share the precision cache; request provenance remains current-call metadata.

`p=1` analytic precision is not stored in the node-wise Lasso cache.

## 11. CV propagation

### LassoCV

Add `nodewise_alpha=None` as final-refit inference configuration only.

- not part of main-alpha selection/cache;
- pass unchanged to final `Lasso`;
- changing it cannot alter selected `alpha_`, `mse_path_`, penalized `coef_`/`intercept_`;
- outer resolved `nodewise_alpha_` mirrors final estimator after successful `p>=2` debiased inference, otherwise `None`;
- reset before every fit.

### ElasticNetCV

Propagate analogously.

The existing hard-coded final `inference_method="debiased"` is an adjacent API inconsistency but not required for this tuning contract. Do not broaden into a full inference-selector redesign unless implementation review proves it is necessary.

### PenalizedGLM_CV

At this baseline the generic CV surface does not expose the affected final-refit inference toggle; no change unless implementation inspection disproves that assumption.

## 12. Tests / evidence

### 12.1 Public API and state

- omission / explicit valid Python and NumPy real scalar;
- invalid bool/complex/non-scalar/NaN/inf/zero/negative;
- signature, get_params, set_params, clone;
- fitted-state invalidation and no warning noise;
- non-debiased path leaves `nodewise_alpha_ is None`;
- direct Lasso/ElasticNet `coef_`/`intercept_` invariant to node-wise alpha.

### 12.2 Statistical invariants

1. **response independence**: fixed `X`/weights/intercept gives identical auto precision for two arbitrary finite `y` vectors, not merely rescaled `y`;
2. positive diagonal feature scaling `A`: `M_{XA} ≈ A^{-1}M_XA^{-1}`;
3. auto→explicit same alpha equivalence for `p>=2`;
4. all-one weight identity;
5. global weight-scale identity;
6. adding/removing zero-weight rows leaves weighted auto alpha and precision unchanged;
7. representative highly unequal weights produce finite `n_nw`, finite alpha, and valid precision/KKT;
8. invalid design scales fail closed;
9. bad/unconverged node-wise solution fails KKT;
10. tau/cross-product inconsistency beyond KKT-derived bound fails closed;
11. simultaneous inference reuses same precision contract;
12. p=1 analytic precision matches `1/(X_w'X_w/n)` and behaves consistently on all backends.

### 12.3 Backend parity

NumPy/CuPy/Torch:

- same auto alpha/effective-n for `p>=2`;
- explicit alpha reaches actual solver;
- same standardized statistical contract;
- KKT/tau gates pass;
- `M`, params, SE/z/p/marginal and simultaneous CI parity within maintained tolerances;
- correct concrete-device provenance;
- no explicit-device fallback;
- GPU Lipschitz uses standardized Gram;
- p=1 analytic path parity.

### 12.4 CV / formula

- node-wise alpha affects only final inference;
- LassoCV/ElasticNetCV selection outputs invariant to it;
- outer/inner fitted nodewise provenance agrees;
- array/formula inference agrees after design construction, including intercept, categorical ordering, and weights where supported.

### 12.5 Independent numeric reference

Build a small independent NumPy reference with explicit alpha:

- obtain canonical working design via existing trusted test helper;
- standardize;
- solve node-wise Lasso at aligned objective scale;
- compute KKT, `tau_j^2`, `Theta_Z`, back-transform;
- compare `M_X` and `M_X Sigma_X` approximation.

Optional aligned-lambda R `hdi` comparison is supplementary; its own CV tuning need not equal statgpu auto tuning.

### 12.6 Simulation evidence

Deterministic validator compares old/new default on representative sparse Gaussian designs and records:

- marginal coverage;
- interval length;
- finite/failure counts;
- precision residual;
- resolved alpha/effective-n/source/rule;
- KKT residual distribution.

For weighted cases include equal, zero-containing, and unequal weights. Simulation is evidence, not a unit-test assertion of exact 95% coverage.

### 12.7 Physical CUDA

Focused physical validator for CuPy + Torch:

- auto/explicit alpha;
- auto-explicit equivalence;
- response independence;
- feature-scale equivariance;
- weight identities and zero-weight-row invariance;
- KKT/tau gates;
- p=1 analytic path;
- simultaneous reuse;
- metadata/backend/device provenance;
- exact SHA/environment and partial failure artifact.

If physical evidence is the only unavailable gate, report `PARTIAL_REMOTE_PENDING`.

## 13. Documentation / #134 integration

Runtime docs/changelog must state:

- `nodewise_alpha` is separate from main `alpha`;
- `None` = standardized design-side auto rule;
- weighted auto tuning uses the documented effective-n heuristic;
- explicit value is on standardized node-wise scale;
- auto default is literature-motivated, not uniquely theorem-mandated;
- precision tuning is independent of `y`;
- scalar-only v1;
- `p=1` uses analytic precision and does not consume node-wise alpha;
- internal numerical settings are provenance-recorded;
- CV parameter affects final-refit inference only.

Remove the old source attribution tying main-response `sigma_hat` scaling to van de Geer et al. (2014).

Do not make #134 depend on this unmerged runtime branch. After runtime behavior is stable and eventually merged, sync master into #132/#134 and reconcile learner-first docs against the shipped API.

## 14. Implementation order

1. Add failing tests for missing override, response dependence, zero-weight-row sensitivity, p=1 backend inconsistency, stale state, and unchecked KKT.
2. Add shared validation/standardization/effective-n/alpha/KKT/tau helpers and internal constants.
3. Plumb public constructor state through generic penalized linear, Lasso, ElasticNet.
4. Add node-wise result/transient state to reset/failure transactions.
5. Implement NumPy p=1 analytic and p>=2 standardized node-wise paths.
6. Implement CuPy/Torch standardized Gram path, recomputed Lipschitz, and analytic p=1.
7. Make publication atomic and preserve metadata through #138 finalizers/simultaneous inference.
8. Propagate through LassoCV/ElasticNetCV.
9. Close cache/formula/weight/direct-fit/CV invariance tests.
10. Add independent reference and simulation validator.
11. Update runtime docs/changelog.
12. Run local-minimal then local-full relevant suites.
13. Run physical GPU acceptance when available.
14. Fresh independent implementation review; fix CRITICAL/HIGH and relevant MEDIUM; rerun affected gates; repeat until clean or user decision required.

## 15. Acceptance criteria

- [ ] public `nodewise_alpha=None|positive real scalar` on all in-scope maintained surfaces;
- [ ] explicit value reaches every actual p>=2 node-wise solve;
- [ ] auto precision tuning never depends on `y`;
- [ ] weighted auto effective-n is scale-invariant and zero-weight-row invariant;
- [ ] standardized design + correct `D^{-1}Theta_ZD^{-1}` transform;
- [ ] paper-style `tau_j^2` normalizer with KKT-consistency check;
- [ ] full KKT gate before publication;
- [ ] p=1 one-backend-contract analytic precision;
- [ ] cache keys actual node-wise numerical contract;
- [ ] GPU Lipschitz uses standardized Gram;
- [ ] direct penalized coefficients and CV selection unaffected by nodewise alpha;
- [ ] all-one/global-weight-scale identities preserved;
- [ ] NumPy/CuPy/Torch parity with no explicit-device fallback;
- [ ] marginal/simultaneous inference share one precision contract;
- [ ] node-wise provenance survives finalizers and all state is atomically/reset safely handled;
- [ ] get_params/set_params/clone/formula/CV regressions pass;
- [ ] source/docs no longer misattribute response-scale tuning;
- [ ] exact-head local blocking suites pass;
- [ ] physical GPU evidence recorded or explicit `PARTIAL_REMOTE_PENDING`;
- [ ] fresh independent implementation review has no unresolved CRITICAL/HIGH.

## 16. Non-goals

- per-coordinate vector node-wise alpha in v1;
- node-wise CV / square-root Lasso as auto selector;
- public nodewise tolerance/max-iter controls;
- main-model CV redesign;
- full ElasticNetCV inference-selector redesign;
- unrelated cache-policy redesign;
- private legacy cosmetic parity;
- performance claims.

## 17. Plan review log

### Round 1

Plan commit: `3b83e3a1a84d18769ba0267ba74fd4aa281a08ec`  
Artifact: `dev/reviews/nodewise-alpha-plan-review-round-1.md`  
Verdict: **PLAN CHANGES REQUIRED**

Closed: #138 working-data ownership; fitted-state cleanup; positive normalizer; raw validation; finalizer metadata; degeneracy tolerance.

### Round 2

Plan commit: `cf71a6915b307763391663be5e0de255c9f8a474`  
Artifact: `dev/reviews/nodewise-alpha-plan-review-round-2.md`  
Verdict: **PLAN CHANGES REQUIRED**

Closed: independent KKT gate; actual node-wise cache settings; standardized-Gram GPU Lipschitz; atomic publication; direct-fit coefficient invariance; numerical provenance.

### Round 3

Plan commit: `ca84b50b550af1b13bb8f8ab04761095167f0286`  
Artifact: `dev/reviews/nodewise-alpha-plan-review-round-3.md`  
Verdict: **PLAN CHANGES REQUIRED**

Closed in this revision:

- HIGH: weighted auto tuning now uses documented response-independent effective sample size and explicitly protects zero-weight-row invariance;
- HIGH: p=1 is now one analytic NumPy/CuPy/Torch precision contract rather than an inherited backend split;
- MEDIUM: production normalizer is paper-style `tau_j^2`; `Z_j'r_j/n` is retained as a KKT-derived numerical consistency check.

Next step: fresh round-4 plan review on this exact revision. Implementation must not start before a clean plan verdict.