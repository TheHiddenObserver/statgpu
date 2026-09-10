# Node-wise Lasso tuning contract — implementation plan

Status: **PLAN REVIEW CLEAN — READY FOR IMPLEMENTATION**

Target baseline:

- repository: `TheHiddenObserver/statgpu`
- implementation branch: `fix/nodewise-alpha-inference-contract`
- base `master`: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
- affected capability: squared-error L1 / Elastic Net debiased inference and the approximate-precision construction used by marginal and simultaneous inference

This work is independent of documentation PR #134. Runtime/API behavior is fixed and reviewed here first; the still-Draft #132/#134 documentation stack is synchronized only after the runtime contract is stable.

## 1. Defects being fixed

1. The node-wise Lasso penalty is inference-critical but internal-only.
2. The historical automatic penalty multiplies by the main-response residual scale, so changing only `y` units can change the design-side precision matrix `M`.
3. Source wording incorrectly presents that response-scale multiplier as the van de Geer et al. (2014) construction.
4. The node-wise numerical contract is underspecified: hidden convergence settings are not fully provenance-recorded, cache identity uses the parent-model tolerance rather than the actual node-wise tolerance, and finite output can be published without an independent full KKT gate.
5. Single-feature (`p=1`) debiased precision lacks one clean shared backend contract despite requiring no nuisance node-wise regression.

## 2. Compatibility / migration decision

This is an **intentional statistical-default correction**, not a numerically backward-compatible additive option.

Before this repair, callers who omit a node-wise control receive an internal response-dependent penalty on an unstandardized node-wise design. After this repair, omission means `nodewise_alpha=None`, which resolves on a standardized design using the design/weight-side rule below.

There is **no legacy public compatibility mode** for the old response-dependent rule because:

- no public parameter previously exposed or promised that rule;
- the rule makes a design-side precision construction depend on response units;
- preserving it as a new legacy selector would turn an internal defect into a public long-term contract.

The behavior change must be explicit in changelog/docs and in old-vs-new numerical evidence. Existing users who require a particular new standardized-scale node-wise penalty can set `nodewise_alpha=` explicitly. Exact reproduction of historical unstandardized internal `M` is not a supported migration promise.

## 3. Impact classification

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

## 4. Public API

### 4.1 Parameter

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

### 4.2 Validation / clone semantics

Store the requested constructor object unchanged. Validate the raw value without replacing it by a coerced float.

Accepted:

- `None`;
- finite real scalar `> 0`, including compatible NumPy real scalars.

Rejected with `ValueError`:

- bool;
- complex scalar;
- non-scalar/array;
- NaN/inf;
- zero/negative.

The same contract must hold through signature introspection, `get_params`, transactional `set_params`, sklearn clone, and internal reconstruction without warning noise.

### 4.3 Meaning

- `alpha`: main penalized prediction/selection estimator.
- `nodewise_alpha`: only the standardized node-wise regressions used to construct approximate precision for `debiased` inference when `p>=2`.

Changing only `nodewise_alpha` must not change direct or CV penalized prediction coefficients, fold scores, selected `alpha_`, or selected `l1_ratio_`.

Version 1 is scalar-only. Per-coordinate node-wise penalties are deferred.

### 4.4 Fitted state / provenance

For `p>=2`, after the **entire requested inference transaction** succeeds, publish:

```python
nodewise_alpha_
```

and include in `_inference_result.metadata`:

```text
precision_method = "nodewise_lasso"
nodewise_alpha_requested = null | float
nodewise_alpha
nodewise_alpha_source = "user" | "auto"
nodewise_alpha_rule = "explicit" | "standardized_universal_v1"
nodewise_design_standardized = true
nodewise_effective_n
nodewise_weighted
nodewise_solver = "fista"
nodewise_stopping = "coef_delta"
nodewise_tol
nodewise_max_iter
nodewise_kkt_tol
nodewise_max_kkt_residual
```

Metadata serializes a valid requested scalar as an ordinary float; the constructor itself still preserves the original object.

For analytic `p=1`:

```text
precision_method = "analytic_univariate"
nodewise_alpha_requested = null | float
nodewise_alpha = null
nodewise_alpha_source = "not_applicable"
nodewise_alpha_rule = "not_applicable"
```

and `nodewise_alpha_ is None`. A requested value remains visible in constructor/get_params state but is not numerically consumed because no node-wise Lasso exists.

Other inference paths also leave `nodewise_alpha_ is None`.

## 5. Canonical working-data ownership

PR #138's installed sparse-Gaussian contracts remain authoritative for centering and analytic-weight normalization. They already create the centered average-loss working design, including row scaling

$$
\sqrt{w_i n / \sum_k w_k}.
$$

The new precision helper must **not** reconstruct centering or weighted rows and must never use `y` to resolve node-wise tuning. It receives the already-canonical working design currently consumed by the debiased routine.

The existing weighted wrapper additionally computes and supplies a response-independent node-wise effective-sample-size context (Section 6). That transient context is installed/restored transactionally; it is not a second weighting transformation and may not leak between fits.

Blocking identities:

- omitted weights = all-one weights;
- global weight scaling `w -> c w` changes neither precision design nor auto tuning;
- adding/removing zero-weight rows changes neither the weighted precision problem nor auto tuning;
- explicit CuPy/Torch inference stays on the selected concrete device.

## 6. Weight-aware effective sample size for auto tuning

For unweighted data:

$$
n_{\mathrm{nw}}=n.
$$

For analytic weights `w_i>=0` with positive finite total:

$$
\boxed{
n_{\mathrm{nw}}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
}
$$

Validate finite and in `(0,n]` up to floating-point roundoff; only tiny roundoff above `n` may be clipped.

This Kish-style quantity is a **statgpu v1 heuristic for the automatic node-wise penalty**, not a theorem claim. It is chosen because it is response-independent, global-weight-scale invariant, equals `n` for equal positive weights, ignores arbitrary zero-weight rows in the equal-weight case, and decreases when a small number of rows dominate the weights. Users can override the resulting penalty directly through `nodewise_alpha`.

Record the resolved `nodewise_effective_n` in metadata.

## 7. Statistical precision contract

### 7.1 Standardized working design

For canonical working design `X_w`:

$$
d_j=\sqrt{\frac1n\sum_iX_{w,ij}^2},\qquad
D=\operatorname{diag}(d_j),\qquad
Z=X_wD^{-1}.
$$

The node-wise objective retains canonical average-loss normalization `1/n`; `n_nw` only sets automatic penalty magnitude.

### 7.2 Design-scale validity

At the maintained float64 inference boundary:

1. all `d_j` and `d_max=max_jd_j` finite;
2. `d_max>0`;
3. `d_tol=64*eps64*d_max`;
4. all `d_j>d_tol`.

Failure aborts inference before publication.

### 7.3 Automatic alpha (`p>=2`)

For `nodewise_alpha=None`:

$$
\boxed{
\lambda_{\mathrm{nw}}^{\mathrm{auto}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}}
}
$$

on standardized `Z`.

The `sqrt(log p/n)` order is literature-motivated; the exact `sqrt(2)` and weighted effective-n convention are statgpu defaults, not uniquely theorem-mandated.

Explicit `nodewise_alpha=x` passes numerical `x` unchanged to every node-wise solve on the same standardized scale.

### 7.4 Node-wise objective (`p>=2`)

For coordinate `j`:

$$
\hat\gamma_j=\arg\min_\gamma\left\{
\frac{1}{2n}\|Z_j-Z_{-j}\gamma\|_2^2
+\lambda_{\mathrm{nw}}\|\gamma\|_1\right\},
$$

$$
r_j=Z_j-Z_{-j}\hat\gamma_j.
$$

Use the paper-style normalizer

$$
\boxed{
\hat\tau_j^2=\frac{\|r_j\|_2^2}{n}
+\lambda_{\mathrm{nw}}\|\hat\gamma_j\|_1.
}
$$

Then

$$
\hat\Theta_{Z,j}^\top
=\frac{(e_j-\tilde\gamma_j)^\top}{\hat\tau_j^2},
\qquad
\boxed{M_X=D^{-1}\hat\Theta_ZD^{-1}.}
$$

Existing debiasing/variance formulas continue with canonical `X_w`, `Sigma_{X_w}`, and `M_X`.

### 7.5 Normalizer / cross-product consistency

Require every `tau_j^2` finite and

$$
\hat\tau_j^2>64\epsilon_{64}.
$$

Also compute

$$
C_j^{\mathrm{cross}}=n^{-1}Z_j^\top r_j.
$$

After the full KKT check, require

$$
|C_j^{\mathrm{cross}}-\hat\tau_j^2|
\le
\|\hat\gamma_j\|_1 R_j^{\mathrm{KKT}}
+64\epsilon_{64}\max(1,\hat\tau_j^2),
$$

apart from one centralized, cross-backend floating-point rounding factor if validation proves it necessary. Any factor change requires tests/evidence; do not use backend-specific magic.

Remove the old tiny-`C_j` identity-row fallback.

### 7.6 Analytic univariate path (`p=1`)

For valid standardized `Z`, `Sigma_Z=[1]`, so

$$
\Theta_Z=[1],\qquad M_X=[1/d_1^2].
$$

Use this on NumPy/CuPy/Torch. No FISTA, alpha resolution, KKT solve, or node-wise cache entry occurs.

## 8. Node-wise numerical solver contract (`p>=2`)

Centralize:

```text
NODEWISE_SOLVER = "fista"
NODEWISE_STOPPING = "coef_delta"
NODEWISE_TOL = 1e-5
NODEWISE_MAX_ITER = 500
NODEWISE_KKT_TOL = 1e-5
```

The internal CPU estimator must pass `solver="fista"` and `stopping="coef_delta"` explicitly; deprecated/shared `cpu_solver` is not authoritative for this numerical contract.

`coef_delta` is only the iterative stopping policy. **Independent post-solve full KKT validation is the publication gate.** Existing batched solver-specific `stopping="kkt"` logic is not reused as a substitute for this full check.

If supported designs cannot satisfy the independent KKT gate with the iteration cap, improve/increase the internal solve rather than silently weakening the gate.

### 8.1 Full KKT gate

With standardized Gram blocks:

$$
g_j=\Sigma_{-j,-j}^{(Z)}\hat\gamma_j-\Sigma_{-j,j}^{(Z)}.
$$

Define

$$
r_{jk}^{\mathrm{KKT}}=
\begin{cases}
|g_{jk}+\lambda_{\mathrm{nw}}\operatorname{sign}(\hat\gamma_{jk})|,&\hat\gamma_{jk}\ne0,\\
\max(|g_{jk}|-\lambda_{\mathrm{nw}},0),&\hat\gamma_{jk}=0,
\end{cases}
$$

$$
R_j^{\mathrm{KKT}}=\max_k r_{jk}^{\mathrm{KKT}},
\qquad
R_j^{\mathrm{KKT}}\le\texttt{NODEWISE_KKT_TOL}.
$$

The independent check is authoritative even if the iterative solver reports convergence.

## 9. Implementation architecture

### 9.1 Shared helper

Add one maintained module under `statgpu/linear_model/penalized/` for:

- raw public-parameter validation;
- design scales/validation;
- weighted effective-n validation/resolution;
- auto/explicit alpha resolution;
- full KKT residual;
- `tau_j^2` and cross-product consistency;
- back-transform;
- provenance metadata.

It never uses `y` for precision tuning. Do not add another install-time monkeypatch solely for this feature.

### 9.2 NumPy

Inside the maintained original debiased routine invoked by #138 wrappers:

1. receive canonical `X_w` plus transient effective-n context;
2. validate/standardize to `Z`;
3. use analytic p=1 branch, otherwise resolve alpha;
4. solve every p>=2 node-wise problem with explicit internal settings;
5. recompute full KKT;
6. compute/validate `tau_j^2` and cross-product consistency;
7. build `Theta_Z`, back-transform `M_X`;
8. reject non-finite standardized Gram/precision state before debiasing;
9. continue existing marginal inference.

### 9.3 CuPy/Torch

Remain on selected concrete GPU device. Form

$$
\Sigma_Z=D^{-1}\Sigma_{X_w}D^{-1}.
$$

Build batched Gram/cross-product inputs from `Sigma_Z`; recompute the FISTA Lipschitz bound from `Sigma_Z`; run finite-state, KKT, and tau checks on device; back-transform and validate finite `M_X` on device. The p=1 analytic path also stays native until the normal reporting boundary.

No CPU numerical fallback.

### 9.4 Existing #138 wrappers/finalizers

Preserve:

- `_post_selection_ols_fifth_review_contract` ownership of centered/weighted working-data formation;
- transactional installation/restoration of node-wise effective-n context;
- `_post_selection_ols_review_fix_contract._finalize_weighted_debiased_result` preserving node-wise metadata;
- simultaneous inference reusing the same validated precision/alpha (or analytic univariate precision).

No second hidden alpha resolver is allowed.

## 10. Atomic publication / reset safety

Candidate alpha/effective-n/scales/precision/KKT/marginal and simultaneous reports remain internal until **all inference requested by the fit** succeeds.

If `enable_simultaneous_inference=False`, commit after the full marginal + centered/weighted finalizer transaction succeeds.

If `enable_simultaneous_inference=True`, the public commit boundary is only after marginal inference, centered/weighted finalization, and max-|Z| simultaneous calibration all succeed. Any simultaneous failure clears the candidate/resolved node-wise state together with all inference fields.

Only then publish `nodewise_alpha_`, `_debiased_M_cpu`/native reporting capture, and final metadata.

Clear node-wise fitted/transient state through:

- canonical `_clear_inference_state`;
- sparse-inference failed-refit invalidation;
- generic failed/no-inference cleanup;
- LassoCV reset;
- ElasticNetCV reset;
- `finally` restoration of transient effective-n context.

Test success→failure→inspection and success→`set_params`→refit.

## 11. Cache contract

For p>=2, key:

- canonical design identity;
- resolved numeric `nodewise_alpha`;
- `NODEWISE_SOLVER`;
- `NODEWISE_STOPPING`;
- `NODEWISE_TOL`;
- `NODEWISE_MAX_ITER`;
- precision/standardization contract version.

Do not key parent-model `tol` unless it becomes part of node-wise settings.

Auto and explicit requests with equal resolved alpha may share the numerical precision cache; current-request provenance is rebuilt separately. p=1 analytic precision does not enter the node-wise cache.

## 12. CV propagation

### LassoCV

Add `nodewise_alpha=None` as final-refit inference configuration only:

- not part of main-alpha selection/cache;
- pass unchanged to final `Lasso`;
- cannot alter selected `alpha_`, `mse_path_`, penalized `coef_`/`intercept_`;
- outer resolved `nodewise_alpha_` mirrors successful final estimator for p>=2, otherwise `None`;
- reset before every fit.

### ElasticNetCV

Propagate analogously.

The existing hard-coded final `inference_method="debiased"` is adjacent scope and is not redesigned here unless implementation review proves required for correctness.

### PenalizedGLM_CV

At the reviewed baseline generic CV does not expose the affected final-refit inference toggle, so no propagation unless implementation inspection disproves that assumption.

## 13. Tests / evidence

### 13.1 API/state

- omission / explicit valid Python and NumPy real scalar;
- invalid bool/complex/non-scalar/NaN/inf/zero/negative;
- signature/get_params/set_params/clone;
- fitted-state invalidation / no internal warning noise;
- non-debiased path leaves `nodewise_alpha_ is None`;
- direct Lasso/ElasticNet `coef_`/`intercept_` invariant to node-wise alpha;
- p=1 requested-value provenance is unambiguous.

### 13.2 Statistical invariants

1. fixed `X`/weights/intercept: auto precision identical for two arbitrary finite `y` vectors;
2. positive diagonal feature scaling `A`: `M_{XA}≈A^{-1}M_XA^{-1}`;
3. auto→explicit same-alpha equivalence for p>=2;
4. all-one weight identity;
5. global weight-scale identity;
6. zero-weight row add/drop invariance of weighted auto alpha/precision;
7. highly unequal valid weights produce finite effective-n/alpha and valid KKT precision;
8. invalid design scales fail closed;
9. bad/unconverged solve fails KKT;
10. tau/cross-product inconsistency beyond derived bound fails closed;
11. standardized Gram or back-transformed precision non-finiteness fails before reporting;
12. simultaneous path uses the same precision contract and leaves no state if calibration fails;
13. p=1 analytic precision equals `1/(X_w'X_w/n)` on all backends.

### 13.3 Backend parity

NumPy/CuPy/Torch:

- same p>=2 auto alpha/effective-n;
- explicit alpha reaches actual solver;
- same standardized contract;
- finite-state/KKT/tau gates;
- `M`, params, SE/z/p/marginal CI/simultaneous CI parity within maintained tolerances;
- concrete-device provenance / no explicit-device fallback;
- GPU Lipschitz from standardized Gram;
- p=1 analytic parity.

### 13.4 CV/formula

- node-wise alpha only affects final inference;
- LassoCV/ElasticNetCV selection outputs invariant;
- outer/inner fitted provenance agrees;
- array/formula inference parity after design construction, including intercept/categorical ordering and weights where supported.

### 13.5 Independent reference

Build a small independent NumPy reference at explicit alpha:

- trusted canonical working-design helper;
- standardize `Z`;
- independently solve aligned node-wise objective;
- compute KKT, tau, Theta_Z, back-transform;
- compare `M_X` and `M_X Sigma_X` approximation.

Optional aligned-lambda R `hdi` comparison is supplementary; its internal CV tuning need not match statgpu auto tuning.

### 13.6 Old/new simulation evidence

Because omission changes default behavior, create a deterministic validation artifact comparing historical and repaired defaults on representative sparse Gaussian designs. Record:

- marginal coverage;
- interval length;
- finite/failure counts;
- precision residual;
- resolved alpha/effective-n/source/rule;
- KKT residual distribution;
- unweighted and representative weighted cases.

This makes the intentional behavior migration auditable; it is not a unit-test claim of exact nominal coverage.

### 13.7 Physical CUDA

Focused validator for CuPy + Torch:

- auto/explicit and auto-explicit equivalence;
- response independence;
- feature-scale equivariance;
- weight identities / zero-weight invariance;
- finite-state/KKT/tau gates;
- p=1 analytic path;
- simultaneous reuse and failure cleanup;
- metadata/backend/device provenance;
- exact SHA/environment and partial failure artifact.

If physical evidence is the only unavailable gate, report `PARTIAL_REMOTE_PENDING`.

## 14. Documentation / #134 integration

Runtime docs/changelog must state:

- intentional change from historical response-dependent internal default;
- no legacy mode for that internal rule;
- `nodewise_alpha` separate from main `alpha`;
- `None` = standardized design-side auto rule;
- weighted auto tuning uses documented effective-n heuristic;
- explicit value is on standardized node-wise scale;
- default is literature-motivated, not uniquely theorem-mandated;
- precision tuning is independent of `y`;
- scalar-only v1;
- p=1 analytic precision does not consume node-wise alpha;
- internal solver/stopping/KKT settings are provenance-recorded;
- CV parameter affects final-refit inference only.

Remove the old source attribution tying main-response `sigma_hat` scaling to van de Geer et al. (2014).

Do not make #134 depend on this unmerged runtime branch. After runtime behavior is stable and eventually merged, sync master into #132/#134 and reconcile learner-first docs against the shipped API.

## 15. Implementation order

1. Add failing tests for missing override, response dependence, zero-weight sensitivity, p=1 inconsistency, stale state, unchecked KKT, simultaneous failure leakage, and non-finite precision publication.
2. Add shared validation/standardization/effective-n/alpha/KKT/tau helpers and internal constants.
3. Plumb public constructor state through generic penalized linear, Lasso, ElasticNet.
4. Add node-wise result/transient state to reset/failure transactions.
5. Implement NumPy p=1 analytic and p>=2 standardized paths.
6. Implement CuPy/Torch standardized-Gram path, recomputed Lipschitz, analytic p=1.
7. Make publication atomic through all requested inference and preserve metadata through #138 finalizers.
8. Propagate through LassoCV/ElasticNetCV.
9. Close cache/formula/weight/direct-fit/CV invariance tests.
10. Add independent reference and old/new simulation validator.
11. Update runtime docs/changelog.
12. Run local-minimal then local-full relevant suites.
13. Run physical GPU acceptance when available.
14. Fresh independent implementation review; fix CRITICAL/HIGH and relevant MEDIUM; rerun affected gates; repeat until clean or a user decision is required.

## 16. Acceptance criteria

- [ ] public `nodewise_alpha=None|positive real scalar` on all in-scope surfaces;
- [ ] intentional default migration/no-legacy-mode documented and evidenced;
- [ ] explicit value reaches every actual p>=2 node-wise solve;
- [ ] auto precision tuning never depends on `y`;
- [ ] weighted effective-n is scale-invariant and zero-weight-row invariant;
- [ ] standardized design + correct `D^{-1}Theta_ZD^{-1}` transform;
- [ ] paper-style tau normalizer and KKT-derived consistency check;
- [ ] full independent KKT gate before publication;
- [ ] p=1 shared analytic precision contract;
- [ ] standardized Gram and back-transformed precision must be finite before debiasing/reporting;
- [ ] cache keys actual node-wise numerical settings including stopping policy;
- [ ] GPU Lipschitz uses standardized Gram;
- [ ] direct penalized coefficients and CV selection unaffected by nodewise alpha;
- [ ] all-one/global-weight-scale identities preserved;
- [ ] NumPy/CuPy/Torch parity with no explicit-device fallback;
- [ ] marginal/simultaneous inference share one precision contract and atomic failure semantics;
- [ ] node-wise provenance survives finalizers and all reset paths;
- [ ] get_params/set_params/clone/formula/CV regressions pass;
- [ ] source/docs no longer misattribute response-scale tuning;
- [ ] exact-head local blocking suites pass;
- [ ] physical GPU evidence recorded or explicit `PARTIAL_REMOTE_PENDING`;
- [ ] fresh independent implementation review has no unresolved CRITICAL/HIGH.

## 17. Non-goals

- per-coordinate vector node-wise alpha in v1;
- node-wise CV / square-root Lasso as auto selector;
- public node-wise tolerance/max-iter controls;
- main-model CV redesign;
- full ElasticNetCV inference-selector redesign;
- unrelated cache-policy redesign;
- private legacy cosmetic parity;
- performance claims.

## 18. Plan review log

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

Closed: weight-aware response-independent effective-n; zero-weight invariance; shared p=1 analytic contract; paper-style tau normalizer with cross-product consistency check.

### Round 4

Plan commit: `236fcb7fbc4e5a51f3e829068c22ae80268ccd14`  
Artifact: `dev/reviews/nodewise-alpha-plan-review-round-4.md`  
Verdict: **PLAN CHANGES REQUIRED (no new HIGH findings)**

Closed: intentional statistical-default migration/no-legacy decision; explicit underlying stopping policy; atomic publication through requested simultaneous calibration; requested-value metadata.

### Round 5

Plan commit: `b09408f0b44e9013d7fb035942bd6148cac9ee3c`  
Artifact: `dev/reviews/nodewise-alpha-plan-review-round-5.md`  
Verdict: **PLAN REVIEW CLEAN**

No CRITICAL/HIGH/actionable MEDIUM plan finding remained. The final status/log edit added the implementation-ready marker and made the non-finite precision gate explicit; a final freshness review is required on this exact plan revision before implementation starts.