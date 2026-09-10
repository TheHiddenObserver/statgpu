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
- standardization-contract version.

Changing only main-model `tol` must not change precision-cache identity. Changing nodewise alpha must.

Auto and explicit equal resolved values may reuse the same numerical entry; metadata still reports the current request source.

## 12. CV propagation

### 12.1 LassoCV

Add `nodewise_alpha=None` as final-refit inference configuration only.

- not part of the main alpha grid or selection cache;
- propagate unchanged to final `Lasso`;
- changing it must not alter selected `alpha_`, `mse_path_`, penalized `coef_`/`intercept_`;
- outer `nodewise_alpha_` mirrors final estimator after successful node-wise inference;
- reset before every CV fit.

### 12.2 ElasticNetCV

Propagate analogously.

The current hard-coded final `inference_method="debiased"` is an adjacent API inconsistency but not required to provide nodewise-alpha control. Do not silently redesign that API in this task unless implementation review proves it necessary.

### 12.3 PenalizedGLM_CV

Current public generic CV does not expose the affected final-refit inference toggle, so no propagation is planned unless implementation inspection disproves that assumption.

## 13. Tests / evidence

### 13.1 Public API

For every changed public surface:

- omitted `None`;
- positive Python/NumPy scalar;
- invalid bool/complex/non-scalar/NaN/inf/zero/negative;
- signature/get_params/set_params/clone;
- fitted-state invalidation;
- non-debiased path keeps `nodewise_alpha_ is None`;
- direct/CV prediction coefficients invariant when only nodewise alpha changes.

### 13.2 Statistical invariants

- precision construction independent of `y` scaling;
- positive feature-scale equivariance;
- auto/explicit resolved-alpha equivalence;
- all-one weight identity;
- global weight-scale identity;
- zero-weight-row identity;
- degenerate scale/tau/non-finite precision fail closed;
- forced bad node-wise solve fails KKT gate;
- simultaneous reuse of same precision/alpha;
- p=1 analytic parity.

### 13.3 Backend

NumPy/CuPy/Torch must agree on the statistical contract, explicit alpha propagation, resolved auto alpha, KKT validity, precision/debiased reports, concrete device provenance, and no hidden CPU fallback.

### 13.4 CV/formula

- LassoCV/ElasticNetCV nodewise alpha affects final inference only;
- main selected hyperparameters/scores unchanged;
- array/formula parity after model-matrix construction.

### 13.5 Reference / simulation / physical GPU

- independent small NumPy reference at fixed explicit alpha;
- KKT/`M Sigma` diagnostic checks;
- deterministic-seed old-vs-new simulation report for coverage/interval-length regression detection, not a theorem claim;
- physical CuPy and Torch validator cases for auto, explicit override, auto/explicit equivalence, response-scale invariance, p=1, weights, simultaneous reuse, and provenance.

## 14. Documentation / migration

Update relevant EN/CN runtime docs and changelog to state:

- `nodewise_alpha` is separate from main `alpha`;
- explicit user value wins;
- omitted value uses the standardized design-side auto rule;
- the exact constant/effective-n convention is a statgpu default, not a unique theorem choice;
- old response-dependent internal behavior is intentionally replaced, with no legacy public mode;
- p=1 uses analytic precision and does not consume nodewise alpha;
- CV control applies only to final-refit inference.

Remove the old source attribution tying main-response `sigma_hat` scaling to van de Geer et al. (2014).

After runtime behavior is stable and merged, synchronize the still-Draft learner-first #132/#134 stack to the new master; do not manually maintain two contradictory runtime truths.

## 15. Implementation/review order

1. shared helper + public API validation/state;
2. NumPy precision construction + p=1;
3. CuPy/Torch batched standardized-Gram path;
4. #138 weighted/intercept/finalizer integration;
5. CV propagation;
6. deterministic API/statistical/formula tests;
7. physical validator extension and simulation evidence;
8. EN/CN docs/changelog;
9. fresh independent code-review/fix loop on exact implementation head;
10. re-resolve branch/master before completion verdict.

Completion requires no unresolved CRITICAL/HIGH review finding. Missing physical CUDA evidence after local/hosted closure is reported as `PARTIAL_REMOTE_PENDING`, not hidden or redefined away.
