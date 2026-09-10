# Node-wise Lasso tuning contract — implementation plan

Status: DRAFT FOR REVIEW

Target baseline:

- repository: `TheHiddenObserver/statgpu`
- branch: `fix/nodewise-alpha-inference-contract`
- base `master`: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
- affected capability: squared-error L1 / Elastic Net debiased inference and the node-wise Lasso precision construction used by marginal and simultaneous inference

This plan intentionally lives outside documentation PR #134. Runtime/API changes must be implemented and reviewed independently, then synchronized into the still-Draft documentation stack after the runtime contract is stable.

## 1. Problem statement

The current debiased-inference implementation has two coupled problems.

1. The node-wise Lasso penalty is an important statistical tuning parameter but is internal-only. Users cannot inspect or override it.
2. The current automatic rule uses the main response-model residual scale,

   $$
   \lambda_{\mathrm{nw}}
   =
   \hat\sigma_y\sqrt{\frac{2\log(\max(p,2))}{n}},
   $$

   even though the node-wise regressions estimate a design-side object, approximately `Sigma_X^{-1}`. Rescaling only `y` can therefore change the estimated precision matrix `M` while `X` is unchanged.

A historical source comment also attributes the response-scale multiplier to van de Geer et al. (2014). The literature supports node-wise penalties of order `sqrt(log(p)/n)` but does not justify multiplying the node-wise penalty by the main response residual scale as the unique construction.

## 2. Impact classification

Active axes under the repository development/review contract:

- public API / sklearn compatibility;
- inference;
- NumPy / CuPy / Torch backend parity;
- sample-weight and intercept semantics;
- CV final-refit propagation for public CV estimators that expose the affected debiased inference;
- formula/model-matrix parity because the changed inference is formula-facing;
- docs / changelog / validation evidence.

Not active by default:

- main Lasso / Elastic Net optimization objective;
- main-model alpha selection or CV scoring;
- unrelated GLM, survival, nonparametric, and feature-selection paths;
- performance claims. Performance regression should still be sanity-checked because GPU node-wise batching is touched, but no speedup claim is part of this task.

## 3. Public API contract

### 3.1 New constructor parameter

Add

```python
nodewise_alpha: Optional[float] = None
```

to the maintained public surfaces that can execute squared-error node-wise debiased inference:

- `PenalizedGeneralizedLinearModel` / `PenalizedLinearRegression` generic path;
- `Lasso`;
- `ElasticNet`;
- `LassoCV` final-refit inference configuration;
- `ElasticNetCV` final-refit inference configuration.

The constructor stores the requested value without lossy coercion so sklearn clone/reconstruction semantics remain stable.

Interpretation:

- `nodewise_alpha=None`: use the automatic design-side rule in Section 4;
- finite scalar `nodewise_alpha > 0`: use that value exactly for every node-wise Lasso regression after the canonical node-wise design standardization in Section 4;
- booleans, non-scalars, NaN, infinity, zero, and negative values are rejected with a clear `ValueError` when the parameter is validated.

The first public version is scalar-only. Per-coordinate `p`-vector penalties are deliberately out of scope because the maintained CuPy/Torch batched Gram solvers currently share one node-wise alpha. Vector support can be added later without changing scalar semantics.

### 3.2 Separation from the main model alpha

`alpha` and `nodewise_alpha` are independent controls:

- `alpha` controls the penalized prediction/selection estimator;
- `nodewise_alpha` controls only the node-wise regressions used to estimate the approximate precision matrix for `inference_method="debiased"`.

`nodewise_alpha` must not alter LassoCV / ElasticNetCV candidate grids, fold scoring, selected main-model hyperparameters, or the penalized final-fit coefficients.

### 3.3 Fitted result/provenance

When node-wise debiased inference actually runs, publish

```python
model.nodewise_alpha_
```

as the resolved scalar used by the node-wise solver.

Also record in `_inference_result.metadata`:

```text
nodewise_alpha
nodewise_alpha_source = "user" | "auto"
nodewise_alpha_rule = "standardized_universal_v1" | "user"
nodewise_design_standardized = true
nodewise_n_samples
```

`nodewise_alpha_` is reset before each fit and remains `None` when the fitted inference path does not use node-wise Lasso.

## 4. Statistical contract for the automatic rule

### 4.1 Canonical working design

The node-wise precision construction must use the same centered / weighted average-loss design convention as the maintained debiased-inference path, not reconstruct a second incompatible weighting convention.

For raw feature rows `x_i` and optional analytic weights `w_i`, define the working design `X_w` as follows.

With an effective intercept and no weights:

$$
X_w = X - \bar X.
$$

Without an effective intercept and no weights:

$$
X_w = X.
$$

With analytic weights, let `W = sum_i w_i` and use the existing statgpu average-loss normalization:

$$
X_{w,i}
=
\sqrt{\frac{n w_i}{W}}
\left(x_i-\bar x_w\right)
$$

when an intercept is present, and

$$
X_{w,i}
=
\sqrt{\frac{n w_i}{W}}x_i
$$

otherwise.

This preserves the existing global-weight-scale invariance: replacing `w` by `c w` does not change `X_w`.

### 4.2 Standardize the node-wise design

Define design-side column scales

$$
d_j
=
\sqrt{\frac{1}{n}\sum_{i=1}^n X_{w,ij}^2},
\qquad
D=\operatorname{diag}(d_1,\ldots,d_p),
$$

and standardized node-wise design

$$
Z=X_wD^{-1}.
$$

Every `d_j` must be finite and strictly positive. A zero/degenerate working-design column makes the corresponding precision direction unidentified; debiased inference must fail closed with a clear inference error rather than silently setting an identity row in `M`. Estimation with `compute_inference=False` remains unaffected.

### 4.3 Automatic penalty

For `nodewise_alpha=None`, resolve

$$
\boxed{
\lambda_{\mathrm{nw}}^{\mathrm{auto}}
=
\sqrt{\frac{2\log(\max(p,2))}{n}}
}
$$

on the standardized design `Z`.

This is a concrete statgpu default motivated by the standard node-wise order `sqrt(log(p)/n)`. It is not documented as the unique theorem-mandated choice. More expensive node-wise CV or square-root-Lasso tuning is not introduced in this repair.

A user-supplied scalar `nodewise_alpha` is interpreted on this same standardized `Z` scale, so setting it to a previously reported `nodewise_alpha_` exactly reproduces the node-wise penalty component of the automatic run.

### 4.4 Node-wise construction and back-transformation

For each coordinate `j`, solve

$$
\hat\gamma_j
=
\arg\min_\gamma
\left\{
\frac{1}{2n}\|Z_j-Z_{-j}\gamma\|_2^2
+
\lambda_{\mathrm{nw}}\|\gamma\|_1
\right\}.
$$

Let

$$
r_j=Z_j-Z_{-j}\hat\gamma_j,
\qquad
C_j^{(Z)}=\frac{1}{n}Z_j^\top r_j.
$$

Construct the standardized approximate-precision row

$$
\hat\Theta_{Z,j}^\top
=
\frac{(e_j-\tilde\gamma_j)^\top}{C_j^{(Z)}}.
$$

Then transform to the original working-design scale:

$$
\boxed{
M_X=D^{-1}\hat\Theta_ZD^{-1}.
}
$$

The existing debiasing correction and variance calculation continue to use the original working design / covariance together with `M_X`; the public prediction coefficients are unchanged.

The implementation must fail closed if a node-wise normalizer `C_j^{(Z)}` is non-finite or numerically degenerate. Do not silently publish a plausible identity-row substitute.

## 5. Backend implementation plan

### 5.1 Shared helpers

Introduce one maintained helper layer for:

- raw parameter validation without constructor coercion;
- canonical working-design scale calculation;
- `nodewise_alpha` resolution;
- scale/back-transform bookkeeping;
- publication of `nodewise_alpha_` and metadata.

Prefer a focused helper module under `statgpu/linear_model/penalized/` rather than adding another runtime monkeypatch contract. Existing node-wise cache and GPU Gram/FISTA kernels can remain where they are unless a small extraction materially reduces duplication.

### 5.2 NumPy

Correctness-first implementation:

1. obtain the canonical `X_w`;
2. compute `d` and `Z`;
3. resolve `nodewise_alpha`;
4. reuse the existing per-coordinate L1 solver on `Z`;
5. construct `Theta_Z`;
6. back-transform to `M_X`;
7. run the existing debiasing/reporting pipeline.

### 5.3 CuPy and Torch

Keep the node-wise numerical transaction on the already selected concrete GPU device.

Avoid materializing a second full standardized design if unnecessary. The existing Gram-based batched solver can use

$$
\Sigma_Z=D^{-1}\Sigma_{X_w}D^{-1}
$$

directly. Construct standardized batched Gram / cross-product blocks from `Sigma_Z`, solve with the single resolved scalar `nodewise_alpha`, build `Theta_Z`, then back-transform on-device.

No new CPU fallback is allowed for an explicit CUDA/Torch inference request.

### 5.4 Cache semantics

Node-wise precision-cache identity must depend on enough information to prevent reuse across different precision problems, including at least:

- design identity under the existing cache contract;
- resolved `nodewise_alpha`;
- solver tolerance;
- standardization contract/version if not already implied by the design hash.

Changing `nodewise_alpha` must force a cache miss. `None` and an explicit numeric value equal to the resolved automatic value should be numerically equivalent; cache reuse between them is allowed only if the cache key is based on the resolved value and identical design contract.

## 6. CV propagation

### 6.1 LassoCV

Add `nodewise_alpha=None` to `LassoCV` as final-refit inference configuration.

- It is not part of the main alpha grid or selection cache key.
- It is propagated unchanged to the final `Lasso(...)` refit.
- `estimator_.nodewise_alpha_` and the outer CV inference result must agree when inference runs.

### 6.2 ElasticNetCV

Add `nodewise_alpha=None` and propagate it to the final `ElasticNet(...)` refit.

The existing `ElasticNetCV` hard-coded `inference_method="debiased"` is an adjacent API inconsistency. Do not silently broaden this repair into a complete ElasticNetCV inference-API redesign unless implementation review shows it is necessary for the nodewise contract. Record it as a follow-up issue/plan item if it remains after this change.

### 6.3 PenalizedGLM_CV

No change is planned unless source inspection shows a public squared-error CV route that exposes node-wise debiased final-refit inference. At the current baseline, the generic `PenalizedGLM_CV` public contract does not expose this inference toggle, so it is not part of this parameter-propagation surface.

## 7. Validation plan

### 7.1 Public API / compatibility

Add deterministic tests for every maintained public surface that gains the parameter:

- omission yields `nodewise_alpha is None`;
- explicit finite positive scalar is preserved by constructor introspection;
- NaN, infinity, zero, negative, bool, and non-scalar inputs fail clearly;
- `get_params(deep=False)` exposes the parameter;
- `set_params(nodewise_alpha=...)` is accepted and invalidates stale fitted state according to the shared estimator contract;
- sklearn `clone()` preserves the requested value without warnings or coercion artifacts;
- inference paths not using node-wise Lasso leave `nodewise_alpha_ is None` and do not accidentally change fit numerics.

### 7.2 Statistical invariants

Add focused tests for:

1. **Response-scale invariance of the precision construction**: with fixed `X`, the automatic resolved `nodewise_alpha_` and `M_X` do not change when only `y` is multiplied by a positive constant. Test the precision helper directly so main-model alpha rescaling does not confound the invariant.
2. **Feature-scale equivariance**: rescaling columns of `X` by diagonal `A` gives the expected precision transformation `M_{XA} = A^{-1} M_X A^{-1}` within tolerance.
3. **Auto/explicit equivalence**: run `None`, capture `nodewise_alpha_`, rerun with that explicit scalar, and require equivalent `M`, debiased params, SE, p-values, and intervals.
4. **Global analytic-weight-scale invariance**: `w` and `c*w` produce the same working-design precision construction and inference.
5. **All-one weight identity**: unweighted and all-one-weight results agree.
6. **Degenerate design fail-closed**: zero/non-finite scale or degenerate `C_j` aborts debiased inference and leaves no stale inference result.
7. **Simultaneous inference reuse**: max-|Z| uses the same resolved `M` / `nodewise_alpha_`; it must not recompute with a different hidden rule.

### 7.3 Backend parity

For NumPy, CuPy, and Torch:

- explicit user `nodewise_alpha` reaches the actual node-wise solver;
- automatic resolution agrees across backends;
- `M`, debiased params, SE, statistics, p-values, marginal CI, and simultaneous CI remain within maintained tolerances;
- actual numerical backend/device provenance remains correct;
- no silent host fallback is introduced.

### 7.4 CV and formula

- `LassoCV(nodewise_alpha=x)` propagates `x` only to final-refit inference, not candidate selection.
- `ElasticNetCV(nodewise_alpha=x)` behaves analogously.
- CV-selected main `alpha` / `l1_ratio` are unchanged when only `nodewise_alpha` changes.
- array and formula routes produce the same resolved node-wise contract after model-matrix construction, including intercept handling.

### 7.5 Analytic/reference checks

Use the strongest deterministic checks before simulation:

- compare the standardized/back-transformed `M` to a small independent NumPy reference implementation at fixed explicit `nodewise_alpha`;
- verify `M Sigma` approximation/KKT residuals under controlled designs;
- use the literature only to justify order/parameterization, not to claim exact numerical parity with packages that select node-wise lambda differently.

An optional non-CI comparison against R `hdi` may be added by forcing an aligned node-wise lambda where practical. It is supplementary, not a substitute for the analytic invariants above.

### 7.6 Coverage/evidence

Because the default tuning rule changes, add a deterministic-seed simulation validator (not a brittle per-commit unit-test assertion) covering representative sparse Gaussian designs and reporting marginal coverage/interval length for the old and new defaults. The goal is to detect an obvious regression, not to claim finite-sample nominal coverage from one simulation grid.

Add a physical CUDA validator or extend a suitable maintained validator so both CuPy and Torch physical-device runs cover:

- automatic node-wise alpha;
- explicit override;
- auto/explicit equivalence;
- response-scale invariance of the precision construction;
- simultaneous inference reuse/provenance.

Physical GPU evidence must record exact SHA, device, environment, resolved alpha/source, and pass/fail metrics.

## 8. Documentation and migration

Runtime branch docs must state:

- `nodewise_alpha` is the node-wise Lasso tuning parameter, separate from main `alpha`;
- omitted value invokes the standardized design-side automatic rule;
- explicit value is interpreted on the standardized node-wise design scale;
- the automatic rule is a statgpu default motivated by `sqrt(log p / n)` theory, not a theorem-mandated unique choice;
- changing `y` units does not change the automatic precision construction;
- scalar-only support in this release;
- CV parameters affect final-refit inference only.

Remove source/docs wording that attributes multiplication by the main response residual `sigma_hat` to van de Geer et al. (2014).

Update EN/CN maintained docs and changelog on the runtime branch as appropriate. Because PR #134 contains newer learner-first pages, after the runtime contract is stable/merged, synchronize the new master into #132/#134 and reconcile those learner pages rather than merging #134 first or duplicating incompatible prose.

## 9. Implementation order

1. Add focused statistical/API tests that expose current response-scale dependence and missing public override; confirm they fail for the intended reason.
2. Add shared validation/resolution/standardization helpers.
3. Plumb `nodewise_alpha` through generic penalized linear, Lasso, ElasticNet, and fitted-state metadata.
4. Implement NumPy standardized-design node-wise construction and back-transform.
5. Implement CuPy/Torch equivalent Gram-space construction on concrete devices.
6. Propagate through LassoCV and ElasticNetCV final refits.
7. Close formula, weight, cache, simultaneous-inference, and fail-closed edge cases.
8. Update docs/changelog and add validation artifact schema.
9. Run local-minimal then local-full relevant suites.
10. Run physical GPU acceptance when available.
11. Invoke a fresh independent code-review pass on the resulting exact head; fix CRITICAL/HIGH and relevant MEDIUM findings, re-run affected validation, then repeat fresh review until clean or an explicit user decision is required.

## 10. Acceptance criteria

The repair is not complete until all applicable items hold:

- [ ] `nodewise_alpha=None|float` is a documented public contract on every maintained direct/CV surface that uses node-wise debiased inference.
- [ ] explicit user value takes precedence everywhere and is provenance-recorded.
- [ ] automatic `nodewise_alpha_` depends only on the canonical design-side working problem, not on response scale.
- [ ] node-wise construction is feature-scale equivariant through standardization and back-transformation.
- [ ] weighted global-scale and all-one-weight identities are preserved.
- [ ] NumPy/CuPy/Torch implementations use the same statistical contract with no explicit-device CPU fallback.
- [ ] LassoCV/ElasticNetCV propagate the setting only to final-refit inference and do not change main hyperparameter selection.
- [ ] marginal and simultaneous inference reuse one resolved node-wise precision contract.
- [ ] cache keys cannot cross-contaminate different resolved node-wise penalties/contracts.
- [ ] `get_params`, `set_params`, clone, refit, fitted-state invalidation, and invalid-input behavior are tested.
- [ ] formula parity and degenerate-design fail-closed behavior are tested.
- [ ] `_inference_result.metadata` and `nodewise_alpha_` report the actual resolved value/source.
- [ ] EN/CN docs and changelog no longer misattribute the old response-scale rule to the literature.
- [ ] local blocking tests pass on the exact implementation head.
- [ ] physical CuPy/Torch evidence is recorded, or completion is explicitly reported as `PARTIAL_REMOTE_PENDING` if that is the only remaining evidence.
- [ ] fresh independent review has no unresolved CRITICAL/HIGH finding.

## 11. Explicit non-goals

- vector-valued per-coordinate node-wise penalties in the first public version;
- adding node-wise CV as the default selector;
- changing the main Lasso/ElasticNet objective or main alpha CV grids;
- redesigning all ElasticNetCV inference controls unless required by this contract;
- modifying private legacy implementations solely for cosmetic parity when no maintained public path routes through them;
- claiming performance improvement.

## 12. Plan review log

The plan must itself go through an independent review/fix loop before implementation starts. Record each review round here with exact plan commit, findings, fixes, and final plan verdict.
