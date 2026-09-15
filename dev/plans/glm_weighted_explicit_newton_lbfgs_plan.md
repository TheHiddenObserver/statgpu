# Weighted explicit Newton/L-BFGS GLM repair plan

Status: PLAN REVIEW CLEAN / IMPLEMENTATION OPEN
Tracked issue: #150
Base: `master` after PR #147 merge (`658a23add3ea54dce95c51eb4bba80b91cb0f1b3`)
Working branch: `fix/glm-weighted-explicit-solver-guard`
Target release: 0.2.6 (unreleased)

## 1. Goal

Repair the ordinary `GeneralizedLinearModel` public fit contract so a genuine non-uniform analytic `sample_weight` is not rejected merely because the caller explicitly requested `solver="newton"` or `solver="lbfgs"`.

The two solver rows have different starting points and are reviewed separately:

- **Newton** already has a maintained backend-native non-uniform weighted numerical implementation. The ordinary GLM wrapper blocks that capability before solver entry. This is existing-capability/public-contract reconciliation.
- **L-BFGS** still rejects genuine non-uniform weights in the shared solver and evaluates value/gradient/line-search trials without weights. This requires a real shared weighted-L-BFGS numerical capability plus consumer/backend closure.

Explicit solver authority is invariant: weights must never cause an explicit Newton/L-BFGS request to be silently rewritten to IRLS, FISTA, another backend, or CPU.

## 2. Change classification and active axes

Change types:

1. existing capability reconciliation / public contract repair for weighted explicit Newton;
2. materially changed shared numerical capability for non-uniform weighted L-BFGS on the maintained GLM loss interface;
3. shared solver consumer reconciliation because penalized GLM already calls the same `lbfgs_solver(..., sample_weight=...)` entry point.

Active axes:

- public fit/solver contract;
- analytic-weight loss/objective semantics;
- solver/convergence/line-search behavior;
- NumPy/CuPy/Torch backend and concrete-device ownership;
- ordinary GLM wrappers and formula/data path;
- shared penalized-GLM / `PenalizedGLM_CV` consumers of `lbfgs_solver`;
- inference/diagnostic integration for already-supported ordinary-GLM rows;
- fit provenance and refit-state semantics;
- direct solver documentation plus model/solver docs and exact-source evidence.

Intentionally unchanged:

- ordinary `solver="auto"` dispatch;
- IRLS and FISTA numerical algorithms;
- the pre-existing meaning of `C` on each ordinary explicit smooth-solver path;
- specialized standalone `LogisticRegression` public API (separate `BaseEstimator`, not a `GeneralizedLinearModel` subclass);
- `OrderedGeneralizedLinearModel`, which continues to reject `sample_weight` explicitly;
- covariance estimators/inferential targets;
- weighted bootstrap semantics (#148) and non-Gaussian parametric bootstrap (#149).

If reconnaissance exposes an unrelated inconsistency in explicit smooth-solver `C` handling, record it separately rather than silently changing it in #150.

## 3. Baseline reconnaissance

### 3.1 Ordinary GLM wrapper

`GeneralizedLinearModel.fit()` already:

- validates `sample_weight`;
- aligns formula-side weights after retained-row/missing-row filtering;
- converts weights once to the selected execution backend;
- resolves explicit solver and routes `newton`/`lbfgs` to `_fit_smooth_solver()`;
- retains fit weights for loglikelihood/AIC/BIC and supported inference.

The stale blocking behavior is inside `_fit_smooth_solver()`:

```python
if sample_weight is not None:
    raise ValueError(...)
```

and downstream calls currently omit `sample_weight` for both smooth solvers.

### 3.2 Newton numerical capability

`newton_solver` already supports non-uniform analytic weights on NumPy/CuPy/Torch. Its maintained contract:

- validates/alines weights to execution backend/device;
- preserves the historical floating-point `allclose` uniform-weight rule by treating such vectors as the unweighted path;
- uses the same active non-uniform vector in value, gradient, Hessian/fused gradient+Hessian, Armijo current objective, and every Armijo trial objective;
- uses normalized average-loss scaling.

Existing tests already cover row-replication equivalence, global weight-rescaling invariance, uniform-weight compatibility, invalid weights, and NumPy/Torch-CPU parity.

### 3.3 L-BFGS numerical gap

`lbfgs_solver` currently:

- accepts `sample_weight` but calls `_validate_uniform_sample_weight`;
- documents only uniform weights as valid;
- computes initial gradient, current line-search objective, candidate objective, and updated gradient without passing the weight vector.

Deleting the ordinary wrapper guard alone would therefore only move non-uniform L-BFGS failure down one layer.

### 3.4 GLM weighted primitives

The maintained GLM loss stack provides the primitives needed by smooth solvers:

- `LossBase.value/gradient/fused_value_and_gradient(..., sample_weight=...)` use `sum_i w_i contribution_i / sum_i w_i`;
- `GLMLoss.fused_value_and_gradient` dispatches weighted calls through `_weighted_loss_and_grad`;
- maintained squared-error, Logistic/Binomial, Poisson, Gamma, Inverse Gaussian, Negative Binomial, and Tweedie losses expose weighted Hessian support for Newton.

This substrate is not itself an acceptance claim. Every public family × solver × backend row is validated before being documented as supported.

### 3.5 Explicit smooth-solver `C` characterization

Current ordinary `_fit_smooth_solver()` calls the smooth solver with `penalty=None`; this issue does **not** reinterpret `C` or introduce a new penalty into explicit Newton/L-BFGS.

Before numerical comparison tests are written:

- characterize the existing unweighted explicit Newton/L-BFGS behavior under multiple `C` values;
- freeze that existing behavior with regression tests;
- use references aligned to the current explicit smooth-solver objective (unpenalized where that is the present behavior).

Any desire to make explicit smooth solvers consume `C` consistently with another solver is a separate public-contract issue.

## 4. Consumer graph and frozen scope

### 4.1 Direct ordinary-GLM target consumers

In scope:

- `GeneralizedLinearModel(family=...)`;
- `PoissonRegression`;
- `GammaRegression`;
- `InverseGaussianRegression`;
- `NegativeBinomialRegression`;
- `TweedieRegression`;
- any maintained thin wrapper that directly inherits `GeneralizedLinearModel` without replacing this fit path.

Generic `GeneralizedLinearModel(family="binomial")` is in scope. The specialized standalone `LogisticRegression(BaseEstimator)` is out of scope and receives preservation tests only if shared changes can reach it indirectly.

### 4.2 Explicitly preserved/unsupported consumers

- `OrderedGeneralizedLinearModel` and ordered-logit/probit remain weighted-unsupported;
- ordinary IRLS/FISTA behavior is unchanged;
- ordinary `solver="auto"` behavior is unchanged.

### 4.3 GLM-only non-uniform L-BFGS capability boundary

`lbfgs_solver` is a generic shared function and is documented for generic losses such as Huber/Quantile. #150 must not accidentally make every `LossBase` a new weighted-L-BFGS public surface.

Freeze this issue's genuinely non-uniform L-BFGS capability to the **GLM loss interface**.

Architecture of the capability check:

- `LossBase` owns a private solver-capability marker such as `_supports_nonuniform_lbfgs_weight = False`;
- `GLMLoss` sets/inherits that marker as `True` because its public subclass contract derives weighted fused value/gradient from per-sample formulas;
- `lbfgs_solver` inspects the generic marker and does **not** import `glm_core`, avoiding a solver→GLM reverse dependency/cycle;
- conforming custom `GLMLoss` subclasses inherit the GLM interface capability; if they override the fused interface, they remain responsible for honoring the inherited `sample_weight` contract;
- non-GLM losses keep the current uniform-only L-BFGS behavior unless separately reviewed.

No hard-coded built-in family list belongs in the L-BFGS iteration loop.

### 4.4 Shared penalized GLM consumers — intentionally included

Existing penalized GLM code already passes `sample_weight` to both smooth solvers. Therefore the new GLM-weighted L-BFGS capability intentionally closes existing smooth-GLM rows that currently fail only because the shared L-BFGS uniform-weight gate fires.

Part of #150 closure:

- explicit `PenalizedGeneralizedLinearModel(..., solver="lbfgs")` for smooth GLM + smooth penalty rows already otherwise weight-capable;
- typed penalized smooth-GLM wrappers using the same path;
- existing weighted `PenalizedGLM_CV` smooth-L2 rows whose maintained CV solver table already resolves to L-BFGS, notably Negative Binomial and Gamma/Inverse-Gaussian candidate/final-refit paths where their existing public weight contract permits weights.

No new penalty/loss/CV dispatch row is introduced. Unsupported independent weight contracts remain fail-closed at their consumer boundary.

## 5. Desired objective and compatibility contract

For supported analytic-weight rows, Newton and L-BFGS optimize exactly

```text
L(beta) = sum_i w_i * ell_i(beta) / sum_i w_i + P(beta)
```

where `P=0` for ordinary explicit smooth-solver fits under the current contract and is the already-existing smooth penalty for shared penalized consumers.

Required identities:

1. multiplying all active non-uniform weights by a positive scalar does not change the optimum beyond maintained tolerance;
2. positive uniform weights reproduce the historical unweighted route;
3. integer weights agree with literal row replication for aligned reference rows;
4. rows with weight exactly zero are equivalent to removing those rows from the objective, provided the remaining weight sum is positive;
5. every value/gradient/Hessian/quasi-Newton/line-search evaluation represents the same weighted objective;
6. explicit solver identity remains exactly the user request.

The historical Newton/L-BFGS floating-point uniformity rule (`allclose(values, values[0])`) remains unchanged.

## 6. Fit provenance and refit-state contract

Ordinary GLM currently does not expose the same stable fit-recorded provenance surface used by penalized GLM. #150 needs machine-auditable proof that explicit weighted solver requests stayed on the requested solver/backend/device.

Add/standardize these private fitted provenance fields on successful ordinary GLM fit:

- `_selected_solver` — actual explicit/auto-resolved solver used;
- `_selected_backend_name` — `numpy`, `cupy`, or `torch`;
- `_selected_backend_device` — `cpu` or concrete `cuda:k` / Torch device string.

Publication timing is part of the contract:

- resolve solver/backend/device into local/staged values before numerical work;
- publish the new provenance only at the same success point at which the fit is considered completed under the existing ordinary-GLM refit semantics;
- do not overwrite a previous successful fit's provenance merely because a new attempted solver was selected before a failed fit;
- if characterization shows current ordinary GLM invalidates fitted state on a failed refit, provenance must be invalidated in the same transaction instead of preserved.

Before broader transaction changes, characterize a previously fitted ordinary GLM under: input validation failure, invalid weight failure, and solver failure. Preserve established semantics unless repository transaction policy already requires invalidation. #150 must not silently redefine unrelated refit behavior.

Adding provenance must not change existing IRLS/FISTA/auto numerical behavior. Include preservation checks showing their successful provenance is accurate as well.

## 7. Planned implementation

### 7.1 Shared analytic-weight preparation

Avoid diverging Newton/L-BFGS backend alignment and uniformity logic.

Preferred implementation:

- extract/generalize Newton's current preparation into one private solver utility such as `_prepare_analytic_sample_weight(sample_weight, n_samples, backend, ref_arr)`;
- Newton calls it with no observable behavior change;
- GLM-capable L-BFGS calls the same utility;
- preserve a thin internal alias if existing static tests/imports require the Newton-private name.

Before/after characterization proves Newton behavior preservation.

### 7.2 Weighted GLM L-BFGS

For losses admitted by `_supports_nonuniform_lbfgs_weight`, thread the prepared active weight vector through every loss evaluation:

- initial fused value/gradient;
- current objective for line search;
- every candidate objective in backtracking;
- updated gradient after the accepted candidate.

Penalty value/gradient handling, two-loop recursion, curvature-history update, convergence rules, and line-search constants remain unchanged.

For losses without the capability marker, genuinely non-uniform weights retain the existing fail-closed behavior; uniform/almost-uniform compatibility remains unweighted as today.

No full weight/data host conversion is allowed on explicit CuPy/Torch. Established scalar synchronization for convergence/line-search booleans is allowed.

### 7.3 Ordinary `GeneralizedLinearModel`

Replace the blanket `_fit_smooth_solver()` rejection with capability-aware dispatch:

- pass already validated/backend-aligned weights to the requested smooth solver;
- stage and publish fit-recorded solver/backend/device provenance according to §6;
- do not change `solver="auto"`;
- do not silently substitute a different solver;
- preserve intercept/result layout and current explicit-solver `C` behavior;
- let precise lower-level loss/solver capability errors propagate before publishing a successful fit.

### 7.4 Shared penalized/CV closure

Once GLM L-BFGS supports non-uniform weights, explicitly validate the included penalized rows from §4.4. Do not add an artificial consumer guard merely to preserve the previous shared-solver limitation.

If a listed row fails because of a genuine independent family/penalty weight limitation, retain that exact consumer-level fail-closed contract and document the reason; do not restore a generic L-BFGS non-uniform rejection.

## 8. Final support matrix to prove

For ordinary `GeneralizedLinearModel`, target these rows on NumPy/CuPy/Torch. A row is removed only by a reviewed plan amendment if characterization proves a genuine family/solver numerical limitation.

| Family/loss | Newton + non-uniform weights | L-BFGS + non-uniform weights | Notes |
|---|---|---|---|
| Gaussian / squared error | target supported | target supported | current explicit smooth-solver objective preserved |
| Binomial / logistic via generic GLM | target supported | target supported | specialized standalone LogisticRegression unchanged |
| Poisson | target supported | target supported | canonical log-link path |
| Gamma, log link | target supported | target supported | validate separately |
| Gamma, inverse-power link if publicly constructible through this surface | target supported | target supported | separate domain/stability row |
| Inverse Gaussian | target supported | target supported | validate noncanonical stability |
| Negative Binomial | target supported | target supported | key shared CV L-BFGS consumer |
| Tweedie | target supported | target supported | power parameter fixed per model |
| Ordered models | unsupported | unsupported | existing contract preserved |

For non-GLM direct `lbfgs_solver` calls, genuinely non-uniform weights remain unsupported in #150.

## 9. Hosted validation plan

### 9.1 Shared solver-level tests

Add weighted L-BFGS regressions analogous to weighted Newton:

- integer-weight row replication on squared error plus at least Logistic and Poisson;
- global positive weight-rescaling invariance;
- uniform and historical almost-uniform floating weights reproduce unweighted path;
- zero-weight rows equal the result after dropping those rows for aligned reference cases;
- invalid shape/negative/non-finite/zero-total weights fail before iteration;
- NumPy vs Torch-CPU parity;
- instrumentation proving every GLM fused value/gradient call receives active weights, including line-search candidates;
- a conforming custom `GLMLoss` inherits the weighted-L-BFGS capability contract;
- non-GLM direct L-BFGS (including representative Huber/Quantile and Cox survival cases) with genuine non-uniform weights remains rejected;
- line-search failure remains fail-visible and never accepts an unverified trial.

Newton tests prove the shared helper extraction preserves all existing weighted/unweighted behavior.

### 9.2 Ordinary public GLM matrix

For explicit `newton` and `lbfgs` separately:

- non-uniform weighted fit reaches the requested solver rather than the old wrapper error;
- `_selected_solver/_selected_backend_name/_selected_backend_device` match execution;
- coefficients/intercept are finite and agree with aligned reference construction;
- positive global weight rescaling invariance;
- uniform-weight equivalence;
- zero-weight-row equivalence on representative families;
- `fit_intercept=True` and `False`;
- generic GLM plus every maintained typed wrapper in §4.1;
- all family/link rows in §8;
- formula/data path aligns side-array weights after dropped rows;
- explicit unsupported rows fail precisely with no solver substitution.

### 9.3 `C`, provenance, and refit-state characterization

Before accepting #150:

- freeze current unweighted explicit Newton/L-BFGS behavior under at least two `C` values to prove the weight change did not alter `C` semantics;
- characterize failed refit behavior on a previously fitted ordinary GLM;
- prove new provenance is published/staged consistently with that established behavior and never contradicts completed work;
- prove existing successful IRLS/FISTA/auto routes still return identical numerical results and now report truthful provenance only.

### 9.4 Inference/diagnostic preservation

For ordinary rows where `compute_inference=True` is already supported:

- fit and inference use the same active weights;
- inference `solver_used` agrees with `_selected_solver`;
- numerical backend/device metadata agrees with fit provenance where present;
- `_sample_weight_inf`, loglikelihood, AIC/BIC, bse/statistic/p-value/CI are coherent with weighted fit;
- covariance support matrix is unchanged.

Cover at least one Newton and one L-BFGS family with inference enabled and an independent aligned reference where feasible.

### 9.5 Shared penalized/CV regressions

Mandatory because the shared L-BFGS capability changes:

- explicit weighted penalized GLM L-BFGS on supported smooth rows;
- weighted `PenalizedGLM_CV` existing L-BFGS rows: Negative Binomial L2 and Gamma/Inverse-Gaussian L2 where public weight support already applies;
- candidate and selected-final-refit solver identity remains L-BFGS;
- penalty scaling/weight-rescaling invariance remains coherent;
- existing Newton/FISTA/IRLS dispatch is unchanged;
- non-GLM/unsupported penalty rows remain fail-closed.

## 10. Backend/device and physical CUDA closure

This is a shared numerical capability change. Final support claims require NumPy/CuPy/Torch closure.

Hosted/static tests prove routing, Torch-CPU parity, formula alignment, capability-marker behavior, public matrix, and no accidental consumer expansion where physical CUDA is unavailable.

Freeze `dev/benchmarks/validate_glm_weighted_explicit_solvers_gpu.py` (schema v1) or an equivalently scoped maintained exact-source validator before first acceptance run.

The physical matrix must include **every ordinary family/link row finally claimed supported in §8 for both explicit Newton and explicit L-BFGS on both CuPy and Torch CUDA**. Small deterministic datasets are acceptable; the goal is capability/provenance/parity, not performance.

Additionally include:

- at least one Torch-input→CuPy and one CuPy-input→Torch crossing for each solver if public explicit-device routing permits it;
- Negative Binomial and Gamma/Inverse-Gaussian weighted penalized/CV L-BFGS representative rows if those shared consumers are claimed supported;
- concrete `cuda:k` provenance and selected solver provenance;
- no CPU numerical fallback;
- coefficient/intercept parity against frozen NumPy references;
- at least one positive weight-rescaling invariant on each GPU backend for L-BFGS.

Validator schema, data seeds, family kwargs, tolerances, route matrix, and source fingerprint are frozen before the first acceptance run. Do not loosen tolerances after a failed run solely to get green status.

## 11. External/reference validation

Use aligned references only:

- Gaussian: analytic weighted least squares / literal row replication under the current unpenalized explicit-smooth-solver objective;
- Binomial/Poisson: statsmodels GLM only after aligning intercept, family/link, weight interpretation, and normalization;
- Gamma/IG/NB/Tweedie: statsmodels or a trusted maintained internal reference only when link/dispersion/power and objective semantics match exactly.

Do not infer a solver defect from coefficient differences until intercept, link, family nuisance parameters, current `C` behavior, weight meaning, normalization, and convergence tolerances are aligned.

## 12. Documentation plan

Update learner-facing documentation and the direct solver surface, not PR-style narratives:

- EN/CN `models/generalized-linear-model.md` — replace categorical weighted explicit Newton/L-BFGS rejection with final capability-aware behavior;
- EN/CN `guides/solver-algorithms.md` — add the weighted Newton/L-BFGS objective and GLM-only non-uniform L-BFGS capability boundary;
- EN/CN `models/losses.md` — keep generic Huber/Quantile L-BFGS examples accurate and state that #150 does not extend genuine non-uniform L-BFGS weights to non-GLM losses;
- EN/CN solver support/matrix pages if their weighting tables are affected;
- typed GLM pages only if they contain conflicting solver/weight claims;
- root + EN/CN changelogs for release history.

Also update the `lbfgs_solver` docstring/API documentation: uniform weights remain backward-compatible for all losses; genuine non-uniform weights are accepted only when the loss advertises the GLM weighted-L-BFGS capability.

Explain to users:

- supported `sample_weight` is normalized analytic objective weighting;
- explicit solver requests are authoritative;
- `solver="auto"` is unchanged;
- GPU changes execution location, not weighted objective semantics;
- exact unsupported rows, including ordered models and non-GLM genuinely non-uniform L-BFGS in this issue.

Physical validator/evidence narration stays in developer/evidence surfaces or changelog, not the main user-learning path.

## 13. Execution and evidence order

1. freeze this plan through plan review/fix;
2. add characterization tests for current Newton, L-BFGS rejection, direct non-GLM boundary, explicit smooth-solver `C`, provenance, and refit-state behavior;
3. implement capability marker + shared analytic-weight preparation while proving Newton preservation;
4. implement GLM-only non-uniform weighted L-BFGS;
5. reopen ordinary explicit smooth-solver weighting and stage/publish fit provenance;
6. close included penalized/CV L-BFGS consumers;
7. run targeted then full hosted tests;
8. update EN/CN direct-solver/model docs and changelogs;
9. freeze physical validator schema/tolerances;
10. run exact-head hosted CI;
11. perform fresh independent code review/fix loop;
12. run physical CUDA gate on exact final numerical source;
13. perform final exact-head/evidence-freshness review.

A docs-only tail after accepted physical validation may reuse evidence only under an explicit scoped exception or validator-native numerical fingerprint contract; any numerical/solver/backend/inference/validator/tolerance change reopens physical validation.

## 14. Acceptance criteria

- [ ] Ordinary weighted explicit Newton reaches `newton_solver` with validated backend-native weights and no stale wrapper rejection.
- [ ] Newton existing weighted behavior/uniform compatibility is unchanged.
- [ ] GLM-capable `lbfgs_solver` supports genuine non-uniform analytic weights with one coherent weighted objective in all value/gradient/line-search evaluations.
- [ ] Non-GLM direct L-BFGS genuinely non-uniform weights remain fail-closed; conforming custom `GLMLoss` follows the GLM interface contract.
- [ ] Explicit weighted L-BFGS reaches L-BFGS for every final supported ordinary GLM row with no fallback.
- [ ] Existing weighted penalized/CV smooth-GLM L-BFGS rows in §4.4 are intentionally closed and tested.
- [ ] `solver="auto"`, IRLS, FISTA, Ordered GLM, specialized LogisticRegression, and existing `C` semantics are unchanged.
- [ ] `_selected_solver/_selected_backend_name/_selected_backend_device` accurately describe successful ordinary GLM execution and follow existing failed-refit transaction semantics without stale/contradictory publication.
- [ ] Weight-rescaling, uniform-weight, and zero-weight-row equivalence hold for supported analytic-weight rows.
- [ ] Existing inference/diagnostics use the same weights and retain previous covariance/estimand support.
- [ ] Formula-side weight alignment is covered.
- [ ] NumPy/CuPy/Torch closure exists for every final claimed ordinary weighted Newton/L-BFGS row.
- [ ] Physical CUDA matrix covers every final ordinary family/link × solver × CuPy/Torch claim plus representative shared penalized/CV L-BFGS consumers.
- [ ] Direct solver docs, EN/CN model/solver docs, and changelogs match the final matrix and remain learner-oriented.
- [ ] Exact final numerical source passes hosted CI and physical CUDA validation.
- [ ] Final fresh code review has no unresolved CRITICAL/HIGH/actionable MEDIUM finding.

## 15. Plan review history

### Round 1 findings fixed in revision `c3796ea...`

- HIGH: generic `lbfgs_solver` consumer graph was under-scoped; non-GLM non-uniform weight expansion is now explicitly blocked in #150.
- HIGH: ordinary fit provenance needed an explicit machine-auditable solver/backend/device contract.
- HIGH: existing weighted penalized/CV L-BFGS rows were left as an implementation-time choice; they are now intentionally included when their independent public weight contract already permits weights.
- MEDIUM: explicit smooth-solver `C` semantics and refit-state behavior now have pre-change characterization gates rather than implicit assumptions.
- MEDIUM: physical acceptance now covers every final claimed ordinary family/link × solver × GPU backend row instead of only a representative pair.

### Round 2 findings fixed in revision `0f55fa5...`

- MEDIUM: direct solver/loss documentation was missing from the docs closure; `models/losses.md`, solver algorithms, and `lbfgs_solver` API docs are now explicit consumers.
- MEDIUM: the GLM-only capability gate now has a layer-safe marker contract (`LossBase=False`, `GLMLoss=True`) and explicit custom-GLMLoss inheritance semantics.
- MEDIUM: ordinary provenance publication is staged until the existing fit transaction's success point, with characterization before any broader refit-state change.
- MEDIUM: zero-weight-row equivalence is now a required objective invariant and regression case.

### Round 3 fresh review

Fresh audit of the complete revised plan found no new CRITICAL/HIGH/actionable MEDIUM plan finding. The generic non-GLM default-fail capability boundary protects Huber/Quantile/Cox from accidental non-uniform weighted L-BFGS expansion; the hosted plan now names representative Huber/Quantile and Cox preservation regressions explicitly.

## 16. Plan-review gate

Plan review is complete. Production implementation may begin from this exact reviewed plan state:

`PLAN REVIEW CLEAN / IMPLEMENTATION OPEN`
