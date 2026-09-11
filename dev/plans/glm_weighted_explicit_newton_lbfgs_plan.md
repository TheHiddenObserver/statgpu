# Weighted explicit Newton/L-BFGS GLM repair plan

Status: DRAFT / PLAN REVIEW OPEN
Tracked issue: #150
Base: `master` after PR #147 merge (`658a23add3ea54dce95c51eb4bba80b91cb0f1b3`)
Working branch: `fix/glm-weighted-explicit-solver-guard`
Target release: 0.2.6 (unreleased)

## 1. Goal

Repair the ordinary `GeneralizedLinearModel` public fit contract so a genuine non-uniform analytic `sample_weight` is not rejected merely because the caller explicitly requested `solver="newton"` or `solver="lbfgs"`.

The two solver rows have different starting points and must not be treated as one generic guard deletion:

- **Newton** already has a maintained backend-native non-uniform weighted numerical implementation. The ordinary GLM wrapper currently blocks that capability before solver entry. This part is an existing-capability/public-contract reconciliation.
- **L-BFGS** still rejects genuine non-uniform weights in the shared solver and then evaluates value/gradient/line-search trials without weights. Supporting the ordinary GLM row therefore requires a real shared weighted-L-BFGS numerical capability, with corresponding consumer and backend closure.

The repair must preserve explicit solver authority: weights must never cause an explicit Newton/L-BFGS request to be silently rewritten to IRLS, FISTA, another backend, or CPU.

## 2. Change classification and active axes

Change types:

1. existing capability reconciliation / public contract repair for weighted explicit Newton;
2. new/materially changed shared numerical capability for non-uniform weighted L-BFGS;
3. shared solver consumer reconciliation because penalized GLM already calls the same `lbfgs_solver(..., sample_weight=...)` entry point.

Active axes:

- public fit/solver contract;
- analytic-weight loss/objective semantics;
- solver/convergence/line-search behavior;
- NumPy/CuPy/Torch backend and concrete-device ownership;
- ordinary GLM wrappers and formula/data path;
- shared penalized-GLM consumers of `lbfgs_solver`;
- inference integration for already-supported ordinary-GLM inference rows;
- docs/changelog and exact-source evidence.

Intentionally unchanged:

- `solver="auto"` dispatch for ordinary `GeneralizedLinearModel`;
- penalty definitions/scaling and ordinary explicit smooth-solver `C` semantics;
- IRLS and FISTA numerical algorithms;
- specialized standalone `LogisticRegression` public API (it is a separate `BaseEstimator`, not a `GeneralizedLinearModel` subclass);
- `OrderedGeneralizedLinearModel`, which continues to reject `sample_weight` explicitly;
- CV APIs that do not reconstruct ordinary `GeneralizedLinearModel` with these explicit solver rows;
- covariance estimators or inferential targets;
- weighted bootstrap semantics (#148) and non-Gaussian parametric bootstrap (#149).

## 3. Baseline reconnaissance

### 3.1 Ordinary GLM wrapper

`GeneralizedLinearModel.fit()` already:

- validates `sample_weight`;
- aligns formula-side weights after missing-row filtering;
- converts the weights once to the selected execution backend;
- resolves the explicit solver and routes `newton`/`lbfgs` to `_fit_smooth_solver()`;
- retains fit weights for loglikelihood/AIC/BIC and supported inference.

The blocking behavior is inside `_fit_smooth_solver()`:

```python
if sample_weight is not None:
    raise ValueError(...)
```

and the downstream calls currently omit `sample_weight` for both `newton_solver` and `lbfgs_solver`.

### 3.2 Newton numerical capability

`newton_solver` already supports non-uniform analytic weights on NumPy/CuPy/Torch. Its maintained contract:

- validates/alines weights to the execution backend;
- preserves the historical floating-point `allclose` uniform-weight rule by treating such vectors as the unweighted path;
- uses the same active non-uniform vector in value, gradient, Hessian/fused gradient+Hessian, the Armijo old objective, and every Armijo trial objective;
- uses normalized average-loss scaling.

Existing solver tests already cover row-replication equivalence, global weight-rescaling invariance, uniform-weight compatibility, invalid weights, and NumPy/Torch-CPU parity.

### 3.3 L-BFGS numerical gap

`lbfgs_solver` currently:

- accepts a `sample_weight` argument but calls `_validate_uniform_sample_weight`;
- documents only uniform weights as valid;
- computes initial gradient, old line-search objective, candidate objective, and updated gradient without passing the weight vector.

Thus removing the ordinary wrapper guard without changing L-BFGS would only move the failure one layer down and would not satisfy #150.

### 3.4 Shared GLM loss support

The maintained GLM loss stack already provides the primitives needed by both smooth solvers:

- `LossBase.value/gradient/fused_value_and_gradient(..., sample_weight=...)` use `sum_i w_i contribution_i / sum_i w_i`;
- `GLMLoss.fused_value_and_gradient` dispatches weighted calls through `_weighted_loss_and_grad`;
- maintained Gaussian, Logistic/Binomial, Poisson, Gamma, Inverse Gaussian, Negative Binomial, and Tweedie losses expose weighted Hessian support for Newton.

This is implementation substrate, not acceptance evidence. Every public family row still requires characterization/tests before being claimed supported.

## 4. Consumer graph

### Direct target consumers

- `GeneralizedLinearModel(family=...)`;
- `PoissonRegression`;
- `GammaRegression`;
- `InverseGaussianRegression`;
- `NegativeBinomialRegression`;
- `TweedieRegression`;
- any other maintained thin wrapper that directly inherits `GeneralizedLinearModel` without replacing `fit()`.

Generic `GeneralizedLinearModel(family="binomial")` is in scope. The separate specialized `LogisticRegression(BaseEstimator)` is not a public-API target for this issue.

### Explicitly preserved/unsupported consumers

- `OrderedGeneralizedLinearModel` and ordered-logit/probit wrappers keep their explicit no-weight contract;
- IRLS/FISTA ordinary GLM paths remain unchanged;
- `solver="auto"` remains unchanged.

### Shared L-BFGS consumers

Because `_lbfgs.py` is shared, weighted-L-BFGS work must also reconcile:

- `PenalizedGeneralizedLinearModel` explicit `solver="lbfgs"` with smooth penalties;
- typed penalized smooth-GLM wrappers that route through `_fit_loss_backend`;
- `PenalizedGLM_CV` rows whose existing solver table may select L-BFGS (notably CV L2 Negative Binomial and Gamma/Inverse-Gaussian rows), including weighted candidate/final-refit behavior if those rows are publicly weight-capable;
- direct `lbfgs_solver` users/tests.

No new penalty or CV row is invented. If an existing consumer has an independent documented weight restriction, it remains fail-closed and is tested as such.

## 5. Desired objective and compatibility contract

For supported analytic-weight rows, both Newton and L-BFGS optimize exactly

```text
L(beta) = sum_i w_i * ell_i(beta) / sum_i w_i + P(beta)
```

where `P=0` for ordinary `GeneralizedLinearModel` smooth-solver fits and is the already-existing smooth penalty for shared penalized consumers.

Required identities:

1. multiplying every non-uniform weight by a positive scalar does not change the optimum beyond maintained numerical tolerance;
2. positive uniform weights reproduce the historical unweighted route;
3. integer frequency-style weights agree with literal row replication for reference rows when the penalty/objective normalization is aligned;
4. every value/gradient/Hessian/quasi-Newton/line-search evaluation represents the same weighted objective;
5. explicit solver identity remains the requested solver.

The historical Newton floating-point uniformity rule (`allclose(values, values[0])`) remains unchanged. Weighted L-BFGS must use the same validation/alignment/uniform-compatibility convention unless characterization proves an existing public L-BFGS rule that must be retained separately.

## 6. Planned implementation

### 6.1 Shared analytic-weight preparation

Avoid two diverging implementations of backend alignment and the historical uniform-weight rule.

Preferred implementation:

- extract/generalize Newton's current weight preparation into one private solver utility (for example `_prepare_analytic_sample_weight(sample_weight, n_samples, backend, ref_arr)`);
- have Newton call that shared utility without changing its observable behavior;
- have L-BFGS use the same utility;
- preserve a thin compatibility alias only if existing internal/static tests import the Newton-private name.

Before/after characterization must prove the Newton path is behavior-preserving.

### 6.2 Weighted L-BFGS

Thread the prepared active weight vector through every loss evaluation:

- initial fused value/gradient;
- current/old objective for line search;
- every candidate objective in backtracking;
- updated gradient after the accepted candidate.

Penalty value/gradient handling remains unchanged.

The two-loop recursion, curvature-history update, convergence tests, and line-search constants remain algorithmically unchanged; only the objective/gradient inputs become weight-coherent.

No host conversion of the full weight vector is allowed for explicit CuPy/Torch execution. Scalar reductions needed by convergence/line-search synchronization may retain the established backend helper behavior.

### 6.3 Ordinary `GeneralizedLinearModel`

Replace the blanket `_fit_smooth_solver()` rejection with capability-aware dispatch:

- pass the already validated/backend-aligned `sample_weight` to `newton_solver` or `lbfgs_solver`;
- do not change `solver="auto"`;
- do not silently substitute a different solver if a loss/backend row rejects weights;
- preserve existing intercept layout and result/reporting layout;
- keep unsupported family/solver combinations failing with a precise solver/loss capability error before publishing fitted state.

### 6.4 Shared penalized consumer closure

Once `lbfgs_solver` genuinely supports non-uniform weights, characterize existing penalized L-BFGS callers rather than leaving an accidental capability expansion untested.

For every penalized row already allowed to call L-BFGS with `sample_weight`:

- either prove the row is now supported with the same weighted objective and backend contract;
- or add/narrow a capability guard at the correct consumer boundary with an explicit message.

Do not reintroduce a generic blanket rejection in the shared solver merely to preserve an unreviewed consumer limitation.

## 7. Intended support matrix to prove

The plan starts with the following candidate matrix. A row becomes a public support claim only after the corresponding deterministic characterization succeeds.

| Family/loss | Newton + non-uniform weights | L-BFGS + non-uniform weights | Notes |
|---|---|---|---|
| Gaussian / squared error | candidate supported | candidate supported | weighted Hessian and weighted fused value/grad exist |
| Binomial / logistic through generic GLM | candidate supported | candidate supported | generic GLM only; standalone `LogisticRegression` API unchanged |
| Poisson | candidate supported | candidate supported | weighted Hessian/fused path exists |
| Gamma (maintained links) | candidate supported | candidate supported | test log and any public inverse-power link separately |
| Inverse Gaussian | candidate supported | candidate supported | family-specific convergence needs characterization |
| Negative Binomial | candidate supported | candidate supported | important shared CV L-BFGS consumer |
| Tweedie | candidate supported | candidate supported | power-specific stability needs characterization |
| Ordered models | unsupported | unsupported | existing explicit no-weight contract preserved |

If a family fails numerical correctness/convergence under the requested solver after the weighting implementation is correct, keep that exact family × solver row fail-closed and document it. Do not fall back to another solver.

## 8. Hosted validation plan

### 8.1 Shared solver-level tests

Add focused L-BFGS tests analogous to the maintained weighted-Newton regressions:

- integer-weight row replication on at least squared error and logistic/Poisson;
- global positive weight-rescaling invariance;
- uniform and historical almost-uniform floating weights reproduce the unweighted path;
- invalid shape/negative/non-finite/zero-total weights fail before iteration;
- NumPy vs Torch-CPU parity;
- direct solver instrumentation proving every fused value/gradient call receives the active non-uniform weights, including line-search candidate calls;
- line-search failure behavior remains fail-visible and does not accept an unverified weighted trial.

Newton characterization tests must show extracting the shared weight helper does not change existing weighted/unweighted results.

### 8.2 Ordinary public GLM tests

For explicit `newton` and `lbfgs` separately:

- non-uniform weighted fit reaches the requested solver rather than the historical wrapper error;
- `model.solver` remains the explicit request;
- coefficients/intercept are finite and agree with an analytic/external/reference construction where available;
- positive constant rescaling of weights leaves fitted coefficients/intercept invariant;
- uniform weights match unweighted fit;
- `fit_intercept=True` and `False`;
- generic `GeneralizedLinearModel` plus maintained typed wrappers;
- formula/data route with dropped/missing rows aligns `sample_weight` correctly;
- failed refits do not leave a stale successful fitted/inference state.

Run a family matrix for Gaussian, Binomial, Poisson, Gamma, Inverse Gaussian, Negative Binomial, and Tweedie. Record rows that are deliberately unsupported instead of silently skipping them.

### 8.3 Inference/diagnostic preservation

For ordinary GLM rows where `compute_inference=True` is already supported:

- fitted numerical objective and inference use the same weights;
- `solver_used` / fit metadata reports the explicit solver;
- `_sample_weight_inf`, loglikelihood, AIC/BIC, bse/statistic/p-value/CI remain coherent with the weighted fit;
- nonrobust/robust covariance support is not expanded beyond the pre-existing contract.

Use at least one Newton and one L-BFGS supported family with inference enabled. Include an external or independently computed weighted baseline where feasible.

### 8.4 Shared penalized/CV regressions

Because weighted L-BFGS changes a shared solver:

- explicit weighted penalized L-BFGS on a smooth supported row;
- weighted `PenalizedGLM_CV` rows that currently select L-BFGS, including candidate fitting and selected final refit;
- solver identity/provenance remains L-BFGS;
- existing Newton/FISTA/IRLS dispatch is unchanged;
- unsupported penalty/loss combinations continue to fail before misleading numerical work.

## 9. Backend/device closure

Weighted L-BFGS is a materially changed shared numerical capability, so completion requires NumPy + CuPy + Torch for every claimed supported public row unless a narrower row is explicitly justified.

Hosted tests should prove what can be proved without physical CUDA, including Torch CPU helper/parity and static routing contracts. Final physical CUDA evidence must cover at least:

- ordinary GLM weighted explicit Newton on CuPy and Torch;
- ordinary GLM weighted explicit L-BFGS on CuPy and Torch;
- at least two representative families, one of which is non-Gaussian;
- a maintained penalized/shared L-BFGS consumer if that capability is claimed;
- concrete device provenance and no CPU numerical fallback;
- heterogeneous input containers if the public fit boundary permits them;
- coefficient/intercept parity against the frozen NumPy reference under fixed data/weights;
- weight-rescaling invariance or an equivalent objective invariant on GPU.

Freeze validator schema, datasets, tolerances, and route matrix before the first acceptance run. A failed physical run must not be made green by silently loosening tolerances.

## 10. External/reference validation

Use the strongest aligned reference available per family:

- Gaussian: analytic weighted least-squares / row-replication identity when unpenalized explicit smooth solver semantics are being tested;
- Binomial/Poisson: statsmodels GLM with aligned analytic/frequency weighting only after confirming equivalent weight semantics and objective scale;
- other families: statsmodels or an internal trusted IRLS/FISTA reference only when family/link/objective parameterizations match exactly.

Do not declare a solver bug from an external coefficient difference until intercept, link, dispersion/power, regularization/C semantics, weight interpretation, objective normalization, and tolerance are aligned.

## 11. Documentation plan

Update only affected user-facing surfaces, keeping learner-facing prose evergreen:

- EN/CN `models/generalized-linear-model.md` — remove the categorical explicit-Newton/L-BFGS weighted rejection and explain capability-aware explicit solver behavior;
- EN/CN solver guide/matrix where the current support table or weighting note is affected;
- typed GLM pages only if they currently make a contradictory weight/solver claim;
- root + EN/CN changelog entries describing the public capability repair.

User docs should explain:

- `sample_weight` is analytic objective weighting for these supported rows;
- explicit solver requests are authoritative;
- `solver="auto"` remains unchanged;
- GPU changes execution location, not the weighted statistical objective;
- exact unsupported family × solver rows, if any.

Do not turn model pages into PR/evidence logs; physical validator details belong in developer/evidence surfaces and changelog only when useful for release auditing.

## 12. Execution and evidence order

1. freeze this plan through plan review/fix;
2. add/adjust characterization tests that capture current Newton behavior and the historical ordinary/L-BFGS rejection;
3. implement shared weight preparation + weighted L-BFGS;
4. reopen ordinary explicit smooth-solver weighting by passing weights to the requested solver;
5. close shared penalized/CV L-BFGS consumers;
6. run targeted hosted tests, then full relevant matrix;
7. update EN/CN user docs and changelogs;
8. freeze physical CUDA validator schema/tolerances;
9. run exact-head hosted CI;
10. perform fresh independent code review/fix loop;
11. run physical CUDA gate on the exact final numerical source;
12. perform final exact-head/evidence-freshness review.

If docs-only changes occur after an accepted physical run, evidence reuse is allowed only under an explicit, scoped evidence exception or a validator contract that fingerprints the unchanged numerical source; otherwise the physical artifact remains historical.

## 13. Acceptance criteria

- [ ] Ordinary weighted explicit Newton no longer fails at the stale wrapper guard and reaches `newton_solver` with the validated backend-native weight vector.
- [ ] Newton's existing weighted numerical behavior and historical uniform-weight compatibility remain unchanged.
- [ ] `lbfgs_solver` supports genuine non-uniform analytic weights using one coherent normalized weighted objective in every value/gradient/line-search evaluation.
- [ ] Explicit weighted L-BFGS reaches L-BFGS for every claimed supported ordinary GLM row; unsupported rows have precise capability errors and no fallback.
- [ ] No explicit Newton/L-BFGS request is silently changed because weights are present.
- [ ] `solver="auto"` behavior is unchanged.
- [ ] NumPy/CuPy/Torch closure exists for every newly claimed weighted L-BFGS row.
- [ ] Shared penalized/CV L-BFGS consumers are inventoried and either covered or explicitly fail-closed at their own capability boundary.
- [ ] Positive constant weight-rescaling invariance and uniform-weight equivalence hold for supported analytic-weight rows.
- [ ] Existing ordinary GLM inference/diagnostics use the same fitted weight convention and retain their previous covariance support matrix.
- [ ] Formula-side weight alignment is covered.
- [ ] Ordered GLM and unrelated specialized APIs remain unchanged.
- [ ] EN/CN docs and changelogs describe the final support matrix without update-log-style user prose.
- [ ] Exact final numerical source passes hosted CI and required physical CuPy/Torch CUDA validation.
- [ ] Final fresh code review has no unresolved CRITICAL/HIGH/actionable MEDIUM finding.

## 14. Plan-review gate

No production implementation starts until this plan has undergone a fresh independent review/fix loop under `.claude/skills/code-review` and reaches:

`PLAN REVIEW CLEAN / IMPLEMENTATION OPEN`
