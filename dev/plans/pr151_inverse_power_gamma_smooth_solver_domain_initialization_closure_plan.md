# PR #151 inverse-power Gamma smooth-solver domain / initialization closure plan

Status: PLAN DRAFT / REVIEW-FIX IN PROGRESS  
Parent PR: #151  
Parent issue: #150  
Follow-up to be folded back after implementation: #152  
Branch: `fix/glm-weighted-explicit-solver-guard`  
Original plan baseline: `b5228fdc4ed78af6fcaa760ecee1475ae4ebe1a7`

## 1. Goal

Close the remaining inverse-power Gamma explicit smooth-solver gap inside PR #151 rather than retaining `fit_intercept=True` as an artificial capability condition.

The final contract is domain based, not weight-shape based:

- inverse-power Gamma has mathematical predictor domain `eta = X @ beta > 0`;
- current Gamma code clips inverse-link predictors to `_ETA_LO/_ETA_HI`; outside the interval where clipping is inactive, its value, gradient and Hessian are not derivatives of one common smooth clipped objective, so maintained explicit Newton/L-BFGS training must keep every evaluated active predictor in a certified smooth numerical interior;
- explicit Newton/L-BFGS start from a family-valid interior point and keep every accepted iterate in that interior;
- `fit_intercept=False` is no longer categorically rejected: it is supported when statgpu can certify an interior start and the requested solver converges without violating the maintained smooth-domain contract;
- inability to certify a start, or domain-boundary stagnation before solver convergence, is fail-visible and must not publish a successful fit;
- unweighted, positive-uniform, effectively-uniform and genuine non-uniform analytic weights use the same domain/initialization policy;
- weight classification still preserves PR #151's historical objective contract: omitted/uniform/effectively-uniform weights execute the unweighted objective, while genuine non-uniform weights execute `sum(w_i * ell_i) / sum(w_i)`;
- genuine zero-weight rows do not constrain the training domain because they do not belong to the executed weighted objective.

Final training flow:

```text
inverse-power Gamma + explicit Newton/L-BFGS
    -> finalize backend/device/dtype and prepared analytic-weight identity
    -> build/validate a smooth-domain interior start on the executed training design
    -> compute an analytic domain step cap for each search direction
    -> Armijo backtracking inside that cap
    -> publish only after ordinary convergence succeeds in the maintained smooth domain
    -> otherwise fail/warn according to the explicit domain/convergence contract
```

## 2. Change classification and active review axes

Classification:

1. existing-capability reconciliation / completion of the original #150 support matrix;
2. numerical correctness repair for inverse-power Gamma initialization and globalization;
3. shared Newton/L-BFGS contract change through loss-owned private optimization-domain hooks;
4. bounded correction of inverse-power Gamma smooth-L2 CV initialization/scoring because the public CV surface otherwise evaluates the wrong link;
5. evidence-invalidating numerical change after the previously accepted `c6781cb6` physical run.

Active axes:

- loss/objective/domain;
- solver initialization, step capping, line search, convergence and failure semantics;
- analytic `sample_weight` and zero-weight-row semantics;
- intercept/no-intercept behavior;
- NumPy/CuPy/Torch backend, device and working dtype;
- ordinary GLM, direct solver, penalized Gamma and smooth-L2 `PenalizedGLM_CV` consumers;
- formula no-intercept behavior;
- inference preservation for successfully fitted public rows;
- tests/docs/changelog/evidence freshness;
- exact-source physical CUDA revalidation.

Not intended to change:

- `solver="auto"` dispatch policy;
- IRLS/FISTA/FISTA-BB/FISTA-LLA/ADMM numerical algorithms;
- Gamma log-link behavior;
- ordinary explicit smooth-solver `C` semantics;
- penalty definitions/scaling;
- covariance estimands;
- current prediction and held-out validation clipping semantics except that inverse-power Gamma CV must use the correct inverse-power loss rather than the log-link loss;
- numerical values of `GammaLoss._ETA_LO/_ETA_HI` unless a separate pre-implementation review proves the constants themselves are incorrect;
- optimized sparse Gamma CV kernels beyond preventing this smooth-L2 closure from using log-link-only formulas;
- ordered models or standalone logistic APIs.

## 3. Current inconsistencies to close

### 3.1 Original #150 support matrix versus the current no-intercept guard

The reviewed parent plan targeted weighted explicit Newton/L-BFGS support for all maintained ordinary GLM family/link rows and explicitly required both `fit_intercept=True` and `False`, including inverse-power Gamma.

The current runtime installer instead:

- supplies a family-aware start only when an intercept is present;
- rejects the newly opened genuine-nonuniform no-intercept row;
- leaves omitted/uniform/effectively-uniform no-intercept calls on the historical zero-start path.

That narrowing was recorded in the implementation addendum and tracked as #152, but it is an implementation limitation rather than a statistical prohibition.

### 3.2 Historical zero start is outside the mathematical inverse-Gamma domain

For inverse-power Gamma,

\[
\ell_i(\eta_i)=y_i\eta_i-\log\eta_i,
\qquad \eta_i>0.
\]

A no-intercept zero start gives `eta=0` and relies on clipping before the first derivative/line-search evaluation. Once PR #151 reopens numerical source and requires fresh evidence anyway, preserving that boundary-dependent trajectory is not a useful compatibility goal.

Uniform and unweighted inputs represent the same normalized objective, so initialization/domain behavior must not jump when a weight vector crosses the historical `allclose` uniformity threshold.

### 3.3 `eta>0` alone is insufficient for the current second-order implementation

Current inverse-Gamma value/gradient/Hessian code clips the predictor. Outside the interval where clipping is inactive, the implemented value and derivatives do not represent one common smooth objective. Explicit Newton/L-BFGS therefore must not evaluate there.

Let

\[
\eta_{\rm lo}=\texttt{GammaLoss.\_ETA\_LO},
\qquad
\eta_{\rm hi}=\texttt{GammaLoss.\_ETA\_HI}.
\]

Define a dtype-aware strict training interior

\[
L=\eta_{\rm lo}+\delta_\eta,
\qquad
U=\eta_{\rm hi}-\delta_\eta,
\]

and require

\[
L < x_i^\top\beta < U\qquad\forall i\in A.
\]

`delta_eta` is derived from the executed floating dtype and bound magnitudes; its formula and constants are frozen by hosted characterization before schema-v4 physical validation and are not loosened after seeing a GPU result.

This is a **maintained numerical smooth-domain contract**, not a claim that the statistical Gamma inverse link has a finite upper domain.

### 3.4 Penalized inverse-power Gamma has a wrong framework-owned intercept start

`_PenalizedFitMixin._fit_loss_backend()` currently initializes Gamma intercepts using `log(mean(y))`, which is appropriate for log-link Gamma but not inverse-power Gamma. Because that produces a non-`None` `init_coef`, it can bypass a loss-level family-valid initializer.

Inverse-power Gamma must stop using that generic log-link initializer; Gamma log-link behavior remains unchanged.

### 3.5 Smooth-L2 Gamma CV currently evaluates a log-link-only validation loss

`PenalizedGLM_CV` registers `"gamma"` validation with

\[
\mu=\exp(\eta),
\qquad \ell=y/\mu+\log\mu,
\]

and optimized scoring dispatches primarily by loss name. Some branches also resolve `"gamma"` without preserving `loss_kwargs`. Thus an inverse-power Gamma candidate can be fitted under one link and ranked under another.

For the public smooth-L2 inverse-power Gamma CV route, candidate construction, scoring, alpha selection and selected refit must all resolve the same `GammaLoss(link="inverse_power")`. This PR should use the generic actual-loss evaluator for that route rather than create a new optimized inverse-power fold-batched kernel.

## 4. Prepared analytic weights and active training rows

Extract Newton/L-BFGS's duplicated preparation into one private helper in `statgpu/solvers/_utils.py`, e.g.

```python
_prepare_analytic_sample_weight(sample_weight, n_samples, backend, ref_arr)
```

with established semantics:

1. validate shape/finiteness/non-negativity/positive total;
2. align to the **executed** design backend, concrete device and dtype;
3. apply historical floating `allclose(values, values[0])` after alignment;
4. return `None` for omitted/uniform/effectively-uniform inputs;
5. return the aligned vector for genuine non-uniform weights.

Newton and L-BFGS both use it. L-BFGS retains its loss-capability gate after preparation. A compatibility alias may remain for the old Newton-private helper, but duplicate classification logic must not remain.

Let the prepared result be `w_eff`:

- `None` -> all retained training rows are active;
- genuine non-uniform vector ->
  \[
  A=\{i:w_{\mathrm{eff},i}>0\}.
  \]

Thus an effectively-uniform public vector intentionally inherits the unweighted all-row objective/domain, while a genuine zero-weight row imposes no training-domain restriction.

## 5. Loss-owned private smooth-domain hooks

Add private no-op hooks to `LossBase` (or an equivalent optional private capability interface) with semantics equivalent to:

```python
_loss_domain_initial_point(X, y, sample_weight=None) -> array | None
_loss_domain_is_feasible(X, coef, sample_weight=None) -> bool
_loss_domain_max_step(X, coef, delta, sample_weight=None) -> float | None
```

`delta` is additive:

\[
\beta(t)=\beta+t\Delta.
\]

Default behavior for other losses is no special start, always feasible, no domain cap. `GammaLoss(link="inverse_power")` implements the hooks from its own `_ETA_LO/_ETA_HI`; log-link Gamma keeps defaults.

Generic solvers consume the private interface without importing `glm_core` or branching on a Gamma name. Tests must prove no non-Gamma solver behavior changes.

## 6. Backend-native interior initialization

### 6.1 Find a positive separating direction

On active executed design `X_A`, first seek `d` with

\[
x_i^\top d>0\qquad\forall i\in A.
\]

Normalize every finite nonzero active row by a positive norm,

\[
u_i=x_i/\|x_i\|_2,
\]

which preserves sign feasibility. An active all-zero/non-finite row is an immediate exact failure.

Fast path: if one executed design column has one certified strict sign on all active rows, use that signed basis vector. An augmented ones column therefore recovers the pure-intercept direction without the geometric fallback.

Otherwise solve the minimum-norm convex-hull problem

\[
\min_{c\in\operatorname{conv}(u_i)}\frac12\|c\|_2^2
\]

with deterministic backend-native Gilbert / Frank-Wolfe iterations:

\[
c_0=|A|^{-1}\sum_{i\in A}u_i,
\]

\[
s_k=u_{j_k},\qquad j_k=\arg\min_i u_i^\top c_k,
\]

\[
m_k=\min_i u_i^\top c_k,
\qquad G_k=\|c_k\|_2^2-m_k,
\]

\[
\gamma_k=
\operatorname{clip}\left(
\frac{\|c_k\|_2^2-c_k^\top s_k}{\|c_k-s_k\|_2^2},0,1\right),
\]

\[
c_{k+1}=(1-\gamma_k)c_k+\gamma_k s_k.
\]

A positive `m_k` above the frozen dtype-aware margin certifies a separating direction. Tiny denominator before certification, convergence near the origin, or iteration exhaustion is **numerically unresolved / no certified strict interior**, not a proof of mathematical infeasibility.

No SciPy/CPU LP/QP fallback is introduced. CuPy/Torch geometry stays on the selected device; only scalar certificate/control synchronization is allowed.

### 6.2 Scale the direction into the unclipped smooth band

For a certified direction let

\[
a_i=x_i^\top d>0,
\qquad a_{\min}=\min a_i,
\qquad a_{\max}=\max a_i.
\]

A positive scalar `c` keeps this ray inside `(L,U)` iff

\[
\frac{L}{a_{\min}}<c<\frac{U}{a_{\max}}.
\]

Along `beta=c d`, the unpenalized inverse-Gamma ray optimum is

\[
c_{\rm ray}
=\frac{s}{\sum_{i\in A}w_i y_i a_i}
\]

or `n / sum(y_i a_i)` under the unweighted objective.

Project `c_ray` into a strict interior of the admissible scale interval and finally verify on the **original, unnormalized** active design that every predictor is finite and in `(L,U)`.

For the pure intercept direction this equals `1/mean(y)` or `1/weighted_mean(y)` whenever that ray optimum is already interior; if it must be adjusted to stay inside the numerical band, docs/tests record the adjustment instead of claiming exact identity.

Failure of this particular certified ray to fit `(L,U)` does not prove another coefficient vector cannot. Report `no numerically certified smooth-domain start` unless an exact contradiction was independently certified. PR #151 deliberately does not add a general two-sided LP solver.

Geometry tolerances, interior margins and maximum iterations are frozen before physical validation.

## 7. Domain-aware maximum step

For feasible current `beta` and additive direction `Delta`, on active rows define

\[
\eta=X_A\beta,
\qquad r=X_A\Delta.
\]

To remain in `(L,U)`:

\[
t_{i,\rm lo}=(\eta_i-L)/(-r_i)\quad(r_i<0),
\]

\[
t_{i,\rm hi}=(U-\eta_i)/r_i\quad(r_i>0).
\]

Take the minimum positive crossing time and apply a frozen interior safety factor. Newton supplies `Delta=-d` under its subtract-direction convention; L-BFGS supplies its ordinary additive direction.

Armijo starts at

\[
t_0=\min(1,t_{\rm domain})
\]

when the domain cap is finite, then retains the existing halving and sufficient-decrease rule. A defensive feasibility check runs before **every** trial objective evaluation.

If the current gradient is not converged but the domain hook reports no numerically meaningful positive step because the iterate is pinned to the maintained smooth boundary, the solver must surface a specific domain-boundary failure. It must not mark the estimator successfully fitted merely because the generic coefficient change became tiny. The implementation may use a private exception propagated by the estimator or another explicit fail-closed mechanism; it must not silently convert this case to convergence.

Ordinary Armijo failure for a positive interior step keeps the existing solver warning policy; PR151 warning-as-error acceptance tests ensure such stagnation is not accepted as closure.

## 8. Newton/L-BFGS integration and warm-start ownership

For `newton_solver` and `lbfgs_solver`:

1. preprocess X/y;
2. prepare analytic weights on final backend/dtype;
3. obtain or validate initial coefficients before any value/gradient/Hessian evaluation;
4. compute the loss-domain step cap before Armijo;
5. validate every trial before objective evaluation;
6. preserve existing Armijo constants, convergence tolerances, Newton rank fallback, L-BFGS history update and ordinary warning semantics;
7. never publish domain-boundary stagnation as successful convergence.

Warm-start ownership:

- direct exported solver `init_coef`: caller-authoritative; invalid -> precise domain error, no silent replacement;
- ordinary GLM explicit smooth fit: no public init control, so use loss initializer;
- penalized/CV starts: framework-owned. Validate at their owning estimator/fold boundary; if invalid for the current executed design, discard and pass `None` for fresh family-valid initialization;
- do not add a generic solver flag that guesses warm-start ownership.

## 9. Estimator and CV integration

### 9.1 Ordinary GLM

Simplify `statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`:

- remove the genuine-nonuniform no-intercept categorical guard;
- remove its duplicate inverse-Gamma start after shared loss/solver initialization proves equivalent supported behavior;
- retain finalized `X_work`, working-dtype authority, explicit solver identity and provenance;
- preserve runtime-installer idempotence/import-order/introspection.

No Gamma feasibility logic remains in the ordinary wrapper.

### 9.2 Penalized smooth Gamma

In `_PenalizedFitMixin._fit_loss_backend()`:

- preserve current log-link Gamma intercept initialization;
- for resolved `GammaLoss(link="inverse_power")`, do **not** create generic `log(mean(y))` init;
- absent a framework warm start, pass `None` so the loss initializer receives finalized `X_work`, prepared weights and backend/dtype;
- validate/reseed framework-owned warm starts per §8;
- retain selective-penalty/intercept non-penalization.

`PenalizedGammaRegression(link="inverse_power")` and the generic equivalent are blocking maintained smooth L2/no-penalty consumers.

### 9.3 PenalizedGLM_CV smooth-L2 inverse-power Gamma

For public `PenalizedGLM_CV(loss="gamma", loss_kwargs={"link":"inverse_power"}, penalty="l2", ...)`:

- candidate estimator construction preserves `loss_kwargs`;
- candidate/fold internal warm starts follow framework-owned validation/reseed semantics;
- validation scoring bypasses the log-link-only `_val_gamma` optimized registry entry and calls the actual resolved inverse-power Gamma loss;
- every loss-resolution branch on this L2 route preserves the supplied `loss_kwargs`;
- alpha selection and selected full-data refit therefore use the same link/objective family;
- domain/contract failures are not converted into MSE or log-link fallback scores.

**Training versus validation distinction:** the strict `(L,U)` smooth-domain contract applies to Newton/L-BFGS **training evaluations**. Held-out CV scoring and public prediction retain the repository's existing clipping semantics in this PR; the CV fix here is that they evaluate the correct inverse-power loss/link. Do not document the training interior as a universal guarantee for unseen covariates.

This PR does not build a new optimized inverse-power fold-batched/sparse Gamma CV kernel. If an optimized sparse route would otherwise use log-link-only formulas, bypass it for inverse-power or preserve an explicit unsupported boundary; do not silently compute the wrong link.

### 9.4 Formula and inference

Ordinary formula fits use the same finalized numerical design after Patsy row filtering/weight alignment, including formulas without an intercept.

For already-supported inference rows, successful fitted parameters entering inference satisfy the training smooth-domain contract. Covariance/estimand/reference-distribution semantics stay unchanged.

## 10. Blocking consumer graph

1. `GammaRegression(link="inverse_power")` and generic ordinary equivalent;
2. direct `newton_solver` / `lbfgs_solver` with inverse-power `GammaLoss`;
3. `PenalizedGammaRegression(link="inverse_power")`;
4. generic penalized Gamma inverse-power smooth L2/no-penalty route;
5. `PenalizedGLM_CV` inverse-power Gamma smooth-L2 candidate scoring/selection/final refit;
6. ordinary formula routes;
7. already-supported inference reached by these fits.

Before coding, inventory every affected call site passing `init_coef` and label it caller-owned or framework-owned.

Preservation: Gamma log link; all non-Gamma Newton/L-BFGS consumers; IRLS/FISTA/etc.; non-GLM weighted-L-BFGS boundaries; ordered models; standalone logistic.

## 11. Hosted validation plan

### 11.1 Objective/derivative consistency inside the smooth band

For predictors strictly inside `(L,U)` verify:

\[
\ell=y\eta-\log\eta,
\qquad
\partial\ell/\partial\eta=y-1/\eta,
\qquad
\partial^2\ell/\partial\eta^2=1/\eta^2,
\]

against analytic and finite-difference checks, with weighted normalization `sum(w contribution)/sum(w)`.

Instrument explicit Newton/L-BFGS so no initial/current/trial value/gradient/Hessian evaluation receives an active training predictor outside `(L,U)`.

### 11.2 Geometric/analytic fixtures

1. **1D exact feasible reference:** positive X with
   \[
   \beta^*=\left(\frac{\sum_iw_i y_i x_i}{\sum_iw_i}\right)^{-1}.
   \]
   Cover omitted, uniform/effectively-uniform and genuine non-uniform weights.
2. **multi-feature feasible half-space** requiring a column combination and exercising Frank-Wolfe;
3. **contradictory active rows** `x,-x`;
4. **active zero row**;
5. **zero-weight contradictory row** matching row deletion;
6. **near-boundary/unresolved geometry**;
7. **ray scale adjustment** into `(L,U)`;
8. **separator ray not certifiable in `(L,U)`**, which reports unresolved rather than exact infeasible.

Successful fits assert coefficient error, gradient/KKT residual, training predictor bounds and positive weight-rescaling invariance.

### 11.3 Independent external baseline

On a well-conditioned CPU fixture whose optimum lies comfortably inside `(L,U)`, align statgpu inverse-power Gamma with statsmodels GLM Gamma + inverse-power link (or the maintained equivalent API) for:

- no-intercept unweighted;
- no-intercept genuine analytic/frequency weight configuration with explicitly matched weight semantics if statsmodels exposes an equivalent;
- an intercept-bearing preservation case if useful.

Before using the external result as an acceptance oracle, document the exact objective/weight mapping. If statsmodels cannot express the same weighted objective cleanly, retain the 1D analytic oracle as blocking and use the external comparison only for the aligned subset rather than changing statgpu semantics.

### 11.4 Initialization regression

- ordinary intercept start equals `1/mean(y)` or `1/weighted_mean(y)` when already inside `(L,U)`;
- omitted/uniform/effectively-uniform calls share initializer/objective classification;
- no-intercept ordinary route never starts from zero;
- penalized inverse-power Gamma no longer uses `log(mean(y))`;
- Gamma log-link penalized initialization unchanged;
- direct invalid explicit init fails before loss evaluation;
- invalid framework-owned start is discarded/reseeded.

### 11.5 Domain-aware line search and failure publication

For Newton and L-BFGS:

- full step crossing lower bound -> domain `t_0<1`;
- full step crossing upper bound -> domain `t_0<1`;
- feasible objective-increasing candidate still fails Armijo;
- no infeasible trial reaches loss evaluator;
- artificial boundary-stagnation case does not report successful fit/convergence;
- existing 20/25 Armijo trial budgets and warning behavior otherwise remain unchanged.

### 11.6 Backend/dtype/weights

- NumPy and Torch-CPU deterministic parity;
- mixed Torch inputs classify weights/domain using finalized promoted `X_work` dtype;
- shared weight prep preserves existing uniform/effectively-uniform classification exactly;
- geometry/domain checks stay device-native; no full X/weight host copy;
- CuPy/Torch CUDA parity is a physical completion gate.

### 11.7 Public/penalized/CV/formula/inference closure

- ordinary Gamma intercept/no-intercept and generic equivalent;
- formula intercept/no-intercept + missing-row weight alignment;
- penalized inverse-power Gamma smooth L2/no-penalty;
- smooth-L2 inverse-power `PenalizedGLM_CV` candidate fit, correct-link scoring, alpha selection and final refit;
- CV fixture where inverse-power and incorrect log-link ranking differ, preventing accidental false-green scoring;
- CV domain errors not converted to MSE/log-link fallback;
- inference smoke/parity on already-supported rows;
- clone/get_params/introspection unchanged.

### 11.8 Preservation

- Gamma log-link Newton/L-BFGS/CV;
- non-Gamma weighted Newton/L-BFGS;
- IRLS/FISTA/auto routes;
- runtime installer idempotence/import order/provenance.

## 12. Documentation and issue handling

After implementation proof, update affected EN/CN GLM/Gamma model pages, solver/support matrix, algorithm/domain note, relevant smooth-L2 CV text, root/EN/CN changelog and PR151 body.

Public wording:

> inverse-power Gamma mathematically requires a positive linear predictor. Explicit Newton/L-BFGS keep training evaluations inside statgpu's maintained smooth numerical interior so value, gradient and Hessian correspond to one coherent objective. An intercept gives an immediate feasible direction; no-intercept designs are supported when statgpu can certify an interior start and the optimizer converges without hitting the maintained numerical-domain boundary.

Do not imply the training interior is guaranteed for unseen prediction rows; prediction retains current clipping semantics.

Issue #152 remains open until this implementation passes hosted review and schema-v4 physical acceptance, then close it as completed by PR151.

## 13. Physical CUDA evidence: schema v4

Any numerical change after `c6781cb6` makes schema-v3 P100 evidence historical for the final implementation. Keep the retained v3 JSON immutable.

Before the new run, freeze schema **v4** with:

- existing v3 tolerances/warning-as-error gates unless a principled pre-run review changes them;
- feasible no-intercept inverse-power Gamma × Newton/L-BFGS × NumPy/CuPy/Torch;
- intercept inverse-power preservation;
- omitted/uniform/genuine non-uniform behavior;
- weight-rescaling invariance;
- min/max active training predictor inside frozen `(L,U)`;
- domain-step-cap characterization;
- truthful solver/backend/device provenance;
- penalized inverse-power Gamma L2 direct fit and smooth-L2 CV selected refit on NumPy/CuPy/Torch;
- existing v3 ordinary/cross-container/shared-consumer coverage unless schema-v4 review explicitly justifies replacement.

Raw v4 evidence records exact source SHA, clean worktree, environment and status and is retained outside benchmark-source scan roots. No post-failure tolerance/domain-margin loosening without a new reviewed schema.

## 14. Implementation order

1. Finish repeated independent plan review/fix passes.
2. Inventory all affected Newton/L-BFGS init owners and inverse-power Gamma L2 CV scoring branches.
3. Extract shared analytic-weight preparation with preservation tests.
4. Add private smooth-domain hooks and inverse-power Gamma `(L,U)` implementation.
5. Add backend-native separator/initializer plus analytic/geometric tests.
6. Integrate hooks into Newton and close Newton tests.
7. Integrate hooks into L-BFGS and close weighted L-BFGS tests.
8. Remove ordinary wrapper guard/duplicate initializer.
9. Repair penalized inverse-power Gamma internal initialization/warm-start handling.
10. Repair smooth-L2 inverse-power Gamma CV link resolution/scoring/refit.
11. Close ordinary/formula/inference/external-reference/preservation tests.
12. Update EN/CN docs/changelogs/support matrices.
13. Freeze validator schema v4 + static contract tests.
14. Run targeted tests, full hosted suite and current-head workflows.
15. Run `.claude/skills/code-review` independently in auto-fix mode until no CRITICAL/HIGH/in-scope MEDIUM remains.
16. Run exact-source P100/CUDA schema-v4 acceptance and retain raw artifact.
17. Fresh post-evidence exact-head review. Do not merge without explicit approval.

## 15. Non-goals

- no general LP/QP package or CPU feasibility fallback;
- no universal constrained-optimization framework;
- no Gamma log-link change;
- no global rewrite of inverse-Gamma clipping semantics for FISTA/IRLS/other solvers;
- no broad sparse inverse-power Gamma CV optimization project beyond preventing wrong-link execution;
- no metric-Proximal-Newton implementation (#157);
- no Huber IRLS implementation (#156);
- no unrelated loss-capability redesign beyond private no-op/optional domain hooks used by Newton/L-BFGS;
- no performance claim.

## 16. Plan review/fix history

### Round 1 — fixed

- **HIGH / SOLVER:** blind reject-and-half could exhaust fixed Armijo budgets while a smaller feasible step exists -> add analytic domain step cap.
- **HIGH / CV:** every `init_coef` was treated as caller-owned -> distinguish direct caller starts from framework/CV starts and reseed invalid internal starts at their owner.
- **MEDIUM / NUMERICAL:** Frank-Wolfe initialization/denominator/certificate semantics under-specified -> deterministic start, explicit margin/gap, tiny-denominator handling and conservative unresolved result.
- **MEDIUM / CONSUMER:** penalized/CV reachability was conditional -> public penalized Gamma and smooth-L2 Gamma CV made blocking consumers.

### Round 2 — fixed

- **HIGH / LOSS:** `eta>0` alone does not keep current clipped value/gradient/Hessian coherent -> maintain explicit smooth training evaluations strictly inside dtype-safe `(ETA_LO, ETA_HI)` and cap both boundaries.
- **HIGH / CONSUMER:** penalized Gamma intercept init assumes log link -> suppress generic `log(mean(y))` init only for inverse-power Gamma and delegate to loss initializer.
- **HIGH / CV:** CV `"gamma"` scoring is log-link-only and some resolution loses `loss_kwargs` -> smooth-L2 inverse-power route uses actual resolved loss/link and may bypass optimized log-link-only kernels.
- **MEDIUM / TEST:** add intercept-start identity, penalized-init, inverse-link CV ranking and lower/upper step-cap regressions.

### Round 3 — fixed

- **HIGH / SOLVER:** certifying an interior start was incorrectly treated as sufficient for supported fitting. Added explicit boundary-stagnation failure: if the domain cap becomes numerically zero before gradient convergence, no successful fit may be published.
- **MEDIUM / TEST:** multi-feature behavior lacked an independent external oracle. Added statsmodels Gamma inverse-link alignment where objective/weight semantics can be matched, while preserving the analytic 1D oracle as the blocking fallback.
- **MEDIUM / CV/DOC:** training-domain guarantees were at risk of being overgeneralized to held-out/prediction rows. Clarified that strict `(L,U)` applies to Newton/L-BFGS training evaluations; CV/prediction retain current clipping semantics, with CV corrected to the actual inverse-power link.

## 17. Plan review/fix closure criteria

Each new pass restarts from the then-current exact plan and checks:

1. mathematical versus maintained numerical-domain claims;
2. value/gradient/Hessian consistency on every explicit smooth training evaluation;
3. floating-point feasibility without overclaiming exact infeasibility;
4. objective/weight/domain alignment, including zero/effectively-uniform weights;
5. intercept/no-intercept and caller/framework warm starts;
6. lower/upper step caps, Armijo and boundary-stagnation failure;
7. NumPy/CuPy/Torch dtype/device ownership and no hidden host fallback;
8. ordinary/direct/penalized/smooth-L2-CV/formula/inference closure;
9. generic-solver blast radius and Gamma-log/non-Gamma preservation;
10. analytic/geometric/external/instrumented negative tests;
11. training versus held-out/prediction claims;
12. docs/support/release wording;
13. schema-v4 evidence freshness and immutable v3 evidence;
14. bounded #150 scope versus accidental constrained-optimization/CV rewrite.

Implementation begins only after a fresh independent pass finds no new CRITICAL/HIGH or in-scope MEDIUM plan issue.
