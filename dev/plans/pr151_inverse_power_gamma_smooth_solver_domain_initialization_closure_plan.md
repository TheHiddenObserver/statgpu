# PR #151 inverse-power Gamma smooth-solver domain / initialization closure plan

Status: PLAN DRAFT / REVIEW-FIX IN PROGRESS  
Parent PR: #151  
Parent issue: #150  
Follow-up to be folded back after implementation: #152  
Branch: `fix/glm-weighted-explicit-solver-guard`  
Original plan baseline: `b5228fdc4ed78af6fcaa760ecee1475ae4ebe1a7`

## 1. Goal

Close the remaining inverse-power Gamma **explicit Newton/L-BFGS** domain/initialization gap inside PR #151 rather than retaining `fit_intercept=True` as an artificial capability condition.

The final contract is domain based, not weight-shape based:

- inverse-power Gamma has mathematical predictor domain `eta = X @ beta > 0`;
- current Gamma code clips inverse-link predictors to `_ETA_LO/_ETA_HI`; outside the interval where clipping is inactive, its value, gradient and Hessian are not derivatives of one common smooth clipped objective, so maintained explicit Newton/L-BFGS training keeps every evaluated active predictor in a certified smooth numerical interior;
- explicit Newton/L-BFGS start from a family-valid interior point and keep every accepted iterate in that interior;
- `fit_intercept=False` is no longer categorically rejected: it is supported when statgpu can certify an interior start and the requested solver converges without violating the maintained smooth-domain contract;
- inability to certify a start, or domain-boundary stagnation before solver convergence, is fail-visible and does not publish a successful fit;
- unweighted, positive-uniform, effectively-uniform and genuine non-uniform analytic weights use the same domain/initialization policy;
- weight classification preserves PR #151's historical objective contract: omitted/uniform/effectively-uniform weights execute the unweighted objective, while genuine non-uniform weights execute `sum(w_i * ell_i) / sum(w_i)`;
- genuine zero-weight rows do not constrain the training domain because they do not belong to the executed weighted objective.

Final training flow:

```text
inverse-power Gamma + explicit Newton/L-BFGS
    -> finalize backend/device/dtype and prepared analytic-weight identity
    -> certify a smooth-domain interior start on the executed training design
    -> form candidate search direction
    -> apply singular/non-descent fallback and freeze the final additive direction
    -> compute the domain step cap for that final direction
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
- solver initialization, search-direction fallback, step capping, line search, convergence and failure semantics;
- analytic `sample_weight` and zero-weight-row semantics;
- intercept/no-intercept behavior;
- NumPy/CuPy/Torch backend, device and working dtype;
- ordinary GLM, direct solver, penalized Gamma and smooth-L2 `PenalizedGLM_CV` consumers;
- formula no-intercept behavior;
- inference and failed-refit state preservation for public rows;
- tests/docs/changelog/evidence freshness;
- exact-source physical CUDA revalidation.

Not intended to change:

- `solver="auto"` dispatch policy;
- IRLS/FISTA/FISTA-BB/FISTA-LLA/ADMM numerical algorithms;
- Gamma log-link behavior;
- ordinary explicit smooth-solver `C` semantics;
- penalty definitions/scaling;
- covariance estimands;
- current prediction and held-out validation clipping semantics except that inverse-power Gamma CV uses the correct inverse-power loss rather than a log-link formula;
- numerical values of `GammaLoss._ETA_LO/_ETA_HI` unless separate pre-implementation review proves those constants themselves incorrect;
- optimized sparse Gamma CV kernels beyond preventing wrong-link execution;
- `proximal_newton_solver` domain integration: the current L2/none damped-Newton implementation is outside #150 and is tracked with the broader Proximal-Newton work in #157; it may later consume the same private loss-domain interface;
- ordered models or standalone logistic APIs.

## 3. Current inconsistencies to close

### 3.1 Original #150 support matrix versus current guard

The reviewed parent plan targeted weighted explicit Newton/L-BFGS support for all maintained ordinary GLM family/link rows and required both `fit_intercept=True` and `False`, including inverse-power Gamma.

The current runtime installer supplies a family-aware start only with an intercept, rejects the newly opened genuine-nonuniform no-intercept row, and leaves omitted/uniform/effectively-uniform no-intercept calls on the historical zero-start path. That narrowing is an implementation limitation, not a statistical prohibition.

### 3.2 Historical zero start is outside the mathematical domain

For inverse-power Gamma,

\[
\ell_i(\eta_i)=y_i\eta_i-\log\eta_i,
\qquad \eta_i>0.
\]

A no-intercept zero start gives `eta=0` and relies on clipping before the first derivative/line-search evaluation. Once PR #151 reopens numerical source and requires fresh evidence, preserving that boundary-dependent trajectory is not a useful compatibility goal.

### 3.3 `eta>0` alone is insufficient for current second-order formulas

Current inverse-Gamma value/gradient/Hessian code clips predictor values. Outside the unclipped interval, the implemented value and derivatives are not one common smooth objective. Explicit Newton/L-BFGS therefore stay inside a dtype-safe strict interior.

Let

\[
\eta_{\rm lo}=\texttt{GammaLoss.\_ETA\_LO},
\qquad
\eta_{\rm hi}=\texttt{GammaLoss.\_ETA\_HI},
\]

\[
L=\eta_{\rm lo}+\delta_\eta,
\qquad
U=\eta_{\rm hi}-\delta_\eta.
\]

Maintained explicit smooth training evaluations require

\[
L<x_i^\top\beta<U\qquad\forall i\in A.
\]

`delta_eta` is derived from the executed dtype and bound magnitudes and frozen by hosted characterization before schema-v4 physical validation. This is a numerical smooth-domain contract, not a statement that the statistical inverse-Gamma model has a finite upper domain.

### 3.4 Penalized inverse-power Gamma currently uses the wrong intercept warm start

`_PenalizedFitMixin._fit_loss_backend()` initializes all Gamma intercept paths using `log(mean(y))`, appropriate for Gamma log link but not inverse-power Gamma. The resulting non-`None` init can bypass a correct loss initializer. Inverse-power Gamma must stop using that branch; Gamma log-link initialization remains unchanged.

### 3.5 Smooth-L2 Gamma CV currently scores a log-link-only Gamma loss

`PenalizedGLM_CV` registers one optimized `"gamma"` validation formula with `mu=exp(eta)` and dispatches primarily by loss name; some branches also resolve Gamma without preserving `loss_kwargs`. An inverse-power candidate can therefore be fitted under one link and ranked under another.

For the public smooth-L2 inverse-power Gamma CV route, candidate construction, scoring, alpha selection and selected refit all resolve the same `GammaLoss(link="inverse_power")`. Use the generic actual-loss evaluator for this route instead of creating another optimized fold-batched kernel.

## 4. Prepared analytic weights and active training rows

Extract Newton/L-BFGS's duplicate weight preparation into one private helper in `statgpu/solvers/_utils.py`, e.g.

```python
_prepare_analytic_sample_weight(sample_weight, n_samples, backend, ref_arr)
```

with existing semantics:

1. validate shape/finiteness/non-negativity/positive total;
2. align to the executed design backend, concrete device and dtype;
3. apply historical floating `allclose(values, values[0])` after alignment;
4. return `None` for omitted/uniform/effectively-uniform inputs;
5. otherwise return the aligned genuine non-uniform vector.

Newton and L-BFGS both consume this helper; L-BFGS retains its loss-capability gate afterward. Preserve a compatibility alias only if required by existing internal tests/imports; duplicate classification logic must not remain.

Let the result be `w_eff`:

- `None`: all retained training rows are active;
- genuine non-uniform vector:
  \[
  A=\{i:w_{\mathrm{eff},i}>0\}.
  \]

Thus an effectively-uniform vector intentionally inherits the unweighted all-row objective/domain, while a genuine zero-weight row imposes no training-domain restriction.

## 5. Loss-owned private smooth-domain hooks

Add private no-op hooks to `LossBase` (or equivalent optional private capability) with semantics equivalent to:

```python
_loss_domain_initial_point(X, y, sample_weight=None) -> array | None
_loss_domain_is_feasible(X, coef, sample_weight=None) -> bool
_loss_domain_max_step(X, coef, delta, sample_weight=None) -> float | None
```

`delta` is additive: `beta(t)=beta+t*delta`.

Other losses default to no special start, feasible, no cap. `GammaLoss(link="inverse_power")` implements all hooks from its `_ETA_LO/_ETA_HI`; Gamma log link keeps defaults. Generic solvers contain no Gamma-name branch. Preservation tests prove non-Gamma behavior is unchanged.

## 6. Backend-native interior initialization

### 6.1 Positive-separator search

On active executed design `X_A`, seek `d` with

\[
x_i^\top d>0\qquad\forall i\in A.
\]

Normalize each finite nonzero active row by a **scale-safe backend-native row norm** to avoid overflow/underflow from naive squaring:

\[
u_i=x_i/\|x_i\|_2.
\]

Positive row scaling preserves sign feasibility. An active all-zero or non-finite row fails before optimization.

#### Candidate fast path

If one executed design column has one certified strict sign on every active row, its signed basis vector is a cheap separator candidate. The augmented intercept ones column therefore yields the pure-intercept candidate immediately.

The fast path is accepted only if §6.2 can also scale it into `(L,U)`. Otherwise continue to the geometric fallback.

#### Gilbert / Frank-Wolfe fallback

Solve

\[
\min_{c\in\operatorname{conv}(u_i)}\frac12\|c\|_2^2
\]

with deterministic backend-native iterations:

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

A positive `m_k` above the frozen dtype-aware margin certifies sign separation but does not alone terminate the search. Each certified separator is passed to §6.2. If it is positive but too poorly conditioned for `(L,U)`, continue Frank-Wolfe. Track the best band-conditioning certificate encountered for deterministic diagnostics.

Stop only on a band-certifiable separator or a reviewed numerical stopping condition/iteration limit. Tiny segment denominator before band certification, convergence near the origin, or exhaustion reports **numerically unresolved / no certified smooth-domain start**, not exact mathematical infeasibility. Exact zero rows and independently certified contradictions may use stronger messages.

No SciPy/CPU LP/QP fallback is added. CuPy/Torch geometry stays on device; only scalar certificate/control synchronization is allowed.

### 6.2 Scale a separator into the unclipped smooth band

For candidate `d`, define on original active design

\[
a_i=x_i^\top d>0,
\qquad a_{\min}=\min a_i,
\qquad a_{\max}=\max a_i.
\]

A scalar `c>0` fits the ray into `(L,U)` iff

\[
\frac{L}{a_{\min}}<c<\frac{U}{a_{\max}}.
\]

Along `beta=c d`, the unpenalized inverse-Gamma ray optimum is

\[
c_{\rm ray}=\frac{s}{\sum_{i\in A}w_i y_i a_i}
\]

or `n/sum(y_i a_i)` under the unweighted objective.

Project `c_ray` into a strict interior of the admissible interval with a frozen safety margin, then verify on original unnormalized active design that every predictor is finite and in `(L,U)`.

For the pure intercept candidate this reproduces `1/mean(y)` / `1/weighted_mean(y)` whenever already interior. If not, adjust safely rather than rely on implicit loss clipping.

Failure of every searched separator to yield a band-certifiable scale is conservative `no numerically certified smooth-domain start`, not proof that a general two-sided LP is infeasible.

Geometry tolerance, band margin, conditioning diagnostic and iteration cap are frozen before physical validation.

## 7. Search direction, domain-aware step cap, and Armijo order

Domain capping is computed from the **final direction that will actually enter line search**.

Required solver order:

1. build the Newton or L-BFGS candidate direction;
2. apply existing singular/ill-conditioned fallback where relevant;
3. compute the descent product and apply the existing non-descent/non-finite fallback to steepest descent;
4. freeze the resulting additive direction `Delta`;
5. compute the loss-domain maximum step for that `Delta`;
6. start Armijo inside that cap.

A cap computed for a direction that is subsequently replaced is invalid and is not allowed.

For feasible current `beta` and final additive `Delta`, define

\[
\eta=X_A\beta,
\qquad r=X_A\Delta.
\]

Crossing times are

\[
t_{i,\rm lo}=(\eta_i-L)/(-r_i)\quad(r_i<0),
\]

\[
t_{i,\rm hi}=(U-\eta_i)/r_i\quad(r_i>0).
\]

Take the minimum positive crossing time and apply a frozen interior safety factor. Newton's final additive direction is `Delta=-d`; L-BFGS's final additive direction is its accepted search direction after any `p=-g` fallback.

Armijo begins at `min(1,t_domain)` and retains existing halving/sufficient decrease. Feasibility is checked before every trial objective evaluation.

If gradient convergence has not occurred but the domain cap collapses below the reviewed meaningful-step floor because the iterate is pinned to `(L,U)`, surface a specific domain-boundary failure. Never publish success from a tiny coefficient change caused only by the bound. Ordinary Armijo failure for a positive interior step keeps existing warning semantics; warning-as-error acceptance prevents it from closing PR151.

## 8. Newton/L-BFGS integration and warm-start ownership

For both solvers:

1. preprocess X/y;
2. prepare analytic weights on final backend/dtype;
3. obtain or validate initial coefficients before any value/gradient/Hessian evaluation;
4. produce the final search direction after all existing fallback logic;
5. compute the domain cap for that final direction;
6. validate every trial before objective evaluation;
7. preserve Armijo constants, convergence tolerances, Newton rank fallback, L-BFGS history and ordinary warning behavior;
8. never report domain-boundary stagnation as successful convergence.

Warm-start ownership:

- direct solver `init_coef` is caller-authoritative; invalid -> precise domain error;
- ordinary GLM has no public init -> loss initializer;
- penalized/CV starts are framework-owned -> validate at owner boundary, discard if invalid, then reseed through `None`;
- no generic solver flag guesses ownership.

## 9. Estimator and CV integration

### 9.1 Ordinary GLM

Simplify `statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`:

- remove genuine-nonuniform no-intercept guard;
- remove duplicate inverse-Gamma start after shared initializer proves equivalent supported behavior;
- retain finalized `X_work`, working dtype, explicit solver identity/provenance and runtime-installer contracts.

The canonical `_glm_base.py` historical weighted rejection remains masked by the installed runtime contract under the already-reviewed PR151 installer architecture; this closure does not require an unrelated rewrite of that installer mechanism.

### 9.2 Penalized smooth Gamma

In `_PenalizedFitMixin._fit_loss_backend()`:

- preserve log-link Gamma `log(mean(y))` initialization;
- inverse-power Gamma does not create that log-link init;
- absent valid framework warm start, pass `None` so shared loss initialization sees finalized `X_work`, prepared weights and backend/dtype;
- validate/reseed framework starts;
- preserve selective penalty/intercept non-penalization.

### 9.3 PenalizedGLM_CV smooth-L2 inverse-power Gamma

For `PenalizedGLM_CV(loss="gamma", loss_kwargs={"link":"inverse_power"}, penalty="l2", ...)`:

- preserve `loss_kwargs` in candidate construction and every L2 loss-resolution branch;
- validate/reseed internal fold/path starts;
- bypass log-link-only `_val_gamma` and score with the actual resolved inverse-power Gamma loss;
- use the same link/objective family for alpha selection and selected refit;
- do not convert domain/contract failures to MSE/log-link fallback.

Strict `(L,U)` applies to Newton/L-BFGS **training evaluations**. Held-out validation and public prediction keep existing clipping semantics; this PR fixes their link identity, not unseen-covariate domain guarantees.

No new optimized inverse-power sparse/fold-batched Gamma kernel is introduced. Wrong-link optimized branches must be bypassed or explicitly unsupported for inverse-power.

### 9.4 Formula and inference

Formula fits use finalized Patsy-retained design and aligned weights, including no-intercept formulas. Existing inference consumes only successfully fitted parameters and preserves covariance/estimand/reference-distribution semantics.

### 9.5 Failure transactions and stale state

New initialization/domain failures are part of the fit transaction. For ordinary, penalized and CV consumers:

- characterize current failed-refit semantics before implementation;
- pre-fit domain-certification failure, unrecoverable internal warm-start/domain failure, or solver domain-boundary failure follows the same maintained transaction policy;
- never publish attempted solver/backend/domain provenance as successful execution;
- never leave new inference/CV-selected-alpha state partially updated;
- add previously-fitted -> failing-refit regressions for ordinary and penalized Gamma and a proportional CV failure-transaction regression.

Do not silently redefine existing transaction policy merely because this closure adds a new failure class.

## 10. Blocking consumer graph

1. ordinary inverse-power `GammaRegression` and generic equivalent;
2. direct `newton_solver` / `lbfgs_solver` with inverse-power `GammaLoss`;
3. `PenalizedGammaRegression(link="inverse_power")`;
4. generic penalized inverse-power Gamma smooth L2/no-penalty;
5. inverse-power Gamma smooth-L2 `PenalizedGLM_CV` scoring/selection/final refit;
6. ordinary formula routes;
7. already-supported inference reached by these fits;
8. failed-refit/provenance state for affected estimators.

Before coding, inventory every affected `init_coef` call site and classify caller/framework ownership.

Preservation: Gamma log link, all non-Gamma Newton/L-BFGS consumers, IRLS/FISTA/etc., non-GLM weighted-L-BFGS boundaries, ordered models, standalone logistic. `proximal_newton_solver` is explicitly not a PR151 consumer.

## 11. Hosted validation plan

### 11.1 Objective/derivative consistency

Inside `(L,U)` verify analytic/finite-difference agreement for

\[
\ell=y\eta-\log\eta,
\quad \ell'=y-1/\eta,
\quad \ell''=1/\eta^2,
\]

plus weighted normalization. Instrument explicit smooth solvers so no active training value/gradient/Hessian evaluation occurs outside `(L,U)`.

### 11.2 Geometric/analytic fixtures

1. 1D exact feasible reference:
   \[
   \beta^*=\left(\frac{\sum_iw_i y_i x_i}{\sum_iw_i}\right)^{-1};
   \]
2. multi-feature half-space requiring column combination;
3. active `x,-x` contradiction;
4. active zero row;
5. zero-weight contradiction matching row deletion;
6. near-boundary/unresolved geometry;
7. basis separator unable to fit `(L,U)` but later geometric separator succeeds;
8. early positive FW separator with bad ratio followed by later band-certifiable separator;
9. no searched ray band-certifiable -> unresolved.

Successful cases assert coefficient error, gradient/KKT residual, predictor bounds and weight-rescaling invariance.

### 11.3 Independent external baseline

For a well-conditioned CPU fixture with optimum comfortably inside the band, align statgpu with statsmodels Gamma inverse-power (or maintained equivalent) for no-intercept unweighted and, where exact semantics can be matched, genuine weighted fitting. Record objective/weight mapping. If no exact weighted mapping exists, 1D analytic results remain blocking and external weighted comparison is advisory only.

### 11.4 Initialization regression

- intercept start equals `1/mean(y)` / `1/weighted_mean(y)` when interior;
- omitted/uniform/effectively-uniform share classification;
- no-intercept ordinary never starts at zero;
- penalized inverse-power Gamma does not use `log(mean(y))`;
- Gamma log-link init unchanged;
- direct invalid explicit init fails before loss evaluation;
- invalid framework start reseeds.

### 11.5 Domain-aware direction/line-search/failure

- singular/ill-conditioned Newton fallback computes cap from fallback direction;
- Newton non-descent fallback computes cap from `Delta=-grad`, not the rejected Newton direction;
- L-BFGS non-descent fallback computes cap from `p=-grad`, not the rejected quasi-Newton direction;
- lower and upper crossings cap `t_0`;
- feasible objective-increasing trial still fails Armijo;
- no infeasible trial reaches evaluator;
- boundary stagnation cannot publish successful convergence;
- existing 20/25 budgets and other warning semantics unchanged.

### 11.6 Backend/dtype/weights

- NumPy/Torch-CPU deterministic parity;
- mixed Torch input uses finalized promoted dtype;
- shared weight prep preserves uniformity classification;
- scale-safe row normalization and geometry/domain operations stay backend-native;
- no full X/weight host copy;
- heterogeneous-container tests prove explicit `device` remains authoritative for inverse-Gamma initialization/domain work;
- CuPy/Torch CUDA parity is a physical gate.

### 11.7 Public/CV/formula/inference/failure closure

- ordinary intercept/no-intercept and generic equivalent;
- formula intercept/no-intercept + missing-row weights;
- penalized inverse-power Gamma smooth L2/no-penalty;
- smooth-L2 inverse-power CV correct-link scoring, alpha selection and final refit;
- CV fixture where inverse/log-link alpha ranking differs;
- CV domain errors not converted to unrelated fallback;
- selected-alpha identity across maintained backends;
- supported inference smoke/parity;
- failed-refit state/provenance/inference/CV-selection regression;
- clone/get_params/introspection unchanged.

### 11.8 Preservation

Gamma log-link Newton/L-BFGS/CV; non-Gamma weighted Newton/L-BFGS; IRLS/FISTA/auto; runtime installer idempotence/import-order/provenance.

## 12. Documentation and issue handling

After implementation proof, update affected EN/CN Gamma/GLM pages, solver/support matrix, explicit Newton/L-BFGS algorithm/domain note, smooth-L2 CV text, root/EN/CN changelog and PR151 body.

Public wording:

> inverse-power Gamma mathematically requires a positive linear predictor. Explicit Newton/L-BFGS keep training evaluations inside statgpu's maintained smooth numerical interior so value, gradient and Hessian correspond to one coherent objective. An intercept gives an immediate feasible direction; no-intercept designs are supported when statgpu can certify an interior start and the optimizer converges without hitting the maintained numerical-domain boundary.

Do not imply the training interior for unseen prediction rows. Do not generalize this PR151 claim to `proximal_newton_solver`; #157 owns the broader Proximal-Newton route.

Issue #152 remains open until implementation, current-head hosted review and schema-v4 physical acceptance complete; then close it as completed by PR151.

## 13. Physical CUDA evidence: schema v4

Any numerical change after `c6781cb6` makes schema-v3 P100 evidence historical for the final implementation. Keep retained v3 JSON immutable.

Before the new run, freeze schema **v4** with:

- existing v3 tolerances/warning gates unless principled pre-run review changes them;
- feasible no-intercept inverse-power Gamma × Newton/L-BFGS × NumPy/CuPy/Torch;
- intercept inverse-power preservation;
- omitted/uniform/genuine non-uniform behavior;
- weight-rescaling invariance;
- min/max active training predictor inside frozen `(L,U)`;
- domain-step-cap characterization using the final post-fallback search direction;
- truthful solver/backend/device provenance;
- at least one heterogeneous-container inverse-Gamma route for each explicit solver or equivalent cross-container matrix proving the domain initializer follows executed backend/device;
- **GPU negative-domain pair on CuPy and Torch:** active contradictory row -> no-intercept fit fails closed; same row with genuine zero analytic weight -> fit succeeds and matches dropping that row;
- penalized inverse-power Gamma L2 direct fit on NumPy/CuPy/Torch;
- smooth-L2 inverse-power Gamma CV with coefficient/intercept parity **and exact selected-alpha identity** versus NumPy on CuPy/Torch;
- existing v3 ordinary/cross-container/shared-consumer coverage unless schema-v4 review explicitly justifies replacement.

The GPU negative pair verifies error classification/provenance without CPU fallback; the zero-weight rescue proves masking uses the prepared backend-native weights.

Raw v4 evidence records exact source SHA, clean source, environment and status and is retained outside benchmark-source scan roots. No post-failure threshold/domain-margin loosening without a new reviewed schema.

## 14. Implementation order

1. Finish independent plan review/fix loop.
2. Inventory affected init owners, failed-refit state ownership and inverse-power L2 CV scoring branches.
3. Extract shared analytic-weight preparation + preservation tests.
4. Add private domain hooks and inverse-power Gamma `(L,U)` contract.
5. Add backend-native separator/initializer + analytic/geometric tests.
6. Integrate Newton, including post-fallback direction capping.
7. Integrate L-BFGS, including post-fallback direction capping.
8. Remove ordinary wrapper guard/duplicate init.
9. Repair penalized inverse-Gamma init/warm-start/failure transaction.
10. Repair smooth-L2 inverse-Gamma CV link resolution/scoring/refit/failure transaction.
11. Close ordinary/formula/inference/external-reference/preservation tests.
12. Update EN/CN docs/changelogs/support matrices.
13. Freeze schema-v4 validator + static contract tests.
14. Run targeted/full hosted validation.
15. Run `.claude/skills/code-review` independently in auto-fix mode until no CRITICAL/HIGH/in-scope MEDIUM remains.
16. Run exact-source P100/CUDA schema-v4 acceptance and retain artifact.
17. Fresh post-evidence exact-head review. Do not merge without explicit approval.

## 15. Non-goals

- no general LP/QP/CPU feasibility fallback;
- no universal constrained optimizer;
- no Gamma log-link change;
- no global clipping rewrite for FISTA/IRLS/other solvers;
- no broad sparse inverse-power Gamma CV optimization project beyond preventing wrong-link execution;
- no `proximal_newton_solver` domain integration in PR151; #157 may reuse the domain hooks when that solver is redesigned;
- no Huber IRLS implementation (#156);
- no unrelated loss-capability redesign beyond private no-op/optional domain hooks used by Newton/L-BFGS;
- no performance claim.

## 16. Plan review/fix history

### Round 1 — fixed

- **HIGH / SOLVER:** blind halving could exhaust Armijo budget despite feasible smaller step -> analytic domain step cap.
- **HIGH / CV:** all `init_coef` treated caller-owned -> distinguish caller/framework starts and reseed invalid internal starts.
- **MEDIUM / NUMERICAL:** Frank-Wolfe initialization/denominator/certification under-specified -> deterministic start, margin/gap, tiny-denominator and conservative unresolved semantics.
- **MEDIUM / CONSUMER:** penalized/CV reachability conditional -> public penalized Gamma and smooth-L2 Gamma CV made blocking consumers.

### Round 2 — fixed

- **HIGH / LOSS:** `eta>0` did not keep clipped value/gradient/Hessian coherent -> strict dtype-safe `(ETA_LO, ETA_HI)` explicit-smooth training interior and two-sided step cap.
- **HIGH / CONSUMER:** penalized Gamma intercept init assumed log link -> inverse-power bypasses `log(mean(y))` and delegates to loss initializer.
- **HIGH / CV:** CV Gamma scoring hard-coded log link / lost `loss_kwargs` -> smooth-L2 inverse-power uses actual resolved loss/link and bypasses log-link-only evaluator.
- **MEDIUM / TEST:** added intercept-start identity, penalized-init, inverse-link CV ranking, lower/upper step-cap tests.

### Round 3 — fixed

- **HIGH / SOLVER:** interior start certification was incorrectly sufficient for support -> explicit boundary-stagnation failure before successful publication.
- **MEDIUM / TEST:** no independent multi-feature external oracle -> statsmodels aligned subset added, analytic 1D remains blocking.
- **MEDIUM / CV/DOC:** training-domain guarantee risked being generalized to held-out/prediction rows -> explicitly separated training interior from current clipped validation/prediction semantics.

### Round 4 — fixed

- **MEDIUM / NUMERICAL:** first positive/basis separator could have predictor spread too large for `(L,U)` although another direction is usable -> fast path falls through when not band-scalable, FW continues until band-certified/stopped, and tests cover it.
- **MEDIUM / NUMERICAL:** naive row norms could overflow/underflow -> scale-safe backend-native row norm required.

### Round 5 — fixed

- **MEDIUM / API:** new domain failures lacked an explicit failed-refit transaction contract -> ordinary/penalized/CV state, inference, selection and provenance preservation/invalidation are blocking characterization/regression targets.
- **MEDIUM / BACKEND/ARTIFACT:** schema-v4 coverage did not explicitly require inverse-Gamma selected-alpha identity or cross-container domain routing -> both are physical gates.
- **MEDIUM / SCOPE/DOC:** “smooth-solver closure” could be read as including `proximal_newton_solver` -> explicit Newton/L-BFGS scope and #157 ownership recorded.

### Round 6 — fixed

- **MEDIUM / BACKEND/ARTIFACT:** feasible GPU routes alone did not prove backend-native negative-domain/active-mask semantics -> schema v4 includes CuPy/Torch contradictory-design failure plus zero-weight rescue matching row deletion.

### Round 7 — fixed

- **MEDIUM / SOLVER:** domain cap could be computed before Newton/L-BFGS replaced a singular/non-descent candidate direction -> required ordering now freezes the final post-fallback additive direction first and tests cap correctness after each fallback type.

## 17. Plan review/fix closure criteria

Each new pass restarts from the then-current exact plan and checks:

1. mathematical versus maintained numerical-domain claims;
2. value/gradient/Hessian consistency on every explicit smooth training evaluation;
3. floating-point feasibility without overclaiming exact infeasibility;
4. objective/weight/domain alignment, including zero/effectively-uniform weights;
5. intercept/no-intercept and caller/framework warm starts;
6. separator search plus band-scalability rather than sign-only feasibility;
7. final-direction ordering, lower/upper step caps, Armijo and boundary-stagnation failure;
8. NumPy/CuPy/Torch dtype/device ownership, active-mask semantics and no hidden host fallback;
9. ordinary/direct/penalized/smooth-L2-CV/formula/inference/failure-state closure;
10. generic-solver blast radius and Gamma-log/non-Gamma preservation;
11. analytic/geometric/external/instrumented negative tests;
12. training versus held-out/prediction claims;
13. docs/support/release wording and Proximal-Newton scope separation;
14. schema-v4 exact-source evidence freshness, selected-alpha/backend identity, GPU negative semantics and immutable v3 evidence;
15. bounded #150 scope versus accidental constrained-optimization/CV rewrite.

Implementation begins only after a fresh independent pass finds no new CRITICAL/HIGH or in-scope MEDIUM plan issue.
