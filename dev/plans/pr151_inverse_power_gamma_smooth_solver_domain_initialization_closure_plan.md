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
- current prediction and held-out validation clipping semantics except that inverse-power Gamma CV uses the correct inverse-power loss rather than a log-link formula;
- numerical values of `GammaLoss._ETA_LO/_ETA_HI` unless separate pre-implementation review proves those constants themselves incorrect;
- optimized sparse Gamma CV kernels beyond preventing wrong-link execution;
- ordered models or standalone logistic APIs.

## 3. Current inconsistencies to close

### 3.1 Original #150 support matrix versus current guard

The reviewed parent plan targeted weighted explicit Newton/L-BFGS support for all maintained ordinary GLM family/link rows and required both `fit_intercept=True` and `False`, including inverse-power Gamma.

The current runtime installer instead supplies a family-aware start only with an intercept, rejects the newly opened genuine-nonuniform no-intercept row, and leaves omitted/uniform/effectively-uniform no-intercept calls on the historical zero-start path. That narrowing is an implementation limitation, not a statistical prohibition.

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

Normalize each finite nonzero row by a **scale-safe backend-native row norm** (avoid overflow/underflow from naive squaring at extreme magnitude):

\[
u_i=x_i/\|x_i\|_2.
\]

Positive row scaling preserves sign feasibility. An active all-zero or non-finite row fails before optimization.

#### Candidate fast path

If one executed design column has one certified strict sign on every active row, its signed basis vector is a cheap separator candidate. The augmented intercept ones column therefore yields the usual pure-intercept candidate immediately.

**The fast path is accepted only if §6.2 can also scale this candidate into `(L,U)`.** If the basis direction is positive but too poorly conditioned to fit the smooth band, do not fail yet; continue to the geometric fallback.

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

A positive `m_k` above the frozen dtype-aware margin certifies **sign separation**, but does not by itself terminate the search. Each certified separator is passed to §6.2:

- if it admits a stable `(L,U)` scale, initialization succeeds;
- if it is positive but its predictor spread is too large for the maintained smooth band, continue Frank-Wolfe and look for a better-conditioned separator rather than incorrectly declaring the design unsupported at the first positive iterate.

Track the best band-conditioning certificate encountered (e.g. predictor ratio / admissible scale interval width) for deterministic diagnostics. Stop only on a band-certifiable separator or on a reviewed numerical stopping condition/iteration limit.

Tiny segment denominator before band certification, convergence near the origin, or exhaustion reports **numerically unresolved / no certified smooth-domain start**, not exact mathematical infeasibility. Exact all-zero rows and other independently certified contradictions may use stronger messages.

No SciPy/CPU LP/QP fallback is added. CuPy/Torch geometry stays on device; only scalar certificate/control synchronization is allowed.

### 6.2 Scale a separator into the unclipped smooth band

For candidate direction `d`, define on the original active design

\[
a_i=x_i^\top d>0,
\qquad a_{\min}=\min a_i,
\qquad a_{\max}=\max a_i.
\]

A positive scalar `c` fits the ray into `(L,U)` iff

\[
\frac{L}{a_{\min}}<c<\frac{U}{a_{\max}}.
\]

Along `beta=c d`, the unpenalized inverse-Gamma ray optimum is

\[
c_{\rm ray}=\frac{s}{\sum_{i\in A}w_i y_i a_i}
\]

or `n/sum(y_i a_i)` under the unweighted objective.

Project `c_ray` into a strict interior of the admissible scale interval using a frozen safety margin, then verify on the original unnormalized active design that all predictors are finite and in `(L,U)`.

For the pure intercept candidate this reproduces `1/mean(y)` or `1/weighted_mean(y)` whenever already interior. If not, it is safely adjusted rather than clipped implicitly by the loss.

Failure of every searched separator to yield a band-certifiable scale is conservative `no numerically certified smooth-domain start`; it is not proof that a general two-sided LP has no solution.

Geometry tolerance, band margin, conditioning diagnostic and iteration cap are frozen before physical validation.

## 7. Domain-aware maximum step

For feasible current `beta` and additive direction `Delta`, define

\[
\eta=X_A\beta,
\qquad r=X_A\Delta.
\]

The lower/upper crossing times are

\[
t_{i,\rm lo}=(\eta_i-L)/(-r_i)\quad(r_i<0),
\]

\[
t_{i,\rm hi}=(U-\eta_i)/r_i\quad(r_i>0).
\]

Take the minimum positive crossing time and apply a frozen interior safety factor. Newton passes `Delta=-d`; L-BFGS passes its additive search direction. Armijo begins at

\[
t_0=\min(1,t_{\rm domain})
\]

and then uses existing halving/sufficient decrease. A feasibility hook runs before every trial objective evaluation.

If the gradient is not converged but the domain cap has collapsed below the reviewed meaningful-step floor because the iterate is pinned to `(L,U)`, surface a specific domain-boundary failure. Do not publish a successful fit merely because parameter change became tiny. Ordinary Armijo failure for a positive interior step keeps existing warning semantics; warning-as-error PR151 acceptance prevents it from being accepted as closure.

## 8. Newton/L-BFGS integration and warm-start ownership

For both solvers:

1. preprocess X/y;
2. prepare analytic weights on final backend/dtype;
3. obtain or validate initial coefficients before any value/gradient/Hessian evaluation;
4. compute domain step cap before Armijo;
5. validate every trial before objective evaluation;
6. preserve Armijo constants, convergence tolerances, Newton rank fallback, L-BFGS history and ordinary warning behavior;
7. never report domain-boundary stagnation as successful convergence.

Warm-start ownership:

- direct exported solver `init_coef` is caller-authoritative; invalid -> precise domain error;
- ordinary GLM has no public init control -> loss initializer;
- penalized/CV starts are framework-owned -> validate at owner boundary, discard if invalid for current design, then pass `None` for fresh family initialization;
- no generic flag guesses ownership.

## 9. Estimator and CV integration

### 9.1 Ordinary GLM

Simplify `statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`:

- remove genuine-nonuniform no-intercept categorical guard;
- remove duplicate inverse-Gamma initializer after shared initializer proves equivalent supported behavior;
- retain finalized `X_work`, working dtype, explicit solver identity/provenance and runtime-installer contracts.

### 9.2 Penalized smooth Gamma

In `_PenalizedFitMixin._fit_loss_backend()`:

- preserve log-link Gamma `log(mean(y))` initialization;
- for resolved inverse-power Gamma, do not create that log-link init;
- absent valid framework warm start, pass `None` so the shared loss initializer sees finalized `X_work`, prepared weights and backend/dtype;
- validate/reseed framework-owned starts;
- retain selective penalty/intercept non-penalization.

`PenalizedGammaRegression(link="inverse_power")` and generic equivalent are blocking smooth L2/no-penalty consumers.

### 9.3 PenalizedGLM_CV smooth-L2 inverse-power Gamma

For public `PenalizedGLM_CV(loss="gamma", loss_kwargs={"link":"inverse_power"}, penalty="l2", ...)`:

- candidate construction preserves `loss_kwargs`;
- candidate/fold internal warm starts use framework-owned validation/reseed;
- validation scoring bypasses log-link-only `_val_gamma` and calls the actual resolved inverse-power Gamma loss;
- every loss-resolution branch on this L2 route preserves `loss_kwargs`;
- alpha selection and selected full-data refit use the same link/objective family;
- domain/contract failures are not converted to MSE/log-link fallback.

Strict `(L,U)` applies to Newton/L-BFGS **training evaluations**. Held-out validation scoring and public prediction retain current clipping semantics; the CV correction here is use of the correct inverse-power link. Do not claim training-domain guarantees for unseen covariates.

No new optimized inverse-power sparse/fold-batched Gamma kernel is built. Any log-link-only optimized branch must be bypassed or explicitly unsupported for inverse-power rather than silently used.

### 9.4 Formula and inference

Formula fits use finalized Patsy-retained numerical design and aligned weights, including no-intercept formulas. Already-supported inference receives a successfully fitted parameter vector inside the training smooth domain; covariance/estimand/reference-distribution semantics remain unchanged.

## 10. Blocking consumer graph

1. ordinary inverse-power `GammaRegression` and generic equivalent;
2. direct `newton_solver` / `lbfgs_solver` with inverse-power `GammaLoss`;
3. `PenalizedGammaRegression(link="inverse_power")`;
4. generic penalized inverse-power Gamma smooth L2/no-penalty;
5. inverse-power Gamma smooth-L2 `PenalizedGLM_CV` scoring/selection/final refit;
6. ordinary formula routes;
7. already-supported inference reached by these fits.

Before coding, inventory every affected `init_coef` call site and classify caller/framework ownership.

Preservation: Gamma log link, all non-Gamma Newton/L-BFGS consumers, IRLS/FISTA/etc., non-GLM weighted-L-BFGS boundaries, ordered models, standalone logistic.

## 11. Hosted validation plan

### 11.1 Objective/derivative consistency

Inside `(L,U)` verify analytic and finite-difference agreement for

\[
\ell=y\eta-\log\eta,
\quad
\ell'=y-1/\eta,
\quad
\ell''=1/\eta^2,
\]

plus weighted normalization. Instrument explicit smooth solvers so no active training value/gradient/Hessian evaluation occurs outside `(L,U)`.

### 11.2 Geometric/analytic fixtures

1. 1D exact feasible reference:
   \[
   \beta^*=\left(\frac{\sum_iw_i y_i x_i}{\sum_iw_i}\right)^{-1};
   \]
2. multi-feature half-space requiring column combination / geometric fallback;
3. active `x,-x` contradiction;
4. active zero row;
5. zero-weight contradictory row matching row deletion;
6. near-boundary/unresolved geometry;
7. basis separator that cannot fit `(L,U)` but a later Frank-Wolfe separator can — proves fast-path fallthrough;
8. early positive Frank-Wolfe separator with poor predictor ratio followed by a later band-certifiable separator — proves search does not stop at first positive margin;
9. no searched ray band-certifiable — conservative unresolved result.

Successful cases assert coefficient error, gradient/KKT residual, predictor bounds and global weight-rescaling invariance.

### 11.3 Independent external baseline

For a well-conditioned CPU fixture with optimum comfortably inside the band, align statgpu with statsmodels Gamma inverse-power (or maintained equivalent) for no-intercept unweighted and, when exact weight semantics can be matched, genuine weighted fitting. Document objective/weight mapping. If no exact weighted mapping exists, the analytic oracle remains blocking and external weighted comparison is advisory only.

### 11.4 Initialization regression

- intercept start equals `1/mean(y)` / `1/weighted_mean(y)` when already interior;
- omitted/uniform/effectively-uniform share classification;
- no-intercept ordinary never starts at zero;
- penalized inverse-power Gamma does not use `log(mean(y))`;
- Gamma log-link init unchanged;
- direct invalid explicit init fails before loss evaluation;
- invalid framework start reseeds.

### 11.5 Domain-aware line search/failure

- lower crossing caps `t_0`;
- upper crossing caps `t_0`;
- feasible objective-increasing trial still fails Armijo;
- no infeasible trial reaches evaluator;
- artificial boundary stagnation cannot publish successful convergence;
- existing 20/25 trial budgets and other warning semantics unchanged.

### 11.6 Backend/dtype/weights

- NumPy/Torch-CPU deterministic parity;
- mixed Torch input uses finalized promoted dtype;
- shared weight prep preserves existing uniformity classification;
- scale-safe row normalization and geometry/domain operations stay backend-native;
- no full X/weight host copy;
- CuPy/Torch CUDA parity is a physical gate.

### 11.7 Public/CV/formula/inference closure

- ordinary Gamma intercept/no-intercept and generic equivalent;
- formula intercept/no-intercept + missing-row weight alignment;
- penalized inverse-power Gamma smooth L2/no-penalty;
- smooth-L2 inverse-power CV correct-link candidate scoring, alpha selection and final refit;
- CV fixture where inverse/log link alpha ranking differs;
- CV domain errors not converted to MSE/log-link fallback;
- supported inference smoke/parity;
- clone/get_params/introspection unchanged.

### 11.8 Preservation

Gamma log-link Newton/L-BFGS/CV; non-Gamma weighted Newton/L-BFGS; IRLS/FISTA/auto; runtime installer idempotence/import-order/provenance.

## 12. Documentation and issue handling

After implementation proof, update affected EN/CN Gamma/GLM pages, solver/support matrix, algorithm/domain note, smooth-L2 CV text, root/EN/CN changelog and PR151 body.

Public wording:

> inverse-power Gamma mathematically requires a positive linear predictor. Explicit Newton/L-BFGS keep training evaluations inside statgpu's maintained smooth numerical interior so value, gradient and Hessian correspond to one coherent objective. An intercept gives an immediate feasible direction; no-intercept designs are supported when statgpu can certify an interior start and the optimizer converges without hitting the maintained numerical-domain boundary.

Do not imply the training interior is guaranteed on unseen prediction rows; prediction retains current clipping semantics.

Issue #152 remains open until implementation, current-head hosted review and schema-v4 physical acceptance complete; then close it as completed by PR151.

## 13. Physical CUDA evidence: schema v4

Any numerical change after `c6781cb6` makes schema-v3 P100 evidence historical for the final implementation. Keep the retained v3 JSON immutable.

Before the new run, freeze schema **v4** with:

- existing v3 tolerances/warning gates unless principled pre-run review changes them;
- feasible no-intercept inverse-power Gamma × Newton/L-BFGS × NumPy/CuPy/Torch;
- intercept inverse-power preservation;
- omitted/uniform/genuine non-uniform behavior;
- weight-rescaling invariance;
- min/max active training predictor inside frozen `(L,U)`;
- domain-step-cap characterization;
- truthful solver/backend/device provenance;
- penalized inverse-power Gamma L2 direct fit and smooth-L2 CV selected refit on NumPy/CuPy/Torch;
- existing v3 ordinary/cross-container/shared-consumer coverage unless schema-v4 review explicitly justifies replacement.

Raw v4 evidence records exact source SHA, clean source, environment and status and is retained outside benchmark-source scan roots. No post-failure threshold/domain-margin loosening without new reviewed schema.

## 14. Implementation order

1. Finish independent plan review/fix loop.
2. Inventory all affected init owners and inverse-power L2 CV scoring branches.
3. Extract shared analytic-weight preparation + preservation tests.
4. Add private domain hooks and inverse-power Gamma `(L,U)` contract.
5. Add backend-native separator/initializer + analytic/geometric tests.
6. Integrate Newton.
7. Integrate L-BFGS.
8. Remove ordinary wrapper guard/duplicate init.
9. Repair penalized inverse-Gamma internal init/warm-start handling.
10. Repair smooth-L2 inverse-Gamma CV link resolution/scoring/refit.
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
- no metric-Proximal-Newton implementation (#157);
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

- **MEDIUM / NUMERICAL:** first positive/basis separator could have predictor spread too large for `(L,U)` although another direction is usable -> fast path now falls through when not band-scalable, Frank-Wolfe continues after positive margin until a band-certifiable separator or reviewed stopping condition, and tests cover both cases.
- **MEDIUM / NUMERICAL:** row normalization could overflow/underflow if implemented as naive `sqrt(sum(x^2))` -> require scale-safe backend-native row norms.

## 17. Plan review/fix closure criteria

Each new pass restarts from the then-current exact plan and checks:

1. mathematical versus maintained numerical-domain claims;
2. value/gradient/Hessian consistency on every explicit smooth training evaluation;
3. floating-point feasibility without overclaiming exact infeasibility;
4. objective/weight/domain alignment, including zero/effectively-uniform weights;
5. intercept/no-intercept and caller/framework warm starts;
6. separator search plus band-scalability rather than sign-only feasibility;
7. lower/upper step caps, Armijo and boundary-stagnation failure;
8. NumPy/CuPy/Torch dtype/device ownership and no hidden host fallback;
9. ordinary/direct/penalized/smooth-L2-CV/formula/inference closure;
10. generic-solver blast radius and Gamma-log/non-Gamma preservation;
11. analytic/geometric/external/instrumented negative tests;
12. training versus held-out/prediction claims;
13. docs/support/release wording;
14. schema-v4 evidence freshness and immutable v3 evidence;
15. bounded #150 scope versus accidental constrained-optimization/CV rewrite.

Implementation begins only after a fresh independent pass finds no new CRITICAL/HIGH or in-scope MEDIUM plan issue.
