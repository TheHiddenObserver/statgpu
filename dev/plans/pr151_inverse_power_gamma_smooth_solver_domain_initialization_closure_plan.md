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
- the current implementation also has clipping bounds `_ETA_LO` / `_ETA_HI`; because the current clipped value, gradient and Hessian are not derivatives of one common smooth clipped objective outside that interval, the maintained explicit Newton/L-BFGS route must keep every training evaluation in a certified **smooth numerical interior** where clipping is inactive;
- explicit Newton/L-BFGS start from a family-valid interior point and keep every accepted iterate in that smooth numerical interior;
- `fit_intercept=False` is supported whenever the executed design admits an interior point that the maintained backend-native initializer can certify; designs that are exactly impossible or numerically unresolved fail before optimization with a precise domain error;
- unweighted, positive-uniform, effectively-uniform, and genuine non-uniform analytic weights use the same domain/initialization policy;
- weight classification still preserves PR #151's historical objective contract: omitted/uniform/effectively-uniform weights execute the unweighted objective, while genuine non-uniform weights execute `sum(w_i * ell_i) / sum(w_i)`;
- zero-weight rows constrain the training domain only when genuine weighting remains active after the maintained solver weight-preparation rule.

The final behavior is:

```text
inverse-power Gamma + explicit Newton/L-BFGS
    -> resolve the actual solver backend/dtype and prepared weight identity
    -> construct/validate a smooth-domain interior start on the executed design
    -> cap/backtrack every trial step so clipping is inactive on active training rows
    -> fit when an interior point is certified
    -> fail with an exact-design or numerically-unresolved domain error otherwise
```

## 2. Change classification and active review axes

Classification:

1. existing-capability reconciliation / completion of the original #150 support matrix;
2. numerical correctness repair for inverse-power Gamma initialization and globalization;
3. shared Newton/L-BFGS solver contract change through loss-owned optimization-domain hooks;
4. bounded correction of inverse-power Gamma L2 CV scoring/initialization because the public CV surface otherwise evaluates the wrong link;
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
- explicit smooth-solver `C` semantics on ordinary GLM;
- penalty definitions/scaling;
- covariance estimands;
- current prediction clipping behavior;
- the numerical values of `GammaLoss._ETA_LO/_ETA_HI` unless a pre-implementation review independently proves those constants themselves are incorrect;
- optimized sparse Gamma CV kernels beyond making sure this PR does not route the smooth-L2 inverse-power case through a log-link-only evaluator;
- ordered models or standalone logistic APIs.

## 3. Current inconsistencies to close

### 3.1 Original PR #151 contract versus current categorical guard

The reviewed parent plan targeted weighted explicit Newton/L-BFGS support for all maintained ordinary GLM family/link rows and explicitly required both `fit_intercept=True` and `False`, including inverse-power Gamma.

The current runtime contract installer instead:

- supplies a family-aware start only when an intercept is present;
- rejects only the newly opened genuine-nonuniform no-intercept row;
- leaves omitted/uniform/effectively-uniform no-intercept calls on the historical zero-start path.

That narrowing was recorded in the implementation addendum and tracked as #152, but it is an implementation limitation rather than a statistical prohibition.

### 3.2 Historical zero start relies on clipping rather than a valid inverse-Gamma interior

For the mathematical inverse-power Gamma loss,

\[
\ell_i(\eta_i)=y_i\eta_i-\log \eta_i,
\qquad \eta_i>0.
\]

A no-intercept zero start gives `eta=0`. Current `GammaLoss` clips the predictor before computing the inverse link, so the historical path can enter numerical work from a point outside the mathematical domain.

Uniform and unweighted inputs represent the same normalized statistical objective; initialization must not change discontinuously merely because a public weight vector crosses the solver's historical `allclose` uniformity threshold.

### 3.3 Current clipping makes `eta>0` alone insufficient for second-order smooth solvers

For inverse-power Gamma, current value/gradient/Hessian code uses clipped predictor values. Outside the clipping interval, the implemented value is flat with respect to the clipped coordinate while the implemented gradient/Hessian still use nonzero inverse-link derivatives. Therefore, outside the interval where clipping is inactive, value, gradient and Hessian are not derivatives of one common smooth objective.

PR #151 must not silently reinterpret those clipped formulas as a valid Newton/L-BFGS objective.

Define the explicit-smooth-solver numerical interior from the loss's actual clipping constants. Let

\[
\eta_{\rm lo}=\texttt{GammaLoss.\_ETA\_LO},
\qquad
\eta_{\rm hi}=\texttt{GammaLoss.\_ETA\_HI}.
\]

The maintained explicit smooth route requires a dtype-aware strict interior

\[
\eta_{\rm lo}+\delta_{\eta}
< x_i^\top\beta
< \eta_{\rm hi}-\delta_{\eta}
\qquad \forall i\in A,
\]

where `delta_eta` is a reviewed machine-precision safety margin computed from the executed dtype and the bound magnitudes. The exact formula/constant is frozen in hosted tests before schema-v4 physical validation; it is not tuned after a GPU failure.

This is a **numerical smooth-domain contract**, not a claim that the statistical inverse-Gamma model has a finite upper predictor bound. Prediction/FISTA/IRLS clipping semantics are not redefined here.

### 3.4 Penalized inverse-power Gamma currently constructs the wrong intercept warm start

`_PenalizedFitMixin._fit_loss_backend()` currently initializes every Gamma-like intercept route as though Gamma used a log link, using approximately

\[
\beta_{0,\mathrm{int}}=\log(\bar y).
\]

That is not a valid family-aware initialization for `GammaLoss(link="inverse_power")` and can bypass a loss-level initializer because the solver sees a non-`None` `init_coef`.

This is blocking for the shared domain closure: inverse-power Gamma must not retain that generic log-link initializer.

### 3.5 PenalizedGLM_CV currently scores `loss="gamma"` with a log-link-only validation formula

The scalar-response CV evaluator registers one optimized `"gamma"` validation function using

\[
\mu=\exp(\eta),
\qquad
\ell=y/\mu+\log\mu,
\]

and `_evaluate_loss_numpy()` dispatches by loss name without inspecting the Gamma link. Some CV branches also resolve the loss without passing `loss_kwargs`.

Therefore `PenalizedGLM_CV(loss="gamma", loss_kwargs={"link":"inverse_power"}, penalty="l2", ...)` cannot be declared closed merely because fitting is repaired: validation scoring could select alpha under the wrong objective.

For this PR's smooth-L2 inverse-power Gamma consumer, validation scoring must use the actual resolved loss/link. The bounded implementation should prefer the generic `loss_fn.value(...)` path for inverse-power Gamma rather than adding a second optimized fold-batched kernel. Log-link Gamma keeps its existing optimized evaluator.

## 4. Executed objective and active-row contract

### 4.1 Prepared analytic weights are authoritative

Move Newton/L-BFGS's duplicated weight preparation into a shared private helper in `statgpu/solvers/_utils.py`, e.g.

```python
_prepare_analytic_sample_weight(sample_weight, n_samples, backend, ref_arr)
```

with existing semantics:

1. validate shape, finiteness, non-negativity and positive total;
2. align to the **executed** design backend, concrete device and dtype;
3. apply the historical floating `allclose(values, values[0])` rule after alignment;
4. return `None` for omitted/uniform/effectively-uniform inputs;
5. return the aligned vector for genuine non-uniform weights.

Newton and L-BFGS both consume this helper. L-BFGS retains its separate loss-capability check after preparation. A thin `_prepare_newton_sample_weight` alias may remain only for internal compatibility; duplicate classification logic must not remain.

Let the result be `w_eff`:

- `w_eff is None`: every retained row belongs to the executed unweighted objective;
- otherwise active rows are

\[
A=\{i:w_{\mathrm{eff},i}>0\}.
\]

An effectively-uniform vector normalized away by the established rule therefore intentionally uses the all-row unweighted domain. A genuine zero-weight row does not constrain the training domain.

### 4.2 Intercept and no-intercept reduce to the same executed-design problem

After each estimator constructs its actual numerical `X_work` (including an augmented intercept column if applicable), the loss/solver layer sees one problem:

\[
\eta_A=X_A\beta.
\]

The initializer and domain checks operate on this executed `X_work`; they do not infer behavior from a public `fit_intercept` flag.

An augmented ones column gives an immediate feasible separator. A no-intercept design may or may not admit one.

## 5. Loss-owned domain interface

Add default private no-op hooks to `LossBase`; exact names may vary but semantics must be equivalent to:

```python
_loss_domain_initial_point(X, y, sample_weight=None) -> array | None
_loss_domain_is_feasible(X, coef, sample_weight=None) -> bool
_loss_domain_max_step(X, coef, delta, sample_weight=None) -> float | None
```

`delta` is an **additive** direction:

\[
\beta(t)=\beta+t\Delta.
\]

Default behavior for other losses:

- no special initial point;
- always domain-feasible;
- no domain step cap.

`GammaLoss(link="inverse_power")` overrides these hooks using its own `_ETA_LO/_ETA_HI`. `GammaLoss(link="log")` keeps defaults.

This keeps ownership in the loss while generic Newton/L-BFGS remain independent of `glm_core` and contain no Gamma-name branch.

## 6. Backend-native interior initialization

### 6.1 Positive-separator geometry

First find a direction `d` with

\[
x_i^\top d>0\qquad \forall i\in A.
\]

Strict positive feasibility is equivalent by Gordan's theorem to the absence of a nonzero nonnegative vector `lambda` with

\[
X_A^\top\lambda=0.
\]

Normalize nonzero active rows by positive row norms,

\[
u_i=x_i/\|x_i\|_2,
\]

which preserves sign feasibility. An active zero/non-finite row fails immediately.

Fast path: if one executed design column has one certified strict sign on every active row, use that signed basis vector. An augmented intercept ones column therefore recovers the pure-intercept direction without running the geometric fallback.

Otherwise solve

\[
\min_{c\in\operatorname{conv}(u_i)}\frac12\|c\|_2^2
\]

with deterministic backend-native Gilbert / Frank-Wolfe iterations.

Start at

\[
c_0=|A|^{-1}\sum_{i\in A}u_i.
\]

At iteration `k`, choose the first backend `argmin` row

\[
s_k=u_{j_k},\qquad j_k=\arg\min_i u_i^\top c_k,
\]

and define

\[
m_k=\min_i u_i^\top c_k,
\qquad
G_k=\|c_k\|_2^2-m_k.
\]

A positive `m_k` above a dtype-scaled certification threshold directly certifies a positive separator. Otherwise use

\[
\gamma_k=
\operatorname{clip}\left(
\frac{\|c_k\|_2^2-c_k^\top s_k}{\|c_k-s_k\|_2^2},0,1\right),
\]

\[
c_{k+1}=(1-\gamma_k)c_k+\gamma_k s_k.
\]

A tiny step denominator before certification, convergence near the origin, or iteration exhaustion reports **numerically unresolved / no certified strict interior**, not exact mathematical infeasibility. Exact contradictory fixtures and active zero rows may receive stronger deterministic errors when directly certified.

No SciPy/CPU LP/QP fallback is added. CuPy/Torch retain `X` and all geometry on their selected device; scalar convergence/certificate synchronization is permitted.

### 6.2 Scale a certified direction into the maintained smooth numerical interior

For a certified direction define

\[
a_i=x_i^\top d>0,
\qquad
a_{\min}=\min_{i\in A}a_i,
\qquad
a_{\max}=\max_{i\in A}a_i.
\]

Let the reviewed interior bounds be

\[
L=\eta_{\rm lo}+\delta_\eta,
\qquad
U=\eta_{\rm hi}-\delta_\eta.
\]

A scale `c>0` keeps that direction in the smooth band iff

\[
\frac{L}{a_{\min}}<c<\frac{U}{a_{\max}}.
\]

For the unpenalized inverse-Gamma objective along `beta=c d`, the ray-optimal scale is

\[
c_{\rm ray}
=\frac{s}{\sum_{i\in A}w_i y_i a_i}
\]

or `n / sum(y_i a_i)` for the unweighted objective.

Choose the initial scale by projecting `c_ray` into a strict interior of the admissible scale interval, using a reviewed multiplicative/absolute safety margin. Then verify on the **original unnormalized active design** that all predictors are finite and satisfy `(L, U)`.

For the pure intercept direction `a_i=1`, this reproduces `1/mean(y)` or `1/weighted_mean(y)` whenever that ray optimum already lies in the interior; if it needs clipping to the numerical band, the test/documentation records that distinction rather than claiming exact identity.

A failure of this particular certified direction to admit an `(L,U)` scale does **not** prove that no other coefficient vector fits the smooth band. The initializer must therefore report `no numerically certified smooth-domain start` unless an exact design contradiction has been established. PR #151 does not introduce a general two-sided linear-program feasibility solver.

Tolerances, interior safety factors and maximum geometry iterations are frozen by hosted analytic/geometric characterization before schema-v4 physical validation and are not relaxed post-failure.

## 7. Domain-aware step cap and Armijo globalization

For current feasible `beta` and additive direction `Delta`, define on active rows

\[
\eta=X_A\beta,
\qquad
r=X_A\Delta.
\]

To stay inside `(L,U)`:

- if `r_i<0`, lower-bound crossing occurs at
  \[
  t_{i,\rm lo}=(\eta_i-L)/(-r_i);
  \]
- if `r_i>0`, upper-bound crossing occurs at
  \[
  t_{i,\rm hi}=(U-\eta_i)/r_i.
  \]

The mathematical maximum is the minimum positive crossing time across constraining rows. The loss hook returns a reviewed interior cap slightly below that boundary. If all `r_i=0` or directions move strictly inside without a finite crossing before `t=1`, no additional cap is needed.

Newton converts its subtract-direction convention to `Delta=-d`. L-BFGS passes its existing additive search direction. Armijo starts at

\[
t_0=\min(1,t_{\rm domain})
\]

and then uses the solver's existing halving and sufficient-decrease condition.

A defensive feasibility hook runs before **every** trial objective evaluation; no explicit smooth-solver value/gradient/Hessian evaluation may receive an active predictor outside `(L,U)`.

Blind reject-and-half without the analytic cap is not accepted as the closure because Newton/L-BFGS have fixed 20/25 backtracking budgets.

## 8. Newton/L-BFGS integration and warm-start ownership

For `newton_solver` and `lbfgs_solver`:

1. preprocess X/y;
2. prepare analytic weights on the executed backend/dtype;
3. obtain or validate the initial point before any value/gradient/Hessian evaluation;
4. compute a domain cap before Armijo;
5. validate every trial before objective evaluation;
6. preserve all existing Armijo constants, convergence tolerances, Newton rank fallback, L-BFGS history rules and warning behavior.

Warm-start ownership is explicit:

- **direct exported solver call:** supplied `init_coef` is caller-authoritative; invalid domain -> precise error, never silent replacement;
- **ordinary GLM explicit smooth fit:** currently has no public `init_coef`, so inverse-power Gamma uses the loss initializer;
- **penalized/CV internal starts:** framework-owned starts must be validated at the owning estimator/fold boundary. If invalid for the current executed design, discard them and pass `None` so the loss builds a fresh interior start;
- do not add a generic solver flag that tries to infer who owns `init_coef`.

## 9. Estimator and CV integration

### 9.1 Ordinary GLM

Simplify `statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`:

- remove the genuine-nonuniform no-intercept categorical guard;
- remove installer-owned inverse-Gamma initialization after the loss/solver initializer reproduces the supported intercept behavior;
- retain actual `X_work` construction, working-dtype authority, explicit solver identity and fit provenance;
- retain runtime-installer idempotence/import-order/introspection contracts.

No family-specific domain rule remains in the wrapper.

### 9.2 Penalized smooth Gamma

In `_PenalizedFitMixin._fit_loss_backend()`:

- preserve current log-link Gamma intercept initialization unchanged;
- when the resolved loss is `GammaLoss(link="inverse_power")`, **do not** construct the generic `log(mean(y))` Gamma intercept start;
- if there is no user/framework warm start, pass `init_coef=None` and let the loss-owned initializer use the finalized `X_work`, prepared weights and actual backend/dtype;
- if a framework-owned warm start exists, validate/reseed it per §8;
- selective penalty/intercept non-penalization remains unchanged.

Both `PenalizedGammaRegression(link="inverse_power")` and the generic penalized Gamma construction are blocking consumers for maintained smooth L2/no-penalty Newton/L-BFGS paths.

### 9.3 PenalizedGLM_CV smooth-L2 inverse-power Gamma

The L2 Gamma CV route is a blocking public consumer because `PenalizedGLM_CV` exposes scalar Gamma, `loss_kwargs`, `fit_intercept`, weights and a smooth-L2 solver/refit path.

For `loss="gamma"` with resolved `link="inverse_power"` and `penalty="l2"`:

- candidate estimator construction must preserve `loss_kwargs={"link":"inverse_power"}`;
- candidate/fold internal warm starts use the framework-owned policy from §8;
- validation scoring must bypass the log-link-only `_val_gamma` optimized registry entry and evaluate the actual resolved inverse-power `GammaLoss` objective (including the current validation/prediction clipping semantics, which this PR does not redefine);
- any branch that resolves `loss_fn` for this L2 route must pass the actual `loss_kwargs` rather than reconstructing default log-link Gamma;
- selected full-data refit must preserve the same link, weight objective, solver and domain initializer;
- domain/contract errors are not converted into MSE or unrelated fallback scores.

This PR does **not** generalize the fold-batched/sparse Gamma kernels to inverse-power. If reconnaissance/tests show an optimized sparse-CV route would otherwise activate for inverse-power Gamma, either bypass that optimization for inverse-power or preserve an explicit existing unsupported boundary; do not silently execute log-link formulas. Any broader sparse inverse-power CV enhancement belongs in a separate issue unless it is already a claimed maintained capability whose correctness blocks this shared change.

### 9.4 Formula and inference

Ordinary formula fits with and without an intercept use the same finalized numerical design/domain logic after Patsy row filtering and weight alignment.

For already-supported `compute_inference=True` inverse-power Gamma fits:

- successful fitted parameters entering inference satisfy the training smooth-domain contract;
- existing covariance/estimand/reference-distribution semantics remain unchanged;
- inference does not reuse an invalid pre-fit/init state.

## 10. Consumer graph

Blocking affected consumers:

1. `GammaRegression(link="inverse_power")` and generic ordinary GLM equivalent;
2. direct exported `newton_solver` / `lbfgs_solver` with inverse-power `GammaLoss`;
3. public `PenalizedGammaRegression(link="inverse_power")`;
4. `PenalizedGeneralizedLinearModel(loss="gamma", loss_kwargs={"link":"inverse_power"})` on smooth L2/no-penalty Newton/L-BFGS;
5. `PenalizedGLM_CV(... gamma inverse_power ..., penalty="l2")` candidate scoring and selected refit;
6. ordinary formula paths;
7. already-supported inference consumers reached by these successful fits.

Before coding, inventory every affected call site that passes `init_coef` into Newton/L-BFGS and classify it as caller-owned or framework-owned. A representative wrapper is not enough.

Preservation consumers:

- Gamma log-link;
- all non-Gamma Newton/L-BFGS callers;
- explicit IRLS/FISTA/etc.;
- current non-GLM weighted-L-BFGS capability boundaries;
- ordered models and standalone logistic APIs.

## 11. Hosted test plan

### 11.1 Objective/derivative and smooth-domain contract

Add direct inverse-power Gamma tests proving, for predictors strictly inside `(L,U)`:

- value agrees with analytic \(y\eta-\log\eta\);
- gradient agrees with finite differences / analytic \(y-1/\eta\);
- Hessian agrees with finite differences / analytic \(1/\eta^2\);
- weighted normalization matches `sum(w_i contribution_i)/sum(w_i)`.

Instrument Newton/L-BFGS so no initial/current/trial value/gradient/Hessian evaluation receives an active predictor outside `(L,U)`.

### 11.2 Feasible-start geometry

Deterministic fixtures:

1. **1D feasible exact reference:** strictly positive X. For unpenalized weighted inverse Gamma,
   \[
   \beta^*=\left(\frac{\sum_i w_i y_i x_i}{\sum_i w_i}\right)^{-1}.
   \]
   Cover omitted, uniform/effectively-uniform and genuine non-uniform weights.
2. **multi-feature feasible half-space** requiring a column combination so the convex-hull fallback runs;
3. **exact contradictory design** containing active `x` and `-x`;
4. **active all-zero row**;
5. **zero-weight contradictory row**, which must match dropping that row under genuine weighting;
6. **near-boundary/unresolved design** that must report the conservative numerical-domain error rather than return a clipped fit;
7. **separator whose ray needs scale adjustment** to enter `(L,U)`;
8. **separator whose particular ray cannot be certified inside `(L,U)`**, which must be reported as unresolved, not exact infeasibility.

For successful cases check final coefficient error, gradient/KKT residual, active predictor bounds and global weight-rescaling invariance.

### 11.3 Initialization regression

- ordinary intercept inverse-power start reproduces `1/mean(y)` or `1/weighted_mean(y)` when that value lies inside the smooth band;
- omitted/uniform/effectively-uniform ordinary calls use the same initializer/objective classification;
- no-intercept ordinary route no longer starts from zero;
- penalized inverse-power Gamma intercept route no longer uses `log(mean(y))`;
- Gamma log-link penalized initialization is unchanged;
- direct explicit invalid `init_coef` fails before loss evaluation;
- framework-owned invalid warm start is discarded and reseeded.

### 11.4 Domain-aware line search

For Newton and L-BFGS:

- construct a direction for which `t=1` crosses the lower boundary and verify `t_0<1`;
- construct an upper-bound crossing and verify the upper cap;
- ensure Armijo still rejects a feasible but objective-increasing candidate;
- no infeasible trial reaches the loss evaluator;
- fixed 20/25 Armijo trial counts and existing warning semantics remain unchanged after the initial domain cap.

### 11.5 Backend/dtype and weights

- NumPy and Torch-CPU deterministic parity in hosted tests;
- mixed Torch input classification uses finalized promoted `X_work` dtype;
- shared weight preparation exactly preserves existing Newton/L-BFGS uniform/effectively-uniform classification;
- geometry and domain checks remain backend-native; no full X/weight host copy;
- physical CuPy/Torch CUDA parity is a completion gate.

### 11.6 Public/penalized/CV/formula/inference closure

Cover:

- ordinary Gamma intercept/no-intercept and generic equivalent;
- ordinary formula intercept/no-intercept plus missing-row weight alignment;
- penalized inverse-power Gamma smooth L2/no-penalty fits;
- smooth-L2 `PenalizedGLM_CV` candidates, scoring, alpha selection and final refit with `loss_kwargs={"link":"inverse_power"}`;
- a CV regression whose inverse-power alpha ranking differs from the incorrectly hard-coded log-link ranking, so a future regression cannot pass accidentally;
- domain failure propagation through CV with no MSE/log-link fallback;
- inference smoke/parity on already-supported successful rows;
- clone/get-params/introspection unchanged because no public constructor control is added.

### 11.7 Preservation

- Gamma log-link Newton/L-BFGS/CV unchanged within established tolerance;
- non-Gamma Newton/L-BFGS weighted tests unchanged;
- IRLS/FISTA/auto preservation tests green;
- runtime installer idempotence/import-order/provenance regressions green.

## 12. Documentation and issue handling

After implementation is proven, update only affected claims:

- EN/CN Gamma/GLM model pages;
- EN/CN solver support matrix and algorithm/domain note;
- relevant EN/CN CV text if inverse-power Gamma L2 scoring behavior is user-visible;
- root/EN/CN changelog PR151 entry;
- PR151 body/current implementation head/evidence status.

Public wording distinguishes mathematical and maintained numerical domains:

> inverse-power Gamma requires a positive linear predictor mathematically. Explicit Newton/L-BFGS additionally keep the training predictor inside statgpu's maintained smooth numerical interior so value, gradient and Hessian remain one coherent objective. An intercept gives an immediate feasible direction; no-intercept designs are supported when statgpu can certify an interior start on the executed design.

Issue #152 remains open during implementation. Close it as completed by PR151 only after implementation, current-head hosted validation, fresh review and schema-v4 physical acceptance are complete.

## 13. Physical CUDA evidence: schema v4

Any production numerical change after `c6781cb6` makes the schema-v3 P100 artifact historical rather than final implementation evidence. Keep the retained v3 JSON immutable.

Before a new physical run, version the validator to schema **v4** and freeze:

- existing v3 thresholds and warning-as-error rules unless a principled pre-run review changes them;
- feasible no-intercept inverse-power Gamma × Newton/L-BFGS × NumPy/CuPy/Torch;
- intercept inverse-power preservation;
- omitted/uniform/genuine non-uniform weight behavior;
- positive weight-rescaling invariance;
- minimum/maximum active training predictor staying within the frozen smooth numerical interior;
- one lower- and one upper-domain step-cap characterization where practical;
- truthful solver/backend/device provenance;
- public penalized inverse-power Gamma L2 direct fit and smooth-L2 CV selected refit on NumPy/CuPy/Torch;
- existing ordinary family/link, cross-container and shared-consumer v3 coverage unless schema-v4 review explicitly justifies a scoped replacement.

The raw v4 artifact records exact `source_sha`, clean source, environment and status and is retained outside benchmark-source scan roots. Do not loosen thresholds or domain margins after observing a failed physical result without a new reviewed schema.

## 14. Implementation order

1. Freeze this plan through repeated independent plan review/fix passes.
2. Inventory all affected Newton/L-BFGS `init_coef` owners plus the inverse-power Gamma L2 CV scoring path.
3. Extract shared analytic-weight preparation with preservation tests.
4. Add private LossBase domain hooks and inverse-power Gamma smooth-domain/step-cap hooks.
5. Add backend-native separator/initializer and analytic/geometric tests.
6. Integrate hooks into Newton and close Newton tests.
7. Integrate hooks into L-BFGS and close weighted L-BFGS tests.
8. Remove ordinary wrapper categorical guard and duplicate inverse-Gamma initializer.
9. Repair penalized inverse-power Gamma internal initialization/warm-start ownership.
10. Repair smooth-L2 inverse-power Gamma CV loss resolution/scoring and close selected-refit behavior.
11. Close ordinary/formula/inference/preservation tests.
12. Update EN/CN docs/changelogs/support matrices.
13. Version/freeze validator schema v4 and static contract tests.
14. Run targeted tests, full hosted suite and current-head workflows.
15. Run `.claude/skills/code-review` in independent auto-fix mode; repeat until no CRITICAL/HIGH or in-scope MEDIUM finding remains.
16. Run exact-source physical P100/CUDA schema-v4 acceptance and retain the raw artifact.
17. Perform a fresh post-evidence exact-head review. Do not merge without explicit approval.

## 15. Non-goals

- no general LP/QP package or CPU feasibility fallback;
- no universal constrained-optimization framework;
- no change to Gamma log link;
- no global rewrite of Gamma clipping semantics for FISTA/IRLS/other solvers;
- no broad sparse inverse-power Gamma CV optimization project beyond preventing this closure from using log-link-only formulas;
- no metric-Proximal-Newton implementation (#157);
- no Huber IRLS implementation (#156);
- no unrelated loss-capability redesign beyond private no-op domain hooks consumed by Newton/L-BFGS;
- no performance/speedup claim.

## 16. Plan review/fix history

### Round 1 — fixed

- **HIGH / SOLVER:** blind reject-and-half could exhaust fixed Armijo budgets even when a feasible smaller step exists. Added an analytic loss-owned step cap.
- **HIGH / CV:** all supplied `init_coef` were treated as caller-authoritative. Separated direct caller-owned warm starts from framework/CV-owned starts and required invalid internal starts to be reseeded at their owner.
- **MEDIUM / NUMERICAL:** convex-hull iteration lacked deterministic initialization, tiny-denominator handling and conservative certificate semantics. Added all three plus final original-design certification.
- **MEDIUM / CONSUMER:** penalized/CV consumers were conditional. Public penalized Gamma and smooth-L2 Gamma CV are now explicit blocking consumers.

### Round 2 — fixed

- **HIGH / LOSS:** the plan initially required only `eta>0`, but current clipped inverse-Gamma value/gradient/Hessian are not one common smooth objective outside the clipping interval. The explicit Newton/L-BFGS contract now stays strictly inside a dtype-safe `(ETA_LO, ETA_HI)` interior and caps both lower and upper crossings.
- **HIGH / CONSUMER:** penalized `_fit_loss_backend()` currently initializes all Gamma intercept paths with `log(mean(y))`, which is wrong for inverse-power Gamma and would bypass the loss initializer. The plan now explicitly removes that init only for inverse-power Gamma while preserving log-link behavior.
- **HIGH / CV:** `PenalizedGLM_CV`'s optimized `"gamma"` validation formula is hard-coded to the log link and some loss resolution drops `loss_kwargs`. The smooth-L2 inverse-power route must use the actual resolved loss/link for scoring and selected refit; it may bypass log-link-only optimization instead of broadening this PR into a new optimized CV kernel.
- **MEDIUM / TEST:** added explicit regression for the existing `1/mean(y)` intercept start when interior, penalized-log-init removal, inverse-link-specific CV alpha ranking, and both lower/upper domain step caps.

## 17. Plan review/fix closure criteria

Every new pass restarts from the then-current exact plan and checks:

1. mathematical versus maintained numerical-domain claims;
2. value/gradient/Hessian consistency on every explicit smooth-solver evaluation;
3. floating-point feasibility/certification without overclaiming exact infeasibility;
4. objective/weight/domain alignment, including zero/effectively-uniform weights;
5. intercept/no-intercept and caller-owned/framework-owned initialization;
6. lower/upper domain step caps plus Armijo sufficient decrease;
7. NumPy/CuPy/Torch working dtype/device ownership and no hidden host fallback;
8. ordinary/direct-solver/penalized/smooth-L2-CV/formula/inference consumer closure;
9. generic-solver blast radius and preservation of non-Gamma/Gamma-log paths;
10. analytic/geometric/instrumented negative tests;
11. docs/support claims and release-boundary wording;
12. schema-v4 evidence freshness and immutable v3 evidence;
13. whether scope remains a bounded #150 closure rather than a general constrained optimization or CV rewrite.

Implementation begins only after a fresh independent pass finds no new CRITICAL/HIGH or in-scope MEDIUM plan issue.
