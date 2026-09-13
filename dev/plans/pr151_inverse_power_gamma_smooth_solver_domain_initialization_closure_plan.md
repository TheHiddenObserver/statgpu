# PR #151 inverse-power Gamma smooth-solver domain / initialization closure plan

Status: PLAN DRAFT / REVIEW-FIX IN PROGRESS  
Parent PR: #151  
Parent issue: #150  
Follow-up to be folded back after implementation: #152  
Branch: `fix/glm-weighted-explicit-solver-guard`  
Plan baseline head: `b5228fdc4ed78af6fcaa760ecee1475ae4ebe1a7`

## 1. Goal

Close the remaining inverse-power Gamma explicit smooth-solver gap inside PR #151 rather than retaining `fit_intercept=True` as an artificial capability condition.

The final contract should be statistical/domain based, not weight-shape based:

- inverse-power Gamma has linear predictor `eta = X @ beta` and requires a strictly positive predictor on every observation that belongs to the executed objective;
- explicit Newton/L-BFGS should start from a family-valid interior point and keep every accepted iterate in that domain;
- `fit_intercept=False` is supported when the executed design admits a strictly positive predictor and fails precisely when no numerically certifiable interior point is available;
- unweighted, positive-uniform, effectively-uniform, and genuine non-uniform analytic weights use the same domain/initialization policy;
- weight classification still preserves PR #151's historical objective contract: omitted/uniform/effectively-uniform weights execute the unweighted objective, while genuine non-uniform weights execute `sum(w_i * ell_i) / sum(w_i)`;
- zero-weight rows constrain the domain only when genuine weighting remains active after the maintained solver weight-preparation rule.

This change replaces the current public rule

```text
inverse-power Gamma + genuine non-uniform weights + explicit Newton/L-BFGS
+ fit_intercept=False -> categorical pre-fit rejection
```

with

```text
inverse-power Gamma + explicit Newton/L-BFGS
    -> construct/validate an interior start for the executed objective
    -> cap/backtrack every trial step to remain in the positive predictor domain
    -> fit when a numerically certified interior exists
    -> fail with a design/domain error when no interior can be certified
```

## 2. Change classification and active review axes

Classification:

1. existing-capability reconciliation / completion of the original #150 support matrix;
2. numerical correctness repair for inverse-power Gamma initialization and globalization;
3. shared Newton/L-BFGS solver contract change through loss-owned optimization-domain hooks;
4. evidence-invalidating numerical change after the previously accepted `c6781cb6` physical run.

Active axes:

- loss/objective/domain;
- solver initialization, step capping, line search, convergence and failure semantics;
- analytic `sample_weight` and zero-weight-row semantics;
- intercept/no-intercept behavior;
- NumPy/CuPy/Torch backend, device and working dtype;
- ordinary GLM, direct solver, penalized Gamma and `PenalizedGLM_CV` consumers;
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
- covariance estimands or prediction clipping policy;
- the current `_ETA_LO/_ETA_HI` numerical clipping constants in `GammaLoss`;
- ordered models or standalone logistic APIs.

## 3. Current inconsistency

### 3.1 Original PR #151 contract

The reviewed parent plan targeted weighted explicit Newton/L-BFGS support for all maintained ordinary GLM family/link rows and explicitly required both `fit_intercept=True` and `False`, including inverse-power Gamma.

### 3.2 Current implementation

`statgpu/linear_model/_glm_weighted_explicit_solver_contract.py` currently:

- supplies a family-valid inverse-power Gamma start when an intercept is present: zero slopes plus `1 / mean(y)` or `1 / weighted_mean(y)`;
- categorically rejects only the newly opened genuine-nonuniform no-intercept row;
- leaves omitted/uniform/effectively-uniform no-intercept calls on the historical zero-start path.

That narrowing was recorded in the implementation addendum and tracked as #152, but it is an initialization/domain limitation rather than a statistical prohibition.

### 3.3 Why preserving the historical zero start is no longer desirable

For inverse-power Gamma,

\[
\ell_i(\eta_i)=y_i\eta_i-\log \eta_i,
\qquad \eta_i>0.
\]

The historical no-intercept zero start gives `eta=0` and relies on `GammaLoss` clipping to `_ETA_LO` before the first gradient/Hessian/line-search evaluation. Once PR #151 reopens numerical source and requires fresh physical evidence anyway, preserving that boundary-dependent trajectory is not a useful compatibility goal.

Uniform and unweighted inputs represent the same normalized statistical objective, so initialization/domain behavior must not change discontinuously when a weight vector crosses the solver's `allclose` uniformity threshold.

## 4. Intended mathematical contract

### 4.1 Executed active sample

Weight handling must use the same prepared analytic-weight identity as the selected solver.

Let `w_eff` denote the backend-aligned prepared weight returned by the shared analytic-weight preparation rule:

- `w_eff is None` for omitted, positive-uniform, and historically effectively-uniform weights; then every retained row belongs to the executed unweighted objective;
- `w_eff` is the genuine non-uniform backend-native vector otherwise; then active rows are

\[
A=\{i:w_{\mathrm{eff},i}>0\}.
\]

The domain is defined on the rows actually present in the executed objective:

\[
\mathcal D_A=\{\beta:x_i^\top\beta>0\ \forall i\in A\}.
\]

This is important for zero-weight equivalence: a conflicting row with genuine weight zero must not make an otherwise feasible weighted design fail.

An effectively-uniform vector that is normalized away by the historical `allclose` rule intentionally inherits the unweighted objective and therefore the all-row domain. Domain membership follows the **prepared solver objective**, not raw weight syntax.

### 4.2 Intercept and no-intercept are one domain problem

Once the estimator has constructed its actual numerical design `X_work`, including an augmented intercept column when applicable, the smooth solver only needs to solve

\[
X_A\beta>0.
\]

An intercept-bearing design has an immediate separator through its constant positive column. A no-intercept design may or may not admit one.

The initialization routine therefore operates on the actual executed `X_work`; it does not branch on the public `fit_intercept` flag except indirectly through the design it receives.

### 4.3 Geometric feasibility characterization

Strict feasibility

\[
\exists\beta:\ X_A\beta>0
\]

is equivalent by Gordan's theorem to the absence of a nonzero nonnegative vector `lambda` satisfying

\[
X_A^\top\lambda=0.
\]

Equivalently, after normalizing each nonzero active row by a positive scalar,

\[
0\notin \operatorname{conv}\{u_i:i\in A\},
\qquad
u_i=\frac{x_i}{\|x_i\|_2}.
\]

If `c*` is the minimum-norm point of that convex hull and `c* != 0`, projection geometry gives

\[
u_i^\top c^*\ge \|c^*\|_2^2>0,
\]

so `c*` is a separating direction.

Any active all-zero design row is immediately infeasible because its predictor is identically zero.

## 5. Implementation design

### A. Unify analytic-weight preparation used by Newton, L-BFGS and domain initialization

Move the duplicated Newton/L-BFGS alignment + uniformity classification into one private helper in `statgpu/solvers/_utils.py`, for example

```python
_prepare_analytic_sample_weight(sample_weight, n_samples, backend, ref_arr)
```

with the current established semantics:

1. validate shape/finite/non-negative/positive total;
2. align to the executed design backend, concrete device and dtype;
3. apply the historical floating-point `allclose(values, values[0])` rule only after alignment;
4. return `None` for omitted/uniform/effectively-uniform inputs;
5. return the aligned vector for genuine non-uniform weights.

Newton and L-BFGS both use this helper. L-BFGS keeps its separate loss-capability gate after preparation. Preserve a thin alias for `_prepare_newton_sample_weight` only if current internal tests/imports require it; do not preserve duplicate classification logic.

The domain initializer, feasibility check and domain step cap consume the prepared identity, not the raw public weight vector.

### B. Add private loss-owned optimization-domain hooks

Add default no-op hooks on `LossBase`, with private names so this is not a new public user API. The exact names can be selected during implementation, but semantics should be equivalent to:

```python
_loss_domain_initial_point(X, y, sample_weight=None) -> array | None
_loss_domain_is_feasible(X, coef, sample_weight=None) -> bool
_loss_domain_max_step(X, coef, delta, sample_weight=None) -> float | None
```

Here `delta` is the **additive update direction**, so a trial point is

\[
\beta(t)=\beta+t\Delta.
\]

Default `LossBase` behavior:

- initial point: `None` (solver retains its normal zero/default start);
- feasibility: `True`;
- max step: `None` / no domain cap.

`GammaLoss(link="inverse_power")` overrides all three. `GammaLoss(link="log")` keeps the defaults.

Ownership rule: Gamma defines its own predictor domain; generic solvers consume the hooks without importing `glm_core` or hard-coding the Gamma loss name.

### C. Construct a backend-native inverse-power Gamma interior start

For active design `X_A` on the actual execution backend/dtype:

1. Reject an active row whose numerical norm is zero/non-finite.
2. Normalize nonzero active rows by positive row norms to improve conditioning; this does not change the sign-feasibility problem.
3. Fast path: if a single feature column has one certified strict sign on every active row, use that signed basis vector as the separating direction. This recovers the current augmented-intercept start because the appended ones column is immediately detected.
4. Otherwise solve the minimum-norm convex-hull problem

   \[
   \min_{c\in\operatorname{conv}(u_i)} \frac12\|c\|_2^2
   \]

   using a deterministic backend-native Gilbert / Frank-Wolfe iteration.

Deterministic initialization:

\[
c_0=\frac1{|A|}\sum_{i\in A}u_i,
\]

which is already a convex combination. At iterate `c_k`, choose

\[
s_k=u_{j_k},\qquad
j_k=\operatorname*{argmin}_i u_i^\top c_k,
\]

using the backend's deterministic first-index tie behavior, and define

\[
m_k=\min_i u_i^\top c_k,
\qquad
G_k=\|c_k\|_2^2-m_k.
\]

A positive `m_k` above the reviewed dtype-scaled certification tolerance is a direct feasible-separator certificate.

Otherwise use the exact segment step

\[
\gamma_k=
\operatorname{clip}\left(
\frac{\|c_k\|_2^2-c_k^\top s_k}
     {\|c_k-s_k\|_2^2},0,1\right),
\]

\[
c_{k+1}=(1-\gamma_k)c_k+\gamma_k s_k.
\]

If the step denominator is below the reviewed numerical floor before a positive separator is certified, classify the geometry as `numerically unresolved` rather than dividing by a tiny value.

Tolerance policy:

- tolerances are derived from the executed floating dtype after row normalization;
- the exact constants and maximum iteration count are frozen by hosted analytic/geometric characterization **before** schema-v4 physical validation;
- they are not loosened after observing a physical failure;
- successful certification always performs a final check on the original, unnormalized active design and rejects non-finite/non-positive predictors.

The negative certificate is intentionally conservative: convergence near the origin or exhaustion without a positive certified margin reports `no numerically certified strict interior` / `numerically unresolved`. Do not claim exact mathematical infeasibility from an approximate floating-point convex-hull residual.

No SciPy/CPU LP or QP fallback is introduced. CuPy/Torch inputs stay on their selected device; only scalar stopping/certificate values may synchronize.

### D. Scale the feasible direction using the Gamma objective

Let `d` be a certified direction and

\[
a_i=x_i^\top d>0.
\]

For the unpenalized Gamma inverse-power objective along `beta = c d`,

\[
L(c)=
\frac{1}{s}\sum_{i\in A}w_i
\left(y_i c a_i-\log(ca_i)\right),
\]

with the obvious unweighted convention. Its ray-optimal positive scale is

\[
c_{\mathrm{ray}}
=\frac{s}{\sum_{i\in A}w_i y_i a_i}
\]

(or `n / sum(y_i a_i)` on the unweighted route).

Use this only as an initialization scale, not as a new statistical restriction. After scaling, explicitly verify on the backend that the coefficient vector is finite and every active predictor is finite and strictly positive. If the ray scale is non-finite or cannot produce a numerically certified interior point, fail closed with the domain-initialization error.

For an intercept-only separator this reduces exactly to the current start `intercept = 1 / mean(y)` or `1 / weighted_mean(y)`.

The initializer need not minimize a smooth penalty; the penalty does not change feasibility. Penalized Newton/L-BFGS may start from the same feasible data-driven point and then optimize the complete penalized objective.

### E. Compute a domain-aware maximum trial step

Blind halving alone is not sufficient for a claimed domain closure because Newton and L-BFGS have fixed 20/25-trial limits. For an additive trial direction `Delta`, let

\[
\eta=X_A\beta,
\qquad
r=X_A\Delta.
\]

Feasibility requires

\[
\eta_i+t r_i>0.
\]

Only rows with `r_i < 0` constrain the step. The mathematical boundary is

\[
t_{\max}
=\min_{i:r_i<0}\frac{\eta_i}{-r_i}.
\]

The Gamma domain hook returns an interior cap slightly below that boundary using a fixed reviewed safety factor derived before physical validation. If no row decreases toward the boundary, no cap is needed.

Newton converts its subtract-direction notation to the additive direction `Delta=-d`; L-BFGS passes its ordinary additive search direction directly. Each solver starts Armijo from

\[
t_0=\min(1,t_{\mathrm{domain}})
\]

when a cap is present, and still performs its existing objective-based Armijo backtracking afterward.

Every trial is also checked by `_loss_domain_is_feasible` before objective evaluation as a defensive certificate; an infeasible trial is never passed into `GammaLoss`.

### F. Make Newton and L-BFGS domain preserving

For `newton_solver` and `lbfgs_solver` only:

1. preprocess X/y and prepare analytic weights on the executed backend/dtype;
2. if no initial coefficient is supplied and the loss supplies a domain initializer, use it before any value/gradient/Hessian evaluation;
3. validate the initial point before the first value/gradient/Hessian evaluation;
4. compute a loss-domain step cap from the additive search direction before Armijo;
5. check every trial point's feasibility before objective evaluation;
6. retain existing Armijo sufficient-decrease conditions after domain feasibility is satisfied;
7. accepted/current iterates are therefore always in the loss domain.

Explicit versus framework-owned warm starts require distinct handling:

- a direct exported solver call that supplies `init_coef` is authoritative; an invalid explicit warm start raises a precise domain error and is not silently replaced;
- internal estimator/CV warm starts are framework-owned. Affected Gamma consumers must validate such warm starts at their boundary and, when one is invalid for the current fold/design, discard it and pass `init_coef=None` so the loss can construct a fresh family-valid start;
- do not add a generic solver flag that silently guesses whether an `init_coef` came from a user or an internal cache.

The existing Armijo constants, convergence tolerances, Newton linear-solve behavior, L-BFGS history update, and warning policy remain unchanged.

This closure deliberately does **not** remove or reinterpret `GammaLoss._ETA_LO/_ETA_HI`; it prevents explicit Newton/L-BFGS from evaluating or accepting non-positive active predictors. Other Gamma solver families retain their current clipping/numerical behavior in this PR.

### G. Remove the ordinary-GLM categorical no-intercept guard

After the loss/solver domain contract exists, simplify `statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`:

- remove the genuine-nonuniform `fit_intercept=False` categorical rejection;
- remove the installer-owned inverse-Gamma start once the shared loss/solver initializer reproduces its intercept behavior;
- continue to construct actual `X_work` on the selected backend/dtype and pass the user's explicit solver and `sample_weight` unchanged;
- keep provenance publication, runtime-installer idempotence/introspection, ordered-model isolation, and explicit solver authority unchanged.

No family-specific feasibility rule should remain in the ordinary wrapper once the shared domain contract owns it.

## 6. Consumer graph and required closure

Because initialization/domain handling moves into `GammaLoss` + shared Newton/L-BFGS, this is broader than the ordinary wrapper even though it closes the original #150 gap.

### Required affected consumers

1. Ordinary `GammaRegression(link="inverse_power")` and generic ordinary GLM surface that resolves to the same loss.
2. Direct exported `newton_solver` and `lbfgs_solver` when called with `GammaLoss(link="inverse_power")`.
3. Public `PenalizedGammaRegression(link="inverse_power")` and `PenalizedGeneralizedLinearModel(loss="gamma", loss_kwargs={"link": "inverse_power"})` on maintained smooth L2/no-penalty Newton/L-BFGS routes.
4. Public `PenalizedGLM_CV(loss="gamma", loss_kwargs={"link": "inverse_power"}, penalty="l2", ...)`, including candidate folds and selected full-data refit. The current class exposes scalar-response Gamma plus `loss_kwargs`; this is therefore an affected public consumer, not an optional discovery item.
5. Formula-driven ordinary Gamma paths, including an explicit no-intercept formula.
6. Existing `compute_inference=True` ordinary/penalized inverse-power Gamma rows where inference is already publicly supported; final fitted parameters handed to inference remain domain-feasible.

### Internal warm-start ownership to trace

Before coding, enumerate every affected call site that passes `init_coef` into Newton/L-BFGS, especially:

- penalized direct-fit `_init_coef` / warm-start controls;
- CV alpha-path/fold warm starts;
- final-refit reconstruction.

Mark each as public/user-owned or internal/framework-owned and apply §5F accordingly. A representative fit is insufficient for this shared change.

### Preservation consumers

- Gamma log link;
- all non-Gamma losses through Newton/L-BFGS;
- explicit IRLS/FISTA/etc.;
- current L-BFGS non-GLM weight capability boundaries;
- ordered models and standalone logistic paths.

## 7. Hosted test plan

### 7.1 Analytic and geometric correctness

Add deterministic no-intercept inverse-Gamma fixtures covering:

1. **1D analytic feasible reference** with strictly positive X. For the unpenalized weighted problem,

   \[
   \beta^*=\left(\frac{\sum_i w_i y_i x_i}{\sum_i w_i}\right)^{-1},
   \]

   providing an exact coefficient reference for Newton and L-BFGS under omitted, uniform and genuine non-uniform weights.
2. **Multi-feature feasible design** whose rows lie in a strict half-space but require a combination of columns rather than a single positive feature; prove the convex-hull fallback is exercised.
3. **Exact contradictory design**, e.g. active rows containing `x` and `-x`, which must fail before numerical optimization.
4. **Active zero row**, which must fail because no coefficient can make its predictor positive.
5. **Zero-weight contradictory row** under genuine non-uniform weights, which must be ignored by the domain and match the result after dropping that row.
6. **Near-boundary / numerically unresolved geometry**, which must fail with the dedicated domain-certification error rather than return a clipped fit.

For every successful case assert:

- initial and final active predictors are strictly positive;
- no instrumented value/gradient/Hessian evaluation receives a non-positive active predictor;
- coefficient error and gradient/KKT residuals meet maintained tolerance;
- positive global weight rescaling leaves the optimum unchanged;
- omitted/uniform/effectively-uniform routes agree according to the historical unweighted objective contract.

### 7.2 Solver behavior

For both Newton and L-BFGS:

- initial domain validation happens before the first loss evaluation;
- direct explicit invalid `init_coef` fails precisely and is not silently repaired;
- framework-owned invalid warm starts are discarded/reseeded by the owning consumer rather than by solver guesswork;
- an intentionally constructed domain-crossing full step receives `t_0 < 1` from the domain cap when appropriate;
- no infeasible trial reaches `GammaLoss.fused_value_and_gradient` / `hessian` instrumentation;
- Armijo still rejects objective-increasing feasible candidates;
- current Newton rank fallback / L-BFGS curvature-history behavior remains unchanged;
- convergence and line-search warnings remain hard failures in PR151 acceptance tests.

### 7.3 Weight/backend/dtype behavior

- NumPy and Torch-CPU deterministic parity in hosted tests;
- CuPy/Torch-CUDA exact-device behavior covered by physical validation;
- mixed Torch input dtype classification uses the final promoted `X_work` dtype;
- the shared analytic-weight helper preserves current Newton/L-BFGS uniformity classification exactly;
- domain initialization and step capping use the same prepared weight identity as objective evaluation;
- no full active design/weight vector is transferred to host merely for feasibility initialization.

### 7.4 Public consumer, CV, formula and inference closure

Cover:

- ordinary `GammaRegression` intercept/no-intercept;
- generic ordinary GLM equivalent surface;
- formula with and without intercept plus missing-row/sample-weight alignment;
- public penalized inverse-power Gamma smooth L2/no-penalty direct fits;
- `PenalizedGLM_CV` inverse-power Gamma L2 candidate/fold/final-refit behavior, including an internal warm-start that becomes invalid under a different fold geometry;
- candidate/domain failure classification: design-domain errors must not be silently converted into an unrelated validation score or MSE fallback;
- inference smoke/parity on successful maintained rows;
- clone/get-params/introspection unchanged because no constructor control is added.

### 7.5 Preservation

- Gamma log-link Newton/L-BFGS unchanged within exact/maintained tolerance;
- non-Gamma Newton/L-BFGS sample-weight tests unchanged;
- IRLS/FISTA and `solver="auto"` preservation tests remain green;
- runtime installer idempotence/import-order and provenance regressions remain green.

## 8. Documentation and issue handling

Update only affected public claims after implementation is proven:

- EN/CN Gamma/GLM model documentation;
- EN/CN solver support matrix and algorithm/domain notes where appropriate;
- root/EN/CN changelog PR151 entry;
- PR #151 body/current implementation head/evidence status.

Documentation should state the statistical condition rather than an intercept proxy:

> inverse-power Gamma explicit Newton/L-BFGS requires a numerically certifiable strictly positive linear predictor on the executed active design; an intercept guarantees an easy feasible start, while no-intercept designs are supported when such a predictor exists.

Issue #152 remains open during implementation. After this closure is implemented, reviewed, hosted-green and physically accepted in PR #151, close #152 as completed by PR #151 rather than leaving a duplicate future task.

## 9. Physical CUDA evidence and schema v4

Any production numerical change after `c6781cb6` invalidates schema-v3 P100 evidence as final implementation evidence. The retained v3 JSON remains historical evidence for `c6781cb6`; it must not be overwritten.

Before the new physical run, version the validator to schema **v4** and freeze the added contract. Preserve all v3 thresholds unless a separate pre-run review finds a principled reason to change them; do not tune thresholds after seeing a failed physical result.

Add physical routes for at least:

- feasible no-intercept inverse-power Gamma × Newton/L-BFGS × NumPy/CuPy/Torch;
- omitted/uniform/genuine non-uniform initialization identity on a maintained inverse-Gamma fixture where appropriate;
- positive global weight-rescaling invariance;
- final active-predictor minimum `> 0` and truthful solver/backend/device provenance;
- one domain-step-cap characterization showing a full step would cross the boundary while the accepted step remains feasible;
- public penalized inverse-power Gamma L2 direct fit and CV final refit on NumPy/CuPy/Torch if the hosted closure confirms those maintained paths.

Retain the existing ordinary family/link routes, cross-container routes, and shared penalized/CV L-BFGS consumers unless schema-v4 review explicitly documents a narrower unaffected subset. No previously accepted threshold may be loosened post-run without a new reviewed schema.

Final physical acceptance requires a clean exact implementation/validator head, CuPy + Torch CUDA on the same source, warning-as-error gates, frozen coefficient/rescaling/domain thresholds, and a newly retained raw artifact with its exact `source_sha` outside benchmark-source scan roots.

## 10. Implementation order

1. Freeze this plan through repeated independent plan review/fix passes.
2. Inventory affected direct/penalized/CV warm-start call sites and classify ownership.
3. Add shared analytic-weight preparation and parity tests without changing behavior.
4. Add `LossBase` private domain hooks and inverse-power `GammaLoss` feasibility/step-cap implementation + unit tests.
5. Add backend-native feasible-start algorithm and analytic/geometric tests.
6. Integrate domain initialization/validation/step cap into Newton; run Newton-focused regression suite.
7. Integrate the same hooks into L-BFGS; run L-BFGS/weighted capability regressions.
8. Remove ordinary wrapper's no-intercept guard/family-specific initializer and close ordinary public/formula/inference tests.
9. Close penalized/CV consumers, including framework-owned invalid warm-start reseeding.
10. Update EN/CN docs/changelogs/support matrices.
11. Version/freeze physical validator schema v4 and its static contract tests.
12. Run targeted tests, full hosted suite and current-head workflows.
13. Run `.claude/skills/code-review` in independent auto-fix mode; repeat until no CRITICAL/HIGH or in-scope MEDIUM finding remains.
14. Run exact-source physical P100/CUDA acceptance for schema v4 and retain the raw artifact outside benchmark-source scan roots.
15. Re-run a fresh exact-head review after physical evidence and only then mark PR151 implementation closure complete. Do not merge without explicit approval.

## 11. Non-goals

- no general LP/QP package or CPU fallback for feasibility;
- no universal constrained-optimization framework;
- no change to Gamma log link;
- no removal of inverse-Gamma clipping from FISTA/IRLS/other solver families;
- no Proximal-Newton metric-prox implementation (#157);
- no Huber IRLS implementation (#156);
- no unrelated loss capability redesign beyond private no-op domain hooks needed by shared Newton/L-BFGS;
- no new performance/speedup claim.

## 12. Plan review/fix history

### Round 1 — findings fixed

- **HIGH / SOLVER — fixed:** the first draft relied on reject-and-half domain backtracking only. Fixed by adding a loss-owned analytic domain step cap, with ordinary Armijo still applied after the cap.
- **HIGH / CV — fixed:** the first draft treated every supplied `init_coef` as user-authoritative. Fixed by separating direct/user-owned warm starts from framework-owned penalized/CV warm starts; invalid internal starts are discarded at their owning consumer and rebuilt through the loss initializer.
- **MEDIUM / NUMERICAL — fixed:** the first draft under-specified convex-hull iteration initialization, denominator failure and certification semantics. Added deterministic `c_0`, explicit gap/margin quantities, tiny-denominator handling, dtype-derived tolerances, final original-design certification, and conservative `numerically unresolved` failure language.
- **MEDIUM / CONSUMER — fixed:** the first draft made inverse-power Gamma penalized/CV closure conditional on later discovery. `PenalizedGammaRegression` publicly supports `link="inverse_power"` and `PenalizedGLM_CV` publicly exposes Gamma plus `loss_kwargs`; both are now explicit affected consumers.

## 13. Plan review/fix closure criteria

Each new plan review pass restarts from the then-current exact plan file and checks at least:

1. mathematical equivalence/limitations of the feasibility characterization;
2. whether floating-point feasibility can be overclaimed as exact infeasibility;
3. objective/weight/domain alignment, especially zero-weight and effectively-uniform rows;
4. initialization behavior under intercept, no intercept, user-owned/internal warm starts and penalties;
5. analytic domain step capping plus Armijo sufficient decrease;
6. NumPy/CuPy/Torch and mixed-dtype/device ownership;
7. ordinary/penalized/CV/formula/inference consumer closure;
8. generic-solver blast radius and preservation of non-Gamma losses;
9. test strength: analytic identity, geometric positive/negative cases, instrumentation, failure behavior;
10. documentation/support-matrix consistency;
11. validator schema/evidence freshness and retained historical artifact semantics;
12. whether the scope remains a bounded #150 closure rather than becoming a general constrained-optimization project.

Implementation may begin only after a fresh independent pass finds no new CRITICAL/HIGH or in-scope MEDIUM plan issue.
