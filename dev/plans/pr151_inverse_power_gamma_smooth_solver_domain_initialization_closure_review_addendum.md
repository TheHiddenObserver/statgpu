# PR #151 inverse-power Gamma domain/initialization plan — review closure addendum

Status: REVIEW-FIX IN PROGRESS  
Normative base plan: `dev/plans/pr151_inverse_power_gamma_smooth_solver_domain_initialization_closure_plan.md` at `3ea92a406ab093eb32e8fce7a98659fb2fb687e2`  
Parent PR: #151  
Parent issue: #150  
Follow-up to fold back after implementation: #152

This addendum is normative for the Round-8/9 findings below. It does not broaden the implementation beyond inverse-power Gamma explicit Newton/L-BFGS and the already-listed affected ordinary/penalized/smooth-L2-CV consumers.

## Round 8 findings and fixes

### 1. MEDIUM / SOLVER / API — domain failure exit semantics were underspecified

The base plan correctly requires domain-boundary stagnation and uncertifiable initialization to avoid publishing a successful fitted state, but `newton_solver` / `lbfgs_solver` currently return only `(params, n_iter)` and their ordinary line-search exhaustion is warning-based. A domain failure therefore needs an explicit fail-hard contract rather than relying on a warning that an estimator could ignore.

Freeze the following behavior:

- **deterministic invalid input/design**: an active non-finite row, active exact zero row, or caller-supplied `init_coef` that is outside the maintained inverse-power Gamma smooth training band raises `ValueError` before the first value/gradient/Hessian evaluation;
- **numerical certification failure**: a finite design for which the backend-native initializer exhausts its reviewed certification procedure without a band-certified interior raises `RuntimeError` (a private `RuntimeError` subclass is acceptable) with wording that says `no numerically certified smooth-domain start`; it must not claim exact mathematical infeasibility unless an exact contradiction was independently established;
- **iteration-domain failure**: if the final post-fallback search direction has no meaningful interior step, or the domain step cap collapses to the reviewed floor before gradient convergence, raise `RuntimeError` (or the same private subclass) with domain-boundary/convergence wording;
- if numerical roundoff causes every backtracking candidate to be rejected by the domain hook before any Armijo objective can be evaluated, treat that as the same fail-hard domain-convergence class rather than falling through to the ordinary warning-only line-search failure path;
- these domain `RuntimeError`s are not ordinary recoverable candidate failures and must not be converted by `PenalizedGLM_CV` into NaN scores, a different loss, MSE fallback, or another solver;
- existing ordinary Armijo line-search warning semantics remain unchanged for failures that are not caused by the new inverse-Gamma domain contract;
- ordinary, penalized and CV fit transactions must preserve/invalidate prior fitted/provenance/inference/selection state exactly according to their already-characterized failure policy.

Targeted tests must assert the exception category and that no objective primitive is called before an invalid explicit start is rejected. Previously-fitted -> domain-failing refit tests must assert fitted/provenance/inference/CV-selection state, not only the raised message.

### 2. MEDIUM / CONSUMER / BACKEND — no-intercept closure was not explicit for public penalized direct-fit consumers

The domain initializer and step-cap hooks are shared solver behavior. It is insufficient to prove `fit_intercept=False` only on ordinary `GammaRegression` while public penalized direct-fit surfaces construct a different `X_work`, selective penalty and framework warm start.

Blocking hosted rows therefore cover **both `fit_intercept=True` and `False`** for:

- `PenalizedGammaRegression(link="inverse_power", penalty="l2", solver="newton")` where that explicit route is maintained;
- the corresponding explicit `solver="lbfgs"` maintained route;
- generic `PenalizedGeneralizedLinearModel(loss="gamma", loss_kwargs={"link":"inverse_power"}, penalty="l2", ...)`;
- feasible no-intercept designs, active contradictory designs, and zero-weight contradiction rescue where the public surface accepts analytic weights;
- framework-owned warm starts that are invalid for the finalized design must be discarded and reseeded; caller-owned direct-solver starts remain authoritative and fail if invalid.

Physical schema v4 includes at least one **no-intercept** inverse-power Gamma penalized-L2 direct-fit route on CuPy and Torch, with NumPy as reference. The existing intercept-bearing penalized direct-fit route remains a preservation row.

## Round 9 finding and fix

### MEDIUM / API / SCOPE — `PenalizedGLM_CV` does not expose `fit_intercept`

Fresh source review found that current `PenalizedGLM_CV.__init__` has no public `fit_intercept` parameter. Its scalar-response CV contract is intercept-bearing; final-refit compatibility uses `getattr(self, "_fit_intercept", True)` rather than exposing a no-intercept CV control. Adding `fit_intercept=False` to CV merely to exercise this domain closure would be a new public API feature outside #150.

Therefore **remove the Round-8 no-intercept CV requirement**. The blocking CV closure is instead the current public intercept-bearing smooth-L2 inverse-power Gamma surface:

- `PenalizedGLM_CV(loss="gamma", loss_kwargs={"link":"inverse_power"}, penalty="l2", ...)` must preserve `loss_kwargs` through fold candidate construction, validation scoring, alpha selection and final refit;
- wrong-link optimized Gamma validation kernels must be bypassed for this route;
- framework-owned fold/path warm starts are validated/reseeded where applicable;
- domain failures must propagate under the fail-hard contract above and must not be converted to unrelated fallback data;
- exact selected-alpha identity, coefficient/intercept parity, and solver/backend/device provenance are required across maintained backends;
- schema v4 keeps the intercept-bearing inverse-power Gamma smooth-L2 CV route on NumPy/CuPy/Torch. **No new no-intercept CV API is introduced.**

This correction does not weaken no-intercept closure where the public API already exposes it: ordinary `GammaRegression` and direct penalized Gamma remain required in both intercept/no-intercept forms.

## Composite-plan closure rule

The implementation plan is the base plan at `3ea92a406ab093eb32e8fce7a98659fb2fb687e2` plus this addendum. A fresh independent review must re-check that composite target from scratch. The plan is `REVIEW CLEAN` only if that pass finds no new CRITICAL/HIGH or in-scope MEDIUM issue.
