# PR #151 inverse-power Gamma domain/initialization plan — review closure addendum

Status: REVIEW-FIX IN PROGRESS  
Normative base plan: `dev/plans/pr151_inverse_power_gamma_smooth_solver_domain_initialization_closure_plan.md` at `3ea92a406ab093eb32e8fce7a98659fb2fb687e2`  
Parent PR: #151  
Parent issue: #150  
Follow-up to fold back after implementation: #152

This addendum is normative for the two Round-8 findings below. It does not broaden the implementation beyond inverse-power Gamma explicit Newton/L-BFGS and the already-listed affected ordinary/penalized/smooth-L2-CV consumers.

## Round 8 findings and fixes

### 1. MEDIUM / SOLVER / API — domain failure exit semantics were underspecified

The base plan correctly requires domain-boundary stagnation and uncertifiable initialization to avoid publishing a successful fitted state, but `newton_solver` / `lbfgs_solver` currently return only `(params, n_iter)` and their ordinary line-search exhaustion is warning-based. A domain failure therefore needs an explicit fail-hard contract rather than relying on a warning that an estimator could ignore.

Freeze the following behavior:

- **deterministic invalid input/design**: an active non-finite row, active exact zero row, or caller-supplied `init_coef` that is outside the maintained inverse-power Gamma smooth training band raises `ValueError` before the first value/gradient/Hessian evaluation;
- **numerical certification failure**: a finite design for which the backend-native initializer exhausts its reviewed certification procedure without a band-certified interior raises `RuntimeError` (a private `RuntimeError` subclass is acceptable) with wording that says `no numerically certified smooth-domain start`; it must not claim exact mathematical infeasibility unless an exact contradiction was independently established;
- **iteration-domain failure**: if the final post-fallback search direction has no meaningful interior step, or the domain step cap collapses to the reviewed floor before gradient convergence, raise `RuntimeError` (or the same private subclass) with domain-boundary/convergence wording;
- these domain `RuntimeError`s are not ordinary recoverable candidate failures and must not be converted by `PenalizedGLM_CV` into NaN scores, a different loss, MSE fallback, or another solver;
- existing ordinary Armijo line-search warning semantics remain unchanged for failures that are not caused by the new inverse-Gamma domain contract;
- ordinary, penalized and CV fit transactions must preserve/invalidate prior fitted/provenance/inference/selection state exactly according to their already-characterized failure policy.

Targeted tests must assert the exception category and that no objective primitive is called before an invalid explicit start is rejected. Previously-fitted -> domain-failing refit tests must assert fitted/provenance/inference/CV-selection state, not only the raised message.

### 2. MEDIUM / CONSUMER / BACKEND — no-intercept closure was not explicit for penalized/CV consumers

The domain initializer and step-cap hooks are shared solver behavior. It is insufficient to physically/hostedly prove `fit_intercept=False` only on ordinary `GammaRegression` while affected public penalized/CV surfaces may construct different `X_work`, selective penalties, fold warm starts and final refits.

Add the following blocking rows to the base plan:

#### Hosted

For both `fit_intercept=True` and `False`, cover:

- `PenalizedGammaRegression(link="inverse_power", penalty="l2", solver="newton")` where that explicit route is maintained;
- the corresponding explicit `solver="lbfgs"` maintained route;
- generic `PenalizedGeneralizedLinearModel(loss="gamma", loss_kwargs={"link":"inverse_power"}, penalty="l2", ...)`;
- `PenalizedGLM_CV(loss="gamma", loss_kwargs={"link":"inverse_power"}, penalty="l2", fit_intercept=...)`, including fold candidate fitting, correct inverse-link validation loss, selected-alpha identity and final refit;
- feasible no-intercept designs, active contradictory designs, and zero-weight contradiction rescue where the public surface accepts analytic weights;
- framework-owned fold/path warm starts that become invalid on the current training design must be discarded and reseeded; caller-owned direct-solver starts remain authoritative and fail if invalid.

#### Physical schema v4

In addition to ordinary no-intercept inverse-Gamma rows already required by the base plan, include at least one **no-intercept** inverse-power Gamma penalized-L2 direct-fit route and one **no-intercept** smooth-L2 CV route on CuPy and Torch, with NumPy as reference. The CV route must record exact selected-alpha identity as well as coefficient/intercept parity and executed solver/backend/device provenance.

The existing intercept-bearing penalized/CV routes remain preservation rows; this addendum does not remove them.

## Composite-plan closure rule

The implementation plan is the base plan at `3ea92a406ab093eb32e8fce7a98659fb2fb687e2` plus this addendum. A fresh independent review must now re-check that composite target from scratch. The plan is `REVIEW CLEAN` only if that pass finds no new CRITICAL/HIGH or in-scope MEDIUM issue.
