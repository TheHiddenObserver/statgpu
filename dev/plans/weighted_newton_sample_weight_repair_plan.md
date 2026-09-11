# Weighted Newton sample-weight repair plan

Status: IMPLEMENTATION / REVIEW-FIX
Target release: 0.2.6 (unreleased)
Parent workflow contract: PR #141 at `2d38f2c5108bfe53f52d389271572240d319496a`
Repair baseline: PR #142 at `e3d77bb63f0cc66770883c4821ff268f80ad5b0d`

## 1. Trigger and exact failure

Physical Tesla P100 validation of PR #142 reached the full fit/inference path but failed deterministically at `logistic/weighted=True/cupy`:

- coefficient max abs error: `5.319437341916311e-06` (threshold `2e-6`);
- intercept abs error: `4.591245370416663e-06` (threshold `2e-6`);
- p-value max abs error: `1.952412395822556e-05` (threshold `1e-5`).

The unweighted logistic routes passed, including CuPy, Torch, Torch→CuPy, and CuPy→Torch. The remaining problem is therefore not the repaired device-binding path. PR #142 had temporarily forced inference-enabled weighted smooth non-Gaussian `solver="auto"` fits to FISTA because `newton_solver` rejected non-uniform `sample_weight`. The physical failure exposed the numerical cost of that workaround.

## 2. Change classification and active axes

Change type:

- existing shared solver capability repair;
- contract reconciliation with PR #142 weighted penalized-GLM inference;
- no new statistical estimand or inference procedure.

Active axes:

- loss/objective weighting;
- solver/convergence and dispatch;
- NumPy/CuPy/Torch backend/device ownership;
- CV candidate/final-refit dispatch;
- inference integration and physical parity evidence;
- docs/evidence.

Intentionally unchanged:

- FISTA/FISTA-BB algorithms and stopping rules;
- penalty formulas/scaling;
- non-Gaussian sparse-inference support matrix;
- Gaussian bootstrap and SCAD/MCP oracle boundaries;
- Cox weighting policy;
- the separate ordinary `GeneralizedLinearModel._fit_smooth_solver` public guard, which currently rejects weighted explicit Newton/L-BFGS before solver entry. Opening that distinct public API boundary is not required by PR #142 and is not silently changed here.

## 3. Current capability matrix

| Consumer | Current baseline behavior | Desired behavior |
|---|---|---|
| direct `newton_solver`, unweighted | supported | preserve |
| direct `newton_solver`, floating-point weights considered uniform by the historical `allclose` rule | accepted and numerically unweighted | preserve equivalent result |
| direct `newton_solver`, genuinely non-uniform weights | rejected before optimization | support when the loss exposes weighted value/gradient/Hessian |
| penalized smooth non-Gaussian L2, explicit `solver="newton"` | rejects non-uniform weights in the shared solver | backend-native weighted Newton |
| PR #142 weighted inference + `solver="auto"` | runtime wrapper forces FISTA | canonical solver dispatch; smooth L2 resolves to Newton where the existing dispatch table says Newton |
| `PenalizedGLM_CV` weighted inference + `solver="auto"` | wrapper forces FISTA for selection and final refit | canonical CV/direct dispatch with the same weighted objective; selected final estimator reports Newton where applicable |
| ordinary `GeneralizedLinearModel(..., solver="newton")` + weights | public wrapper rejects before solver entry | preserve this separate fail-closed public boundary in this PR |
| Cox + non-uniform weights | unsupported | remain fail-closed |
| FISTA and FISTA-BB consumers | weight-capable paths unchanged | preserve; no stopping/convergence edits |

## 4. Consumer graph

Shared numerical source:

- `statgpu.solvers.newton_solver` / `statgpu/solvers/_newton.py`.

Direct maintained consumers:

- penalized GLM `_fit_loss_backend` on NumPy/CuPy/Torch;
- smooth L2/no-penalty typed penalized GLMs;
- direct solver users/tests/benchmarks;
- robust losses that already expose weighted curvature.

Reviewed but intentionally unchanged consumer boundary:

- ordinary `GeneralizedLinearModel._fit_smooth_solver` keeps its existing public `sample_weight` rejection for explicit Newton/L-BFGS. The guard is visible and occurs before solver entry, so this is not a silent objective change. Enabling that separate surface should be reviewed as its own API expansion rather than smuggled into PR #142.

Meta-estimator consumers:

- `PenalizedGLM_CV` candidate fitting;
- selected full-data final refit;
- PR #142 final-refit-only inference.

Preservation / unsupported consumers:

- Cox loss continues to reject non-uniform weighting explicitly;
- non-smooth penalties remain invalid for Newton through `_validate_smooth_penalty`;
- FISTA/FISTA-BB/LLA/proximal-Newton paths are not modified by this repair.

Documentation/evidence consumers:

- root/EN/CN changelogs;
- EN/CN cross-validation guides;
- EN/CN penalized-GLM inference guides;
- EN/CN generalized-linear-model pages where the temporary FISTA workaround is described;
- physical `validate_penalized_glm_inference_gpu.py` contract and PR #142 evidence wording.

## 5. Desired numerical contract

For genuinely non-uniform analytic weights `w_i >= 0`, `sum(w_i) > 0`, Newton must solve the same normalized average-loss objective used by `LossBase`:

`L(beta) = sum_i w_i l_i(beta) / sum_i w_i + P(beta)`.

The exact same weight vector must be used by:

1. loss value used by Armijo backtracking;
2. gradient;
3. Hessian or fused gradient+Hessian;
4. every trial-point objective evaluation.

Penalty scaling remains unchanged because global rescaling of all analytic weights must leave the normalized loss and therefore the fitted penalized optimum unchanged.

Floating-point weight vectors satisfying the historical Newton uniformity rule (`allclose(values, values[0])`) remain on the established unweighted numerical path. Exact integer uniformity retains its exact-equality behavior. This avoids changing prior compatibility semantics for consumers that treat uniform weighting as the unweighted problem.

Weights are converted once to the execution backend/device relative to the processed design matrix. Explicit CUDA/Torch execution must not transfer numerical work to CPU.

## 6. Implementation plan

1. Replace Newton's `_validate_uniform_sample_weight` rejection gate with general weight validation and backend/device alignment.
2. Preserve the historical floating-point `allclose` uniform-weight classification and normalize those vectors to the existing unweighted path.
3. Thread active genuinely non-uniform weights through constant-Hessian construction, fused/non-fused gradient/Hessian calls, and both sides of Armijo line search.
4. Remove PR #142's fit/CV wrapper that changed public `solver="auto"` execution to FISTA solely because Newton lacked weights.
5. Leave the canonical `_preferred_penalized_glm_solver` table unchanged; it again owns smooth-L2 solver resolution.
6. Update physical validation so weighted `auto` proves Newton selection while unweighted explicit-FISTA cases continue exercising the FISTA inference/device path.
7. Update public docs/changelogs so no PR #142 surface claims that weighted penalized-GLM inference requires the temporary FISTA workaround.

## 7. Deterministic validation

Solver-level:

- weighted logistic Newton equals literal row replication for integer weights after matching average-loss normalization;
- multiplying every weight by a positive constant leaves coefficients unchanged;
- historical uniform / almost-uniform floating-point weights equal the unweighted path;
- invalid shape, negative/non-finite, and zero-total weights fail before numerical work;
- Cox non-uniform weights remain visibly unsupported;
- smooth-penalty validation remains unchanged;
- Torch CPU weighted Newton matches NumPy; physical CuPy/Torch CUDA parity remains a remote gate.

Estimator/CV:

- explicit weighted penalized logistic L2 Newton works independently of inference;
- weighted logistic/Poisson L2 with public `solver="auto"` selects Newton and publishes finite M-estimation inference;
- public `solver` remains `auto`;
- weighted `PenalizedGLM_CV` selection and selected final refit use the canonical dispatch and final estimator reports Newton for applicable rows;
- inference still runs only on the selected full-data final refit.

Backend/evidence:

- hosted NumPy/static regression suite must pass on the exact final head;
- physical P100 validator must cover CuPy, Torch, Torch→CuPy, and CuPy→Torch for weighted logistic and Poisson with `auto`→Newton, retaining coefficient/inference parity thresholds;
- physical evidence is valid only for the exact clean source SHA recorded by the artifact.

## 8. Review-fix loop

After implementation/tests/docs:

1. restart review from the exact final parent/head rather than inheriting the previous clean verdict;
2. inspect solver weighting, objective normalization, backend ownership, direct/CV dispatch, inference integration, compatibility consumers, and documentation claims;
3. fix every in-scope CRITICAL/HIGH and relevant MEDIUM finding;
4. rerun affected validation after each runtime fix;
5. repeat fresh review until no new actionable findings remain;
6. report `PARTIAL_REMOTE_PENDING` until a clean exact-head physical CUDA artifact exists.
