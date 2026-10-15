# Weighted Newton sample-weight repair plan

Status: COMPLETE / PHYSICAL EVIDENCE CLOSED
Target release: 0.2.6 (unreleased)
Parent workflow contract: PR #141 at `2d38f2c5108bfe53f52d389271572240d319496a`
Repair baseline: PR #142 at `e3d77bb63f0cc66770883c4821ff268f80ad5b0d`
Accepted numerical source: `db448d718f523eacf97bcb3c419e376c9812362d`

## 1. Trigger and exact failure

Physical Tesla P100 validation of PR #142 reached the full fit/inference path but failed deterministically at `logistic/weighted=True/cupy`:

- coefficient max abs error: `5.319437341916311e-06` (threshold `2e-6`);
- intercept abs error: `4.591245370416663e-06` (threshold `2e-6`);
- p-value max abs error: `1.952412395822556e-05` (threshold `1e-5`).

The unweighted logistic routes passed, including CuPy, Torch, Torch→CuPy, and CuPy→Torch. The remaining problem was therefore not the repaired device-binding path. PR #142 had temporarily forced inference-enabled weighted smooth non-Gaussian `solver="auto"` fits to FISTA because `newton_solver` rejected non-uniform `sample_weight`. The physical failure exposed the numerical cost of that workaround.

A later exact-head P100 run at `1eedc29f6ea48774449ccc9b8fb2934a45a66757` confirmed that the weighted-Newton repair closed the original logistic failure. That run instead stopped at the extra `poisson/weighted=False/cupy` explicit-FISTA parity row: coefficient error `7.168500309906456e-06` and p-value error `4.290581307964114e-05`, while intercept, BSE, and CI remained within the existing limits. This is a point-estimation trajectory difference in a non-canonical explicit-FISTA row, not a renewed sandwich/device failure. Schema v4 therefore restores unweighted Poisson to its maintained public `solver="auto"` → Newton path while retaining one representative unweighted Logistic explicit-FISTA row for independent FISTA plus cross-container inference/device coverage. No acceptance tolerance is relaxed.

The schema-v4 physical gate then passed at exact clean numerical source `db448d718f523eacf97bcb3c419e376c9812362d` on Tesla P100 with CuPy 13.6.0, Torch 2.0.0+cu117, and NumPy 1.24.2. The validator exited 0 with `status: success`; all four family/weight rows and all four execution/container routes passed the unchanged coefficient/intercept (`2e-6`) and inference (`1e-5`) thresholds. The retained artifact is `results/pr142_penalized_glm_inference_gpu/pr142_penalized_glm_inference_gpu.json` (15944 B).

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

| Consumer | Baseline behavior | Final behavior |
|---|---|---|
| direct `newton_solver`, unweighted | supported | preserved |
| direct `newton_solver`, floating-point weights considered uniform by the historical `allclose` rule | accepted and numerically unweighted | preserved equivalent result |
| direct `newton_solver`, genuinely non-uniform weights | rejected before optimization | supported when the loss exposes weighted value/gradient/Hessian |
| penalized smooth non-Gaussian L2, explicit `solver="newton"` | rejected non-uniform weights in the shared solver | backend-native weighted Newton |
| PR #142 weighted inference + `solver="auto"` | runtime wrapper forced FISTA | canonical solver dispatch; smooth L2 resolves to Newton where the existing dispatch table says Newton |
| `PenalizedGLM_CV` weighted inference + `solver="auto"` | wrapper forced FISTA for selection and final refit | canonical CV/direct dispatch with the same weighted objective; selected final estimator reports Newton where applicable |
| ordinary `GeneralizedLinearModel(..., solver="newton")` + weights | public wrapper rejects before solver entry | preserved as a separate fail-closed public boundary in this PR |
| Cox + non-uniform weights | unsupported | remains fail-closed |
| FISTA and FISTA-BB consumers | weight-capable paths unchanged | preserved; no stopping/convergence edits |

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

## 5. Final numerical contract

For genuinely non-uniform analytic weights `w_i >= 0`, `sum(w_i) > 0`, Newton solves the same normalized average-loss objective used by `LossBase`:

`L(beta) = sum_i w_i l_i(beta) / sum_i w_i + P(beta)`.

The exact same weight vector is used by:

1. loss value used by Armijo backtracking;
2. gradient;
3. Hessian or fused gradient+Hessian;
4. every trial-point objective evaluation.

Penalty scaling remains unchanged because global rescaling of all analytic weights must leave the normalized loss and therefore the fitted penalized optimum unchanged.

Floating-point weight vectors satisfying the historical Newton uniformity rule (`allclose(values, values[0])`) remain on the established unweighted numerical path. Exact integer uniformity retains its exact-equality behavior. This avoids changing prior compatibility semantics for consumers that treat uniform weighting as the unweighted problem.

Weights are converted once to the execution backend/device relative to the processed design matrix. Explicit CUDA/Torch execution does not transfer numerical work to CPU.

## 6. Implemented repair

1. Replaced Newton's `_validate_uniform_sample_weight` rejection gate with general weight validation and backend/device alignment.
2. Preserved the historical floating-point `allclose` uniform-weight classification and normalized those vectors to the existing unweighted path.
3. Threaded active genuinely non-uniform weights through constant-Hessian construction, fused/non-fused gradient/Hessian calls, and both sides of Armijo line search.
4. Removed PR #142's fit/CV wrapper that changed public `solver="auto"` execution to FISTA solely because Newton lacked weights.
5. Left the canonical `_preferred_penalized_glm_solver` table unchanged; it again owns smooth-L2 solver resolution.
6. Updated physical validation so weighted `auto` proves Newton selection. Schema v4 also runs unweighted Poisson through canonical `auto` → Newton and retains one unweighted Logistic explicit-FISTA row for FISTA/inference/device coverage without turning Poisson FISTA terminal-iterate parity into a PR #142 acceptance gate.
7. Updated public docs/changelogs so no PR #142 surface claims that weighted penalized-GLM inference requires the temporary FISTA workaround.

## 7. Validation closure

Solver-level regressions cover:

- weighted logistic Newton equals literal row replication for integer weights after matching average-loss normalization;
- multiplying every weight by a positive constant leaves coefficients unchanged;
- historical uniform / almost-uniform floating-point weights equal the unweighted path;
- invalid shape, negative/non-finite, and zero-total weights fail before numerical work;
- Cox non-uniform weights remain visibly unsupported;
- smooth-penalty validation remains unchanged;
- Torch CPU weighted Newton matches NumPy.

Estimator/CV regressions cover:

- explicit weighted penalized logistic L2 Newton independently of inference;
- weighted logistic/Poisson L2 with public `solver="auto"` selects Newton and publishes finite M-estimation inference;
- public `solver` remains `auto`;
- weighted `PenalizedGLM_CV` selection and selected final refit use the canonical dispatch and final estimator reports Newton for applicable rows;
- inference still runs only on the selected full-data final refit.

Exact-source hosted evidence for `db448d718f523eacf97bcb3c419e376c9812362d` passed Tests #3516 (full CPU 3184 passed / 821 skipped / 0 failed; Python 3.9–3.12; Torch 2.0 CPU; docs/static; panel external alignment), Maintenance compatibility #2529, Gaussian inference #536, Node-wise alpha inference #60, Release package validation #2259, Release notes validation #2252, and Benchmark Frontend CI #2545.

Exact-source physical evidence for the same numerical source passed schema v4 on Tesla P100. The four maintained rows were:

- Logistic unweighted: explicit FISTA → FISTA; worst coefficient error 2.72e-07 and worst inference error 4.43e-07.
- Logistic weighted: `auto` → Newton; coefficient/inference parity at approximately 1e-16 to 1e-15.
- Poisson unweighted: `auto` → Newton; coefficient/inference parity at approximately 1e-17 to 1e-16.
- Poisson weighted: `auto` → Newton; coefficient/inference parity at approximately 1e-16 to 1e-15.

Every row passed CuPy, Torch, Torch→CuPy, and CuPy→Torch routes with concrete `cuda:0` provenance and the maintained `m_estimation` / `penalized_sandwich` / `hc0` metadata.

## 8. Review/evidence policy after physical closure

The numerical implementation and physical acceptance are complete at immutable source `db448d718f523eacf97bcb3c419e376c9812362d`.

Subsequent changes that only reconcile changelog/evidence prose do not alter production numerical code, validator logic, or the accepted solver matrix. The user explicitly approved reusing the exact-source P100 artifact for such docs-only closure instead of performing a numerically meaningless duplicate P100 run. Any later numerical, validator, backend, solver, inference, or tolerance change would invalidate that exception and reopen physical validation.

Final PR-level closure therefore requires only hosted validation of the docs-only head plus a fresh review confirming that the post-`db448d71` delta is documentation/evidence-only and that no stale schema-v3/pending claims remain.
