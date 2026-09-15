# Issue #163 — Quantile solver provenance reconciliation plan

## Goal

Make penalized Quantile requested/resolved/executed solver identity truthful across generic direct fit, the typed `PenalizedQuantileRegression` wrapper, and `PenalizedGLM_CV(loss="quantile")`, while preserving the caller's requested quantile level in CV scoring.

## Classification

Existing-capability solver/API/statistical-semantics reconciliation. Active axes: solver dispatch, public API/provenance, direct fit, CV/final refit, validation scoring, tests, EN/CN documentation, and proportional physical GPU evidence for changed maintained CUDA routes. This is not a new Quantile numerical method.

## Current mismatch

- The canonical `solver="auto"` dispatch table historically labeled all Quantile rows as FISTA.
- `_fit_loss_backend()` silently ran `QuantileLoss.irls()` from inside the `solver_name == "fista"` branch for L2/no-penalty Quantile fits.
- Therefore smooth Quantile `auto` and explicit `solver="fista"` could report FISTA while numerically executing IRLS.
- Sparse L1/ElasticNet Quantile paths genuinely use FISTA-family solvers; SCAD/MCP uses the dedicated Proximal IRLS-CD continuation path.
- Fresh review also found that non-median `PenalizedGLM_CV(loss="quantile", loss_kwargs={"quantile": q})` could fit candidates at the requested `q` while some optimized validation evaluators fell back to median (`q=0.5`) pinball loss.

## Intended contract

1. Quantile + L2/no penalty + `solver="auto"` resolves truthfully to ordinary Quantile IRLS.
2. Explicit `solver="irls"` remains supported only for L2/no penalty.
3. Quantile + L1/ElasticNet + `solver="auto"` remains FISTA-family and never routes through ordinary IRLS.
4. Explicit `solver="fista"` must execute FISTA if that combination is maintained; if smooth Quantile FISTA is not numerically maintained, reject the explicit request before iteration rather than substituting IRLS.
5. SCAD/MCP remains a separate dedicated Proximal IRLS-CD algorithm; its provenance/documentation must not be mislabeled as ordinary IRLS.
6. `proximal_irls_cd` is an internal resolved/executed provenance label, not a new public explicit `solver=` keyword. User-supplied explicit requests for that spelling fail closed.
7. Direct generic, typed wrapper, CV candidate/fold path, and selected final refit must agree on requested/resolved/executed identity.
8. Quantile CV training and validation scoring use the same requested `q`, including analytic validation weights.
9. Private optimized Quantile CV helpers whose loss-parameter contract is incomplete fall back to the maintained per-fold estimator path rather than being productized inside this reconciliation PR.

## Implementation reconnaissance

Before code edits:

- trace `_preferred_penalized_glm_solver`, `_select_solver`, `_fit_loss_backend`, `_solver_for_cv`, and `_refit_best`;
- inventory which Quantile penalties reach optimized CV paths versus generic estimator fallback;
- verify whether explicit FISTA for smooth Quantile is actually numerically sound. Do not make it public merely to avoid an error;
- trace every Quantile validation-loss evaluator, including weighted and two-stage CV paths, so the configured quantile level is not lost at an optimized scoring boundary.

## Implemented reconciliation

- Split Quantile auto dispatch by penalty class: smooth → IRLS, convex sparse → maintained FISTA-family, non-convex → dedicated Proximal IRLS-CD provenance.
- Added pre-dispatch rejection for explicit smooth-Quantile FISTA/FISTA-BB and incompatible explicit non-convex solver requests.
- Kept `proximal_irls_cd` internal; `PenalizedGLM_CV` may pass the resolved label to its private SCAD/MCP child estimator only inside a call-local context.
- Added a call-local Quantile CV scoring context so optimized NumPy validation evaluation uses the caller's configured `q` instead of defaulting to 0.5.
- Routed incomplete private Quantile fold-batched sparse and generic SCAD/MCP fast helpers back to the maintained general/per-fold estimator path instead of adding a new Quantile residual/fast-solver implementation.

## Tests

Coverage includes:

- generic + typed direct L2/no-penalty `auto` resolves/executes IRLS;
- explicit IRLS L2/no-penalty;
- explicit FISTA/FISTA-BB smooth Quantile fails visibly before backend work;
- L1 auto remains FISTA-family;
- SCAD/MCP remain dedicated Proximal IRLS-CD and do not report ordinary IRLS;
- `proximal_irls_cd` cannot be requested explicitly as a public keyword;
- `PenalizedGLM_CV(loss="quantile")` candidate policy and final refit provenance agree for L2, sparse, and SCAD paths;
- public `cv_strategy="two_stage"` L1 and SCAD paths complete through the maintained per-fold fallback after incomplete private fast helpers decline;
- formula routing and installer import-order/signature/idempotence boundaries;
- weighted and unweighted non-median Quantile CV scores agree with manual pinball loss and differ visibly from the median objective;
- incomplete private Quantile fast helpers fail safely back to maintained paths.

## Documentation

Reconcile EN/CN:

- Quantile model page;
- solver × penalty matrix, including the internal-only `proximal_irls_cd` boundary;
- loss × penalty × solver framework;
- solver algorithms dispatch summary;
- root/EN/CN release/changelog surfaces.

Learner pages state current behavior, not Issue/PR tracking status.

## Validation and review loop

1. Implement the smallest truthful routing/scoring repair.
2. Run targeted tests and exact-head hosted CI.
3. Fresh-review the whole PR diff under `.claude/skills/code-review`.
4. Fix findings and re-review the new exact head from scratch.
5. Because the final repair changes publicly reachable `cv_strategy="two_stage"` Quantile CUDA execution paths by routing incomplete sparse and non-convex private fast helpers back to the maintained per-fold estimator implementation, hosted CI alone is not final acceptance.
6. Freeze the schema-v1 matrix in `dev/benchmarks/validate_quantile_solver_provenance_gpu.py`, and publish evidence only through `dev/benchmarks/run_quantile_solver_provenance_gpu_gate.py`. The wrapper requires the same clean HEAD before and after the matrix run, verifies the inner schema/source/status, and writes the artifact only after those checks pass.
7. Run the exact-source wrapper with both CuPy CUDA and Torch CUDA. The matrix covers direct L2/L1/SCAD solver provenance, full weighted pinball + SCAD penalized-objective parity, non-median weighted L2 CV score/alpha parity, and public two-stage L1/SCAD CV stage-1/strict/final-refit provenance. It makes no timing or universal performance claim.
8. Only after exact-head hosted CI, fresh review, and the exact-source physical CUDA artifact all pass may PR #164 be marked merge-ready.
