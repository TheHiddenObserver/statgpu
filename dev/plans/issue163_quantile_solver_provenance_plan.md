# Issue #163 — Quantile solver provenance reconciliation plan

## Goal

Make penalized Quantile requested/resolved/executed solver identity truthful across generic direct fit, the typed `PenalizedQuantileRegression` wrapper, and `PenalizedGLM_CV(loss="quantile")`.

## Classification

Existing-capability solver/API reconciliation. Active axes: solver dispatch, public API/provenance, direct fit, CV/final refit, tests, EN/CN documentation. This is not a new Quantile numerical method.

## Current mismatch

- The canonical `solver="auto"` dispatch table labels all Quantile rows as FISTA.
- `_fit_loss_backend()` silently runs `QuantileLoss.irls()` from inside the `solver_name == "fista"` branch for L2/no-penalty Quantile fits.
- Therefore smooth Quantile `auto` and explicit `solver="fista"` may report FISTA while numerically executing IRLS.
- Sparse L1/ElasticNet Quantile paths genuinely use FISTA-family solvers; SCAD/MCP uses the dedicated Proximal IRLS-CD continuation path.

## Intended contract

1. Quantile + L2/no penalty + `solver="auto"` resolves truthfully to ordinary Quantile IRLS.
2. Explicit `solver="irls"` remains supported only for L2/no penalty.
3. Quantile + L1/ElasticNet + `solver="auto"` remains FISTA-family and never routes through ordinary IRLS.
4. Explicit `solver="fista"` must execute FISTA if that combination is maintained; if smooth Quantile FISTA is not numerically maintained, reject the explicit request before iteration rather than substituting IRLS.
5. SCAD/MCP remains a separate dedicated Proximal IRLS-CD algorithm; its provenance/documentation must not be mislabeled as ordinary IRLS.
6. Direct generic, typed wrapper, CV candidate/fold path, and selected final refit must agree on requested/resolved/executed identity.

## Implementation reconnaissance

Before code edits:

- trace `_preferred_penalized_glm_solver`, `_select_solver`, `_fit_loss_backend`, `_solver_for_cv`, and `_refit_best`;
- inventory which Quantile penalties reach optimized CV paths versus generic estimator fallback;
- verify whether explicit FISTA for smooth Quantile is actually numerically sound. Do not make it public merely to avoid an error.

## Likely implementation

- Split Quantile auto dispatch by penalty class: smooth → IRLS, convex sparse → FISTA/FISTA-BB as maintained, non-convex → dedicated continuation path.
- Remove or make unreachable the hidden `fista -> irls` substitution for smooth Quantile.
- Add a precise pre-dispatch rejection for explicit smooth-Quantile FISTA if current FISTA is not maintained for the non-smooth check-loss geometry.
- Ensure CV uses the same helper and selected final refit carries the same solver identity.

## Tests

Cover at minimum:

- generic + typed direct L2/no-penalty `auto` resolves/executes IRLS;
- explicit IRLS L2/no-penalty;
- explicit FISTA smooth Quantile either genuinely executes FISTA or fails visibly according to the selected contract;
- L1 and ElasticNet auto remain FISTA-family;
- SCAD/MCP remain dedicated Proximal IRLS-CD and do not report ordinary IRLS;
- `PenalizedGLM_CV(loss="quantile")` candidate solver and final refit provenance agree for L2 and a sparse penalty;
- sample weights and formula routing are preserved where those consumers already support them;
- NumPy and Torch CPU hosted coverage, plus deterministic CUDA skips where physical devices are unavailable.

## Documentation

Reconcile EN/CN:

- Quantile model page;
- solver × penalty matrix;
- loss × penalty × solver framework;
- solver algorithms dispatch summary;
- release/changelog surfaces.

Learner pages should state current behavior, not Issue/PR tracking status.

## Validation and review loop

1. Implement the smallest truthful routing repair.
2. Run targeted tests and exact-head hosted CI.
3. Fresh-review the whole PR diff under `.claude/skills/code-review`.
4. Fix findings and re-review the new exact head from scratch.
5. If the repair changes only solver labeling/routing to an algorithm already executed on that same backend, physical CUDA is required only if the actual maintained CUDA numerical path changes; otherwise record preservation evidence explicitly.
