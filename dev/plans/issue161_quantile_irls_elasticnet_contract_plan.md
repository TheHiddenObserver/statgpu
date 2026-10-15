# Issue #161 — Quantile low-level IRLS ElasticNet contract repair plan

## Goal

Reconcile the low-level `QuantileLoss.irls()` contract with the already-correct public estimator boundary. Direct low-level IRLS must not silently optimize only the L2 component of an ElasticNet penalty while ignoring its L1 term.

## Change classification

- Existing capability reconciliation / public-contract repair.
- Active axes: loss, penalty, solver, backend preservation, tests, docs.
- No new numerical capability and no new backend claim.

## Current state

- `PenalizedQuantileRegression(..., solver="auto")` routes convex Quantile problems to FISTA.
- Explicit estimator-level `solver="irls"` is maintained only for L2 / no-penalty Quantile fitting and already rejects ElasticNet and other non-smooth penalties.
- `QuantileLoss.irls()` currently documents ElasticNet as supported and, when it sees `l1_ratio`, adds only the L2 curvature term. It therefore does not solve the declared ElasticNet objective.

## Intended contract

1. `QuantileLoss.irls(..., penalty=None)` remains supported.
2. `QuantileLoss.irls(..., penalty=L2Penalty(...))` remains supported on NumPy/CuPy/Torch.
3. Any non-smooth penalty, including `ElasticNetPenalty`, L1, adaptive/group penalties, SCAD, and MCP, fails before numerical iteration with a precise error directing users to FISTA or the dedicated Quantile non-convex path.
4. Unknown penalty objects fail closed rather than being partially interpreted from incidental attributes such as `alpha` or `l1_ratio`.
5. Existing sample-weight normalization, intercept exclusion from L2 curvature, backend/device behavior, and public estimator dispatch remain unchanged.

## Implementation

- Replace class-name substring heuristics in `QuantileLoss.irls()` with an explicit low-level penalty capability gate based on the declared penalty name.
- Allow only `None` / `l2` / explicit null aliases if they occur at this boundary.
- Remove the ElasticNet-only L2 curvature branch.
- Update the low-level docstring so it no longer claims ElasticNet support.

## Tests

Add a focused regression file covering:

- direct `QuantileLoss.irls()` + `ElasticNetPenalty` rejects before the linear solve;
- direct L1 and another representative non-smooth penalty reject;
- direct L2 remains numerically executable;
- unpenalized IRLS remains executable;
- weighted L2 IRLS remains invariant to positive global rescaling of analytic weights;
- estimator-level `solver="irls" + ElasticNet` remains fail-closed with its existing public error;
- where Torch CPU is available, direct L2 IRLS still preserves backend execution.

No new physical CUDA evidence is required because the repair removes an unsupported route and leaves every supported numerical path unchanged. Hosted NumPy/Torch regressions and the existing GPU-capable implementation contract are the active gate.

## Documentation / release notes

- Keep learner-facing Quantile pages focused on the current user contract: explicit IRLS is L2/no-penalty only; ElasticNet uses FISTA.
- Add a concise Issue #161 changelog entry in root / EN / CN changelogs; do not put Issue tracking prose back into learner-facing model pages.

## Review / completion loop

1. Implement the minimal fail-closed repair and focused tests.
2. Run hosted CI on the exact branch head.
3. Review the complete PR diff under `.claude/skills/code-review` with exact base/head identity.
4. Fix all in-scope findings.
5. Re-run targeted/hosted validation and review the new exact head from scratch.
6. Stop only when the fresh review has no actionable findings and evidence belongs to that exact head.
