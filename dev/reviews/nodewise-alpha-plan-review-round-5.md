# Node-wise alpha plan review — round 5

Reviewed plan commit: `b09408f0b44e9013d7fb035942bd6148cac9ee3c`
Base master: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
Target kind: branch-plan
Verdict: **PLAN REVIEW CLEAN**

This was a fresh review of the revised plan, not an inherited round-4 verdict.

## Rechecked contracts

- Public API is explicit: scalar `nodewise_alpha=None|positive real`, raw-value validation, clone/get_params/set_params behavior, and inference-only semantics.
- The default numerical behavior change is explicitly classified as an intentional statistical correction with no legacy public mode.
- #138 remains authoritative for centered/weighted average-loss working data; the new precision helper does not repeat those transforms or use `y` for tuning.
- Automatic tuning is standardized-design-side and response-independent; weighted tuning uses a documented, overrideable Kish-style effective-n heuristic with global-weight and zero-weight-row invariants.
- Standardization/back-transformation is algebraically consistent: `Z=X_w D^-1`, `Theta_Z` is constructed on `Z`, and `M_X=D^-1 Theta_Z D^-1` is used with the original canonical working design.
- The production node-wise normalizer is the paper-style `tau_j^2 = ||r_j||^2/n + lambda||gamma_j||_1`; `Z_j' r_j/n` is an independently checked KKT-consistency identity rather than the sole normalizer.
- Internal FISTA settings and stopping policy are explicit and provenance-recorded; an independently recomputed full KKT residual is the publication gate.
- CuPy/Torch use standardized Gram matrices and recompute the relevant Lipschitz bound on-device; no explicit-device CPU fallback is introduced.
- `p=1` has one analytic NumPy/CuPy/Torch precision contract and does not pretend to execute a node-wise Lasso.
- Precision/result publication is transactional through all requested marginal/finalizer/simultaneous stages, with node-wise fitted and transient state included in cleanup/reset contracts.
- Cache identity keys the actual node-wise numerical contract, not unrelated parent-estimator tolerance.
- LassoCV/ElasticNetCV propagation is final-refit inference-only and cannot affect main hyperparameter selection.
- Test/evidence plan covers API compatibility, direct-fit invariance, response/feature/weight invariants, bad KKT/normalizer failures, formula/CV paths, independent NumPy reference, old-vs-new simulation evidence, and physical CuPy/Torch acceptance.

## Non-blocking implementation watchpoints

These are implementation-review checks, not plan defects:

1. Validate standardized Gram, back-transformed `M_X`, and candidate reporting arrays are finite before the transaction commits; existing fail-closed reporting guards remain the final safety net.
2. Ensure the transient weighted effective-n context is restored in `finally` blocks on every exception path.
3. If the existing 500-iteration FISTA cap cannot meet the declared independent KKT gate on supported cases, increase/improve the internal solve rather than weakening the gate silently.
4. Do not accidentally widen the task into the adjacent full ElasticNetCV inference-selector redesign.

No CRITICAL, HIGH, or actionable MEDIUM plan finding remains for this exact plan commit and base.