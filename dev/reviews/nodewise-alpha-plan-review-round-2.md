# Node-wise alpha plan review — round 2

Reviewed plan commit: `cf71a6915b307763391663be5e0de255c9f8a474`
Base master: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
Verdict: **PLAN CHANGES REQUIRED**

Round 1 findings are closed. The remaining issues are numerical-execution details that must be decided before implementation.

## Findings

### HIGH — node-wise solver convergence is not a production acceptance gate

The revised plan validates design scales and `C_j`, but an unconverged node-wise Lasso can still produce finite positive `C_j` and a plausible `M`. Current internal solves use hidden `max_iter=500`, `tol=1e-5`, usually with coefficient-delta stopping. The repair must define a node-wise solver convergence contract and verify KKT residuals before publishing each precision row.

Recommended plan change:

- central internal constants such as `NODEWISE_MAX_ITER=500`, `NODEWISE_TOL=1e-5` for v1;
- run node-wise FISTA with `stopping="kkt"` where the maintained solver supports it;
- independently recompute the standardized Lasso KKT residual from `Sigma_Z`, `gamma_j`, and `nodewise_alpha` before accepting the row;
- fail closed if the residual exceeds the declared tolerance/bound;
- record convergence tolerance/max-iter in result metadata.

This does not require exposing `nodewise_tol` as a new public parameter in the same API change.

### HIGH — cache identity currently names the wrong tolerance concept

The plan says the cache includes solver tolerance, but the current implementation hashes `self._tol` while the actual node-wise solve uses a fixed `tol=1e-5`. The revised contract must explicitly decouple main-model tolerance from node-wise solver tolerance and key the cache by the **actual node-wise numerical contract**, not the parent estimator's `tol` unless those are intentionally made the same.

### MEDIUM — GPU Lipschitz bound must be recomputed on `Sigma_Z`

The CuPy/Torch batched path currently derives a global FISTA Lipschitz constant from the unstandardized `Sigma_hat`. Once the node-wise objective is defined on `Z`, the valid smooth-gradient Lipschitz constant is determined by the standardized Gram matrix `Sigma_Z`. Reusing the old bound can make the step size inconsistent with the new objective.

### MEDIUM — publication should be atomic

Resolve alpha and build/validate `M` in local candidate state. Publish `nodewise_alpha_`, `_debiased_M_cpu`/native equivalent, and result metadata only after every node-wise row, KKT gate, marginal report, and required finalizer stage succeeds. Existing cleanup remains the safety net, not the primary publication mechanism.

### MEDIUM — direct-fit coefficient invariance needs an explicit test

The plan protects CV selection/final coefficients but should also require direct `Lasso` and `ElasticNet` penalized `coef_`/`intercept_` to be identical when only `nodewise_alpha` changes. The parameter is inference-only.

### MEDIUM — metadata should record the internal node-wise numerical contract

For reproducibility, add at least `nodewise_tol`, `nodewise_max_iter`, and `nodewise_stopping="kkt"` to inference metadata. These are not public constructor controls in v1, but they affect the computed `M`.

## Required fixes

1. Make KKT/convergence a production node-wise acceptance gate.
2. Define internal node-wise tol/max-iter/stopping constants and use them consistently in cache identity.
3. Recompute GPU FISTA Lipschitz scaling from `Sigma_Z`.
4. Specify atomic result publication.
5. Add direct-fit prediction-coefficient invariance tests.
6. Record internal node-wise numerical settings in metadata.

After these changes, perform a fresh third review rather than inheriting this verdict.