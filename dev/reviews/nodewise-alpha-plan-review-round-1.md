# Node-wise alpha plan review — round 1

Reviewed plan commit: `3b83e3a1a84d18769ba0267ba74fd4aa281a08ec`
Base master: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
Verdict: **PLAN CHANGES REQUIRED**

## Findings

### HIGH — state reset / failed-refit leakage is underspecified

The plan says `nodewise_alpha_` is reset before fit, but the current sparse-inference runtime has multiple installed cleanup layers (`_clear_inference_state`, `_invalidate_failed_sparse_inference_fit`, no-inference cleanup, CV reset). A new fitted value can become stale after a failed refit unless it is explicitly part of the canonical inference-state cleanup chain. The plan must name this as an implementation and test requirement, including LassoCV/ElasticNetCV outer-state propagation/reset.

### HIGH — do not reconstruct centered/weighted working data inside the new node-wise helper

PR #138 already installs the centered average-loss working problem before the original debiased CPU/CuPy/Torch routines are called. The plan's wording can be read as asking a new helper to derive `X_w` again from raw `X, y, w`, which risks double centering/weight scaling. The new node-wise standardization must operate on the **already canonical design supplied to the node-wise precision construction**. Existing #138 working-data wrappers remain authoritative.

### HIGH — degenerate `C_j` must be positive, not merely nonzero in absolute value

For the standardized node-wise Lasso problem, the normalizer

`C_j = Z_j' (Z_j - Z_-j gamma_j) / n`

should be finite and positive under a valid solve. The current `abs(C_j) < 1e-30 -> identity row` fallback can publish a plausible but invalid precision row. The plan should require `C_j > c_tol`, shared across backends, with fail-closed semantics and a regression covering negative/non-finite/tiny values.

### MEDIUM — constructor validation timing is not explicit enough

The plan requires raw-value preservation for sklearn clone semantics but does not say when invalid values are rejected. Add a raw validation helper that checks `None` or a finite real scalar, rejects bool/non-scalar/non-positive values, and does not coerce the stored constructor value. The same validation must be exercised by constructor reconstruction through `set_params`/clone.

### MEDIUM — installed wrapper/finalizer metadata propagation must be explicit

Weighted/centered debiased inference reconstructs `DebiasedInferenceResult` in a later finalizer. The plan should require that node-wise provenance survives this reconstruction and that simultaneous inference observes the same resolved `M`/alpha. A test must cover unweighted, weighted, and centered-GPU publication paths.

### MEDIUM — current p=1 behavior is not scoped

The existing GPU debiased path has an explicit `p >= 2` guard while the CPU path is historically different. This task should not accidentally change the minimum-feature contract while refactoring node-wise setup. The plan must either unify p=1 intentionally or explicitly preserve current support/failure behavior with regression coverage.

### MEDIUM — numerical degeneracy tolerance is undefined

The plan says zero/degenerate design scales and `C_j` fail closed but does not define how tolerances remain backend/dtype consistent. Require one float64, scale-aware validation rule shared by NumPy/CuPy/Torch rather than backend-specific magic constants. The exact threshold may be selected during implementation from analytic/numerical evidence, but it must be centralized and regression-tested before completion.

## Required fixes before implementation

1. Make #138 working-data wrappers authoritative; standardize only the already-resolved node-wise working design.
2. Add node-wise fitted state to every inference reset/failure/CV reset path.
3. Require positive finite `C_j` with one shared tolerance policy.
4. Define raw constructor validation and clone/set_params behavior.
5. Require provenance survival through weighted/centered finalizers and simultaneous inference.
6. Scope p=1 behavior explicitly.
7. Centralize degeneracy tolerance semantics.

No implementation should start until the revised plan closes these findings and receives a fresh review.