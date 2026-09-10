# Node-wise alpha plan review — round 3

Reviewed plan commit: `ca84b50b550af1b13bb8f8ab04761095167f0286`
Base master: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
Verdict: **PLAN CHANGES REQUIRED**

This is a fresh review. Round-1/2 fixes were rechecked rather than inherited; they remain sound. Three remaining design issues should be closed before implementation.

## Findings

### HIGH — weighted automatic-alpha sample-size convention breaks zero-weight equivalence

The plan defines automatic alpha with the raw row count `n` of the transformed working problem. For analytic weights, however, a zero-weight observation contributes nothing to the weighted objective/Gram matrix. Adding or removing zero-weight rows should not change the precision problem, yet raw-row `n` changes `sqrt(2 log p / n)`.

The plan needs a weight-aware, response-independent effective sample-size convention. A defensible v1 default is the Kish-style design-weight effective sample size

`n_eff = (sum w)^2 / sum(w^2)`

clipped/validated in `(0, n]`, with `n_eff=n` when weights are omitted. It is invariant to global weight scaling, equals the number of positive rows for equal nonzero weights plus arbitrary zero-weight rows, and shrinks when a few observations dominate the analytic weights.

This is a statgpu default heuristic, not a theorem claim. The existing #138 weighted working-data transform remains authoritative; it should additionally make the normalized weight/effective-n information available to the node-wise alpha resolver without reconstructing the transform inside the precision helper.

Required tests: all-one identity, global weight-scale identity, zero-weight-row add/drop invariance, and highly unequal finite weights.

### HIGH — leaving p=1 backend-specific is inconsistent with a newly shared public nodewise contract

The plan explicitly preserves historical p=1 behavior even though the changed capability is shared across NumPy/CuPy/Torch. A new public `nodewise_alpha` contract should not retain an avoidable backend semantic split.

For `p=1` no nuisance node-wise regression is needed. On standardized `Z`, `Sigma_Z=[1]`, so the approximate precision is exactly `Theta_Z=[1]` and `M_X=[1/d_1^2]`. Define this analytic univariate path consistently on NumPy/CuPy/Torch.

Because no node-wise Lasso solve occurs, `nodewise_alpha_` should remain `None` and metadata should use `precision_method="analytic_univariate"` plus `nodewise_alpha_source="not_applicable"`. A supplied `nodewise_alpha` remains a valid stored constructor setting but is not numerically consumed for p=1; document this explicitly. Alternatively, a uniform public rejection would be consistent but would discard an easy exact construction, so analytic support is preferred.

### MEDIUM — use the paper-style tau_j^2 normalizer and treat cross-product equality as a numerical check

The plan normalizes row `j` by

`C_j = Z_j' r_j / n`.

Under the node-wise Lasso KKT equations and the plan's objective scaling,

`Z_j' r_j / n = ||r_j||^2/n + lambda * ||gamma_j||_1`.

The right-hand side is the standard node-wise `tau_j^2` form, is nonnegative by construction, and is less exposed to cancellation. The production normalizer should therefore be

`tau_j_sq = ||r_j||^2/n + nodewise_alpha * ||gamma_j||_1`,

with finite-positive validation. Compute `Z_j' r_j/n` as an equivalence diagnostic and require agreement within a tolerance implied by the KKT residual / floating-point scale. Record max normalizer-consistency error in validation evidence if useful; it need not become a public fitted attribute.

This also makes the documentation map more directly to van de Geer-style node-wise precision notation.

## Required fixes

1. Define a response-independent weighted effective sample size for auto alpha and test zero-weight add/drop invariance.
2. Close p=1 as one shared analytic backend contract instead of preserving a historical split.
3. Use `tau_j^2 = ||r_j||^2/n + lambda||gamma_j||_1` as the production normalizer, with the cross-product form as a KKT/numerical consistency check.

After these fixes, perform a fresh round-4 review.