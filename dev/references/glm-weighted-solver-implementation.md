# Weighted GLM solver implementation notes

This developer reference preserves implementation detail moved from the GLM
learner page at source snapshot `94758a78f3acc741ac0ba2a8c85076763117e7b0`.
It does not record a new numerical or physical-GPU validation run. Public
solver combinations, analytic-weight semantics, and initialization failures
remain in the [GLM model guide](../../docs/en/models/generalized-linear-model.md).

The ordinary public adapter is implemented in
`statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`; Gamma domain
construction lives in `statgpu/glm_core/_gamma.py` and shared solver domain
hooks. Penalized/CV reconstruction uses
`statgpu/linear_model/_inverse_gamma_smooth_domain_contract.py`.
These ownership details are not additional public API guarantees.

### Explicit Newton and L-BFGS with analytic weights

For supported ordinary GLM combinations, explicit `solver="newton"` and `solver="lbfgs"` accept non-uniform analytic weights when that family/link combination supports the solver.

Both solvers optimize the [normalized weighted objective](../../docs/en/models/generalized-linear-model.md#objective-function-and-sample_weight). The same weight vector is used consistently throughout the optimization:

- Newton uses it in objective, gradient, Hessian, and Armijo line-search evaluations;
- L-BFGS uses it in the initial gradient, current objective, every line-search candidate, and the accepted-point gradient.

This consistency is important: a weighted search direction is never paired with an unweighted line-search objective.

Adding `sample_weight` does **not** replace an explicit Newton/L-BFGS request with IRLS or FISTA. Likewise, an explicit CUDA/Torch request does not fall back to CPU. Uniform and historically effectively-uniform weights retain the historical unweighted objective.

#### Initializing `inverse_power` Gamma

`GammaRegression(link="inverse_power")` uses the inverse link, so fitting requires

$$
\eta_i=b+x_i^\top\beta>0.
$$

Before the first objective evaluation, explicit Newton/L-BFGS constructs an interior starting point that satisfies this condition. With an intercept, the intercept column provides an immediate feasible direction. Without an intercept, statgpu searches the positive-weight training rows for a direction $d$ with $Xd>0$ and scales that direction to a valid interior point. If such a start cannot be numerically certified, fitting fails before optimization begins. Subsequent updates are kept inside the valid inverse-link domain by solver-owned feasibility checks and step caps; these are numerical safeguards rather than additional statistical assumptions on the Gamma model.

#### Scope of weighted L-BFGS support

Weighted L-BFGS is a **GLM loss capability**, not a blanket promise for non-GLM `LossBase` implementations. Other model families retain their own weight and solver semantics; consult their model pages and the compatibility matrix. Ordered GLMs also retain their separate weight policy.

Weighted penalized smooth GLMs use the same analytic-weight convention. Their existing direct-fit/CV dispatch remains authoritative: a supported L2 row may use Newton or L-BFGS according to that policy, not merely because weights are present.

