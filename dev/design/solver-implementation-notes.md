# Solver implementation notes

Audience: contributors maintaining numerical kernels and their public wrappers.

These notes describe implementation choices inspected at PR #168 head
`ccacbb69303d74e50190f68f1e969a462fea3ae0`. They are not public dispatch,
performance, or synchronization guarantees. Public algorithm equations,
supported inputs, return values, and observable warnings belong in the paired
[English](../../docs/en/guides/solver-algorithms.md) and
[Chinese](../../docs/cn/guides/solver-algorithms.md) solver references.

## FISTA-LLA curvature refresh

In [`_fista_lla.py`](../../statgpu/solvers/_fista_lla.py), the generic non-quadratic
inner loop checks the loss's step-scale estimate every 20 iterations after the
initial iteration. After response scaling, it replaces the current estimate
when `L_new > 1.5 * L` or `L_new < L / 1.5`, with
`L = max(L_new, 0.1 * L_base)` and `step = 1 / L`.

The refresh cadence, factor-of-1.5 trigger, and lower clamp are private heuristic
choices. They are distinct from the ordinary FISTA backtracking multiplier,
which appears in the algorithm's quadratic-majorization update. Do not describe
the refresh trigger as a user-configurable tolerance or a convergence guarantee.

The same source's unweighted squared-error GPU specialization precomputes the
Gram products and evaluates `(X.T @ X @ y_k - X.T @ y) / n`. It is gated on
`sample_weight is None`; applying this unweighted expression to weighted data
would change the objective. Public documentation may explain product reuse and
its algebraic equivalence without promising this exact specialization.

## Ordinary FISTA synchronization policy

[`_fista.py`](../../statgpu/solvers/_fista.py) chooses a device-resident loop for
supported non-smooth-penalty CuPy/Torch problems according to the quadratic/CV
conditions and the loss exclusion hint. This loop omits per-trial Armijo
backtracking and combines finite-state, objective, divergence, and convergence
checks at deferred reporting points. Quantile non-smooth paths additionally
apply objective-based restart and step contraction.

Smooth-penalty Quantile GPU fits instead check accepted-iterate behavior every
iteration and combine reporting quantities with the accepted line-search
trial's host-visible decision. These synchronization placements and cadences
are implementation policy, not public promises about host transfers or timing.
Preserve the weighted objective, support restrictions, accepted-iterate
semantics, and observable convergence warnings when changing them.

## ADMM direct-factorization selection

[`_admm.py`](../../statgpu/solvers/_admm.py) currently selects Cholesky when the
loss advertises `_supports_cholesky` and the processed feature count is at most
2000. It forms `A = Hessian + rho * I`, which for squared error is
`X.T @ X / n + rho * I`. The private size cutoff can change independently of
the public ADMM objective.

This branch sets `adaptive_rho=False` so that the prefactorized system does not
become stale. Recognized positive-definiteness/rank failures use the iterative
Nesterov inner solve; unrelated failures remain visible. The iterative step
scale is initialized after a failed factorization as well. Public reference
material retains the linear system, the meaning of `cg_max_iter`/`cg_tol`, and
the fact that rho stays fixed on the direct solve; it does not expose a private
feature-count threshold as a support boundary.

## Quantile continuation routing and accounting

The public wrapper in
[`_fista_lla_group_contract.py`](../../statgpu/solvers/_fista_lla_group_contract.py)
delegates scalar Quantile SCAD/MCP to
[`proximal_irls_quantile_solver`](../../statgpu/solvers/_quantile_proximal_public_contract.py)
when `lla_penalty_factory`, `init_coef`, and `init_intercept` are all `None` and
`return_path` is false. Group penalties, custom factories, warm starts, and
path-reporting calls retain the fused FISTA-LLA engine. This exact branching
policy belongs here; the public reference explains the caller-visible
possibility of different numerical methods, counts, and warnings.

The delegated solver counts completed IRLS updates and reports target-alpha
IRLS/LLA budget exhaustion. The fused engine counts completed proximal updates,
including the update satisfying its stopping rule, and warns when target-alpha
inner FISTA or outer LLA convergence was not established. With `return_path=True`,
the fourth return value stores coefficient/intercept snapshots and cumulative
iteration counts for each alpha. Reporting mode is therefore not necessarily
numerically passive; caller documentation must not imply otherwise.

Quantile group calls use a group-aware convex surrogate factory, and a loss
proxy prevents the generic proximal-Newton inner branch. The Quantile weighted
step-scale proxy retains analytic weights during periodic step-scale refreshes.
These proxy names and call-graph details are implementation mechanics, while
group-norm scaling and weight-normalized objectives are mathematical contracts.

## Shared loss primitives versus solver support

[`LossBase`](../../statgpu/losses/_base.py) derives `value`, `gradient`, and
`fused_value_and_gradient` from per-sample formulas. With analytic weights, the
shared primitive objective is `sum(w_i * loss_i) / sum(w_i)` and its gradient is
`X.T @ (w * score) / sum(w)`. `QuantileLoss` overrides these primitives with its
own corresponding weighted implementations.

The existence of a generic weighted primitive does not authorize a weighted
solver combination. In particular, direct non-uniform L-BFGS weights require
loss-level opt-in; Quantile rejects them, and estimator/CV Quantile L-BFGS is
unsupported even though low-level unweighted/uniform compatibility remains.
Keep this distinction when changing abstractions or documentation.

The five public Quantile input boundaries are enforced in
[`_quantile_solver_guard.py`](../../statgpu/solvers/_quantile_solver_guard.py),
[`_quantile_irls_validation_contract.py`](../../statgpu/losses/_quantile_irls_validation_contract.py),
[`_quantile_proximal_public_contract.py`](../../statgpu/solvers/_quantile_proximal_public_contract.py),
and [`_fista_lla_group_contract.py`](../../statgpu/solvers/_fista_lla_group_contract.py).
Their strict one-dimensional low-level response requirement is separate from
the estimator-level single-column response normalization.
