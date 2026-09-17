# L-BFGS Float32 Numerical Behavior

> Language: English  
> Last updated: 2026-09-17  
> This page: user-facing numerical guidance  
> Switch: [Chinese](../../cn/guides/lbfgs-float32-precision-contract.md)

## Scope

This page explains how to interpret native float32 L-BFGS results across NumPy, CuPy, and Torch. It does not change the L-BFGS algorithm, the statistical objective, analytic-weight semantics, solver selection, or explicit device behavior.

For the algorithm itself, see [Solver Algorithms](solver-algorithms.md).

## Why float32 coefficients can differ across backends

L-BFGS is path dependent at finite precision. Small rounding differences in objective and gradient reductions can change:

- which curvature pairs enter the limited-memory history;
- how many Armijo backtracking steps are taken;
- the final sequence of accepted iterates; and
- whether termination occurs through a gradient criterion or because a further parameter step is below float32 resolution.

As a result, two valid float32 runs can have coefficient differences that are noticeably larger than their difference in objective value.

This means that a NumPy float32 coefficient vector should **not** be treated as an exact coordinate-by-coordinate oracle for CuPy or Torch float32 execution.

## What to compare instead

When checking whether two float32 L-BFGS fits are numerically consistent, look at the complete numerical picture rather than only the largest coefficient difference:

1. **Objective value** — both fits should optimize the same declared objective to a float32-appropriate numerical scale.
2. **Stationarity** — the final gradient or equivalent stopping diagnostic should indicate that the fit is close to a stationary point.
3. **Same-backend float64 result** — when tighter diagnosis is needed, compare a float32 run with the same backend solving the same problem in float64.
4. **Statistical invariances** — transformations that should leave the objective unchanged, such as multiplying all positive analytic weights by the same constant under the normalized-weight convention, should continue to describe the same statistical problem.
5. **Requested execution path** — an explicit backend/device request should still execute on that backend rather than being silently replaced by a CPU solve.

A modest coefficient discrepancy by itself is therefore not sufficient evidence of a backend error when the objective and stationarity agree at the expected float32 scale.

## Analytic weights

For L-BFGS routes that support analytic `sample_weight`, the normalized weighted objective is

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

Multiplying all positive weights by a common constant leaves this objective unchanged. Finite-precision optimization trajectories may still differ slightly, but the statistical target is the same.

See [Penalized GLM inference](penalized-glm-inference.md) for the inferential meaning of analytic weights on supported penalized GLM routes.

## When to use float64

Use float64 when your application requires tighter cross-backend coefficient reproducibility or when small coefficient differences are substantively important.

Float32 is appropriate when native lower-precision execution is desired and objective/stationarity accuracy at float32 scale is sufficient. Float64 reduces the room for backend-specific rounding paths and is the better diagnostic reference when investigating a suspected numerical discrepancy.

## Practical interpretation

If NumPy, CuPy, and Torch float32 L-BFGS runs produce slightly different coefficient vectors:

- first compare the objective values;
- check convergence/stationarity diagnostics;
- confirm that all runs used the same data, objective normalization, weights, penalty, and stopping controls;
- compare each backend with a float64 run when a tighter reference is needed;
- use float64 if the application requires close coefficient agreement rather than merely an equivalent optimized objective.

Do not loosen the statistical objective or change solver semantics merely to force bitwise-like float32 coefficient agreement across backends.

## Related documentation

- [Solver Algorithms](solver-algorithms.md) — L-BFGS update and line-search algorithm
- [Device and GPU Memory](device-and-memory.md) — explicit backend/device behavior
- [Penalized GLM inference](penalized-glm-inference.md) — weighted objective and inference semantics
