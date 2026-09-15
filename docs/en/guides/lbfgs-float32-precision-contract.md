# L-BFGS Float32 Precision Contract

> Language: English  
> Last updated: 2026-09-15  
> This page: numerical-precision contract  
> Switch: [Chinese](../../cn/guides/lbfgs-float32-precision-contract.md)

## Scope

This page describes the maintained **float32 numerical comparison contract** for ordinary smooth L-BFGS routes. It does not change the L-BFGS algorithm, the declared statistical objective, solver selection, analytic-weight semantics, or explicit backend/device authority.

The algorithm itself is documented in [Solver Algorithms](solver-algorithms.md).

## Why float32 is not a coefficient-by-coefficient backend oracle

L-BFGS is path dependent at finite precision. Small rounding differences in objective/gradient reductions can change late-iteration curvature-pair history, Armijo backtracking, and whether termination occurs through the gradient criterion or through an unrepresentably small parameter step.

Consequently, two valid native-float32 executions can have coefficient differences that are materially larger than their objective-value difference. A NumPy float32 result is therefore **not** treated as an exact coefficient oracle for CuPy or Torch float32 execution.

This distinction is specific to the float32 numerical contract. Float64 remains the strict cross-backend reference regime.

## Maintained acceptance dimensions

A maintained float32 L-BFGS result is assessed jointly through:

1. **objective agreement** — the complete declared objective must agree with the corresponding references at float32-appropriate numerical scale;
2. **stationarity** — the final gradient norm must be small enough to certify a numerically stationary solution, including when a backend terminates through parameter resolution rather than the gradient test;
3. **same-backend float64 comparison** — the float32 solution is compared with the same backend solving the same numerical data in float64;
4. **cross-backend diagnostics** — float32 parameter differences across backends are recorded, but are not interpreted as exact-oracle disagreement by themselves;
5. **analytic-weight rescaling** — multiplying all positive analytic weights by a common constant must preserve the statistical objective, while finite-precision optimization-path differences are judged under the same float32 objective/stationarity contract;
6. **execution fidelity** — diagnostic traces must reproduce the production solver, and public estimator routes must preserve the requested backend/device and executed L-BFGS identity.

The reviewed Issue #160 physical-evidence gate uses the following conservative bounds for the maintained diagnostic fixture matrix:

| Quantity | Float32 acceptance bound |
|---|---:|
| objective absolute difference vs same-backend float64 | `2e-6` |
| parameter max-absolute difference vs same-backend float64 | `2e-3` |
| final gradient norm | `2e-4` |
| objective absolute difference vs NumPy float32 | `2e-6` |
| parameter max-absolute difference vs NumPy float32 | `2e-3` |
| objective difference under global analytic-weight rescaling | `2e-6` |
| parameter max-absolute difference under global analytic-weight rescaling | `1e-3` |
| gradient-norm difference under global analytic-weight rescaling | `2e-4` |

The cross-backend float32 parameter bound is a **diagnostic bound**, not a claim that the NumPy float32 coefficient vector is the mathematically preferred solution.

## Float64 remains strict

Issue #160 does not weaken the float64 L-BFGS contract. In the retained physical matrix, float64 NumPy/CuPy/Torch solutions and global-weight-rescaling controls agree at or near float64 roundoff. Float64 remains the appropriate regime when strict cross-backend parameter reproducibility is required.

## Practical guidance

For ordinary use, no special action is required: statgpu preserves native float32 execution when float32 data reach a maintained L-BFGS route.

If an application requires tight cross-backend coefficient reproducibility rather than float32-level objective/stationarity agreement, use float64 inputs. Do not infer a backend failure solely from a small float32 coefficient discrepancy when the objective, stationarity, same-backend float64 comparison, and execution provenance are all within contract.

The physical evidence and the reviewed Option-A decision are retained in `dev/reviews/issue160_float32_lbfgs_matrix.json` and `dev/reviews/issue160_float32_lbfgs_contract_decision.md`.
