# Solver Algorithms

> Language: English  
> Last updated: 2026-09-12  
> This page: Algorithm reference  
> Switch: [Chinese](../../cn/guides/solver-algorithms.md)

## Overview

statgpu provides a collection of first-order, second-order, proximal, and closed-form solvers. Most model users should start with `solver="auto"`; this page is an algorithm-level reference for understanding what the solvers do and where their capability boundaries come from.

Three rules are worth keeping in mind while reading the tables below:

1. **Backend support is not the same as model support.** A solver may run on NumPy, CuPy, and Torch while a particular loss/penalty/weight combination is still unsupported.
2. **Explicit solver requests stay explicit.** Adding `sample_weight` does not silently replace Newton/L-BFGS with another solver.
3. **Weight support is route-specific.** In particular, non-uniform direct L-BFGS weights are supported by maintained GLM losses, not automatically by every `LossBase` implementation.

For model-level dispatch, see [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md).

## Solver summary

| Solver | Best for | Backend support |
|--------|----------|:---:|
| Proximal IRLS-CD | quantile + SCAD/MCP | numpy, cupy, torch |
| Proximal Newton | smooth loss + L2/no penalty; non-smooth requests use FISTA | numpy, cupy, torch |
| FISTA | general non-smooth penalties | numpy, cupy, torch |
| FISTA-BB | GLM + sparse penalties | numpy, cupy, torch |
| FISTA-LLA | non-convex penalties via continuation/LLA | numpy, cupy, torch |
| IRLS | losses with a maintained IRLS representation | numpy, cupy, torch |
| Newton | smooth losses with Hessian support | numpy, cupy, torch |
| L-BFGS | smooth losses, moderate dimensions | numpy, cupy, torch |
| L-BFGS-B | box-constrained smooth problems | numpy, cupy, torch |
| ADMM | separable/proximal formulations | numpy, cupy, torch |
| exact | squared error + L2 closed-form path | numpy, cupy, torch |

The backend column describes numerical implementation capability only. Estimator and loss contracts can further narrow the valid combinations.

---

## 1. Proximal IRLS-CD

**File**: `statgpu/solvers/_proximal_irls_quantile.py`

**Use case**: Quantile regression with SCAD/MCP penalties. It combines an IRLS majorization of the pinball loss with local linear approximation (LLA) for the non-convex penalty.

### Algorithm

1. **Continuation path**: move from $\lambda_{\max}$ to the target $\alpha$ on a short geometric path.
2. **LLA outer loop**:
   - compute the local SCAD/MCP weights from the current coefficients;
   - solve the resulting weighted L1-like surrogate with an IRLS-coordinate-descent inner loop;
   - stop when the LLA coefficients stabilize.
3. **IRLS-CD inner loop**:
   - construct the quadratic majorizer for the pinball loss;
   - compute the weighted gradient/diagonal curvature;
   - apply coordinate-wise soft-thresholding;
   - stop when the coefficient change is below tolerance.

### Convergence

- IRLS inner loop: maximum coefficient change below `tol`;
- LLA outer loop: maximum coefficient change below `lla_tol`;
- GPU convergence checks remain on device except for the final boolean synchronization.

### Backend

- NumPy: NumPy linear algebra and array operations;
- CuPy: CuPy matrix operations and GPU kernels where available;
- Torch: device-native tensor operations.

### Hyperparameters

| Parameter | Default | Description |
|---|---:|---|
| `max_lla_per_step` | `2` | Maximum LLA iterations per continuation step |
| `lla_tol` | `1e-6` | LLA convergence tolerance |
| `max_iter` | `200` | Maximum IRLS iterations per LLA step |
| `tol` | `1e-6` | IRLS convergence tolerance |

---

## 2. Proximal Newton

**File**: `statgpu/solvers/_proximal_newton.py`

**Use case**: Smooth losses with L2/no penalty, where an ordinary Newton system is well defined.

A general non-smooth proximal-Newton method requires a proximal subproblem in the Hessian metric. A Euclidean-prox shortcut would optimize a different composite objective, so direct non-smooth requests do not silently use that approximation. They warn and use FISTA instead; FISTA-LLA likewise keeps its backend-native FISTA inner solve until a correct Hessian-metric proximal capability is explicitly implemented.

### Algorithm

1. Compute the declared objective gradient and Hessian.
2. Solve the Newton system; use least squares only for a genuine rank failure.
3. Run Armijo backtracking on the full declared objective.
4. If the Newton direction is not a descent direction, fall back to steepest descent.

### Convergence

The maintained implementation uses its gradient/step convergence checks together with the Armijo acceptance rule. A line-search failure is surfaced rather than treated as a successful step.

### Backend

The numerical backend is resolved from the model/device request and input arrays; supported NumPy/CuPy/Torch paths use the corresponding native linear algebra.

---

## 3. FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

**File**: `statgpu/solvers/_fista.py`

**Use case**: General proximal solver for a smooth data-fit term plus a penalty with a proximal operator.

### Algorithm

1. Initialize $\beta_0$, the momentum point, and the Nesterov scalar.
2. At each iteration:
   - evaluate the smooth gradient at the momentum point;
   - take a proximal-gradient step;
   - update Nesterov momentum;
   - test the maintained convergence rule.

### GPU asynchronous path

On supported GPU routes, gradient evaluation, proximal updates, momentum updates, and most convergence/divergence checks remain device-native. Synchronization is batched where possible.

### Weighted path

On maintained weighted routes:

- `sample_weight` is converted to the selected backend once at solver entry;
- the data-fit gradient uses the normalized weighted convention;
- weighted objective tracking uses the same normalization.

Weight semantics still come from the estimator/loss route; the generic existence of a weighted FISTA implementation does not automatically validate every model combination.

### Hyperparameters

| Parameter | Default | Description |
|---|---:|---|
| `max_iter` | `500` | Maximum FISTA iterations |
| `tol` | `1e-6` | Convergence tolerance |

---

## 4. FISTA-BB (Barzilai-Borwein)

**File**: `statgpu/solvers/_fista_bb.py`

**Use case**: FISTA with adaptive Barzilai-Borwein step sizes, especially useful for supported GLM sparse-penalty routes.

### Algorithm

FISTA-BB keeps the Nesterov/proximal structure of FISTA but adapts the step size using BB1/BB2 secant information from successive coefficient and gradient differences. The implementation also uses adaptive restart when momentum conflicts with descent.

### Non-convex penalties

BB updates are disabled for SCAD/MCP and their group variants. LLA reweighting can change the effective subgradient abruptly, which makes secant-based BB steps unreliable on those continuation paths.

---

## 5. FISTA-LLA

**File**: `statgpu/solvers/_fista_lla.py`

**Use case**: SCAD, MCP, adaptive-L1, and related non-convex/iteratively reweighted penalties.

### Algorithm

1. Build a short continuation path from a large regularization value to the requested `alpha`.
2. At each continuation step, run an LLA outer loop.
3. Each LLA iteration replaces the non-convex penalty by its current convex surrogate and solves that surrogate with backend-native FISTA.
4. Stop when the LLA coefficients stabilize.

A future proximal-Newton inner path is gated on an explicit and mathematically correct Hessian-metric proximal implementation; it is not silently approximated today.

### GPU path

Supported GPU routes use fused proximal/momentum kernels and batch scalar checks to reduce device-to-host synchronization.

---

## 6. IRLS (Iteratively Reweighted Least Squares)

**Implementation**: Loss/family-specific IRLS methods.

**Use case**: Losses for which statgpu exposes a maintained IRLS representation, typically with L2/no penalty.

### Generic pattern

1. Build the current working response and working weights.
2. Solve the corresponding weighted least-squares surrogate.
3. Update coefficients and repeat until the maintained convergence rule is met.

For GLMs, the working response/weights are determined by the family and link. Quantile-specific IRLS uses a different majorization and should not be confused with GLM analytic `sample_weight`.

---

## 7. Newton-Raphson

**File**: `statgpu/solvers/_newton.py`

**Use case**: Smooth losses with L2/no penalty and Hessian support. It is attractive when second-order curvature is stable and the parameter dimension is moderate.

### Algorithm

1. Evaluate the gradient and Hessian of the declared objective.
2. Solve the Newton system.
3. Use Armijo backtracking to select an acceptable step.
4. Apply the maintained small ridge stabilization where required for numerical conditioning.

### Analytic `sample_weight`

For loss/estimator routes that expose weighted curvature, Newton supports non-uniform analytic weights through one normalized objective:

$$
L(\beta)=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

The same weights are used in objective values, gradients, Hessians, and every Armijo trial. Therefore multiplying all active weights by one positive constant leaves the optimum unchanged.

Uniform/effectively-uniform weights retain the historical unweighted numerical path where that compatibility route is defined.

---

## 8. L-BFGS / L-BFGS-B

**Files**: `statgpu/solvers/_lbfgs.py`, `statgpu/solvers/_lbfgs_b.py`

**Use case**: Smooth losses with smooth/no penalty when a limited-memory quasi-Newton method is preferable to forming a full Hessian.

### L-BFGS algorithm

L-BFGS uses the standard limited-memory two-loop recursion and Armijo line search. The current objective, every line-search candidate, and the accepted-point gradient are all evaluated under the same declared objective.

### Analytic `sample_weight`

Non-uniform weighted direct L-BFGS is intentionally opt-in at the loss-contract level:

| Direct L-BFGS route | Non-uniform `sample_weight` |
|---|---|
| Maintained `GLMLoss` | ✅ Supported |
| Generic robust / quantile / Cox `LossBase` consumers | ❌ Not implied by unweighted support |

For maintained GLMs, the same normalized weight vector is used in the initial gradient, current objective, every line-search candidate, and accepted-point gradient. NumPy, CuPy, and Torch execution stays on the selected numerical backend.

Uniform weights remain compatible with the historical unweighted L-BFGS route. A loss that supports unweighted L-BFGS does **not** automatically support genuine non-uniform `sample_weight`.

`L-BFGS-B` is a separate box-constrained implementation and should not be assumed to inherit every `lbfgs_solver` weighting capability.

---

## 9. ADMM (Alternating Direction Method of Multipliers)

**File**: `statgpu/solvers/_admm.py`

**Use case**: Supported separable/proximal formulations where variable splitting is useful.

### Generic pattern

1. Update the primary coefficient variable under the smooth objective plus the augmented quadratic term.
2. Update the split variable through the declared proximal operator.
3. Update the scaled dual variable.
4. Adapt the penalty parameter according to the maintained residual rule.

---

## 10. `exact` (closed-form path)

**Implemented in**: `_fit_mixin._solve_exact_*`

**Use case**: Squared-error + L2 rows where the maintained dispatch selects the closed-form/eigendecomposition path.

---

## Solver dispatch

For direct model fitting, `solver="auto"` follows the maintained model-level table. A simplified view is:

```
direct fit with solver="auto"
├── squared_error + L2 + NumPy/CPU → exact
├── squared_error + L2 + GPU       → Newton
├── squared_error + sparse penalty → FISTA/FISTA-BB
├── smooth non-Gaussian GLM + L2   → Newton
├── SCAD/MCP/adaptive paths         → LLA + FISTA-family inner solve
├── quantile                        → quantile-specific FISTA/IRLS path
└── group penalties                 → group-aware FISTA / FISTA-LLA
```

`PenalizedGLM_CV` has a related but intentionally separate smooth-L2 policy. In particular, Gamma, Inverse-Gaussian, and Negative-Binomial L2 CV/final-refit routes use L-BFGS, while logistic, Poisson, and Tweedie L2 rows use Newton. Consult the compatibility matrix rather than inferring CV behavior from the direct-fit tree.

`sample_weight` never silently rewrites an explicit solver request. If a requested weighted route is unsupported, statgpu raises instead of substituting another solver.

## References

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- O'Donoghue, B. & Candes, E. (2015). Adaptive Restart for Accelerated Gradient Schemes. *Foundations of Computational Mathematics*, 15(3), 715-732.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods for Minimizing Composite Functions. *SIAM J. Optimization*, 24(3), 1420-1443.
- Boyd, S. et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in ML*, 3(1), 1-122.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
- Zou, H. & Li, R. (2008). One-step Sparse Estimates in Nonconcave Penalized Likelihood Models. *Annals of Statistics*, 36(4), 1509-1533.
