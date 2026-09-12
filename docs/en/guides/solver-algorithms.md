# Solver Algorithms

> Language: English  
> Last updated: 2026-09-12  
> This page: Algorithm reference  
> Switch: [Chinese](../../cn/guides/solver-algorithms.md)

## Overview

statgpu provides first-order, second-order, proximal, and closed-form solvers. Most model users should start with `solver="auto"`; this page is the algorithm-level reference and therefore keeps the mathematical update rules, convergence criteria, backend behavior, and important capability boundaries explicit.

Three rules are useful when reading this page:

1. Backend support does not imply that every loss/penalty/weight combination is supported.
2. `sample_weight` does not change an explicitly requested solver. If the requested weighted combination is unsupported, fitting raises an error.
3. Weight support is route-specific. In particular, non-uniform direct L-BFGS weights are supported by maintained GLM losses, not automatically by every `LossBase` implementation.

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

**Use case**: Quantile regression with SCAD/MCP penalties. It combines an IRLS quadratic majorization of the pinball loss with local linear approximation (LLA) for the non-convex penalty.

### Algorithm

For each continuation value of `alpha`:

1. **LLA outer loop.** Compute the local penalty derivative

   $$
   d_j=P'(|\beta_j|),
   $$

   using the current SCAD/MCP iterate. The implementation uses the threshold vector

   $$
   t_j=n\,d_j.
   $$

2. **IRLS-CD inner loop.** With residuals

   $$
   r_i=y_i-x_i^\top\beta,
   $$

   define

   $$
   q_i=\begin{cases}
   \tau, & r_i\ge 0,\\
   1-\tau, & r_i<0,
   \end{cases}
   \qquad
   w_i^{\mathrm{IRLS}}=\frac{q_i}{\max(|r_i|,\varepsilon)}.
   $$

   If analytic `sample_weight=s` is supplied, statgpu first normalizes it so that

   $$
   \tilde s_i=\frac{n s_i}{\sum_j s_j},
   $$

   and uses

   $$
   w_i=\tilde s_i\,w_i^{\mathrm{IRLS}}.
   $$

3. **Parallel diagonal majorization.** Let `W = diag(w)`. The implementation computes

   $$
   g=X^\top W(y-X\beta),
   \qquad
   h=\operatorname{diag}(X^\top W X),
   $$

   then updates all coordinates in parallel with

   $$
   u=g+h\odot\beta,
   \qquad
   \beta_j^{\mathrm{new}}=\frac{S(u_j,t_j)}{h_j},
   $$

   where

   $$
   S(u,t)=\operatorname{sign}(u)\max(|u|-t,0).
   $$

   This is a Jacobi-style parallel diagonal-majorization update rather than a cyclic coordinate-descent sweep.

4. **Convergence.** Stop the IRLS inner loop when

   $$
   \|\beta^{\mathrm{new}}-\beta\|_\infty<\texttt{tol},
   $$

   and stop the LLA outer loop when

   $$
   \|\beta-\beta_{\mathrm{before\,LLA}}\|_\infty<\texttt{lla\_tol}.
   $$

### Continuation and defaults

- continuation path: `lambda_max` to target `alpha`;
- `max_lla_per_step=2` by default;
- `lla_tol=1e-6` by default;
- `tol=1e-6` by default;
- GPU convergence checks remain on device except for the final boolean synchronization.

### Backend

- NumPy: NumPy matrix operations;
- CuPy: CuPy matrix operations and GPU kernels where available;
- Torch: device-native tensor operations.

---

## 2. Proximal Newton

**File**: `statgpu/solvers/_proximal_newton.py`

**Use case**: Smooth losses with L2/no penalty, where an ordinary Newton system is well defined.

For a general non-smooth composite objective

$$
F(\beta)=\ell(\beta)+P(\beta),
$$

a true proximal-Newton step would solve a Hessian-metric proximal subproblem such as

$$
\Delta_k
=\arg\min_{\Delta}
\left\{
\nabla\ell(\beta_k)^\top\Delta
+\frac12\Delta^\top H_k\Delta
+P(\beta_k+\Delta)
\right\}.
$$

Applying an ordinary Euclidean proximal operator to a Newton step would optimize a different composite objective. The maintained implementation therefore uses Newton only on the smooth L2/no-penalty path; non-smooth requests warn and delegate to FISTA until a correct Hessian-metric proximal subproblem is implemented.

### Algorithm

At iterate $\beta_k$, write the smooth objective as

$$
F(\beta)=\ell(\beta)+P(\beta),
$$

where $P$ is L2 or zero on this path. Compute

$$
g_k
=\nabla\ell(\beta_k)+\nabla P(\beta_k),
$$

and

$$
H_k
=\nabla^2\ell(\beta_k)+\nabla^2P(\beta_k).
$$

The implementation symmetrizes the Hessian and adds a small ridge stabilization,

$$
\widetilde H_k
=\frac12\left(H_k+H_k^\top\right)+10^{-10}I.
$$

If

$$
\|g_k\|_2\le \texttt{tol},
$$

optimization stops.

The Newton system is then solved as

$$
\widetilde H_k d_k=g_k.
$$

The code uses a subtractive direction convention, so trial points are

$$
\beta_k(t)=\beta_k-t d_k.
$$

This is equivalent to the more common notation $p_k=-\widetilde H_k^{-1}g_k$ followed by $\beta_k+t p_k$. If the linear solve is classified as singular or ill-conditioned, the current implementation **does not call a least-squares solver**; it instead falls back to

$$
d_k=g_k,
$$

which is steepest descent under the subtractive convention above.

Before line search, descent is checked explicitly. Because the update is $\beta_k-t d_k$, a valid descent direction requires

$$
g_k^\top d_k>0.
$$

If $g_k^\top d_k$ is non-finite or non-positive, the implementation again sets

$$
d_k=g_k,
\qquad
 g_k^\top d_k=\|g_k\|_2^2.
$$

### Armijo backtracking

Starting from

$$
t_0=1,
$$

the solver accepts the first step satisfying

$$
F(\beta_k-t d_k)
\le
F(\beta_k)-c\,t\,g_k^\top d_k,
$$

with

$$
c=10^{-4}.
$$

If the condition fails, the step is halved,

$$
t\leftarrow \frac{t}{2},
$$

for at most 25 trials. The first accepted point becomes

$$
\beta_{k+1}=\beta_k-t d_k.
$$

If all 25 trials fail, the solver restores

$$
\beta_{k+1}=\beta_k,
$$

emits a line-search warning, and stops the current solve.

### Defaults and backend

- default `max_iter=50`;
- default `tol=1e-6`;
- supported NumPy/CuPy/Torch routes use the corresponding native linear algebra.

---

## 3. FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

**File**: `statgpu/solvers/_fista.py`

**Use case**: Smooth data-fit term plus a penalty with a proximal operator.

### Algorithm

Initialize

$$
\beta_0=y_0,\qquad t_0=1.
$$

At iteration $k$:

1. evaluate the smooth gradient

   $$
   g_k=\nabla \ell(y_k);
   $$

2. take the proximal step

   $$
   \beta_{k+1}=\operatorname{prox}_{\alpha/L}
   \left(y_k-\frac{1}{L}g_k\right);
   $$

3. update Nesterov momentum

   $$
   t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2},
   $$

   $$
   y_{k+1}=\beta_{k+1}
   +\frac{t_k-1}{t_{k+1}}(\beta_{k+1}-\beta_k);
   $$

4. apply the maintained convergence rule, including coefficient-change checks such as

   $$
   \|\beta_{k+1}-\beta_k\|_1<\texttt{tol}
   $$

   on the corresponding route.

### Weighted path

On maintained weighted routes, `sample_weight` is converted to the selected backend and the data-fit gradient uses the normalized weighting convention, for example

$$
g=\frac{X^\top(s\odot\psi)}{\sum_i s_i}.
$$

Weighted objective tracking uses the same normalization. The existence of a weighted FISTA implementation does not by itself imply that every estimator/loss combination supports weights.

### GPU path

Supported GPU routes keep gradient evaluation, proximal updates, momentum updates, and most convergence/divergence checks on device and batch synchronization where possible.

### Defaults

- default `max_iter=500`;
- default `tol=1e-6`.

---

## 4. FISTA-BB (Barzilai-Borwein)

**File**: `statgpu/solvers/_fista_bb.py`

**Use case**: FISTA with adaptive Barzilai-Borwein step sizes, especially on supported GLM sparse-penalty routes.

### Algorithm

Let

$$
s_{k-1}=\beta_k-\beta_{k-1},
\qquad
y_{k-1}=\nabla\ell(\beta_k)-\nabla\ell(\beta_{k-1}).
$$

The two standard BB step estimates are

$$
\alpha_k^{\mathrm{BB1}}
=\frac{\langle s_{k-1},s_{k-1}\rangle}
{\langle s_{k-1},y_{k-1}\rangle},
$$

and

$$
\alpha_k^{\mathrm{BB2}}
=\frac{\langle s_{k-1},y_{k-1}\rangle}
{\langle y_{k-1},y_{k-1}\rangle}.
$$

The maintained route alternates BB1/BB2 on its schedule, applies step bounds, and uses adaptive restart when momentum conflicts with the descent direction.

BB updates are disabled for SCAD/MCP and their group variants because LLA reweighting can change the effective subgradient abruptly and make secant-based steps unreliable.

---

## 5. FISTA-LLA

**File**: `statgpu/solvers/_fista_lla.py`

**Use case**: SCAD, MCP, adaptive-L1, and related non-convex or iteratively reweighted penalties.

### Algorithm

1. Build a continuation path from `lambda_max` to the requested `alpha` (maintained defaults use five steps, or three on non-smooth routes).
2. At each continuation step, run the LLA outer loop.
3. Compute the local penalty weights at the current coefficient vector.
4. Solve the resulting convex surrogate with backend-native FISTA.
5. Stop the LLA loop when

   $$
   \|\beta-\beta_{\mathrm{before\,LLA}}\|_1<\texttt{lla\_tol}.
   $$

Supported GPU routes use fused proximal/momentum kernels and batch scalar checks to reduce device-to-host synchronization.

---

## 6. IRLS (Iteratively Reweighted Least Squares)

**Implementation**: Loss/family-specific `irls()` methods.

### Quantile IRLS

The current `QuantileLoss.irls()` implementation uses the Frisch-Newton-style reweighting implemented in the code. Starting from an initial coefficient vector (OLS when no explicit initialization is supplied), each iteration computes

$$
r_i=y_i-x_i^\top\beta
$$

and

$$
w_i^{\mathrm{IRLS}}
=\frac{\tau+(1-2\tau)\mathbf 1\{r_i<0\}}
{\max(|r_i|,\varepsilon)}.
$$

If analytic `sample_weight=s` is supplied, it is normalized to sum to $n$,

$$
\tilde s_i=\frac{n s_i}{\sum_j s_j},
$$

and the effective weight is

$$
w_i=\tilde s_i w_i^{\mathrm{IRLS}}.
$$

With $W=\operatorname{diag}(w)$, the unpenalized update solves

$$
(X^\top W X+\varepsilon I)\beta_{\mathrm{new}}=X^\top W y.
$$

For an L2 penalty, the maintained path adds the corresponding diagonal ridge term. When `fit_intercept=True`, the intercept coordinate is excluded from the penalty. Convergence is checked with

$$
\|\beta_{\mathrm{new}}-\beta\|_2<\texttt{tol}.
$$

### GLM IRLS

GLM IRLS has the same high-level weighted-least-squares structure, but its working response and working weights are family/link-specific. They should not be conflated with analytic `sample_weight`.

---

## 7. Newton-Raphson

**File**: `statgpu/solvers/_newton.py`

**Use case**: Smooth losses with L2/no penalty and Hessian support.

### Algorithm

For gradient $g$ and Hessian $H$ of the declared objective,

$$
d=-H^{-1}g
$$

is the Newton direction. The maintained solver then performs Armijo backtracking (up to the maintained retry limit) and applies a small ridge stabilization where required for numerical conditioning.

### Analytic `sample_weight`

For routes that expose weighted curvature, Newton uses one normalized objective throughout value, gradient, Hessian, and Armijo trials:

$$
L(\beta)=\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}+P(\beta).
$$

Multiplying all active weights by one positive constant therefore leaves the optimum unchanged. Uniform/effectively-uniform weights retain the historical unweighted numerical path where that compatibility route is defined.

---

## 8. L-BFGS / L-BFGS-B

**Files**: `statgpu/solvers/_lbfgs.py`, `statgpu/solvers/_lbfgs_b.py`

**Use case**: Smooth losses with smooth/no penalty when a limited-memory quasi-Newton method is preferable to forming a full Hessian.

### L-BFGS algorithm

L-BFGS uses the standard limited-memory two-loop recursion with Armijo line search (history size `m=10` on the maintained route). The current objective, every line-search candidate, and the accepted-point gradient are evaluated under the same declared objective.

### Analytic `sample_weight`

Non-uniform weighted direct L-BFGS is opt-in at the loss level:

| Direct L-BFGS route | Non-uniform `sample_weight` |
|---|---|
| Maintained `GLMLoss` | ✅ Supported |
| Generic robust / quantile / Cox `LossBase` consumers | ❌ Not implied by unweighted support |

For maintained GLMs, the initial gradient, current objective, every line-search candidate, and accepted-point gradient use the same normalized weight vector. NumPy, CuPy, and Torch execution stays on the selected numerical backend.

`L-BFGS-B` is a separate box-constrained implementation and should not be assumed to inherit every `lbfgs_solver` weighting capability.

---

## 9. ADMM (Alternating Direction Method of Multipliers)

**File**: `statgpu/solvers/_admm.py`

For a variable split $\beta=z$, the maintained structure is

1. coefficient update

   $$
   \beta^{k+1}=\arg\min_\beta
   L(\beta)+\frac{\rho}{2}\|\beta-z^k+u^k\|_2^2;
   $$

2. proximal split-variable update

   $$
   z^{k+1}=\operatorname{prox}_{P/\rho}(\beta^{k+1}+u^k);
   $$

3. scaled-dual update

   $$
   u^{k+1}=u^k+\beta^{k+1}-z^{k+1}.
   $$

The implementation may adapt `rho` according to its residual rule.

---

## 10. `exact` (closed-form path)

**Implemented in**: `_fit_mixin._solve_exact_*`

**Use case**: Squared-error + L2 rows where the maintained dispatch selects the closed-form/eigendecomposition path, based on systems of the form

$$
\left(\frac{X^\top X}{n}+\alpha I\right)\beta=\frac{X^\top y}{n},
$$

with the estimator-specific intercept treatment applied separately.

---

## Solver dispatch

For direct model fitting, `solver="auto"` follows the maintained model-level table. A simplified view is:

```text
direct fit with solver="auto"
├── squared_error + L2 + NumPy/CPU → exact
├── squared_error + L2 + GPU       → Newton
├── squared_error + sparse penalty → FISTA/FISTA-BB
├── smooth non-Gaussian GLM + L2   → Newton
├── SCAD/MCP/adaptive path          → LLA + FISTA-family inner solve
├── quantile                        → quantile-specific FISTA/IRLS path
└── group penalty                   → group-aware FISTA / FISTA-LLA
```

`PenalizedGLM_CV` has a related but intentionally separate smooth-L2 policy. In particular, Gamma, Inverse-Gaussian, and Negative-Binomial L2 CV/final-refit routes use L-BFGS, while logistic, Poisson, and Tweedie L2 rows use Newton. Consult the compatibility matrix rather than inferring CV behavior from the direct-fit tree.

`sample_weight` does not change an explicitly requested solver. Unsupported weighted combinations raise instead of selecting a different solver.

## References

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- O'Donoghue, B. & Candes, E. (2015). Adaptive Restart for Accelerated Gradient Schemes. *Foundations of Computational Mathematics*, 15(3), 715-732.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods for Minimizing Composite Functions. *SIAM J. Optimization*, 24(3), 1420-1443.
- Boyd, S. et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in ML*, 3(1), 1-122.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
- Zou, H. & Li, R. (2008). One-step Sparse Estimates in Nonconcave Penalized Likelihood Models. *Annals of Statistics*, 36(4), 1509-1533.