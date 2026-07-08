# Solver Algorithms

> Language: English  
> Last updated: 2026-07-01

## Overview

statgpu provides 10 solvers for penalized loss minimization. This page documents the algorithm, convergence criteria, backend support, and hyperparameters for each solver.

## Solver Summary

| Solver | Best For | Backend Support |
|--------|----------|:---:|
| Proximal IRLS-CD | quantile + SCAD/MCP | numpy, cupy, torch |
| Proximal Newton | Huber/Bisquare/Cox + SCAD/MCP | numpy, cupy, torch |
| FISTA | general non-smooth penalties | numpy, cupy, torch |
| FISTA-BB | GLM + sparse penalties | numpy, cupy, torch |
| FISTA-LLA | nonconvex penalties (continuation path) | numpy, cupy, torch |
| IRLS | smooth losses + L2 | numpy, cupy, torch |
| Newton | smooth losses + L2 | numpy, cupy, torch |
| L-BFGS | smooth losses, moderate dims | numpy, cupy, torch |
| L-BFGS-B | box-constrained problems | numpy |
| ADMM | sum of separable penalties | numpy, cupy, torch |
| exact | squared_error + L2 (closed-form) | numpy, cupy, torch |

---

## 1. Proximal IRLS-CD

**File**: `statgpu/solvers/_proximal_irls_quantile.py`

**Use case**: Quantile regression + SCAD/MCP penalties. Combines IRLS majorization of the non-smooth pinball loss with LLA for non-convex penalties.

### Algorithm

1. **Continuation path**: λ_max → target α (geometric sequence, 3 steps)
2. **LLA outer loop** (2-5 iterations per step):
   a. Compute LLA weights: w_j = P'(|β_j|) from SCAD/MCP
   b. **IRLS-CD inner loop**:
      - Compute IRLS weights: w_i = τ_i / max(|r_i|, ε)
      - Quadratic majorization: Q(β) = ½ Σ w_i(y_i − X_iβ)²
      - Parallel diagonal majorization (Jacobi step):
        g = X' @ W @ (y − Xβ)
        h = diag(X' @ W @ X)
        β = S(g + h·β, n·α·w) / h
      - Check convergence: max(|β_new − β_old|) < tol

### Convergence

- IRLS inner: max coefficient change < tol (typically 1e-6)
- LLA outer: max coefficient change < lla_tol
- GPU: convergence check stays on device, only syncs bool

### Backend

- numpy: all matrix ops via numpy
- cupy: ElementwiseKernel for LLA weights, cupy for matrix ops
- torch: device-native ops, `.to(device)` for sample_weight

### Hyperparameters

| Parameter | Default | Description |
|---|---:|---|
| max_lla_per_step | 2 | Max LLA iterations per continuation step |
| lla_tol | 1e-6 | LLA convergence tolerance |
| max_iter | 200 | Max IRLS iterations per LLA step |
| tol | 1e-6 | IRLS convergence tolerance |

---

## 2. Proximal Newton

**File**: `statgpu/solvers/_proximal_newton.py`

**Use case**: Smooth losses with Hessian (Huber, Bisquare, Cox PH) + non-smooth penalties (SCAD/MCP via LLA). Converges in 5-10 iterations.

### Algorithm

1. Compute Hessian H = X'WX and gradient g = X'ψ / n
2. Newton direction: d = -H⁻¹·g
3. Armijo line search (max 25 retries):
   a. Trial point: β_try = proximal(β − step·d, step)
   b. Check composite Armijo: f(β_try) + g(β_try) ≤ f(β) + g(β) + c·step·g'd
   c. Halve step if not satisfied
4. If Hessian singular or g'd ≤ 0: fall back to gradient descent

### Convergence

- Gradient norm: ||∇f + prox(∇g)|| < tol (typically 1e-6)
- Line search failure after 25 retries → warning

### Backend

- Backend-detect via `_resolve_backend("auto", X)`
- numpy/cupy/torch for linalg.solve and matrix ops

### Hyperparameters

| Parameter | Default | Description |
|---|---:|---|
| max_iter | 50 | Max Newton iterations |
| tol | 1e-6 | Convergence tolerance |

---

## 3. FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

**File**: `statgpu/solvers/_fista.py`

**Use case**: General solver for any loss + any penalty with proximal operator.

### Algorithm

1. Initialize β₀, y₀ = β₀, t₀ = 1
2. For k = 1, 2, ...:
   a. Compute gradient: g_k = ∇ℓ(y_k)
   b. Proximal step: β_{k+1} = prox(β_k − (1/L)·g_k, α/L)
   c. Nesterov momentum: t_{k+1} = (1 + √(1+4t_k²))/2
      y_{k+1} = β_{k+1} + ((t_k−1)/t_{k+1})(β_{k+1} − β_k)
   d. Check convergence: ||β_{k+1} − β_k||₁ < tol

### GPU Async Path

When conditions met (GPU backend + non-smooth penalty + CV/quadratic):
- Gradient computation on device
- Fused proximal + momentum kernel
- Batch convergence/divergence/Lipschitz checks

### Weighted Path

- Convert sample_weight to backend-native array at entry
- Weighted gradient: g = X' @ (sw * ψ) / Σsw
- Weighted objective tracking in GPU path

### Hyperparameters

| Parameter | Default | Description |
|---|---:|---|
| max_iter | 500 | Max FISTA iterations |
| tol | 1e-6 | Convergence tolerance |

---

## 4. FISTA-BB (Barzilai-Borwein)

**File**: `statgpu/solvers/_fista_bb.py`

**Use case**: FISTA with adaptive BB step sizes. Good for GLM + sparse penalties on GPU.

### Algorithm

1. FISTA body with Nesterov momentum
2. Instead of fixed L⁻¹ step, use BB1 or BB2:
   - BB1 (long): α_k = ⟨s_{k-1}, s_{k-1}⟩ / ⟨s_{k-1}, y_{k-1}⟩
   - BB2 (short): α_k = ⟨s_{k-1}, y_{k-1}⟩ / ⟨y_{k-1}, y_{k-1}⟩
   where s = β_k − β_{k-1}, y = ∇ℓ(β_k) − ∇ℓ(β_{k-1})
3. Alternate BB1/BB2 every 2 iterations
4. Adaptive restart (O'Donoghue & Candes 2015): reset momentum when it opposes descent direction
5. Step bounds: [L/step_max_factor, L·step_max_factor]

### Disabled for Non-convex Penalties

BB steps are disabled for SCAD/MCP/group MCP/group SCAD. The abrupt subgradient changes from LLA reweighting amplify noise through the BB step, causing divergence.

---

## 5. FISTA-LLA

**File**: `statgpu/solvers/_fista_lla.py`

**Use case**: Non-convex penalties (SCAD/MCP/adaptive L1) via LLA. Runs the continuation path + LLA + FISTA/proximal Newton in one fused function.

### Algorithm

1. **Continuation path**: λ_max → target α (5 steps, 3 for non-smooth)
2. **LLA outer** (2-5 iterations per step):
   a. Compute LLA weights from SCAD/MCP at current β
   b. **Inner solver**:
      - Losses with Hessian → Proximal Newton (5-10 iter)
      - Losses without Hessian → FISTA (300+ iter)
   c. LLA convergence: ||β − β_before_lla||₁ < lla_tol

### Fused Kernels (GPU)

- Squared error + GPU: fused proximal + momentum kernel (X'X precomputed)
- Generic path: fused gradient clipping + proximal + momentum
- Batch GPU syncs: convergence + divergence + Lipschitz in one D2H transfer

---

## 6. IRLS (Iteratively Reweighted Least Squares)

**Implementation**: Each loss class has its own `irls()` method.

**Use case**: Smooth penalties (L2, none) with GLM or quantile losses.

### Algorithm (Quantile IRLS)

1. Initialize β₀ = OLS estimate
2. For each iteration:
   a. Compute residuals: r = y − Xβ
   b. IRLS weights: w_i = (τ + (1−2τ)·1_{r_i<0}) / max(|r_i|, ε)
   c. Solve weighted LS: (X'WX + n·α·I)β = X'Wy
   d. ||β_new − β|| < tol → stop

### Algorithm (GLM IRLS)

Same pattern but weights from GLM working response: (y−μ)/Var(μ)·g'(μ)²

---

## 7. Newton-Raphson

**File**: `statgpu/solvers/_newton.py`

**Use case**: Smooth losses + L2 penalty. Fast convergence when Hessian is positive-definite.

### Algorithm

1. Compute gradient g = ∇ℓ(β) + λ·β and Hessian H = ∇²ℓ(β) + λ·I
2. Newton direction: d = -H⁻¹·g
3. Armijo line search with backtracking (max 25)
4. Ridge regularization: 1e-10·I for stability

---

## 8. L-BFGS / L-BFGS-B

**Files**: `statgpu/solvers/_lbfgs.py`, `statgpu/solvers/_lbfgs_b.py`

**Use case**: Smooth losses + L2, moderate dimensions, non-canonical GLM links.

### Algorithm

Standard L-BFGS with Armijo line search. History size m=10. Fused GLM gradient + penalty gradient in one call.

---

## 9. ADMM (Alternating Direction Method of Multipliers)

**File**: `statgpu/solvers/_admm.py`

**Use case**: Any loss + any penalty (alternative formulation).

### Algorithm

1. β-update: argmin L(β) + (ρ/2)||β − z + u||² (via sub-solver)
2. z-update: proximal operator on z
3. u-update: u = u + β − z
4. Adaptive ρ: increase by 10% every 10 iterations if primal residual > 10·dual

---

## 10. exact (Closed-form)

**Implemented in**: `_fit_mixin._solve_exact_*`

**Use case**: squared_error + L2 penalty on numpy. Eigendecomposition of X'X/n + αI.

---

## Solver Dispatch Logic

```
fit() with solver="auto"
├── squared_error + L2 + numpy → exact (eigendecomposition)
├── squared_error + L2 + GPU  → newton
├── SCAD/MCP/adaptive → fista (LLA wrapper)
│   ├── squared_error → fista_lla (fused)
│   ├── quantile      → proximal_irls_cd
│   ├── has_hessian   → fista_lla → proximal_newton
│   └── no_hessian    → fista_lla → fista
├── quantile (any penalty) → fista
├── squared_error + sparse → fista
├── GLM + GPU + sparse → fista_bb (if size < 2M elements)
├── CV + L2 (loss-specific) → lbfgs / newton
├── smooth penalties + smooth losses → newton / irls
└── default sparse → fista_bb
```

## References

- Beck, A. & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm. *SIAM J. Imaging Sciences*, 2(1), 183-202.
- Barzilai, J. & Borwein, J. M. (1988). Two-Point Step Size Gradient Methods. *IMA J. Numer. Anal.*, 8(1), 141-148.
- O'Donoghue, B. & Candes, E. (2015). Adaptive Restart for Accelerated Gradient Schemes. *Foundations of Computational Mathematics*, 15(3), 715-732.
- Lee, J. D., Sun, Y. & Saunders, M. A. (2014). Proximal Newton-Type Methods for Minimizing Composite Functions. *SIAM J. Optimization*, 24(3), 1420-1443.
- Boyd, S. et al. (2011). Distributed Optimization and Statistical Learning via ADMM. *Foundations and Trends in ML*, 3(1), 1-122.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
- Zou, H. & Li, R. (2008). One-step Sparse Estimates in Nonconcave Penalized Likelihood Models. *Annals of Statistics*, 36(4), 1509-1533.
