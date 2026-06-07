# Cross-Validation Implementation

> Language: English  
> Last updated: 2026-06-07  
> This page: CV architecture, acceleration techniques, and GPU optimization  
> Switch: [Chinese](../cross-validation.md)

Language switch: [Chinese](../cross-validation.md)

## Overview

`PenalizedGLM_CV` performs k-fold cross-validation for penalized generalized linear models across 7 loss families and 10+ penalty types. The implementation uses multiple acceleration techniques to minimize total CV time, with specialized paths for different loss×penalty×device combinations.

## Architecture

```
PenalizedGLM_CV.fit(X, y)
  │
  ├─ 1. Auto-device selection (_effective_cv_device)
  │     └─ Selects CPU/CuPy/Torch based on problem size and loss
  │
  ├─ 2. Alpha grid generation (_generate_alpha_grid)
  │     └─ Generates descending alpha grid from alpha_max
  │
  ├─ 3. CV scoring (_compute_cv_scores)
  │     ├─ Fast path: Ridge eigendecomposition (squared_error + l2)
  │     ├─ Fold-batched path (logistic, poisson, gamma, NB, inv.gauss, tweedie)
  │     ├─ Sparse CV path (squared_error + l1/en)
  │     ├─ LLA path (SCAD/MCP)
  │     └─ General per-fold path (fallback)
  │
  ├─ 4. Best alpha selection
  │
  └─ 5. Refit on full data (_refit_best)
```

## Device Auto-Selection

When `device="auto"`, the CV estimator selects the backend based on problem size and loss×penalty combination:

| Condition | Selected Device | Reason |
|-----------|----------------|--------|
| n×p < 200k | CPU | Kernel launch overhead dominates |
| squared_error + l1/en, p≥256, n×p≥1M | Torch | Batched alpha path benefits |
| logistic + l1/en, n≥5000, n×p≥500k | Torch | Fold-batched path |
| poisson + l1/en, p≥500, n×p≥1M | Torch | Fold-batched path |
| gamma + l1/en, p≥500, n×p≥2M | Torch | Fold-batched path |
| SCAD/MCP, n×p≥1M | Torch | Async FISTA path |
| NB (any penalty) | CPU | Complex gradient overhead |
| Otherwise | CPU | Default fallback |

The thresholds are benchmark-backed and stored in `_effective_cv_device()`.

## CV Scoring Paths

### Path 1: Ridge Eigendecomposition (squared_error + l2)

**When**: `loss="squared_error"`, `penalty="l2"`, `device` is CPU/auto, `sample_weight=None`.

**Method**: Batch eigendecomposition per fold.

```python
# For each fold:
XtX = Xc.T @ Xc              # Centered Gram matrix
eigvals, Q = eigh(XtX)       # One eigendecomposition
# Solve all alphas at once:
coef = Q @ (1/(eigvals + n*alpha) * Q.T @ Xc.T @ yc)
```

**Complexity**: O(p³) per fold (eigendecomposition), independent of n_alphas.

**Why it's fast**: All alphas are solved from a single eigendecomposition. For 20 alphas × 5 folds, this is 5 eigendecompositions instead of 100 model fits.

### Path 2: Fold-Batched CV (logistic, poisson, gamma, NB, inv.gauss, tweedie)

**When**: `loss` is a GLM family, `penalty` is l1/elasticnet, `device` is Torch/CuPy, `strict=False` (two-stage mode).

**Method**: All folds run simultaneously on GPU using mask tensors.

```python
# Setup: all folds share X on device
train_mask = ones(n_samples, n_folds)   # 1 = train, 0 = val
val_mask = zeros(n_samples, n_folds)    # 1 = val, 0 = train

# Per-fold Lipschitz and step sizes
for fold in folds:
    train_mask[val_idx, fold] = 0
    L[fold] = lipschitz(X_train_fold)
    step[fold] = 1 / L[fold]

# FISTA loop: all folds simultaneously
for alpha in alphas:
    for iteration in range(max_iter):
        eta = X @ coef + intercept           # (n, n_folds) matrix
        resid = loss_residual(eta, y) * train_mask
        grad = (X.T @ resid) / n_train_vec   # (p, n_folds) matrix
        coef = proximal(coef - step * grad, alpha * step)
        # Convergence check: all folds at once
        active = active & (delta >= tol)
        if not any(active): break
```

**Key advantages**:
- Single `X @ coef` GEMM for all folds (vs n_folds separate GEMVs)
- Single `X.T @ resid` GEMM for all folds
- Convergence check across all folds in one operation
- No per-fold Python loop overhead

**Supported losses** (with inline gradient formulas):

| Loss | Gradient (residual) | Lipschitz Scaling |
|------|--------------------|--------------------|
| logistic | sigmoid(η) - y | eig_max(X'X) / 4n |
| poisson | exp(η) - y | eig_max(X'X) / n × y_scale |
| gamma | 1 - y/exp(η) | eig_max(X'X) / n × max(y/ȳ) |
| inverse_gaussian | (exp(η) - y) / exp(2η) | eig_max(X'X) / n × y_scale |
| negative_binomial | (exp(η) - y) / (1 + exp(η)) | eig_max(X'X) / n × y_scale |
| tweedie | exp((1-p)·log(μ)) · (μ - y) | eig_max(X'X) / n × y_scale |

### Path 3: Sparse CV (squared_error + l1/elasticnet)

**When**: `loss="squared_error"`, `penalty` is l1/elasticnet.

**Method**: Precomputed Gram matrix + warm-started FISTA.

```python
# Per fold: precompute once
XtX = X_train.T @ X_train
Xty = X_train.T @ y_train

# Warm-start across descending alphas
coef = zeros(p)
for alpha in alphas_sorted_desc:
    for iteration in range(max_iter):
        grad = XtX @ coef - Xty
        coef = proximal(coef - step * grad, alpha * step)
```

**Key advantage**: `XtX` and `Xty` are computed once per fold, reused across all alphas.

### Path 4: LLA Path (SCAD/MCP)

**When**: `penalty` is SCAD or MCP.

**Method**: Local linear approximation (LLA) outer loop + FISTA inner loop.

```python
for alpha in alphas:
    for lla_iter in range(max_lla):
        # LLA: approximate non-convex penalty as weighted L1
        lla_w = scad_penalty.lla_weights(coef)
        inner_penalty = AdaptiveL1Penalty(alpha=1.0, weights=lla_w)
        # FISTA inner solve
        coef = fista_solver(loss, inner_penalty, X, y, init_coef=coef)
```

**For squared_error**: Uses precomputed Gram matrix (same as Path 3).

### Path 5: General Per-Fold (fallback)

**When**: No specialized path applies.

**Method**: Standard per-fold, per-alpha model fitting.

```python
for fold in folds:
    for alpha in alphas:
        model = PenalizedGeneralizedLinearModel(...)
        model.fit(X_train, y_train)
        val_loss = evaluate(model, X_val, y_val)
```

**Used for**: NB with l2, tweedie with l2, any case where specialized paths are unavailable.

## Two-Stage CV

When `cv_strategy="two_stage"`:

1. **Stage 1 (screening)**: Run relaxed CV (reduced max_iter, looser tol) on full alpha grid
2. **Select top-k candidates**: Identify alpha values with best stage-1 scores
3. **Stage 2 (refinement)**: Run strict CV only on candidate alphas

This can skip 50-80% of alphas in the expensive strict pass.

## GPU Acceleration Techniques

### 1. Async FISTA Loop

For non-smooth penalties (l1, elasticnet, SCAD, MCP) on GPU:

```python
# Traditional FISTA: Armijo backtracking = GPU→CPU sync every iteration
for iteration in range(max_iter):
    coef_new = proximal(coef - step * grad, alpha * step)
    if loss(coef_new) > bound:  # GPU→CPU sync!
        step /= 2
        continue

# Async FISTA: no backtracking, conservative fixed step
step = 1 / (L * safety_factor)  # Precomputed, no per-iteration sync
for iteration in range(max_iter):
    coef = proximal(coef - step * grad, alpha * step)
    # All ops stay on GPU
```

**Safety factors**: logistic 2x, gamma 3x, inverse_gaussian 3x, tweedie 5x.

**Sync reduction**: From 2000 syncs (one per iteration) to ~80 (one every 25 iterations).

### 2. torch.compile Fusion

FISTA step operations are fused via `torch.compile`:

```python
@torch.compile
def fista_step(X, coef, step, alpha):
    eta = X @ coef
    mu = torch.exp(eta)
    grad = X.T @ (mu - y) / n
    w = coef - step * grad
    return torch.sign(w) * torch.clamp(torch.abs(w) - alpha*step, min=0)
```

This reduces ~6 kernel launches to 1-2 compiled kernels.

### 3. Device-Side Convergence Check

```python
# CPU path: sync every iteration
delta = float(to_numpy(abs(coef - coef_old)))  # GPU→CPU sync

# GPU path: check every 50 iterations, batch with other checks
if iteration % 50 == 0:
    delta = torch.sum(torch.abs(coef - coef_old), dim=0)
    active = active & (delta >= tol)  # All on GPU
    if not torch.any(active).item():  # One sync point
        break
```

### 4. Batched Validation Scoring

```python
# Per-alpha scoring: 20 syncs
for alpha in alphas:
    val_loss = loss(X_val, y_val, coef)  # GPU→CPU sync
    scores.append(val_loss)

# Batched scoring: 1 sync
scores_dev = []
for alpha in alphas:
    scores_dev.append(loss(X_val, y_val, coef))  # Stay on GPU
scores = to_numpy(torch.stack(scores_dev))  # One sync
```

### 5. Warm-Start Across Alphas

Descending alpha grid (strongest regularization first). Each alpha's solution initializes the next:

```python
coef = zeros(p)
for alpha in alphas_descending:
    coef = fista_solver(init_coef=coef, ...)  # Warm start
```

This reduces iterations by 3-5x compared to cold start.

## Result Caching

See [CV Cache Hash](cv_cache_hash.md) for the cache mechanism that avoids redundant CV runs on identical data.

## Alpha Convention

All penalties use `alpha` consistently across `PenalizedGeneralizedLinearModel` and the specialized wrappers.

| Penalty | statgpu Alpha | sklearn Alpha | Internal Consistency |
|---------|--------------|---------------|---------------------|
| L1 | `alpha` | `alpha` | `Lasso(a) == PGLM(a, penalty='l1')` |
| ElasticNet | `alpha` | `alpha` | `ElasticNet(a) == PGLM(a, penalty='elasticnet')` |
| L2 (Ridge) | `alpha` | `alpha * n` | `Ridge(a) == PGLM(a, penalty='l2')` |

**sklearn mapping**: Ridge requires `statgpu_alpha = sklearn_alpha * n`. Lasso/ElasticNet use the same alpha directly.

Internal consistency is verified to machine precision (diff ~1e-16).

## Performance Characteristics

### CPU vs GPU Break-Even Points

| Loss | p=100 | p=500 |
|------|-------|-------|
| squared_error | CPU wins | GPU wins at n≥2000 |
| logistic | CPU wins | GPU wins at n≥2000 |
| poisson | CPU wins | GPU wins at n≥2000 |
| gamma | CPU wins | GPU wins at n≥5000 |
| NB | CPU wins | CPU wins |

GPU wins at large p because the GEMM operations (`X @ coef`, `X.T @ resid`) dominate, and GPU GEMM throughput exceeds CPU for matrices above ~100×100.

### Fold-Batch vs Per-Fold Speedup

On Tesla P100 (from benchmark):

| Loss | n=2000, p=500 | n=5000, p=500 |
|------|---------------|---------------|
| poisson + l1 | 7.4x | 4.5x |
| gamma + l1 | 6.5x | 9.3x |
| logistic + l1 | 1.4x | 2.5x |

The speedup comes from eliminating per-fold overhead (Lipschitz computation, model initialization, Python loop) and batching GPU operations.
