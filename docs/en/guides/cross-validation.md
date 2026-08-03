# Cross-Validation

> Language: English  
> Last updated: 2026-08-03
> This page: Unified CV guide — API reference, architecture, GPU acceleration, and caching  
> Switch: [Chinese](../../cn/guides/cross-validation.md)

---

## Part I: User Guide

### Overview

statgpu provides cross-validated estimators for all penalized models. Each CV estimator automatically searches over a grid of regularization parameters and selects the best one via k-fold cross-validation.

| CV Estimator | Base Model | Penalty | Path |
|-------------|------------|---------|------|
| `RidgeCV` | `Ridge` | l2 | `statgpu.linear_model.RidgeCV` |
| `LassoCV` | `Lasso` | l1 | `statgpu.linear_model.LassoCV` |
| `ElasticNetCV` | `ElasticNet` | elasticnet | `statgpu.linear_model.ElasticNetCV` |
| `LogisticRegressionCV` | `LogisticRegression` | l2 | `statgpu.linear_model.LogisticRegressionCV` |
| `PenalizedGLM_CV` | `PenalizedGeneralizedLinearModel` or `PenalizedCoxPHModel` | family-supported | `statgpu.linear_model.PenalizedGLM_CV` |

### Quick Start

#### RidgeCV

```python
from statgpu.linear_model import RidgeCV

model = RidgeCV(
    alphas=None,           # auto-generate log-spaced grid
    n_alphas=100,          # number of alpha candidates
    cv=5,                  # number of folds
    fit_intercept=True,
    device="auto",         # "cpu", "cuda", or "auto"
)
model.fit(X, y)

print(f"Best alpha: {model.alpha_}")
print(f"CV MSE path shape: {model.cv_results_['mse_path'].shape}")
print(f"R²: {model.score(X_test, y_test):.4f}")
```

#### ElasticNetCV

```python
from statgpu.linear_model import ElasticNetCV

model = ElasticNetCV(
    l1_ratio=0.5,          # or [0.1, 0.5, 0.9] to search over
    alphas=None,
    cv=5,
    device="auto",
)
model.fit(X, y)

print(f"Best alpha: {model.alpha_}")
print(f"Best l1_ratio: {model.l1_ratio_}")
```

#### PenalizedGLM_CV (universal)

```python
from statgpu.linear_model import PenalizedGLM_CV

# Poisson + SCAD with automatic CV
model = PenalizedGLM_CV(
    loss="poisson",
    penalty="scad",
    penalty_kwargs={"a": 3.7},
    cv=5,
    device="auto",
)
model.fit(X, y)
pred = model.predict(X_test)
```

#### Penalized Cox CV

Cox targets must remain two-dimensional throughout selection:

```python
survival_y = np.column_stack([time, event])
model = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="elasticnet",        # l1, l2, elasticnet, scad, or mcp
    l1_ratio=0.4,
    alpha_grid=[0.2, 0.05, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cuda",               # NumPy CPU and Torch CUDA are also supported
).fit(X_cuda, survival_y_cuda)
```

This path uses finite held-out Cox partial-likelihood evidence from every
evaluable fold and refits `PenalizedCoxPHModel` without an intercept. It raises
instead of selecting a default alpha if no candidate has complete evidence.
The branch is estimation-only: it does not publish post-selection coefficient
inference. `two_stage`, sample weights, and dictionary targets are unsupported.

#### LogisticRegressionCV

```python
from statgpu.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(
    cv=5,
    device="auto",
)
model.fit(X, y)
print(f"Best C: {model.C_}")
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

### Parameters Reference

#### Common Parameters (all CV estimators)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cv` | int | `5` | Number of CV folds. Must be >= 2. |
| `random_state` | int | `None` | Random seed for fold shuffling. |
| `device` | str/Device | `"auto"` | `"cpu"`, `"cuda"`, or `"auto"`. |
| `fit_intercept` | bool | `True` | Whether to fit an intercept. |
| `gpu_memory_cleanup` | bool | `False` | Free GPU memory after fit (CuPy). |

#### RidgeCV-Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alphas` | array | `None` | Alpha grid. `None` = auto-generate. |
| `n_alphas` | int | `100` | Number of alphas when auto-generating. |
| `alpha_min_ratio` | float | `1e-3` | Ratio of min to max alpha. |
| `compute_inference` | bool | `False` | Compute SE/p-values/CI after CV. |
| `cov_type` | str | `"nonrobust"` | Covariance type for inference. |
| `gpu_cv_mixed_precision` | bool | `True` | Use float32 for CV (faster on GPU). |

#### ElasticNetCV-Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `l1_ratio` | float/list | `0.5` | L1 mixing. Pass a list to search over multiple values. |
| `alphas` | array | `None` | Alpha grid. |
| `n_alphas` | int | `100` | Number of alphas. |

#### PenalizedGLM_CV-Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss` | str | `"squared_error"` | Loss family (see [Solver x Penalty Matrix](solver-penalty-matrix.md)). |
| `penalty` | str | `"l2"` | Penalty type. |
| `penalty_kwargs` | dict | `{}` | Penalty parameters (e.g., `{"a": 3.7}` for SCAD). |
| `alpha_grid` | array | `None` | Alpha grid. |
| `n_alphas` | int | `100` | Number of alphas. |
| `cv_splits` | list | `None` | Custom fold splits `[(train_idx, val_idx), ...]`. |
| `loss_kwargs` | dict | `{}` | Loss options; Cox accepts `ties="breslow"` or `ties="efron"`. |

### Custom CV Splits

All CV estimators support custom fold generators via `cv_splits`:

```python
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

# Time series CV
tscv = TimeSeriesSplit(n_splits=5)
model = PenalizedGLM_CV(
    loss="poisson", penalty="l1",
    cv_splits=list(tscv.split(X)),
)
model.fit(X, y)

# Stratified CV for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegressionCV(
    cv_splits=list(skf.split(X, y)),
)
model.fit(X, y)
```

When `cv_splits=None` (default), the estimator uses `kfold_indices(n, cv, random_state)` with shuffled folds.

For penalized Cox CV, each custom train/validation pair may be any non-empty,
disjoint split; training need not be the validation complement, and validation
rows need not form a one-time partition across folds. This supports forward
`TimeSeriesSplit` and repeated holdout designs. Indices must be one-dimensional,
exact integers within signed-int64 and sample bounds. Boolean, numeric-string,
fractional, non-finite, overflowing, duplicate, overlapping, or out-of-range
indices are rejected before any candidate fit. Each evaluated Cox train and
validation partition must contain an observed event.

### Sample Weight

Most scalar-response CV estimators support `sample_weight`; see the survival limitation below:

```python
model = RidgeCV(cv=5)
model.fit(X, y, sample_weight=w)
print(f"Weighted R²: {model.score(X_test, y_test, sample_weight=w_test):.4f}")
```

**Limitations** (see [Known Limitations](#known-limitations) below):
- Non-uniform weights with l1/elasticnet/SCAD/MCP raise `ValueError` at the solver level.
- Uniform weights (all equal) work for supported scalar-response penalties.
- `loss="cox_ph"` rejects `sample_weight`; weighted penalized Cox CV is not implemented.

### Alpha Grid

#### Auto-Generated Grid

When `alphas=None`, the grid is generated as:
1. Compute `alpha_max = max(|X'y|) / n` (or weighted variant)
2. Generate `n_alphas` values from `alpha_max` down to `alpha_max * alpha_min_ratio`
3. Grid is log-spaced: `np.logspace(log10(alpha_max * ratio), log10(alpha_max), n_alphas)`

Penalized Cox uses the infinity norm of the partial-likelihood gradient at the
zero model. For ElasticNet with `l1_ratio=rho > 0`, the first value is the
zero-model KKT boundary `alpha_max = ||gradient L(0)||_inf / rho`; a string
penalty uses the estimator's `l1_ratio`, while an `ElasticNetPenalty` object
uses its own value. `rho=0` is pure L2 and has no finite all-zero KKT threshold,
so `||gradient L(0)||_inf` is recorded as an explicit grid heuristic. The
no-penalty aliases `"none"`, `"null"`, and `""` are non-tunable and are rejected
by Cox CV; fit `PenalizedCoxPHModel` directly for an unpenalized run.

#### Custom Grid

```python
import numpy as np

model = RidgeCV(
    alphas=np.logspace(-4, 2, 50),  # custom 50-point grid
    cv=5,
)
model.fit(X, y)
```

Scalar-response CV estimators search strictly positive alpha values. A
one-dimensional numeric user grid keeps its original order after non-positive
and non-finite entries are filtered with a `RuntimeWarning`; an empty or
fully-filtered grid emits a warning and regenerates the default grid. Zero is
therefore not a scalar-CV candidate for L1, L2, ElasticNet, SCAD, or MCP; use a
direct estimator with `alpha=0` for an unpenalized fit. Non-one-dimensional,
complex, boolean, or non-numeric grids raise `ValueError` before device routing
or candidate work. Penalized Cox uses a stricter contract and does not filter
or replace a user grid: non-finite or negative values raise `ValueError`, and
SCAD/MCP additionally require every alpha to be strictly positive. L1, L2, and
ElasticNet Cox grids may include zero.

### Fitted Attributes

After `fit()`, all CV estimators expose:

| Attribute | Description |
|-----------|-------------|
| `alpha_` | Best alpha selected by CV |
| `best_score_` | Best CV score (negative MSE for regression, accuracy for classification) |
| `cv_results_` | Dict with `mse_path`, `alpha_grid`, `best_idx` |
| `estimator_` | Refit model on full data with best alpha |
| `coef_` | Coefficients from refit model |
| `intercept_` | Intercept from refit model |

`ElasticNetCV` additionally has `l1_ratio_` (best l1_ratio if a list was passed).

### Scoring

```python
# Predict
pred = model.predict(X_test)

# Score (R² for regression, accuracy for classification)
r2 = model.score(X_test, y_test)

# Weighted score
r2_w = model.score(X_test, y_test, sample_weight=w_test)
```

`score()` delegates to the refit estimator (`model.estimator_`), so the scoring method matches the base model.

### Device Selection

When `device="auto"`, the CV estimator selects the backend based on problem size:

| Condition | Selected | Reason |
|-----------|----------|--------|
| n*p < 200,000 | CPU | Kernel launch overhead dominates |
| squared_error + l1/en, p>=256, n*p>=1M | Torch GPU | Batched alpha path |
| logistic + l1/en, p>=100, n*p>=500k | Torch GPU | Fold-batched path |
| poisson + l1/en, p>=500, n*p>=1M | Torch GPU | Fold-batched path |
| gamma + l1/en, p>=500, n*p>=2M | Torch GPU | Fold-batched path |
| non-squared-error SCAD/MCP, n*p>=1M | Torch GPU | Async FISTA |
| NB + l1/l2/en | CPU | Complex gradient overhead |
| Generic fallback work < 100M | CPU | Below measured GPU break-even |
| Generic fallback work >= 100M | Torch, then CuPy; CPU if neither is operational | Large aggregate CV work |

For explicit control: `device="cpu"` forces CPU, `device="cuda"` forces GPU. The thresholds are benchmark-backed and stored in `_effective_cv_device()`.
`device="auto"` selects a GPU only after the backend reports an operational
CUDA driver and device; an installed but unusable CuPy wheel does not prevent a
CPU fallback. The generic fallback work is `n * p * n_work_folds * n_alphas`,
with a continuation factor of 20 for non-squared-error SCAD/MCP. Scalar-
response CV uses the normalized generated/custom fold count; Cox CV uses only
folds with events in both training and validation. The earlier empirical
loss/penalty rows are evaluated first and remain driven by their documented
`n * p` and feature conditions, without a fold multiplier. Explicit
`device="cuda"` remains strict and raises when CuPy CUDA is unavailable.

### Inference After CV

For `RidgeCV` with `compute_inference=True`:

```python
model = RidgeCV(compute_inference=True, cov_type="hc1")
model.fit(X, y)

# Standard errors, t-stats, p-values, confidence intervals
print(model.summary())
```

For `PenalizedGLM_CV` with `penalty="l1"` and `compute_inference=True`:
- Debiased Lasso inference is computed via nodewise regression
- Provides SE, z-stat, p-value, and CI for each coefficient

**Status**: l2 inference is fully available. l1 debiased inference is available. ElasticNet/SCAD/MCP inference is not yet implemented.

### Performance Tips

1. **Use GPU for large problems**: n*p > 200k. Set `device="cuda"` or let `auto` decide.
2. **Reduce alpha grid**: `n_alphas=50` is often sufficient; 100 is the default.
3. **Use mixed precision**: `gpu_cv_mixed_precision=True` (default) uses float32 for CV, 2-4x faster on GPU.
4. **Two-stage CV**: `cv_strategy="two_stage"` screens alphas quickly, then refines top candidates.
5. **Custom folds**: Pre-generate folds to avoid re-shuffling across repeated runs.

### See Also

- [Solver x Penalty Compatibility Matrix](solver-penalty-matrix.md) -- full dispatch table and CV fast path details
- [Ridge Model](../models/ridge.md) -- RidgeCV in model context
- [ElasticNet Model](../models/elastic-net.md) -- ElasticNetCV in model context
- [GLM Model](../models/generalized-linear-model.md) -- PenalizedGLM_CV in model context

---

## Part II: Architecture and Implementation

### Architecture

`PenalizedGLM_CV.fit()` dispatches to two preparation and routing sequences.
They share selection and final-refit contracts, but automatic-grid construction
runs at different points and therefore must not be represented as one ordered
pipeline.

#### Scalar-response sequence

```
PenalizedGLM_CV._fit_standard(X, y)
  |
  +-- 1. Validate or generate the complete alpha grid
  |     +-- Automatic grids are generated before CV-device selection
  |
  +-- 2. Materialize generated/custom folds exactly once
  |     +-- One-shot generators become a reusable fold list
  |
  +-- 3. Select the CV device (_effective_cv_device)
  |     +-- Generic work sizing uses len(folds)
  |
  +-- 4. Score the alpha grid (_compute_cv_scores)
  |     +-- Ridge eigendecomposition, fold-batched, sparse, LLA, or fallback
  |
  +-- 5. Select the best alpha and refit
        +-- squared_error + l2: exact float64 eigensolve on CPU
        |   while retaining cv_selected_device_ for prediction/output
        +-- other paths: refit on the resolved selected backend
```

#### Penalized-Cox sequence

```
fit_penalized_cox_cv(estimator, X, (time, event))
  |
  +-- 1. Normalize the survival target and materialize folds
  |
  +-- 2. Validate the alpha-grid request
  |     +-- Explicit grids are validated; automatic grids are not built yet
  |
  +-- 3. Validate event support for every fold
  |     +-- Compute fold_valid and n_effective_folds
  |
  +-- 4. Select the CV device (_effective_cv_device)
  |     +-- Generic work sizing uses n_effective_folds
  |
  +-- 5. Convert to the selected backend and preprocess the Cox loss
  |     +-- Automatic alpha grids are generated here on that backend
  |
  +-- 6. Score only evaluable folds; retain skipped-fold diagnostics
  |
  +-- 7. Require complete finite candidate evidence, select, and refit
```

### CV Scoring Paths

The numbered paths below describe scalar-response scoring. Penalized Cox uses
the survival-aware fold path after its preparation sequence above.

#### Path 1: Ridge Eigendecomposition (squared_error + l2)

**When**: `loss="squared_error"`, `penalty="l2"`, the resolved CV device is CPU, and `sample_weight=None`.

**Method**: Batch eigendecomposition per fold.

```python
# For each fold:
XtX = Xc.T @ Xc              # Centered Gram matrix
eigvals, Q = eigh(XtX)       # One eigendecomposition
# Solve all alphas at once:
coef = Q @ (1/(eigvals + n*alpha) * Q.T @ Xc.T @ yc)
```

**Complexity**: O(p^3) per fold (eigendecomposition), independent of n_alphas.

**Why it's fast**: All alphas are solved from a single eigendecomposition. For 20 alphas x 5 folds, this is 5 eigendecompositions instead of 100 model fits.

This batched scoring path depends on the resolved CV device, not the constructor
spelling: `device="auto"` uses it only when auto routing resolves to CPU. If
auto routing selects CUDA/Torch, CV uses the corresponding GPU scoring path.
After selection, squared-error L2 always transfers the full refit data to NumPy
and executes the exact float64 `_ridge_eig_single()` solve on CPU so CV/refit
coefficient precision is stable. The fitted estimator still retains
`cv_selected_device_` as its prediction/output backend contract; that metadata
does not claim the refit eigensolve ran on the selected accelerator.

#### Path 2: Fold-Batched CV (logistic, poisson, gamma, NB, inv.gauss, tweedie)

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
| logistic | sigmoid(eta) - y | eig_max(X'X) / 4n |
| poisson | exp(eta) - y | eig_max(X'X) / n x y_scale |
| gamma | 1 - y/exp(eta) | eig_max(X'X) / n x max(y/y_mean) |
| inverse_gaussian | (exp(eta) - y) / exp(2*eta) | eig_max(X'X) / n x y_scale |
| negative_binomial | (exp(eta) - y) / (1 + exp(eta)) | eig_max(X'X) / n x y_scale |
| tweedie | exp((1-p)*log(mu)) * (mu - y) | eig_max(X'X) / n x y_scale |

#### Path 3: Sparse CV (squared_error + l1/elasticnet)

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

#### Path 4: LLA Path (SCAD/MCP)

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

#### Path 5: General Per-Fold (fallback)

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

### Two-Stage CV

When `cv_strategy="two_stage"`:

1. **Stage 1 (screening)**: Run relaxed CV (reduced max_iter, looser tol) on full alpha grid
2. **Select top-k candidates**: Identify alpha values with best stage-1 scores
3. **Stage 2 (refinement)**: Run strict CV only on candidate alphas

This can skip 50-80% of alphas in the expensive strict pass.

### GPU Acceleration Techniques

#### 1. Async FISTA Loop

For non-smooth penalties (l1, elasticnet, SCAD, MCP) on GPU:

```python
# Traditional FISTA: Armijo backtracking = GPU->CPU sync every iteration
for iteration in range(max_iter):
    coef_new = proximal(coef - step * grad, alpha * step)
    if loss(coef_new) > bound:  # GPU->CPU sync!
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

#### 2. torch.compile Fusion

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

#### 3. Device-Side Convergence Check

```python
# CPU path: sync every iteration
delta = float(to_numpy(abs(coef - coef_old)))  # GPU->CPU sync

# GPU path: check every 50 iterations, batch with other checks
if iteration % 50 == 0:
    delta = torch.sum(torch.abs(coef - coef_old), dim=0)
    active = active & (delta >= tol)  # All on GPU
    if not torch.any(active).item():  # One sync point
        break
```

#### 4. Batched Validation Scoring

```python
# Per-alpha scoring: 20 syncs
for alpha in alphas:
    val_loss = loss(X_val, y_val, coef)  # GPU->CPU sync
    scores.append(val_loss)

# Batched scoring: 1 sync
scores_dev = []
for alpha in alphas:
    scores_dev.append(loss(X_val, y_val, coef))  # Stay on GPU
scores = to_numpy(torch.stack(scores_dev))  # One sync
```

#### 5. Warm-Start Across Alphas

Descending alpha grid (strongest regularization first). Each alpha's solution initializes the next:

```python
coef = zeros(p)
for alpha in alphas_descending:
    coef = fista_solver(..., init_coef=coef)  # Warm start
```

This reduces iterations by 3-5x compared to cold start.

### Result Caching

`LassoCV`, `ElasticNetCV`, and `RidgeCV` implement hash-based result caching to avoid redundant cross-validation runs on identical data and parameters. This benefits both CPU and GPU paths equally.

**Motivation**: Cross-validation is expensive -- e.g., `LassoCV(n_alphas=100, cv=5)` performs 500 model fits. Caching avoids redundant computation when the same estimator is fit multiple times on the same data:

```python
# Same data, different max_iter -> cache miss, recomputed
m1 = LassoCV(max_iter=100).fit(X, y)
m2 = LassoCV(max_iter=500).fit(X, y)

# Same data, same params -> cache hit, instant return
m3 = LassoCV(max_iter=100).fit(X, y)
```

#### Cache Architecture

```
+---------------------------------------------------------+
| LassoCV.fit(X, y, sample_weight)                        |
|                                                          |
|  1. data_digest = _hash_data(X, y, sample_weight)        |
|     +-- Sample 100 rows + shape + summary stats -> 16 B  |
|                                                          |
|  2. cache_key = _make_cache_key(params + data_digest)    |
|     +-- All CV params + data fingerprint -> 32 B hash    |
|                                                          |
|  3. Lookup                                               |
|     +-- Hit -> return cached alpha, mse_path, coef_      |
|     +-- Miss -> run CV -> store result in LRU cache      |
+---------------------------------------------------------+
```

#### Data Fingerprint: `_hash_data(X, y, sample_weight)`

**Design goals**: distinguish different datasets, low computation cost (O(100*p) not O(n*p)), and support GPU arrays (CuPy/torch auto-convert to numpy).

```python
def _hash_data(X, y, sample_weight=None) -> bytes:
    h = blake2b(digest_size=16)

    # 1. Record shape
    h.update(shape_bytes)           # (n, p) -> 8 bytes

    # 2. Sample 100 evenly-spaced rows
    idx = arange(0, n, n//100)[:100]
    h.update(X[idx].tobytes())      # 100 x p x 8 bytes
    h.update(y[idx].tobytes())      # 100 x 8 bytes

    # 3. Summary stats (fallback uniqueness)
    h.update([mean(X), std(X)])     # 16 bytes
    h.update([mean(y), std(y)])     # 8 bytes

    # 4. sample_weight (if provided)
    h.update(sw[idx].tobytes())     # 100 x 8 bytes
    h.update([mean(sw)])            # 8 bytes

    return h.digest()               # 16 bytes
```

**Why sample 100 rows?**

| Approach | Cost | Collision Risk |
|----------|------|----------------|
| Full data | O(n*p) | ~ 0 |
| First/last + summary | O(1) | High (middle rows differ) |
| **100-row sample** | O(100*p) | Negligible |

With 100 sampled rows, the probability of two different datasets having identical samples is approximately 2^(-128) for random data.

#### Parameter Fingerprint: `_make_cache_key(...)`

The cache key includes all parameters that affect CV results:

- `X_shape`, `y_shape` -- data dimensions
- `alphas` -- alpha grid (if provided)
- `n_alphas`, `alpha_min_ratio` -- grid generation params
- `fit_intercept`, `use_gpu`, `max_iter`, `tol` -- solver params
- `cpu_solver`, `cv_method`, `cd_kkt_check_every` -- algorithm params
- `fold_indices` -- first 5 indices per fold
- `sample_weight_shape` -- weight dimensions
- `data_digest` -- from `_hash_data`

#### LRU Cache

```python
_LASSO_CV_ALPHA_CACHE = {}      # Global dict
_LASSO_CV_ALPHA_CACHE_MAXSIZE = 16  # Max 16 cached results

def _cache_get(key):
    val = cache.get(key)
    if val is not None:
        cache.move_to_end(key)  # LRU: move to end on access
    return val

def _cache_put(key, value):
    cache[key] = value
    while len(cache) > MAXSIZE:
        cache.popitem(last=False)  # Evict least recently used
```

#### GPU Impact

Cache hash is not a GPU-specific optimization, but it is particularly valuable for GPU paths:

| Overhead Source | CPU | GPU |
|-----------------|-----|-----|
| Data transfer (H2D) | None | ~1-10ms |
| JIT compilation (torch.compile) | None | ~100ms first call |
| CV computation | Same | Same or faster |

A cache hit on GPU saves not only CV computation time but also JIT + H2D fixed overhead. Cached results store complete CV outputs and do not affect estimation precision.

### Alpha Convention

All penalties use `alpha` consistently across `PenalizedGeneralizedLinearModel` and the specialized wrappers.

| Penalty | statgpu Alpha | sklearn Alpha | Internal Consistency |
|---------|--------------|---------------|---------------------|
| L1 | `alpha` | `alpha` | `Lasso(a) == PGLM(a, penalty='l1')` |
| ElasticNet | `alpha` | `alpha` | `ElasticNet(a) == PGLM(a, penalty='elasticnet')` |
| L2 (Ridge) | `alpha` | `alpha / n` | `Ridge(a) == PGLM(a, penalty='l2')` |

**sklearn mapping**: `sklearn_alpha = statgpu_alpha * n` for Ridge. Lasso/ElasticNet use the same alpha directly.

Internal consistency is verified to machine precision (diff ~1e-16).

### Known Limitations

#### Non-uniform sample_weight with non-L2 penalties

Non-uniform `sample_weight` is **not supported** for penalties other than L2:

| Penalty | Solver | Non-uniform weights |
|---------|--------|-------------------|
| L2 | IRLS | Supported |
| L1, ElasticNet | FISTA | Raises ValueError |
| SCAD, MCP | FISTA | Raises ValueError |
| Adaptive L1 | FISTA | Raises ValueError |
| Group Lasso/MCP/SCAD | FISTA | Raises ValueError |

The underlying solvers (`fista`, `fista_bb`) reject non-uniform `sample_weight`. This is a solver-level limitation, not a CV limitation. Passing non-uniform weights with these penalties raises a clear `ValueError`.

**Workaround**: Use `penalty='l2'` with `solver='irls'` for weighted GLM fits.

**Future work**: Implement weighted FISTA gradient computation (`X' diag(w) residual / sum(w)`) in `fista_solver` and `fista_bb_solver` to support non-uniform weights with all penalties.

#### Other Limitations

- **`n_jobs` parameter**: Currently accepted but fold loops execute sequentially. Reserved for future parallelization.
- **SCAD/MCP CV speed**: Slower than L1/ElasticNet due to iterative LLA (Local Linear Approximation) rounds per alpha value.
- **NB with GPU**: Negative Binomial always falls back to CPU due to complex gradient overhead.

### Performance Characteristics

#### CPU vs GPU Break-Even Points

| Loss | p=100 | p=500 |
|------|-------|-------|
| squared_error | CPU wins | GPU wins at n>=2000 |
| logistic | CPU wins | GPU wins at n>=2000 |
| poisson | CPU wins | GPU wins at n>=2000 |
| gamma | CPU wins | GPU wins at n>=5000 |
| NB | CPU wins | CPU wins |

GPU wins at large p because the GEMM operations (`X @ coef`, `X.T @ resid`) dominate, and GPU GEMM throughput exceeds CPU for matrices above ~100x100.

#### Fold-Batch vs Per-Fold Speedup

On Tesla P100 (from benchmark):

| Loss | n=2000, p=500 | n=5000, p=500 |
|------|---------------|---------------|
| poisson + l1 | 7.4x | 4.5x |
| gamma + l1 | 6.5x | 9.3x |
| logistic + l1 | 1.4x | 2.5x |

The speedup comes from eliminating per-fold overhead (Lipschitz computation, model initialization, Python loop) and batching GPU operations.

### FAQ

**Q: Why is the CV cache not hitting?**
The CV cache uses blake2b hashing to detect data changes. Cache misses occur when:
- The data array memory address changes (even if values are the same)
- `sample_weight` changes
- `alpha_grid` changes
- Data shape changes

**Q: Why is `PenalizedGLM_CV`'s `alpha_grid` different from sklearn?**
statgpu uses a data-driven alpha grid: `alpha_max` is computed from `max(|X'y|)/n`, then decays in a geometric sequence. sklearn uses a similar but potentially slightly different strategy.

**Q: How do I choose the number of CV folds?**
- Default 5-fold: balances bias and variance
- 10-fold: more accurate error estimates, but slower
- Leave-one-out: usable when n is small, but high variance

### External Validation

**Test scripts:**
- `dev/tests/test_pr49_regression.py` -- 2400+ lines of regression tests covering CV parameter validation, kfold integrity, cache consistency
- `dev/tests/test_glm_penalty_review_fixes.py` -- 2015 lines of penalty tests
- `dev/tests/test_elasticnet_cv.py` -- ElasticNetCV dedicated tests
- `dev/tests/test_ridge_cv.py` -- RidgeCV dedicated tests

**Benchmark scripts:**
- `dev/tests/benchmark_cv_full.py` -- Full CV benchmark
- `dev/benchmarks/benchmark_lassocv_impls.py` -- LassoCV implementation comparison

**External framework comparison:**
- RidgeCV vs sklearn `RidgeCV`: alpha selection and MSE alignment
- ElasticNetCV vs sklearn `ElasticNetCV`: l1_ratio and alpha selection alignment
- PenalizedGLM vs R `glmnet`: coefficient path and deviance alignment
