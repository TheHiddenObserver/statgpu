# GPU CV 性能优化建议

> 基于当前 benchmark 数据，GPU CV 在多数场景下仍慢于 CPU。以下分析瓶颈并给出可行的优化方案。

---

## 一、当前 GPU CV 性能现状

### GPU 赢的场景（已有）

| 场景 | CPU | Torch | 加速比 |
|------|-----|-------|--------|
| tweedie+l1 n=2000 | 5193ms | 2941ms | 1.77x |
| tweedie+elasticnet n=2000 | 4667ms | 2792ms | 1.67x |
| poisson+none/l2 (newton) | ~150ms | ~45ms | 3.3x |
| NB+elasticnet (fista_bb) | ~900ms | ~680ms | 1.3x |

### GPU 输的场景（瓶颈）

| 场景 | CPU | CuPy | Torch | 原因 |
|------|-----|------|-------|------|
| squared_error+scad n=500 | 3.1s | 16.9s | 12.1s | 每 alpha 重算 LLA weights + kernel launch |
| logistic+scad n=500 | 8.9s | 117.5s | 66.9s | GLM loss.value() 每次 GPU→CPU sync |
| gamma+scad n=500 | 7.6s | 81.9s | 47.5s | 同上 |
| NB+scad n=500 | 25.0s | 249.8s | 147.8s | NB gradient 计算开销 + 多次 sync |

---

## 二、瓶颈分析

### 瓶颈 1：每个 alpha × 每个 fold 都调用 `model.fit()`（最大瓶颈）

当前 CV 循环：
```python
for fold in folds:
    for alpha in alphas:  # 20-100 个 alpha
        model.fit(X_train, y_train)  # 每次都有完整开销
```

每次 `model.fit()` 的固定开销：
- 设备检测和数据传输：~0.5ms
- XtX 计算（即使有 _cv_cache，仍有 overhead）：~0.2ms
- Lipschitz 常数计算：~0.1ms
- 求解器初始化：~0.1ms

**20 alphas × 5 folds = 100 次 model.fit()，固定开销 ~100ms**

### 瓶颈 2：GLM loss 的 `value()` 和 `gradient()` 频繁 GPU→CPU sync

在 FISTA 内循环中：
```python
for iteration in range(max_iter):  # 可能 2000 次
    q_yk_dev, grad = _fused_glm_value_and_gradient(loss, X, y, y_k)  # GPU
    # Armijo backtracking:
    q_new_dev = loss.value(X, y, coef_new)  # GPU
    _armijo_ok = bool(_to_numpy(slack_dev >= 0))  # GPU→CPU sync!
```

**每次 Armijo 检查都有 1 次 GPU→CPU sync**，对于 2000 次迭代 = 2000 次 sync，每次 ~0.01ms = 20ms 额外开销。

### 瓶颈 3：小矩阵 GPU kernel launch 开销

n=500, p=50 时，矩阵乘法 `X.T @ X` 只有 50×50 = 2500 个元素。GPU kernel launch 开销（~0.01ms）远大于计算本身（~0.001ms）。

### 瓶颈 4：CV 中 validation loss 每次 alpha 都计算

```python
for alpha in alphas:
    model.fit(...)
    val_loss = _evaluate_single(model, X_val, y_val)  # 每次都算
```

validation loss 计算涉及 `model.predict(X_val)` + loss 计算，每次都有 GPU→CPU 转移。

---

## 三、优化方案

### 方案 1：批量 alpha 求解（最高优先级，预计 3-5x 加速）

**核心思路**：不要每个 alpha 单独调用 `model.fit()`，而是在一次 `fista_lla_path` 调用中处理所有 alpha。

**实现**：
```python
def cv_batch_solve(X_train, y_train, X_val, y_val, alphas, loss, penalty, backend):
    """一次调用处理所有 alpha，避免 per-alpha model.fit() 开销。"""
    # 1. 预计算所有共享量（只算一次）
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    L = lipschitz_constant(XtX, loss)
    
    # 2. 按 alpha 从大到小排序，warm-start
    sorted_alphas = sorted(alphas, reverse=True)
    coef = zeros(p)
    val_losses = []
    
    for alpha in sorted_alphas:
        # 3. 直接调用 FISTA 内循环（不经过 model.fit()）
        coef = fista_inner_loop(XtX, Xty, coef, alpha, L, loss, penalty)
        val_losses.append(compute_loss(X_val, y_val, coef))
    
    return val_losses
```

**关键改动**：在 `_penalized_cv.py` 中直接调用 `_solver.fista_solver()` 而非 `model.fit()`，跳过 PGLM 的初始化开销。

**预计加速**：20 alphas × 5 folds = 100 次 → 5 次调用（每个 fold 一次），减少 ~95% 的固定开销。

### 方案 2：在 device 上批量计算 validation loss（预计 1.5-2x 加速）

**核心思路**：不要每次 alpha 都把 validation loss 从 GPU 传回 CPU，而是在 GPU 上累积所有 alpha 的 validation loss，最后一次传回。

**实现**：
```python
# 旧代码：每次 alpha 都 sync
for alpha in alphas:
    model.fit(...)
    val_loss = float(_to_numpy(loss_fn.value(X_val, y_val, coef)))  # GPU→CPU
    val_losses.append(val_loss)

# 新代码：在 GPU 上累积
val_losses_dev = xp.zeros(len(alphas))
for i, alpha in enumerate(alphas):
    coef = solve_one_alpha(...)
    val_losses_dev[i] = loss_fn.value(X_val, y_val, coef)  # 留在 GPU
val_losses = _to_numpy(val_losses_dev)  # 只传一次
```

**预计加速**：20 alphas × 5 folds = 100 次 GPU→CPU sync → 5 次，减少 ~95% sync 开销。

### 方案 3：预计算 XtX + Lipschitz per fold（预计 1.2-1.5x 加速）

**核心思路**：每个 fold 的 XtX 和 Lipschitz 常数对所有 alpha 都一样，只算一次。

**当前状态**：已有 `_cv_cache`，但只对 squared_error 的 exact path 生效。需要扩展到所有 GLM loss。

**实现**：
```python
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_train, y_train = X[train_idx], y[train_idx]
    
    # 预计算一次（当前已有，但需要确保所有 path 都用到）
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    L = compute_lipschitz(XtX, loss)
    
    cv_cache = {'XtX': XtX, 'Xty': Xty}
    
    for alpha in alphas:
        model._cv_cache = cv_cache
        model.lipschitz_L = L
        model.fit(X_train, y_train)  # 内部跳过 XtX 计算
```

### 方案 4：减少 FISTA 内循环的 GPU→CPU sync（预计 1.5-2x 加速）

**核心思路**：把 convergence check 和 divergence check 改成 device-side。

**当前问题**：
```python
# 每 5 次迭代就 sync
if iteration % _conv_interval == 0:
    coef_diff = float(_to_numpy(_abs_sum_dev(coef - coef_old)))  # GPU→CPU
    if coef_diff < tol:
        break
```

**优化**：
```python
# 在 GPU 上做 convergence check，只在真正收敛时才 sync
if iteration % _conv_interval == 0:
    coef_diff_dev = _abs_sum_dev(coef - coef_old)
    # 用 device-side 判断
    if backend == "cupy":
        converged = bool(coef_diff_dev < tol)  # CuPy 支持 device-side bool
    elif backend == "torch":
        converged = bool((coef_diff_dev < tol).item())
    if converged:
        break
```

**更进一步**：把 convergence check 频率从每 5 次改为每 20 次，减少 sync 次数。

### 方案 5：对小问题自动 fallback 到 CPU（已部分实现，可更激进）

**当前阈值**：`n * p < 200_000` 时 auto fallback 到 CPU。

**建议更激进**：对于 CV 场景，每个 fold 的 `n_train * p` 比全量小。如果 `n_train * p < 50_000`，直接用 CPU。

```python
def _cv_backend_for_fold(n_train, p, requested_backend):
    if n_train * p < 50_000:
        return "cpu"  # GPU 开销大于收益
    return requested_backend
```

### 方案 6：GPU warm-up 批量化（预计 1.1-1.3x 加速）

**核心思路**：在 CV 开始前，一次性 warm-up 所有 GPU kernel，避免首次调用的 JIT 编译开销。

**实现**：
```python
def _warmup_gpu_for_cv(loss, penalty, backend):
    """用小数据预热所有 GPU kernel。"""
    X_dummy = xp.randn(10, 5)
    y_dummy = xp.randn(10)
    for alpha in [0.1, 0.01]:
        model = PGLM(loss=loss, penalty=penalty, alpha=alpha, device=backend)
        model.fit(X_dummy, y_dummy)
```

---

## 四、实施优先级

| 优先级 | 方案 | 预计加速 | 实施难度 | 影响范围 |
|--------|------|---------|---------|---------|
| P0 | 方案 1：批量 alpha 求解 | 3-5x | 高 | 所有 CV |
| P1 | 方案 2：device 上批量 validation loss | 1.5-2x | 中 | 所有 CV |
| P2 | 方案 4：减少 FISTA sync | 1.5-2x | 中 | 所有 FISTA |
| P3 | 方案 3：预计算 XtX per fold | 1.2-1.5x | 低 | 所有 CV |
| P4 | 方案 5：更激进的 CPU fallback | 1.5-3x | 低 | 小问题 |
| P5 | 方案 6：GPU warm-up 批量化 | 1.1-1.3x | 低 | 首次运行 |

**组合效果**：如果实施方案 1+2+4，预计 GPU CV 总体加速 3-8x，使 GPU 在更多场景下超越 CPU。

---

## 五、验证方法

```bash
# 修改后运行
PYTHONPATH=/root/statgpu python dev/tests/benchmark_cv_full.py --solvers auto

# 对比修改前后
PYTHONPATH=/root/statgpu python dev/tests/benchmark_cv_full.py --solvers auto 2>&1 | tee /tmp/cv_after.log

# 重点检查
# 1. 精度不退化（corr > 0.999）
# 2. GPU 时间减少
# 3. alpha 选择一致（alpha=OK）
```
