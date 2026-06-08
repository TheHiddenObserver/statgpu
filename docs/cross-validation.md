# 交叉验证实现

> 语言: 中文  
> 最后更新: 2026-06-07  
> 页面定位: CV 架构、加速技巧与 GPU 优化  
> 切换: [English](en/cross-validation.md)

语言切换：[English](en/cross-validation.md)

## 概述

`PenalizedGLM_CV` 对惩罚广义线性模型执行 k 折交叉验证，支持 7 种损失函数和 10+ 种惩罚类型。实现使用了多种加速技巧来最小化总 CV 时间，针对不同的 loss×penalty×device 组合使用专门的求解路径。

## 架构

```
PenalizedGLM_CV.fit(X, y)
  │
  ├─ 1. 自动设备选择 (_effective_cv_device)
  │     └─ 根据问题规模和损失函数选择 CPU/CuPy/Torch
  │
  ├─ 2. Alpha 网格生成 (_generate_alpha_grid)
  │     └─ 从 alpha_max 生成递减 alpha 网格
  │
  ├─ 3. CV 评分 (_compute_cv_scores)
  │     ├─ 快速路径: Ridge 特征分解 (squared_error + l2)
  │     ├─ Fold-batch 路径 (logistic, poisson, gamma, NB, inv.gauss, tweedie)
  │     ├─ Sparse CV 路径 (squared_error + l1/en)
  │     ├─ LLA 路径 (SCAD/MCP)
  │     └─ 通用逐 fold 路径 (兜底)
  │
  ├─ 4. 最优 alpha 选择
  │
  └─ 5. 全数据重拟合 (_refit_best)
```

## 自动设备选择

当 `device="auto"` 时，CV 估计器根据问题规模和 loss×penalty 组合选择后端：

| 条件 | 选择设备 | 原因 |
|------|---------|------|
| n×p < 200k | CPU | Kernel launch 开销主导 |
| squared_error + l1/en, p≥256, n×p≥1M | Torch | 批量 alpha 路径受益 |
| logistic + l1/en, n≥5000, n×p≥500k | Torch | Fold-batch 路径 |
| poisson + l1/en, p≥500, n×p≥1M | Torch | Fold-batch 路径 |
| gamma + l1/en, p≥500, n×p≥2M | Torch | Fold-batch 路径 |
| SCAD/MCP, n×p≥1M | Torch | Async FISTA 路径 |
| NB（任意惩罚） | CPU | 复杂梯度开销 |
| 其他 | CPU | 默认回退 |

阈值基于 benchmark 数据，存储在 `_effective_cv_device()` 中。

## CV 评分路径

### 路径 1：Ridge 特征分解（squared_error + l2）

**条件**：`loss="squared_error"`、`penalty="l2"`、`device` 为 CPU/auto、`sample_weight=None`。

**方法**：每 fold 批量特征分解。

```python
# 每个 fold：
XtX = Xc.T @ Xc              # 中心化 Gram 矩阵
eigvals, Q = eigh(XtX)       # 一次特征分解
# 一次求解所有 alpha：
coef = Q @ (1/(eigvals + n*alpha) * Q.T @ Xc.T @ yc)
```

**复杂度**：每 fold O(p³)（特征分解），与 n_alphas 无关。

**为什么快**：所有 alpha 从一次特征分解求解。对于 20 alpha × 5 fold，这是 5 次特征分解而非 100 次模型拟合。

### 路径 2：Fold-Batch CV（logistic, poisson, gamma, NB, inv.gauss, tweedie）

**条件**：`loss` 为 GLM 系列、`penalty` 为 l1/elasticnet、`device` 为 Torch/CuPy、`strict=False`（两阶段模式）。

**方法**：所有 fold 在 GPU 上同时运行，使用 mask 张量。

```python
# 设置：所有 fold 共享设备上的 X
train_mask = ones(n_samples, n_folds)   # 1 = 训练, 0 = 验证
val_mask = zeros(n_samples, n_folds)    # 1 = 验证, 0 = 训练

# 每 fold 的 Lipschitz 和步长
for fold in folds:
    train_mask[val_idx, fold] = 0
    L[fold] = lipschitz(X_train_fold)
    step[fold] = 1 / L[fold]

# FISTA 循环：所有 fold 同时
for alpha in alphas:
    for iteration in range(max_iter):
        eta = X @ coef + intercept           # (n, n_folds) 矩阵
        resid = loss_residual(eta, y) * train_mask
        grad = (X.T @ resid) / n_train_vec   # (p, n_folds) 矩阵
        coef = proximal(coef - step * grad, alpha * step)
        # 收敛检查：所有 fold 一次
        active = active & (delta >= tol)
        if not any(active): break
```

**核心优势**：
- 单次 `X @ coef` GEMM 处理所有 fold（vs n_folds 次独立 GEMV）
- 单次 `X.T @ resid` GEMM 处理所有 fold
- 所有 fold 的收敛检查一次完成
- 无逐 fold Python 循环开销

**支持的损失函数**（内联梯度公式）：

| 损失 | 梯度 (residual) | Lipschitz 缩放 |
|------|----------------|-----------------|
| logistic | sigmoid(η) - y | eig_max(X'X) / 4n |
| poisson | exp(η) - y | eig_max(X'X) / n × y_scale |
| gamma | 1 - y/exp(η) | eig_max(X'X) / n × max(y/ȳ) |
| inverse_gaussian | (exp(η) - y) / exp(2η) | eig_max(X'X) / n × y_scale |
| negative_binomial | (exp(η) - y) / (1 + exp(η)) | eig_max(X'X) / n × y_scale |
| tweedie | exp((1-p)·log(μ)) · (μ - y) | eig_max(X'X) / n × y_scale |

### 路径 3：Sparse CV（squared_error + l1/elasticnet）

**条件**：`loss="squared_error"`、`penalty` 为 l1/elasticnet。

**方法**：预计算 Gram 矩阵 + warm-start FISTA。

```python
# 每 fold：预计算一次
XtX = X_train.T @ X_train
Xty = X_train.T @ y_train

# 递减 alpha 的 warm-start
coef = zeros(p)
for alpha in alphas_sorted_desc:
    for iteration in range(max_iter):
        grad = XtX @ coef - Xty
        coef = proximal(coef - step * grad, alpha * step)
```

**核心优势**：`XtX` 和 `Xty` 每 fold 只计算一次，所有 alpha 复用。

### 路径 4：LLA 路径（SCAD/MCP）

**条件**：`penalty` 为 SCAD 或 MCP。

**方法**：局部线性近似（LLA）外循环 + FISTA 内循环。

```python
for alpha in alphas:
    for lla_iter in range(max_lla):
        # LLA：将非凸惩罚近似为加权 L1
        lla_w = scad_penalty.lla_weights(coef)
        inner_penalty = AdaptiveL1Penalty(alpha=1.0, weights=lla_w)
        # FISTA 内求解
        coef = fista_solver(loss, inner_penalty, X, y, init_coef=coef)
```

**对于 squared_error**：使用预计算的 Gram 矩阵（同路径 3）。

### 路径 5：通用逐 Fold（兜底）

**条件**：无专门路径适用时。

**方法**：标准逐 fold、逐 alpha 模型拟合。

```python
for fold in folds:
    for alpha in alphas:
        model = PenalizedGeneralizedLinearModel(...)
        model.fit(X_train, y_train)
        val_loss = evaluate(model, X_val, y_val)
```

**用于**：NB+l2、tweedie+l2、以及专门路径不可用的场景。

## 两阶段 CV

当 `cv_strategy="two_stage"` 时：

1. **阶段 1（筛选）**：在完整 alpha 网格上运行宽松 CV（减少 max_iter，放宽 tol）
2. **选择 top-k 候选**：识别阶段 1 得分最优的 alpha
3. **阶段 2（精炼）**：仅对候选 alpha 运行严格 CV

这可以跳过严格路径中 50-80% 的 alpha。

## GPU 加速技巧

### 1. Async FISTA 循环

对于 GPU 上的非平滑惩罚（l1, elasticnet, SCAD, MCP）：

```python
# 传统 FISTA：Armijo 回溯 = 每次迭代 GPU→CPU 同步
for iteration in range(max_iter):
    coef_new = proximal(coef - step * grad, alpha * step)
    if loss(coef_new) > bound:  # GPU→CPU 同步！
        step /= 2
        continue

# Async FISTA：无回溯，保守固定步长
step = 1 / (L * safety_factor)  # 预计算，无逐迭代同步
for iteration in range(max_iter):
    coef = proximal(coef - step * grad, alpha * step)
    # 所有操作留在 GPU
```

**安全系数**：logistic 2x、gamma 3x、inverse_gaussian 3x、tweedie 5x。

**同步减少**：从 2000 次（每次迭代一次）减少到 ~80 次（每 25 次迭代一次）。

### 2. torch.compile 融合

FISTA 步操作通过 `torch.compile` 融合：

```python
@torch.compile
def fista_step(X, coef, step, alpha):
    eta = X @ coef
    mu = torch.exp(eta)
    grad = X.T @ (mu - y) / n
    w = coef - step * grad
    return torch.sign(w) * torch.clamp(torch.abs(w) - alpha*step, min=0)
```

这将 ~6 次 kernel launch 减少到 1-2 次编译后的 kernel。

### 3. 设备端收敛检查

```python
# CPU 路径：每次迭代同步
delta = float(to_numpy(abs(coef - coef_old)))  # GPU→CPU 同步

# GPU 路径：每 50 次迭代检查，与其他检查批量处理
if iteration % 50 == 0:
    delta = torch.sum(torch.abs(coef - coef_old), dim=0)
    active = active & (delta >= tol)  # 全在 GPU
    if not torch.any(active).item():  # 一个同步点
        break
```

### 4. 批量验证评分

```python
# 逐 alpha 评分：20 次同步
for alpha in alphas:
    val_loss = loss(X_val, y_val, coef)  # GPU→CPU 同步
    scores.append(val_loss)

# 批量评分：1 次同步
scores_dev = []
for alpha in alphas:
    scores_dev.append(loss(X_val, y_val, coef))  # 留在 GPU
scores = to_numpy(torch.stack(scores_dev))  # 一次同步
```

### 5. Alpha 间 Warm-Start

递减 alpha 网格（最强正则化优先）。每个 alpha 的解初始化下一个：

```python
coef = zeros(p)
for alpha in alphas_descending:
    coef = fista_solver(init_coef=coef, ...)  # Warm start
```

这比冷启动减少 3-5 倍迭代次数。

## 结果缓存

参见 [CV Cache Hash](cv_cache_hash.md) 了解避免相同数据重复 CV 的缓存机制。

## Alpha 约定

所有惩罚在 `PenalizedGeneralizedLinearModel` 和专用包装器中使用一致的 `alpha`。

| 惩罚 | statgpu Alpha | sklearn Alpha | 内部一致性 |
|------|--------------|---------------|-----------|
| L1 | `alpha` | `alpha` | `Lasso(a) == PGLM(a, penalty='l1')` |
| ElasticNet | `alpha` | `alpha` | `ElasticNet(a) == PGLM(a, penalty='elasticnet')` |
| L2 (Ridge) | `alpha` | `alpha / n` | `Ridge(a) == PGLM(a, penalty='l2')` |

**sklearn 映射**：Ridge 需要 `sklearn_alpha = statgpu_alpha * n`。Lasso/ElasticNet 直接使用相同的 alpha。

内部一致性已验证到机器精度（diff ~1e-16）。

## 性能特征

### CPU vs GPU 盈亏平衡点

| 损失 | p=100 | p=500 |
|------|-------|-------|
| squared_error | CPU 赢 | GPU 在 n≥2000 时赢 |
| logistic | CPU 赢 | GPU 在 n≥2000 时赢 |
| poisson | CPU 赢 | GPU 在 n≥2000 时赢 |
| gamma | CPU 赢 | GPU 在 n≥5000 时赢 |
| NB | CPU 赢 | CPU 赢 |

GPU 在大 p 时赢，因为 GEMM 操作（`X @ coef`、`X.T @ resid`）占主导，GPU GEMM 吞吐量在 ~100×100 以上矩阵超过 CPU。

### Fold-Batch vs Per-Fold 加速比

Tesla P100 上的 benchmark 数据：

| 损失 | n=2000, p=500 | n=5000, p=500 |
|------|---------------|---------------|
| poisson + l1 | 7.4x | 4.5x |
| gamma + l1 | 6.5x | 9.3x |
| logistic + l1 | 1.4x | 2.5x |

加速来自消除逐 fold 开销（Lipschitz 计算、模型初始化、Python 循环）和批量化 GPU 操作。
