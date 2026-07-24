# 交叉验证

> 语言：中文  
> 最后更新：2026-06-12  
> 页面定位：CV 用户指南 + 架构实现 + 缓存机制（统一页面）  
> 切换：[English](../../en/guides/cross-validation.md)

## 概述

statgpu 为所有惩罚模型提供交叉验证估计器。每个 CV 估计器自动在正则化参数网格上搜索，通过 k 折交叉验证选择最优参数。`PenalizedGLM_CV` 支持 7 种损失函数和 10+ 种惩罚类型，使用多种加速技巧最小化总 CV 时间，针对不同的 loss×penalty×device 组合使用专门的求解路径。

| CV 估计器 | 基础模型 | 惩罚 | 路径 |
|-----------|---------|------|------|
| `RidgeCV` | `Ridge` | l2 | `statgpu.linear_model.RidgeCV` |
| `LassoCV` | `Lasso` | l1 | `statgpu.linear_model.LassoCV` |
| `ElasticNetCV` | `ElasticNet` | elasticnet | `statgpu.linear_model.ElasticNetCV` |
| `LogisticRegressionCV` | `LogisticRegression` | l2 | `statgpu.linear_model.LogisticRegressionCV` |
| `PenalizedGLM_CV` | `PenalizedGeneralizedLinearModel` | 任意 | `statgpu.linear_model.PenalizedGLM_CV` |

## 快速开始

### RidgeCV

```python
from statgpu.linear_model import RidgeCV

model = RidgeCV(
    alphas=None,           # 自动生成 log 等距网格
    n_alphas=100,          # alpha 候选数量
    cv=5,                  # 折数
    fit_intercept=True,
    device="auto",         # "cpu"、"cuda" 或 "auto"
)
model.fit(X, y)

print(f"最优 alpha: {model.alpha_}")
print(f"CV MSE 路径形状: {model.cv_results_['mse_path'].shape}")
print(f"R²: {model.score(X_test, y_test):.4f}")
```

### ElasticNetCV

```python
from statgpu.linear_model import ElasticNetCV

model = ElasticNetCV(
    l1_ratio=0.5,          # 或 [0.1, 0.5, 0.9] 搜索多个值
    alphas=None,
    cv=5,
    device="auto",
)
model.fit(X, y)

print(f"最优 alpha: {model.alpha_}")
print(f"最优 l1_ratio: {model.l1_ratio_}")
```

### PenalizedGLM_CV（通用）

```python
from statgpu.linear_model import PenalizedGLM_CV

# Poisson + SCAD 自动 CV
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

### LogisticRegressionCV

```python
from statgpu.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(
    cv=5,
    device="auto",
)
model.fit(X, y)
print(f"最优 C: {model.C_}")
print(f"准确率: {model.score(X_test, y_test):.4f}")
```

## 参数参考

### 通用参数（所有 CV 估计器）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cv` | int | `5` | CV 折数。必须 ≥ 2。 |
| `random_state` | int | `None` | 折洗牌的随机种子。 |
| `device` | str/Device | `"auto"` | `"cpu"`、`"cuda"` 或 `"auto"`。 |
| `fit_intercept` | bool | `True` | 是否拟合截距。 |
| `gpu_memory_cleanup` | bool | `False` | 拟合后释放 GPU 内存（CuPy）。 |

### RidgeCV 专用

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alphas` | array | `None` | Alpha 网格。`None` = 自动生成。 |
| `n_alphas` | int | `100` | 自动生成时的 alpha 数量。 |
| `alpha_min_ratio` | float | `1e-3` | 最小 alpha 与最大 alpha 的比值。 |
| `compute_inference` | bool | `False` | CV 后计算 SE/p 值/CI。 |
| `cov_type` | str | `"nonrobust"` | 推断的协方差类型。 |
| `gpu_cv_mixed_precision` | bool | `True` | CV 使用 float32（GPU 更快）。 |

### ElasticNetCV 专用

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `l1_ratio` | float/list | `0.5` | L1 混合比。传列表可搜索多个值。 |
| `alphas` | array | `None` | Alpha 网格。 |
| `n_alphas` | int | `100` | Alpha 数量。 |

### PenalizedGLM_CV 专用

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loss` | str | `"squared_error"` | 损失族（见 [Solver × Penalty 矩阵](solver-penalty-matrix.md)）。 |
| `penalty` | str | `"l2"` | 惩罚类型。 |
| `penalty_kwargs` | dict | `{}` | 惩罚参数（如 SCAD 的 `{"a": 3.7}`）。 |
| `alphas` | array | `None` | Alpha 网格。 |
| `n_alphas` | int | `100` | Alpha 数量。 |
| `cv_splits` | list | `None` | 自定义折分割 `[(train_idx, val_idx), ...]`。 |
| `scoring` | str | `"auto"` | 评分指标。`"auto"` 根据损失自动选择。 |
| `compute_inference` | bool | `False` | 计算 debiased 推断（仅 l1）。 |

## 自定义 CV 分割

所有 CV 估计器通过 `cv_splits` 支持自定义折生成器：

```python
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

# 时间序列 CV
tscv = TimeSeriesSplit(n_splits=5)
model = PenalizedGLM_CV(
    loss="poisson", penalty="l1",
    cv_splits=list(tscv.split(X)),
)
model.fit(X, y)

# 分层 CV 用于分类
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegressionCV(
    cv_splits=list(skf.split(X, y)),
)
model.fit(X, y)
```

`cv_splits=None`（默认）时，估计器使用 `kfold_indices(n, cv, random_state)` 生成随机洗牌的折。

## 样本权重

所有 CV 估计器支持 `sample_weight`：

```python
model = RidgeCV(cv=5)
model.fit(X, y, sample_weight=w)
print(f"加权 R²: {model.score(X_test, y_test, sample_weight=w_test):.4f}")
```

**限制**（见 [已知限制](#已知限制)）：
- 非均匀权重 + l1/elasticnet/SCAD/MCP 在求解器层面抛出 `ValueError`。
- 均匀权重（所有值相等）对所有惩罚有效。

## Alpha 网格

### 自动生成网格

`alphas=None` 时，网格生成方式：
1. 计算 `alpha_max = max(|X'y|) / n`（或加权变体）
2. 生成从 `alpha_max` 到 `alpha_max * alpha_min_ratio` 的 `n_alphas` 个值
3. 网格为 log 等距：`np.logspace(log10(alpha_max * ratio), log10(alpha_max), n_alphas)`

### 自定义网格

```python
import numpy as np

model = RidgeCV(
    alphas=np.logspace(-4, 2, 50),  # 自定义 50 点网格
    cv=5,
)
model.fit(X, y)
```

非正和非有限值会被自动过滤。如果所有提供的 alpha 都被过滤，会发出警告并使用默认网格。

## 拟合属性

`fit()` 后，所有 CV 估计器暴露：

| 属性 | 说明 |
|------|------|
| `alpha_` | CV 选择的最优 alpha |
| `best_score_` | 最优 CV 分数（回归为负 MSE，分类为准确率） |
| `cv_results_` | 包含 `mse_path`、`alpha_grid`、`best_idx` 的字典 |
| `estimator_` | 用最优 alpha 在全数据上重拟合的模型 |
| `coef_` | 重拟合模型的系数 |
| `intercept_` | 重拟合模型的截距 |

`ElasticNetCV` 额外有 `l1_ratio_`（传入列表时的最优 l1_ratio）。

## 评分

```python
# 预测
pred = model.predict(X_test)

# 评分（回归为 R²，分类为准确率）
r2 = model.score(X_test, y_test)

# 加权评分
r2_w = model.score(X_test, y_test, sample_weight=w_test)
```

`score()` 委托给重拟合估计器（`model.estimator_`），评分方法与基础模型一致。

## 设备选择

`device="auto"` 时，CV 估计器根据问题规模和 loss×penalty 组合选择后端：

| 条件 | 选择设备 | 原因 |
|------|---------|------|
| n×p < 200k | CPU | Kernel launch 开销主导 |
| squared_error + l1/en, p≥256, n×p≥1M | Torch | 批量 alpha 路径受益 |
| logistic + l1/en, n≥5000, n×p≥500k | Torch | Fold-batch 路径 |
| poisson + l1/en, p≥500, n×p≥1M | Torch | Fold-batch 路径 |
| gamma + l1/en, p≥500, n×p≥2M | Torch | Fold-batch 路径 |
| SCAD/MCP, n×p≥1M | Torch | 异步 FISTA 路径 |
| NB（任意惩罚） | CPU | 复杂梯度开销 |
| 其他 | CPU | 默认回退 |

阈值基于 benchmark 数据，存储在 `_effective_cv_device()` 中。显式控制：`device="cpu"` 强制 CPU，`device="cuda"` 强制 GPU。

## CV 后推断

`RidgeCV` 设置 `compute_inference=True` 时：

```python
model = RidgeCV(compute_inference=True, cov_type="hc1")
model.fit(X, y)

# 标准误、t 统计量、p 值、置信区间
print(model.summary())
```

`PenalizedGLM_CV` 设置 `penalty="l1"` 和 `compute_inference=True` 时：
- 通过 nodewise 回归计算 Debiased Lasso 推断
- 提供每个系数的 SE、z 统计量、p 值和 CI

**状态**：l2 推断完全可用。l1 debiased 推断可用。ElasticNet/SCAD/MCP 推断待实现。

## 性能建议

1. **大规模用 GPU**：n×p > 200k 时设置 `device="cuda"` 或让 `auto` 决定。
2. **减少 alpha 网格**：`n_alphas=50` 通常足够；默认 100。
3. **使用混合精度**：`gpu_cv_mixed_precision=True`（默认）CV 使用 float32，GPU 上快 2-4x。
4. **两阶段 CV**：`cv_strategy="two_stage"` 快速筛选 alpha，再精炼 top 候选。
5. **自定义折**：预生成折可避免重复运行时的重新洗牌。

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

`LassoCV`、`ElasticNetCV`、`RidgeCV` 实现了基于 hash 的结果缓存，避免对相同数据和参数重复执行交叉验证。**这不是 GPU 特有的优化**，而是通用的计算缓存机制，对 CPU 和 GPU 路径同等生效。

### 为什么需要缓存？

交叉验证（CV）是昂贵的操作：

```
LassoCV(n_alphas=100, cv=5) → 100 × 5 = 500 次模型拟合
```

在以下场景中，用户可能对相同数据多次调用 fit：

```python
# 场景 1：超参数搜索
for max_iter in [100, 500, 1000]:
    m = LassoCV(max_iter=max_iter).fit(X, y)  # 同一数据，不同参数

# 场景 2：逐步建模
m1 = LassoCV().fit(X, y)          # 第一次 fit
m2 = LassoCV().fit(X, y)          # 相同参数，应直接返回缓存

# 场景 3：调试/实验
m = LassoCV().fit(X, y)           # 运行一次
# ... 修改其他代码 ...
m = LassoCV().fit(X, y)           # 再次运行，应命中缓存
```

### 缓存架构

```
┌─────────────────────────────────────────────────────────┐
│ LassoCV.fit(X, y, sample_weight)                        │
│                                                          │
│  1. data_digest = _hash_data(X, y, sample_weight)        │
│     └─ 采样 100 行 + shape + 摘要统计 → 16 字节指纹      │
│                                                          │
│  2. cache_key = _make_cache_key(参数 + data_digest)      │
│     └─ 所有 CV 参数 + 数据指纹 → 32 字节哈希             │
│                                                          │
│  3. 查缓存                                               │
│     ├─ 命中 → 直接返回缓存的 alpha, mse_path, coef_      │
│     └─ 未命中 → 执行 CV → 结果存入 LRU 缓存              │
└─────────────────────────────────────────────────────────┘
```

### 数据指纹 `_hash_data(X, y, sample_weight)`

**设计目标**：
- 能区分不同数据集（避免碰撞）
- 计算成本低（O(100×p) 而非 O(n×p)）
- 支持 GPU 数组（CuPy/torch → 自动转 numpy）

**实现**：

```python
def _hash_data(X, y, sample_weight=None) -> bytes:
    h = blake2b(digest_size=16)

    # 1. 记录 shape
    h.update(shape_bytes)           # (n, p) → 8 bytes

    # 2. 采样 100 行（均匀间距）
    idx = arange(0, n, n//100)[:100]
    h.update(X[idx].tobytes())      # 100 × p × 8 bytes
    h.update(y[idx].tobytes())      # 100 × 8 bytes

    # 3. 摘要统计（兜底唯一性）
    h.update([mean(X), std(X)])     # 16 bytes
    h.update([mean(y), std(y)])     # 8 bytes

    # 4. sample_weight（如有）
    h.update(sw[idx].tobytes())     # 100 × 8 bytes
    h.update([mean(sw)])            # 8 bytes

    return h.digest()               # 16 bytes
```

**为什么采样 100 行？**

| 方案 | 成本 | 碰撞风险 |
|------|------|---------|
| 全量数据 | O(n×p) | ≈ 0 |
| 首末行 + 摘要 | O(1) | 高（中间行不同无法检测） |
| **100 行采样** | O(100×p) | 极低 |

100 行采样使得两个不同数据集的 100 个采样点完全相同的概率极低（对随机数据约为 2^(-128)）。

### 参数指纹 `_make_cache_key(...)`

Cache key 包含所有影响 CV 结果的参数：

- `X_shape`, `y_shape` — 数据维度
- `alphas` — alpha 网格（如有）
- `n_alphas`, `alpha_min_ratio` — 网格生成参数
- `fit_intercept`, `use_gpu`, `max_iter`, `tol` — 求解器参数
- `cpu_solver`, `cv_method`, `cd_kkt_check_every` — 算法参数
- `fold_indices` — 每 fold 前 5 个 index
- `sample_weight_shape` — 权重维度
- `data_digest` — 来自 `_hash_data` 的数据指纹

### LRU 缓存

```python
_LASSO_CV_ALPHA_CACHE = {}  # 全局字典
_LASSO_CV_ALPHA_CACHE_MAXSIZE = 16  # 最多缓存 16 个结果

def _cache_get(key):
    val = cache.get(key)
    if val is not None:
        cache.move_to_end(key)  # LRU: 最近使用移到末尾
    return val

def _cache_put(key, value):
    cache[key] = value
    while len(cache) > MAXSIZE:
        cache.popitem(last=False)  # 淘汰最久未用
```

### 缓存与 GPU 的关系

Cache hash **不是 GPU 特有的优化**，但对 GPU 路径特别有价值：

| 开销来源 | CPU | GPU |
|----------|-----|-----|
| 数据传输 (H2D) | 无 | ~1-10ms |
| JIT 编译 (torch.compile) | 无 | ~100ms 首次 |
| CV 计算本身 | 相同 | 相同或更快 |

GPU 的首次调用开销（JIT + H2D）使得缓存命中时的节省更大。

### 精度影响

Cache hash **不影响估计精度**：

- 缓存存储的是完整的 CV 结果（alpha, mse_path, coef_）
- 碰撞概率极低（blake2b 128-bit + 参数区分）
- 缓存未命中时正常计算，结果完全相同

唯一的理论风险是 hash 碰撞导致返回错误数据集的结果，但对随机数据概率约为 2^(-128)。

## Alpha 约定

所有惩罚在 `PenalizedGeneralizedLinearModel` 和专用包装器中使用一致的 `alpha`。

| 惩罚 | statgpu Alpha | sklearn Alpha | 内部一致性 |
|------|--------------|---------------|-----------|
| L1 | `alpha` | `alpha` | `Lasso(a) == PGLM(a, penalty='l1')` |
| ElasticNet | `alpha` | `alpha` | `ElasticNet(a) == PGLM(a, penalty='elasticnet')` |
| L2 (Ridge) | `alpha` | `alpha / n` | `Ridge(a) == PGLM(a, penalty='l2')` |

**sklearn 映射**：Ridge 需要 `sklearn_alpha = statgpu_alpha * n`。Lasso/ElasticNet 直接使用相同的 alpha。

内部一致性已验证到机器精度（diff ~1e-16）。

## 已知限制

### 非均匀 sample_weight + 非 L2 惩罚

非均匀 `sample_weight` 对 L2 以外的惩罚**不支持**：

| 惩罚 | 求解器 | 非均匀权重 |
|------|--------|-----------|
| L2 | IRLS | ✅ 支持 |
| L1, ElasticNet | FISTA | ❌ 抛出 ValueError |
| SCAD, MCP | FISTA | ❌ 抛出 ValueError |
| Adaptive L1 | FISTA | ❌ 抛出 ValueError |
| Group Lasso/MCP/SCAD | FISTA | ❌ 抛出 ValueError |

底层求解器（`fista`, `fista_bb`）拒绝非均匀 `sample_weight`。这是求解器层面的限制，不是 CV 限制。

**临时方案**：对加权 GLM 使用 `penalty='l2'` 配合 `solver='irls'`。

**后续工作**：在 `fista_solver` 和 `fista_bb_solver` 中实现加权梯度计算（`X' diag(w) residual / sum(w)`），以支持所有惩罚的非均匀权重。

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

## FAQ

**Q: 为什么 CV 缓存没有命中？**
CV 缓存使用 blake2b 哈希检测数据变化。以下情况会导致缓存未命中：
- 数据数组的内存地址变化（即使数值相同）
- `sample_weight` 变化
- `alpha_grid` 变化
- 数据形状变化

**Q: `n_jobs` 参数有什么作用？**
当前 `n_jobs` 被接受但 fold 循环是顺序执行的。这是为了未来并行化预留的接口。

**Q: 为什么 SCAD/MCP 的 CV 比 L1/ElasticNet 慢？**
SCAD/MCP 使用 LLA（Local Linear Approximation）迭代求解，每个 alpha 值需要多轮 LLA 迭代。L1/ElasticNet 只需要一轮 FISTA 求解。

**Q: 如何选择 `cv` 折数？**
- 默认 5 折：平衡偏差和方差
- 10 折：更准确的误差估计，但更慢
- Leave-one-out：n 很小时可用，但方差高

**Q: 为什么 `PenalizedGLM_CV` 的 `alpha_grid` 与 sklearn 不同？**
statgpu 使用数据驱动的 alpha 网格：`alpha_max` 从 `max(|X'y|)/n` 计算，然后按几何级数衰减。sklearn 使用类似但可能有细微差异的策略。

## 参见

- [Solver × Penalty 兼容性矩阵](solver-penalty-matrix.md) — 完整分发表和 CV 快速路径详情
- [Ridge 模型](../models/ridge.md) — RidgeCV 模型上下文
- [ElasticNet 模型](../models/elastic-net.md) — ElasticNetCV 模型上下文
- [GLM 模型](../models/generalized-linear-model.md) — PenalizedGLM_CV 模型上下文

## External Validation

**测试脚本：**
- `dev/tests/test_pr49_regression.py` — 2400+ 行回归测试，覆盖 CV 参数验证、kfold 完整性、缓存一致性
- `dev/tests/test_glm_penalty_review_fixes.py` — 2015 行 penalty 测试
- `dev/tests/test_elasticnet_cv.py` — ElasticNetCV 专项测试
- `dev/tests/test_ridge_cv.py` — RidgeCV 专项测试

**基准测试：**
- `dev/tests/benchmark_cv_full.py` — CV 全量基准测试
- `dev/benchmarks/benchmark_lassocv_impls.py` — LassoCV 实现对比

**外部框架对比：**
- RidgeCV vs sklearn `RidgeCV`：alpha 选择和 MSE 对齐
- ElasticNetCV vs sklearn `ElasticNetCV`：l1_ratio 和 alpha 选择对齐
- PenalizedGLM vs R `glmnet`：系数路径和 deviance 对齐
