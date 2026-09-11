# 交叉验证

> 语言：中文  
> 最后更新：2026-09-11
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
| `PenalizedGLM_CV` | `PenalizedGeneralizedLinearModel` 或 `PenalizedCoxPHModel` | family 支持的 penalty | `statgpu.linear_model.PenalizedGLM_CV` |

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

### 惩罚 Cox 交叉验证

Cox target 在整个选择流程中必须保持二维：

```python
survival_y = np.column_stack([time, event])
model = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="elasticnet",        # l1、l2、elasticnet、scad 或 mcp
    l1_ratio=0.4,
    alpha_grid=[0.2, 0.05, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cuda",               # 同时支持 NumPy CPU 与 Torch CUDA
).fit(X_cuda, survival_y_cuda)
```

该路径要求每个可评估 fold 都提供有限的 held-out Cox partial-likelihood
证据，并以无截距 `PenalizedCoxPHModel` 完成最终重拟合。若所有候选均无
完整证据，会直接抛错而不是默认选择第一个 alpha。该分支仅提供估计，
不发布 post-selection 系数推断；不支持 `two_stage`、sample weights 或字典 target。

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

### LassoCV 专用

`LassoCV` 将交叉验证阶段与最终全数据重拟合阶段的 solver 控制明确分开。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alphas` | array | `None` | Alpha 网格；`None` 时自动生成。 |
| `n_alphas` | int | `12` | 自动生成时的 alpha 数量。 |
| `solver` | str | `"fista"` | 最终全数据 `Lasso` 重拟合使用的 solver。 |
| `cv_solver` | str | `"auto"` | CV folds/path 的 solver；`auto` 在 CPU 上解析为 coordinate descent，在 CUDA/Torch 上解析为 FISTA。 |
| `cpu_solver` | str/None | `None` | **已弃用**的旧 CPU-CV 控制。CPU 上在新控制为 `auto` 时作为 `cv_solver` alias；CUDA/Torch 上只 warning，不会替换 GPU FISTA。 |
| `method` | str | `"standard"` | CV path profile；CPU 的 `glmnet` 固定 coordinate descent，CUDA/Torch 仍保持 FISTA。 |
| `cd_kkt_check_every` | int/None | `None` | 适用时 coordinate descent 的 KKT 扫描频率。 |
| `gpu_cv_mixed_precision` | bool | `True` | GPU CV path 是否使用混合精度。 |
| `compute_inference` | bool | `False` | 仅对选定后的最终全数据重拟合执行推断。 |

拟合后，`cv_solver_` 记录设备与 `method` 解析后**实际执行**的 CV 算法。`cpu_solver` 的弃用与迁移语义见 [penalized solver API 迁移指南](penalized-solver-api-migration.md)。

### ElasticNetCV 专用

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `l1_ratio` | float/list | `0.5` | L1 混合比。传列表可搜索多个值。 |
| `alphas` | array | `None` | Alpha 网格。 |
| `n_alphas` | int | `100` | Alpha 数量。 |
| `compute_inference` | bool | `False` | 对最终全数据 ElasticNet 重拟合执行 debiased 推断。 |

各折拟合仍仅用于估计；只有在所选 `alpha` 与 `l1_ratio` 使用全部观测重拟合后才计算推断。

### PenalizedGLM_CV 专用

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loss` | str | `"squared_error"` | 损失族（见 [Solver × Penalty 矩阵](solver-penalty-matrix.md)）。 |
| `penalty` | str | `"l2"` | 惩罚类型。 |
| `penalty_kwargs` | dict | `{}` | 惩罚参数（如 SCAD 的 `{"a": 3.7}`）。 |
| `alpha_grid` | array | `None` | Alpha 网格。 |
| `n_alphas` | int | `100` | Alpha 数量。 |
| `cv_splits` | list | `None` | 自定义折分割 `[(train_idx, val_idx), ...]`。 |
| `loss_kwargs` | dict | `{}` | loss 选项；Cox 接受 `ties="breslow"` 或 `ties="efron"`。 |
| `compute_inference` | bool | `False` | 只在 selected full-data final refit 上运行 coefficient inference；fold/path/grid 拟合保持 estimation-only。 |
| `inference_method` | str | `"auto"` | final-refit inference request；受支持的 non-Gaussian L2/no-penalty 行解析为 fixed-penalty M-estimation。 |
| `cov_type` | str | `"nonrobust"` | final-refit covariance；non-Gaussian penalized M-estimation 当前支持 nonrobust/HC0/HC1。 |
| `hac_maxlags` | int/None | `None` | 保留底层控制；不表示 non-Gaussian penalized HAC 已受支持。 |

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

对于 penalized Cox CV，每个自定义 train/validation pair 只需非空且互不重叠；
train 不必是 validation 的补集，各 fold 的 validation 也不必把样本恰好覆盖一次。
因此可使用前向 `TimeSeriesSplit` 和 repeated holdout。索引必须是一维、位于
signed-int64 与样本边界内的精确整数。布尔值、数字字符串、小数、NaN/Inf、溢出、
重复、交叠或越界索引都会在任何 candidate fit 前被拒绝。每个参与评估的 Cox
train 与 validation partition 都必须至少包含一个观察事件。

## 样本权重

大多数标量响应 CV 估计器支持 `sample_weight`；生存路径见下方限制：

```python
model = RidgeCV(cv=5)
model.fit(X, y, sample_weight=w)
print(f"加权 R²: {model.score(X_test, y_test, sample_weight=w_test):.4f}")
```

**边界**（见 [已知限制](#已知限制)）：
- 标量响应 sample-weight 能力取决于具体 loss/penalty/solver path；不受支持的显式 solver 组合会直接报错，而不会改变用户请求的 objective。
- 对 PR #142 的 inference-enabled smooth non-Gaussian L2/no-penalty contract，non-uniform analytic weights 受支持。public `solver="auto"` 时，candidate selection 与 selected final refit 使用已有 weight-capable FISTA，因为 Newton 当前拒绝 non-uniform weights；public solver request 仍保持 `auto`。
- `loss="cox_ph"` 会拒绝 `sample_weight`；加权惩罚 Cox CV 尚未实现。

## Alpha 网格

### 自动生成网格

`alphas=None` 时，网格生成方式：
1. 计算 `alpha_max = max(|X'y|) / n`（或加权变体）
2. 生成从 `alpha_max` 到 `alpha_max * alpha_min_ratio` 的 `n_alphas` 个值
3. 网格为 log 等距：`np.logspace(log10(alpha_max * ratio), log10(alpha_max), n_alphas)`

Penalized Cox 使用零模型处 partial-likelihood gradient 的无穷范数。ElasticNet
在 `l1_ratio=rho > 0` 时，首个值是零模型 KKT 边界
`alpha_max = ||gradient L(0)||_inf / rho`；字符串 penalty 使用 estimator 的
`l1_ratio`，`ElasticNetPenalty` 对象使用对象自身的值。`rho=0` 是纯 L2，没有
有限的全零 KKT 阈值，因此明确把 `||gradient L(0)||_inf` 作为网格 heuristic
并记录。无 penalty 别名 `"none"`、`"null"` 与 `""` 不可调，Cox CV 会拒绝；
无惩罚运行应直接拟合 `PenalizedCoxPHModel`。

### 自定义网格

```python
import numpy as np

model = RidgeCV(
    alphas=np.logspace(-4, 2, 50),  # 自定义 50 点网格
    cv=5,
)
model.fit(X, y)
```

标量响应 CV estimator 只搜索严格为正的 alpha。一维数值用户网格会保留原顺序，
非正与非有限项会伴随 `RuntimeWarning` 被过滤；空网格或过滤后为空会 warning 并重新
生成默认网格。因此 L1、L2、ElasticNet、SCAD、MCP、Adaptive L1、Group Lasso、
Group SCAD 与 Group MCP 的标量 CV 都不会把零作为候选；无惩罚拟合应直接使用
`alpha=0` 的 estimator。在 NumPy dtype promotion 前，Python sequence 与 object array
会逐元素检查：布尔值与字符串/bytes（包括 `"0.2"` 这样的数字文本）会被拒绝，而不会
变成 1.0 或 0.2。非一维、复数、布尔、字符串/bytes 或其他非数值网格会在设备路由与
candidate 工作前抛出 `ValueError`。Penalized Cox 使用更严格的契约，
不会过滤或替换用户网格：非有限或负值会抛出 `ValueError`，SCAD/MCP 还要求每个
alpha 严格为正；L1、L2 与 ElasticNet Cox 网格允许零值。

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

`LassoCV` 额外暴露 `cv_solver_`，表示设备/`method` 解析后实际执行的 CV 算法；`ElasticNetCV` 额外有 `l1_ratio_`（传入列表时的最优 l1_ratio）。

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
| logistic + l1/en, p≥100, n×p≥500k | Torch | Fold-batch 路径 |
| poisson + l1/en, p≥500, n×p≥1M | Torch | Fold-batch 路径 |
| gamma + l1/en, p≥500, n×p≥2M | Torch | Fold-batch 路径 |
| 非 squared-error SCAD/MCP, n×p≥1M | Torch | 异步 FISTA 路径 |
| NB + l1/l2/en | CPU | 复杂梯度开销 |
| 通用 fallback 工作量 < 100M | CPU | 低于实测 GPU break-even |
| 通用 fallback 工作量 ≥ 100M | 优先 Torch，其次 CuPy；均不可用时 CPU | 聚合 CV 工作量较大 |

阈值基于 benchmark 数据，存储在 `_effective_cv_device()` 中。显式控制：`device="cpu"` 强制 CPU，`device="cuda"` 强制 GPU。
`device="auto"` 只在 backend 报告 CUDA driver 与设备实际可用后选择 GPU；仅安装
但无法运行的 CuPy wheel 不会阻止回退 CPU。通用 fallback 工作量为
`n * p * n_work_folds * n_alphas`；非 squared-error 的 SCAD/MCP 另乘 20 的
continuation factor。标量响应 CV 使用规范化后的 generated/custom fold 数，Cox CV
只统计 training 与 validation 都含事件的可评估 fold。前面的经验 loss/penalty 行优先
执行，仍只使用各自行中记录的 `n * p` 与 feature 条件，不乘 fold 数。显式
`device="cuda"` 仍采用严格契约，CuPy CUDA 不可用时会抛错。

## CV 后推断

`RidgeCV` 设置 `compute_inference=True` 时：

```python
model = RidgeCV(compute_inference=True, cov_type="hc1")
model.fit(X, y)
print(model.summary())
```

`PenalizedGLM_CV` 明确把 selection 与 coefficient inference 分开：

1. fold/path/grid candidate 拟合全部关闭 inference；
2. 先根据 held-out evidence 选择 alpha；
3. `compute_inference=True` 时，只在 selected full-data final refit 上运行一次 inference；
4. CV estimator 委托 final estimator 的 inference result，并记录 `penalty_conditioning_="cv_selected_penalty"` 与 `penalty_selection_adjusted_=False`。

支持的 final-refit 行包括既有 Gaussian inference contract，以及 smooth non-Gaussian L2/no-penalty 的 `m_estimation`（nonrobust/HC0/HC1）。non-Gaussian L1/ElasticNet coefficient inference 尚未实现，penalized Cox branch 仍为 estimation-only。完整 method/target matrix 见 [Penalized GLM inference](penalized-glm-inference.md)。

## 性能建议

1. **大规模用 GPU**：n×p > 200k 时设置 `device="cuda"` 或让 `auto` 决定。
2. **减少 alpha 网格**：`n_alphas=50` 通常足够；默认 100。
3. **使用混合精度**：`gpu_cv_mixed_precision=True`（默认）CV 使用 float32，GPU 上快 2-4x。
4. **两阶段 CV**：`cv_strategy="two_stage"` 快速筛选 alpha，再精炼 top 候选。
5. **自定义折**：预生成折可避免重复运行时的重新洗牌。

## 架构

`PenalizedGLM_CV.fit()` 会分派到两套 preparation 与 routing 顺序。二者共享选择
和最终重拟合契约，但 automatic grid 的构造位置不同，因此不能画成一条统一的有序
流水线。

### 标量响应顺序

```
PenalizedGLM_CV._fit_standard(X, y)
  │
  ├─ 1. 校验或生成完整 alpha 网格
  │     └─ automatic grid 在 CV 设备选择前生成
  │
  ├─ 2. 只 materialize 一次 generated/custom folds
  │     └─ 一次性 generator 转为可复用 fold list
  │
  ├─ 3. 选择 CV 设备 (_effective_cv_device)
  │     └─ 通用工作量估算使用 len(folds)
  │
  ├─ 4. 对 alpha 网格评分 (_compute_cv_scores)
  │     └─ Ridge 特征分解、fold-batch、sparse、LLA 或兜底路径
  │
  └─ 5. 选择最优 alpha 并重拟合
        ├─ squared_error + l2：在 CPU 上执行精确 float64 特征分解
        │  同时保留 cv_selected_device_ 作为预测/输出后端契约
        └─ 其他路径：在解析后的选定 backend 上重拟合
```

### 惩罚 Cox 顺序

```
fit_penalized_cox_cv(estimator, X, (time, event))
  │
  ├─ 1. 规范化 survival target 并 materialize folds
  │
  ├─ 2. 校验 alpha-grid request
  │     └─ 显式网格在此校验；automatic grid 尚不构造
  │
  ├─ 3. 校验每个 fold 的事件支持
  │     └─ 计算 fold_valid 与 n_effective_folds
  │
  ├─ 4. 选择 CV 设备 (_effective_cv_device)
  │     └─ 通用工作量估算使用 n_effective_folds
  │
  ├─ 5. 转换到选定 backend 并预处理 Cox loss
  │     └─ automatic alpha grid 在此 backend 上生成
  │
  ├─ 6. 仅对可评估 fold 评分，并保留 skipped-fold 诊断
  │
  └─ 7. 要求完整有限 candidate 证据，完成选择与重拟合
```

## CV 评分路径

以下编号路径描述标量响应 scoring。惩罚 Cox 在完成上述 preparation 后使用
survival-aware fold 路径。

### 路径 1：Ridge 特征分解（squared_error + l2）

**条件**：`loss="squared_error"`、`penalty="l2"`、解析后的 CV device 为 CPU，且 `sample_weight=None`。

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

这条批量 scoring 路径取决于解析后的 CV device，而不是 constructor 的字面值：
`device="auto"` 只有在自动路由解析为 CPU 时才使用它；若自动路由选择 CUDA/Torch，
CV 会使用相应的 GPU scoring 路径。完成选择后，squared-error L2 总会把完整重拟合
数据转换到 NumPy，并在 CPU 上执行精确 float64 `_ridge_eig_single()`，以保持 CV 与
重拟合系数的精度一致。拟合后的 estimator 仍保留 `cv_selected_device_` 作为预测/输出
backend 契约；该 metadata 并不表示重拟合特征分解运行在所选 accelerator 上。

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
    coef = fista_solver(..., init_coef=coef)  # Warm start
```

这比冷启动减少 3-5 倍迭代次数。

## 结果缓存

`LassoCV` 在 `statgpu.linear_model.wrappers._lasso` 中使用 selection-only LRU cache。缓存 payload 只包含 `alpha`、实际评估的 `alphas`、`mse_path` 和 `mean_mse`；最终全数据重拟合得到的 estimator 与系数不会存入这个 selection cache。

### LassoCV 数据身份

`X`、`y` 和 `sample_weight` 分别通过 `_array_identity_token(...)` 表示：

- `None` 有独立 token；
- NumPy、CuPy、Torch 数组的 token 包含 backend tag、shape、dtype 和 BLAKE2b digest；
- 行数不超过 100 时 hash 全部行；更大数组 hash 100 个均匀抽样行；
- CuPy/Torch 只把用于 hash 的抽样行传回 host。

这是**基于内容**的身份，而不是内存地址身份：重新分配一个 shape、dtype 与抽样内容相同的数组，不会仅因地址变化而 miss。

### LassoCV selection key

当前 `_make_lasso_cv_auto_cache_key(...)` 包含：

- `X`、`y`、`sample_weight` 的 identity token；
- 完整已评估 alpha 网格的 digest；
- **每个 fold 的完整 train/validation index 数组**的 digest；
- `fit_intercept` 与 CV 是否在 GPU 上执行；
- `max_iter`、`tol`；
- 解析后的 CV solver（helper 的历史字段名仍为 `cpu_solver`）；
- 规范化后的 `method` / `cv_method`；
- `cd_kkt_check_every`；
- `gpu_cv_mixed_precision`。

最终 refit 的 `solver` 被刻意排除，因为它不会改变 alpha scoring。因此只改变最终重拟合算法时，可以复用同一份 alpha-selection evidence。

### LRU payload 与容量

缓存使用 `OrderedDict`；读取命中后把条目移到末尾，插入超出容量后淘汰最久未使用的条目。默认容量是 **64**，导入时由 `STATGPU_LASSO_CV_CACHE_SIZE` 控制。命中时返回 selection payload 中 NumPy 数组的副本，避免调用者意外修改缓存内部对象。

本节描述的是当前 **LassoCV selection cache**。`RidgeCV` 与 `ElasticNetCV` 有各自的缓存实现，不应从这份 key 反推它们的具体字段。

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

### sample-weight solver 边界

sample-weight 能力是 path-specific 的，不能从某个历史 solver 的限制外推出所有 penalty。显式请求不受支持的 solver 会直接报错。

对 PR #142 的 coefficient-inference contract，smooth non-Gaussian L2/no-penalty + analytic weights 受支持。启用 inference 且 public request 为 `solver="auto"` 时，`PenalizedGLM_CV` candidate selection 与 selected final refit 都使用已有 weight-capable FISTA，同时 public request 保持 `auto`。penalized Cox CV branch 仍拒绝 `sample_weight`。

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
对 `LassoCV`，当抽样后的数据/权重内容或其他 selection-key 字段变化时会 miss，包括实际评估的 alpha 网格、完整 fold indices、CV solver/method 控制、容差/迭代设置、截距模式、CPU/GPU 执行方式以及 mixed-precision 配置。仅重新分配一个抽样内容、shape 与 dtype 相同的数组，本身不会导致 miss。

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
- `dev/tests/test_penalized_solver_api_cleanup.py` — LassoCV 阶段化 solver/deprecation 回归覆盖

**基准测试：**
- `dev/tests/benchmark_cv_full.py` — CV 全量基准测试
- `dev/benchmarks/benchmark_lassocv_impls.py` — LassoCV 实现对比

**外部框架对比：**
- RidgeCV vs sklearn `RidgeCV`：alpha 选择和 MSE 对齐
- ElasticNetCV vs sklearn `ElasticNetCV`：l1_ratio 和 alpha 选择对齐
- PenalizedGLM vs R `glmnet`：系数路径和 deviance 对齐
