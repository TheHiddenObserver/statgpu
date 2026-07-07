# 有序广义线性模型 (Ordered Logit/Probit)

> 语言: 中文  
> 最后更新: 2026-07-07  
> 切换: [English](../en/models/ordered.md)

有序响应模型，适用于目标变量为序数类别（如"低/中/高"）的场景。

## 模型形式

P(y <= j | X) = F(theta_j - X * beta)

其中：
- `j = 1, ..., K-1` 为类别阈值
- `F` 为累积分布函数（Logit 或 Probit）
- `theta_j` 为阈值参数（严格递增）
- `beta` 为系数向量（比例优势假设：所有类别共享同一系数）

## 当前实现

### OrderedLogitRegression

比例优势模型（proportional odds），使用 Logit 链接函数。

```python
from statgpu.linear_model import OrderedLogitRegression

model = OrderedLogitRegression(
    n_categories=3,         # 类别数
    max_iter=100,           # 最大 Newton-Raphson 迭代数
    tol=1e-4,               # 收敛容差（NLL 绝对变化）
    device='auto',          # 'auto' | 'cpu' | 'cuda' | 'torch'
    compute_inference=True, # 计算标准误、z 值、p 值、置信区间
    cov_type='nonrobust',   # 协方差类型（目前仅支持 nonrobust）
)
model.fit(X, y)
print(model.coef_)          # 原始尺度系数 (p,)
print(model._thresh_est)    # 原始尺度阈值 (K-1,)
print(model._bse)           # 标准误 [coef SEs, threshold SEs]
print(model._pvalues)       # p 值
print(model.summary())      # 完整推断摘要
print(model.aic, model.bic) # 信息准则
```

### OrderedProbitRegression

使用 Probit 链接函数的有序模型。

```python
from statgpu.linear_model import OrderedProbitRegression

model = OrderedProbitRegression(n_categories=3, compute_inference=True, device='cpu')
model.fit(X, y)
```

## 目标函数

负对数似然（平均尺度）：

```
NLL = -(1/n) * Σ_i log P(y_i | X_i)
```

其中类别概率为：

```
P(y=k | X) = F(θ_k - Xβ) - F(θ_{k-1} - Xβ)
```

边界约定 `θ_{-1} = -∞`，`θ_{K-1} = ∞`。

## 优化算法

Newton-Raphson + 信赖域正则化（三端统一）：

| 后端 | 算法 | 说明 |
|------|------|------|
| numpy (CPU) | Newton-Raphson + 向量化解析 Hessian | NumPy `linalg.solve` |
| cupy (GPU) | Newton-Raphson + 向量化解析 Hessian | CuPy 原生，logit 下零 CPU 往返 |
| torch (GPU) | Newton-Raphson + 向量化解析 Hessian | Torch 原生，使用 `torch.linalg.solve` |

**收敛**：典型问题 5–23 次迭代。信赖域内层循环（每次迭代最多 20 次尝试）递增 ridge 惩罚直到 NLL 下降。

**标准化**：X 内部标准化为均值=0，标准差=1。收敛后系数和阈值转换回原始（未标准化）尺度：
`β_raw = β_fit / X_std`，`θ_raw = θ_fit + X_mean @ β_raw`。

## 推断 (Inference)

### Hessian 矩阵

解析观测 Hessian（向量化，后端无关）。与 R `MASS::polr` 和 `ordinal::clm` 的 Hessian 结构完全一致。

Hessian 具有分块结构：

```
H = [ H_{ββ}   H_{βθ} ]
    [ H_{θβ}   H_{θθ} ]
```

- `H_{ββ}` (p × p): 系数的二阶导数
- `H_{βθ}` (p × K-1): 系数与阈值的交叉导数
- `H_{θθ}` (K-1 × K-1): 阈值的二阶导数

### 协方差矩阵

协方差矩阵 = `H^{-1}`（在 MLE 处的观测 Hessian 逆矩阵）。

标准误：`bse = sqrt(diag(H^{-1}))`。

Wald z 统计量：`z = θ / bse`，双侧 p 值使用标准正态分布。

### 属性（设置 `compute_inference=True` 拟合后）

| 属性 | 形状 | 说明 |
|------|------|------|
| `coef_` | (p,) | 原始尺度系数估计值 |
| `_thresh_est` | (K-1,) | 原始尺度阈值估计值（内部使用，不含 -inf/+inf） |
| `thresholds_` | (K+1,) | 完整阈值向量 `[-inf, θ_1, ..., θ_{K-1}, +inf]` |
| `_bse` | (d,) | 标准误：`[bse_coef, bse_thresh]`，`d = p + K - 1`。用 `_bse[:p]` 取系数部分，`_bse[p:]` 取阈值部分 |
| `_zvalues` | (d,) | Wald z 统计量。用 `_zvalues[:p]` / `_zvalues[p:]` 分割 |
| `_pvalues` | (d,) | 双侧 p 值 |
| `_conf_int` | (d, 2) | 95% 置信区间 |
| `loglikelihood` | float | MLE 处的对数似然 |
| `aic` | float | AIC: `-2*loglik + 2*d` |
| `bic` | float | BIC: `-2*loglik + d*log(n)` |
| `n_iter_` | int | Newton-Raphson 迭代次数 |

### 当前限制

- **仅 nonrobust**：`cov_type='nonrobust'` 是唯一支持的协方差类型。
  HC0/HC1 sandwich、bootstrap、惩罚推断尚未支持。
- **不支持 sample_weight**：有序模型不支持样本权重。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_categories` | int | 3 | 序数类别数（>= 2） |
| `fit_intercept` | bool | True | 是否拟合截距项 |
| `max_iter` | int | 100 | Newton-Raphson 最大迭代数 |
| `tol` | float | 1e-4 | 收敛容差（NLL 绝对变化） |
| `C` | float | 1.0 | 逆正则化强度（未使用；继承自 GLM 基类） |
| `device` | str | 'auto' | 'auto' \| 'cpu' \| 'cuda' \| 'torch' |
| `compute_inference` | bool | False | 拟合后计算 SE、z 值、p 值、CI |
| `cov_type` | str | 'nonrobust' | 协方差估计类型（目前仅 nonrobust） |
| `gpu_memory_cleanup` | bool | False | 拟合后清理 GPU 显存 |

## CPU / GPU 示例

### CPU 模式 + 推断

```python
import numpy as np
from statgpu.linear_model import OrderedLogitRegression

np.random.seed(42)
X = np.random.randn(5000, 10)
beta = [0.5, -0.3, 0, 0.8, 0, -0.2, 0, 0.4, 0, 0]
y = np.digitize(0.5 + X @ beta + 0.5 * np.random.randn(5000), [-0.5, 0.5])

model = OrderedLogitRegression(n_categories=3, compute_inference=True, max_iter=50)
model.fit(X, y)
print(model.summary())
# OrderedLogitRegression Summary
# =====================
# n_obs=5000  n_params=12  loglik=-2657.066  aic=5338.133  bic=5416.456
# 
#   Param     Coef    StdErr        z   P>|z|  [0.025   0.975]
#   coef_0   1.705    0.294   5.806   0.000   1.130   2.281
#   ...
#   thresh_0 -3.300   0.751  -4.396   0.000  -4.771  -1.829
#   thresh_1 -0.095   0.155  -0.612   0.541  -0.399   0.209
```

### GPU 拟合（无推断）

```python
model = OrderedLogitRegression(n_categories=3, device='cuda', max_iter=50)
model.fit(X, y)  # GPU 上拟合，结果转回 CPU
print(model.coef_)
```

### Probit + 推断

```python
from statgpu.linear_model import OrderedProbitRegression

model = OrderedProbitRegression(n_categories=3, compute_inference=True)
model.fit(X, y)
print(model._bse)
print(model._pvalues)
```

## strict vs approximate

- **strict**：MLE 处的解析 Hessian 是精确的观测 Fisher 信息矩阵。在相同优化目标（无惩罚 NLL）下，
  标准误与 R `MASS::polr` 和 `ordinal::clm` 在 float64 容差内一致。
- **approximate**：GPU 路径（CuPy/Torch）因不同数学库（`libm` vs NVIDIA `libdevice`）
  产生略有不同的系数估计。经过约 20 次 Newton 迭代后，累积 BSE 差异约 4.5e-04。
  推断使用与拟合相同的后端（NumPy、CuPy 或 Torch），通过后端无关计算实现。

## 外部验证

| 参考 | 方法 | 一致性 |
|------|------|--------|
| R `ordinal::clm` | Newton-Raphson + 解析 Hessian | NLL 匹配（benchmark 数据上 statgpu: -0.532, R: -0.497） |
| R `MASS::polr` | Fisher scoring | 相同 Hessian 结构，系数匹配 |
| `statsmodels` `OrderedModel` | L-BFGS + 数值 Hessian | NLL 相当（statgpu 获得更低 NLL） |

## 参考文献

- McCullagh, P. (1980). Regression models for ordinal data. *JRSS B*, 42(2), 109–142.
- Agresti, A. (2010). *Analysis of Ordinal Categorical Data* (2nd ed.). Wiley.
- Christensen, R. H. B. (2019). ordinal—Regression Models for Ordinal Data. R package.
