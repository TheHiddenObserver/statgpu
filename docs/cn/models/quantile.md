# 分位数回归

> 语言：中文  
> 最后更新：2026-07-01  
> 页面定位：模型文档  
> 切换：[English](../en/models/quantile.md)

## 概述

`QuantileLoss` 实现 quantile 回归的 pinball（check）损失。`PenalizedQuantileRegression` 封装了最多 10 种惩罚和 8 种求解器，包括专门针对 SCAD/MCP 的 Proximal IRLS-CD 求解器。

| 组件 | 路径 |
|------|------|
| 损失 | `statgpu.losses.QuantileLoss` |
| 独立模型 | `statgpu.linear_model.QuantileRegression` |
| 惩罚模型 | `statgpu.linear_model.penalized.PenalizedQuantileRegression` |
| 专用求解器 | `statgpu.solvers._proximal_irls_quantile.proximal_irls_quantile_solver` |
| R 等价 | `quantreg::rq()` |

## 目标函数

Pinball 损失，在分位数 τ ∈ (0, 1) 处：

$$
\ell(\eta, y) = \rho_\tau(y - \eta), \quad \rho_\tau(u) = u \cdot (\tau - \mathbf{1}\{u < 0\})
$$

逐样本梯度（subgradient，在 u=0 处）：

$$
\frac{\partial \ell}{\partial \eta} = -\tau + \mathbf{1}\{y - \eta < 0\}
$$

关键属性：梯度是阶梯函数，不随残差大小变化。因此 `has_hessian = False`、`smooth_gradient = False`。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值范围 (0, 1)。τ=0.5 为中位数回归。 |

无尺度参数；quantile 回归是尺度无关的。

## 求解器兼容性

| 求解器 | 支持 | 说明 |
|--------|:---:|------|
| Proximal IRLS-CD | ✅ | 专用：IRLS 上界 + LLA 处理 SCAD/MCP。大规模 ~49x GPU 加速。 |
| FISTA | ✅ | 非光滑惩罚（L1、SCAD、MCP）和非凸 group 惩罚。 |
| IRLS | ✅ | 光滑惩罚（L2、none）。使用 Frisch-Newton 算法（匹配 statsmodels QuantReg）。 |
| L-BFGS | ✅ | 光滑惩罚，中低维度。 |
| ADMM | ✅ | 所有惩罚的替代方案。 |
| Newton | ❌ | Quantile 无 Hessian。 |
| Proximal Newton | ❌ | Quantile 无 Hessian。 |

## 惩罚兼容性

| 惩罚 | 求解器 (auto) | 说明 |
|---------|---------------|-------|
| l2 / none | IRLS | 5-15 次迭代收敛。 |
| l1 / elasticnet | FISTA | 基于 subgradient。 |
| SCAD / MCP | Proximal IRLS-CD | 最快：CPU ~3x / GPU ~49x 加速。 |
| adaptive_l1 | FISTA-LLA | 加权 L1 proximal。 |
| group_* | FISTA-LLA | Group proximal 算子。 |

## 示例

### CPU

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

# 中位数回归 (τ=0.5)
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)
print(model.coef_)

# 上四分位数 + L2 惩罚
model = PenalizedQuantileRegression(quantile=0.75, penalty='l2', alpha=0.01)
model.fit(X, y)

# 下四分位数 + MCP
model = PenalizedQuantileRegression(quantile=0.25, penalty='mcp', alpha=0.1)
model.fit(X, y)
```

### GPU (torch-CUDA)

```python
import torch
X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X_t, y_t)
```

### 加权 Quantile

```python
sample_weight = np.ones(n)
sample_weight[:50] = 5.0  # 前 50 个样本权重加倍

model = PenalizedQuantileRegression(quantile=0.5, penalty='l2', alpha=0.01)
model.fit(X, y, sample_weight=sample_weight)
```

## 算法详解

### Proximal IRLS-CD (SCAD/MCP)

对于 quantile + 非凸惩罚，专用求解器使用：

1. **IRLS 二次上界**：每次迭代计算权重 w_i = τ_i / max(|r_i|, ε)。形成非光滑 pinball 损失的二次上界：Q(β) = ½ Σ w_i(y_i − X_iβ)²。

2. **LLA（局部线性近似）**：非凸 SCAD/MCP 通过 P'(|β_j|) 权重转为加权 L1。

3. **并行对角化**：Jacobi 风格更新使用矩阵运算（每次 O(np)）——GPU 友好。

4. **GPU 优化**：收敛检查在 device 上比较，仅同步 bool 到 CPU。每 5 次迭代检查。

### IRLS (L2/none)

使用 Frisch-Newton 算法（匹配 statsmodels `QuantReg`）：
1. IRLS 权重：w_i = (τ + (1−2τ)·1_{r_i<0}) / max(|r_i|, ε)
2. 求解加权最小二乘：(X'WX + n·α·I) β = X'Wy
3. 重复至收敛（~5-15 次迭代）

## 输出

| 属性 | 类型 | 说明 |
|------|------|------|
| `coef_` | (p,) float | 估计系数 |
| `intercept_` | float | 估计截距 |
| `n_iter_` | int | 迭代次数 |
| `quantile` | float | 目标分位数 |

## 外部验证

- **R `quantreg::rq()`**: IRLS 路径系数与 Frisch-Newton IRLS 匹配到 1e-6。
- **sklearn `QuantileRegressor`**: HiGHS LP 求解器产生相同的 active set 和系数（tol=1e-8）。
- **FISTA-LLA 对等性**: Proximal IRLS-CD 与 FISTA-LLA 的 active set 一致（rtol=0.15）。

## 注意事项

- Score 使用加权 pinball 损失：`score()` 返回负平均 pinball 损失以兼容 sklearn。
- `sample_weight` 全求解器支持。
- GPU 设备（`cuda`/`torch`）不静默回退 CPU。
- 大规模问题（n=10K, p=500）GPU 比 CPU 快 ~49x。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.
