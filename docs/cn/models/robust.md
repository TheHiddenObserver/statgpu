# 稳健回归

> 语言：中文  
> 最后更新：2026-07-01  
> 页面定位：模型文档  
> 切换：[English](../../en/models/robust.md)

## 概述

通过 M-估计实现稳健回归，支持自动尺度估计。`PenalizedRobustRegression` 封装了 Huber、Bisquare 和 Fair 损失，支持最多 10 种惩罚和 8 种求解器，包括专门针对 SCAD/MCP 的 Proximal Newton 求解器。

| 组件 | 路径 |
|------|------|
| Huber 损失 | `statgpu.losses.HuberLoss` |
| Bisquare 损失 | `statgpu.losses.BisquareLoss` |
| Fair 损失 | `statgpu.losses.FairLoss` |
| 惩罚模型 | `statgpu.linear_model.penalized.PenalizedRobustRegression` |
| R 等价 | `MASS::rlm()` |

## 损失函数

### Huber 损失

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & |y - \eta| \le \delta \\
\delta|y - \eta| - \frac{1}{2}\delta^2 & \text{否则}
\end{cases}
$$

- `smooth_gradient=True`、`has_hessian=True`
- δ→∞ 退化为 OLS；δ→0 退化为 LAD
- 默认 ε=1.345 在高斯分布下 95% 效率

### Bisquare (Tukey biweight) 损失

$$
\ell(\eta, y) = \rho_c(y - \eta),\quad
\rho_c(u) = \begin{cases}
\frac{c^2}{6}\bigl[1 - (1 - (u/c)^2)^3\bigr] & |u| \le c \\
c^2/6 & |u| > c
\end{cases}
$$

- `smooth_gradient=True`、`has_hessian=True`
- |u|>c 时完全忽略残差（梯度=0）
- 比 Huber 更高的 breakdown point
- 默认 ε=4.685 在高斯分布下 95% 效率

### Fair 损失

$$
\ell(\eta, y) = c^2\left[\frac{|y-\eta|}{c} - \log(1 + \frac{|y-\eta|}{c})\right]
$$

- `smooth_gradient=True`、`has_hessian=True`
- 比 Huber 更温和，小残差更接近 OLS

## 参数

### HuberLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `1.0` | 固定阈值 |
| `epsilon` | `1.345` | 稳健性调节（auto-scale 模式） |
| `method` | `"MAD"` | 尺度估计方法 |

### BisquareLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `epsilon` | `4.685` | 稳健性调节 |
| `method` | `"MAD"` | 尺度估计方法 |

## 尺度估计

`epsilon` 模式（auto-scale）下，尺度 σ 在拟合前估计：

- **MAD**: σ̂ = median(|r_i|) / 0.6745
- **Huber Proposal 2**: 迭代重估计

然后 δ = ε · σ̂（Huber）或 c = ε · σ̂（Bisquare）。

## 求解器兼容性

| 求解器 | Huber | Bisquare | Fair | 说明 |
|--------|:---:|:---:|:---:|------|
| Proximal Newton | ✅ | ✅ | ✅ | SCAD/MCP 最快：5-10 次迭代 |
| FISTA | ✅ | ✅ | ✅ | 任意惩罚 |
| FISTA-BB | ✅ | ✅ | ✅ | 自适应步长 |
| IRLS | ✅ | ✅ | ✅ | 仅光滑惩罚 |
| Newton | ✅ | ✅ | ✅ | L2 惩罚 |
| L-BFGS | ✅ | ✅ | ✅ | 中低维度 |

## 示例

```python
from statgpu.linear_model.penalized import PenalizedRobustRegression

# Huber + SCAD
model = PenalizedRobustRegression(loss='huber', penalty='scad', alpha=0.1)
model.fit(X, y)

# Bisquare + MCP
model = PenalizedRobustRegression(loss='bisquare', penalty='mcp', alpha=0.1)
model.fit(X, y)
```

### GPU (torch-CUDA)

```python
import torch
X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedRobustRegression(loss='huber', penalty='scad', alpha=0.1)
model.fit(X_t, y_t)
```

## 算法详解

### Proximal Newton (SCAD/MCP)

1. 计算 Hessian H = X'WX 和梯度 g
2. Newton 方向：d = -H⁻¹·g
3. Armijo 线搜索 + proximal 步
4. 更新：β_new = proximal(β − step·d, step)
5. 通常每 LLA 步 5-10 次迭代

## 输出

| 属性 | 类型 | 说明 |
|------|------|------|
| `coef_` | (p,) float | 估计系数 |
| `intercept_` | float | 估计截距 |
| `n_iter_` | int | 迭代次数 |
| `loss` | str | 损失名称 |

## 外部验证

- **Huber**: 与 R `MASS::rlm(psi=psi.huber)` 对齐，系数一致。
- **Bisquare**: 与 R `MASS::rlm(psi=psi.bisquare)` 对齐；SCAD/MCP active set 与 FISTA-LLA 一致。
- **Fair**: 与 R `MASS::rlm(psi=psi.fair)` 对齐。

## 注意事项

- `BisquareLoss` + SCAD/MCP：在 LAST continuation step（target α）warm-start（v0.2.1 修复）。
- 尺度估计使用 CPU numpy；GPU 数据自动转换。
- 所有损失接受 `sample_weight`。
- 三种损失 `has_hessian=True`，支持 proximal Newton。

## 参考文献

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Holland, P. W. & Welsch, R. E. (1977). Robust Regression using IRLS. *Communications in Statistics*, A6(9), 813-827.
