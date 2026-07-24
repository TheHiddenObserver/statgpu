# MCP

> 语言：中文  
> 最后更新：2026-06-14  
> 页面定位：模型文档  
> 切换：[English](../../../en/models/mcp.md)

语言切换：[English](../../../en/models/mcp.md)

## 概述

`MCPRegression` 提供 MCP 惩罚（Minimax Concave Penalty）线性回归（Zhang, 2010）。MCP 是非凸惩罚，具有 **oracle property**，且惩罚函数连续——同时解决了 Lasso 的偏差和硬阈值的不连续性问题。

## 路径

`statgpu.linear_model.MCPRegression`

## 目标函数

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \sum_{j=1}^p p_{\lambda,\gamma}(|\beta_j|)
$$

其中 MCP 惩罚定义为：

$$
p_{\lambda,\gamma}(\theta) = \begin{cases}
\lambda\theta - \frac{\theta^2}{2\gamma} & \text{if } \theta \le \gamma\lambda \\
\frac{\gamma\lambda^2}{2} & \text{if } \theta > \gamma\lambda
\end{cases}
$$

凹度参数 $\gamma > 1$（默认 3.0，Zhang 推荐）。

## 算法

MCP 使用与 SCAD 相同的 **LLA + FISTA** 算法：

1. **延续路径**：从 $\lambda_{max}$ 沿几何网格递减
2. **LLA 内循环**（每个 $\lambda$ 迭代 1-6 次）：
   - 计算 LLA 权重：$w_j = p'_{\lambda,\gamma}(|\beta_j|) = \max(\lambda - |\beta_j|/\gamma, 0)$
   - 通过 FISTA 求解加权 L1 问题
3. **热启动**：前一个 $\lambda$ 的解作为初始点

## Oracle Property

在正则条件下（Zhang 2010, Theorem 1）：
- **选择一致性**：$\Pr(\hat{S} = S_0) \to 1$
- **渐近正态性**：$\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

MCP 产生**几乎无偏**的估计，偏差随 $\gamma$ 增大而减小。

## 推断

- 默认 `compute_inference=False`（MCP 不支持 debiased 推断）
- 对选中变量做推断可用 oracle 方法：在选中的 support set 上重新拟合 OLS
- 未来：oracle 推断和 BIC 超参选择（见 TO_DO.md）

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 正则化强度（$\lambda$） |
| `gamma` | `3.0` | 凹度参数（$\gamma > 1$，Zhang 推荐 3.0） |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 每个 LLA 步骤的最大 FISTA 迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` |
| `solver` | `"auto"` | 求解器选择 |
| `gpu_memory_cleanup` | `False` | CuPy 内存池清理 |

## MCP vs SCAD vs Lasso

| 性质 | Lasso | SCAD | MCP |
|---|---|---|---|
| 凸性 | 凸 | 非凸 | 非凸 |
| Oracle property | 无 | 有 | 有 |
| 大 $\beta_j$ 的偏差 | 向零收缩 | 几乎无偏 | 几乎无偏 |
| 惩罚连续性 | 连续 | 连续 | 连续 |
| 惩罚凹性 | 线性（凸） | 分段线性-二次 | 分段线性-二次 |
| 默认凹度参数 | — | $a = 3.7$ | $\gamma = 3.0$ |

## 示例

```python
from statgpu.linear_model import MCPRegression

model = MCPRegression(alpha=0.1, gamma=3.0)
model.fit(X, y)
print(model.coef_)        # 稀疏系数
print(model.score(X, y))  # R-squared

# 调整 gamma（凹度）
model_aggressive = MCPRegression(alpha=0.1, gamma=1.5)  # 更激进的阈值
```

## 参考文献

- Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. *Annals of Statistics*, 38(2), 894-942.
- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
