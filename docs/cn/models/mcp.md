# MCP

> 语言：中文  
> 最后更新：2026-10-06  
> 页面定位：模型文档  
> 切换：[English](../../en/models/mcp.md)

语言切换：[English](../../en/models/mcp.md)

## 概述

`MCPRegression` 提供 MCP 惩罚（Minimax Concave Penalty）线性回归（Zhang, 2010）。MCP 是连续的非凸惩罚，可以减小对大系数的收缩。其 **oracle 性质**依赖渐近正则条件和调参条件，并非每次有限样本拟合都成立的保证。

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
2. **LLA 内循环**：
   - 计算 LLA 权重：$w_j = p'_{\lambda,\gamma}(|\beta_j|) = \max(\lambda - |\beta_j|/\gamma, 0)$
   - 通过 FISTA 求解加权 L1 问题
3. **热启动**：前一个 $\lambda$ 的解作为初始点

## Oracle 性质

在正则条件下（Zhang 2010, Theorem 1）：
- **选择一致性**：$\Pr(\hat{S} = S_0) \to 1$
- **渐近正态性**：$\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

当系数绝对值超过 $\gamma\lambda$ 时，MCP 惩罚导数为零。固定 $\lambda$ 时增大 $\gamma$ 会减弱凹性，使惩罚更接近 Lasso，并不意味着收缩偏差一般会减小。惩罚导数为零也不保证有限样本估计无偏或区间已校正变量选择。

## 推断

`MCPRegression` 默认设置 `compute_inference=False`。其构造函数不接受
`inference_method`；开启推断后拟合会报错，因为继承的自动模式不会替用户选择
以已选变量集合为条件的推断方法。
若要显式请求 高斯 `oracle` 或 `bootstrap`，请使用
`PenalizedLinearRegression(penalty="mcp", penalty_kwargs={"gamma": 3.0}, ...)`，
并设置所需的 `inference_method` 和 `compute_inference=True`。

`oracle` 在选中的活跃集上进行无惩罚重拟合，预测仍使用原惩罚系数。
这些普通区间不校正在同一响应数据上选择变量的影响。oracle 接口拒绝 GPU
父模型，但子模型目前默认使用 `device="auto"`，因此 CPU 父模型不保证子模型
也在 CPU 上运行。残差 `bootstrap` 仅适用于无样本权重、
`cov_type="nonrobust"` 的 高斯 模型；它固定调参配置，并支持拟合时使用的
数值后端，但不会自动校正调参或变量选择的不确定性。
完整的通用估计器示例和方法限制见[推断模式](../guides/inference-modes.md#scadmcp-active-set-inference)。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 正则化强度（$\lambda$） |
| `gamma` | `3.0` | 凹度参数（$\gamma > 1$，Zhang 推荐 3.0） |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 每个 LLA 步骤的最大 FISTA 迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto`；显式 GPU 请求需要可用的 CUDA 后端 |
| `compute_inference` | `False` | 此封装类应保持关闭；显式推断方法请使用上面的通用估计器 |
| `solver` | `"auto"` | 求解器选择 |
| `gpu_memory_cleanup` | `False` | CuPy 内存池清理 |

## MCP vs SCAD vs Lasso

| 性质 | Lasso | SCAD | MCP |
|---|---|---|---|
| 凸性 | 凸 | 非凸 | 非凸 |
| Oracle 性质 | 一般不成立 | 依赖正则与调参条件 | 依赖正则与调参条件 |
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
