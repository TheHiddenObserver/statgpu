# MCP

> 语言：中文  
> 最后更新：2026-10-06  
> 页面定位：模型文档  
> 切换：[English](../../en/models/mcp.md)

## 概述

`MCPRegression` 提供 MCP 惩罚（Minimax Concave Penalty）线性回归（Zhang, 2010）。MCP 是连续的非凸惩罚，可以减小对大系数的收缩。其 **oracle 性质**依赖渐近正则条件和调参条件，并非每次有限样本拟合都成立的保证。

当连续响应需要稀疏预测，且你关心 Lasso 对大斜率的收缩时，可以考虑这个模型。
若优先需要凸目标和更简单的调参过程，可先使用 [Lasso](lasso.md) 或
[Elastic Net](elastic-net.md)。非凸拟合可能得到不同的局部解；变量入选不自动
代表统计显著性或因果效应。

## 路径

`statgpu.linear_model.MCPRegression`

## 目标函数

$$
\min_{b,\beta} \frac{1}{2n}\|y - b\mathbf{1} - X\beta\|_2^2 + \sum_{j=1}^p p_{\lambda,\gamma}(|\beta_j|)
$$

其中 MCP 惩罚定义为：

$$
p_{\lambda,\gamma}(\theta) = \begin{cases}
\lambda\theta - \frac{\theta^2}{2\gamma} & \text{if } \theta \le \gamma\lambda \\
\frac{\gamma\lambda^2}{2} & \text{if } \theta > \gamma\lambda
\end{cases}
$$

凹度参数 $\gamma > 1$（默认 3.0，Zhang 推荐）。

这里 n 为观测数，X 含 p 列特征，b 是不受惩罚的截距，`alpha` 对应
$\lambda$；标量惩罚的自变量为 $\theta=|\beta_j|\geq0$。
`fit_intercept=False` 时固定 b=0。样本权重须有限、非负且总和为正；加权时将
数据拟合项改为 $\sum_i w_i(y_i-b-x_i^\top\beta)^2/(2\sum_i w_i)$。
所有权重同乘一个正数不改变该目标。特征不会自动标准化；如需缩放，应仅从
训练行学习变换。

## 完整 CPU 示例

下面的特征已具有相近尺度。拟合前留出最后 40 行；只有前两列特征产生信号。

<!-- learner-example: mcp-prediction -->
```python
import numpy as np
from statgpu.linear_model import MCPRegression

rng = np.random.default_rng(64)
X = rng.normal(size=(160, 5))
y = 1.5 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.4, size=160)
model = MCPRegression(
    alpha=0.1, gamma=3.0, device="cpu", compute_inference=False,
    max_iter=5000, tol=1e-8,
).fit(X[:120], y[:120])
prediction = model.predict(X[120:])
print("Slopes:", np.round(model.coef_, 3))
print("Intercept:", round(model.intercept_, 3))
print("Test R2:", round(model.score(X[120:], y[120:]), 3))
```

该随机种子下，斜率约为 `[1.987, -0.982, 0, 0, 0]`，截距约为 `1.430`，
留出集 R² 约为 `0.967`；预测形状为 `(40,)`。这些是惩罚预测系数，并非活跃集
重拟合或显著性检验结果。换一个样本仍可能选入噪声或遗漏真实信号。

## 参数选择与结果检查

- 在训练数据内通过验证选择 `alpha`。示例中的 0.1 不是通用最优值；内部延续
  路径只服务于求解，不会自动执行交叉验证。
- `gamma=3.0` 可作为常用起点。改变凹度会改变惩罚和优化难度；应比较
  预测表现及入选集合的稳定性，不能把更稀疏直接等同于更好。
- 增大 `max_iter`、收紧 `tol` 可以检查数值稳定性。`n_iter_` 本身不证明
  全局最优，局部最优仍可能存在。
- 测试集应与调参分开；每个验证训练折中重新学习预处理。这两个封装类不提供
  内置交叉验证方法。

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
若要显式请求高斯 `oracle` 或 `bootstrap`，请使用
`PenalizedLinearRegression(penalty="mcp", penalty_kwargs={"gamma": 3.0}, ...)`，
并设置所需的 `inference_method` 和 `compute_inference=True`。

`oracle` 在选中的活跃集上进行无惩罚重拟合，预测仍使用原惩罚系数。
这些普通区间不校正在同一响应数据上选择变量的影响。oracle 接口拒绝 GPU
父模型，但子模型目前默认使用 `device="auto"`，因此 CPU 父模型不保证子模型
也在 CPU 上运行。残差 `bootstrap` 仅适用于无样本权重、
`cov_type="nonrobust"` 的高斯模型；它固定调参配置，并支持拟合时使用的
数值后端，但不会自动校正调参或变量选择的不确定性。
完整的通用估计器示例和方法限制见[推断模式](../guides/inference-modes.md#scadmcp-active-set-inference)。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 有限正数正则化强度（$\lambda$） |
| `gamma` | `3.0` | 大于 1 的有限凹度参数（常用 3.0） |
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
| 大 $\beta_j$ 的偏差 | 向零收缩 | 阈值以上导数为零；不保证有限样本无偏 | 阈值以上导数为零；不保证有限样本无偏 |
| 惩罚连续性 | 连续 | 连续 | 连续 |
| 惩罚凹性 | 线性（凸） | 分段线性-二次 | 分段线性-二次 |
| 默认凹度参数 | — | $a = 3.7$ | $\gamma = 3.0$ |

## 可选 GPU 使用

准备好 CPU 示例中的数据后，在同一构造函数中设置 `device="cuda"` 可请求
CuPy CUDA，设置 `device="torch"` 可请求 Torch CUDA。显式后端不可用时会
报错；只有 `auto` 可以选择其他可用后端。详见[设备与内存](../guides/device-and-memory.md)。
上述 CPU 示例不需要 GPU。

## API 与输出形状

[完整 MCPRegression 方法参考](../reference/linear-model-api.md#mcpregression)
包含拟合、公式与权重、预测位置、评分、继承辅助方法及诊断限制。
上表列出了全部构造参数。应从 `statgpu.linear_model` 导入；顶层 `statgpu`
不导出这个类。

## 参考文献

- Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax concave penalty. *Annals of Statistics*, 38(2), 894-942.
- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
