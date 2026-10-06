# SCAD

> 语言：中文  
> 最后更新：2026-10-06  
> 页面定位：模型文档  
> 切换：[English](../../en/models/scad.md)

语言切换：[English](../../en/models/scad.md)

## 概述

`SCADRegression` 提供 SCAD 惩罚（Smoothly Clipped Absolute Deviation）线性回归（Fan & Li, 2001）。SCAD 是非凸惩罚，可以减小对大系数的收缩。其 **oracle 性质**是依赖正则条件和调参条件的渐近结论，不是每次有限样本拟合都成立的保证。

## 路径

`statgpu.linear_model.SCADRegression`

## 目标函数

$$
\min_{\beta} \frac{1}{2n}\|y - X\beta\|_2^2 + \sum_{j=1}^p p_{\lambda,a}(|\beta_j|)
$$

其中 SCAD 惩罚定义为：

$$
p_{\lambda,a}(\theta) = \begin{cases}
\lambda \theta & \text{if } \theta \le \lambda \\
\frac{2a\lambda\theta - \theta^2 - \lambda^2}{2(a-1)} & \text{if } \lambda < \theta \le a\lambda \\
\frac{(a+1)\lambda^2}{2} & \text{if } \theta > a\lambda
\end{cases}
$$

凹度参数 $a = 3.7$（Fan & Li 推荐）。

## 算法

SCAD 使用 **LLA（局部线性近似）** + FISTA：

1. **延续路径**：从 $\lambda_{max}$ 沿几何网格递减
2. **LLA 内循环**：
   - 计算 LLA 权重：$w_j = p'_{\lambda,a}(|\beta_j|)$（SCAD 在当前估计处的次梯度）
   - 求解加权 L1 问题：$\min \frac{1}{2n}\|y - X\beta\|_2^2 + \sum w_j |\beta_j|$
   - 加权 L1 通过 FISTA 求解
3. **热启动**：用前一个 $\lambda$ 的解作为下一个 $\lambda$ 的初始点

## Oracle 性质

在正则条件下（Fan & Li 2001, Theorem 2）：
- **选择一致性**：$\Pr(\hat{S} = S_0) \to 1$
- **渐近正态性**：$\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

当系数绝对值超过 $a\lambda$ 时，SCAD 惩罚的导数为零，不再直接收缩这些坐标。这不保证有限样本估计无偏，也不保证数据驱动选择之后的区间有效。

## 推断

`SCADRegression` 默认设置 `compute_inference=False`。其构造函数不接受
`inference_method`；开启推断后拟合会报错，因为继承的自动模式不会替用户选择
以已选变量集合为条件的推断方法。
若要显式请求 高斯 `oracle` 或 `bootstrap`，请使用
`PenalizedLinearRegression(penalty="scad", penalty_kwargs={"a": 3.7}, ...)`，
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
| `a` | `3.7` | 凹度参数（Fan & Li 推荐 3.7） |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 每个 LLA 步骤的最大 FISTA 迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto`；显式 GPU 请求需要可用的 CUDA 后端 |
| `compute_inference` | `False` | 此封装类应保持关闭；显式推断方法请使用上面的通用估计器 |
| `solver` | `"auto"` | 求解器选择 |
| `gpu_memory_cleanup` | `False` | CuPy 内存池清理 |

## SCAD vs Lasso

| 性质 | Lasso | SCAD |
|---|---|---|
| 凸性 | 凸 | 非凸 |
| Oracle 性质 | 一般不成立 | 依赖正则条件与调参条件 |
| 大 $\beta_j$ 的偏差 | 向零收缩 | 几乎无偏 |
| 优化 | 凸目标；仍需检查数值收敛 | 可能存在多个局部最优 |
| 稀疏性 | 有 | 有（通常更稀疏） |

## 示例

```python
from statgpu.linear_model import SCADRegression

model = SCADRegression(alpha=0.1, a=3.7)
model.fit(X, y)
print(model.coef_)        # 稀疏系数
print(model.score(X, y))  # R-squared
```

## 参考文献

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
- Wang, H., Li, R., & Tsai, C.-L. (2007). Tuning parameter selectors for the smoothly clipped absolute deviation method. *Biometrika*, 94(3), 553-568.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509-1533.
