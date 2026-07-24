# SCAD

> 语言：中文  
> 最后更新：2026-06-14  
> 页面定位：模型文档  
> 切换：[English](../../../en/models/scad.md)

语言切换：[English](../../../en/models/scad.md)

## 概述

`SCADRegression` 提供 SCAD 惩罚（Smoothly Clipped Absolute Deviation）线性回归（Fan & Li, 2001）。SCAD 是非凸惩罚，具有 **oracle property**，对大系数产生几乎无偏的估计——解决了 Lasso 的估计偏差问题。

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
2. **LLA 内循环**（每个 $\lambda$ 迭代 1-6 次）：
   - 计算 LLA 权重：$w_j = p'_{\lambda,a}(|\beta_j|)$（SCAD 在当前估计处的次梯度）
   - 求解加权 L1 问题：$\min \frac{1}{2n}\|y - X\beta\|_2^2 + \sum w_j |\beta_j|$
   - 加权 L1 通过 FISTA 求解
3. **热启动**：用前一个 $\lambda$ 的解作为下一个 $\lambda$ 的初始点

## Oracle Property

在正则条件下（Fan & Li 2001, Theorem 2）：
- **选择一致性**：$\Pr(\hat{S} = S_0) \to 1$
- **渐近正态性**：$\sqrt{n}(\hat{\beta}_{\hat{S}} - \beta_{0,S_0}) \xrightarrow{d} N(0, \Sigma_0)$

SCAD 对大系数产生**无偏**估计（不像 Lasso 将所有系数向零收缩）。

## 推断

- `compute_inference=True` + `inference_method='oracle'` 支持 active-set 推断（Fan & Li 2001）。当前仅 CPU；CuPy/Torch 抛出 ``NotImplementedError``。
- `inference_method='bootstrap'` 也可用（仅 CPU）。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 正则化强度（$\lambda$） |
| `a` | `3.7` | 凹度参数（Fan & Li 推荐 3.7） |
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `1000` | 每个 LLA 步骤的最大 FISTA 迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` |
| `solver` | `"auto"` | 求解器选择 |
| `gpu_memory_cleanup` | `False` | CuPy 内存池清理 |

## SCAD vs Lasso

| 性质 | Lasso | SCAD |
|---|---|---|
| 凸性 | 凸 | 非凸 |
| Oracle property | 无 | 有 |
| 大 $\beta_j$ 的偏差 | 向零收缩 | 几乎无偏 |
| 全局最优 | 保证 | 可能存在多个局部最优 |
| 稀疏性 | 有 | 有（通常更稀疏） |

## 示例

```python
from statgpu.linear_model import SCADRegression

model = SCADRegression(alpha=0.1, a=3.7)
model.fit(X, y)
print(model.coef_)        # 稀疏系数（比 Lasso 更无偏）
print(model.score(X, y))  # R-squared
```

## 参考文献

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
- Wang, H., Li, R., & Tsai, C.-L. (2007). Tuning parameter selectors for the smoothly clipped absolute deviation method. *Biometrika*, 94(3), 553-568.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509-1533.
