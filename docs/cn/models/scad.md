# SCAD

> 语言：中文  
> 最后更新：2026-10-09<br>
> 页面定位：模型文档  
> 切换：[English](../../en/models/scad.md)

## 概述

`SCADRegression` 提供 SCAD 惩罚（Smoothly Clipped Absolute Deviation）线性回归（Fan & Li, 2001）。SCAD 是非凸惩罚，可以减小对大系数的收缩。其 **oracle 性质**是依赖正则条件和调参条件的渐近结论，不是每次有限样本拟合都成立的保证。

当连续响应需要稀疏预测，且你关心 Lasso 对大斜率的收缩时，可以考虑这个模型。
若优先需要凸目标和更简单的调参过程，可先使用 [Lasso](lasso.md) 或
[Elastic Net](elastic-net.md)。非凸拟合可能得到不同的局部解；变量入选不自动
代表统计显著性或因果效应。

## 路径

`statgpu.linear_model.SCADRegression`

## 目标函数

$$
\min_{b,\beta} \frac{1}{2n}\|y - b\mathbf{1} - X\beta\|_2^2 + \sum_{j=1}^p p_{\lambda,a}(|\beta_j|)
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

这里 n 为观测数，X 含 p 列特征，b 是不受惩罚的截距，`alpha` 对应
$\lambda$；标量惩罚的自变量为 $\theta=|\beta_j|\geq0$。
`fit_intercept=False` 时固定 b=0。样本权重须有限、非负且总和为正；加权时将
数据拟合项改为 $\sum_i w_i(y_i-b-x_i^\top\beta)^2/(2\sum_i w_i)$。
所有权重同乘一个正数不改变该目标。特征不会自动标准化；如需缩放，应仅从
训练行学习变换。

## 完整 CPU 示例

下面的特征已具有相近尺度。拟合前留出最后 40 行；只有前两列特征产生信号。

请在同一个 Python 会话中按顺序运行以下步骤，先导入所需的库。

<!-- learner-example: scad-prediction -->
```python
import numpy as np
from statgpu.linear_model import SCADRegression
```

<a id="cpu-data"></a>

### 准备训练与测试数据

`X` 的形状为 `(160, 5)`：每行是一条观测，每列是一个预测变量。`y` 是形状为 `(160,)` 的一维连续响应。最后 40 行不参与拟合或调参。

```python
rng = np.random.default_rng(64)
X = rng.normal(size=(160, 5))
y = 1.5 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.4, size=160)
X_train, X_test = X[:120], X[120:]
y_train, y_test = y[:120], y[120:]
```

### 拟合预测模型

先用一个示意性的固定惩罚强度。关闭推断，让这一步只估计用于预测的系数。

```python
model = SCADRegression(
    alpha=0.1, a=3.7, device="cpu", compute_inference=False,
    max_iter=5000, tol=1e-8,
)
model.fit(X_train, y_train)
```

### 预测并查看拟合结果

按训练时的列顺序传入测试特征；`score` 用留出的响应计算 R²。

```python
prediction = model.predict(X_test)
print("Slopes:", np.round(model.coef_, 3))
print("Intercept:", round(model.intercept_, 3))
print("Test R2:", round(model.score(X_test, y_test), 3))
```
<!-- example-end: scad-prediction -->

该随机种子下，斜率约为 `[1.987, -0.982, 0, 0, 0]`，截距约为 `1.430`，
留出集 R² 约为 `0.967`；预测形状为 `(40,)`。这些是惩罚预测系数，并非活跃集
重拟合或显著性检验结果。换一个样本仍可能选入噪声或遗漏真实信号。

## 参数选择与结果检查

- 在训练数据内通过验证选择 `alpha`。示例中的 0.1 不是通用最优值；内部延续
  路径只服务于求解，不会自动执行交叉验证。
- `a=3.7` 可作为常用起点。改变凹度会改变惩罚和优化难度；应比较
  预测表现及入选集合的稳定性，不能把更稀疏直接等同于更好。
- 增大 `max_iter`、收紧 `tol` 可以检查数值稳定性。`n_iter_` 本身不证明
  全局最优，局部最优仍可能存在。
- 测试集应与调参分开；每个验证训练折中重新学习预处理。这两个封装类不提供
  内置交叉验证方法。

## 算法

SCAD 使用 **LLA（局部线性近似）** + FISTA：

1. **延续路径**：从 $\lambda_{max}$ 沿几何网格递减
2. **LLA 内循环**：
   - 计算 LLA 权重：$w_j = p'_{\lambda,a}(|\beta_j|)$（SCAD 在当前估计处的次梯度）
   - 求解加权 L1 问题：$\min_{b,\beta} \frac{1}{2n}\|y - b\mathbf{1} - X\beta\|_2^2 + \sum w_j |\beta_j|$
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
若要显式请求高斯 `oracle` 或 `bootstrap`，请使用
`PenalizedLinearRegression(penalty="scad", penalty_kwargs={"a": 3.7}, ...)`，
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
| `a` | `3.7` | 大于 2 的有限凹度参数（常用 3.7） |
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
| 大 $\beta_j$ 的偏差 | 向零收缩 | 阈值以上导数为零；不保证有限样本无偏 |
| 优化 | 凸目标；仍需检查数值收敛 | 可能存在多个局部最优 |
| 稀疏性 | 有 | 有；入选数量取决于数据与调参 |

## 可选 GPU 使用

完成 [CPU 数据准备](#cpu-data)后，在同一构造函数中设置 `device="cuda"` 可请求
CuPy CUDA，设置 `device="torch"` 可请求 Torch CUDA。显式后端不可用时会
报错；只有 `auto` 可以选择其他可用后端。详见[设备与内存](../guides/device-and-memory.md)。
上述 CPU 示例不需要 GPU。

## API 与输出形状

[完整 SCADRegression 方法参考](../reference/linear-model-api.md#scadregression)
包含拟合、公式与权重、预测位置、评分、继承辅助方法及诊断限制。
上表列出了全部构造参数。应从 `statgpu.linear_model` 导入；顶层 `statgpu`
不导出这个类。

## 参考文献

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
- Wang, H., Li, R., & Tsai, C.-L. (2007). Tuning parameter selectors for the smoothly clipped absolute deviation method. *Biometrika*, 94(3), 553-568.
- Zou, H., & Li, R. (2008). One-step sparse estimates in nonconcave penalized likelihood models. *Annals of Statistics*, 36(4), 1509-1533.
