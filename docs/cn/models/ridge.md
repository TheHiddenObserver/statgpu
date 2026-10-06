# Ridge

> 语言：中文  
> 最后更新：2026-10-06  
> 页面定位：模型文档  
> 切换：[English](../../en/models/ridge.md)

## 概览

`Ridge` 在普通最小二乘基础上加入 L2 正则化，用于缓解多重共线性、稳定系数估计，并保留与 `LinearRegression` 对齐的推断接口，包括 `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3` 和 `hac` 协方差选项。

正 alpha 下的区间围绕惩罚拟合计算，不会自动消除收缩偏差，也不校正选择 alpha 的不确定性。

公开路径：`statgpu.linear_model.Ridge`

## 目标函数

对于无权重样本，statgpu 使用平均损失目标：

$$
\min_{b,\beta}
\frac{1}{2n}\sum_{i=1}^n
\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\|\beta\|_2^2.
$$

当传入 `sample_weight=w` 时，数据拟合项按照总权重归一化：

$$
\min_{b,\beta}
\frac{1}{2\sum_i w_i}\sum_{i=1}^n
w_i\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\|\beta\|_2^2.
$$

截距项不受 L2 惩罚。因此，把所有样本权重同时乘以任意正常数，不会改变拟合结果。

## 估计方程

使用普通均值或加权均值对数据中心化后，一阶条件为：

$$
\left(X_c^\top W X_c + \alpha\,s_w I\right)\hat\beta
= X_c^\top W y_c,
$$

其中，无权重时 $W=I$、$s_w=n$；加权时 $W=\operatorname{diag}(w)$、$s_w=\sum_iw_i$。

`Ridge` 默认使用 `solver="exact"`。闭式解与 FISTA 路径、`PenalizedLinearRegression(penalty="l2")` 以及 `RidgeCV` 都使用同一个平均损失尺度。

scikit-learn 使用未归一化的残差平方和。比较系数时应使用：

- 无权重：`sklearn_alpha = n_samples * statgpu_alpha`；
- 加权：`sklearn_alpha = sample_weight.sum() * statgpu_alpha`。

如果直接使用相同数值的 `alpha`，实际比较的是两个不同的目标函数。

<a id="large-feature-offsets"></a>

## 特征均值远大于变化幅度时

带截距的 CPU `solver="exact"` 优化路径通过大数原始矩相减计算中心化交叉乘积。
当特征均值远大于变化幅度时，数值消去可能产生严重错误的系数与预测，即使
结果有限且拟合已返回。加权与无权重拟合均受影响；开启推断不会修复该问题。
例如，将普通预测变量整体平移 `1e8` 后，原本约为 0.91 的正斜率可能变为约 −0.40。

在拟合前减去仅由训练行确定的原点，并在每次预测时使用同一个原点；不要单独
中心化测试集。下面保留拟合截距，因此该平移不改变 Ridge 的统计目标：

<!-- learner-example: ridge-training-origin -->
```python
import numpy as np
from statgpu.linear_model import Ridge

rng = np.random.default_rng(113)
variation = rng.normal(size=(80, 3))
X = variation + 1e8
y = 0.4 + variation @ np.array([1.0, -0.5, 0.3]) + rng.normal(scale=0.1, size=80)
X_train, X_test = X[:60], X[60:]
y_train, y_test = y[:60], y[60:]
origin = X_train.mean(axis=0)
model = Ridge(alpha=0.1, device="cpu", compute_inference=True).fit(
    X_train - origin, y_train,
)
prediction = model.predict(X_test - origin)
original_intercept = model.intercept_ - origin @ model.coef_
print(np.round(model.coef_, 3))
print(round(float(np.mean((prediction - y_test)**2)), 3))
```

系数约为 `[0.919, -0.430, 0.256]`，留出集 MSE 约为 `0.033`。
`original_intercept` 将方程映射回原特征坐标；实际预测仍应通过中心化后的模型
计算，以避免大数相减。有训练权重时，可使用加权训练均值作为原点，权重和
alpha 保持不变。此时截距推断描述该原点处的响应，其区间不能直接当作
`original_intercept` 的区间。未预先中心化的 FISTA 拟合也避免了这项原始矩系数
计算，但仍需检查收敛，并不保证所有大偏移数值计算都安全。

## 协方差与推断

- `cov_type="nonrobust"`：经典 Ridge 协方差；
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`：sandwich 形式的稳健协方差；
- `cov_type="hac"`：Newey–West Bartlett 核协方差，`hac_maxlags` 控制最大滞后阶；
- `compute_inference=True` 时返回 `_bse`、`_tvalues`、`_pvalues`、`_conf_int`；
- 加权推断使用加权设计矩阵 `[sqrt(w), sqrt(w) * X]`，因此截距列、残差以及协方差的 bread/meat 与估计阶段采用同一权重约定。

推断中的 Ridge 正规方程与拟合阶段使用同一个平均损失惩罚尺度：无权重时数值 Ridge 项为 `n * alpha`，加权时为 `sample_weight.sum() * alpha`；截距始终不受惩罚。

对于共享的 Gaussian 路径，协方差、标准误、检验统计量、参考分布 p 值与置信区间临界值都在实际执行拟合的 NumPy/CuPy/Torch 后端上完成。数值推断结束后，小型结果数组才转换为 NumPy 用于统一展示。显式 CUDA/Torch 请求如果缺少对应后端执行条件会报错，而不会静默切换成 NumPy 推断。

非稳健区间使用 Student-t 参考分布；HC/HAC 区间使用正态参考分布，尽管属性名仍为 `_tvalues`。这些代入式区间不保证在收缩或同一数据调参之后仍覆盖无惩罚的总体系数。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | 平均损失尺度下的 L2 正则化强度 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | 并行任务数 |
| `compute_inference` | `True` | 是否计算标准误、t 值、p 值和置信区间 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时的最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后是否请求释放可回收的 GPU 缓存内存 |
| `solver` | `"exact"` | 默认使用 L2 闭式解；`fista` 使用同一目标函数 |
| `max_iter` | `1000` | 迭代求解器的迭代预算；闭式路径只需求解一次 |
| `tol` | `1e-4` | 迭代收敛容差 |
| `cpu_solver` | `"fista"` | 已弃用的兼容参数；请用 `solver` 选择算法 |
| `lipschitz_L` | `None` | 适用迭代求解器的可选光滑梯度 Lipschitz 上界 |

## CPU 与 GPU 示例

```python
from statgpu.linear_model import Ridge

# CPU
m_cpu = Ridge(alpha=1.0, device="cpu", cov_type="hc3", compute_inference=True)
m_cpu.fit(X, y, sample_weight=w)

# CuPy CUDA
m_gpu = Ridge(
    alpha=1.0,
    device="cuda",
    cov_type="hc3",
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y, sample_weight=w)
```

## 精确与近似计算

当前 `Ridge` 没有单独公开的近似拟合模式。闭式解与 FISTA 都优化同一个公开目标函数；两者的选择属于数值算法差异，而不是统计模型定义的差异。

如果应用需要固定执行后端，应显式设置 `device`。关于后端选择和 GPU 内存行为，见 [设备与 GPU 内存](../guides/device-and-memory.md)。

## 输出

- 系数：`intercept_`、`coef_`
- 推断：`_bse`、`_tvalues`、`_pvalues`、`_conf_int`
- 诊断：`rsquared`、`rsquared_adj`、`fvalue`、`aic`、`bic`
- 方法：`fit`、`predict`、`score`、`summary`

## 常见问题

- **`alpha` 如何选择？** 可以使用 `RidgeCV`，或在 statgpu 的平均损失尺度下根据任务设置对数网格。
- **为什么相同 `alpha` 与 sklearn 不一致？** 两者残差项的归一化方式不同，应使用上面的显式映射。
- **把所有样本权重同时缩放会改变模型吗？** 不会，因为加权损失按 `sum(sample_weight)` 归一化。
- **什么时候设置 `hac_maxlags`？** 当 `cov_type="hac"` 且存在时间相关时可以显式设置；否则使用默认规则。
- **GPU 推断会把公开数组保留为 CuPy/Torch 吗？** 数值推断在对应后端上完成，但统一的公开结果数组在数值推断结束后转换为 NumPy。

## 相关文档

- [交叉验证](../guides/cross-validation.md) — `RidgeCV` 的选择与最终重拟合语义
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备行为
- [推断模式](../guides/inference-modes.md) — 系数推断方法的解释
- [求解器算法](../guides/solver-algorithms.md) — exact/FISTA 等数值算法

## 参考文献

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
