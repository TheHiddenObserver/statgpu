# Ridge

> 语言: 中文  
> 最后更新: 2026-07-12  
> 页面定位: 模型文档  
> 切换: [English](../en/models/ridge.md)

语言切换: [English](../en/models/ridge.md)

## Overview

`Ridge` 在普通最小二乘基础上加入 L2 正则化，用于缓解多重共线性、稳定系数估计，并保留与 `LinearRegression` 对齐的推断接口，包括 `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3` 和 `hac` 协方差选项。

## Path

`statgpu.linear_model.Ridge`

## Objective Function

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

截距项不受 L2 惩罚。因而将所有样本权重同时乘以任意正数，不会改变拟合结果。

## Estimating Equation

使用普通均值或加权均值对数据中心化后，一阶条件为：

$$
\left(X_c^\top W X_c + \alpha\,s_w I\right)\hat\beta
= X_c^\top W y_c,
$$

其中，无权重时 $W=I$、$s_w=n$；加权时 $W=\operatorname{diag}(w)$、$s_w=\sum_iw_i$。

`Ridge` 默认使用 `solver="exact"`。exact 与 FISTA 路径、`PenalizedLinearRegression(loss="squared_error", penalty="l2")` 以及 `RidgeCV` 均使用同一个平均损失尺度。

scikit-learn 使用未归一化的残差平方和。比较系数时应使用：

- 无权重：`sklearn_alpha = n_samples * statgpu_alpha`；
- 加权：`sklearn_alpha = sample_weight.sum() * statgpu_alpha`。

直接使用相同数值的 `alpha`，实际比较的是两个不同目标函数。

## Covariance/Inference

- `cov_type="nonrobust"`：经典 ridge 协方差。
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`：sandwich 风格稳健协方差。
- `cov_type="hac"`：Newey-West Bartlett kernel 协方差，`hac_maxlags` 控制最大滞后阶。
- `compute_inference=True` 时返回 `_bse`、`_tvalues`、`_pvalues`、`_conf_int`。
- 加权推断使用加权设计矩阵 `[sqrt(w), sqrt(w) * X]`，因此截距列、残差、bread 和 meat 与估计阶段采用相同权重约定。

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | 平均损失尺度下的 L2 正则化强度 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | 并行任务数 |
| `compute_inference` | `True` | 是否计算标准误、t 值、p 值和置信区间 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时的最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后尽可能释放 GPU 内存 |
| `solver` | `"exact"` | 默认使用 L2 闭式解；`fista` 使用相同目标函数 |

## CPU+GPU Examples

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

## strict/approx difference

当前没有单独公开的 approximate 模式。CPU 测试覆盖 exact/FISTA、加权/无权重、formula、推断和 RidgeCV 契约；真实 CuPy/Torch CUDA 数值与性能验证仍属于远程验证门禁。

## Outputs

- 系数：`intercept_`、`coef_`
- 推断：`_bse`、`_tvalues`、`_pvalues`、`_conf_int`
- 诊断：`rsquared`、`rsquared_adj`、`fvalue`、`aic`、`bic`
- 方法：`fit`、`predict`、`score`、`summary`

## FAQ

- `alpha` 如何选择？建议使用 `RidgeCV`，或在 statgpu 的平均损失尺度下使用任务相关的对数网格。
- 为什么相同 `alpha` 与 sklearn 不一致？两者残差项的归一化方式不同，应使用上面的显式映射。
- 将所有样本权重同时缩放会改变模型吗？不会，因为加权损失除以 `sum(sample_weight)`。
- 什么时候设置 `hac_maxlags`？当 `cov_type="hac"` 且存在时间相关时建议显式设置，否则使用默认规则。

## External Validation

- 内部一致性通过平均损失闭式解以及通用 penalized-linear estimator 进行验证。
- 与 sklearn 比较时使用无权重或加权情况下的显式 alpha 映射。
- 加权 exact/FISTA、formula 缺失行权重对齐、推断和 RidgeCV 权重整体缩放不变性由 `dev/tests/test_ridge_weighted_consistency.py` 覆盖。

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
