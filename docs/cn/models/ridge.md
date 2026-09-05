# 岭回归（Ridge）

> 语言: 中文  
> 最后更新: 2026-09-04
> 页面定位: 模型文档  
> 切换: [English](../../en/models/ridge.md)

语言切换: [English](../../en/models/ridge.md)

## 概述

`Ridge` 在普通最小二乘基础上加入 L2 正则化，用于缓解多重共线性、稳定系数估计，并保留与 `LinearRegression` 对齐的推断接口，包括 `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3` 和 `hac` 协方差选项。

## 导入路径

`statgpu.linear_model.Ridge`

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

截距项不受 L2 惩罚。因而将所有样本权重同时乘以任意正数，不会改变拟合结果。

## 估计方程

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

## 求解器支持

`Ridge` 是唯一默认采用模型专属 `solver="exact"` 路径的线性模型 wrapper。若需要可复现的数值路径，应显式指定求解器；`solver="auto"` 会在后端选择完成后分发到不同路径。

| `solver` 值 | CPU | CuPy / Torch | 适用场景 |
|---|:---:|:---:|---|
| `exact`（默认） | 支持 | 支持 | 稠密 L2 normal-equation 求解；普通 Ridge 的首选 |
| `auto` | exact | Newton | 交给后端感知的自动分发 |
| `fista` / `fista_bb` | 支持 | 支持 | 迭代路径对照或受控优化实验 |
| `newton` / `lbfgs` | 支持 | 支持 | 光滑目标的替代路径 |
| `admm` | 支持 | 支持 | 实验性拆分路径；只支持均匀样本权重 |
| `irls` | 不支持 | 不支持 | squared-error loss 未声明 IRLS contract，因此会被拒绝 |

`coordinate_descent`、`quantile_cd_solver` 与 `lbfgs_b_solver` 都不是 Ridge 估计器选项。exact 路径的推导就是上方估计方程；通用算法原理见[求解器指南](../guides/solver-algorithms.md)。

## 协方差与推断

- `cov_type="nonrobust"`：经典 ridge 协方差。
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`：sandwich 风格稳健协方差。
- `cov_type="hac"`：Newey-West Bartlett kernel 协方差，`hac_maxlags` 控制最大滞后阶。
- `compute_inference=True` 时返回 `_bse`、`_tvalues`、`_pvalues`、`_conf_int`。
- 加权推断使用加权设计矩阵 `[sqrt(w), sqrt(w) * X]`，因此截距列、残差、bread 和 meat 与估计阶段采用相同权重约定。

推断中的 ridge normal equations 与拟合阶段使用同一个平均损失 penalty mapping：无权重时数值 ridge 项为 `n * alpha`，加权时为 `sample_weight.sum() * alpha`；截距始终不惩罚。

对于已经迁移的共享 Gaussian 路径，协方差、标准误、检验统计量、参考分布 p 值以及置信区间临界值都在实际执行拟合的 NumPy/CuPy/Torch backend 上完成。全部数值工作结束以后，既有 reporting 数组才 snapshot 到 NumPy。`_inference_result.metadata` 会记录 numerical backend/device 以及 `post_numerical_inference` reporting boundary。显式 CUDA/Torch 拟合若缺失执行 backend provenance，会 fail closed，而不会静默切换到 NumPy inference。

低自由度 Student-t 路径复用维护中的 df=1 与 df=2 稳定恒等式，因此极端但仍可表示的尾概率不会因为 `1-CDF` 消去或 `t**2` 中间量 overflow 被错误压成 0。

## 参数

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

## strict 与 approx 的差异

当前没有单独公开的 approximate 模式。hosted tests 覆盖 exact/FISTA、加权/无权重、formula、协方差/推断、RidgeCV final-refit inference、backend provenance 和 numerical/reporting transfer boundary。真实 CuPy/Torch CUDA acceptance 仍是独立的 exact-source remote gate；只要维护的 validator contract 后续发生变化，就必须重新执行 physical validation。

## 输出

- 系数：`intercept_`、`coef_`
- 推断：`_bse`、`_tvalues`、`_pvalues`、`_conf_int`
- 诊断：`rsquared`、`rsquared_adj`、`fvalue`、`aic`、`bic`
- 方法：`fit`、`predict`、`score`、`summary`

## 常见问题

- `alpha` 如何选择？建议使用 `RidgeCV`，或在 statgpu 的平均损失尺度下使用任务相关的对数网格。
- 为什么相同 `alpha` 与 sklearn 不一致？两者残差项的归一化方式不同，应使用上面的显式映射。
- 将所有样本权重同时缩放会改变模型吗？不会，因为加权损失除以 `sum(sample_weight)`。
- 什么时候设置 `hac_maxlags`？当 `cov_type="hac"` 且存在时间相关时建议显式设置，否则使用默认规则。
- GPU inference 会把公开数组保留为 CuPy/Torch 吗？不会。数值推断保持 backend-native，但既有公开 reporting 属性在数值推断完成后仍然是 NumPy snapshot。

## 外部验证

- 内部一致性通过平均损失闭式解以及通用 penalized-linear estimator 进行验证。
- 与 sklearn 比较时使用无权重或加权情况下的显式 alpha 映射。
- 加权 exact/FISTA、formula 缺失行权重对齐、推断和 RidgeCV 权重整体缩放不变性由 `dev/tests/test_ridge_weighted_consistency.py` 覆盖。
- Issue #127 的 backend-native inference regressions 与 physical CUDA acceptance contract 位于 `dev/tests/test_gaussian_inference_*.py` 和 `dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py`。

## 参考文献

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
