# Ridge

> 语言: 中文  
> 最后更新: 2026-05-20  
> 页面定位: 模型文档  
> 切换: [English](../en/models/ridge.md)

语言切换: [English](../en/models/ridge.md)

## Overview

`Ridge` 在普通最小二乘基础上加入 L2 正则化，用于缓解多重共线性、稳定系数估计，并保留与 `LinearRegression` 对齐的推断接口，包括 `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3` 和 `hac` 协方差选项。

## Path

`statgpu.linear_model.Ridge`

## Objective Function

估计目标为：

$$
\min_{\beta} \|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2
$$

其中截距项按当前实现单独处理，不作为 L2 惩罚对象。

## Estimating Equation

Ridge 的一阶条件为：

$$
(X^\top X + \alpha I)\hat\beta = X^\top y
$$

当前 `Ridge` 默认使用 `solver="exact"`，即闭式 normal-equation 路径。该路径支持 CPU、CuPy 和 Torch 后端。与 sklearn 做目标函数尺度对齐时，请使用 `sklearn_alpha = n_samples * statgpu_alpha`。

## Covariance/Inference

- `cov_type="nonrobust"`：经典 ridge 协方差。
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`：sandwich 风格稳健协方差。
- `cov_type="hac"`：Newey-West Bartlett kernel 协方差，`hac_maxlags` 控制最大滞后阶。
- `compute_inference=True` 时返回 `_bse`、`_tvalues`、`_pvalues`、`_conf_int`。

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L2 正则化强度 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `n_jobs` | `None` | 并行任务数 |
| `compute_inference` | `True` | 是否计算标准误、t 值、p 值和置信区间 |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时的最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy memory pool |
| `solver` | `"exact"` | thin-wrapper 路径的求解器；默认使用 L2 闭式解 |

## CPU+GPU Examples

```python
from statgpu.linear_model import Ridge

# CPU
m_cpu = Ridge(alpha=1.0, device="cpu", cov_type="hc3", compute_inference=True)
m_cpu.fit(X, y)

# GPU
m_gpu = Ridge(
    alpha=1.0,
    device="cuda",
    cov_type="hc3",
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y)
```

## strict/approx difference

当前没有单独公开的 approx 模式。默认路径是经过验证的发布路径；CPU/GPU 差异预期主要来自浮点线性代数实现差异。

## Outputs

- 系数：`intercept_`、`coef_`
- 推断：`_bse`、`_tvalues`、`_pvalues`、`_conf_int`
- 诊断：`r_squared`、`adj_r_squared`、`f_statistic`、`aic`、`bic`
- 方法：`fit`、`predict`、`score`、`summary`

## FAQ

- `alpha` 如何选择？建议先使用对数网格交叉验证，例如 `1e-4` 到 `1e2`，再固定任务参数。
- 什么时候设置 `hac_maxlags`？当 `cov_type="hac"` 且存在时间相关时建议显式设置，否则使用默认规则。
- 为什么和 sklearn 的 `alpha` 不能直接同名比较？statgpu 的 penalized 目标使用平均 loss 尺度；与 sklearn 对比 Ridge 时应使用 `sklearn_alpha = n_samples * statgpu_alpha`。

## External Validation

- Ridge inference/covariance 行为复用与 `LinearRegression` 对齐的稳健协方差实现。
- 相关一致性检查维护在 `dev/tests/test_external_consistency.py`。
- penalized Gaussian 路径的外部对比由远程 `myconda` accuracy/benchmark 脚本覆盖。

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
