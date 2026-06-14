# PoissonRegression

> 语言: 中文  
> 最后更新: 2026-05-20  
> 页面定位: 模型文档  
> 切换: [English](../en/models/poisson-regression.md)

语言切换: [English](../en/models/poisson-regression.md)

## Overview

`PoissonRegression` 是用于计数数据的普通 Poisson GLM 入口，内部走共享的 `GeneralizedLinearModel` 架构。它表示非惩罚的 Poisson 回归；如果需要 L1、L2、ElasticNet、group 或 adaptive penalty，请使用 `PenalizedPoissonRegression`。

当前范围是估计和预测。robust standard error、confidence interval 等完整 inference 字段尚未在该 GLM wrapper 中公开。

## Path

`statgpu.linear_model.PoissonRegression`

也支持顶层导入：

```python
from statgpu import PoissonRegression
```

## Objective Function

在 log link 下：

$$
\mu_i = \exp(x_i^\top\beta)
$$

模型最小化平均 Poisson negative log-likelihood，忽略与参数无关的常数项：

$$
\min_\beta \frac{1}{n}\sum_i \left[\mu_i - y_i \log(\mu_i)\right]
$$

当 `C` 为有限值时，继承的 GLM IRLS 路径可以通过内部机制加入 L2 风格 ridge 项。

## Estimating Equation

非惩罚 Poisson GLM 的 score equation 为：

$$
\sum_i x_i(y_i - \mu_i)=0
$$

`PoissonRegression` 默认使用 `solver="auto"`，当前会调度到 IRLS。smooth Poisson GLM 也支持显式 `solver="newton"` 和 `solver="lbfgs"`，并运行在用户选择的后端。v23c 起，`solver="lbfgs"` 正确支持 L2 惩罚。该模型继承 GLM formula 接口，因此 formula 中的截距语义遵循 patsy/R 习惯。

## Covariance/Inference

该 wrapper 当前没有暴露 `LinearRegression`、`Ridge` 或 `LogisticRegression` 那类 inference-rich 接口。也就是说，本页不承诺公开 `cov_type`、robust covariance、`_bse`、`_zvalues`、`_pvalues` 或 `_conf_int`。

本轮应通过系数、预测、目标函数和外部框架对比验证 Poisson estimation。Poisson GLM 的 strict inference 应留到后续 inference 专项里补齐。

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `100` | 最大 IRLS 迭代次数 |
| `tol` | `1e-4` | 收敛阈值 |
| `C` | `1.0` | 继承 GLM IRLS 路径使用的 inverse regularization strength |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `solver` | `"auto"` | `auto` / `irls` / `fista` / `newton` / `lbfgs` |
| `n_jobs` | `None` | 并行任务数 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy memory pool |
| `formula` | `None` | `fit` 中可选的 patsy 风格公式 |
| `data` | `None` | formula 模式下的数据表 |

## CPU+GPU Examples

```python
from statgpu.linear_model import PoissonRegression

# CPU count model
m_cpu = PoissonRegression(device="cpu", max_iter=100, tol=1e-6)
m_cpu.fit(X, y_count)
mu_cpu = m_cpu.predict(X)

# CUDA backend 可用时使用 GPU
m_gpu = PoissonRegression(device="cuda", max_iter=100, tol=1e-6)
m_gpu.fit(X_gpu, y_count_gpu)
mu_gpu = m_gpu.predict(X_gpu)
```

Formula 用法：

```python
from statgpu.linear_model import PoissonRegression

model = PoissonRegression()
model.fit(formula="count ~ exposure + x1 + C(group)", data=df)
pred = model.predict(df_new)
```

大规模 GPU 任务建议直接传入显式 `X, y` 数组，因为 formula 解析是 CPU 侧便利层。

## strict/approx difference

`PoissonRegression` 当前没有公开 strict/approx inference 开关。发布验证重点是不同后端和外部框架之间的系数、预测、目标函数与 runtime 一致性。

## Outputs

- 系数：`intercept_`、`coef_`
- 迭代次数：`n_iter_`
- 方法：`fit`、`predict`
- 使用 `formula` 和 `data` 拟合时，会在内部保存 design metadata 供预测转换使用

`predict` 返回 inverse-link 后的均值响应；对 Poisson 来说，即估计的计数/发生率 \(\hat\mu\)，不是 linear predictor。

## FAQ

- 什么时候使用 `PoissonRegression` 而不是 `GeneralizedLinearModel(family="poisson")`？当你希望 public API 更明确时使用 `PoissonRegression`；二者共享 GLM 实现。
- 什么时候使用 `PenalizedPoissonRegression`？需要 L1、L2、ElasticNet、group 或 adaptive penalty 时使用。
- `PoissonRegression` 当前提供 robust standard error 吗？不提供，这部分留到后续 inference 里程碑。
- `device="cuda"` 是否一定使用 GPU？对已支持的 Poisson GLM solver 路径，是的：核心计算保持在 CuPy，否则清晰报错。`device="torch"` 同样要求 Torch CUDA。

## External Validation

Poisson GLM 验证应包括：

- CPU/GPU 系数与预测一致性。
- L2 对齐设置下与 sklearn `PoissonRegressor` 对比。
- 普通 GLM estimation 与 statsmodels GLM Poisson 对比。
- 远程 CUDA 环境下包含 warm-up 与 GPU synchronization 的 runtime benchmark。

当前远程 GLM 验证入口：

```bash
python dev/tests/run_remote_v10_accuracy.py
python dev/benchmarks/run_remote_v10_benchmark.py
```

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- scikit-learn PoissonRegressor documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
