# 泊松回归（PoissonRegression）

> 语言: 中文  
> 最后更新: 2026-05-20  
> 页面定位: 模型文档  
> 切换: [English](../../en/models/poisson-regression.md)

语言切换: [English](../../en/models/poisson-regression.md)

## 概述

`PoissonRegression` 是用于计数数据的普通 Poisson GLM 入口，内部走共享的 `GeneralizedLinearModel` 架构。它表示非惩罚的 Poisson 回归；如果需要 L1、L2、ElasticNet、group 或 adaptive penalty，请使用 `PenalizedPoissonRegression`。

支持 M-estimation sandwich inference：通过 ``compute_inference=True`` 获取标准误、z 统计量、p 值和 95% 置信区间。模型协方差使用 expected Fisher information（``cov_type='nonrobust'``，与 ``statsmodels.GLM`` 对齐），稳健 sandwich 使用 observed Hessian（``cov_type='hc0'``、``'hc1'``）。当前仅 CPU；GPU 显式 raise ``NotImplementedError``。

## 导入路径

`statgpu.linear_model.PoissonRegression`

也支持顶层导入：

```python
from statgpu import PoissonRegression
```

## 目标函数

在 log link 下：

$$
\mu_i = \exp(x_i^\top\beta)
$$

模型最小化平均 Poisson negative log-likelihood，忽略与参数无关的常数项：

$$
\min_\beta \frac{1}{n}\sum_i \left[\mu_i - y_i \log(\mu_i)\right]
$$

当 `C` 为有限值时，继承的 GLM IRLS 路径可以通过内部机制加入 L2 风格 ridge 项。

## 估计方程

非惩罚 Poisson GLM 的 score equation 为：

$$
\sum_i x_i(y_i - \mu_i)=0
$$

`PoissonRegression` 默认使用 `solver="auto"`，当前会调度到 IRLS。smooth Poisson GLM 也支持显式 `solver="newton"` 和 `solver="lbfgs"`，并运行在用户选择的后端。v23c 起，`solver="lbfgs"` 正确支持 L2 惩罚。该模型继承 GLM formula 接口，因此 formula 中的截距语义遵循 patsy/R 习惯。

## 求解器支持

| `solver` 值 | CPU / CuPy / Torch | 惩罚与权重语义 |
|---|:---:|---|
| `auto`（默认） | 支持 | 分发到 IRLS |
| `irls` | 支持 | `C` 控制继承的 L2 项；支持解析样本权重 |
| `fista` | 支持 | 零惩罚的近端梯度对照路径；支持解析样本权重 |
| `newton` | 支持 | 光滑无惩罚路径；只支持均匀样本权重 |
| `lbfgs` | 支持 | 光滑无惩罚路径；只支持均匀样本权重 |

`exact`、`admm` 与低层 L-BFGS-B 不是 `PoissonRegression` 的取值。需要 L1、Elastic Net、SCAD 或 MCP 时，应使用 `PenalizedPoissonRegression` 并核对 [GLM 求解器表](generalized-linear-model.md#求解器支持)。

## 协方差与推断

设置 ``compute_inference=True`` 即可获得推断结果：

```python
from statgpu import PoissonRegression
import numpy as np

m = PoissonRegression(solver='newton', compute_inference=True, cov_type='nonrobust')
m.fit(X, y)
print(m._bse)       # 标准误
print(m._pvalues)   # 双尾 p 值（正态）
print(m._conf_int)  # 95% 置信区间
```

**协方差类型**：``'nonrobust'``（默认，expected Fisher，与 statsmodels 对齐）、``'hc0'``、``'hc1'``（sandwich）。HC2/HC3/HAC 暂不支持（raise ``NotImplementedError``）。

**严格推断**：nonrobust Poisson 与 statsmodels 对齐到机器精度（n≥200 时 \|bse diff\| < 1e-9）。GPU 暂不支持（显式报错，不静默回退）。

## 参数

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

## CPU 与 GPU 示例

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

## strict 与 approx 的差异

`PoissonRegression` 当前没有公开 strict/approx inference 开关。发布验证重点是不同后端和外部框架之间的系数、预测、目标函数与 runtime 一致性。

## 输出

- 系数：`intercept_`、`coef_`
- 迭代次数：`n_iter_`
- 方法：`fit`、`predict`
- 使用 `formula` 和 `data` 拟合时，会在内部保存 design metadata 供预测转换使用

`predict` 返回 inverse-link 后的均值响应；对 Poisson 来说，即估计的计数/发生率 \(\hat\mu\)，不是 linear predictor。

## 常见问题

- 什么时候使用 `PoissonRegression` 而不是 `GeneralizedLinearModel(family="poisson")`？当你希望 public API 更明确时使用 `PoissonRegression`；二者共享 GLM 实现。
- 什么时候使用 `PenalizedPoissonRegression`？需要 L1、L2、ElasticNet、group 或 adaptive penalty 时使用。
- `PoissonRegression` 提供标准误和 p 值吗？提供——设置 ``compute_inference=True``。支持 model-based（``cov_type='nonrobust'``）和 sandwich（``hc0``、``hc1``）协方差。与 statsmodels 对齐到 \|bse diff\| < 1e-9。
- `device="cuda"` 是否一定使用 GPU？对已支持的 Poisson GLM solver 路径，是的：核心计算保持在 CuPy，否则清晰报错。`device="torch"` 同样要求 Torch CUDA。

## 外部验证

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

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
- scikit-learn PoissonRegressor documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
