# PoissonRegression

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：模型文档  
> 切换：[English](../../en/models/poisson-regression.md)

## 概述

`PoissonRegression` 是 statgpu 面向计数数据提供的普通 Poisson 广义线性模型入口，底层复用 `GeneralizedLinearModel` 的通用 GLM 基础设施。它表示非惩罚的 Poisson 回归；需要 L1、L2、ElasticNet、分组或自适应惩罚时，应使用 `PenalizedPoissonRegression`。

模型支持 NumPy、CuPy 与 Torch 三种数值后端。设置 `compute_inference=True` 后，可以计算标准误、z 统计量、p 值和置信区间；显式 GPU 请求若对应后端不可用会直接报错，不会静默改为 CPU。

导入路径为：

```python
from statgpu.linear_model import PoissonRegression
# 或
from statgpu import PoissonRegression
```

## 统计模型与目标函数

在对数链接下，条件均值为

$$
\mu_i = \exp(x_i^\top\beta).
$$

忽略与参数无关的常数项后，模型最小化平均 Poisson 负对数似然：

$$
\min_\beta
\frac{1}{n}\sum_i
\left[\mu_i-y_i\log(\mu_i)\right].
$$

当继承的 `C` 参数对应有限正则化强度时，共享 IRLS 路径可以加入 L2 风格的 Ridge 项。需要更一般的惩罚结构时，应使用专门的惩罚 Poisson 模型。

非惩罚 Poisson GLM 的得分方程为

$$
\sum_i x_i(y_i-\mu_i)=0.
$$

`solver="auto"` 当前选择 IRLS。对光滑的 Poisson GLM 目标，也可以显式使用 `solver="newton"` 或 `solver="lbfgs"`；这些求解器在受支持组合下运行于所选数值后端。

## 协方差与统计推断

设置 `compute_inference=True` 即可获得拟合后推断：

```python
from statgpu import PoissonRegression

model = PoissonRegression(
    solver="newton",
    compute_inference=True,
    cov_type="nonrobust",
)
model.fit(X, y)

print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

当前协方差选项包括：

- `cov_type="nonrobust"`：基于期望 Fisher 信息的模型协方差；
- `cov_type="hc0"`：基于观测 Hessian 的夹心协方差；
- `cov_type="hc1"`：在 HC0 基础上加入自由度修正；
- `hc2`、`hc3` 与 `hac`：当前 Poisson 路径不支持，显式请求会报错。

Poisson 推断使用渐近正态参考分布，因此报告 z 统计量和双侧 p 值。Poisson 离散参数固定为 1；相关拟合信息可通过模型元数据查看。

在 CuPy 或 Torch CUDA 上成功拟合后，受支持的推断计算继续使用同一数值后端和具体设备；不会为了计算协方差或参考分布而静默切换到 CPU。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `max_iter` | `100` | 最大迭代次数 |
| `tol` | `1e-4` | 收敛容差 |
| `C` | `1.0` | 继承的 GLM IRLS 路径使用的逆正则化强度 |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `solver` | `"auto"` | `auto` / `irls` / `fista` / `newton` / `lbfgs`；实际可用性取决于完整模型路径 |
| `n_jobs` | `None` | 适用路径的并行任务数 |
| `gpu_memory_cleanup` | `False` | 拟合后尽可能释放可回收的 GPU 缓存 |
| `formula` | `None` | `fit` 中可选的 patsy 风格公式 |
| `data` | `None` | 公式模式使用的数据表 |

求解器组合的完整支持范围见 [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md)。

## 使用示例

### CPU

```python
from statgpu.linear_model import PoissonRegression

model = PoissonRegression(
    device="cpu",
    max_iter=100,
    tol=1e-6,
)
model.fit(X, y_count)
mu = model.predict(X)
```

### CuPy CUDA

```python
model = PoissonRegression(
    device="cuda",
    max_iter=100,
    tol=1e-6,
)
model.fit(X_gpu, y_count_gpu)
mu_gpu = model.predict(X_gpu)
```

### 公式接口

```python
model = PoissonRegression()
model.fit(
    formula="count ~ exposure + x1 + C(group)",
    data=df,
)
pred = model.predict(df_new)
```

公式解析属于 CPU 侧的数据准备步骤；对于大规模 GPU 工作负载，直接传入已经准备好的 `X, y` 数组通常可以减少额外的数据转换。

## 输出与解释

常用结果包括：

- `intercept_`、`coef_`：拟合参数；
- `n_iter_`：迭代次数；
- `_bse`、`_zvalues`、`_pvalues`、`_conf_int`：启用推断时的统计结果；
- `fit()`、`predict()`：拟合与预测接口。

`predict()` 返回逆链接后的条件均值。对 Poisson 模型而言，这是估计的计数或发生率均值 $\widehat\mu$，而不是线性预测子 $X\widehat\beta$。

## 常见问题

**什么时候使用 `PoissonRegression`，什么时候直接使用通用 GLM？**  
当希望接口明确表达“这是 Poisson 回归”时使用 `PoissonRegression`；它与通用 GLM 的 Poisson 配置共享核心实现。

**什么时候使用 `PenalizedPoissonRegression`？**  
需要 L1、L2、ElasticNet、分组、自适应或其他惩罚路径时使用。

**是否支持 GPU 上的统计推断？**  
支持的 Poisson 推断路径可以在 CuPy 或 Torch CUDA 上执行。显式 GPU 请求只有在对应路径和后端可用时才成功，否则直接报错。

**为什么不支持 HC2、HC3 或 HAC？**  
这些协方差形式目前没有在 Poisson 模型的公开推断路径中实现；应使用已支持的 `nonrobust`、`hc0` 或 `hc1`，或者根据研究问题选择其他模型/推断方案。

## 相关文档

- [广义线性模型](generalized-linear-model.md)
- [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md)
- [推断模式](../guides/inference-modes.md)
- [设备与 GPU 内存](../guides/device-and-memory.md)

## 参考文献

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.
