# Distribution API 使用指南

> 语言: 中文
> 最后更新: 2026-04-24
> 页面定位: 指南文档
> 切换: [English](../../en/guides/distribution-api.md)

语言切换：[English](../../en/guides/distribution-api.md)

本页说明当前 distribution API 的推荐调用方式。

## 1) 推荐入口（对象式 API）

建议从 `statgpu.inference` 导入分布对象，使用 `cdf/sf/ppf/isf/pdf/pmf/rvs` 等方法。

```python
import cupy as cp
from statgpu.inference import norm, t, chi2, gamma, beta, f, poisson, binom

x = cp.array([0.0, 1.0, 2.0], dtype=cp.float64)
q = cp.array([0.1, 0.5, 0.9], dtype=cp.float64)

# 连续分布
norm_cdf = norm.cdf(x)
norm_ppf = norm.ppf(q)
t_sf = t.sf(x, df=10)
chi2_ppf = chi2.ppf(q, df=8)

# 离散分布
poi_cdf = poisson.cdf(k=cp.array([1, 2, 3]), mu=3.5)
binom_ppf = binom.ppf(q, n=20, p=0.2)
```

已原生实现的分布名（可直接对象调用）：

- `norm`
- `t`
- `uniform`
- `expon`
- `cauchy`
- `laplace`
- `logistic`
- `chi2`
- `gamma`
- `beta`
- `f`
- `weibull_min`
- `lognorm`
- `poisson`
- `binom`

## 2) 后端选择

模块级代理对象（`norm`、`t`、`chi2` 等）自动选择最佳可用后端（GPU > Torch > NumPy）。可按次覆盖后端：

```python
from statgpu.inference import norm, t

# 自动检测（默认）— 优先使用 CuPy，然后 Torch，最后 NumPy
result = norm.cdf(x)

# 强制 Torch 后端
result = norm.cdf(x, backend="torch")

# 强制 NumPy/scipy 后端
result = norm.cdf(x, backend="numpy")

# 所有分布和方法都支持逐次覆盖
result = t.ppf(q, df=10, backend="cupy")
```

也可以创建固定后端的分布对象：

```python
from statgpu.inference import get_distribution

# 显式指定后端
norm_cupy = get_distribution("norm", backend="cupy")
norm_torch = get_distribution("norm", backend="torch")
norm_numpy = get_distribution("norm", backend="numpy")

# Torch 特有：控制设备
norm_torch_gpu = get_distribution("norm", backend="torch", device="cuda:0")

# 禁用 LUT 缓存（全精度迭代求解）
norm_full_precision = get_distribution("norm", backend="numpy", use_lut=False)
```

## 3) 按名称动态获取分布

如需按字符串动态选择分布，可使用 `get_distribution(name, backend=...)`。

```python
from statgpu.inference import get_distribution

dist = get_distribution("norm", backend="auto")
y = dist.cdf(0.0)
```

`backend` 参数可选：`"auto"`（默认）、`"numpy"`、`"cupy"`、`"torch"`。

## 4) 显式启用 SciPy fallback（可选）

对于尚未原生实现的长尾分布，可以显式开启 fallback：

```python
import numpy as np
from statgpu.inference import get_distribution

dist = get_distribution("gumbel_r", backend="numpy")
out = dist.cdf(np.array([0.0, 1.0, 2.0]))
```

说明：

- 对于非原生分布名，`get_distribution` 配合 `backend="numpy"` 会包装对应的 `scipy.stats` 分布。
- GPU 后端仅支持原生实现的分布。

## 5) 逆函数的 LUT 加速

对于 `t`、`f`、`beta`、`chi2`、`gamma` 的逆 CDF 方法（`ppf`/`isf`），默认使用 LUT（查找表）+ 1 步 Newton 修正。这提供了 10-50 倍加速，精度损失可忽略（~1e-11）。

```python
# 默认：启用 LUT（快速）
t.ppf(q, df=10)

# 禁用 LUT 获取全精度（较慢）
t.ppf(q, df=10, use_lut=False)

# 或在创建分布对象时全局禁用
t_full = get_distribution("t", backend="torch", use_lut=False)
t_full.ppf(q, df=10)  # 始终使用全迭代求解器
```

**精度权衡**:

| 后端 | `use_lut=True` | `use_lut=False` |
|---|---|---|
| numpy | LUT + 1 Newton (误差 ~1e-11) | scipy.special (全迭代) |
| cupy | 原生 `cupyx.scipy.special` | 相同（LUT 不影响） |
| torch | LUT + 1 Newton (误差 ~1e-5 for t/f) | Newton + 64K Chebyshev 积分 |

## 6) R 语言兼容风格接口（兼容保留）

以下接口与 R 的命名风格对齐，按分布家族分组如下。

### 正态分布（norm）

- `dnorm_gpu` -> `norm.pdf`
- `rnorm_gpu` -> `norm.rvs`
- `pnorm_gpu` -> `norm.cdf`
- `qnorm_gpu` -> `norm.ppf`

### t 分布（t）

- `dt_gpu` -> `t.pdf`
- `rt_gpu` -> `t.rvs`
- `pt_gpu` -> `t.cdf`
- `qt_gpu` -> `t.ppf`

### 卡方分布（chi2）

- `dchisq_gpu` -> `chi2.pdf`
- `pchisq_gpu` -> `chi2.cdf`
- `qchisq_gpu` -> `chi2.ppf`
- `rchisq_gpu` -> `chi2.rvs`

### Gamma 分布（gamma）

- `dgamma_gpu` -> `gamma.pdf`
- `pgamma_gpu` -> `gamma.cdf`
- `qgamma_gpu` -> `gamma.ppf`
- `rgamma_gpu` -> `gamma.rvs`

### Beta 分布（beta）

- `dbeta_gpu` -> `beta.pdf`
- `pbeta_gpu` -> `beta.cdf`
- `qbeta_gpu` -> `beta.ppf`
- `rbeta_gpu` -> `beta.rvs`

### F 分布（f）

- `df_gpu` -> `f.pdf`
- `pf_gpu` -> `f.cdf`
- `qf_gpu` -> `f.ppf`
- `rf_gpu` -> `f.rvs`

### Poisson 分布（poisson）

- `dpois_gpu` -> `poisson.pmf`
- `ppois_gpu` -> `poisson.cdf`
- `qpois_gpu` -> `poisson.ppf`
- `rpois_gpu` -> `poisson.rvs`

### 二项分布（binom）

- `dbinom_gpu` -> `binom.pmf`
- `pbinom_gpu` -> `binom.cdf`
- `qbinom_gpu` -> `binom.ppf`
- `rbinom_gpu` -> `binom.rvs`

调用示例：

```python
from statgpu.inference import (
    dnorm_gpu, pnorm_gpu, qnorm_gpu, rnorm_gpu,
    dt_gpu, pt_gpu, qt_gpu, rt_gpu,
    dpois_gpu, ppois_gpu, qpois_gpu, rpois_gpu,
)

pdf = dnorm_gpu(0.0)
p = pnorm_gpu(1.96)
q = qnorm_gpu(0.975)
sample_norm = rnorm_gpu(size=8)
pdf_t = dt_gpu(2.0, df=10)
pt = pt_gpu(2.0, df=10)
qt = qt_gpu(0.975, df=10)
sample_t = rt_gpu(df=10, size=8)
pmf_pois = dpois_gpu(3, 4.0)
sample_pois = rpois_gpu(4.0, size=8)
```

这些接口建议用于兼容已有脚本；新代码仍建议使用对象式 API。常见迁移关系：

- `dnorm_gpu(x)` -> `norm.pdf(x)`
- `pnorm_gpu(x)` -> `norm.cdf(x)`
- `qnorm_gpu(q)` -> `norm.ppf(q)`
- `rnorm_gpu(size=...)` -> `norm.rvs(size=...)`
- `dpois_gpu(k, mu)` -> `poisson.pmf(k, mu)`
- `rpois_gpu(mu, size=...)` -> `poisson.rvs(mu, size=...)`
- `pt_gpu(x, df)` -> `t.cdf(x, df=df)`
- `qt_gpu(q, df)` -> `t.ppf(q, df=df)`

## 7) 旧函数接口（非 R 风格，软弃用）

以下非 R 风格历史接口仍可用，但会发出 `DeprecationWarning`。同样按分布家族分组：

### norm 相关旧接口

- `norm_cdf_gpu` -> `norm.cdf`
- `norm_sf_gpu` -> `norm.sf`
- `norm_ppf_gpu` -> `norm.ppf`
- `norm_isf_gpu` -> `norm.isf`
- `norm_two_sided_pvalue_gpu` -> `norm.two_sided_pvalue`
- `norm_two_sided_critical_value_gpu` -> `norm.two_sided_critical_value`

### t 相关旧接口

- `t_cdf_gpu` -> `t.cdf`
- `t_sf_gpu` -> `t.sf`
- `t_ppf_gpu` -> `t.ppf`
- `t_two_sided_pvalue_gpu` -> `t.two_sided_pvalue`
- `t_two_sided_critical_value_gpu` -> `t.two_sided_critical_value`

建议将以上接口迁移到对象式 API（`norm.*` / `t.*`）。

## 8) 后端行为说明

- **自动模式（默认）**：代理对象按 CuPy > Torch > NumPy/scipy 顺序尝试。输入数组类型不变。
- **NumPy 后端**：使用 `scipy.stats` 和 `scipy.special`。接受 numpy 数组。
- **CuPy 后端**：使用 `cupyx.scipy.special`。接受 CuPy 数组。
- **Torch 后端**：使用 `torch.special`，缺失函数有回退实现。接受 Torch 张量。

原生分布内核在所需特殊函数不可用时不会隐式回退到 CPU。

## 9) 常见问题

1. 报错：GPU 后端缺少特殊函数

- 原因：所需的 `cupyx.scipy.special` 或 `torch.special` 函数不可用。
- 建议：检查 CUDA 驱动 + CuPy/Torch 版本匹配，或临时使用 `backend="numpy"`。

2. 查看可用分布名

```python
from statgpu.inference import list_available_distributions

native_only = list_available_distributions()
```

3. 后端精度差异

- CuPy 和 NumPy 后端对大多数函数与 `scipy.stats` 匹配到机器精度。
- Torch 2.0 缺少 `torch.special.betainc` — t/f/beta CDF/PPF 可能有 ~1e-5 到 1e-7 误差。
- 升级到 Torch >= 2.1（具有原生 `torch.special.betainc`）可解决。
