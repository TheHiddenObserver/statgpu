# Distribution API 使用指南

> 语言: 中文  
> 最后更新: 2026-04-12  
> 页面定位: 指南文档  
> 切换: [English](../en/guides/distribution-api.md)

语言切换：[English](../en/guides/distribution-api.md)

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

## 2) 按名称动态获取分布

如需按字符串动态选择分布，可使用 `get_distribution_gpu(name)`。

```python
from statgpu.inference import get_distribution_gpu

dist = get_distribution_gpu("norm")
y = dist.cdf(0.0)
```

当前默认策略是“仅原生 GPU 分布”：

- `allow_fallback=False`（默认）：只允许原生实现。
- 传入非原生分布名会抛出 `ValueError`。

## 3) 显式启用 SciPy fallback（可选）

对于尚未原生实现的长尾分布，可以显式开启 fallback：

```python
import cupy as cp
from statgpu.inference import get_distribution_gpu

dist = get_distribution_gpu("gumbel_r", allow_fallback=True)
out = dist.cdf(cp.asarray([0.0, 1.0, 2.0]))
```

说明：

- fallback 会调用 `scipy.stats` 计算，再将结果转回 CuPy。
- 这是显式行为，不会在默认路径下自动发生。

## 4) R 语言兼容风格接口（兼容保留）

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

## 5) 旧函数接口（非 R 风格，软弃用）

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

## 6) CPU 与 GPU 路径说明

- 本页分布对象 API 是 GPU 优先路径（输入/输出为 CuPy）。
- 原生分布实现缺少 `cupyx.scipy.special` 关键能力时会直接报错，不做隐式 CPU 回退。
- 模型推断（例如线性/逻辑/Cox 的 p 值、临界值）在 CPU 场景下仍会直接使用 `scipy.stats`，这是独立于本页分布对象 API 的另一条路径。

## 7) 常见问题

1. 报错 `cupyx.scipy.special.* is required for GPU backend`

- 原因：当前环境缺少可用的 `cupyx.scipy.special` 对应函数或 CUDA/CuPy 不可用。
- 建议：检查 CUDA、CuPy 与驱动版本匹配；或临时改用 CPU 模型推断路径。

2. 想查看所有可用分布名

```python
from statgpu.inference import list_available_distributions_gpu

native_only = list_available_distributions_gpu(include_scipy=False)
all_names = list_available_distributions_gpu(include_scipy=True)
```
