# 推断 API 参考

> **模块：** `statgpu.inference`  
> **最后更新：** 2026-09-09  
> **后端：** NumPy、CuPy、PyTorch

`statgpu.inference` 模块提供常用统计推断工具，包括概率分布、多重检验、排列检验和自助法。所有继承自 `BaseEstimator` 的公开 statgpu 估计器还会继承一组**估计器绑定的推断辅助方法**；这些方法会根据估计器当前的设备设置解析默认计算后端。

## 快速参考

```python
from statgpu.inference import norm, poisson, t, adjust_pvalues, combine_pvalues, permutation_test
```

| 函数/类 | 说明 |
|---|---|
| `norm`, `t`, `chi2`, `f`, `beta`, `gamma`, `poisson`, `binom`, `uniform`, `expon`, `cauchy`, `laplace`, `logistic`, `lognorm`, `weibull_min` | 分布对象（与 SciPy 兼容的 API） |
| `get_distribution(name, backend=...)` | 按名称动态获取分布对象 |
| `adjust_pvalues(pvals, method=...)` | 多重检验 p 值校正 |
| `combine_pvalues(pvals, method=...)` | 合并多个 p 值，构造全局检验 |
| `permutation_test(statistic, X, y, ...)` | 基于排列的假设检验 |
| `bootstrap_statistic(statistic, arrays, ...)` | 通用自助法引擎 |
| `multipletests(...)` | `adjust_pvalues` 的兼容别名 |

## 估计器绑定的推断辅助方法

每个公开 `BaseEstimator` 子类都继承下列模型上下文包装方法。`Ridge`、`Lasso`、`ElasticNet` 等模型不需要各自重新实现这些通用功能。

| 估计器方法 | 签名 | 模型上下文行为 |
|---|---|---|
| `adjust_pvalues` | `adjust_pvalues(pvalues=None, method="bh", alpha=0.05, axis=0, backend="auto")` | 未显式传 `pvalues` 时使用估计器的 `_pvalues`。返回原始/校正后 p 值、拒绝掩码、方法、`alpha`、`axis` 与解析出的后端。 |
| `combine_pvalues` | `combine_pvalues(pvalues=None, method="fisher", weights=None, axis=None, backend="auto")` | 未显式传 `pvalues` 时使用 `_pvalues`；返回合并统计量、全局 p 值和后端元数据。 |
| `bootstrap_statistic` | `bootstrap_statistic(statistic, *arrays, n_resamples=200, strategy="iid", strata=None, clusters=None, block_size=None, confidence_level=0.95, random_state=None, statistic_name="statistic", backend="auto")` | `backend="auto"` 按估计器**当前设备设置**解析。未显式传输入数组时，会在可用的情况下读取拟合后缓存的设计矩阵与响应，并转换到解析出的后端。 |
| `permutation_test` | `permutation_test(statistic, X, y, n_resamples=1000, strategy="iid", strata=None, groups=None, alternative="two-sided", random_state=None, statistic_name="statistic", backend="auto")` | 在调用共享排列检验引擎之前，把数据以及可选的分层标签/分组标签转换到由估计器当前设备设置解析出的后端。 |

`backend="auto"` 跟随估计器的**当前设备解析结果**：CPU 对应 NumPy，`device="cuda"` 对应 CuPy，`device="torch"` 对应 Torch。它不会读取之前某次拟合记录的 `_selected_backend_name`。因此，如果 `device="auto"` 的一次拟合曾通过自动路由选择了其他后端，之后再次调用这些模型上下文辅助方法时，仍可能重新解析到不同后端。

需要固定辅助方法实际使用的后端时，请显式传入 `backend=`。只要共享推断引擎支持，显式指定的后端就会覆盖模型上下文给出的默认选择。

如果没有显式传入 p 值，而估计器也不存在 `_pvalues`，`adjust_pvalues()` 与 `combine_pvalues()` 会抛出 `RuntimeError`，不会静默构造输入。同样，如果 `bootstrap_statistic()` 没有传入输入数组，则要求模型已经拟合，并保留可用的训练数组缓存。

一个典型的模型绑定用法是：

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.05,
    inference_method="debiased",
    compute_inference=True,
).fit(X, y)

adjusted = model.adjust_pvalues(method="bh")
combined = model.combine_pvalues(method="fisher")

boot = model.bootstrap_statistic(
    lambda X_, y_: float((X_[:, 0] * y_).mean()),
    n_resamples=500,
    random_state=7,
)

perm = model.permutation_test(
    lambda X_, y_: float((X_[:, 0] * y_).mean()),
    X,
    y,
    n_resamples=999,
    random_state=7,
)
```

当你希望推断工具根据估计器的**当前设备上下文**自动选择后端，并在适用时复用拟合状态时，可以使用这些估计器绑定方法；需要精确固定实际后端时则显式传 `backend=`。如果推断计算与任何已拟合模型无关，直接使用下面的 `statgpu.inference` 模块级函数通常更清晰。

---

## 分布函数

### 直接导入（默认 NumPy）

```python
from statgpu.inference import norm, poisson, t

# 生成随机样本
X = norm.rvs(size=1000)

# CDF、生存函数、PPF
p = norm.cdf(1.96)           # 0.975
s = norm.sf(1.96)            # 0.025
q = norm.ppf(0.975)          # 1.96

# 带参数的 Poisson
y = poisson.rvs(mu=3.0, size=1000)

# 带自由度的 t 分布
p = t.cdf(2.0, df=10)
```

### GPU 后端

```python
from statgpu.inference import norm

# Torch 后端
X_torch = norm.rvs(size=1000, backend="torch")    # CUDA 上的 Torch 张量
p = norm.cdf(x_torch, backend="torch")

# CuPy 后端
X_cupy = norm.rvs(size=1000, backend="cupy")      # GPU 上的 CuPy 数组

# 从输入类型自动检测后端
import torch
x = torch.tensor([0.0, 1.96]).cuda()
p = norm.cdf(x)  # 自动使用 Torch 后端
```

### 可用分布

| 分布 | 参数 | 方法 |
|---|---|---|
| `norm` | — | rvs, cdf, sf, ppf, isf, pdf |
| `t` | `df` | rvs, cdf, sf, ppf, isf, pdf |
| `chi2` | `df` | rvs, cdf, sf, ppf, isf, pdf |
| `f` | `dfn, dfd` | rvs, cdf, sf, ppf, isf, pdf |
| `beta` | `a, b` | rvs, cdf, sf, ppf, isf, pdf |
| `gamma` | `a` | rvs, cdf, sf, ppf, isf, pdf |
| `uniform` | — | rvs, cdf, sf, ppf, isf, pdf |
| `expon` | — | rvs, cdf, sf, ppf, isf, pdf |
| `cauchy` | — | rvs, cdf, sf, ppf, isf, pdf |
| `laplace` | — | rvs, cdf, sf, ppf, isf, pdf |
| `logistic` | — | rvs, cdf, sf, ppf, isf, pdf |
| `lognorm` | `s` | rvs, cdf, sf, ppf, isf, pdf |
| `weibull_min` | `c` | rvs, cdf, sf, ppf, isf, pdf |
| `poisson` | `mu` | rvs, cdf, sf, ppf, pmf |
| `binom` | `n, p` | rvs, cdf, sf, ppf, pmf |

### 动态查找

```python
from statgpu.inference import get_distribution

# 按名称获取
norm = get_distribution("norm", backend="torch")
pois = get_distribution("poisson", backend="cupy")

# 列出可用分布
from statgpu.inference import list_available_distributions
print(list_available_distributions())
```

---

## 多重检验

### adjust_pvalues（p 值校正）

```python
from statgpu.inference import adjust_pvalues
import numpy as np

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.5])

# Benjamini-Hochberg（控制错误发现率 FDR）
reject, pvals_adj = adjust_pvalues(pvals, method='bh')

# 其他方法：'bonferroni', 'holm', 'hochberg', 'by'（Benjamini-Yekutieli）
reject, pvals_adj = adjust_pvalues(pvals, method='bonferroni')
```

### combine_pvalues（全局 p 值）

```python
from statgpu.inference import combine_pvalues

pvals = np.array([0.01, 0.04, 0.03, 0.40])

# Fisher 方法
stat, p_global = combine_pvalues(pvals, method='fisher')

# Cauchy 合并检验（ACAT）
stat, p_global = combine_pvalues(pvals, method='cauchy')

# Stouffer 方法
stat, p_global = combine_pvalues(pvals, method='stouffer')
```

---

## 排列检验

```python
from statgpu.inference import permutation_test
import numpy as np

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = X @ np.ones(5) + rng.standard_normal(100)

# 检验 X[:,0] 和 y 的相关性
result = permutation_test(
    lambda X_, y_: np.corrcoef(X_[:, 0], y_)[0, 1],
    X, y,
    n_resamples=999,
    random_state=42,
)
print(f"p 值: {result.pvalue:.4f}")
```

---

## 自助法

```python
from statgpu.inference import bootstrap_statistic
import numpy as np

rng = np.random.default_rng(42)
data = rng.standard_normal(1000)

# 用自助法估计均值的不确定性
result = bootstrap_statistic(
    np.mean, (data,),
    n_resamples=9999,
    random_state=42,
)
print(f"均值: {result.statistic:.4f}")
print(f"95% CI: [{result.confidence_interval.low:.4f}, {result.confidence_interval.high:.4f}]")
```

---

## R 兼容性

从 R 迁移的用户可以使用 R 风格的函数名：

```python
from statgpu.inference import norm

# R 风格：dnorm, pnorm, qnorm, rnorm
from statgpu.inference import dnorm_gpu, pnorm_gpu, qnorm_gpu, rnorm_gpu

# 这些函数对应 R 的 dnorm/pnorm/qnorm/rnorm，并提供 GPU 加速实现
```

---

## 常见问题

**Q: 什么时候用 `get_distribution()`，什么时候直接导入？**  
A: NumPy 后端可以直接导入（`from statgpu.inference import norm`）。需要明确控制后端时使用 `get_distribution("norm", backend="torch")`。

**Q: 可以用 statgpu 分布替代 SciPy 吗？**  
A: 可以。API 与 SciPy 兼容：`rvs`, `cdf`, `sf`, `ppf`, `isf`, `pdf`/`pmf` 的调用方式保持一致。可以将 `scipy.stats.norm` 替换为 `statgpu.inference.norm`。

**Q: 如何使用 GPU 加速的分布？**  
A: 给分布方法传 `backend="torch"` 或 `backend="cupy"`，例如 `norm.rvs(size=1000, backend="torch")`。

**Q: `sf` 和 `1 - cdf` 有什么区别？**  
A: `sf(x)` 是生存函数（survival function），数学上等于 `1 - CDF`；但当 CDF 接近 1 时，直接计算 `sf` 通常数值更稳定。

---

## 参考文献

- **SciPy 统计模块：** [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
- **R 概率分布：** [https://stat.ethz.ch/R-manual/R-patched/library/stats/html/Distributions.html](https://stat.ethz.ch/R-manual/R-patched/library/stats/html/Distributions.html)
- **多重检验：** Benjamini & Hochberg (1995), "Controlling the False Discovery Rate"
- **Cauchy 合并检验：** Liu & Xie (2020), "Cauchy Combination Test"
