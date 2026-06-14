# 推断 API 参考

> **模块:** `statgpu.inference`  
> **最后更新:** 2026-06-14  
> **后端:** NumPy, CuPy, PyTorch

`statgpu.inference` 模块提供统计推断工具：分布函数、多重检验、排列检验和自助法。

## 快速参考

```python
from statgpu.inference import norm, poisson, t, adjust_pvalues, combine_pvalues, permutation_test
```

| 函数/类 | 说明 |
|---|---|
| `norm`, `t`, `chi2`, `f`, `beta`, `gamma`, `poisson`, `binom`, `uniform`, `expon`, `cauchy`, `laplace`, `logistic`, `lognorm`, `weibull_min` | 分布对象（与 scipy 兼容的 API） |
| `get_distribution(name, backend=...)` | 动态分布查找 |
| `adjust_pvalues(pvals, method=...)` | 多重检验校正 |
| `combine_pvalues(pvals, method=...)` | 全局 p 值合并 |
| `permutation_test(statistic, X, y, ...)` | 基于排列的假设检验 |
| `bootstrap_statistic(statistic, arrays, ...)` | 通用自助法引擎 |
| `multipletests(...)` | `adjust_pvalues` 的别名（科学命名） |

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
X_torch = norm.rvs(size=1000, backend="torch")    # CUDA 上的 torch tensor
p = norm.cdf(x_torch, backend="torch")

# CuPy 后端
X_cupy = norm.rvs(size=1000, backend="cupy")      # GPU 上的 CuPy array

# 从输入类型自动检测后端
import torch
x = torch.tensor([0.0, 1.96]).cuda()
p = norm.cdf(x)  # 自动使用 torch 后端
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

# 按名称查找
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

# Benjamini-Hochberg（FDR 控制）
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

# 自助法均值
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

从 R 迁移的用户可以使用 R 兼容的函数名：

```python
from statgpu.inference import norm

# R 风格：dnorm, pnorm, qnorm, rnorm
from statgpu.inference import dnorm_gpu, pnorm_gpu, qnorm_gpu, rnorm_gpu

# 这些是 R 的 dnorm/pnorm/qnorm/rnorm 的 GPU 加速等价物
```
