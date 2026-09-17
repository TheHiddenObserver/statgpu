# 推断 API 参考

> 语言：中文  
> 最后更新：2026-09-17  
> 模块：`statgpu.inference`  
> 切换：[English](../../en/guides/inference-api.md)

`statgpu.inference` 汇总可复用的统计工具，包括概率分布、多重检验、p 值合并、排列检验与 bootstrap。

本页只作为**模块入口**。分布函数的详细行为统一放在 [分布 API](distribution-api.md)；模型 coefficient inference 的方法选择统一放在 [推断模式](inference-modes.md)。

## 快速参考

```python
from statgpu.inference import (
    norm,
    t,
    adjust_pvalues,
    combine_pvalues,
    permutation_test,
    bootstrap_statistic,
)
```

| API | 用途 | 详细文档 |
|---|---|---|
| `norm`、`t`、`chi2`、`poisson` 等 distribution object | CDF/SF/PPF/PDF/PMF/随机采样 | [分布 API](distribution-api.md) |
| `get_distribution(...)` | 动态选择 distribution/backend | [分布 API](distribution-api.md) |
| `adjust_pvalues(...)` | 多重检验校正 | [多重检验](multiple-testing-combine-pvalues.md) |
| `combine_pvalues(...)` | 合并多个 p 值中的证据 | [多重检验](multiple-testing-combine-pvalues.md) |
| `permutation_test(...)` | permutation-based hypothesis test | 本页 |
| `bootstrap_statistic(...)` | 对用户给定 statistic 做通用 bootstrap | 本页 |

## 分布函数

Distribution object 提供与 scipy 风格接近的 `cdf`、`sf`、`ppf`、`isf`、`pdf`/`pmf` 与 `rvs` 方法。

```python
from statgpu.inference import norm, t

p = norm.cdf(1.96)
q = t.ppf(0.975, df=10)
```

backend 选择、可用 distribution、inverse-function 精度、R-style compatibility alias 与 legacy name 都统一见 [分布 API](distribution-api.md)，本页不再复制一套。

## 多重检验与 p 值合并

```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.5])

reject, pvals_bh = adjust_pvalues(pvals, method="bh")
stat, p_global = combine_pvalues(pvals, method="fisher")
```

可用的 adjustment/combination method 及其统计解释见 [多重检验](multiple-testing-combine-pvalues.md)。

## 排列检验

`permutation_test` 根据 API 定义对输入数据重复排列，并重新计算用户提供的 statistic，从而构造 permutation reference distribution。

```python
import numpy as np
from statgpu.inference import permutation_test

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = X @ np.ones(5) + rng.standard_normal(100)

result = permutation_test(
    lambda X_, y_: np.corrcoef(X_[:, 0], y_)[0, 1],
    X,
    y,
    n_resamples=999,
    random_state=42,
)

print(result.pvalue)
```

statistic 与 permutation scheme 应与应用中的 null hypothesis 相匹配；通用 permutation engine 无法替用户判断具体数据是否满足 exchangeability 假设。

## 通用 bootstrap

`bootstrap_statistic` 对调用者提供的 statistic 执行 bootstrap。

```python
import numpy as np
from statgpu.inference import bootstrap_statistic

rng = np.random.default_rng(42)
data = rng.standard_normal(1000)

result = bootstrap_statistic(
    np.mean,
    (data,),
    n_resamples=9999,
    random_state=42,
)

print(result.statistic)
print(result.confidence_interval)
```

这个通用工具与 estimator-specific inference mode（例如 penalized Gaussian residual bootstrap）不是同一件事。若目标是 fitted model 的 coefficient inference，应先看 [推断模式](inference-modes.md)，不要假定 generic bootstrap 自动复现某个模型专属推断程序。

## 文档导航

可以按问题选择页面：

- **如何在 NumPy/CuPy/Torch 上计算概率分布？** → [分布 API](distribution-api.md)
- **如何校正或合并多个 p 值？** → [多重检验](multiple-testing-combine-pvalues.md)
- **如何运行通用 permutation/bootstrap？** → 本页
- **回归 estimator 应该选哪种 inference method？** → [推断模式](inference-modes.md)
- **penalized-GLM coefficient inference 的统计 target 是什么？** → [Penalized GLM 推断](penalized-glm-inference.md)
