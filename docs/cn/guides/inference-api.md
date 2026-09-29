# 推断 API 参考

> 语言：中文  
> 最后更新：2026-09-17  
> 模块：`statgpu.inference`  
> 切换：[English](../../en/guides/inference-api.md)

`statgpu.inference` 汇总可复用的统计工具，包括概率分布、多重检验、p 值合并、排列检验与自助法（bootstrap）。

本页只作为**模块入口**。分布函数的详细行为统一放在 [分布 API](distribution-api.md)；模型系数推断的方法选择统一放在 [推断模式](inference-modes.md)。

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
| `norm`、`t`、`chi2`、`poisson` 等分布对象 | CDF/SF/PPF/PDF/PMF/随机采样 | [分布 API](distribution-api.md) |
| `get_distribution(...)` | 动态选择概率分布与计算后端 | [分布 API](distribution-api.md) |
| `adjust_pvalues(...)` | 多重检验校正 | [多重检验](multiple-testing-combine-pvalues.md) |
| `combine_pvalues(...)` | 合并多个 p 值提供的证据 | [多重检验](multiple-testing-combine-pvalues.md) |
| `permutation_test(...)` | 基于排列的假设检验 | 本页 |
| `bootstrap_statistic(...)` | 对用户给定的统计量执行通用自助法 | 本页 |

## 分布函数

分布对象提供与 SciPy 风格接近的 `cdf`、`sf`、`ppf`、`isf`、`pdf`/`pmf` 与 `rvs` 方法。

```python
from statgpu.inference import norm, t

p = norm.cdf(1.96)
q = t.ppf(0.975, df=10)
```

后端选择、可用分布、反函数精度、R 风格兼容别名以及历史名称统一见 [分布 API](distribution-api.md)，本页不再重复说明。

## 多重检验与 p 值合并

```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.5])

reject, pvals_bh = adjust_pvalues(pvals, method="bh")
stat, p_global = combine_pvalues(pvals, method="fisher")
```

可用的校正方法、合并方法及其统计解释见 [多重检验](multiple-testing-combine-pvalues.md)。

## 排列检验

`permutation_test` 按照接口定义对输入数据反复排列，并重新计算用户提供的统计量，从而构造排列参考分布。

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

统计量与排列方案应当和应用中的原假设相匹配；通用排列检验工具无法替用户判断具体数据是否满足可交换性假设。

## 通用自助法

`bootstrap_statistic` 对调用者提供的统计量执行自助法重抽样。

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

这个通用工具与模型专属的推断模式并不是同一件事。例如，惩罚 Gaussian 模型的残差自助法有自己的统计定义。如果目标是已拟合模型的系数推断，应先查看 [推断模式](inference-modes.md)，不要假定通用自助法会自动复现某个模型专属的推断程序。

## 文档导航

可以按问题选择页面：

- **如何在 NumPy/CuPy/Torch 上计算概率分布？** → [分布 API](distribution-api.md)
- **如何校正或合并多个 p 值？** → [多重检验](multiple-testing-combine-pvalues.md)
- **如何运行通用排列检验或自助法？** → 本页
- **回归估计器应该选择哪种推断方法？** → [推断模式](inference-modes.md)
- **惩罚 GLM 系数推断的统计目标是什么？** → [惩罚 GLM 推断](penalized-glm-inference.md)
