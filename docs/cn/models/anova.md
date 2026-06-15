# ANOVA

> 语言: 中文  
> 最后更新: 2026-05-28  
> 页面定位: 模型文档  
> 切换: [English](../en/models/anova.md)

语言切换：[English](../en/models/anova.md)

## 概览（Overview）

`f_oneway` 执行单因素方差分析（One-Way ANOVA），检验各组均值是否相等。它是 `scipy.stats.f_oneway` 的 GPU 加速替代实现，支持 numpy、cupy 和 torch 后端。

## 路径（Path）

`statgpu.anova.f_oneway`、`statgpu.anova.AnovaResult`

## 目标函数（Objective Function）

总均值：
$$
\bar{y} = \frac{\sum_i n_i \bar{y}_i}{\sum_i n_i}
$$

组间平方和：
$$
SSB = \sum_{i=1}^k n_i (\bar{y}_i - \bar{y})^2
$$

组内平方和：
$$
SSW = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij} - \bar{y}_i)^2
$$

F 统计量：
$$
F = \frac{SSB / (k-1)}{SSW / (N-k)}
$$

其中 $k$ 为组数，$n_i$ 为第 $i$ 组的样本量，$N = \sum_i n_i$ 为总观测数。

## 估计方程（Estimating Equation）

直接计算，无需迭代求解器。F 统计量通过后端原生的归约操作在单次数据遍历中完成计算。

## 协方差与推断（Covariance/Inference）

p 值由 F 分布的生存函数 $1 - F_{k-1,\,N-k}(F)$ 得到。效应量报告为：
$$
\eta^2 = \frac{SSB}{SSB + SSW}
$$

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `*groups` | （必填） | 两个或更多一维数组，每组一个 |
| `backend` | `"auto"` | `"auto"` / `"numpy"` / `"cupy"` / `"torch"` |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.anova import f_oneway
import numpy as np

# CPU
g1 = np.random.randn(100)
g2 = np.random.randn(100) + 0.5
result = f_oneway(g1, g2, backend="numpy")
print(f"F={result.statistic:.4f}, p={result.pvalue:.4e}, eta2={result.eta_squared:.4f}")

# GPU (cupy)
import cupy as cp
g1_gpu = cp.asarray(g1)
g2_gpu = cp.asarray(g2)
result_gpu = f_oneway(g1_gpu, g2_gpu, backend="cupy")

# GPU (torch)
import torch
g1_t = torch.from_numpy(g1).cuda()
g2_t = torch.from_numpy(g2).cuda()
result_torch = f_oneway(g1_t, g2_t, backend="torch")
```

## strict/approx 差异（strict/approx difference）

无 strict/approx 模式区分。单一计算路径，仅需选择后端。

## 输出（Outputs）

`AnovaResult` dataclass，包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `statistic` | float | F 统计量 |
| `pvalue` | float | F 分布的 p 值 |
| `df_between` | int | 组间自由度（$k - 1$） |
| `df_within` | int | 组内自由度（$N - k$） |
| `eta_squared` | float | 效应量 $\eta^2$ |

## 常见问题（FAQ）

- **支持多少组？** 两组或更多。
- **所有观测值完全相同时会怎样？** `statistic`、`pvalue` 和 `eta_squared` 均返回 `NaN`。
- **各组完全分离时会怎样？** `statistic` 返回 `inf`，`pvalue` 返回 `0.0`，`eta_squared` 返回 `1.0`。
- **能否直接替代 scipy？** 可以。函数签名和输出字段与 `scipy.stats.f_oneway` 兼容，并额外提供 `eta_squared`、`df_between` 和 `df_within`。

## 外部验证（External Validation）

针对 `scipy.stats.f_oneway` 进行验证，在多种组数和效应量组合下相对误差 < 1e-15。

## 参考文献（References）

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.
