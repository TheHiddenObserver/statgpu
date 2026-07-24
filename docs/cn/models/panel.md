# Panel 模型

> 语言：中文  
> 最后更新：2026-07-24  
> 页面定位：模型文档  
> 切换：[English](../../en/models/panel.md)

## 概览

`statgpu.panel` 提供六类面板数据估计器：

- `PanelOLS`：个体和/或时间固定效应；
- `RandomEffects`：可行 GLS 随机效应；
- `PooledOLS`：不去均值的堆叠 OLS；
- `BetweenOLS`：在个体均值上回归；
- `FirstDifferenceOLS`：个体内一阶差分；
- `FamaMacBeth`：逐期横截面回归后对系数取平均。

数组输入的数值路径支持 NumPy、CuPy CUDA 与 Torch CUDA。formula 构造以及字符串/分类 entity、time、cluster 标签属于明确的 CPU 元数据边界，只会将对齐后的紧凑编码传入数值后端。显式 GPU device 不会静默回退 CPU。

## 路径

```python
from statgpu.panel import (
    PanelOLS,
    RandomEffects,
    PooledOLS,
    BetweenOLS,
    FirstDifferenceOLS,
    FamaMacBeth,
    clustered_covariance,
    two_way_clustered_covariance,
    hac_covariance,
)
```

## 模型汇总

| 模型 | 变换 | 主要推断选项 |
|---|---|---|
| `PanelOLS` | 个体/时间组内变换 | nonrobust、HC1 robust、clustered |
| `RandomEffects` | Swamy-Arora 可行 GLS | nonrobust |
| `PooledOLS` | 堆叠 OLS | nonrobust、robust、clustered、HAC |
| `BetweenOLS` | 个体均值 | nonrobust、robust、clustered |
| `FirstDifferenceOLS` | 个体内一阶差分 | nonrobust、robust |
| `FamaMacBeth` | 逐期横截面回归 | nonrobust、Newey-West |

## 核心估计方程

`PanelOLS` 在移除指定固定效应后拟合 OLS。仅使用个体效应时，

$$
y_{it}^{\mathrm{within}} = y_{it} - \bar y_{i\cdot},
\qquad
X_{it}^{\mathrm{within}} = X_{it} - \bar X_{i\cdot}.
$$

个体与时间双向固定效应会进一步减去时间均值并加回总体均值。

`PooledOLS` 拟合

$$
\hat\beta = X^+ y,
$$

其中 \(X^+\) 在需要时表示 Moore-Penrose 伪逆。`BetweenOLS` 对个体均值执行 OLS，`FirstDifferenceOLS` 对 Δ\(X\) 和 Δ\(y\) 执行 OLS，`FamaMacBeth` 对逐期系数向量求平均。

## 协方差与推断

| `cov_type` | 行为 |
|---|---|
| `"nonrobust"` | 经典 OLS 协方差和 t 推断 |
| `"robust"` | HC1 sandwich 协方差和渐近正态推断 |
| `"clustered"` | 单向或双向聚类稳健协方差 |
| `"hac"` | `PooledOLS` 使用 Bartlett/Newey-West HAC |
| `"newey-west"` | 对 `FamaMacBeth` 系数路径应用 HAC |

### PooledOLS HAC 时间排序

对 `PooledOLS(cov_type="hac")`，应在 `fit` 中传入 `time_index=`。实现会验证该侧数组，并使用稳定时间排序，同时保持所有数值数组对齐。因此，只要时间标签不变，对原始行进行排列不会改变 HAC 协方差（除数值误差外）。

### 秩亏 PooledOLS

秩亏设计需要区分拟合空间与系数空间：

- 拟合、预测、残差、RSS、有效秩和拟合空间比较仍然有效；
- `df_resid` 按 `nobs - rank(X)` 计算，而不是 `nobs - n_columns`；
- 精确共线时，单个系数不唯一；
- 因此系数级协方差、BSE、检验统计量、p 值和置信区间不可唯一识别，应标记为 `NOT_COMPARABLE`，而不是运行错误，也不能作为唯一推断结果报告。

PR79 验证管线在秩亏场景中继续检查 prediction、RSS、rank 与拟合空间合同，同时排除不可识别的系数空间比较。

## 参数与 fit 签名

### `PanelOLS`

```python
PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",
    device="auto",
)
```

```python
model.fit(y, X, entity_ids=entity_ids, time_ids=time_ids, cluster=cluster)
```

### `PooledOLS`

```python
PooledOLS(
    cov_type="nonrobust",
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
    device="auto",
)
```

```python
model.fit(X, y, cluster=None, time_index=None)
```

clustered 推断需要 `cluster`。HAC 推断强烈建议传入 `time_index`，该参数用于定义稳定时间顺序。

### 其他模型

```python
RandomEffects(device="auto")
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")
FamaMacBeth(
    cov_type="newey-west",
    bandwidth=None,
    alpha=0.05,
    min_obs_per_period=1,
    device="auto",
)
```

## CPU 与 GPU 示例

```python
import numpy as np
from statgpu.panel import PanelOLS, PooledOLS, FamaMacBeth

n_entities, n_times = 50, 10
n = n_entities * n_times
entity_ids = np.repeat(np.arange(n_entities), n_times)
time_ids = np.tile(np.arange(n_times), n_entities)
X = np.random.default_rng(0).normal(size=(n, 3))
y = X @ np.array([1.0, -0.5, 0.3]) + np.random.default_rng(1).normal(size=n) * 0.1

# CPU 固定效应。
fe = PanelOLS(entity_effects=True, cov_type="robust", device="cpu")
fe.fit(y, X, entity_ids=entity_ids)

# 使用显式时间顺序的 HAC PooledOLS。
pooled_hac = PooledOLS(cov_type="hac", device="cpu")
pooled_hac.fit(X, y, time_index=time_ids)

# CuPy CUDA Fama-MacBeth；元数据标签可保留在 CPU。
fm = FamaMacBeth(cov_type="newey-west", device="cuda")
fm.fit(X, y, time_ids=time_ids)
```

Torch CUDA 路径应传入 CUDA tensor，并设置 `device="torch"`。数组输入的公开预测方法会保留 estimator 后端。

## 输出

常见拟合属性包括：

- `coef_`；
- 在系数空间可识别时的 `bse_`、`tvalues_`、`pvalues_`、`conf_int_`；
- 适用模型的 `rsquared` 或 `rsquared_within`；
- `nobs`、`df_resid` 与有效秩；
- `FamaMacBeth` 的 `betas_`、`cov_params_` 与 `n_periods`。

对精确秩亏的 `PooledOLS`，下游使用者不得将系数级推断解释为唯一识别结果。

## Formula 与元数据边界

formula 计算可能因缺失值删除行。entity、time、cluster 等侧数组会按保留行同步对齐。字符串或分类标签在 CPU 上 factorize；数值变换和回归仍在所选后端执行。

## 验证

PR #79 已验证 NumPy、CuPy CUDA 与 Torch CUDA 的维护中面板路径。最终维护中的真实 GPU 测试在 Tesla P100 上 **33/33** 通过，覆盖后端保持的 `PooledOLS.predict()` 与秩亏 `NOT_COMPARABLE` 合同。exact-head GitHub Actions 同时通过 Python 3.9–3.12 regression matrix 与完整 CPU suite。

相关内容：

- `dev/reviews/pr79_physical_gpu_validation.md`；
- `dev/tests/test_pr79_physical_gpu.py`；
- Issue #83：清理未纳入维护测试树的旧 GPU 诊断脚本。

## 参考文献

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering.
