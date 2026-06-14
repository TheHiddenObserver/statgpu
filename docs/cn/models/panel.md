# Panel

> 语言: 中文
> 最后更新: 2026-05-28
> 页面定位: 模型文档
> 切换: [English](../en/models/panel.md)

语言切换：[English](../en/models/panel.md)

## 概览（Overview）

`panel` 模块提供面板数据（纵向数据）模型。`PanelOLS` 估计固定效应（个体效应和/或时间效应），支持非稳健、HC1 稳健和聚类标准误。`RandomEffects` 使用 Swamy-Arora 方差分量估计器实现可行 GLS 随机效应。两个类均支持 CPU、CuPy 和 PyTorch 后端，并自动检测设备。

## 路径（Path）

- `statgpu.panel.PanelOLS`
- `statgpu.panel.RandomEffects`
- `statgpu.panel.clustered_covariance`
- `statgpu.panel.two_way_clustered_covariance`

## 目标函数（Objective Function）

**PanelOLS** 求解组内变换 OLS。对因变量和回归变量进行去均值处理以消除固定效应。对于个体效应，变换为：

$$
y_{it}^{within} = y_{it} - \bar{y}_{i\cdot}
$$

对于双向（个体 + 时间）固定效应，双重去均值残差为：

$$
y_{it}^{within} = y_{it} - \bar{y}_{i\cdot} - \bar{y}_{\cdot t} + \bar{y}_{\cdot\cdot}
$$

其中 \(\bar{y}_{i\cdot}\) 为个体均值，\(\bar{y}_{\cdot t}\) 为时间均值，\(\bar{y}_{\cdot\cdot}\) 为总均值。对 \(X\) 逐列施加相同变换。

**RandomEffects** 估计方差分量模型：

$$
y_{it} = \alpha + X_{it}'\beta + a_i + \epsilon_{it}
$$

其中 \(a_i \sim \text{iid}(0, \sigma^2_a)\) 为个体随机效应，\(\epsilon_{it} \sim \text{iid}(0, \sigma^2_e)\) 为特异性误差。Swamy-Arora 估计器从组内估计器获得 \(\hat{\sigma}^2_e\)，从组间估计器获得 \(\hat{\sigma}^2_a\)，然后应用可行 GLS。

## 估计方程（Estimating Equation）

**PanelOLS** 在去均值数据上拟合 OLS：

$$
\hat{\beta} = (X_d^\top X_d)^{-1} X_d^\top y_d
$$

其中 \(X_d\) 和 \(y_d\) 为个体（可选加上时间）去均值后的回归变量和因变量。

**RandomEffects** 分六步进行：

1. **组间估计**：计算组均值 \(\bar{y}_i, \bar{X}_i\)，在组间数据上运行 OLS 得到 \(\hat{\beta}_{between}\)。

2. **组内估计**：个体去均值 OLS 得到 \(\hat{\beta}_{within}\) 及残差平方和 \(RSS_{within}\)。

3. **方差分量**：
   \[
   \hat{\sigma}^2_e = \frac{RSS_{within}}{N - n_{entities} - k}, \qquad
   \hat{\sigma}^2_a = \max\!\left(0,\; \frac{RSS_{between}}{n_{entities}} - \frac{\hat{\sigma}^2_e}{\bar{T}}\right)
   \]
   其中 \(\bar{T}\) 为每个个体的平均观测数。

4. **GLS 权重**（逐个体）：
   \[
   \theta_i = 1 - \sqrt{\frac{\hat{\sigma}^2_e}{\hat{\sigma}^2_e + T_i\,\hat{\sigma}^2_a}}
   \]

5. **准去均值 OLS**：施加部分去均值变换
   \[
   y^*_{it} = y_{it} - \theta_i\,\bar{y}_{i\cdot}, \qquad
   X^*_{it} = X_{it} - \theta_i\,\bar{X}_{i\cdot}
   \]
   在变换后的数据上运行 OLS。

6. **推断**：在准去均值数据上计算 OLS 协方差。

## 协方差与推断（Covariance/Inference）

`PanelOLS` 的 `cov_type` 参数选择推断方法：

- **`nonrobust`**：经典 OLS 协方差 \(\hat{\sigma}^2 (X_d^\top X_d)^{-1}\)。p 值基于 \(t\) 分布，自由度为 \(df_{resid}\)。
- **`robust`**：HC1 sandwich 估计器（White 1980，含有限样本 \(n/(n-k)\) 修正）。p 值基于标准正态分布。
- **`clustered`**：聚类稳健 sandwich 估计器（Cameron & Miller 2015）。p 值基于标准正态分布。对于双向聚类，传入 2 列聚类数组；方差通过 Cameron, Gelbach & Miller (2011) 方法计算。

`RandomEffects` 默认在准去均值数据上使用非稳健 OLS 推断。

`fit()` 后的输出：`coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared_within`（PanelOLS）。

## 参数（Parameters）

### PanelOLS

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `entity_effects` | `False` | 是否包含个体（实体）固定效应 |
| `time_effects` | `False` | 是否包含时间固定效应 |
| `cov_type` | `'nonrobust'` | 协方差类型：`'nonrobust'`、`'robust'` 或 `'clustered'` |
| `device` | `"auto"` | 计算设备：`"cpu"`、`"cuda"` 或 `"auto"` |

### RandomEffects

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `device` | `"auto"` | 计算设备：`"cpu"`、`"cuda"` 或 `"auto"` |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.panel import PanelOLS, RandomEffects
import numpy as np

# 生成面板数据
n_entities, n_times = 50, 10
n = n_entities * n_times
entity_ids = np.repeat(np.arange(n_entities), n_times)
time_ids = np.tile(np.arange(n_times), n_entities)
X = np.random.randn(n, 3)
y = X @ [1.0, -0.5, 0.3] + np.random.randn(n) * 0.1

# --- 固定效应（CPU）---

fe = PanelOLS(entity_effects=True, cov_type='robust', device='cpu')
fe.fit(y, X, entity_ids=entity_ids)
print(f"系数: {fe.coef_}, 标准误: {fe.bse_}")
print(f"组内 R 方: {fe.rsquared_within:.4f}")

# 双向固定效应 + 聚类标准误
fe2 = PanelOLS(entity_effects=True, time_effects=True,
               cov_type='clustered', device='cpu')
fe2.fit(y, X, entity_ids=entity_ids, time_ids=time_ids,
        cluster=entity_ids)
print(f"双向 FE 系数: {fe2.coef_}")

# --- 随机效应（CPU）---

re = RandomEffects(device='cpu')
re.fit(y, X, entity_ids=entity_ids)
print(f"RE 系数: {re.coef_}, theta: {re.theta_}")
print(f"方差分量: {re.variance_components_}")

# --- 随机效应（GPU）---

re_gpu = RandomEffects(device='cuda')
re_gpu.fit(y, X, entity_ids=entity_ids)
print(f"GPU RE 系数: {re_gpu.coef_}, theta: {re_gpu.theta_}")

# --- GPU + PyTorch 张量 ---

import torch
y_torch = torch.from_numpy(y).cuda().float()
X_torch = torch.from_numpy(X).cuda().float()
fe_torch = PanelOLS(entity_effects=True, cov_type='robust', device='cuda')
fe_torch.fit(y_torch, X_torch, entity_ids=entity_ids)
print(f"Torch FE 系数: {fe_torch.coef_}")
```

## strict/approx 差异（strict/approx difference）

面板模型没有 strict/approx 模式之分。`cov_type` 参数控制推断方法：

- `'nonrobust'`：经典 OLS 标准误，假设同方差且无组内相关。p 值使用 \(t\) 分布。
- `'robust'`：HC1 异方差稳健标准误（White sandwich）。p 值使用正态分布。
- `'clustered'`：聚类稳健标准误，允许组内任意相关。p 值使用正态分布。支持单向和双向聚类。

## 输出（Outputs）

### 拟合属性

| 属性 | 形状 | 说明 |
|---|---|---|
| `coef_` | `(k,)` | 估计系数 |
| `bse_` | `(k,)` | 标准误 |
| `tvalues_` | `(k,)` | t 统计量（稳健/聚类时为 Z 统计量） |
| `pvalues_` | `(k,)` | p 值 |
| `conf_int_` | `(k, 2)` | 95% 置信区间 |
| `rsquared_within` | 标量 | 组内 R 方（仅 PanelOLS） |
| `theta_` | 标量 | GLS 变换权重（仅 RandomEffects） |
| `variance_components_` | dict | `{'sigma2_e': float, 'sigma2_a': float}`（仅 RandomEffects） |
| `nobs` | int | 观测数 |
| `df_resid` | int | 残差自由度 |

### 方法

| 方法 | 返回值 | 说明 |
|---|---|---|
| `fit(y, X, entity_ids, ...)` | `self` | 拟合面板模型。需要 `entity_ids`（一维个体标签数组）。可选：`time_ids`、`cluster`。 |
| `predict(X, entity_ids)` | `ndarray` | 预测值 |
| `summary()` | str | 格式化汇总表 |

## 常见问题（FAQ）

**FE 和 RE 如何选择？**
当个体效应可能与回归变量相关时，使用固定效应（`PanelOLS`）。无论是否存在相关性，FE 估计器都是一致的。当效应与回归变量不相关时，使用随机效应（`RandomEffects`）以提高效率。Hausman 检验可帮助判断：若 Hausman 统计量显著，则优先选择 FE。

**如何做双向聚类？**
将 `cluster` 作为 2 列数组（或两个数组的列表）传入 `PanelOLS.fit()`。每列定义一个聚类维度。双向聚类方差通过 Cameron, Gelbach & Miller (2011) 方法计算，该方法投影到两个聚类集的并集上。

**与 `linearmodels` 有什么区别？**
统计方法与 `linearmodels.panel.PanelOLS` 和 `linearmodels.panel.RandomEffects` 相同。主要区别在于 GPU 加速：statgpu 将核心线性代数分派到 CuPy 或 PyTorch 后端，在有 GPU 的大型面板数据集上可提供加速。

**可以直接传入 CuPy 或 PyTorch 数组吗？**
可以。将 CuPy ndarray 或 PyTorch 张量作为 `y` 或 `X` 传入，后端会根据输入类型自动检测。也可以对 NumPy 输入显式设置 `device="cuda"` 来强制 GPU 计算。

**非平衡面板如何处理？**
`PanelOLS` 和 `RandomEffects` 均支持非平衡面板。个体和时间标识符定义数据结构；每个个体可以有不同数量的观测 \(T_i\)。RandomEffects 中的 GLS 权重 \(\theta_i\) 按个体变化，以考虑不同的 \(T_i\)。

## 外部验证（External Validation）

与 `linearmodels.panel.PanelOLS` 和 `linearmodels.panel.RandomEffects`（来自 `linearmodels` 包）进行了验证。在测试数据集上，系数估计、标准误和方差分量的相对误差 < 1e-12。一致性检查维护在 `dev/tests/test_external_consistency.py` 中。

## 参考文献（References）

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. *Journal of Human Resources*, 50(2), 317-372. [https://doi.org/10.3368/jhr.50.2.317](https://doi.org/10.3368/jhr.50.2.317)
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(3), 238-249. [https://doi.org/10.1198/jbes.2010.07136](https://doi.org/10.1198/jbes.2010.07136)
- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909125](https://doi.org/10.2307/1909125)
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
