# Panel 模型

> 语言：中文  
> 最后更新：2026-08-10
> 页面定位：模型文档  
> 切换：[English](../../en/models/panel.md)

## 概览

`statgpu.panel` 提供六类面板数据估计器：

- `PanelOLS`：个体和/或时间固定效应；
- `RandomEffects`：Swamy-Arora 可行 GLS 随机效应；
- `PooledOLS`：不去均值的堆叠 OLS；
- `BetweenOLS`：在个体均值上回归；
- `FirstDifferenceOLS`：个体内一阶差分；
- `FamaMacBeth`：逐期横截面回归后对系数取平均。

数组输入的数值路径支持 NumPy、CuPy CUDA 与 Torch CUDA。formula 构造以及字符串/分类 entity、time、cluster 标签属于明确的 CPU 元数据边界，只会把对齐后的紧凑编码传入数值后端。显式 GPU device 不会静默回退 CPU。

Tier-1 Panel 路线的 Stage C 在 Stage-B diagnostics 之上补齐 residual-sandwich covariance 层：历史默认行为保持不变，同时加入 HC0/HC2/HC3、RandomEffects robust inference、显式 cluster group debias 与 Driscoll-Kraay，并保持 NumPy/CuPy/Torch 数值累积后端原生。修复后的 covariance/provenance 实现已在 exact-clean head `aad53587...` 上重新完成 Tesla P100 验证：CuPy 与 Torch 各自通过全部 26 个 estimator covariance case 和 6 个 direct public covariance primitive（每个 backend 32/32），包括 full-rank ill-conditioned HC0/HC2/HC3/DK；同步 performance 也覆盖有界的 `N=10,000`、`k=2`、`T=200` QS 场景。此前 `c151550a...` 与 `9c0b3050...` 产物继续作为不可变历史证据保留。

## 路径

```python
from statgpu.panel import (
    PanelOLS,
    RandomEffects,
    PooledOLS,
    BetweenOLS,
    FirstDifferenceOLS,
    FamaMacBeth,
    PanelTestResult,
    PanelFitStatistics,
    hausman_test,
    pooling_f_test,
    breusch_pagan_lm_test,
    clustered_covariance,
    two_way_clustered_covariance,
    hac_covariance,
    driscoll_kraay_covariance,
)
```

诊断结果类和三类检验函数也从顶层 `statgpu` 导出。

## 模型汇总

| 模型 | 变换 | 主要推断选项 | 标准化 fit statistics |
|---|---|---|---|
| `PanelOLS` | entity/time 组内变换 | nonrobust；HC0/HC1/HC2/HC3；one-/two-way clustered；Driscoll-Kraay | within/between/overall R²、adjusted R²、classical model F、pooling F |
| `RandomEffects` | Swamy-Arora 可行 GLS | nonrobust；HC0/HC1/HC2/HC3；one-/two-way clustered；Driscoll-Kraay | within/between/overall R²、adjusted R²、classical model F；nonrobust 时可作为 Hausman 输入 |
| `PooledOLS` | 堆叠 OLS | nonrobust；HC0/HC1/HC2/HC3；clustered；legacy row-HAC；Driscoll-Kraay | overall R² 始终可用；提供 `entity_ids` 后增加 within/between R² 与 BP-LM；另有 adjusted R² 和 classical model F |
| `BetweenOLS` | entity 均值 | nonrobust；HC0/HC1/HC2/HC3 | within/between/overall R²、adjusted R²、classical model F |
| `FirstDifferenceOLS` | entity 内一阶差分 | nonrobust；HC0/HC1/HC2/HC3 | within/between/overall R²、差分拟合空间 adjusted R²、classical model F |
| `FamaMacBeth` | 逐期横截面回归 | nonrobust、Newey-West | 参数型 within/between/overall R²；不定义 residual-OLS adjusted R² 或 model F |

## 核心估计方程

`PanelOLS` 在移除指定固定效应后拟合 OLS。仅使用个体效应时，

$$
y_{it}^{\mathrm{within}} = y_{it} - \bar y_{i\cdot},
\qquad
X_{it}^{\mathrm{within}} = X_{it} - \bar X_{i\cdot}.
$$

个体和时间双向固定效应会进一步移除时间均值并加回总体均值。

`PooledOLS` 拟合

$$
\hat\beta = X^+ y,
$$

其中 \(X^+\) 在需要时表示 Moore-Penrose 伪逆。`BetweenOLS` 对个体均值执行 OLS，`FirstDifferenceOLS` 对 \(\Delta X\) 和 \(\Delta y\) 执行 OLS，`FamaMacBeth` 对逐期系数向量求平均。

## Stage-C 协方差与推断

Stage C 是增量式扩展：系数估计、Stage-B fit statistics 与历史默认推断不改变。协方差名称规范如下。

| `cov_type` | 行为 |
|---|---|
| `"nonrobust"` | 经典拟合空间 OLS covariance 与 Student-t 推断 |
| `"robust"`、`"hc1"` | 历史 statgpu HC1 sandwich 与渐近正态推断；`hc1` 规范化为 canonical `robust` |
| `"hc0"` | estimator 实际拟合空间上的未缩放 Eicker-White sandwich |
| `"hc2"`、`"hc3"` | estimator 实际拟合空间上的 leverage-adjusted sandwich |
| `"clustered"` | one-/two-way cluster sandwich；支持时可用 `group_debias=True` 显式修正 |
| `"driscoll-kraay"`、`"dk"`、`"kernel"` | 按 time 聚合 score 后计算 Driscoll-Kraay，可选 Bartlett、Parzen、Quadratic Spectral kernel |
| `"hac"` | `PooledOLS` 历史 row-order Bartlett/Newey-West；与 Driscoll-Kraay 明确区分 |
| `"newey-west"` | `FamaMacBeth` 系数路径已有 HAC；不进入 Stage-C residual-OLS covariance 层 |

### HC0/HC2/HC3 的拟合空间定义

设 estimator 实际数值回归使用的 fit-space design 为 `Z`，残差为 `e`，`B=(Z'Z)^+`。leverage 为

$$
h_i=z_i^\top Bz_i.
$$

HC0 的 meat 为 $\sum_i z_i z_i^\top e_i^2$；HC2 将每个残差平方除以 $1-h_i$；HC3 除以 $(1-h_i)^2$。实现逐行计算 leverage，不会构造 `n x n` hat matrix。当 leverage 在数值上等于 1 时，HC2/HC3 本身未定义，因此显式报错而不是裁剪成看似有效的 covariance。

不同模型使用不同 fit space：`PooledOLS` 为含 fitted intercept 的 level design；`PanelOLS` 为 fixed-effect transformed slope design；`RandomEffects` 为 quasi-demeaned `X_star`；`BetweenOLS` 为 entity-mean design；`FirstDifferenceOLS` 为保留的一阶差分 design。因此 Panel HC2/HC3 明确称为 **transformed-fit-space HC2/HC3**，不会暗中改成完整 dummy regression 的 HC2/HC3。

### Cluster covariance 与 `group_debias`

one-way clustering 在 cluster 内汇总 score vector。two-way clustering 对 cluster 1、cluster 2 与精确 paired-label intersection 做 inclusion-exclusion。默认 `group_debias=False` 完全保留历史 statgpu clustered covariance。设某一 component 有 `G` 个 group，则 `group_debias=True` 会在 inclusion-exclusion 前将该 component 的 meat 乘以

$$
\frac{G}{G-1}\frac{n-1}{n}.
$$

这只改变 covariance 大小；Stage C 不会静默把 coefficient test 改成 finite-group t reference。字符串/分类 cluster label 属于 metadata，可在 CPU factorize，但数值 score matrix 不会搬回 CPU。

### Driscoll-Kraay covariance

Driscoll-Kraay 首先按 observed time 聚合 fit-space score：

$$
g_t=\sum_i z_{it}e_{it},
$$

再对有序 `g_t` 序列计算 kernel HAC。`PooledOLS` 使用 `time_index=`；`PanelOLS` 与 `RandomEffects` 使用对齐后的 `time_ids=`。unbalanced panel 直接按每个 time 实际存在的观测聚合。

对 full-rank fit-space design（列数 `k`），Stage C 使用与 `linearmodels==7.0` 对齐的 debiased scale：

$$
\mathrm{scale}_{DK}=\frac{n}{n-\mathrm{extra\_df}-k}.
$$

`PooledOLS` 与 `RandomEffects` 的 `extra_df=0`；`PanelOLS` 使用 Stage-B standard fixed-effect nuisance rank（`N`、`T` 或 `N+T-C`）。若 statgpu 合法到达 rank-deficient fit，则记录为扩展：用 numerical rank 替代 `k` 并配合 pseudoinverse；该 corner 不宣称与 `linearmodels` 完全相等。

`bandwidth=None` 使用 `floor(4*(T/100)^(2/9))`，其中 `T` 是 observed distinct time period 数。Bartlett/Newey-West 与 Parzen/Gallant 在 bandwidth 处截断。Quadratic Spectral（`qs`、Andrews）把 bandwidth 当作平滑尺度；bandwidth 为正时对**所有 observed lag**赋权，而不是在 `bw` 截断。numeric 与 datetime time key 按自然排序处理；ordered pandas categorical 保留显式声明的 category chronology，并仅压缩实际 observed categories。普通 string/object label 仍采用 deterministic sorted-label order；若时间顺序与字典序不同，应传入 ordered categorical 或显式 numeric/datetime time key。

### RandomEffects covariance

Stage C 不改变 Swamy-Arora variance component 或 coefficient estimate。robust、HC、cluster 与 Driscoll-Kraay 都基于 quasi-demeaned GLS design `X_star` 与相应 residual 计算，因此改变 `cov_type` 只改变 inference。Stage-B classical Hausman 要求 **FE 与 RE 两端都使用 nonrobust covariance**；robust auxiliary Hausman 不在 Stage C 范围内，并返回结构化 inapplicable 结果。

### Backend 与验证状态

HC leverage、row score、cluster/time grouped score、lag product、bread/meat/covariance 都保留在 NumPy/CuPy/Torch 数值后端。CPU transfer 只允许 label/group code、小型配置和 scalar audit reduction。显式 GPU device 不静默回退 CPU。

hosted Stage-C tests 已将 HC2/HC3 与 analytic/statsmodels fit-space 计算对齐，并将 cluster/Driscoll-Kraay definition 与 `linearmodels==7.0` 固定版本对齐。fresh exact-clean head `aad53587...` 的 Tesla P100 acceptance 已闭合：CuPy 与 Torch 每个 backend 均通过 26/26 estimator covariance case 与 6/6 direct public covariance primitive（每个 backend 32/32），包括 full-rank ill-conditioned HC0/HC2/HC3/DK regressions，requested/executed backend 一致且无 CPU fallback。同步 performance rerun 覆盖三个 base scale 以及显式 `N=10,000`、`k=2`、`T=200` QS all-lag 场景，只记录 timing，不声明 speedup；此前 `c151550a...` 与 `9c0b3050...` 产物继续作为历史审计证据。

### PooledOLS HAC 时间排序

对 `PooledOLS(cov_type="hac")`，应向 `fit` 传入 `time_index=`。实现采用稳定时间排序，并让 X、y 和 Stage-B entity diagnostic metadata 使用同一个 permutation。因此 BP-LM 与参数型 R² 不会因为 HAC 排序后仍使用旧 entity metadata 而发生错位。

### 秩亏 PooledOLS

秩亏设计需要区分拟合空间与系数空间：

- 拟合、预测、残差、RSS、有效 rank 与拟合空间比较仍然有效；
- `df_resid` 使用 `nobs - rank(X)`，而不是 `nobs - n_columns`；
- 精确共线时单个系数不唯一；
- 因而系数级 covariance、BSE、test statistic、p-value 和 confidence interval 不应被解释为唯一识别的系数推断。

Stage-B model-F 的 restriction rank 使用有效数值 rank，而不是直接使用原始列数。

## 标准化 `fit_statistics_`

支持的拟合对象提供结构化 `PanelFitStatistics`：

```python
stats = model.fit_statistics_

stats.rsquared_within
stats.rsquared_between
stats.rsquared_overall
stats.rsquared_adj
stats.f_statistic
stats.f_pvalue
stats.f_df
stats.metadata
```

### 参数型 R²

within/between/overall R² 采用与 `linearmodels` 对齐的 parameter-based 定义：各统计量直接评价同一个拟合系数向量，而不是使用 fitted-value correlation 的平方。

给定 \(\hat\beta\)：

- **overall R²** 在 level panel 上评价 \(y-X\hat\beta\)；
- **between R²** 在个体均值上评价 \(\bar y_i-\bar X_i\hat\beta\)；
- **within R²** 在 entity-demeaned 的 y 和 X 上评价同一系数向量。

只有 level regressor design 中存在实际可识别的常数项时，overall 和 between total sum of squares 才中心化。固定效应本身不会自动改变这一规则。`RandomEffects` 会检测传入 level design 中显式的非零常数列，并在 adjusted R² 与 restricted model-F 中保留其 quasi-demeaned transformed column。常数检测的容差相对于该列自身量级定义，因此仅改变单位不会把一个非零常数误判成 slope，反之亦然。若 total sum of squares 为 0，Stage-B 标准字段返回 `0.0`，并在 `metadata["degenerate_total_ss"]` 中标记。

### Legacy `PanelOLS.rsquared_within`

`PanelOLS.rsquared_within` 为兼容 Stage A 保持原样。双向 FE 下，它描述历史上的 entity+time 完整 transformed fit，因此可能与标准化的 entity-within `fit_statistics_.rsquared_within` 不同。Stage B 不覆盖旧属性，并在 fit-statistics metadata 中保留兼容值。

### Adjusted R² 与 diagnostic df

新的标准化 diagnostics 使用完整 fixed-effect nuisance space 的 rank。当前 `PanelOLS` 不保留 exogenous intercept，因此标准 nuisance-effect rank 为：

- 仅 entity effects：\(N\)；
- 仅 time effects：\(T\)；
- entity + time effects：\(N+T-C\)，其中 \(C\) 是观测到的 entity-time incidence graph 的 connected-component 数。

因此常见的 \(N+T-1\) 只是连通面板 \(C=1\) 的特例；若 incomplete panel 的 incidence graph 不连通，diagnostic df 使用实际 dummy-space rank，而不会硬编码连通情形。

若 transformed slope design 的数值 rank 为 \(r_X\)，Stage B 使用

$$
\mathrm{df}_{\mathrm{resid,diag}}
= n-r_X-r_{\mathrm{effects}},
$$

以及

$$
\mathrm{df}_{\mathrm{total,diag}}
= n-r_{\mathrm{effects}}
$$

来计算标准 FE model F 与 adjusted R²。

这套 diagnostic df 与历史公开 `PanelOLS.df_resid` 明确分离；旧 df 继续服务已有 covariance、t statistic、p-value、confidence interval 与 summary。Hausman 使用一份仅供 diagnostic 的小型 FE covariance，并按标准 nuisance-rank denominator 重标度；Stage-A 公共 BSE/CI 不受影响。

### Classical model F

对 OLS-style panel estimator，Stage B 报告主拟合空间上的 classical homoskedastic joint-slope F：

$$
F=
\frac{(RSS_R-RSS_U)/q}
     {RSS_U/\mathrm{df}_{\mathrm{resid}}},
$$

其中 \(q\) 是有效 restriction rank。即使 estimator 的 coefficient covariance 使用 robust 或 clustered 选项，这一字段也不会静默变成 robust Wald test。当 unrestricted regression 精确拟合而 restricted regression 的 RSS 为正时，标准化结果采用经典极限值 `F=inf`、`p=0`，而不是把统计量标成 unavailable。RSS 的 zero/nesting tolerance 按 RSS 自身量级缩放，因此将 response 与 fitted coefficient 同时乘一个单位变换常数不会改变无量纲 F statistic 或 applicability。

`FamaMacBeth` 不定义 residual-OLS model F 或 adjusted R²，因为其 covariance 来自逐期横截面系数时间序列；Stage B 不会把 beta-series inference 重命名成 residual OLS inference。

## Specification Tests

所有 specification test 都返回 `PanelTestResult`。在计量意义上不适用的情况返回 `applicable=False` 并给出 `reason`；真正的 API 编程错误仍正常抛出异常。

### Pooling F test

对至少包含一个固定效应的已拟合 `PanelOLS`：

```python
result = fe.pooling_f_test()
# 或
result = pooling_f_test(fe)
```

经典 poolability 原假设是所有纳入的固定效应联合为 0。统计量使用**同一个对齐后的估计样本**比较 fixed-effect model 与其 nested pooled regression。

当 FE level design 没有显式常数时，pooled null 会先从 y 和 X 中投影掉共同常数，再拟合 slope，避免把共同均值误计入被检验的 fixed effects。分子和分母自由度由有效 nested-model rank 推导，而不是硬编码 effect count。

如果 fixed-effects fit 精确拟合（`RSS_FE` 在数值上为 0），而 restricted pooled RSS 实质为正，则经典极限结果为 `F=inf`、`p=0`；若 pooled 与 FE RSS 都为 0，则比值不定，返回结构化 inapplicable。以上 zero/nesting 判定使用相对于 RSS 量级的 tolerance，不使用固定的 unit-size floor。

即使 FE 对象的系数推断采用 robust 或 clustered covariance，这个 pooling F 仍是 classical/homoskedastic test。

### 个体随机效应 Breusch-Pagan LM

向 `PooledOLS.fit()` 提供 `entity_ids`：

```python
pooled = PooledOLS().fit(X, y, entity_ids=entity_ids)
result = pooled.breusch_pagan_lm_test()
# 或
result = breusch_pagan_lm_test(pooled)
```

这里是 **panel error-components Breusch-Pagan LM test**，不是横截面异方差的 Breusch-Pagan test。Stage B 实现 one-way entity 版本，并包含 `plm::plmtest(type="bp", effect="individual")` 使用的 Baltagi-Li incomplete/unbalanced-panel 公式。

原假设是 entity random-effect variance 为 0。至少需要两个 entity、正的 pooled RSS，并且至少一个 entity 有重复观测。没有 `entity_ids` 时返回结构化 inapplicable 结果，不会根据行顺序猜测 panel structure。

### Classical Hausman：FE 与 RE

```python
fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
    X, y, entity_ids=entity_ids
)
re = RandomEffects().fit(X, y, entity_ids=entity_ids)

result = fe.hausman_test(re)
# 或
result = hausman_test(fe, re)
```

Stage B 实现 one-way entity FE-versus-RE 的经典二次型 Hausman：

$$
H=(\hat\beta_{FE}-\hat\beta_{RE})^\top
  (V_{FE}-V_{RE})^{-1}
  (\hat\beta_{FE}-\hat\beta_{RE}).
$$

适用性规则是显式的：

- FE 必须只有 one-way entity effects；
- FE coefficient covariance 必须是 classical/nonrobust；
- RE coefficient covariance 同样必须是 classical/nonrobust；Stage-C robust/HC/cluster/DK RE fit 不属于该 classical test 的输入；
- FE 与 RE 必须来自同一个对齐后的 y/entity 样本和同一个 canonical slope design；
- RE 可以比 FE 多一个显式常数，因为 entity FE 已吸收共同 intercept；该常数不会进入 Hausman coefficient vector；
- 行/样本一致性使用所有对齐 float64 **slope-X/y** 值的 collision-resistant SHA-256 digest，并结合 entity-code signature 与 canonical feature metadata，而不是只比较 shape 或低阶 moments；
- canonical slope position 会保留到各模型原始 coefficient/covariance position 的映射，因此 RE 的第 0 列 intercept 不会把 `x1`、`x2` 错配到错误系数；
- covariance difference 若实质上 indefinite，则返回 inapplicable，不通过 eigenvalue clipping 强行制造统计量。

数组输入下，在移除 RE-only constant 后 slope 会重新按 `x1`、`x2`、... 做 canonical 编号；named/formula design 则保留 slope 名称。Hausman covariance rank 与 identified-range tolerance 按 covariance/coefficient 自身量级缩放，所以改变 outcome 单位不会改变同一数学问题的 applicability。

GPU 拟合下，canonical slope full-content digest 仅为了 hashing 而通过有界 chunk 分批复制到 host。只有可进入 Stage-B Hausman 的 one-way entity nonrobust FE 与 `RandomEffects` 保留该 identity；robust/clustered、time-only、two-way FE 在 identity compare 前就会被判不适用，因此不承担完整 X/y hashing 开销。拟合对象只保存 digest/index metadata，不保留第二份完整 CPU design copy。统计估计、covariance 构造与 fit-statistic reduction 仍在所选数值 backend 上完成。

若 covariance difference 为 positive semidefinite 但 rank deficient，statgpu 提供显式记录的 generalized-inverse extension：只在 coefficient difference 位于 identified range 内时计算，并使用 numerical rank 作为 chi-square df。metadata 会记录 `used_pinv=True` 和 `singular PSD generalized-inverse Hausman`。Stage B 不实现 robust auxiliary-regression Hausman。

## 参数与 fit 签名

### `PanelOLS`

```python
PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",  # 同时支持 robust/hc0/hc1/hc2/hc3/clustered/dk
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
    device="auto",
)
```

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=time_ids, cluster=cluster)
```

formula 输入也可使用已有 pipe syntax，例如 `"y ~ x1 + x2 | entity"`。formula 的 missing-row filtering 会同步对齐 observation-level side arrays。

### `PooledOLS`

```python
PooledOLS(
    cov_type="nonrobust",  # 包含 HC、clustered、legacy hac 与 dk
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
    device="auto",
)
```

```python
model.fit(
    X,
    y,
    cluster=None,
    time_index=None,
    entity_ids=None,
)
```

clustered inference 需要 `cluster`。`time_index` 定义 HAC 的稳定时间排序。`entity_ids` 是可选项，不改变系数；提供后会启用标准化 within/between R² 与 panel BP-LM。`PooledOLS` 与 `BetweenOLS` 始终包含截距，因此显式移除截距的公式（例如 `0 +` 或 `-1`）会直接报错，而不会被静默忽略。

### 其他模型

```python
RandomEffects(
    device="auto",
    cov_type="nonrobust",  # robust/hc0/hc1/hc2/hc3/clustered/dk
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
)
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")  # 支持 HC0/1/2/3
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")  # 支持 HC0/1/2/3
FamaMacBeth(
    cov_type="newey-west",
    bandwidth=None,
    alpha=0.05,
    min_obs_per_period=1,
    device="auto",
)
```

`FamaMacBeth.fit(..., entity_ids=None)` 中的可选 entity IDs 只用于 Stage-B within/between R²；beta-series estimation 与 covariance path 保持不变。

## CPU 与 GPU 示例

```python
import numpy as np
from statgpu.panel import PanelOLS, PooledOLS, RandomEffects

n_entities, n_times = 50, 10
n = n_entities * n_times
entity_ids = np.repeat(np.arange(n_entities), n_times)
time_ids = np.tile(np.arange(n_times), n_entities)
X = np.random.default_rng(0).normal(size=(n, 3))
y = X @ np.array([1.0, -0.5, 0.3]) + np.random.default_rng(1).normal(size=n) * 0.1

fe = PanelOLS(entity_effects=True, device="cpu").fit(
    X, y, entity_ids=entity_ids
)
print(fe.fit_statistics_.rsquared_within)
print(fe.pooling_f_test())

pooled = PooledOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
print(pooled.breusch_pagan_lm_test())

re = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
print(fe.hausman_test(re))
```

CuPy CUDA 使用 `device="cuda"`；Torch CUDA 使用 CUDA tensor 并设置 `device="torch"`。Stage-B 的统计变换与 sufficient-statistic accumulation 跟随所选数值 backend；formula/label metadata、最终 scalar 与小型 covariance matrix 使用 CPU metadata boundary；Hausman-compatible one-way FE/RE 拟合还会仅为 collision-resistant identity hashing 对 canonical slope X/y 做有界分块 host copy。

## 输出

常见拟合属性包括：

- `coef_`；
- 在系数空间可识别时的 `bse_`、`tvalues_`、`pvalues_`、`conf_int_`；
- 历史兼容的 `rsquared` 或 `rsquared_within`；
- 标准化的 `fit_statistics_`；
- `nobs`、`df_resid` 与有效 rank；
- `FamaMacBeth` 的 `betas_`、`cov_params_` 与 `n_periods`。

`PanelTestResult` 包含 `statistic`、`pvalue`、`distribution`、`df`、`null`、`alternative`、`applicable`、`reason` 和 `metadata`。

## Formula 与元数据边界

formula evaluation 可能因为 missing value 删除行。entity、time、cluster 等 side array 会与保留行同步对齐。字符串和分类标签在 CPU 上 factorize；数值变换与 sufficient-statistic calculation 继续留在所选 backend。对 Hausman-compatible 拟合，会先排除 RE-only 显式常数，再把 canonical slope X/y 以有界 chunk 复制到 host 计算 SHA-256 identity；随后只保存 digest、原始 coefficient-index 映射、feature/entity metadata 与小型 covariance matrix，不保存第二份完整 CPU design copy。

## 验证

Stage A / PR #119 建立共享 Panel framework，并在 Tesla P100 上通过 10 个 CuPy + 10 个 Torch exact-head physical cases。

Stage B 增加 maintained analytic/fitted-model regression tests、formula/missing-row alignment、Python 3.9 + Torch 2.0 CPU parity，以及可执行的 `linearmodels==7.0` external-definition gate。physical runner 每个 backend 包含 17 个 estimator cases 与 4 个 Hausman diagnostic cases，其中包括 balanced/unbalanced 显式常数 RandomEffects，以及 FE 吸收 intercept、RE 显式估计 intercept 的 Hausman 参数化。最终 promotion 要求 `dev/benchmarks/validate_panel_stage_b_gpu.py` 在 clean exact commit 上同时通过 CuPy 与 Torch CUDA；该 runner 是 correctness/provenance gate，而不是性能 benchmark。另有独立 physical benchmark 用于测量 Hausman-compatible FE/RE 上剩余 full-content identity 开销。

## 参考文献

- Hausman, J. A. (1978). Specification Tests in Econometrics.
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange Multiplier Test and its Applications to Model Specification in Econometrics.
- Baltagi, B. H., & Li, Q. (1990). A Lagrange Multiplier Test for the Error Components Model with Incomplete Panels.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering.
