# Cox 比例风险模型

> 语言：中文
>
> 最后更新：2026-07-12
>
> 页面定位：模型文档
>
> 切换：[English](../../en/models/coxph.md)

## 概览

statgpu 提供三层 Cox 比例风险模型接口：

| 接口 | 用途 | 当前边界 |
|---|---|---|
| `statgpu.survival.CoxPH` | 无惩罚或固定 L2 惩罚的估计、推断、基线风险与预测 | 支持 Breslow、Efron、Exact ties |
| `statgpu.survival.CoxPHCV` | 用 K 折部分似然选择 L2 强度并在全量数据上重拟合 | 支持与 `CoxPH` 相同的 ties、计数过程和三后端轴 |
| `statgpu.linear_model.PenalizedCoxPHModel` | L1、L2、Elastic Net、SCAD、MCP 惩罚估计 | 仅估计；无截距、无推断和基线风险 |

三者均支持 NumPy CPU、CuPy CUDA 和 Torch CUDA。显式选择 `device="cuda"` 或
`device="torch"` 时，后端不可用或执行失败会直接报错，不会静默回退 CPU。

## 数据与风险集

对第 $s$ 个分层，计数过程数据在事件时刻 $t$ 的风险集定义为

$$
R_s(t)=\{j:\operatorname{strata}_j=s,\;\operatorname{start}_j<t\leq
\operatorname{stop}_j\}.
$$

因此区间采用 **$(\text{start},\text{stop}]$** 约定。`time` 表示 `stop`；
`entry` 与 `start` 是同一含义的两个入口，二者不能同时传入。未传入时等价于普通
右删失数据。`strata` 为每个分层建立独立风险集和基线风险，但所有分层共享系数。
对同一受试者的多行时变协变量数据，传入 `subject_id` 可让 C-index 排除受试者内部配对，
并在 HC0/HC1 稳健推断中按受试者聚合行级得分。

公式接口接受以下两种响应：

```python
from statgpu.survival import CoxPH

CoxPH().fit(formula="Surv(time, event) ~ age + treatment", data=df)
CoxPH().fit(formula="Surv(start, stop, event) ~ age + treatment", data=df)
```

拟合时 Patsy 会删除生存响应或设计项中含缺失值的行，并将外部逐行数组同步到保留行。
预测和评分不会静默复用这种删行行为：公式协变量含缺失值时会抛出 `ValueError`，从而
保证每个输入行都对应一个输出。

## 部分似然与 ties

模型为

$$
h_s(t\mid x)=h_{0s}(t)\exp(x^\top\beta).
$$

`ties` 支持：

- `"breslow"`：Breslow 近似；
- `"efron"`：Efron 近似，通常更适合存在较多并列事件的普通规模数据；
- `"exact"`：Exact 部分似然，通过基本对称多项式动态规划计算并列事件组合。

普通右删失的 Breslow/Efron 路径使用专用向量化实现；`start`/`entry`、`strata`、
`subject_id` 或 Exact 会进入共享的后端原生计数过程实现。Exact 的成本随风险集大小和
同一时刻事件数增长，当前更适合并列事件数受控的场景。

## 推断、基线风险与预测

`CoxPH(compute_inference=True)` 支持：

- `cov_type="nonrobust"`：观测信息矩阵协方差；
- `cov_type="hc0"`、`"hc1"`：sandwich 稳健协方差；
- `cov_type="cluster"`：聚类稳健协方差，需要 `fit(..., cluster=...)`。

Breslow/Efron、普通右删失和 start-stop/分层数据均可在 NumPy、CuPy、Torch 上执行这些
推断路径。**当 `compute_inference=True` 时，Exact 只支持
`cov_type="nonrobust"`**；Exact 与 HC0、HC1 或 cluster 组合会抛出
`NotImplementedError`，不会改用近似协方差。`compute_inference=False` 时
`cov_type` 不参与计算，因此不会阻止纯估计拟合。

系数无论使用 Breslow、Efron 还是 Exact 拟合，基线风险、累计基线风险和
生存概率都统一使用常规 **Breslow 基线估计量**；分层模型为每个 stratum 单独保存基线。
这些结果在 `compute_inference=True` 时计算；分层模型调用 `predict_survival` 时需为预测行
提供 `strata`。

当 `penalty > 0` 时，协方差来自惩罚后的观测曲率，并以给定 penalty 为条件；
`CoxPHCV` 最终重拟合给出的标准误和区间属于未经选择校正的朴素 post-selection
推断。经典 likelihood-ratio、AIC 与 BIC 因此只对无惩罚拟合报告。

主要输出包括：

- `coef_`、`hazard_ratios_`；
- `_bse`、`_zvalues`、`_pvalues`、`_conf_int`（启用推断时）；
- `log_likelihood`、`concordance_index`，以及无惩罚拟合的 `aic`、`bic`；
- 基线风险、累计基线风险和 `predict_survival(...)`。

## `CoxPH` 示例

### 分层 start-stop 与受试者级稳健推断

```python
from statgpu.survival import CoxPH

model = CoxPH(
    ties="efron",
    cov_type="hc1",
    device="cuda",
    compute_inference=True,
)
model.fit(
    X,
    stop,
    event,
    start=start,
    strata=strata,
    subject_id=subject_id,
)

survival, eval_times = model.predict_survival(
    X_new,
    times=[1.0, 2.0, 5.0],
    strata=strata_new,
)
```

### Exact ties

```python
from statgpu.survival import CoxPH

model = CoxPH(
    ties="exact",
    cov_type="nonrobust",  # Exact 当前仅支持非稳健协方差
    device="torch",
)
model.fit(X_torch, time_torch, event_torch)
```

## `CoxPHCV`：L2 交叉验证

`CoxPHCV` 对候选 `penalties` 计算 held-out Cox 部分似然，选择最佳 L2 强度后在全量数据
上用同一后端重拟合 `CoxPH`。评分遵循所选 Breslow/Efron/Exact ties、
$(\text{start},\text{stop}]$ 风险集和分层边界。

当提供 `subject_id` 时，自动 K 折按受试者分组，避免同一受试者同时出现在训练集与验证集；
自定义 `cv_splits` 若产生受试者泄漏会直接报错。`cv_results_` 包含每个 penalty/fold 的
部分似然、收敛状态、迭代数、停止原因和 fold 元数据。拟合失败会传播或记录为失败诊断，
不会把 CPU 或另一个 GPU 后端的结果伪装成当前设备结果。
fold 构造与诊断记录由主机编排；显式 CuPy/Torch 模式下，候选拟合和 held-out
部分似然评分都保留在所请求后端，`cv_results_` 分别记录拟合、评分和编排设备。

```python
from statgpu.survival import CoxPHCV

cv = CoxPHCV(
    penalties=[0.0, 1e-4, 1e-3, 1e-2],
    cv=5,
    ties="exact",
    device="cuda",
    cov_type="nonrobust",
    random_state=42,
)
cv.fit(
    X_cupy,
    stop_cupy,
    event_cupy,
    start=start_cupy,
    strata=strata_cupy,
    subject_id=subject_cupy,
)

print(cv.penalty_, cv.best_score_)
print(cv.cv_results_["converged_path"])
```

Exact 的最终重拟合也受“仅 `nonrobust` 推断”的限制。

## `PenalizedCoxPHModel`：稀疏与非凸惩罚

`PenalizedCoxPHModel` 当前公开并验证五类惩罚：`l1`、`l2`、`elasticnet`、`scad`、
`mcp`。求解使用 FISTA 家族；SCAD/MCP 使用 FISTA-LLA 的局部线性近似路径。

```python
import numpy as np
from statgpu.linear_model import PenalizedCoxPHModel

y_surv = np.column_stack([time, event])
model = PenalizedCoxPHModel(
    penalty="scad",
    alpha=0.05,
    ties="efron",
    fit_intercept=False,
    compute_inference=False,
    device="cuda",
)
model.fit(X, y_surv)
hazard_ratio = model.predict_hazard_ratio(X_new)

# 右删失公式支持分类变量、交互项、变换和 Patsy 的 NA 删除；
# 不可识别的截距会自动从设计矩阵中移除。
formula_model = PenalizedCoxPHModel(
    penalty="l2", alpha=0.05, ties="efron", device="cpu"
)
formula_model.fit(
    formula="Surv(time, event) ~ age * C(group) + np.log(marker)",
    data=frame,
)
```

当前限制必须显式考虑：

- Cox 部分似然不能识别截距，因此 `fit_intercept=True` 会报错；
- 该类仅提供惩罚估计、风险比和 C-index；`compute_inference=True` 会抛出
  `NotImplementedError`；
- 需要标准误、显著性检验、置信区间、基线风险或生存曲线时，使用无惩罚 `CoxPH`；
- 该惩罚接口接收形如 `[time, event]` 的二维响应，或右删失
  `Surv(time, event)` 公式；公式支持分类变量、交互项、变换和 NA 删除；
- 尚不提供 `CoxPH` 的 start-stop/strata/subject 公共接口，也不支持 Exact ties；
- 仅接受已公开验证的 L1、L2/Ridge、Elastic Net、SCAD、MCP 或无惩罚；可传入字符串
  或对应的内置 penalty 对象。对象参数会在每次拟合前重新校验，并支持 `sklearn.clone`；
  adaptive/group penalty 的 Cox 路径尚未验证，因此会显式报错；
- 非有限的 penalty、容差及非法迭代次数会在优化前显式报错。

## 性能与验证

2026-07-12 的两份可复现实验产物为：

- [`results/survival_completion_2026-07-12.json`](../../../results/survival_completion_2026-07-12.json)：quick 规模；
- [`results/survival_completion_full_2026-07-12.json`](../../../results/survival_completion_full_2026-07-12.json)：full 规模。

实验使用 NVIDIA RTX 5880 Ada Generation、Python 3.11.15、NumPy 2.4.6、
CuPy 14.1.1、Torch 2.8.0+cu128、float64；每个场景 1 次 warmup、2 次计时重复。
`fit` 计时包含优化、推断和基线估计，主机到设备传输单独计时。

full delayed-entry 场景（$n=2500,p=16$，Breslow）中，CuPy 和 Torch 相对 NumPy
分别为 **1.044×** 和 **1.374×**。但性能高度依赖风险集结构和问题规模：同一 full
产物中的 stratified start-stop 场景仅为 **0.241×** 和 **0.411×**；Exact 与普通
重 ties 场景也慢于 NumPy。quick delayed-entry 中 CuPy 为 0.647×、Torch 为 0.959×。
这两份产物未确定通用 crossover 规模，因此不能据此承诺所有生存分析工作负载都有 GPU
加速。

三后端在 delayed-entry、Exact、普通重 ties 和 stratified start-stop 场景均通过兼容性、
收敛和推断矩阵；后端间系数、标准误、对数部分似然和预测误差在所列容差内。Breslow/Efron
CPU 结果与 statsmodels PHReg 比较；Exact 由小规模暴力枚举测试验证，当前产物未调用
R `survival`。两份产物中的 stratified start-stop + subject-grouped `CoxPHCV` 在三后端
均选择相同 penalty，最终 refit 系数和标准误的最大后端差异小于 $10^{-16}$。

## 参考文献

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y. & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
