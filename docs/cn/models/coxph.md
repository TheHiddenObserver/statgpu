# CoxPH

> 语言：中文<br>
> 最后更新：2026-08-02<br>
> 页面定位：模型文档<br>
> 切换：[English](../../en/models/coxph.md)

## 概览

`CoxPH` 在 NumPy、CuPy CUDA 与 Torch CUDA 后端实现比例风险回归，支持
Breslow、Efron 与 Exact 三种 ties 处理，同时覆盖普通右删失、delayed entry、
计数过程 `(start, stop]` 行、独立 strata、时变协变量、稳健/聚类协方差，以及
通过 `CoxPHCV` 选择 L2 惩罚。

重要行为：

- 显式 `device="cuda"` 与 `device="torch"` 不会静默回退 CPU；
- `entry=` 与 `start=` 是互斥的别名；
- 某行在时刻 `t` 进入风险集，当且仅当 `start < t <= stop`，且其 stratum
  与事件所属 stratum 相同；
- `subject_id=` 标识同一受试者的重复行，用于 concordance、sandwich 聚合和
  保持受试者完整的 CV folds；
- `compute_inference=False` 仅执行估计，推断与 baseline-hazard 字段保持未设置。

## 导入

```python
from statgpu.survival import CoxPH, CoxPHCV
```

## CPU 与 GPU 示例

三个后端使用相同的统计输入，并在拟合后端返回预测数组。先运行一次以下确定性数据准备：

```python
import numpy as np

from statgpu.survival import CoxPH, CoxPHCV

rng = np.random.default_rng(20260730)
n = 256
X = rng.normal(size=(n, 3))
log_risk = X @ np.array([0.45, -0.30, 0.20])
event_time = rng.exponential(scale=np.exp(-log_risk))
censor_time = rng.exponential(scale=1.8, size=n)
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(np.float64)
```

NumPy / CPU：

```python
cpu_model = CoxPH(
    ties="efron",
    device="cpu",
    compute_inference=False,
).fit(X, time, event)
cpu_log_risk = cpu_model.predict_risk_score(X[:3])
```

CuPy / CUDA：

```python
import cupy as cp

X_cp = cp.asarray(X)
time_cp = cp.asarray(time)
event_cp = cp.asarray(event)
cupy_model = CoxPH(
    ties="efron",
    device="cuda",
    compute_inference=False,
).fit(X_cp, time_cp, event_cp)
cupy_log_risk = cupy_model.predict_risk_score(X_cp[:3])
```

Torch / CUDA：

```python
import torch

X_t = torch.as_tensor(X, dtype=torch.float64, device="cuda")
time_t = torch.as_tensor(time, dtype=torch.float64, device="cuda")
event_t = torch.as_tensor(event, dtype=torch.float64, device="cuda")
torch_model = CoxPH(
    ties="efron",
    device="torch",
    compute_inference=False,
).fit(X_t, time_t, event_t)
torch_log_risk = torch_model.predict_risk_score(X_t[:3])
```

当对应 package、CUDA runtime 或设备不可用时，显式 CUDA 请求会报错，不会静默转到
CPU。需要协方差、检验或生存曲线时，设置 `compute_inference=True`。

## 目标函数与估计方程

对第 `i` 行的起始时间 `a_i`、终止时间 `b_i`、事件指示 `delta_i` 与分层
`s_i`，时刻 `t` 的风险集为

$$
R_s(t)=\{i : a_i < t \le b_i,\ s_i=s\}.
$$

无并列失败时，分层 Cox 部分对数似然为

$$
\ell(\beta)=\sum_s\sum_{i:\delta_i=1,\ s_i=s}
\left[x_i^\top\beta-
\log\left\{\sum_{j\in R_s(b_i)}\exp(x_j^\top\beta)\right\}\right].
$$

Breslow、Efron 与 Exact 按各自定义替换并列事件分母，但沿用相同的
`(start, stop]` 风险集。令 `penalty=lambda`，StatGPU 最大化总和尺度的目标：

$$
Q_\lambda(\beta)=\ell(\beta)-\lambda\lVert\beta\rVert_2^2.
$$

记 `U(beta)` 为未惩罚 partial-likelihood score，拟合系数满足

$$
U_\lambda(\beta)=U(\beta)-2\lambda\beta=0.
$$

若 $J(\beta)=-\partial U(\beta)/\partial\beta$ 是未惩罚观测信息，则 penalized Newton
使用的导数为 $A(\beta)=J(\beta)+2\lambda I_p$。

## 风险集与 ties 方法

`ties="breslow"` 和 `ties="efron"` 使用对应的并列事件部分似然；
`ties="exact"` 通过 elementary-symmetric 动态规划计算 Exact 分母。
delayed entry、strata、Exact ties、L2 惩罚拟合与 GPU 稳健推断共用同一套
计数过程风险集引擎，因此三个后端遵循一致的 `(start, stop]` 约定。
公开 `CoxPH` 与 `CoxPHCV` estimator 会 factorize 一维标签：host 字符串/对象以及
有限的 CuPy/Torch 数值标签都会在内部编码为连续 int64 code。低层
counting-process primitive 不执行 factorize，因此要求数值 code 有限、整数值且
可由有符号 int64 表示。

对于普通 right-censored Exact 拟合，风险集在各 stratum 内具有嵌套结构。
StatGPU 先按 stratum、再按 stop time 降序排列样本，并在 NumPy、CuPy、Torch 上让
所有失败组复用同一个分段 elementary-symmetric 前缀动态规划，不再通过 Python
逐 stratum 循环，也避免随失败组数量重复扫描风险集。
失败分子改用后端原生的分组归约，不再构造 `失败组 × 样本` 密集掩码。
前缀工作区默认上限为 512 MiB，由 `STATGPU_EXACT_NESTED_MAX_BYTES` 控制，且在
分配前完成检查。

在 Torch CUDA 上，PyTorch 2.0 对长轴执行多维 `cumsum(dim=0)` 时，可能成为这条
线性前缀 DP 的主要耗时。当样本数至少为 2,048、尾部矩通道数不超过 64 时，
StatGPU 会将每个通道连续布局，分别执行高效的一维 CUDA 扫描，再在设备上拼回原
形状。`STATGPU_TORCH_EXACT_SCAN_MIN_ROWS` 与
`STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS` 可配置这两个保守门禁。CPU、小样本和宽
张量保留 Torch 原生多维扫描。`STATGPU_TORCH_EXACT_SCAN_STRATEGY` 可设为 `auto`、
`native` 或 `channelwise`；`auto` 仅在已有实测证据的 Torch 2.0 + Pascal/P100
组合启用分通道扫描，未经验证的 Torch/GPU 组合使用原生扫描。额外的转置与输出
工作区也计入 nested 工作区检查：
若基础 DP 能容纳而通道扫描额外空间不足，则继续使用 nested 算法的原生 Torch
扫描，不会回退到开销更高的通用 Exact 路径。

delayed entry 不满足嵌套前缀条件。当 strata 至少为 8 时，GPU 后端会先在一次后端
原生批量路径中处理所有合格失败组；更少的 GPU strata 与 NumPy 使用逐-stratum
batch，避免计算跨 stratum 的空掩码。独立的 512 MiB 上限由
`STATGPU_EXACT_BATCH_MAX_BYTES` 控制。全局批量工作区超限时先按 stratum 重试，
再使用逐组内存受限路径；构造 score residuals 或触发保守数值范围门禁时也保留
normalized 实现。这些都是显式算法回退，不会隐式回退到 CPU。

对于 Breslow/Efron delayed-entry objective，
`STATGPU_COX_GROUP_MAX_BYTES` 控制密集 failure-group 工作区，默认
512 MiB。如果单个 failure group 已超过上限，所选 GPU 后端会改用
数值稳定的多遍 row-streaming moment 计算，从而避免最小 batch size 为 1 时仍分配
不受限的 `O(n)` mask。

完整拟合的推断阶段还需要构造 Breslow baseline hazard。对于普通右删失行，
StatGPU 现在在每个 stratum 内按 stop time 降序排列，并通过一次 log-risk 前缀
得到所有风险分母：NumPy 使用 `logaddexp.accumulate`，Torch 使用
`logcumsumexp`，CuPy 在保守的 predictor-range 门禁内使用平移后的指数累积和。
极端 CuPy predictor 与 delayed-entry 行继续使用数值稳定的后端原生逐失败组实现。
这移除了普通右删失常用路径中原先的 `失败组 × 样本` 风险掩码扫描。

## Formula 接口

支持两种生存响应：

```python
CoxPH().fit(formula="Surv(time, event) ~ age + C(group)", data=df)
CoxPH().fit(
    formula="Surv(start, stop, event) ~ age + treatment",
    data=df,
    strata=df["clinic"],
    subject_id=df["patient_id"],
)
```

Formula 删除缺失行时，会同步对齐 `entry`/`start`、`cluster`、`strata` 与
`subject_id`。三列 `Surv(start, stop, event)` 已定义起始时间，不能再同时传入
`entry=` 或 `start=`。

## 优化与收敛

Newton 迭代使用 line search，并在最终参数处执行 KKT 检查。line search
失败不会更新系数，也不能报告收敛。公开拟合状态包括：

- `converged_`；
- `termination_reason_`；
- `optimization_stop_reason_`；
- `n_iter_`；
- `final_kkt_inf_`；
- `final_kkt_normalized_`。

likelihood、gradient、Hessian、协方差、baseline hazard 与公开收敛状态均从
最终系数向量重新计算。
`termination_reason_` 是解释后的用户级分类，只会是 `kkt_converged`、
`line_search_failed` 或 `stalled_with_large_kkt`。`optimization_stop_reason_`
保留底层 solver 的原始退出原因（包括 `max_iter`），warning 也报告该原始值；
因此预算耗尽可以审计，但不会被误当作独立的收敛证书。

## Penalty 缩放与惩罚推断

`penalty` 就是上述总和尺度 partial-likelihood 目标中的 `lambda`，不会除以样本数或
事件数；CoxPH 也没有需要惩罚的截距。因此，复制全部观测会令 likelihood 与 score
贡献加倍，却不会自动加倍用户提供的 penalty，从而改变有效正则强度。跨数据集或
样本规模比较时，应在目标抽样尺度下用 `CoxPHCV` 调参；复现采用平均 loss 的外部
软件时，需要显式换算其 penalty 口径，不能假设数值直接相同。

正 L2 惩罚下，记 `J` 为拟合系数处未加惩罚的 Cox 观测信息，
`A = J + 2 * penalty * I_p`，则固定惩罚强度的频率学派 plug-in 协方差为：

```text
A^-1 J A^-1
```

而不是 `A^-1`；后者更接近惩罚曲率或 Laplace-style 量，不能直接作为频率学派
抽样协方差发布。带惩罚的稳健推断同样使用 penalized bread，而 meat 仍来自未加
惩罚的聚合 score outer product。

因此 SE/z/p/CI 与 penalized Wald test 都以给定 penalty 为条件，目标是 penalized
estimating equation；它们不是无惩罚系数的 debiased inference，也不校正 shrinkage
bias 或交叉验证选择 penalty 带来的不确定性。`CoxPHCV` 从最终重拟合复制相同契约，
并明确报告 `penalty_selection_adjusted_=False`。沿用 `PenalizedGLM` 的结果命名，
正 penalty 拟合的 `inference_method_` 使用简洁的 `"m_estimation"`；bread、meat、
协方差口径、推断目标和条件化方式仍分别保留在 inference metadata 中。

带惩罚拟合会关闭经典 likelihood-ratio、score test 与 AIC/BIC，不会把惩罚估计
当作无约束最大似然结果报告。该契约与 `PenalizedCoxPHModel` 分开；后者的
L1/elastic-net/SCAD/MCP 接口仍是 estimation-only。

## 协方差与推断

| `cov_type` | 含义 |
|---|---|
| `"nonrobust"` | 模型协方差；无惩罚时为信息逆，有惩罚时为固定惩罚 sandwich |
| `"hc0"` | score-sandwich 协方差 |
| `"hc1"` | 带有限独立单元修正的 score-sandwich 协方差 |
| `"cluster"` | 聚类稳健协方差；在 `fit` 时传入 `cluster=` |

无惩罚拟合的 nonrobust 协方差仍是通常的观测信息逆；正 penalty 协方差遵循
上一节的专门契约。

Breslow 与 Efron 的 strict 稳健推断使用 statgpu 内部的精确计数过程 score
residual，不依赖 statsmodels。同一受试者的重复行会先按 `subject_id` 汇总再
形成 HC0/HC1 meat；cluster 协方差按 `cluster` 汇总。

稳健推断必须具有可识别的独立单元变异。按 subject 或 cluster 汇总后，HC0 与
cluster 协方差至少需要两个独立单元；HC1 还要求
`n_units > n_features`，因为其有限单元修正严格为
`n_units / (n_units - n_features)`。违反这些条件会抛出 `RuntimeError`，
不会把非正自由度分母替换为任意有限值。实质性负协方差对角线或非正稳健边际
方差同样会令 strict inference 失败，而不会发布零标准误与误导性的显著性结果。

边际方差为正并不保证稳健协方差在完整参数空间有效。StatGPU 会先用尺度感知容忍度
分类对称化后的 covariance spectrum：正定矩阵同时支持边际推断和 joint Wald；PSD
但秩亏的矩阵仍保留逐系数 robust SE/z/p/CI，同时设置
`wald_test_available_=False` 并记录 `wald_test_failure_reason_`，summary 显示
`Robust Wald test unavailable`，不会使用不稳定逆矩阵或打印裸 `nan`。若存在实质性
负特征值，该矩阵已不是合法 covariance estimator；strict inference 会抛出
`RuntimeError` 并清空本次 fit 状态，而不会仅凭正对角线发布边际推断。即使逐系数与
Wald 推断使用稳健协方差，likelihood-ratio 与 score test 仍是经典的 model-based
test；summary 会明确标注这一差异。

`inference_mode="strict"` 是默认值。为保持向后兼容，公开 API 仍接受
`inference_mode="approx"`，但统一 fit 路径会把它作为 compatibility-only alias，
继续计算精确的 counting-process score sandwich。因此成功拟合会报告
`inference_approximate_=False`，且没有 approximation fallback reason。

Exact ties 当前只支持模型协方差（`cov_type="nonrobust"`）。若在
`ties="exact"` 下请求 HC0、HC1 或 cluster 推断，会抛出
`NotImplementedError`。当 `compute_inference=False` 时，可以保留稳健
`cov_type` 标签，但不会计算协方差。

推断来源通过以下字段公开：

- `inference_method_`；
- `inference_backend_`；
- `inference_approximate_`；
- `inference_fallback_reason_`；
- `inference_target_`；
- `penalty_conditioning_`；
- `penalty_selection_adjusted_`；
- `wald_test_available_` 与 `wald_test_failure_reason_`；
- `full_host_transfer_performed_`。

对于 `CoxPHCV`，`full_host_transfer_performed_` 描述整个 fit，包括在 host
上组织的 fold 构造与 penalty 选择。`cv_full_host_transfer_performed_` 与
`final_refit_full_host_transfer_performed_` 分别标记 CV 与最终重拟合阶段是否
将至少一个完整的 device 训练组件移到 host；这包括排序后的 target，以及需要
保留的 entry、strata 或 subject 向量，即使设计矩阵仍留在 GPU 也会如实标记。
`orchestration_device_` 记录 CV 编排设备。
普通 GPU Breslow/Efron 预处理在选定后端完成排序，再把完整的已排序 time 与
event 向量复制到 host 以构建失败组元数据，因此会报告
`full_host_transfer_performed_=True`。普通 `CoxPHCV` 在一次完整 selector 调用中
为每个 fold 只准备一次元数据，并由所有 staged penalty pass 复用。
该复用仅在估算的保留 workspace 不超过
`STATGPU_COXPHCV_FOLD_CACHE_MAX_BYTES`（默认 512 MiB）时启用；超限时各 stage
会重新准备 fold，以避免无界的多 fold GPU 常驻内存。路由由
`fold_state_cache_enabled` 及估算/上限字段记录。
`selection_cache_hit`、
`requested_fit_device`、`fold_backend_preparation_count_this_call` 与
`candidate_target_host_transfer_count_this_call` 描述本次调用；
`selection_origin_device`、`candidate_preparation_origin_device` 和
`scoring_device` 保留选择结果的来源；`effective_device` 记录本次请求/最终 refit
设备。一次 target preparation 代表一整套
time/event 元数据准备；vector-transfer 计数记录实际发生的两条向量复制。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"`、`"efron"` 或 `"exact"` |
| `tol` | `1e-9` | Newton/KKT 收敛阈值 |
| `max_iter` | `100` | 最大迭代次数 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`、`"torch"` 或 `"auto"` |
| `compute_inference` | `True` | 计算协方差、检验与 baseline hazard |
| `compute_cindex` | `True` | 计算训练集 concordance |
| `cov_type` | `"nonrobust"` | `"nonrobust"`、`"hc0"`、`"hc1"` 或 `"cluster"` |
| `penalty` | `0.0` | 非负 L2 惩罚 |
| `inference_mode` | `"strict"` | `"strict"` 或兼容别名 `"approx"`；两者均执行精确推断 |
| `gpu_memory_cleanup` | `False` | 尝试释放 CuPy/Torch 缓存 |

## 支持矩阵

| 能力 | Breslow | Efron | Exact | NumPy | CuPy | Torch |
|---|---|---|---|---|---|---|
| 普通右删失 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| delayed entry / `(start, stop]` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 独立 `strata` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| 非负 L2 `penalty` | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| nonrobust 推断 | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| HC0 / HC1 / cluster 推断 | 支持 | 支持 | 未实现 | 支持 | 支持 | 支持 |
| 后端原生预测数组 | 支持 | 支持 | 支持 | NumPy | CuPy | Torch |

`predict_survival` 需要已拟合的 baseline hazard，因此需要生存曲线时应保留
`compute_inference=True`。risk-score 与 hazard-ratio 预测不依赖 baseline。

## 交叉验证

`CoxPHCV` 使用相同的 ties、start/entry、strata 与后端语义评估 L2 penalty
网格，再以最佳 penalty 重拟合 `CoxPH`。传入 `subject_id` 后，同一受试者的
全部行会被保留在同一自动生成 fold 中；若用户提供的 `cv_splits` 把同一
受试者泄漏到 train 与 validation，会被拒绝。`inference_mode` 与
`compute_inference` 会转发到最终 refit。

```python
cpu_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1],
    cv=5,
    device="cpu",
    compute_inference=False,
).fit(X, time, event)
```

同一 penalty 搜索也可直接使用前述 CuPy 或 Torch CUDA 数组：

```python
cupy_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1], cv=5, device="cuda",
    compute_inference=False,
).fit(X_cp, time_cp, event_cp)

torch_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1], cv=5, device="torch",
    compute_inference=False,
).fit(X_t, time_t, event_t)
```

### L1/L2/ElasticNet/SCAD/MCP 模型族交叉验证

上面的 `CoxPHCV` 是 canonical L2 Cox selector，并可按配置执行最终重拟合推断。
公开 penalized model family 则使用 `PenalizedGLM_CV` 的独立生存感知分支：

```python
from statgpu.linear_model import PenalizedGLM_CV

survival_y = np.column_stack([time, event])
penalized_cv = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="mcp",               # l1、l2、elasticnet、scad 或 mcp
    alpha_grid=[0.1, 0.03, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cpu",                # 也可用 "cuda" / "torch"
).fit(X, survival_y)
```

该分支始终保留二维 `(time, event)` target，禁止截距，用未惩罚的 held-out
partial likelihood 评分，并要求每个可评估 fold 都提供有限证据。若不存在满足
契约的 alpha，fit 会抛错，且不会发布已选 alpha 或拟合 estimator。最终重拟合为
`PenalizedCoxPHModel(compute_inference=False)`；不支持 post-selection 系数推断、
`two_stage`、sample weights 或字典 target。

## 预测与评分

对数组输入，`predict`、`predict_risk_score`、`predict_hazard_ratio`、
`predict_survival` 与 `score` 都在拟合后端执行。使用 `device="auto"` 拟合后，模型会固定实际的
`effective_device_`；后续修改全局 device 不会迁移既有模型的预测或评分后端。分层生存预测要求每个预测行
提供一个训练时已知的 stratum 标签；即使拟合时只有一个显式 stratum，也不能省略
标签，缺失或未知标签会抛出 `ValueError`。生存曲线在 log-domain 中累计 baseline，
以提高数值稳定性。Formula 拟合模型会在预测前应用已保存的设计矩阵转换。

`score()` 复用同一行标签编码器：传入的 strata 必须具有 `(n_samples,)` shape；
显式 stratified 模型只接受训练时已知标签，多 stratum 拟合在评分时必须提供标签。
scalar、二维、长度错误或未知标签都会在 backend concordance 计算前统一抛出
`ValueError`。

若某个已拟合 stratum 没有观察到任何 failure，其空 baseline-hazard state 是合法状态。
该 stratum 在任意时间的累计 baseline hazard 均为零，因此 `predict_survival()` 精确返回
1。显式 times、自动 times、混合 strata 预测行和 `CoxPHCV` 委托路径都遵守此契约；
存储的 time/hazard shape 不匹配仍属于非法状态。

`predict_risk_score()` 返回未取指数的 log-risk。canonical、CV 与 penalized
Cox 的 hazard-ratio 预测 API 共享严格的 float64 指数边界；canonical/CV 拟合后
`hazard_ratios_` 采用相同边界。会溢出为无穷或下溢为零的值，在 canonical/CV
fit 时抛出 `CoxFitNumericalError`，在预测时抛出 `FloatingPointError`，不会按 estimator
专属阈值静默截断。`PenalizedCoxPHModel` 也提供 `predict_risk_score()`，因此
极端但有限的 log-risk 仍可直接读取。

## 输出

- 参数：`coef_`、`hazard_ratios_`；
- 推断：启用时的 `_bse`、`_zvalues`、`_pvalues`、`_conf_int`；
- 诊断：定义时的 `log_likelihood`、`aic`、`bic`、`concordance_index`；
- 收敛：`converged_`、`termination_reason_`、`optimization_stop_reason_`、`n_iter_`、
  `final_kkt_inf_`、`final_kkt_normalized_`；
- provenance：`inference_method_`、`inference_backend_`、
  `inference_approximate_`、`inference_fallback_reason_`、
  `inference_target_`、`penalty_conditioning_`、`penalty_selection_adjusted_`、
  `full_host_transfer_performed_`。

`CoxPHCV` 还会公开 `cv_full_host_transfer_performed_`、
`final_refit_full_host_transfer_performed_` 与 `orchestration_device_`，避免数据移动审计
将 host CV 选择与最终重拟合混淆。
`cv_results_` 会区分 selection 来源字段（`scoring_device`、
`selection_origin_device`、`candidate_preparation_origin_device` 与总准备次数）和本次调用字段（`selection_cache_hit`、
`requested_fit_device`、`effective_device` 与 `*_this_call` 次数）。Cache 命中时，本次 fold 准备和
target 传输次数均为零，同时不会改写 selection 来源设备。
若有限输入的候选返回非有限系数或 likelihood，`CoxPH` 会抛出
`CoxFitNumericalError`（`FloatingPointError` 子类）；`CoxPHCV` 只排除这类
候选，输入、allocator、CUDA 与非预期 runtime 错误仍原样传播。

## 外部验证与可复现性

维护的 R 基线使用 R 4.4.1 与 `survival` 3.8.9，并对齐 ties、Newton
`max_iter=80` 和 `tol=1e-8`。在 `n=3000`、`p=10` 的 Breslow/Efron 比较中，
HC1 使用 3,000 个独立单元，cluster 使用 120 个单元。StatGPU 相对 R 的最大
系数/SE/p-value 差异：HC1 为 `5.55e-16`/`1.39e-16`/`8.00e-19`，cluster 为
`5.55e-16`/`1.32e-16`/`2.22e-16`。statsmodels 不支持的协方差模式会明确记录为
unsupported，不会换名后充当外部证据。

机器可读 R 对齐产物：

- `results/benchmark_frontend_sources/coxph_robust_inference_breslow_pr80_20260729_schema11.json`；
- `results/benchmark_frontend_sources/coxph_robust_inference_efron_pr80_20260729_schema11.json`。

这些是绑定精确源码和特定 shape 的比较，不是普遍精度或性能保证。Exact ties 与
性能结论仍绑定到 `dev/reviews/pr80_review_fix.md` 中列出的专用产物。

### 精确源码物理 GPU 证据

物理 GPU 证据固定到精确 source commit，后续代码或文档变更不会自动继承更宽的
验证声明。

| 字段 | 当前可审计证据 |
|---|---|
| Source commit | `d688f760d8a0678c3c52c657a50178dad1b5ab3d` |
| Artifact | `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260802_schema16.json` |
| Artifact SHA-256 | `f0b47df704d2a0895cd1d66019c8676ff8a525d0f85e827d90ba816ad02b4837` |
| Schema / tier | `16` / `remote-full` |
| 硬件 | Tesla P100-SXM2-16GB |
| 软件 | Python 3.9.16、NumPy 1.24.2、CuPy 13.6.0、Torch 2.0.0+cu117 |
| Structured GPU cases | CuPy 14/14；Torch 14/14 |
| 定向测试 | 516 passed，7 个预期 warning |
| 源码审计 | `source_clean=true`；记录的 43/43 个 Git-blob hash 全部匹配 |
| Gate failures | `[]` |

schema-16 保留 schema-15 的预测/评分、CV fold 准备、prepared state、packed target、
数值边界、workspace、concordance、稳健推断、固定 penalty 推断、共享 strata 评分、
无事件 stratum 和 canonical validator 门禁；并在两个 GPU 后端新增 L1、L2、
ElasticNet、SCAD、MCP 的生存感知 penalized-Cox CV，覆盖完整有限 fold 证据、
selected-alpha、无截距、direct final-refit coefficient parity，以及全局设备改变后的拟合
后端固定。定向矩阵还执行 sklearn <=1.2 的 `CompositePenalty` 构造器对象身份回归。

该 artifact 不是新的性能 crossover benchmark，也不是新的 R 外部对齐；这些结论仍
分别绑定到专用 artifact，详细历史保留在 `dev/reviews/pr80_review_fix.md`。上述 source
commit 之后的运行时或维护测试变更必须刷新自己的精确源码证据，才能声明获得相同的
物理 GPU 覆盖。

## FAQ 与常见失败模式

| 现象 | 含义与处理 |
|---|---|
| 显式 `device="cuda"` 或 `device="torch"` 失败 | 对应 package、CUDA runtime 或设备不可用。安装兼容后端或改用 `device="cpu"`；StatGPU 不会静默回退。 |
| `predict_survival()` 提示 baseline 不可用 | 使用 `compute_inference=True` 重新拟合；risk-score 与 hazard-ratio 预测不需要 baseline。 |
| 分层生存预测拒绝标签 | 只要拟合时显式分层，每个预测行都必须提供一个训练时已知的 stratum，shape 为 `(n_samples,)`；训练时只有一个 stratum 也不能省略。 |
| 分层评分拒绝标签 | 多 stratum 拟合必须逐行提供已知标签；单 stratum 拟合可省略标签，但一旦提供，仍必须具有 `(n_samples,)` shape 且属于训练标签。 |
| 已知 stratum 的生存率恒为 1 | 该拟合 stratum 没有观察到 failure，累计 baseline hazard 恒为零；这是合法拟合状态，不是 baseline 数据缺失。 |
| `HC1 covariance requires n_units > n_features` | 增加独立 subject/cluster、减少特征，或采用研究设计能够支持的协方差契约。 |
| 稳健协方差要求至少两个独立单元 | 单 subject/cluster 无法估计单元间变异；可用 `compute_inference=False` 仅执行估计。 |
| observed information singular | 检查共线性、常量列、separation/saturation 与事件支持；减少设计或使用有明确依据的 L2 penalty。 |
| hazard-ratio 预测抛出 `FloatingPointError` | `exp(X @ coef_)` 超出有限 float64 范围。检查 `predict_risk_score()`、缩放特征并检查外推。 |
| `converged_` 为 false | 检查 `optimization_stop_reason_`、`final_kkt_inf_` 与 `final_kkt_normalized_`；单纯增加 `max_iter` 不能修复 line-search 失败或病态设计。 |
| Exact ties 很慢或触发 workspace gate | Exact likelihood 对最大并列事件组具有组合复杂度；科学上允许时使用 Breslow/Efron，或减小最大 Exact tie block。 |
| `score()` 返回 `0.5` | 数据中不存在 permissible concordance pair；`0.5` 是文档化的中性返回值。 |

## 限制

- Exact ties 尚不支持 robust/cluster 协方差；
- Exact ties 使用组合动态规划，适合规模适中的并列事件组，不适合无限制的大型 tie block；
- 尚未实现 frailty/random-effect 项；
- 可选 `torch.compile` 加速要求兼容 Triton 的硬件，不属于可移植 correctness 契约。

## 参考文献

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
- R survival 文档：[`coxph`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html)。
