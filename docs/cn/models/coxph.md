# CoxPH

> 语言：中文<br>
> 最后更新：2026-07-26<br>
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

## 风险集与 ties 方法

对第 `i` 行的起始时间 `a_i`、终止时间 `b_i`、事件指示 `delta_i` 与分层
`s_i`，时刻 `t` 的风险集为

$$
R_s(t)=\{i : a_i < t \le b_i,\ s_i=s\}.
$$

`ties="breslow"` 和 `ties="efron"` 使用对应的并列事件部分似然；
`ties="exact"` 通过 elementary-symmetric 动态规划计算 Exact 分母。
delayed entry、strata、Exact ties、L2 惩罚拟合与 GPU 稳健推断共用同一套
计数过程风险集引擎，因此三个后端遵循一致的 `(start, stop]` 约定。

对于普通 right-censored、单个 stratum 的 Exact 拟合，风险集具有嵌套结构。
StatGPU 按 stop time 降序排列样本，并在 NumPy、CuPy、Torch 上让所有失败组复用
同一个 elementary-symmetric 前缀动态规划，避免随失败组数量重复扫描风险集。
失败分子改用按事件时间排序的分段前缀和，不再构造 `失败组 × 样本` 密集掩码。
前缀工作区默认上限为 512 MiB，由 `STATGPU_EXACT_NESTED_MAX_BYTES` 控制，且在
分配前完成检查。

在 Torch CUDA 上，PyTorch 2.0 对长轴执行多维 `cumsum(dim=0)` 时，可能成为这条
线性前缀 DP 的主要耗时。当样本数至少为 2,048、尾部矩通道数不超过 64 时，
StatGPU 会将每个通道连续布局，分别执行高效的一维 CUDA 扫描，再在设备上拼回原
形状。`STATGPU_TORCH_EXACT_SCAN_MIN_ROWS` 与
`STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS` 可配置这两个保守门禁。CPU、小样本和宽
张量保留 Torch 原生多维扫描。额外的转置与输出工作区也计入 nested 工作区检查：
若基础 DP 能容纳而通道扫描额外空间不足，则继续使用 nested 算法的原生 Torch
扫描，不会回退到开销更高的通用 Exact 路径。

delayed entry、多个 strata、构造 score residuals、前缀工作区超限，或触发保守的
数值范围门禁时，会使用原有的 normalized Exact 实现。CuPy/Torch 可先使用失败组
批量路径，其独立的 512 MiB 上限由 `STATGPU_EXACT_BATCH_MAX_BYTES` 控制；批量
工作区超限时在同一后端使用逐组内存受限路径。这些都是显式算法回退，不会隐式
回退到 CPU。

完整拟合的推断阶段还需要构造 Breslow baseline hazard。对于普通右删失行，
StatGPU 现在在每个 stratum 内按 stop time 降序排列，并通过一次 log-risk 前缀
得到所有风险分母：NumPy 使用 `logaddexp.accumulate`，Torch 使用
`logcumsumexp`，CuPy 在保守的 predictor-range 门禁内使用平移后的指数累积和。
极端 CuPy predictor 与 delayed-entry 行继续使用数值稳定的后端原生逐失败组实现。
这移除了普通右删失常用路径中原先的 `失败组 × 样本` 风险掩码扫描。

当 `penalty > 0` 时，优化目标为部分对数似然减去
`penalty * ||beta||^2`。惩罚估计不是无约束最大似然估计，因此不会把普通
likelihood-ratio 统计量与信息准则作为经典无惩罚结果报告。

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
- `n_iter_`；
- `final_kkt_inf_`；
- `final_kkt_normalized_`。

likelihood、gradient、Hessian、协方差、baseline hazard 与公开收敛状态均从
最终系数向量重新计算。

## 协方差与推断

| `cov_type` | 含义 |
|---|---|
| `"nonrobust"` | 基于观测信息矩阵的模型协方差 |
| `"hc0"` | score-sandwich 协方差 |
| `"hc1"` | 带有限独立单元修正的 score-sandwich 协方差 |
| `"cluster"` | 聚类稳健协方差；在 `fit` 时传入 `cluster=` |

Breslow 与 Efron 的 strict 稳健推断使用 statgpu 内部的精确计数过程 score
residual，不依赖 statsmodels。同一受试者的重复行会先按 `subject_id` 汇总再
形成 HC0/HC1 meat；cluster 协方差按 `cluster` 汇总。

`inference_mode="strict"` 是默认值。`inference_mode="approx"` 仅用于显式选择
旧路径的 event-row Efron sandwich 近似。近似推断会写入公开 provenance 字段，
不会被静默启用。

Exact ties 当前只支持模型协方差（`cov_type="nonrobust"`）。若在
`ties="exact"` 下请求 HC0、HC1 或 cluster 推断，会抛出
`NotImplementedError`。当 `compute_inference=False` 时，可以保留稳健
`cov_type` 标签，但不会计算协方差。

推断来源通过以下字段公开：

- `inference_method_`；
- `inference_backend_`；
- `inference_approximate_`；
- `inference_fallback_reason_`；
- `full_host_transfer_performed_`。

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
| `inference_mode` | `"strict"` | `"strict"` 或显式 `"approx"` |
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
cv_model = CoxPHCV(
    penalties=[0.0, 0.01, 0.1],
    cv=5,
    ties="efron",
    device="cpu",
).fit(
    X_rows,
    stop,
    event,
    start=start,
    strata=clinic,
    subject_id=patient_id,
)
```

## 预测与评分

对数组输入，`predict`、`predict_risk_score`、`predict_hazard_ratio`、
`predict_survival` 与 `score` 都在拟合后端执行。分层生存预测要求每个预测行
提供一个训练时已知的 stratum 标签。生存曲线在 log-domain 中累计 baseline，
以提高数值稳定性。Formula 拟合模型会在预测前应用已保存的设计矩阵转换。

## 输出

- 参数：`coef_`、`hazard_ratios_`；
- 推断：启用时的 `_bse`、`_zvalues`、`_pvalues`、`_conf_int`；
- 诊断：定义时的 `log_likelihood`、`aic`、`bic`、`concordance_index`；
- 收敛：`converged_`、`termination_reason_`、`n_iter_`、
  `final_kkt_inf_`、`final_kkt_normalized_`；
- provenance：`inference_method_`、`inference_backend_`、
  `inference_approximate_`、`inference_fallback_reason_`、
  `full_host_transfer_performed_`。

## 验证

截至 2026-07-26 的 PR #80 review 已通过本地 NumPy quick gate，覆盖普通
heavy ties、delayed entry、Exact ties、分层 start-stop、推断、
subject-grouped CV，以及模型可比场景下的 statsmodels 对齐；结果 schema 通过且
没有本地 gate failure。随后通过 Paramiko 将准确的 reviewed source 放入远程
隔离 worktree，并在 Tesla P100-SXM2-16GB 的 `myconda` 环境中验证。首次真实
GPU 执行暴露了 11 个可修复的后端/测试边界及 scikit-learn 1.2.2 兼容问题；
review-fix 后，原失败节点与相邻契约共 **15 项全部通过**。最终 nested-Exact
源码的 13 文件真实 GPU 完整矩阵为 **392 passed、2 个预期 skip、0 failed**。
远程 quick 与 full artifact 均报告 `validation_tier="remote-full"`、
`schema_status="ok"`、零 gate failure。新增回归测试在 NumPy/Torch 上比较前缀
路径与强制 normalized fallback，真实 GPU target 还覆盖 CuPy、delayed-entry
批量 fallback、两个 GPU 后端的 baseline parity、极端 predictor 的 CuPy 稳定
回退、Torch 通道扫描与原生扫描的一致性，以及通道扫描内存门禁。本地 13 文件
矩阵为 **297 passed、97 skipped、0 failed**；skip 来自可选 GPU/R 可用性分支。

随后使用 R 4.4.1、survival 3.8.9 的
`survival::coxph(ties="exact")` 对同一源码做外部 Exact 对齐。bounded scaling 以及
独立的 right-censored、delayed-entry、strata、delayed-entry+strata 场景中，
NumPy/CuPy/Torch 全部收敛且 artifact 为零 gate failure。相对 R 的最大系数、
exact partial log-likelihood、model-based covariance 差异分别为 `1.30e-09`、
`5.12e-09`、`5.01e-12`。

性能结论仍依赖风险集形状与后端。在 Tesla P100 的 bounded-tie right-censored
工作负载（`p=4`、最大 tie size 为 8）中，`n=1920` 的
R/NumPy/CuPy/Torch 完整拟合中位时间为 0.0460/0.0354/0.0838/0.0571 秒；
这个小规模下 GPU 仍受 kernel launch 开销限制。`n=15,360` 时四者为
0.295/0.273/0.0949/0.0558 秒，`n=61,440` 时为
1.323/1.465/0.1114/0.0662 秒，`n=122,880` 时为
2.691/3.043/0.1430/0.1000 秒。最大规模下，Torch 通道扫描相对先前多维原生扫描
结果提速 30.32 倍；Torch 比 R 快 26.92 倍、比 NumPy 快 30.44 倍、比 CuPy
快 1.43 倍。两个 GPU 后端都在实测 `n=15,360` 超过 R。

`n=61,440` 的分阶段 profiling 将 baseline 构造确定为剩余的完整拟合热点。
优化前 NumPy/CuPy/Torch 的 baseline 阶段分别为 6.847/5.988/3.328 秒，
现在为 0.0202/0.00701/0.00265 秒，同时保持 R 与跨后端精度。在另一个
`n=160` delayed-entry 场景中（按设计保留 normalized fallback），R 与
NumPy/CuPy/Torch 分别为 57.031/0.182/0.594/0.345 秒。这些时间只证明实测
形状，不能当作通用 crossover。StatGPU 计时包含输入转换和推断；R 计时包含
`coxph` 调用及推断，但排除进程启动、包加载和 CSV 解析。

相关验证入口：

- `dev/tests/test_survival_risk_sets.py`；
- `dev/tests/test_cox_phase1_completion.py`；
- `dev/tests/test_cox_cv.py`；
- `dev/benchmarks/benchmark_survival_completion.py`；
- `dev/benchmarks/benchmark_exact_ties_scaling.py`（写入
  `results/exact_ties_scaling.json`）。

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
