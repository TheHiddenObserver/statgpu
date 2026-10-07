# CoxPH 设备与 CV 诊断

> 语言：中文  
> 最后更新：2026-10-07  
> 本页：进阶 API 参考  
> 切换：[English](../../en/reference/coxph-diagnostics.md)

以下字段用于检查已完成的 `CoxPH` 或 `CoxPHCV` 拟合。模型选择、示例、推断和
统计限制见 [Cox 模型指南](../models/coxph.md)。传输标记描述的是数据移动，
不能仅凭它判断数值拟合是否在 CPU 上执行。

## 主机传输属性

| 属性 | 含义 |
|---|---|
| `full_host_transfer_performed_` | 至少一个完整的设备端训练组件传到了主机；对于 `CoxPHCV`，涵盖选择阶段和最终重拟合。 |
| `cv_full_host_transfer_performed_` | CV 选择阶段发生的传输，包括主机端组织数据折时的传输。 |
| `final_refit_full_host_transfer_performed_` | 使用选定配置在全部数据上重拟合时发生的传输。 |
| `orchestration_device_` | 组织 CV 选择过程的设备。 |

后三个属性属于 `CoxPHCV`。完整训练组件可以是排序后的响应向量，也可以是
保留的 `entry`、`strata` 或 `subject_id` 向量；标记为真不要求设计矩阵也被复制。
普通 GPU Breslow/Efron 预处理先在选定后端排序，再把完整的已排序 `time` 与
`event` 向量复制到主机端以构建事件组元数据，因此
`full_host_transfer_performed_` 为真。

## 本次调用与选择结果的来源

以下键位于 `CoxPHCV.cv_results_`。复用的选择结果可能来自之前的调用，应分别
查看本次调用字段和来源字段，不能把当前请求的设备当作复用评分的计算设备。

| 键 | 含义 |
|---|---|
| `selection_cache_hit` | 本次调用是否复用了缓存的选择结果。 |
| `requested_fit_device` | 本次请求的拟合设备。 |
| `effective_device` | 本次请求及最终重拟合所用设备。 |
| `selection_origin_device` | 原始选择结果记录的设备。 |
| `candidate_preparation_origin_device` | 原始选择过程中准备候选项所用设备。 |
| `scoring_device` | 原始选择过程的评分设备。 |
| `fold_backend_preparation_count_this_call` | 本次调用执行的数据折后端准备次数。 |
| `candidate_right_censored_preparation_count_this_call` | 本次调用执行的右删失候选数据准备次数。 |
| `candidate_target_host_transfer_count_this_call` | 本次调用中，需要传输响应数据的完整 `time`/`event` 元数据准备次数。 |
| `candidate_target_host_vector_transfer_count_this_call` | 本次调用实际传输的响应向量数。 |

相应的不带 `_this_call` 的准备与传输计数描述原始选择过程。复用缓存选择结果
时，上述本次调用的准备与响应传输计数为零，但仍可能有输入传输和新的最终
重拟合。因此，不能用候选响应传输计数为零来替代分阶段主机传输标记。这些
计数用于诊断，不是固定操作次数或性能保证。

## 实验性筛选诊断

请求 `STATGPU_COXPHCV_TWO_STAGE` 或 `STATGPU_COXPHCV_SUCCESSIVE_HALVING`
时，当前实现会告警，并以完整精度评估所有候选项。
`staged_safety_strategy="single_pass_exhaustive"` 标识这一行为。
请求与实际状态字段、候选项掩码见[筛选控制参考](../guides/cox-cv-staged-safety.md#诊断字段)。
