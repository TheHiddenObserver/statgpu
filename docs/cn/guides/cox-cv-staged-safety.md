# CoxPHCV 实验性筛选控制

> 最后更新：2026-09-17  
> 适用对象：`statgpu.survival.CoxPHCV`  
> 切换：[English](../../en/guides/cox-cv-staged-safety.md)

`CoxPHCV` 识别两个实验性环境变量：

- `STATGPU_COXPHCV_TWO_STAGE`
- `STATGPU_COXPHCV_SUCCESSIVE_HALVING`

## 当前用户可见行为

无论是否设置这两个环境变量，当前 `CoxPHCV` 都会对**完整惩罚参数网格**执行全精度交叉验证，再选择惩罚强度并进行最终重拟合。

因此，开启这些实验控制目前**不会减少实际评估的候选惩罚参数数量**。它们保留的是公开配置入口和诊断信息：用户可以明确知道自己请求了哪种实验性筛选方式，同时也能确认本次运行实际仍采用完整候选评估。

这意味着：

- 候选惩罚参数集合不会因实验开关而改变；
- 各数据折仍以完整精度拟合并评分；
- 选出的惩罚强度来自完整候选集合；
- 最终模型仍使用选定惩罚强度在全部数据上重新拟合；
- 实验开关不会静默改变当前的参数选择结果。

NumPy、CuPy 与 Torch 路径遵循相同的统计语义。显式选择 CuPy 或 Torch 时，完整候选评估仍在对应受支持的执行后端上进行；实验开关不会把 GPU 请求静默改成另一套近似选择问题。

## 诊断字段

拟合完成后，`cv_results_` 中与实验性筛选有关的字段用于区分“用户请求了什么”和“本次实际执行了什么”。当前字段包括：

| 字段 | 当前含义 |
|---|---|
| `two_stage_requested` | 是否请求了两阶段筛选环境开关 |
| `two_stage_enabled` | 当前为 `False`；实际两阶段筛选尚未启用 |
| `successive_halving_requested` | 是否请求了 successive halving 环境开关 |
| `successive_halving_enabled` | 当前为 `False`；实际 successive halving 尚未启用 |
| `staged_execution_mode` | 当前为 `"exhaustive_safety_fallback"` |
| `staged_safety_strategy` | 当前为 `"single_pass_exhaustive"` |
| `staged_fallback_reason` | 说明为什么本次仍采用完整候选评估 |
| `fast_pass_candidate_mask` | 当前全部为 `False` |
| `full_precision_candidate_mask` | 当前全部为 `True` |
| `screened_out_candidate_mask` | 当前全部为 `False` |

这些字段是用户可观察的诊断信息。它们描述当前行为，但不意味着调用者应该依赖内部候选循环、私有函数或具体执行顺序。

## 如何理解实验开关

需要区分三件事：

- **请求状态**：是否设置了实验性环境变量；
- **实际行为**：当前仍完整评估全部候选项；
- **选择结果**：最终惩罚强度来自完整交叉验证，而不是近似筛选结果。

因此，不应把“实验开关已设置”解释成“两阶段筛选或 successive halving 已经实际启用”。例如：

```python
import os
from statgpu.survival import CoxPHCV

os.environ["STATGPU_COXPHCV_TWO_STAGE"] = "1"

model = CoxPHCV(
    penalties=[0.8, 0.4, 0.2, 0.1, 0.05],
    cv=3,
).fit(X, time, event)

assert model.cv_results_["staged_execution_mode"] == "exhaustive_safety_fallback"
assert model.cv_results_["staged_safety_strategy"] == "single_pass_exhaustive"
assert model.cv_results_["full_precision_candidate_mask"].all()
```

## 为什么保留这些控制项

这些环境变量用于保留实验接口和可诊断性，使未来的候选筛选研究不必重新定义公开配置拼写。

如果以后真正启用分阶段筛选，用户文档需要直接说明新的可观察语义，例如：

- 哪些候选项会进入高精度阶段；
- 各数据折如何分配计算预算；
- 近似筛选是否可能改变最终候选集合；
- 最终重拟合是否仍在全部数据上使用选定配置。

## 使用建议

如果只需要标准的 Cox 惩罚参数选择，无需设置这些实验环境变量；默认行为已经完整评估候选网格。

如果正在测试实验性控制，可以设置相应环境变量，并通过 `cv_results_` 中的公开诊断字段同时确认“请求了什么”和“实际执行了什么”。当前应预期完整候选评估，而不是计算量减少。

一般的 CV 数据折、选择和最终重拟合语义见 [交叉验证](cross-validation.md)；Cox 专属的数据结构、风险集与评分语义见 [Cox 比例风险模型](../models/coxph.md)。
