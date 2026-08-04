# CoxPHCV 实验性筛选安全策略

> 最后更新：2026-08-04  
> 适用对象：`statgpu.survival.CoxPHCV`

## 当前状态

`CoxPHCV` 提供两个由环境变量控制的实验性优化开关：

- `STATGPU_COXPHCV_TWO_STAGE`
- `STATGPU_COXPHCV_SUCCESSIVE_HALVING`

当前这两个开关**不会删除或近似处理任何 penalty candidate**。只要请求其中任一开关，statgpu 就会发出 `RuntimeWarning`，并以完整 solver 精度评估整个 penalty grid。该 correctness-first fallback 可避免初步分数、数值并列或近似并列改变最终选择的正则化参数。

## 后端行为

NumPy、CuPy 和 Torch CUDA 遵循相同的统计契约：

- 每个 candidate 都接受 full-precision evaluation；
- 不筛除任何 candidate；
- 最终选择基于完整 candidate set；
- 使用所选 penalty 在完整数据上重新拟合。

显式 CuPy 运行可能继续使用 staged fold-workspace machinery，以复用已经准备好的 fold state；但是 coarse、refinement 和 finalist 集合都会扩展为完整 grid。CPU 和 Torch 使用单次 exhaustive pass。该实现差异不会改变 candidate set 或最终选择结果。

## 诊断字段

请求实验性开关后，`cv_results_` 包含以下字段：

| 字段 | 含义 |
|---|---|
| `two_stage_requested` | 是否请求 two-stage 环境开关 |
| `two_stage_enabled` | screening 安全禁用期间恒为 `False` |
| `successive_halving_requested` | 是否请求 successive halving |
| `successive_halving_enabled` | screening 安全禁用期间恒为 `False` |
| `staged_execution_mode` | `"exhaustive_safety_fallback"` |
| `staged_safety_strategy` | exhaustive fallback 所采用的后端执行策略 |
| `staged_fallback_reason` | 禁用 screening 的用户可见原因 |
| `fast_pass_candidate_mask` | 全部为 `False` |
| `full_precision_candidate_mask` | 全部为 `True` |
| `screened_out_candidate_mask` | 全部为 `False` |

## 示例

```python
import os
from statgpu.survival import CoxPHCV

os.environ["STATGPU_COXPHCV_TWO_STAGE"] = "1"
os.environ["STATGPU_COXPHCV_SUCCESSIVE_HALVING"] = "1"

model = CoxPHCV(
    penalties=[0.8, 0.4, 0.2, 0.12, 0.1, 0.06, 0.04, 0.02],
    cv=3,
    device="cuda",
    compute_inference=False,
).fit(X, time, event)

assert model.cv_results_["staged_execution_mode"] == "exhaustive_safety_fallback"
assert model.cv_results_["full_precision_candidate_mask"].all()
assert not model.cv_results_["screened_out_candidate_mask"].any()
```

这些环境变量目前应被视为预留的实验性控制项。只有在 deterministic candidate ranking 以及 NumPy、CuPy、Torch 三后端 correctness evidence 完整之后，未来版本才可能重新启用实际 screening。
