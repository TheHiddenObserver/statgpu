# CoxPHCV 实验性筛选控制

> 最后更新：2026-09-17  
> 适用对象：`statgpu.survival.CoxPHCV`  
> 切换：[English](../../en/guides/cox-cv-staged-safety.md)

`CoxPHCV` 识别两个实验性环境变量：

- `STATGPU_COXPHCV_TWO_STAGE`
- `STATGPU_COXPHCV_SUCCESSIVE_HALVING`

## 当前用户可见行为

请求任一控制项时，statgpu **不会**删除、近似处理或跳过任何 penalty candidate。它会发出 `RuntimeWarning`，并按普通 solver 精度评估完整 penalty grid。

用户可观察到的选择流程仍然是：

```text
完整 penalty grid
    -> 评估每个 candidate
    -> 从完整 candidate set 中选择
    -> 在全部数据上 refit 所选 penalty
```

这一行为在受支持的 NumPy、CuPy 与 Torch 执行路径上相同。

如果应用真正需要 staged screening 或 successive halving，目前不能把这两个环境变量解释为已经提供相应算法。

## 诊断字段

请求实验性控制后，`cv_results_` 会通过以下字段说明实际发生的执行方式：

| 字段 | 含义 |
|---|---|
| `two_stage_requested` | 是否请求 two-stage control |
| `two_stage_enabled` | 当前 exhaustive 行为下为 `False` |
| `successive_halving_requested` | 是否请求 successive halving |
| `successive_halving_enabled` | 当前 exhaustive 行为下为 `False` |
| `staged_execution_mode` | `"exhaustive_safety_fallback"` |
| `staged_safety_strategy` | `"single_pass_exhaustive"` |
| `staged_fallback_reason` | 未使用 staged screening 的用户可见原因 |
| `fast_pass_candidate_mask` | 全部为 `False` |
| `full_precision_candidate_mask` | 全部为 `True` |
| `screened_out_candidate_mask` | 全部为 `False` |

字段名称保留现有 API spelling，即使当前实际计算采用 exhaustive evaluation。

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

## 实际使用建议

普通 Cox penalty selection 应直接依赖上面描述的 exhaustive candidate-selection 行为。如果程序必须知道是否真正执行了 screening，应检查 enabled/diagnostic 字段，而不是根据环境变量是否设置来推断。

fold、final refit 与 Cox-specific 统计限制见 [Cox 比例风险模型](../models/coxph.md)和 [交叉验证](cross-validation.md)。
