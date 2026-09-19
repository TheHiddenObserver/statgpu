# CoxPHCV Experimental Screening Controls

> Last updated: 2026-09-17  
> Applies to: `statgpu.survival.CoxPHCV`  
> Switch: [Chinese](../../cn/guides/cox-cv-staged-safety.md)

`CoxPHCV` recognizes two experimental environment controls:

- `STATGPU_COXPHCV_TWO_STAGE`
- `STATGPU_COXPHCV_SUCCESSIVE_HALVING`

## Current user-visible behavior

Requesting either control does **not** currently remove, approximate, or skip any penalty candidate. statgpu emits a `RuntimeWarning` and evaluates the complete penalty grid at ordinary solver precision.

The observable selection sequence remains:

```text
complete penalty grid
    -> evaluate every candidate
    -> select from the complete candidate set
    -> refit the selected penalty on all data
```

This behavior is the same for the supported NumPy, CuPy, and Torch execution paths.

If an application requires actual staged screening or successive halving, these environment controls should not be interpreted as providing that algorithm at present.

## Diagnostic fields

When an experimental control is requested, `cv_results_` exposes diagnostic fields describing what actually happened:

| Field | Meaning |
|---|---|
| `two_stage_requested` | whether the two-stage control was requested |
| `two_stage_enabled` | `False` under the current exhaustive behavior |
| `successive_halving_requested` | whether successive halving was requested |
| `successive_halving_enabled` | `False` under the current exhaustive behavior |
| `staged_execution_mode` | `"exhaustive_safety_fallback"` |
| `staged_safety_strategy` | `"single_pass_exhaustive"` |
| `staged_fallback_reason` | user-visible reason staged screening was not used |
| `fast_pass_candidate_mask` | all `False` |
| `full_precision_candidate_mask` | all `True` |
| `screened_out_candidate_mask` | all `False` |

The field names retain their existing API spelling even though the effective computation is exhaustive.

## Example

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

## Practical guidance

For ordinary Cox penalty selection, rely on the exhaustive candidate-selection behavior documented above. Code that needs to know whether actual screening occurred should inspect the enabled/diagnostic fields rather than assuming that setting an environment variable implies staged execution.

For the statistical CV contract, folds, final refit, and Cox-specific restrictions, see [Cox Proportional Hazards](../models/coxph.md) and [Cross-Validation](cross-validation.md).
