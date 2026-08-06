# CoxPHCV Experimental Screening Safety

> Last updated: 2026-08-04  
> Applies to: `statgpu.survival.CoxPHCV`

## Status

`CoxPHCV` exposes two experimental, environment-controlled optimization switches:

- `STATGPU_COXPHCV_TWO_STAGE`
- `STATGPU_COXPHCV_SUCCESSIVE_HALVING`

These switches currently **do not remove or approximate any penalty candidate**. When either switch is requested, statgpu emits a `RuntimeWarning` and evaluates the complete penalty grid at full solver precision. This correctness-first fallback prevents preliminary scores or numerical ties from changing the selected regularization parameter.

## Backend behavior

The statistical and execution contract is the same on NumPy, CuPy, and Torch CUDA:

- every candidate receives full-precision evaluation;
- no candidate is screened out;
- the raw staged and successive-halving branches are disabled;
- exactly one exhaustive candidate pass is used;
- final selection uses the complete candidate set;
- the selected penalty is refitted on the full data.

In particular, CuPy no longer expands the staged candidate sets to the complete grid and then repeats the same full-precision finalists. All three backends use the ordinary exhaustive selector once. This avoids a backend-specific double full-grid fit while retaining the same selected-penalty contract.

## Diagnostics

When an experimental switch is requested, `cv_results_` includes:

| Field | Meaning |
|---|---|
| `two_stage_requested` | Whether the two-stage environment switch was requested |
| `two_stage_enabled` | Always `False` while screening is safety-disabled |
| `successive_halving_requested` | Whether successive halving was requested |
| `successive_halving_enabled` | Always `False` while screening is safety-disabled |
| `staged_execution_mode` | `"exhaustive_safety_fallback"` |
| `staged_safety_strategy` | Always `"single_pass_exhaustive"` |
| `staged_fallback_reason` | User-visible reason screening was disabled |
| `fast_pass_candidate_mask` | All `False` |
| `full_precision_candidate_mask` | All `True` |
| `screened_out_candidate_mask` | All `False` |

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
assert model.cv_results_["staged_safety_strategy"] == "single_pass_exhaustive"
assert model.cv_results_["full_precision_candidate_mask"].all()
assert not model.cv_results_["screened_out_candidate_mask"].any()
```

The environment switches should be treated as reserved experimental controls. A future release may re-enable screening only after deterministic candidate ranking and NumPy/CuPy/Torch correctness and performance evidence are complete.
