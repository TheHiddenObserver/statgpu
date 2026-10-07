# CoxPH device and CV diagnostics

> Language: English  
> Last updated: 2026-10-07  
> This page: Advanced API reference  
> Switch: [Chinese](../../cn/reference/coxph-diagnostics.md)

Use these fields to inspect a completed `CoxPH` or `CoxPHCV` fit. Start with the
[Cox model guide](../models/coxph.md) for model choice, examples, inference, and
statistical limitations. A transfer flag describes memory movement; it does
not by itself imply that numerical fitting ran on CPU.

## Host-transfer attributes

| Attribute | Interpretation |
|---|---|
| `full_host_transfer_performed_` | At least one complete device-resident training component was transferred to the host. On `CoxPHCV`, this covers selection and the final refit. |
| `cv_full_host_transfer_performed_` | Transfer during CV selection, including host-orchestrated fold construction. |
| `final_refit_full_host_transfer_performed_` | Transfer during the selected full-data refit. |
| `orchestration_device_` | Device used to organize CV selection. |

The last three attributes describe `CoxPHCV`. A complete training component can
be a sorted response vector or retained `entry`, `strata`, or `subject_id`
vector; a true flag does not require copying the design matrix. Ordinary GPU
Breslow/Efron preprocessing sorts on the selected backend, then copies complete
sorted `time` and `event` vectors to the host for event-group metadata, so its
`full_host_transfer_performed_` is true.

## Current call and selection origin

The following keys are in `CoxPHCV.cv_results_`. Reused selection results can
come from an earlier call. Inspect the current-call fields separately from
selection-origin fields rather than treating the current requested device as
proof of where reused selection scores were calculated.

| Key | Interpretation |
|---|---|
| `selection_cache_hit` | Whether this call reused cached selection results. |
| `requested_fit_device` | Device requested for this call. |
| `effective_device` | Current requested/final-refit device. |
| `selection_origin_device` | Device recorded for the original selection. |
| `candidate_preparation_origin_device` | Device recorded for preparation of that selection's candidates. |
| `scoring_device` | Device recorded for that selection's scoring. |
| `fold_backend_preparation_count_this_call` | Fold/backend preparations performed by this call. |
| `candidate_right_censored_preparation_count_this_call` | Right-censored candidate-data preparations performed by this call. |
| `candidate_target_host_transfer_count_this_call` | Complete time/event metadata preparations requiring target transfers during this call. |
| `candidate_target_host_vector_transfer_count_this_call` | Individual target-vector transfers during this call. |

The corresponding preparation/transfer keys without `_this_call` describe the
original selection. Cached selection reuse reports zero current-call
preparations and target transfers for these counters; it can still have input
transfers and a new final refit. Do not use a zero candidate-transfer count as a
substitute for the phase-specific host-transfer flags. Counts are diagnostics,
not fixed operation-count or performance guarantees.

## Experimental screening diagnostics

When `STATGPU_COXPHCV_TWO_STAGE` or `STATGPU_COXPHCV_SUCCESSIVE_HALVING` is
requested, the current implementation warns and evaluates all candidates at
full precision. `staged_safety_strategy="single_pass_exhaustive"` identifies
this behavior. The requested/effective flags and candidate masks are defined
in the [screening-control reference](../guides/cox-cv-staged-safety.md#diagnostic-fields).
