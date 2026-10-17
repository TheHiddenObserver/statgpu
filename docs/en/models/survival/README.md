# Survival Analysis

> Language: English  
> Last updated: 2026-10-17  
> Switch: [Chinese](../../../cn/models/survival/README.md)

[Models and Statistical Methods](../README.md) / Survival analysis

Use survival methods when the response is **time until an event** and follow-up
can end before the event is observed. The inputs include both the observed time
and an event indicator, so censored observations still contribute information.
Cox regression relates predictors to the event hazard; it does not treat censored
times as ordinary continuous responses.

The current tutorials cover the Cox proportional-hazards family. Start with the
[Cox tutorial](../coxph.md) for assumptions, input meaning, and a first CPU fit,
then choose the estimator according to the outputs and penalties you need.

## Choosing a Cox estimator

| Need | Estimator | Import | Contract |
|---|---|---|---|
| Full Cox fitting, baseline hazards, survival prediction, formula input, and inference | `CoxPH` | `from statgpu.survival import CoxPH` | Breslow/Efron/Exact ties, delayed entry, `(start, stop]`, strata, robust/cluster covariance, NumPy/CuPy/Torch |
| Select a non-negative L2 penalty by held-out partial likelihood | `CoxPHCV` | `from statgpu.survival import CoxPHCV` | Uses the canonical Cox semantics during CV and performs a final `CoxPH` refit |
| Estimate with L1, L2, ElasticNet, SCAD, or MCP | `PenalizedCoxPHModel` | `from statgpu.linear_model import PenalizedCoxPHModel` | Broad penalty and generic solver path; currently estimation-only and rejects `compute_inference=True` |

`CoxPH(penalty=...)` and `PenalizedCoxPHModel` are not interchangeable aliases.
Use the canonical `CoxPH`/`CoxPHCV` path when counting-process features,
stratification, baseline prediction, or statistical inference are required. Use
`PenalizedCoxPHModel` when the broader penalty family is the primary requirement
and estimation-only output is sufficient.

The [Cox model page](../coxph.md) is the authoritative user-facing source for
Breslow/Efron/Exact ties, delayed-entry and `(start, stop]` data, strata,
robust/cluster inference, subject-grouped CV, prediction boundaries, and the
NumPy/CuPy/Torch support matrix.

## Continue the workflow

- [Cox proportional hazards](../coxph.md): worked examples, prediction, inference, and cross-validation
- [CoxPH and CoxPHCV API](../../reference/survival-smoothing-api.md#coxph-and-coxphcv): constructors, methods, and fitted attributes
- [Broader Cox penalties and model-family CV](../coxph.md#l1l2elasticnetscadmcp-model-family-cv): penalized fitting and tuning
- [CoxPH device and CV diagnostics](../../reference/coxph-diagnostics.md): device controls and tuning diagnostics
- [Regression overview](../regression/README.md): methods for other response types
