# CV and Inference Frontend Contract

## 1. Purpose

This document freezes the frontend contract before the missing cross-validation and broader inference benchmarks are generated.

The dashboard treats:

- **cross-validation** as a benchmark task that may also belong to a base model category such as Linear Models, Penalized GLM, or Survival Analysis;
- **inference** as a metric scope attached to the corresponding model, not as a separate statistical model family.

The frontend must therefore remain usable when one scope has current rows and another has no current source yet.

## 2. Current status

### Inference

Current June-or-later canonical data already includes:

- Ordered Logit/Probit inference;
- Quantile kernel-sandwich and bootstrap inference;
- penalized-logistic L2 HC0 sandwich inference;
- penalized-logistic SCAD oracle inference;
- penalized-linear/Lasso bootstrap inference.

These rows are selected through:

```text
Metric scope → Inference
```

The Inference Metrics panel is placed above the overview table and displays model, inference method/variant, penalty, backend, scale, timing scope, BSE, Wald statistic, p-value, status, and source.

### Cross-validation

No source dated 2026-06-01 or later is currently registered. The frontend therefore displays a disabled:

```text
CV (0)
```

scope control. This is intentional: it shows that the interaction contract is implemented without reconnecting the excluded April 2026 LassoCV artifact.

## 3. Metric-scope classification

A run is classified as **Inference** when at least one of the following is true:

- `metrics.inference` is present;
- `parameters.compute_inference == true`;
- `parameters.inference_method` is present;
- `parameters.timing_scope` contains `inference`.

A run is classified as **Cross-validation** when at least one of the following is true:

- `model_id` ends in `CV` or `CrossValidation`;
- `parameters.metric_scope`, `benchmark_scope`, `task_scope`, or `timing_scope` equals `cv` or `cross_validation`;
- one of `parameters.cv`, `cv_folds`, `fold_count`, or `n_folds` is present.

A CV run may also expose Prediction or Selection metrics. Scope filtering is non-exclusive: such a row can be found through CV, Prediction, and Selection views.

## 4. Required fields for new CV results

A new current CV source should emit, at minimum:

```json
{
  "category_ids": ["linear_models"],
  "model_id": "LassoCV",
  "variant": "path-cv",
  "loss": "squared_error",
  "penalty": "l1",
  "solver": "auto",
  "parameters": {
    "metric_scope": "cross_validation",
    "timing_scope": "cv_total",
    "cv_folds": 5,
    "grid_size": 100,
    "warm_start": true,
    "refit": true,
    "selected_alpha": 0.01
  },
  "metrics": {
    "timing": {
      "fit_time_ms": 123.4,
      "quality": "measured",
      "source_file": "cv_models_202607xx.json"
    }
  }
}
```

`metrics.timing.fit_time_ms` should represent the total user-facing CV wall time. Stage-specific values such as path construction, fold fitting, selection, and refit may be retained in `parameters` until the schema gains dedicated timing-stage fields.

Recommended additional metrics:

- selected hyperparameter and its variability;
- CV score/path summary;
- test prediction error;
- support precision/recall/F1/Jaccard for sparse models;
- convergence and failed-fold counts;
- NumPy/CuPy/Torch agreement;
- matched external-framework timing where objectives are aligned.

## 5. Required fields for new inference results

New inference rows should retain:

```text
model family
loss and penalty
inference method
covariance type
bootstrap replicate count or HAC bandwidth/kernel
fit-only, inference-only, or fit-plus-inference timing scope
backend
scale
BSE/Wald/p-value or method-appropriate uncertainty metrics
correctness reference and tolerance
```

Recommended `parameters.timing_scope` values are:

```text
fit_only
inference_only
fit_plus_inference
```

Separate fit and inference timing is preferred for new sources. Existing PR #74 rows remain labelled `fit_plus_inference` because that is what the runner measured.

## 6. Acceptance criteria

A new CV or inference source is frontend-ready when:

1. it is dated 2026-06-01 or later;
2. its JSON is structured and SHA256-registered;
3. method, case, timing scope, and backend identities are complete;
4. unsupported, failed, and OOM cases are explicit;
5. GPU timing uses explicit backend synchronization;
6. parser tests verify the complete source matrix;
7. the Metric scope control obtains a non-zero count automatically;
8. overview Scope cells and the corresponding metric panels show the correct semantics;
9. deterministic generation, Python 3.9/3.11, TypeScript, build, staleness, and Playwright checks pass.
