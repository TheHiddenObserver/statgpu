# Cross-Validation

> Language: English  
> Last updated: 2026-09-11  
> This page: CV user guide + architecture + final-refit semantics  
> Switch: [Chinese](../../cn/guides/cross-validation.md)

## Overview

statgpu provides cross-validation estimators for regularized models. Candidate scoring and the selected full-data refit are separate stages, and coefficient inference—when available—belongs to the selected final refit rather than the CV folds.

| CV estimator | Base model | Main tuning |
|---|---|---|
| `RidgeCV` | `Ridge` | L2 alpha |
| `LassoCV` | `Lasso` | L1 alpha |
| `ElasticNetCV` | `ElasticNet` | alpha / optional l1_ratio |
| `LogisticRegressionCV` | `LogisticRegression` | logistic regularization |
| `PenalizedGLM_CV` | `PenalizedGeneralizedLinearModel` or `PenalizedCoxPHModel` | family-specific penalty alpha |

For the coefficient-inference target and method resolver used by `PenalizedGLM_CV`, see [Penalized GLM inference](penalized-glm-inference.md).

## Quick start

```python
from statgpu.linear_model import PenalizedGLM_CV

model = PenalizedGLM_CV(
    loss="poisson",
    penalty="l2",
    alpha_grid=[0.2, 0.05, 0.01],
    cv=5,
    device="auto",
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
)
model.fit(X, y)

print(model.alpha_)
print(model.inference_method_)          # m_estimation
print(model.penalty_conditioning_)     # cv_selected_penalty
```

CV folds/path/grid fits remain estimation-only. If inference is requested, statgpu selects the tuning parameter first and then runs coefficient inference exactly once on the full-data selected-penalty refit.

## Main parameters

### RidgeCV

| Parameter | Typical default | Meaning |
|---|---:|---|
| `alphas` | `None` | User alpha grid or automatic grid |
| `cv` | estimator default | Number of folds |
| `compute_inference` | estimator default | Inference on the selected full-data refit |
| `cov_type` | `"nonrobust"` | Final-refit covariance convention |

### LassoCV

`LassoCV` separates the CV path solver from the final full-data refit solver.

| Parameter | Default | Meaning |
|---|---:|---|
| `alphas` | `None` | Alpha grid |
| `n_alphas` | `12` | Automatic-grid size |
| `solver` | `"fista"` | Final full-data `Lasso` refit solver |
| `cv_solver` | `"auto"` | Fold/path solver |
| `cpu_solver` | `None` | Deprecated historical CPU-CV control |
| `method` | `"standard"` | CV-path profile |
| `compute_inference` | estimator default | Inference only on selected final refit |

After fitting, `cv_solver_` records the algorithm actually used by the CV stage.

### ElasticNetCV

| Parameter | Default | Meaning |
|---|---:|---|
| `l1_ratio` | `0.5` | Mixing parameter or search list |
| `alphas` | `None` | Alpha grid |
| `compute_inference` | estimator default | Maintained sparse-Gaussian inference on the selected final refit |

### PenalizedGLM_CV

| Parameter | Default | Meaning |
|---|---:|---|
| `loss` | `"squared_error"` | Scalar-response loss or `"cox_ph"` survival branch |
| `penalty` | `"l2"` | Penalty name/object |
| `alpha_grid` | `None` | Candidate alpha grid |
| `n_alphas` | `100` | Automatic-grid size |
| `cv` | `5` | Number of folds |
| `cv_splits` | `None` | Optional custom folds |
| `device` | `"auto"` | CV execution device policy |
| `compute_inference` | `False` | Run coefficient inference on the selected full-data final refit only |
| `inference_method` | `"auto"` | Final-refit inference request |
| `cov_type` | `"nonrobust"` | Final-refit covariance; non-Gaussian L2 M-estimation currently supports nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | Retained lower-level control; does not imply non-Gaussian penalized HAC support |

A successful `PenalizedGLM_CV` inference-enabled fit delegates the final estimator's result/provenance and additionally records:

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

Thus the reported SEs, p-values, and confidence intervals are conditional on the selected alpha. They do **not** adjust for tuning-selection uncertainty.

## Custom folds

`cv_splits` may provide explicit `(train_idx, validation_idx)` pairs. One-shot generators are materialized once so repeated candidate work sees the same folds.

```python
from sklearn.model_selection import TimeSeriesSplit

folds = list(TimeSeriesSplit(n_splits=5).split(X))
model = PenalizedGLM_CV(
    loss="poisson",
    penalty="l1",
    cv_splits=folds,
)
model.fit(X, y)
```

The survival branch applies stricter event-support validation: every evaluable Cox training/validation partition must contain observed events, and invalid indices fail before candidate fitting.

## Sample weights

Sample-weight support is path-specific; a solver limitation must not be generalized into a claim that an estimator never supports weights.

For the new non-Gaussian L2 coefficient-inference contract, analytic weights are supported by the selected final refit and M-estimation covariance. When an inference-enabled weighted non-Gaussian L2/no-penalty final fit keeps `solver="auto"`, the fit transaction uses the maintained backend-native FISTA path for that fit because Newton currently rejects non-uniform analytic weights. The public solver request remains `auto`; explicit solver choices stay authoritative.

The penalized Cox CV branch currently rejects `sample_weight`.

## Alpha grids

Automatic grids are generated on statgpu's average-loss scale. The exact zero-model/KKT rule depends on the loss/penalty pair; ElasticNet adjusts the L1 threshold by its mixing parameter.

User grids are validated before candidate fitting. Scalar-response and survival CV intentionally have different zero/positivity rules where their statistical contracts differ.

## Fitted attributes

Common CV outputs include:

- `alpha_`;
- `best_score_`;
- `cv_results_`;
- `estimator_` — selected hyperparameter, full-data final refit;
- `coef_` and `intercept_` where defined.

When `PenalizedGLM_CV(compute_inference=True)` resolves to a supported final-refit row, it also publishes `_inference_result`, `_bse`, `_pvalues`, `_conf_int`, and requested/resolved/target/conditioning provenance.

## Device selection

`device="auto"` combines problem-size and loss×penalty heuristics with actual backend availability. Explicit `device="cuda"` and `device="torch"` are strict: they never silently become CPU execution when the requested backend is unavailable.

Current scalar CV code contains benchmark-backed thresholds for small-problem CPU execution, selected high-dimensional sparse GLM Torch paths, and a large-effective-work fallback. Those thresholds are execution policy, not part of the statistical definition.

## Inference after CV

### Ridge / sparse Gaussian CV

Ridge/Lasso/ElasticNet keep their estimator-specific maintained inference contracts. Their folds are used for selection; inference is associated with the final selected full-data fit.

### PenalizedGLM_CV

Coefficient inference is **final-refit-only**:

1. folds/grid/path candidate fits run with inference disabled;
2. alpha is selected from held-out evidence;
3. if requested, the selected model is refit on full data with the public inference controls;
4. the final estimator's `_inference_result` and provenance are delegated to the CV estimator;
5. selection adjustment is explicitly reported as false.

The current supported rows include:

- smooth non-Gaussian L2/no penalty → `m_estimation`;
- maintained Gaussian L2 and sparse-Gaussian contracts;
- explicit narrow SCAD/MCP oracle/bootstrap rows only where the underlying estimator supports them.

Non-Gaussian L1/ElasticNet coefficient inference is not implemented. The Cox branch remains estimation-only.

See [Penalized GLM inference](penalized-glm-inference.md) for the full method matrix.

## Strict and two-stage CV

`PenalizedGLM_CV` defaults to `cv_strategy="strict"`. Strict mode scores the requested candidate grid with the maintained iteration/tolerance settings.

`cv_strategy="two_stage"` is opt-in approximate screening:

1. relaxed stage-1 scoring over the full grid;
2. keep top candidates;
3. strict refinement for those candidates;
4. strict/full final refit.

Because stage-1 approximation can change rankings when CV curves are very close, the mode is visibly approximate and requires acknowledgement to silence its warning.

## Architecture

Scalar-response and survival CV have different preparation order, but they share one key rule: **selection and inference are separate operations**.

### Scalar response

```text
validate/generate alpha grid
        ↓
materialize folds
        ↓
resolve CV device + solver
        ↓
score candidates (inference disabled)
        ↓
select alpha
        ↓
full-data final refit
        ↓
optional coefficient inference exactly once
```

### Penalized Cox

```text
normalize (time,event)
        ↓
validate/materialize event-supported folds
        ↓
resolve CV device
        ↓
survival-aware held-out scoring
        ↓
require complete finite evidence
        ↓
PenalizedCoxPHModel final refit (inference disabled)
```

## Main scoring paths

### Ridge eigendecomposition

Squared-error L2 CPU scoring can reuse one eigendecomposition per fold across many alphas. The maintained full-data refit preserves its exact float64 semantics.

### Fold-batch sparse GLM

Selected GPU sparse-GLM paths batch fold linear algebra to reduce kernel launches and host synchronization.

### Sparse Gram + warm starts

Squared-error L1/ElasticNet paths can reuse fold Gram matrices and warm-start the descending alpha path.

### SCAD/MCP LLA

Non-convex penalties require continuation/LLA outer iterations, so their CV path is more expensive than a convex sparse path.

### Generic fallback

When no optimized path applies, CV fits maintained estimators per fold/candidate and evaluates the declared loss. Infrastructure recovery must not silently replace a non-Gaussian objective with MSE.

## Caching

`LassoCV` uses a selection-only cache: it caches alpha-selection evidence, not the final estimator. The key includes data/weight identity, evaluated alpha grid, full fold indices, CV solver/method controls, tolerances, and execution-mode fields that can change selection. A final-refit solver change is intentionally excluded when it cannot change CV scoring.

## Alpha conventions

statgpu penalized estimators use an average-loss objective. Cross-framework comparisons must align penalty scaling first. For example, Ridge often requires:

```text
sklearn_alpha = statgpu_alpha * n_samples
```

Same-named alpha values are not automatically comparable across frameworks with different objective normalization.

## Penalized Cox branch

`PenalizedGLM_CV(loss="cox_ph")` keeps the target as an `(n_samples, 2)` `[time, event]` array throughout selection. It uses survival-aware held-out partial-likelihood scoring, validates event support, and refits `PenalizedCoxPHModel` without an intercept.

It currently does not support:

- coefficient inference;
- `sample_weight`;
- `cv_strategy="two_stage"`;
- dictionary targets.

## Known limits

- Non-Gaussian L1/ElasticNet coefficient inference is not implemented.
- Non-Gaussian penalized M-estimation currently supports nonrobust/HC0/HC1 covariance only.
- Weighted/robust/family-aware bootstrap is outside the maintained Gaussian residual-bootstrap contract.
- Explicit solver sample-weight support remains solver-specific; the inference contract does not override an explicit solver request.
- Penalized Cox CV remains estimation-only.

## FAQ

**Does CV inference adjust for hyperparameter selection?**  
No. `penalty_selection_adjusted_=False`; the result conditions on the selected alpha.

**Why did my cache miss?**  
Selection-relevant data/fold/grid/solver/tolerance/execution fields changed.

**Why is SCAD/MCP CV slower?**  
It requires LLA/continuation outer work.

**Can non-Gaussian L1/ElasticNet use bootstrap as a fallback?**  
No. The maintained bootstrap in this repair is Gaussian residual bootstrap only.

## See also

- [Penalized GLM inference](penalized-glm-inference.md)
- [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md)
- [Ridge](../models/ridge.md)
- [ElasticNet](../models/elastic-net.md)
- [GeneralizedLinearModel and Penalized GLM](../models/generalized-linear-model.md)

## External validation

The maintained CV suite covers parameter validation, fold safety, solver migration, selection caches, Ridge/Lasso/ElasticNet paths, backend routing, and penalized-Cox evidence rules. The penalized-GLM inference repair additionally tests final-refit-only inference, selected-alpha reuse, clone/API compatibility, formula parity, and selection-conditioning provenance.
