# Cross-Validation

> Language: English  
> Last updated: 2026-09-17  
> This page: user guide for CV selection and refit behavior  
> Switch: [Chinese](../../cn/guides/cross-validation.md)

## Overview

Cross-validation in statgpu is a **model-selection layer** around an estimator. A typical fit follows the same statistical sequence:

```text
candidate grid
    -> fit candidates on training folds
    -> evaluate held-out scores
    -> select a candidate
    -> refit the selected configuration on all observations
```

This page documents the task-oriented public behavior: how to configure folds and tuning grids, how solver/device choices interact with CV, what happens during the final refit, and how inference after CV should be interpreted.

For the public execution model behind these behaviors—including the selection/refit split, pathwise reuse, GPU batching, and selection-cache semantics—see [How statgpu Cross-Validation Works](cross-validation-design.md). Exact private fast paths, cache-key fields, helper names, and benchmark-derived routing thresholds remain implementation details.

## Available CV estimators

| CV estimator | Base model | Main tuning target |
|---|---|---|
| `RidgeCV` | `Ridge` | `alpha` |
| `LassoCV` | `Lasso` | `alpha` |
| `ElasticNetCV` | `ElasticNet` | `alpha`, optionally `l1_ratio` |
| `LogisticRegressionCV` | `LogisticRegression` | regularization strength |
| `PenalizedGLM_CV` | penalized GLM / penalized Cox | `alpha` for the selected loss and penalty |
| `CoxPHCV` | `CoxPH` | Cox penalty strength |

For exact loss × penalty × solver availability, use the [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md). Model-specific statistical restrictions belong to the corresponding model page.

## Quick start

### RidgeCV

```python
from statgpu.linear_model import RidgeCV

model = RidgeCV(
    alphas=None,
    n_alphas=100,
    cv=5,
    device="auto",
)
model.fit(X, y)

print(model.alpha_)
print(model.score(X_test, y_test))
```

### LassoCV

`LassoCV` distinguishes the algorithm used to score the CV path from the algorithm used for the final full-data refit:

```python
from statgpu.linear_model import LassoCV

model = LassoCV(
    cv=5,
    cv_solver="auto",   # CV folds/path
    solver="fista",     # final full-data refit
    device="auto",
)
model.fit(X, y)

print(model.alpha_)
print(model.cv_solver_)
```

`cv_solver_` records the algorithm that actually ran during selection. The older `cpu_solver` control is deprecated; see the [penalized solver API migration guide](penalized-solver-api-migration.md).

### PenalizedGLM_CV

```python
from statgpu.linear_model import PenalizedGLM_CV

model = PenalizedGLM_CV(
    loss="poisson",
    penalty="scad",
    cv=5,
    device="auto",
)
model.fit(X, y)

print(model.alpha_)
pred = model.predict(X_test)
```

An explicit solver request remains authoritative wherever that loss/penalty combination supports it. Unsupported explicit combinations raise an error rather than silently changing algorithms. `solver="auto"` uses the documented dispatch for the selected model family.

## Folds

With an integer `cv`, statgpu constructs k-fold training/validation splits. `random_state` controls reproducible shuffling where the estimator exposes that option.

Estimators that expose `cv_splits` also accept explicit train/validation index pairs:

```python
from sklearn.model_selection import TimeSeriesSplit
from statgpu.linear_model import PenalizedGLM_CV

tscv = TimeSeriesSplit(n_splits=5)

model = PenalizedGLM_CV(
    loss="poisson",
    penalty="l1",
    cv_splits=list(tscv.split(X)),
)
model.fit(X, y)
```

Use custom splits when ordinary randomly shuffled folds are statistically inappropriate, for example with ordered or grouped data. The split itself is part of the statistical design; statgpu does not infer whether a user-supplied split is scientifically appropriate for the application.

For `PenalizedGLM_CV`, `cv_splits` may also be a one-shot iterator such as a generator. statgpu materializes that iterator privately once and reuses the snapshot for repeated `fit()` calls, scikit-learn cloning, and pickle serialization, while leaving the public `cv_splits` attribute unchanged. Reusable lists and tuples continue to be read directly.

### Cox data

Cox CV has additional requirements because validation scores depend on event information and survival-risk sets. For `PenalizedGLM_CV(loss="cox_ph", ...)`, survival targets remain two-dimensional and every evaluated train/validation partition must contain the event information required by the Cox score. Invalid partitions raise before candidate selection.

The penalized Cox CV branch does not support `sample_weight` or post-selection coefficient inference. For the complete survival-data contract, including `CoxPHCV`, see [Cox Proportional Hazards](../models/coxph.md).

## Tuning grids

Specialized estimators such as `RidgeCV`, `LassoCV`, and `ElasticNetCV` expose `alphas`; `PenalizedGLM_CV` exposes `alpha_grid`.

When a grid is omitted, the estimator constructs a data-dependent grid appropriate to its model. A user-supplied grid is treated as the requested candidate set after the estimator's public validation rules are applied.

For Quantile rows in `PenalizedGLM_CV`, the automatic grid uses the intercept-only check-loss score at the requested quantile rather than a squared-residual surrogate. Analytic `sample_weight` enters the same normalized pinball subgradient. For Group SCAD/MCP, the feature score is mapped to the public group-penalty scale through `max_g ||score_g||_2 / sqrt(p_g)`, matching the penalty's `alpha * sqrt(p_g)` local threshold. With fixed positive Adaptive L1 weights, the same score is divided coordinatewise by the effective adaptive weights before taking the maximum; fixed positive Adaptive Group Lasso weights analogously use `max_g ||score_g||_2 / (w_g sqrt(p_g))`. When adaptive weights are not yet fixed and will be learned from an initialization fit, the generated grid remains a pre-initialization heuristic rather than an exact all-zero KKT threshold.

```python
import numpy as np
from statgpu.linear_model import RidgeCV

model = RidgeCV(
    alphas=np.logspace(-4, 2, 50),
    cv=5,
)
model.fit(X, y)
```

Scalar-response penalized CV generally searches positive regularization strengths. If an unpenalized fit is the scientific target, use the corresponding direct estimator with its documented zero/no-penalty configuration rather than assuming every CV class treats zero as an ordinary candidate.

Cox grids have survival-specific validation rules; see the Cox model page rather than inferring those rules from scalar-response CV.

## Sample weights

Where a CV estimator supports `sample_weight`, the weights enter the training-fold objective and the corresponding weighted validation criterion, and the selected final model is refit with the full-data weights. For `PenalizedGLM_CV`, every evaluated training fold and validation fold must retain a finite positive total analytic weight. A fold with zero weight mass is undefined for the declared weighted objective and raises before candidate fitting; statgpu does not replace that fold by an unweighted score.

For Quantile rows, strict selection also requires complete finite fold evidence for each alpha. A fold fit that fails and produces no finite score makes that alpha ineligible rather than allowing the remaining folds to determine its mean score. Automatic scalar/Group non-convex Quantile routes expose target-level convergence failure as such an unscoreable fold; an ordinary solver `ConvergenceWarning` by itself does not discard an otherwise finite result. With `cv_strategy="two_stage"`, the screening pass remains relaxed; the complete-evidence rule is enforced in strict refinement and ordinary strict CV.

```python
from statgpu.linear_model import RidgeCV

model = RidgeCV(cv=5)
model.fit(X, y, sample_weight=w)
```

Weight support is **not** a blanket property of the word “CV”: it depends on the base loss, penalty, and solver route. If an explicit solver does not support the requested weighted objective, statgpu raises instead of dropping the weights or silently switching to a different objective. The [compatibility matrix](solver-penalty-matrix.md) and model pages are the canonical places to check those combinations.

## Selection and final refit

CV fold fits are temporary candidate fits. After scoring the candidate grid, the selected hyperparameter configuration is fitted again on the complete dataset.

This distinction matters for interpretation:

- `alpha_` (and `l1_ratio_` where applicable) belongs to the selection stage;
- `coef_`, `intercept_`, prediction, and ordinary fitted-model diagnostics belong to the final full-data refit;
- a solver used only for the CV path need not be the same solver used for the final refit when the estimator exposes separate controls;
- inference, when supported, is computed from the selected full-data refit rather than separately inside every fold.

For why this separation is part of the design rather than just an implementation detail, see [How statgpu Cross-Validation Works](cross-validation-design.md).

## Inference after CV

For estimators that support `compute_inference=True`, candidate fits remain selection-only. statgpu first selects the tuning parameter, refits the selected model on all observations, and then runs the requested inference on that final estimator.

The resulting intervals and p-values therefore condition on the selected tuning configuration; they do **not** automatically account for uncertainty introduced by choosing that configuration with CV. `PenalizedGLM_CV` exposes this distinction through its inference metadata.

See [Inference Modes](inference-modes.md) and [Penalized GLM inference](penalized-glm-inference.md) for the statistical interpretation and supported methods.

## Device behavior

CV follows the same explicit-device rule as direct estimators:

- `device="cpu"` requests NumPy CPU computation;
- `device="cuda"` requests CuPy CUDA and raises if that backend is unavailable;
- `device="torch"` requests the Torch CUDA route and raises if it is unavailable;
- `device="auto"` may choose among available backends according to the estimator and workload.

Automatic routing is an implementation choice and may evolve with measured performance. Do not write application logic that depends on a particular internal size threshold. If a specific execution backend is required, request it explicitly.

The public design page explains why automatic backend choice and GPU batching are allowed to vary without changing the CV statistical problem: [How statgpu Cross-Validation Works](cross-validation-design.md). See [Device and GPU Memory](device-and-memory.md) for the device contract.

## Fitted results

Exact result dictionaries vary by estimator, but CV estimators expose the selected tuning value and the fitted final estimator through their documented attributes. Common examples include:

| Attribute | Meaning |
|---|---|
| `alpha_` | selected regularization strength |
| `l1_ratio_` | selected ElasticNet mixing value, when searched |
| `cv_results_` | candidate/fold scoring information exposed by that estimator |
| `estimator_` | selected model refit on the full dataset, where exposed |
| `coef_`, `intercept_` | parameters of the final refit |
| `cv_solver_` | actual CV-path solver for `LassoCV` |

Do not assume every CV class uses an identical `cv_results_` schema; use the corresponding estimator/model reference when consuming detailed diagnostic fields programmatically.

## Choosing a CV configuration

A practical sequence is:

1. choose the statistical model and penalty first;
2. choose folds that respect the data-generating structure;
3. use the estimator's automatic grid unless there is a substantive reason to supply one;
4. leave solver/device selection on `auto` unless reproducibility, hardware, or algorithm choice requires an explicit setting;
5. interpret inference after tuning as inference conditional on the selected configuration unless a method explicitly states otherwise.

## Related documentation

- [How statgpu Cross-Validation Works](cross-validation-design.md) — public execution model, acceleration concepts, and statistical invariants
- [Implemented Methods](implemented-methods.md) — available public estimators
- [Solver × Penalty Compatibility Matrix](solver-penalty-matrix.md) — explicit compatibility
- [Solver Algorithms](solver-algorithms.md) — optimization algorithms
- [Device and GPU Memory](device-and-memory.md) — backend/device semantics
- [Inference Modes](inference-modes.md) — choosing an inference method
- [Penalized GLM inference](penalized-glm-inference.md) — inference after penalized fitting/tuning
- [Cox Proportional Hazards](../models/coxph.md) — survival-specific CV behavior
