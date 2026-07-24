# Panel Models

> Language: English  
> Last updated: 2026-07-24  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/panel.md)

## Overview

The `statgpu.panel` module provides six panel-data estimators:

- `PanelOLS`: entity and/or time fixed effects;
- `RandomEffects`: feasible GLS random effects;
- `PooledOLS`: stacked OLS without demeaning;
- `BetweenOLS`: regression on entity means;
- `FirstDifferenceOLS`: within-entity first differences;
- `FamaMacBeth`: period-by-period cross-sectional regressions with coefficient averaging.

Array-input numerical paths support NumPy, CuPy CUDA, and Torch CUDA. Formula construction and categorical entity/time/cluster labels are intentional CPU metadata boundaries; compact aligned codes are transferred to the selected numerical backend. Explicit GPU devices do not silently fall back to CPU.

## Paths

```python
from statgpu.panel import (
    PanelOLS,
    RandomEffects,
    PooledOLS,
    BetweenOLS,
    FirstDifferenceOLS,
    FamaMacBeth,
    clustered_covariance,
    two_way_clustered_covariance,
    hac_covariance,
)
```

## Model Summary

| Model | Transformation | Main inference choices |
|---|---|---|
| `PanelOLS` | Entity/time within transformation | nonrobust, HC1 robust, clustered |
| `RandomEffects` | Swamy-Arora feasible GLS | nonrobust |
| `PooledOLS` | Stacked OLS | nonrobust, robust, clustered, HAC |
| `BetweenOLS` | Entity means | nonrobust, robust, clustered |
| `FirstDifferenceOLS` | Within-entity first differences | nonrobust, robust |
| `FamaMacBeth` | Cross-sectional regressions by period | nonrobust, Newey-West |

## Core Estimating Equations

`PanelOLS` fits OLS after removing requested fixed effects. With entity effects,

$$
y_{it}^{\mathrm{within}} = y_{it} - \bar y_{i\cdot},
\qquad
X_{it}^{\mathrm{within}} = X_{it} - \bar X_{i\cdot}.
$$

With entity and time effects, the two-way transformation adds back the grand mean.

`PooledOLS` fits

$$
\hat\beta = X^+ y,
$$

where \(X^+\) denotes the inverse or Moore-Penrose pseudoinverse as required. `BetweenOLS` applies OLS to entity means, `FirstDifferenceOLS` applies OLS to Δ\(X\) and Δ\(y\), and `FamaMacBeth` averages period-specific coefficient vectors.

## Covariance and Inference

| `cov_type` | Behavior |
|---|---|
| `"nonrobust"` | Classical OLS covariance and t-based inference |
| `"robust"` | HC1 sandwich covariance and asymptotic normal inference |
| `"clustered"` | One-way or two-way cluster-robust covariance |
| `"hac"` | Bartlett/Newey-West HAC for `PooledOLS` |
| `"newey-west"` | HAC applied to the `FamaMacBeth` coefficient path |

### PooledOLS HAC ordering

For `PooledOLS(cov_type="hac")`, pass `time_index=` to `fit`. The implementation validates the side array and uses a stable time ordering while keeping all numerical arrays aligned. Consequently, a row permutation with unchanged time labels produces the same HAC covariance, up to numerical tolerance.

### Rank-deficient PooledOLS

A rank-deficient design separates fitted-space validity from coefficient-space identifiability:

- fitting, prediction, residuals, RSS, rank, and fitted-space comparisons remain valid;
- `df_resid` is computed as `nobs - rank(X)`, not `nobs - n_columns`;
- individual coefficients are not unique under exact collinearity;
- coefficient-level covariance, BSE, test statistics, p-values, and confidence intervals are therefore non-identifiable and should be reported as `NOT_COMPARABLE`, not as a runtime error or a successful unique inference result.

The PR79 validation pipeline preserves prediction/RSS/rank contracts for rank-deficient cases while excluding non-identifiable coefficient-space comparisons.

## Parameters and Fit Signatures

### `PanelOLS`

```python
PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",
    device="auto",
)
```

```python
model.fit(y, X, entity_ids=entity_ids, time_ids=time_ids, cluster=cluster)
```

### `PooledOLS`

```python
PooledOLS(
    cov_type="nonrobust",
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
    device="auto",
)
```

```python
model.fit(X, y, cluster=None, time_index=None)
```

`cluster` is required for clustered inference. `time_index` is strongly recommended for HAC inference and is used to define stable temporal ordering.

### Other models

```python
RandomEffects(device="auto")
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")
FamaMacBeth(
    cov_type="newey-west",
    bandwidth=None,
    alpha=0.05,
    min_obs_per_period=1,
    device="auto",
)
```

## CPU and GPU Examples

```python
import numpy as np
from statgpu.panel import PanelOLS, PooledOLS, FamaMacBeth

n_entities, n_times = 50, 10
n = n_entities * n_times
entity_ids = np.repeat(np.arange(n_entities), n_times)
time_ids = np.tile(np.arange(n_times), n_entities)
X = np.random.default_rng(0).normal(size=(n, 3))
y = X @ np.array([1.0, -0.5, 0.3]) + np.random.default_rng(1).normal(size=n) * 0.1

# Fixed effects on CPU.
fe = PanelOLS(entity_effects=True, cov_type="robust", device="cpu")
fe.fit(y, X, entity_ids=entity_ids)

# HAC PooledOLS with explicit time ordering.
pooled_hac = PooledOLS(cov_type="hac", device="cpu")
pooled_hac.fit(X, y, time_index=time_ids)

# Fama-MacBeth on CuPy CUDA; metadata labels may remain on CPU.
fm = FamaMacBeth(cov_type="newey-west", device="cuda")
fm.fit(X, y, time_ids=time_ids)
```

For Torch CUDA, pass CUDA tensors for numerical arrays and use `device="torch"`. Public prediction methods preserve the estimator backend for array inputs.

## Outputs

Common fitted attributes include:

- `coef_`;
- `bse_`, `tvalues_`, `pvalues_`, `conf_int_` when coefficient-space inference is identifiable;
- `rsquared` or `rsquared_within` as applicable;
- `nobs`, `df_resid`, and effective rank where exposed;
- `betas_`, `cov_params_`, and `n_periods` for `FamaMacBeth`.

For an exactly rank-deficient `PooledOLS` design, downstream consumers must not interpret coefficient-level inference as uniquely identified.

## Formula and Metadata Boundaries

Formula evaluation may drop rows with missing values. Entity, time, cluster, and other side arrays are aligned to the retained rows. String and categorical labels are factorized on CPU; the numerical transformations and regression calculations remain on the selected backend.

## Validation

PR #79 validated maintained panel behavior across NumPy, CuPy CUDA, and Torch CUDA. The final maintained physical-GPU suite passed **33/33** checks on a Tesla P100, including backend-preserving `PooledOLS.predict()` and the rank-deficient `NOT_COMPARABLE` contract. GitHub Actions also passed the Python 3.9–3.12 regression matrix and full CPU suite on the exact head.

See:

- `dev/reviews/pr79_physical_gpu_validation.md`;
- `dev/tests/test_pr79_physical_gpu.py`;
- Issue #83 for cleanup of ignored legacy GPU diagnostic scripts.

## References

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering.
