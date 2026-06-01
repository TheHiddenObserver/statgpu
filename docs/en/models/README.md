# Models Overview

> Language: English  
> Last updated: 2026-05-28  
> This page: Model index  
> Switch: [Chinese](../../models/README.md)

Language switch: [Chinese](../../models/README.md)

## Linear and GLM Models

- [LinearRegression](linear-regression.md)
- [GeneralizedLinearModel and Penalized GLM](generalized-linear-model.md)
- [PoissonRegression](poisson-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [ElasticNet](elastic-net.md)
- [LogisticRegression](logistic-regression.md)
- [Ordered Generalized Linear Models (Logit/Probit)](ordered.md)

## ANOVA

- [One-Way ANOVA](anova.md)

## Covariance Estimation

- [EmpiricalCovariance, LedoitWolf, OAS](covariance.md)

## Panel Data

- [PanelOLS and RandomEffects](panel.md)

## Nonparametric Methods

- [Kernel Density Estimation and Kernel Regression](nonparametric.md)
- [Kernel Ridge Regression](nonparametric/kernel-methods.md)
- [Spline Basis Functions](nonparametric/splines.md)

## Semiparametric Models

- [GAM (Generalized Additive Model)](semiparametric.md)

## Survival

- [CoxPH](coxph.md)

## Feature Selection

- [Knockoff](knockoff.md)

## Current Coverage Notes

- All current models support `device="cpu"` / `device="cuda"` / `device="torch"` / `device="auto"` where the documented backend implementation is available.
- Explicit `device="cuda"` and `device="torch"` raise when their matching CUDA backend is unavailable; only `device="auto"` may choose another backend.
- All current models support `gpu_memory_cleanup`.
- `GeneralizedLinearModel` and typed penalized GLMs are documented in [GeneralizedLinearModel and Penalized GLM](generalized-linear-model.md).
- `PoissonRegression` is documented separately as the ordinary Poisson GLM estimator.
- Inference-rich models:
  - `LinearRegression`: classical + `HC0/HC1/HC2/HC3/HAC`
  - `Ridge`: classical + `HC0/HC1/HC2/HC3/HAC`
  - `Lasso`: CPU/GPU OLS-style inference + bootstrap
  - `LogisticRegression`: classical + `HC0/HC1/HC2/HC3/HAC`
- `CoxPH` supports Breslow/Efron ties and CPU/GPU fitting paths.
- `OrderedLogitRegression` / `OrderedProbitRegression` support CPU/CuPy/Torch backends with cross-backend precision fix (coef diff < 1e-2).
- `CoxPH` delayed entry (`entry`) support:
  - `entry + breslow`: CPU/CUDA/Torch
  - `entry + efron`: CPU/CUDA/Torch
- Feature selection:
  - `Knockoff`: fixed-X/model-X unified API + selector wrappers
- `LassoCV` is implemented and trainable.
- Exported CV classes status:
  - `RidgeCV`, `LogisticRegressionCV`, and `CoxPHCV` are implemented and trainable.
  - Current `CoxPHCV` boundary: on GPU paths, `entry` currently supports only `ties='breslow'`; `cluster`-robust CV is not yet supported and raises `NotImplementedError`.
- New modules (validated 38/38 ALL PASS on Tesla P100):
  - `ANOVA`: `f_oneway` — drop-in replacement for `scipy.stats.f_oneway`
  - `Covariance`: `EmpiricalCovariance`, `LedoitWolf`, `OAS` — equivalent to `sklearn.covariance`
  - `KernelMethods`: `KernelRidge`, `KernelRidgeCV` — equivalent to `sklearn.kernel_ridge`
  - `Panel`: `PanelOLS`, `RandomEffects` — equivalent to `linearmodels.panel`
  - `Splines`: `bspline_basis`, `natural_cubic_spline_basis`, `GAM` — penalized B-spline GAM with GCV
