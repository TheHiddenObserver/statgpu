# Models Overview

> Language: English  
> Last updated: 2026-04-11  
> This page: Model index  
> Switch: [Chinese](../../models/README.md)

Language switch: [Chinese](../../models/README.md)

## Linear Models

- [LinearRegression](linear-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [LogisticRegression](logistic-regression.md)

## Survival

- [CoxPH](coxph.md)

## Feature Selection

- [Knockoff](knockoff.md)

## Nonparametric

- [Nonparametric](nonparametric.md)

## Current Coverage Notes

- All current models support `device="cpu"` / `device="cuda"` / `device="auto"`.
- All current models support `gpu_memory_cleanup`.
- Inference-rich models:
  - `LinearRegression`: classical + `HC0/HC1/HC2/HC3/HAC`
  - `Ridge`: classical + `HC0/HC1/HC2/HC3/HAC`
  - `Lasso`: CPU/GPU OLS-style inference + bootstrap
  - `LogisticRegression`: classical + `HC0/HC1/HC2/HC3/HAC`
- `CoxPH` supports Breslow/Efron ties and CPU/GPU fitting paths.
- Feature selection:
  - `Knockoff`: fixed-X/model-X unified API + selector wrappers
- `LassoCV` is implemented and trainable.
- Exported CV classes (`RidgeCV`, `LogisticRegressionCV`, `CoxPHCV`) are currently interface skeletons only; CV training/search logic is pending.
