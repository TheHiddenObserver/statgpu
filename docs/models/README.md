# Models Overview

> 语言: 中文  
> 最后更新: 2026-04-02  
> 页面定位: 模型索引  
> 切换: [English](../en/models/README.md)

语言切换：[English](../en/models/README.md)

This section organizes method-level docs so the documentation scales as more
statistical methods are added.

## Linear Models

- [LinearRegression](linear-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [LogisticRegression](logistic-regression.md)

## Survival

- [CoxPH](coxph.md)

## Adding a New Model Doc

When adding a new estimator:

1. Create `docs/models/<model-name>.md`
2. Add it to this index
3. Add it to `USAGE.md` navigation
4. If benchmarked, add script reference in `docs/benchmarks.md`

## Current Coverage Notes

- All current models support `device="cpu"` / `device="cuda"` / `device="auto"`.
- All current models support `gpu_memory_cleanup`.
- Inference-rich models:
  - `LinearRegression`: classical + `HC0/HC1`
  - `Lasso`: CPU/GPU OLS-style inference + bootstrap
  - `LogisticRegression`: classical + `HC0/HC1`
- `CoxPH` supports Breslow/Efron ties and CPU/GPU fitting paths.
