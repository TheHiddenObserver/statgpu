# Models Overview

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
