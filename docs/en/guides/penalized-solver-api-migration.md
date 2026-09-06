# Penalized solver API migration

statgpu's current penalized solver engine uses one backend-neutral `solver`
parameter for a direct estimator fit. The older `cpu_solver` argument came from
the previous implementation, where CPU and GPU paths exposed different solver
controls. That split is now deprecated.

## Direct penalized estimators

Use `solver` regardless of device:

```python
from statgpu.linear_model import Lasso

cpu_model = Lasso(alpha=0.1, device="cpu", solver="coordinate_descent")
gpu_model = Lasso(alpha=0.1, device="cuda", solver="fista")
```

`cpu_solver` remains accepted for one compatibility cycle, but it does **not**
select the direct-fit algorithm in the unified engine. Meaningful legacy use
emits `FutureWarning`; migrate to `solver=...`.

This applies to the public penalized estimator family, including `Ridge`,
`Lasso`, `ElasticNet`, the typed `Penalized*Regression` estimators, penalized
robust/quantile models, and penalized Cox.

## LassoCV

`LassoCV` has two genuinely different optimization stages, so the API names the
stages instead of the hardware:

```python
from statgpu.linear_model import LassoCV

model = LassoCV(
    solver="fista",            # final full-data refit
    cv_solver="auto",          # CV folds/path
)
```

`cv_solver="auto"` resolves to coordinate descent on CPU and FISTA on CUDA or
Torch. `cv_solver="coordinate_descent"` is CPU-only. `method="glmnet"` fixes the
CV path to coordinate descent.

The old `LassoCV(cpu_solver=...)` argument is a deprecated alias for
`cv_solver=...`. Supplying both with conflicting values raises `ValueError`
rather than silently choosing one. The resolved CV algorithm is published as
`cv_solver_` after fitting.

The final refit no longer receives `cpu_solver`; only `solver` controls that
stage.

## Migration table

| Old call | Replacement |
|---|---|
| `Lasso(device="cpu", cpu_solver="coordinate_descent")` | `Lasso(device="cpu", solver="coordinate_descent")` |
| `ElasticNet(device="cpu", cpu_solver="fista")` | `ElasticNet(device="cpu", solver="fista")` |
| `LassoCV(cpu_solver="fista")` | `LassoCV(cv_solver="fista")` |
| `LassoCV(solver="fista", cpu_solver="coordinate_descent")` | `LassoCV(solver="fista", cv_solver="coordinate_descent")` |

`cpu_solver` is scheduled for removal in a future breaking release after this
deprecation cycle.
