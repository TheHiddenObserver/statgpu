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
select the direct-fit algorithm in the unified engine. Caller-owned legacy use
emits `FutureWarning`; framework reconstruction and statgpu's own helper-model
construction do not turn an omitted/default value into a user-facing warning.
Migrate new code to `solver=...`.

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
Torch. An explicitly new `cv_solver="coordinate_descent"` request is CPU-only.
`method="glmnet"` forces coordinate descent on the CPU CV path; CUDA/Torch CV
retains the maintained backend-native FISTA path, and `cv_solver_` records the
algorithm that actually executes.

The old `LassoCV(cpu_solver=...)` argument is deprecated. On CPU it acts as the
legacy alias for `cv_solver`. On CUDA/Torch it warns but remains
non-authoritative, preserving the historical behavior in which this CPU-only
control did not replace the GPU FISTA CV path. Conflicting new/legacy controls
are rejected on CPU, where both would otherwise select the same CV stage.

The final refit no longer receives `cpu_solver`; only `solver` controls that
stage.

## Migration table

| Old call | Replacement |
|---|---|
| `Lasso(device="cpu", cpu_solver="coordinate_descent")` | `Lasso(device="cpu", solver="coordinate_descent")` |
| `ElasticNet(device="cpu", cpu_solver="fista")` | `ElasticNet(device="cpu", solver="fista")` |
| `LassoCV(device="cpu", cpu_solver="fista")` | `LassoCV(device="cpu", cv_solver="fista")` |
| `LassoCV(device="cuda", cpu_solver="coordinate_descent")` | `LassoCV(device="cuda", cv_solver="auto")` |
| `LassoCV(solver="fista", cpu_solver="coordinate_descent", device="cpu")` | `LassoCV(solver="fista", cv_solver="coordinate_descent", device="cpu")` |

`cpu_solver` is scheduled for removal in a future breaking release after this
deprecation cycle.
