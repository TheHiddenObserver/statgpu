# Penalized Solver API Migration

statgpu's penalized solver interface uses a backend-neutral `solver` parameter for direct estimator fits. The older `cpu_solver` argument came from an earlier interface in which CPU and GPU paths exposed different solver controls; that argument is now deprecated.

## Direct penalized estimators

Use `solver` regardless of device:

```python
from statgpu.linear_model import Lasso

cpu_model = Lasso(alpha=0.1, device="cpu", solver="coordinate_descent")
gpu_model = Lasso(alpha=0.1, device="cuda", solver="fista")
```

`cpu_solver` remains accepted during the deprecation period, but it does **not** select the direct-fit algorithm in the unified interface. Caller-supplied legacy use emits `FutureWarning`; omitted/default compatibility values used during estimator reconstruction do not create a user-facing warning.

For a **behavior-preserving migration**, remove `cpu_solver` and leave the estimator's existing `solver` value unchanged. Do not mechanically copy a direct estimator's old `cpu_solver` value into `solver`: the legacy argument is not authoritative for direct fitting, so copying it can select a different algorithm.

If the old value is the algorithm you now intentionally want, set that value through `solver` explicitly and treat the change as an algorithm-selection change.

This applies to the public penalized estimator family, including `Ridge`, `Lasso`, `ElasticNet`, typed `Penalized*Regression` estimators, penalized robust/quantile models, and penalized Cox.

## LassoCV has two solver stages

`LassoCV` has separate optimization stages for CV scoring and the final full-data refit. The API therefore names the stage rather than the hardware:

```python
from statgpu.linear_model import LassoCV

model = LassoCV(
    solver="fista",            # final full-data refit
    cv_solver="auto",          # CV folds/path
)
```

Current behavior is:

- `solver` controls the final full-data `Lasso` refit;
- `cv_solver` controls the CV folds/path;
- `cv_solver="auto"` selects coordinate descent on CPU and FISTA on CUDA/Torch;
- explicit `cv_solver="coordinate_descent"` is CPU-only;
- `method="glmnet"` selects coordinate descent for the CPU CV path, while CUDA/Torch CV continues to use FISTA;
- after fitting, `cv_solver_` records the CV algorithm that actually executed.

The old `LassoCV(cpu_solver=...)` argument is deprecated. On CPU it acts as the legacy alias for `cv_solver` when the new control has not already selected a conflicting algorithm. On CUDA/Torch it warns but does not replace the GPU FISTA CV path.

The final refit no longer receives `cpu_solver`; that stage is controlled by `solver`.

## Migration examples

| Legacy call | Behavior-preserving replacement | Optional explicit algorithm choice |
|---|---|---|
| `Lasso(device="cpu", cpu_solver="coordinate_descent")` | `Lasso(device="cpu")` | `Lasso(device="cpu", solver="coordinate_descent")` only if you intentionally want coordinate descent |
| `ElasticNet(device="cpu", cpu_solver="fista")` | `ElasticNet(device="cpu")` | `ElasticNet(device="cpu", solver="fista")` if you want the choice explicit |
| `PenalizedLinearRegression(device="cpu", cpu_solver="fista")` | `PenalizedLinearRegression(device="cpu")` | `solver="fista"` only if you intentionally want to pin FISTA instead of automatic dispatch |
| `LassoCV(device="cpu", cpu_solver="fista")` | `LassoCV(device="cpu", cv_solver="fista")` | same; here the legacy control selected the CPU CV stage |
| `LassoCV(device="cuda", cpu_solver="coordinate_descent")` | `LassoCV(device="cuda", cv_solver="auto")` or omit both CV controls | `cv_solver="fista"` to state the GPU CV algorithm explicitly |
| `LassoCV(solver="fista", cpu_solver="coordinate_descent", device="cpu")` | `LassoCV(solver="fista", cv_solver="coordinate_descent", device="cpu")` | same; `solver` remains the final-refit control |

## Removal timeline

`cpu_solver` is deprecated and is intended for removal in a future breaking release. New code should use `solver` for direct/final-refit optimization and `cv_solver` for the `LassoCV` selection stage.

See [Cross-Validation](cross-validation.md) for the general selection/refit distinction and [Solver × Penalty Matrix](solver-penalty-matrix.md) for explicit solver compatibility.
