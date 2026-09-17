# Penalized 求解器 API 迁移

statgpu 的 penalized solver interface 对 direct estimator fit 使用与 backend 无关的 `solver` 参数。旧的 `cpu_solver` 来自更早的接口，当时 CPU 与 GPU path 分别暴露不同的 solver 控制；该参数现在已经弃用。

## 直接 penalized estimator

无论 device 是什么，都使用 `solver`：

```python
from statgpu.linear_model import Lasso

cpu_model = Lasso(alpha=0.1, device="cpu", solver="coordinate_descent")
gpu_model = Lasso(alpha=0.1, device="cuda", solver="fista")
```

`cpu_solver` 在 deprecation period 内仍可接受，但在统一接口中它**不会**选择 direct-fit 算法。调用者显式提供旧参数时会产生 `FutureWarning`；estimator reconstruction 中省略/默认的 compatibility value 不会产生用户可见 warning。

如果目标是**保持当前实际数值行为不变**，应删除 `cpu_solver`，并保持 estimator 现有的 `solver` 值不变。不要机械地把 direct estimator 的旧 `cpu_solver` 值复制到 `solver`：legacy argument 并不是 direct fitting 的权威控制，因此复制它可能主动或意外地切换实际算法。

如果旧值正是你现在有意请求的算法，可以显式通过 `solver` 设置，并把它视为一次 algorithm-selection change。

这适用于公开的 penalized estimator family，包括 `Ridge`、`Lasso`、`ElasticNet`、typed `Penalized*Regression`、penalized robust/quantile model 与 penalized Cox。

## LassoCV 有两个 solver 阶段

`LassoCV` 对 CV scoring 与最终 full-data refit 使用两个独立的优化阶段，因此 API 按**阶段**命名，而不是按硬件命名：

```python
from statgpu.linear_model import LassoCV

model = LassoCV(
    solver="fista",            # 最终 full-data refit
    cv_solver="auto",          # CV folds/path
)
```

当前行为是：

- `solver` 控制最终 full-data `Lasso` refit；
- `cv_solver` 控制 CV folds/path；
- `cv_solver="auto"` 在 CPU 上选择 coordinate descent，在 CUDA/Torch 上选择 FISTA；
- 显式 `cv_solver="coordinate_descent"` 只支持 CPU；
- `method="glmnet"` 在 CPU CV path 上选择 coordinate descent，而 CUDA/Torch CV 继续使用 FISTA；
- 拟合完成后，`cv_solver_` 记录 CV 阶段实际执行的算法。

旧的 `LassoCV(cpu_solver=...)` 已弃用。在 CPU 上，如果新控制没有选择冲突算法，它作为 `cv_solver` 的 legacy alias；在 CUDA/Torch 上会 warning，但不会替换 GPU FISTA CV path。

最终 refit 不再接收 `cpu_solver`；该阶段由 `solver` 控制。

## 迁移示例

| 旧调用 | 保持当前行为的迁移 | 可选的显式算法选择 |
|---|---|---|
| `Lasso(device="cpu", cpu_solver="coordinate_descent")` | `Lasso(device="cpu")` | 只有确实希望使用 coordinate descent 时才写 `Lasso(device="cpu", solver="coordinate_descent")` |
| `ElasticNet(device="cpu", cpu_solver="fista")` | `ElasticNet(device="cpu")` | 若希望显式写出算法，可用 `ElasticNet(device="cpu", solver="fista")` |
| `PenalizedLinearRegression(device="cpu", cpu_solver="fista")` | `PenalizedLinearRegression(device="cpu")` | 只有希望主动固定为 FISTA 而不是 automatic dispatch 时才设置 `solver="fista"` |
| `LassoCV(device="cpu", cpu_solver="fista")` | `LassoCV(device="cpu", cv_solver="fista")` | 相同；这里旧参数控制的是 CPU CV 阶段 |
| `LassoCV(device="cuda", cpu_solver="coordinate_descent")` | `LassoCV(device="cuda", cv_solver="auto")`，或省略两个 CV control | `cv_solver="fista"` 可显式写出 GPU CV 算法 |
| `LassoCV(solver="fista", cpu_solver="coordinate_descent", device="cpu")` | `LassoCV(solver="fista", cv_solver="coordinate_descent", device="cpu")` | 相同；`solver` 仍控制 final refit |

## 删除时间线

`cpu_solver` 已弃用，计划在未来 breaking release 中删除。新代码应使用 `solver` 控制 direct/final-refit optimization，并用 `cv_solver` 控制 `LassoCV` 的 selection stage。

一般的 selection/refit 区分见 [交叉验证](cross-validation.md)，显式 solver 兼容性见 [Solver × Penalty 矩阵](solver-penalty-matrix.md)。
