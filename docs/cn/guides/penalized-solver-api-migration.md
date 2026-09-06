# Penalized 求解器 API 迁移

statgpu 当前的 penalized solver engine 对单次模型拟合使用统一、与后端无关的
`solver` 参数。旧的 `cpu_solver` 来自更早的实现，当时 CPU 与 GPU 分别暴露了
不同求解器控制。这个硬件维度的双接口现在进入弃用流程。

## 直接 penalized estimator

无论设备是什么，都使用 `solver`：

```python
from statgpu.linear_model import Lasso

cpu_model = Lasso(alpha=0.1, device="cpu", solver="coordinate_descent")
gpu_model = Lasso(alpha=0.1, device="cuda", solver="fista")
```

`cpu_solver` 暂时保留一个兼容周期，但在统一引擎中它**不会**选择 direct-fit
算法。具有实际意义的旧式用法会产生 `FutureWarning`；请迁移到 `solver=...`。

这适用于公开的 penalized estimator 家族，包括 `Ridge`、`Lasso`、
`ElasticNet`、typed `Penalized*Regression`、penalized robust/quantile 以及
penalized Cox。

## LassoCV

`LassoCV` 确实存在两个不同优化阶段，因此新 API 按**阶段**命名，而不是按硬件：

```python
from statgpu.linear_model import LassoCV

model = LassoCV(
    solver="fista",            # 最终 full-data refit
    cv_solver="auto",          # CV folds/path
)
```

`cv_solver="auto"` 在 CPU 上解析为 coordinate descent，在 CUDA/Torch 上解析为
FISTA。`cv_solver="coordinate_descent"` 仅支持 CPU。`method="glmnet"` 会固定
CV path 使用 coordinate descent。

旧的 `LassoCV(cpu_solver=...)` 是 `cv_solver=...` 的 deprecated alias。如果同时
提供两个冲突值，会抛出 `ValueError`，而不是静默选择其中一个。拟合后实际使用的
CV 算法记录在 `cv_solver_`。

最终 refit 不再接收 `cpu_solver`；这个阶段只由 `solver` 控制。

## 迁移表

| 旧调用 | 替代调用 |
|---|---|
| `Lasso(device="cpu", cpu_solver="coordinate_descent")` | `Lasso(device="cpu", solver="coordinate_descent")` |
| `ElasticNet(device="cpu", cpu_solver="fista")` | `ElasticNet(device="cpu", solver="fista")` |
| `LassoCV(cpu_solver="fista")` | `LassoCV(cv_solver="fista")` |
| `LassoCV(solver="fista", cpu_solver="coordinate_descent")` | `LassoCV(solver="fista", cv_solver="coordinate_descent")` |

在本次 deprecation 周期之后，`cpu_solver` 计划在未来 breaking release 中删除。
