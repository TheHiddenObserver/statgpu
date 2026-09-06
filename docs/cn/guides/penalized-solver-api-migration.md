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
算法。调用者自己显式使用旧参数时会产生 `FutureWarning`；框架重建以及 statgpu
内部 helper estimator 的构造不会把原本省略的默认值误报成用户弃用 warning。
新代码请迁移到统一的 `solver` 接口。

如果目标是**保持当前实际数值行为不变**，应删除 `cpu_solver`，同时保持 estimator
现有的 `solver` 值不变。不要机械地把 direct estimator 的旧 `cpu_solver` 值复制到
`solver`：在当前统一引擎里 `cpu_solver` 已经不是权威的 direct-fit 控制，直接复制
可能会有意或无意地切换实际算法。如果旧值恰好表达了你现在确实想显式选择的算法，
可以把它迁到 `solver`，但应把这视为一次主动的 solver 选择变更，并验证拟合结果。

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
FISTA。显式的新接口 `cv_solver="coordinate_descent"` 仅支持 CPU。
`method="glmnet"` 只在 CPU CV path 上固定使用 coordinate descent；CUDA/Torch
仍保留维护中的 backend-native FISTA 路径，拟合后的 `cv_solver_` 记录实际执行的
算法。

旧的 `LassoCV(cpu_solver=...)` 已弃用。在 CPU 上，它作为旧版 `cv_solver` alias
保留原行为；在 CUDA/Torch 上，它会 warning 但仍保持**非权威**，从而保留旧版本
中“CPU-only 控制不改变 GPU FISTA CV 路径”的行为。在 CPU 上同时提供冲突的新旧
控制会抛出 `ValueError`。

最终 refit 不再接收 `cpu_solver`；这个阶段只由 `solver` 控制。

## 迁移示例

| 旧调用 | 保持当前行为的迁移 | 可选的显式算法选择 |
|---|---|---|
| `Lasso(device="cpu", cpu_solver="coordinate_descent")` | `Lasso(device="cpu")`（保持当前默认 direct solver，即 FISTA） | 只有确实希望把实际 direct solver 切到 coordinate descent 时，才改为 `Lasso(device="cpu", solver="coordinate_descent")` |
| `ElasticNet(device="cpu", cpu_solver="fista")` | `ElasticNet(device="cpu")`（保持现有 `solver` 值） | 若希望显式写出算法，可用 `ElasticNet(device="cpu", solver="fista")` |
| `PenalizedLinearRegression(device="cpu", cpu_solver="fista")` | `PenalizedLinearRegression(device="cpu")`（保持 `solver="auto"`） | 只有希望主动固定为 FISTA 而不是 auto dispatch 时，才用 `solver="fista"` |
| `LassoCV(device="cpu", cpu_solver="fista")` | `LassoCV(device="cpu", cv_solver="fista")` | 相同；这里旧参数确实控制 CPU CV 算法 |
| `LassoCV(device="cuda", cpu_solver="coordinate_descent")` | `LassoCV(device="cuda", cv_solver="auto")`（或直接省略两个控制参数） | 可用 `cv_solver="fista"` 显式写出维护中的 GPU CV 算法 |
| `LassoCV(solver="fista", cpu_solver="coordinate_descent", device="cpu")` | `LassoCV(solver="fista", cv_solver="coordinate_descent", device="cpu")` | 相同；`solver` 继续控制最终 refit，`cv_solver` 接管 CPU CV |

在本次 deprecation 周期之后，`cpu_solver` 计划在未来 breaking release 中删除。
