# 惩罚模型求解器 API 迁移

statgpu 的惩罚模型求解器接口，在直接拟合估计器时统一使用与后端无关的 `solver` 参数。旧的 `cpu_solver` 来自更早的接口，当时 CPU 与 GPU 分别暴露不同的求解器控制；该参数现在已经弃用。

## 直接拟合惩罚估计器

无论使用什么设备，都通过 `solver` 选择算法：

```python
from statgpu.linear_model import Lasso

cpu_model = Lasso(alpha=0.1, device="cpu", solver="coordinate_descent")
gpu_model = Lasso(alpha=0.1, device="cuda", solver="fista")
```

在弃用过渡期内，`cpu_solver` 仍然可以传入，但在统一接口中它**不会**选择直接拟合所使用的算法。调用者显式提供旧参数时会收到 `FutureWarning`；估计器在内部重建时省略该参数，或只使用默认兼容值，则不会产生用户可见警告。

如果目标是**保持当前实际数值行为不变**，应删除 `cpu_solver`，并保持估计器现有的 `solver` 值不变。不要机械地把直接估计器中旧的 `cpu_solver` 值复制到 `solver`：历史参数并不是直接拟合阶段的权威控制项，复制它反而可能主动或意外切换实际算法。

如果旧值正是现在希望显式使用的算法，可以通过 `solver` 明确指定，并把这视为一次主动的算法选择变更。

这一规则适用于公开的惩罚估计器族，包括 `Ridge`、`Lasso`、`ElasticNet`、各类 `Penalized*Regression`、惩罚稳健回归、惩罚分位数回归以及惩罚 Cox 模型。

## LassoCV 包含两个求解阶段

`LassoCV` 的交叉验证评分与最终全数据重拟合属于两个独立的优化阶段，因此 API 按**阶段**命名，而不是按硬件命名：

```python
from statgpu.linear_model import LassoCV

model = LassoCV(
    solver="fista",            # 最终全数据重拟合
    cv_solver="auto",          # 交叉验证阶段
)
```

当前行为是：

- `solver` 控制最终全数据 `Lasso` 重拟合；
- `cv_solver` 控制交叉验证各数据折和参数路径上的拟合；
- `cv_solver="auto"` 在 CPU 上选择坐标下降，在 CUDA/Torch 上选择 FISTA；
- 显式 `cv_solver="coordinate_descent"` 只支持 CPU；
- `method="glmnet"` 在 CPU 的 CV 阶段使用坐标下降，而 CUDA/Torch 的 CV 阶段继续使用 FISTA；
- 拟合完成后，`cv_solver_` 记录 CV 阶段实际执行的算法。

旧的 `LassoCV(cpu_solver=...)` 已弃用。在 CPU 上，如果新的控制项没有指定冲突算法，它会作为 `cv_solver` 的历史兼容别名；在 CUDA/Torch 上会产生警告，但不会替换 GPU 上的 FISTA 交叉验证路径。

最终重拟合阶段不再接收 `cpu_solver`；这一阶段由 `solver` 控制。

## 迁移示例

| 旧调用 | 保持当前行为的迁移 | 可选的显式算法选择 |
|---|---|---|
| `Lasso(device="cpu", cpu_solver="coordinate_descent")` | `Lasso(device="cpu")` | 只有确实希望使用坐标下降时才写 `Lasso(device="cpu", solver="coordinate_descent")` |
| `ElasticNet(device="cpu", cpu_solver="fista")` | `ElasticNet(device="cpu")` | 若希望显式写出算法，可用 `ElasticNet(device="cpu", solver="fista")` |
| `PenalizedLinearRegression(device="cpu", cpu_solver="fista")` | `PenalizedLinearRegression(device="cpu")` | 只有希望主动固定为 FISTA 而不是自动分发时才设置 `solver="fista"` |
| `LassoCV(device="cpu", cpu_solver="fista")` | `LassoCV(device="cpu", cv_solver="fista")` | 相同；这里旧参数控制的是 CPU 上的 CV 阶段 |
| `LassoCV(device="cuda", cpu_solver="coordinate_descent")` | `LassoCV(device="cuda", cv_solver="auto")`，或省略两个 CV 控制项 | `cv_solver="fista"` 可显式写出 GPU 上的 CV 算法 |
| `LassoCV(solver="fista", cpu_solver="coordinate_descent", device="cpu")` | `LassoCV(solver="fista", cv_solver="coordinate_descent", device="cpu")` | 相同；`solver` 仍控制最终重拟合 |

## 删除时间线

`cpu_solver` 已弃用，并计划在未来的不兼容版本中删除。新代码应使用 `solver` 控制直接拟合或最终重拟合的优化算法，并使用 `cv_solver` 控制 `LassoCV` 的选择阶段。

一般的“选择—最终重拟合”区分见 [交叉验证](cross-validation.md)，显式求解器兼容性见 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。
