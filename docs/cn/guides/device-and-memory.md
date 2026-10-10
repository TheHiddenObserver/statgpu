# 设备与 GPU 内存

> 语言：中文\
> 最后更新：2026-10-10\
> 页面定位：设备选择与 GPU 内存控制\
> 切换：[English](../../en/guides/device-and-memory.md)

## 选择设备

对于采用通用设备处理方式的估计器（如 `Ridge`），可使用以下设置：

- `device="cpu"`：使用 NumPy 在 CPU 上计算。
- `device="cuda"`：使用 CuPy 在 CUDA GPU 上计算。
- `device="torch"`：使用 Torch 在 CUDA GPU 上计算。
- `device="auto"`：由估计器结合全局设置选择后端。

这些估计器显式请求 `"cuda"` 或 `"torch"` 时，如果对应的 CUDA 后端不可用，会报错。需要 CPU 计算时请设置 `device="cpu"`；允许自动选择时可用 `device="auto"`。部分核方法与样条估计器对这些设置的处理有所不同；依赖 GPU 执行前，请先阅读[例外与设备检查方法](#current-smoothing-and-spline-exceptions)。

以下示例在 CPU 上运行，无需安装 CuPy 或 Torch：

<!-- api-example: device-cpu -->
```python
import numpy as np
from statgpu.linear_model import Ridge

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 5))
y = 1.0 + X @ np.arange(1.0, 6.0) + rng.normal(size=200)
model = Ridge(alpha=1.0, device="cpu", compute_inference=False).fit(X, y)
predictions = model.predict(X[:3])
print(predictions)
```

后端支持范围取决于模型和具体操作。为拟合、交叉验证或推断选择后端前，请查看对应模型页或[已实现方法](implemented-methods.md)。某个后端可用，并不代表任意损失函数、惩罚项和求解器组合都能使用；组合限制见[求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

### 估计器的 `device` 与函数的 `backend`

采用通用设备处理方式的估计器用 `device="torch"` 请求 Torch CUDA。函数的 `backend="torch"` 则用于选择 Torch 库，可以支持 Torch CPU。例如，`get_backend(backend="torch", device="cpu")` 返回 Torch CPU 后端；`get_backend(backend="torch")` 在 Torch CUDA 可用时使用 CUDA，否则使用 Torch CPU。两种方式都需要安装 Torch。请查看具体函数的后端说明，不要将这两个参数视为相同的设置。

## 全局设置与估计器设置

`statgpu.set_device` 会修改整个进程的设备设置。对于采用通用设备处理方式的估计器：

- 估计器显式指定的 `device="cpu"`、`"cuda"` 或 `"torch"` 优先于全局设置。
- 估计器的 `device="auto"` 会读取全局设置，不会将全局设置重置为自动选择。

例如，调用 `set_device("torch")` 后，即使估计器设置了 `device="auto"`，Torch CUDA 不可用时仍可能报错。需要恢复全局自动选择时请调用 `set_device("auto")`；如果只想让当前模型在 CPU 上拟合，请设置 `device="cpu"`。将全局设备设为不可用的 GPU 后端时会产生警告，并保留请求的设置。

全局设置为自动选择时，`statgpu.get_device()` 依次检查 CuPy CUDA、Torch CUDA、CPU，返回可用的设备。这只是全局设置解析后的结果，并不表示每个模型都在该设备上运行。包括惩罚 GLM 在内的部分模型，只要自身的 `device` 为 `"auto"`，就可能针对某些工作量选择 CPU，即使全局设置为 GPU。需要固定硬件位置时，请在模型上显式指定设备，同时注意下文的例外。

如果只有一次拟合需要更换设备，优先在模型上设置。如果临时修改全局设置，请在操作结束后恢复先前配置的策略，发生异常时也要恢复。请自行记录这一策略：`get_device()` 会把 `"auto"` 解析为具体设备，仅保存其返回值无法保留原先的 `"auto"` 策略。

<a id="current-smoothing-and-spline-exceptions"></a>

## 核方法与样条例外：检查数组所在设备

以下估计器仅设置 GPU 参数，并不能保证实际使用 GPU：

- `KernelDensityEstimator`（含 `KDE`）和 `KernelRegression`（含 `KernelRegressionRegressor`）处理 NumPy 或 Torch CPU 输入时，即使设置 `device="torch"` 且 `backend="auto"` 或 `"torch"`，仍可能在 Torch CPU 上拟合。显式 `backend="torch"` 搭配 `device="cuda"` 也可能在 CPU 上执行；显式 `backend="numpy"` 会覆盖这两种 GPU 请求。使用 `device="cuda", backend="auto"` 时，如果 CuPy 不可用而 Torch CUDA 可用，这些估计器可能选择 Torch，但 CPU 输入仍可能留在 CPU 上。
- `SplineTransformer` 会保留传入 Torch 张量所在的设备。因此，Torch CPU 输入即使搭配 `device="torch"` 或 `"cuda"`，拟合节点与变换特征仍可能留在 CPU 上。相同设置改用 NumPy 输入时，如果请求的加速器不可用，则可能报错。
- `KernelPCA` 与 `Nystroem` 在 Torch CUDA 不可用时会拒绝 `device="torch"`。但通过检查后，NumPy 或 Torch CPU 输入仍可能得到 Torch CPU 特征。通过可用性检查不代表张量已经位于 GPU。

请检查这些方法实际使用和返回的数组：

- KDE/核回归：`samples_` 及返回的密度或预测。
- `SplineTransformer`：`knots_` 中的各个数组及变换后的基矩阵。
- `KernelPCA`/`Nystroem`：`fit_transform`、`transform` 或 `predict` 返回的特征。两者的公开拟合数组即使在 GPU 计算后也以 NumPy 数组存储，不能用来判断计算所在设备。

Torch 张量查看 `.device` 与 `.is_cuda`，CuPy 数组查看 `.device`。NumPy 数组位于 CPU。`model.device` 或 `backend_="torch"` 这样的库名称都不能证明 CUDA 执行。

如果必须使用 GPU，请不要接受这些方法返回的 CPU 结果。请使用受支持的 GPU 输入与后端组合，并在使用结果前检查上述数组。需要明确的 CPU 替代方案时，请传入 NumPy 数组并设 `device="cpu"`；KDE/核回归还需设 `backend="numpy"`。

其他限制与受支持的组合见[核平滑](../models/nonparametric.md#可选-gpu-路径与外部对照)、[核特征](../models/kernel-methods.md#后端与执行边界)与[样条](../models/splines.md#后端执行与验证边界)。

## 输入转换与返回数组

公式（Formula）/DataFrame 解析与列信息准备可以先在 CPU 上完成，再进行数值拟合。GPU 拟合也可能需要把输入数组从 CPU 复制到 GPU。比较端到端运行时间时，请计入这些传输；复用 GPU 数组前，也请检查模型接受的输入类型。

输出类型由具体方法决定。例如，`Ridge.predict(X)` 默认返回 NumPy 数组，包括在 GPU 上拟合之后。`Ridge.predict(X, return_cpu=False)` 则在已拟合模型的数值后端上返回预测：CPU 拟合返回 NumPy 数组，GPU 拟合返回 CuPy 数组或 Torch 张量。后续还要在同一 GPU 后端上继续计算时，可使用这一选项。`Ridge.coef_` 始终是 NumPy 数组，不能据此判断拟合所在设备。

方法明示的 NumPy 输出转换，与上述核方法/样条例外中的 CPU 执行是两种情况。请先查看返回值说明，再判断能否用输出类型识别计算位置。

## GPU 内存清理

部分支持 GPU 的估计器提供 `gpu_memory_cleanup` 参数。以 `Ridge` 为例：

- `gpu_memory_cleanup=False`（默认值）会保留后端可复用的内存分配，有利于重复拟合的吞吐量。
- `gpu_memory_cleanup=True` 会在拟合的清理阶段请求释放 GPU 内存池中未使用的分配。这可以减少保留的显存，但后续操作可能需要重新分配内存。

清理不会丢弃 `predict()`、`score()` 或受支持推断所需要的已拟合状态，也无法释放模型或用户代码中仍被引用的数组占用的显存。因此，该选项不保证归还全部显存，也不保证降低一次拟合所需的峰值显存。参数是否可用以及清理时机取决于具体估计器。

以下自包含示例需要可用的 CuPy/CUDA 环境：

<!-- api-example: cuda-cleanup -->
```python
import numpy as np
from statgpu.linear_model import Ridge

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 5))
y = 1.0 + X @ np.arange(1.0, 6.0) + rng.normal(size=200)
model = Ridge(
    alpha=1.0,
    device="cuda",
    gpu_memory_cleanup=True,
)
model.fit(X, y)
print(model.score(X, y))
```

多个模型共享 GPU，或归还未使用显存比重复拟合延迟更重要时，可开启清理。持续重复拟合、希望复用已有分配且显存足够时，可保持关闭。

## 相关文档

- [已实现方法](implemented-methods.md) — 模型与后端清单
- [交叉验证](cross-validation.md) — 选择与最终重拟合中的设备行为
- [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md) — 求解器兼容性
- [求解器算法](solver-algorithms.md) — 算法定义
- [PyTorch 后端](pytorch-backend.md) — PyTorch 专属使用说明
